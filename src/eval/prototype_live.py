"""
prototype_live.py — W13 bước 8: prototype chạy THẬT (OCR trực tiếp + model) có còn đúng số đã báo cáo không,
nhanh tới đâu, và trả lời ảnh thật ra sao.

Vấn đề: R-PAR (bước 4–6) kiểm backend với OCR lấy từ CACHE. Giao diện thật thì OCR ảnh trực tiếp, ở tiến trình
riêng (src/prototype/ocr_worker.py), có thể trên môi trường Paddle khác lúc dựng cache. Nếu OCR trực tiếp đọc khác
cache thì prompt khác -> câu trả lời của demo có thể khác số trong khoá luận.

Luật ghi trước (docs/W13/README.md, mục "Bước 8"):
  R-OCR  : trên mẫu ảnh test (200 ảnh / tập, seed 42), văn bản OCR đưa vào prompt (lines_to_texts, 600 ký tự)
           của OCR trực tiếp TRÙNG HỆT cache ở ≥ 98% ảnh, từng tập.
  R-LIVE : câu hỏi của các ảnh đó, chạy đúng đường của giao diện (prepare_image -> Backend.ask, OCR trực tiếp):
           câu trả lời trùng hệt run eval W11 ở ≥ 97% câu VÀ EM không thấp hơn W11 quá 1,5 điểm, từng tập
           (bỏ các câu bị detector có/không bắt — đã đánh giá ở R-YB-det).
Chỉ mô tả: latency p50/p95 (OCR / model / tổng, bỏ 5 câu khởi động), thời gian khởi động, VRAM; ảnh thật của nhóm.

Chạy (Colab L4, notebook 24):
  python -m src.eval.prototype_live --run --drive_root … --adapter … --expect_adapter_sha … --ocr_device gpu
  python -m src.eval.prototype_live --analyze --exp_root … --out …
  python -m src.eval.prototype_live --realphotos --photos_dir … --questions_csv … --adapter … --out …
Kiểm thử offline: python -m src.eval.prototype_live --selftest
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

LIVE_VERSION = "prototype-live-v1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = ("vivqa", "vitextvqa_official")
SAMPLE_SEED = 42
N_IMAGES = 200                 # ảnh / tập
OCR_IDENTICAL_MIN = 0.98       # R-OCR
ANSWER_IDENTICAL_MIN = 0.97    # R-LIVE
EM_DROP_MAX = 1.5              # R-LIVE (điểm EM, như δ W11)
WARMUP = 5                     # bỏ 5 câu đầu khỏi thống kê latency
LIVE_RUN = "W13_LIVE_sh_ocrlive_sample200img_bf16_L4_seed42"
OUT_NAME = "W13_prototype_live_L4_seed42"
REAL_RUN = "W13_REAL_sh_realphotos_bf16_L4"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# =============================================================================
# 1) MẪU (tất định theo seed — ghi trước bằng code)
# =============================================================================
def build_sample(samples: List[Dict[str, Any]], n_images: int = N_IMAGES, seed: int = SAMPLE_SEED) -> List[Dict[str, Any]]:
    """n_images ảnh (sắp tên, random.Random(seed).shuffle, lấy đầu) -> MỌI câu hỏi của các ảnh đó, giữ thứ tự gốc."""
    imgs = sorted({s["image"] for s in samples})
    random.Random(seed).shuffle(imgs)
    keep = set(imgs[:n_images])
    return [s for s in samples if s["image"] in keep]


def sample_id_hash(sample: List[Dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps([s["question_id"] for s in sample]).encode()).hexdigest()


# =============================================================================
# 2) CHẠY TRỰC TIẾP (đúng đường của giao diện)
# =============================================================================
def run_items(items: List[Dict[str, Any]], ask: Callable[[str, str], Dict[str, Any]], out_jsonl: str,
              tmp_dir: str) -> List[Dict[str, Any]]:
    """Mỗi item {question_id, dataset, image, image_path, question, answers?} -> prepare_image -> ask.
    Ghi nối tiếp từng dòng (resume: bỏ câu đã có)."""
    from src.prototype.app import prepare_image
    done = {}
    if os.path.exists(out_jsonl):
        with open(out_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[(r["dataset"], r["question_id"])] = r
    todo = [it for it in items if (it["dataset"], it["question_id"]) not in done]
    _log(f"[live] {len(items)} câu | đã có {len(done)} | còn {len(todo)}")
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for k, it in enumerate(todo):
            used, info = prepare_image(it["image_path"], tmp_dir)
            try:
                r = ask(used, it["question"])
            finally:
                if used != it["image_path"] and os.path.exists(used):
                    os.remove(used)
            rec = {"dataset": it["dataset"], "question_id": it["question_id"], "image": it["image"],
                   "cache_key": os.path.basename(it["image_path"]),
                   "question": it["question"], "answers": it.get("answers"), "prediction": r["answer"],
                   "mode": r["mode"], "level": r["level"], "confidence": r["confidence"],
                   "ocr_texts": r.get("ocr_texts") or [], "n_ocr_lines": len(r.get("ocr_lines") or []),
                   "sec_ocr": r.get("sec_ocr"), "sec_vlm": r.get("sec"), "sec_total": r.get("sec_total"),
                   "image_untouched": not info["reencoded"], "img_size": info["orig_size"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[(rec["dataset"], rec["question_id"])] = rec
            if (k + 1) % 50 == 0:
                _log(f"[live] {k + 1}/{len(todo)}")
    return [done[(it["dataset"], it["question_id"])] for it in items]


def gpu_memory_snapshot() -> Dict[str, Any]:
    """VRAM theo tiến trình (nvidia-smi) — gồm cả tiến trình OCR con."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        procs = [{"pid": int(a), "mib": int(b)} for a, b in (l.split(",") for l in out.strip().splitlines() if "," in l)]
        return {"processes": procs, "total_mib": sum(p["mib"] for p in procs)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def run_live(drive_root: str, adapter: str, expect_adapter_sha: Optional[str], ocr_device: str, exp_root: Optional[str],
             config: str, n_images: int = N_IMAGES) -> Dict[str, Any]:
    import torch
    from src.data.vivqa_dataset import load_vivqa
    from src.eval.prototype_check import DATA_REL
    from src.eval.run_baseline import prepare_images
    from src.prototype.app import load_ocr_selection, load_ui_settings, make_real_backend
    exp_root = exp_root or os.path.join(drive_root, "experiments")
    out_dir = os.path.join(exp_root, LIVE_RUN)
    items, sample_info = [], {}
    for ds in DATASETS:
        data_file = os.path.join(drive_root, DATA_REL[ds])
        samples = load_vivqa(data_file, image_dir=prepare_images(ds, drive_root))
        for s in samples:                                   # tên ảnh = khoá OCR cache (như prototype_check)
            s["image"] = os.path.basename(s["image_path"])
        smp = build_sample(samples, n_images)
        miss = [s["image_path"] for s in smp if not os.path.exists(s["image_path"])]
        if miss:
            raise FileNotFoundError(f"{len(miss)} ảnh thiếu, vd {miss[0]}")
        sample_info[ds] = {"n_images": len({s['image'] for s in smp}), "n_questions": len(smp),
                           "question_ids_sha256": sample_id_hash(smp), "data_file": DATA_REL[ds]}
        items += [{"dataset": ds, "question_id": s["question_id"], "image": s["image"], "image_path": s["image_path"],
                   "question": s["question"], "answers": s.get("answers")} for s in smp]
    _log(f"[live] mẫu: {json.dumps(sample_info, ensure_ascii=False)}")
    st, ocr_cfg = load_ui_settings(), load_ocr_selection()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    backend, client, start = make_real_backend(adapter, config, st, ocr_cfg, ocr_device=ocr_device,
                                               ocr_log=os.path.join(out_dir, "ocr_worker.log"),
                                               expect_adapter_sha=expect_adapter_sha)
    _log(f"[live] khởi động {time.time() - t0:.1f} s | OCR {json.dumps(start['ocr_check'], ensure_ascii=False)}")
    try:
        t1 = time.time()
        recs = run_items(items, backend.ask, os.path.join(out_dir, "records.jsonl"), os.path.join(out_dir, "_tmp"))
        mem = gpu_memory_snapshot()
    finally:
        client.close()
    import transformers
    meta = {"run_name": LIVE_RUN, "live": LIVE_VERSION, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sample": sample_info, "n_images_per_dataset": n_images, "seed": SAMPLE_SEED,
            "ocr_device": ocr_device, "ocr_selection": ocr_cfg, "startup": start,
            "sec_run": round(time.time() - t1, 1),
            "peak_vram_gb_main": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
            "gpu_memory_end": mem,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
    _write_json(os.path.join(out_dir, "meta.json"), meta)
    _log(f"== xong {len(recs)} câu | {meta['sec_run'] / 60:.1f} phút")
    return {"meta": meta, "records": recs}


# =============================================================================
# 3) PHÂN TÍCH (hàm thuần)
# =============================================================================
def percentile(vals: Sequence[float], q: float) -> Optional[float]:
    v = sorted(x for x in vals if isinstance(x, (int, float)))
    if not v:
        return None
    i = (len(v) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def ocr_agreement(recs: List[Dict[str, Any]], cache_texts: Dict[str, List[str]]) -> Dict[str, Any]:
    """Theo ẢNH: văn bản OCR vào prompt của OCR trực tiếp (câu đầu của ảnh) == cache?"""
    first = {}
    for r in recs:
        first.setdefault(r.get("cache_key", r["image"]), r["ocr_texts"])
    diff = [{"image": im, "live": t, "cache": cache_texts.get(im)} for im, t in first.items() if t != cache_texts.get(im)]
    n = len(first)
    return {"n_images": n, "identical": 1 - len(diff) / max(n, 1), "n_diff": len(diff), "diffs": diff[:15]}


def answer_agreement(recs: List[Dict[str, Any]], ref_preds: Dict[str, str]) -> Dict[str, Any]:
    from src.eval.metrics import exact_match
    from src.prototype.core import is_yes_no_question
    kept = [r for r in recs if not is_yes_no_question(r["question"])]
    skipped = len(recs) - len(kept)
    miss = [r["question_id"] for r in kept if r["question_id"] not in ref_preds]
    if miss:
        raise ValueError(f"{len(miss)} câu không có trong run W11, vd {miss[0]}")
    bad = [{"question_id": r["question_id"], "live": r["prediction"], "w11": ref_preds[r["question_id"]]}
           for r in kept if r["prediction"].strip() != str(ref_preds[r["question_id"]]).strip()]
    em_live = 100 * sum(exact_match(r["prediction"], r["answers"]) for r in kept) / max(len(kept), 1)
    em_w11 = 100 * sum(exact_match(ref_preds[r["question_id"]], r["answers"]) for r in kept) / max(len(kept), 1)
    return {"n": len(kept), "skipped_yesno": skipped, "identical": 1 - len(bad) / max(len(kept), 1),
            "n_mismatch": len(bad), "mismatches": bad[:15], "em_live": em_live, "em_w11": em_w11,
            "em_delta": em_live - em_w11}


def latency(recs: List[Dict[str, Any]], warmup: int = WARMUP) -> Dict[str, Any]:
    rs = recs[warmup:]
    out = {"n": len(rs), "warmup_excluded": min(warmup, len(recs))}
    for k in ("sec_ocr", "sec_vlm", "sec_total"):
        vals = [r[k] for r in rs]
        out[k] = {"p50": percentile(vals, .5), "p95": percentile(vals, .95), "max": max(vals) if vals else None}
    return out


def verdicts(ocr: Dict[str, Dict[str, Any]], ans: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    r_ocr = all(ocr[d]["identical"] >= OCR_IDENTICAL_MIN for d in ocr)
    r_live = all(ans[d]["identical"] >= ANSWER_IDENTICAL_MIN and ans[d]["em_delta"] >= -EM_DROP_MAX for d in ans)
    return {"R-OCR": "ĐẠT" if r_ocr else "KHÔNG ĐẠT", "R-LIVE": "ĐẠT" if r_live else "KHÔNG ĐẠT"}


def analyze(exp_root: str, drive_root: str, out: str) -> Dict[str, Any]:
    from src.data.ocr import default_cache_path, lines_to_texts, load_ocr_cache
    from src.eval.prototype_check import REF_RUN
    from src.prototype.app import load_ocr_selection
    run_dir = os.path.join(exp_root, LIVE_RUN)
    meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "records.jsonl"), encoding="utf-8") if l.strip()]
    want = sum(meta["sample"][d]["n_questions"] for d in DATASETS)
    if len(recs) != want:
        raise ValueError(f"records.jsonl có {len(recs)} câu, mẫu cần {want} -> chạy --run lại (tự chạy tiếp)")
    sel = load_ocr_selection()
    res: Dict[str, Any] = {"live": LIVE_VERSION, "rules": {"n_images": meta["n_images_per_dataset"], "seed": SAMPLE_SEED,
                           "ocr_identical_min": OCR_IDENTICAL_MIN, "answer_identical_min": ANSWER_IDENTICAL_MIN,
                           "em_drop_max": EM_DROP_MAX, "warmup": WARMUP},
                           "ocr": {}, "answers": {}, "latency": {}, "meta": meta}
    for ds in DATASETS:
        rr = [r for r in recs if r["dataset"] == ds]
        cache = load_ocr_cache(default_cache_path(drive_root, ds, sel["backend"]))
        ct = {im: lines_to_texts(cache["images"][im], sel["max_chars"], sel["min_conf"])[0]
              for im in {r["cache_key"] for r in rr} if im in cache["images"]}
        res["ocr"][ds] = ocr_agreement(rr, ct)
        preds = json.load(open(os.path.join(exp_root, REF_RUN[ds], "predictions.json"), encoding="utf-8"))
        res["answers"][ds] = answer_agreement(rr, {p["question_id"]: p["prediction"] for p in preds})
        res["latency"][ds] = latency(rr, WARMUP if ds == DATASETS[0] else 0)
    res["latency"]["all"] = latency(recs)
    res["levels"] = {lv: sum(r["level"] == lv for r in recs) for lv in ("Cao", "Trung bình", "Thấp")}
    res["n_untouched_images"] = sum(r["image_untouched"] for r in recs)
    res["verdicts"] = verdicts(res["ocr"], res["answers"])
    os.makedirs(out, exist_ok=True)
    _write_json(os.path.join(out, "prototype_live_summary.json"), res)
    md = report(res)
    with open(os.path.join(out, "prototype_live_summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(md)
    return res


def _f(x: Optional[float], nd: int = 2) -> str:
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def report(res: Dict[str, Any]) -> str:
    m = res["meta"]
    L = ["# W13 bước 8 — prototype chạy thật (OCR trực tiếp + model)", "",
         f"OCR: tiến trình riêng, device `{m['ocr_device']}` · engine trùng cache: "
         f"{'✅' if m['startup']['ocr_check']['ok'] else '❌'}"
         + (f" ({m['startup']['ocr_check']['device_note']})" if m['startup']['ocr_check'].get('device_note') else ""), "",
         "## 1. R-OCR — văn bản OCR vào prompt: trực tiếp = cache?", "",
         "| Tập | Ảnh | Trùng hệt | Khác |", "|---|---|---|---|"]
    for d, o in res["ocr"].items():
        L.append(f"| {d} | {o['n_images']} | {_f(100 * o['identical'], 1)}% | {o['n_diff']} |")
    L += ["", f"➜ **R-OCR: {res['verdicts']['R-OCR']}** (ngưỡng ≥ {_f(100 * OCR_IDENTICAL_MIN, 0)}% từng tập)", ""]
    for d, o in res["ocr"].items():
        for x in o["diffs"][:3]:
            L.append(f"- {d} `{x['image']}`: trực tiếp {x['live'][:6]} · cache {(x['cache'] or [])[:6]}")
    L += ["", "## 2. R-LIVE — câu trả lời qua đường giao diện = run eval W11?", "",
          "| Tập | Câu (bỏ có/không) | Trùng hệt | EM trực tiếp | EM W11 | Δ |", "|---|---|---|---|---|---|"]
    for d, a in res["answers"].items():
        L.append(f"| {d} | {a['n']} ({a['skipped_yesno']}) | {_f(100 * a['identical'], 1)}% | {_f(a['em_live'])} | "
                 f"{_f(a['em_w11'])} | {_f(a['em_delta'])} |")
    L += ["", f"➜ **R-LIVE: {res['verdicts']['R-LIVE']}** (trùng ≥ {_f(100 * ANSWER_IDENTICAL_MIN, 0)}% và Δ EM ≥ "
          f"−{_f(EM_DROP_MAX, 1)} điểm, từng tập)", ""]
    for d, a in res["answers"].items():
        for x in a["mismatches"][:3]:
            L.append(f"- {d} {x['question_id']}: trực tiếp «{x['live']}» · W11 «{x['w11']}»")
    L += ["", "## 3. Chỉ mô tả — tốc độ, bộ nhớ", "",
          f"Khởi động: OCR {_f(m['startup']['sec_start_ocr'], 1)} s · tổng (gồm nạp model) {_f(m['startup']['sec_start_total'], 1)} s · "
          f"GPU {m['gpu']} · VRAM đỉnh tiến trình chính {_f(m['peak_vram_gb_main'])} GB · "
          f"VRAM cuối (mọi tiến trình) {m['gpu_memory_end'].get('total_mib', '—')} MiB", "",
          "| Tập | n | OCR p50 / p95 (s) | Model p50 / p95 (s) | Tổng p50 / p95 (s) |", "|---|---|---|---|---|"]
    for d, l in res["latency"].items():
        L.append(f"| {d} | {l['n']} | {_f(l['sec_ocr']['p50'])} / {_f(l['sec_ocr']['p95'])} | "
                 f"{_f(l['sec_vlm']['p50'])} / {_f(l['sec_vlm']['p95'])} | {_f(l['sec_total']['p50'])} / {_f(l['sec_total']['p95'])} |")
    L += ["", f"Mức tin cậy trên mẫu: {res['levels']} · ảnh đi nguyên file: {res['n_untouched_images']}", ""]
    return "\n".join(L)


# =============================================================================
# 4) ẢNH THẬT CỦA NHÓM (chỉ mô tả; không commit ảnh)
# =============================================================================
COLS = {"image": ("anh", "ảnh", "image"), "question": ("cau_hoi", "câu hỏi", "question"),
        "gold": ("dap_an", "đáp án", "gold", "answer")}


def read_questions_csv(path: str) -> List[Dict[str, Any]]:
    """CSV (Excel 'CSV UTF-8', dấu phân cách ',' hoặc ';') cột anh, cau_hoi, [dap_an]. Đáp án nhiều cách viết: tách bằng '|'."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    head = text.splitlines()[0] if text.strip() else ""
    delim = ";" if head.count(";") > head.count(",") else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delim))
    if not rows:
        raise ValueError("file câu hỏi rỗng")

    def col(name):
        keys = {k.strip().lower(): k for k in rows[0].keys() if k}
        for c in COLS[name]:
            if c in keys:
                return keys[c]
        return None
    ci, cq, cg = col("image"), col("question"), col("gold")
    if not ci or not cq:
        raise ValueError(f"CSV cần cột 'anh' và 'cau_hoi' (đang có {list(rows[0].keys())})")
    out = []
    for i, r in enumerate(rows, 1):
        im, q = (r.get(ci) or "").strip(), (r.get(cq) or "").strip()
        if not im and not q:
            continue
        if not im or not q:
            raise ValueError(f"dòng {i + 1}: thiếu ảnh hoặc câu hỏi")
        gold = [g.strip() for g in (r.get(cg) or "").split("|") if g.strip()] if cg else []
        out.append({"question_id": f"real_{i:03d}", "image": im, "question": q, "answers": gold or None})
    return out


def run_realphotos(photos_dir: str, questions_csv: str, adapter: str, expect_adapter_sha: Optional[str],
                   ocr_device: str, out: str, config: str) -> Dict[str, Any]:
    from src.eval.metrics import exact_match
    from src.prototype.app import load_ocr_selection, load_ui_settings, make_real_backend
    qs = read_questions_csv(questions_csv)
    for q in qs:
        q["image_path"] = os.path.join(photos_dir, q["image"])
        q["dataset"] = "real"
        if not os.path.exists(q["image_path"]):
            raise FileNotFoundError(f"không thấy ảnh {q['image']} trong {photos_dir}")
    st, ocr_cfg = load_ui_settings(), load_ocr_selection()
    backend, client, start = make_real_backend(adapter, config, st, ocr_cfg, ocr_device=ocr_device,
                                               ocr_log=os.path.join(out, "ocr_worker.log"),
                                               expect_adapter_sha=expect_adapter_sha)
    try:
        recs = run_items(qs, backend.ask, os.path.join(out, "records.jsonl"), os.path.join(out, "_tmp"))
    finally:
        client.close()
    for r in recs:
        r["em"] = exact_match(r["prediction"], r["answers"]) if r.get("answers") else None
    graded = [r for r in recs if r["em"] is not None]
    L = ["# W13 bước 8 — ảnh thật của nhóm (chỉ mô tả)", "",
         f"{len(recs)} câu · {len({r['image'] for r in recs})} ảnh"
         + (f" · EM trên {len(graded)} câu có đáp án: {_f(100 * sum(r['em'] for r in graded) / len(graded), 1)}%" if graded else ""), "",
         "| # | Ảnh | Câu hỏi | Trả lời | Đáp án | Mức | Chế độ | Dòng OCR | Ảnh gốc → dùng | Tổng (s) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(recs, 1):
        L.append(f"| {i} | {r['image']} | {r['question']} | {r['prediction']} | {' / '.join(r['answers'] or []) or '—'} | "
                 f"{r['level']} | {r['mode']} | {r['n_ocr_lines']} | {r['img_size'][0]}×{r['img_size'][1]}"
                 f"{'' if r['image_untouched'] else ' (xoay/thu nhỏ)'} | {_f(r['sec_total'])} |")
    md = "\n".join(L) + "\n"
    _write_json(os.path.join(out, "realphotos_summary.json"), {"live": LIVE_VERSION, "startup": start, "records": recs,
                                                             "latency": latency(recs, 1)})
    with open(os.path.join(out, "realphotos_summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(md)
    return {"records": recs}


# =============================================================================
# 5) KIỂM THỬ OFFLINE
# =============================================================================
def _selftest() -> None:
    import tempfile
    from PIL import Image
    tmp = tempfile.mkdtemp(prefix="vvlive_")
    # (1) mẫu tất định, lấy MỌI câu của ảnh được chọn
    smp = [{"question_id": f"q{i}", "image": f"im{i // 2}.jpg", "question": "x"} for i in range(40)]
    a, b = build_sample(smp, 5), build_sample(list(smp), 5)
    assert a == b and len({s["image"] for s in a}) == 5 and len(a) == 10
    assert sample_id_hash(a) == sample_id_hash(b) and build_sample(smp, 5, seed=1) != a
    # (2) chạy qua đường giao diện với ask giả + resume
    p = os.path.join(tmp, "a.jpg")
    Image.new("RGB", (64, 48), "white").save(p)
    calls = []

    def ask(path, q):
        calls.append(path)
        return {"answer": "đỏ" if "màu" in q else "có", "mode": "open", "level": "Cao", "confidence": 0.99,
                "ocr_texts": ["ABC"], "ocr_lines": [{"text": "ABC"}], "sec_ocr": 0.1, "sec": 0.5, "sec_total": 0.6}
    items = [{"dataset": "vivqa", "question_id": f"t{i}", "image": "a.jpg", "image_path": p,
              "question": "màu gì" if i % 2 else "có mèo không", "answers": ["đỏ"]} for i in range(8)]
    out = os.path.join(tmp, "r", "records.jsonl")
    run_items(items[:5], ask, out, os.path.join(tmp, "t"))
    recs = run_items(items, ask, out, os.path.join(tmp, "t"))
    assert len(calls) == 8 and len(recs) == 8 and all(c == p for c in calls)      # resume + ảnh đi nguyên file
    assert recs[0]["image_untouched"] and recs[0]["n_ocr_lines"] == 1
    # (3) so OCR, so câu trả lời (bỏ câu có/không), latency, phán quyết
    o = ocr_agreement(recs, {"a.jpg": ["ABC"]})
    assert o["identical"] == 1.0 and ocr_agreement(recs, {"a.jpg": ["ABD"]})["n_diff"] == 1
    ref = {f"t{i}": "đỏ" for i in range(8)}
    an = answer_agreement(recs, ref)
    assert an["skipped_yesno"] == 4 and an["n"] == 4 and an["identical"] == 1.0 and an["em_delta"] == 0
    an2 = answer_agreement(recs, {**ref, "t1": "xanh"})
    assert an2["n_mismatch"] == 1 and abs(an2["em_delta"] - 25.0) < 1e-9          # W11 sai, live đúng -> Δ +25
    lat = latency(recs, 2)
    assert lat["n"] == 6 and lat["sec_total"]["p50"] == 0.6 and percentile([1, 2, 3, 4], .5) == 2.5
    good = {"vivqa": {"identical": 0.99}, "vitextvqa_official": {"identical": 0.98}}
    ga = {"vivqa": {"identical": 0.97, "em_delta": -1.5}, "vitextvqa_official": {"identical": 1.0, "em_delta": 0.3}}
    assert verdicts(good, ga) == {"R-OCR": "ĐẠT", "R-LIVE": "ĐẠT"}
    assert verdicts({**good, "vivqa": {"identical": 0.979}}, ga)["R-OCR"] == "KHÔNG ĐẠT"
    assert verdicts(good, {**ga, "vivqa": {"identical": 0.99, "em_delta": -1.6}})["R-LIVE"] == "KHÔNG ĐẠT"
    assert verdicts(good, {**ga, "vivqa": {"identical": 0.96, "em_delta": 0.0}})["R-LIVE"] == "KHÔNG ĐẠT"
    # (4) CSV ảnh thật: Excel ';' + BOM + đáp án nhiều cách viết; dấu ',' ; thiếu cột -> lỗi rõ
    c1 = os.path.join(tmp, "q1.csv")
    with open(c1, "w", encoding="utf-8-sig") as f:
        f.write("anh;cau_hoi;dap_an\nphoto 1.jpg;quán này tên gì?;phở hà nội|phở hà nội 24\nb.jpg;màu gì;\n;;\n")
    q = read_questions_csv(c1)
    assert len(q) == 2 and q[0]["image"] == "photo 1.jpg" and q[0]["answers"] == ["phở hà nội", "phở hà nội 24"]
    assert q[1]["answers"] is None
    c2 = os.path.join(tmp, "q2.csv")
    with open(c2, "w", encoding="utf-8") as f:
        f.write("image,question\na.jpg,có gì\n")
    assert read_questions_csv(c2)[0]["question"] == "có gì"
    c3 = os.path.join(tmp, "q3.csv")
    with open(c3, "w", encoding="utf-8") as f:
        f.write("ten,noi_dung\na.jpg,x\n")
    try:
        read_questions_csv(c3)
        raise AssertionError("thiếu cột phải báo lỗi")
    except ValueError as e:
        assert "anh" in str(e)
    # (5) báo cáo dựng được
    res = {"meta": {"ocr_device": "gpu", "startup": {"ocr_check": {"ok": True, "device_note": None}, "sec_start_ocr": 30.0,
                                                     "sec_start_total": 95.0},
                    "gpu": "NVIDIA L4", "peak_vram_gb_main": 2.6, "gpu_memory_end": {"total_mib": 5000}},
           "ocr": {"vivqa": o}, "answers": {"vivqa": an}, "latency": {"vivqa": lat}, "levels": {"Cao": 8},
           "n_untouched_images": 8, "verdicts": verdicts({"vivqa": o}, {"vivqa": an})}
    md = report(res)
    assert "R-OCR: ĐẠT" in md and "R-LIVE: ĐẠT" in md and "0,60" in md
    print("✅ selftest prototype_live OK (mẫu tất định, đường giao diện + resume, R-OCR, R-LIVE bỏ câu có/không, "
          "latency, CSV ảnh thật, báo cáo)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--realphotos", action="store_true")
    ap.add_argument("--drive_root", default="/content/drive/MyDrive/ViVQA-VLM")
    ap.add_argument("--exp_root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--adapter")
    ap.add_argument("--expect_adapter_sha")
    ap.add_argument("--ocr_device", default="gpu")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "qwen_lora.yaml"))
    ap.add_argument("--n_images", type=int, default=N_IMAGES)
    ap.add_argument("--photos_dir")
    ap.add_argument("--questions_csv")
    a = ap.parse_args()
    exp_root = a.exp_root or os.path.join(a.drive_root, "experiments")
    if a.selftest:
        _selftest()
    elif a.run:
        if a.n_images != N_IMAGES:
            raise SystemExit(f"--n_images khác {N_IMAGES} = không phải run của luật")
        run_live(a.drive_root, a.adapter, a.expect_adapter_sha, a.ocr_device, exp_root, a.config)
    elif a.analyze:
        analyze(exp_root, a.drive_root, a.out or os.path.join(exp_root, OUT_NAME))
    elif a.realphotos:
        run_realphotos(a.photos_dir, a.questions_csv, a.adapter, a.expect_adapter_sha, a.ocr_device,
                       a.out or os.path.join(exp_root, REAL_RUN), a.config)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
