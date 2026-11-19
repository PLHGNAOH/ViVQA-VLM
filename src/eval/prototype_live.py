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

W14 — phiên bản v2 (mặc định; `--version v1` tái lập đúng run W13). Luật ghi trước ở docs/W14/README.md, mục "Bước 3":
  - giao diện thu nhỏ ảnh > 6.000 px (W13: 2.048 px) -> mọi ảnh dataset đi nguyên file như lúc eval;
  - OCR cache ViTextVQA so sánh = cache đã sửa EXIF (paddledet_vietocr_exiffix.json);
  R-OCR v2 : kiểm ĐƯỜNG GIAO DIỆN, tách khỏi sai khác số học giữa GPU: văn bản OCR vào prompt qua giao diện
             (prepare_image -> tiến trình OCR) TRÙNG HỆT OCR đọc thẳng file gốc bằng CÙNG tiến trình OCR, cùng phiên
             L4, ở ≥ 98% ảnh từng tập. So với cache (dựng trên T4) chỉ mô tả: % trùng hệt + độ giống chuỗi.
  R-LIVE v2: như v1, BỎ thêm câu của ảnh có cờ xoay EXIF (giao diện xoay ảnh, eval W11 thì không).
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

VERSIONS: Dict[str, Dict[str, Any]] = {
    "v1": {"live": "prototype-live-v1", "run": "W13_LIVE_sh_ocrlive_sample200img_bf16_L4_seed42",
           "out": "W13_prototype_live_L4_seed42", "real": "W13_REAL_sh_realphotos_bf16_L4", "max_side": 2048,
           "cache_suffix": {}, "ocr_sim_min": None, "exclude_rotated": False, "title": "W13 bước 8"},
    "v2": {"live": "prototype-live-v2", "run": "W14_LIVE_sh_ocrlive_v2_sample200img_bf16_L4_seed42",
           "out": "W14_prototype_live_v2_L4_seed42", "real": "W14_REAL_sh_realphotos_v2_bf16_L4", "max_side": 6000,
           "cache_suffix": {"vitextvqa_official": "_exiffix"}, "ocr_sim_min": 0.9, "exclude_rotated": True,
           "ocr_pipeline_ref": "direct",
           "title": "W14 bước 3"},
}
DEFAULT_VERSION = "v2"
LIVE_VERSION = VERSIONS[DEFAULT_VERSION]["live"]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = ("vivqa", "vitextvqa_official")
SAMPLE_SEED = 42
N_IMAGES = 200                 # ảnh / tập
OCR_IDENTICAL_MIN = 0.98       # R-OCR
ANSWER_IDENTICAL_MIN = 0.97    # R-LIVE
EM_DROP_MAX = 1.5              # R-LIVE (điểm EM, như δ W11)
WARMUP = 5                     # bỏ 5 câu đầu khỏi thống kê latency
LIVE_RUN = VERSIONS[DEFAULT_VERSION]["run"]
OUT_NAME = VERSIONS[DEFAULT_VERSION]["out"]
REAL_RUN = VERSIONS[DEFAULT_VERSION]["real"]
DIRECT_OCR_FILE = "direct_ocr.json"   # v2: OCR đọc thẳng file gốc, cùng tiến trình OCR với giao diện


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
              tmp_dir: str, max_side: Optional[int] = None) -> List[Dict[str, Any]]:
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
            used, info = (prepare_image(it["image_path"], tmp_dir) if max_side is None
                          else prepare_image(it["image_path"], tmp_dir, max_side=max_side))
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
                   "image_untouched": not info["reencoded"], "img_size": info["orig_size"],
                   "rotated": bool(info.get("rotated")), "resized": bool(info.get("resized"))}
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
             config: str, n_images: int = N_IMAGES, version: str = DEFAULT_VERSION) -> Dict[str, Any]:
    import torch
    from src.data.vivqa_dataset import load_vivqa
    from src.eval.prototype_check import DATA_REL
    from src.eval.run_baseline import prepare_images
    from src.prototype.app import load_ocr_selection, load_ui_settings, make_real_backend
    V = VERSIONS[version]
    exp_root = exp_root or os.path.join(drive_root, "experiments")
    out_dir = os.path.join(exp_root, V["run"])
    os.makedirs(out_dir, exist_ok=True)                   # notebook 24 lần 1: thiếu dòng này -> lỗi mở log OCR
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
        recs = run_items(items, backend.ask, os.path.join(out_dir, "records.jsonl"), os.path.join(out_dir, "_tmp"),
                         max_side=V["max_side"])
        mem = gpu_memory_snapshot()
        if V.get("ocr_pipeline_ref") == "direct":
            direct_path = os.path.join(out_dir, DIRECT_OCR_FILE)
            direct = json.load(open(direct_path, encoding="utf-8")) if os.path.exists(direct_path) else {}
            todo = {(it["dataset"], os.path.basename(it["image_path"])): it["image_path"] for it in items}
            todo = {k: p for k, p in todo.items() if k[1] not in direct.get(k[0], {})}
            _log(f"[live] OCR đọc thẳng file gốc: còn {len(todo)} ảnh")
            from src.data.ocr import lines_to_texts
            for i, ((ds, key), path) in enumerate(sorted(todo.items()), 1):
                direct.setdefault(ds, {})[key] = lines_to_texts({"lines": client.lines(path)}, ocr_cfg["max_chars"],
                                                                ocr_cfg["min_conf"])[0]
                if i % 50 == 0 or i == len(todo):
                    _write_json(direct_path, direct)
                    _log(f"[live] OCR thẳng {i}/{len(todo)}")
    finally:
        client.close()
    import transformers
    meta = {"run_name": V["run"], "live": V["live"], "version": version, "max_side": V["max_side"], "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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


def text_similarity(a: Optional[List[str]], b: Optional[List[str]]) -> float:
    """Độ giống (difflib ratio) của 2 văn bản OCR vào prompt (ghép ' | '). Cả hai rỗng = 1; một bên rỗng = 0."""
    import difflib
    sa, sb = " | ".join(a or []), " | ".join(b or [])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return difflib.SequenceMatcher(None, sa, sb, autojunk=False).ratio()


def ocr_agreement(recs: List[Dict[str, Any]], cache_texts: Dict[str, List[str]],
                  sim_min: Optional[float] = None) -> Dict[str, Any]:
    """Theo ẢNH: văn bản OCR vào prompt của OCR trực tiếp (câu đầu của ảnh) == cache? (v2: thêm độ giống)"""
    first, rot = {}, {}
    for r in recs:
        first.setdefault(r.get("cache_key", r["image"]), r["ocr_texts"])
        rot.setdefault(r.get("cache_key", r["image"]), bool(r.get("rotated")))
    diff = [{"image": im, "live": t, "cache": cache_texts.get(im), "rotated": rot[im],
             "similarity": round(text_similarity(t, cache_texts.get(im)), 4)}
            for im, t in first.items() if t != cache_texts.get(im)]
    n = len(first)
    out = {"n_images": n, "identical": 1 - len(diff) / max(n, 1), "n_diff": len(diff),
           "diffs": sorted(diff, key=lambda x: x["similarity"])[:15]}
    if sim_min is not None:
        low = [x for x in diff if x["similarity"] < sim_min]
        out.update({"sim_min": sim_min, "n_below_sim": len(low), "similar": 1 - len(low) / max(n, 1),
                    "min_similarity": min([x["similarity"] for x in diff], default=1.0)})
    return out


def answer_agreement(recs: List[Dict[str, Any]], ref_preds: Dict[str, str],
                     exclude_rotated: bool = False) -> Dict[str, Any]:
    from src.eval.metrics import exact_match
    from src.prototype.core import is_yes_no_question
    no_yn = [r for r in recs if not is_yes_no_question(r["question"])]
    kept = [r for r in no_yn if not (exclude_rotated and r.get("rotated"))]
    skipped = len(recs) - len(no_yn)
    skipped_rot = len(no_yn) - len(kept)
    miss = [r["question_id"] for r in kept if r["question_id"] not in ref_preds]
    if miss:
        raise ValueError(f"{len(miss)} câu không có trong run W11, vd {miss[0]}")
    bad = [{"question_id": r["question_id"], "live": r["prediction"], "w11": ref_preds[r["question_id"]]}
           for r in kept if r["prediction"].strip() != str(ref_preds[r["question_id"]]).strip()]
    em_live = 100 * sum(exact_match(r["prediction"], r["answers"]) for r in kept) / max(len(kept), 1)
    em_w11 = 100 * sum(exact_match(ref_preds[r["question_id"]], r["answers"]) for r in kept) / max(len(kept), 1)
    return {"n": len(kept), "skipped_yesno": skipped, "skipped_rotated": skipped_rot,
            "identical": 1 - len(bad) / max(len(kept), 1),
            "n_mismatch": len(bad), "mismatches": bad[:15], "em_live": em_live, "em_w11": em_w11,
            "em_delta": em_live - em_w11}


def latency(recs: List[Dict[str, Any]], warmup: int = WARMUP) -> Dict[str, Any]:
    rs = recs[warmup:]
    out = {"n": len(rs), "warmup_excluded": min(warmup, len(recs))}
    for k in ("sec_ocr", "sec_vlm", "sec_total"):
        vals = [r[k] for r in rs]
        out[k] = {"p50": percentile(vals, .5), "p95": percentile(vals, .95), "max": max(vals) if vals else None}
    return out


def verdicts(ocr: Dict[str, Dict[str, Any]], ans: Dict[str, Dict[str, Any]],
             pipeline: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, str]:
    """v1: R-OCR = giao diện vs cache. v2 (truyền `pipeline`): R-OCR = giao diện vs OCR đọc thẳng file, cùng phiên."""
    base = pipeline if pipeline is not None else ocr
    r_ocr = all(base[d]["identical"] >= OCR_IDENTICAL_MIN for d in base)
    r_live = all(ans[d]["identical"] >= ANSWER_IDENTICAL_MIN and ans[d]["em_delta"] >= -EM_DROP_MAX for d in ans)
    return {"R-OCR": "ĐẠT" if r_ocr else "KHÔNG ĐẠT", "R-LIVE": "ĐẠT" if r_live else "KHÔNG ĐẠT"}


def analyze(exp_root: str, drive_root: str, out: str, version: str = DEFAULT_VERSION) -> Dict[str, Any]:
    from src.data.ocr import default_cache_path, lines_to_texts, load_ocr_cache
    from src.eval.prototype_check import REF_RUN
    from src.prototype.app import load_ocr_selection
    V = VERSIONS[version]
    run_dir = os.path.join(exp_root, V["run"])
    meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "records.jsonl"), encoding="utf-8") if l.strip()]
    want = sum(meta["sample"][d]["n_questions"] for d in DATASETS)
    if len(recs) != want:
        raise ValueError(f"records.jsonl có {len(recs)} câu, mẫu cần {want} -> chạy --run lại (tự chạy tiếp)")
    sel = load_ocr_selection()
    res: Dict[str, Any] = {"live": V["live"], "version": version,
                           "rules": {"n_images": meta["n_images_per_dataset"], "seed": SAMPLE_SEED,
                           "ocr_identical_min": OCR_IDENTICAL_MIN, "answer_identical_min": ANSWER_IDENTICAL_MIN,
                           "em_drop_max": EM_DROP_MAX, "warmup": WARMUP, "max_side": V["max_side"],
                           "ocr_sim_min_descriptive": V["ocr_sim_min"], "ocr_pipeline_ref": V.get("ocr_pipeline_ref"),
                           "exclude_rotated": V["exclude_rotated"], "cache_suffix": V["cache_suffix"]},
                           "ocr": {}, "answers": {}, "latency": {}, "meta": meta}
    for ds in DATASETS:
        rr = [r for r in recs if r["dataset"] == ds]
        cpath = default_cache_path(drive_root, ds, sel["backend"]).replace(".json", V["cache_suffix"].get(ds, "") + ".json")
        cache = load_ocr_cache(cpath)
        res.setdefault("caches", {})[ds] = os.path.basename(cpath)
        ct = {im: lines_to_texts(cache["images"][im], sel["max_chars"], sel["min_conf"])[0]
              for im in {r["cache_key"] for r in rr} if im in cache["images"]}
        res["ocr"][ds] = ocr_agreement(rr, ct, V["ocr_sim_min"])
        if V.get("ocr_pipeline_ref") == "direct":
            direct = json.load(open(os.path.join(run_dir, DIRECT_OCR_FILE), encoding="utf-8"))
            miss = [r["cache_key"] for r in rr if r["cache_key"] not in direct.get(ds, {})]
            if miss:
                raise ValueError(f"{DIRECT_OCR_FILE} thiếu {len(miss)} ảnh {ds} -> chạy --run lại (tự chạy tiếp)")
            res.setdefault("ocr_pipeline", {})[ds] = ocr_agreement(rr, direct[ds], V["ocr_sim_min"])
        preds = json.load(open(os.path.join(exp_root, REF_RUN[ds], "predictions.json"), encoding="utf-8"))
        res["answers"][ds] = answer_agreement(rr, {p["question_id"]: p["prediction"] for p in preds}, V["exclude_rotated"])
        res["latency"][ds] = latency(rr, WARMUP if ds == DATASETS[0] else 0)
    res["latency"]["all"] = latency(recs)
    res["levels"] = {lv: sum(r["level"] == lv for r in recs) for lv in ("Cao", "Trung bình", "Thấp")}
    res["n_untouched_images"] = sum(r["image_untouched"] for r in recs)
    res["n_rotated_records"] = sum(bool(r.get("rotated")) for r in recs)
    res["n_resized_records"] = sum(bool(r.get("resized")) for r in recs)
    res["verdicts"] = verdicts(res["ocr"], res["answers"], res.get("ocr_pipeline"))
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
    V = VERSIONS[res.get("version", "v1")]
    L = [f"# {V['title']} — prototype chạy thật (OCR trực tiếp + model), {V['live']}", "",
         f"OCR: tiến trình riêng, device `{m['ocr_device']}` · engine trùng cache: "
         f"{'✅' if m['startup']['ocr_check']['ok'] else '❌'}"
         + (f" ({m['startup']['ocr_check']['device_note']})" if m['startup']['ocr_check'].get('device_note') else ""), "",
         ("## 1. R-OCR — văn bản OCR vào prompt qua giao diện = OCR đọc thẳng file gốc (cùng phiên)?"
          if res.get("ocr_pipeline") else "## 1. R-OCR — văn bản OCR vào prompt: trực tiếp = cache?"), "",
         "| Tập | Ảnh | Trùng hệt | Khác | Độ giống ≥ ngưỡng | Độ giống thấp nhất |", "|---|---|---|---|---|---|"]
    for d, o in res["ocr"].items():
        sim = f"{_f(100 * o['similar'], 1)}%" if "similar" in o else "—"
        L.append(f"| {d} | {o['n_images']} | {_f(100 * o['identical'], 1)}% | {o['n_diff']} | {sim} | "
                 f"{_f(o.get('min_similarity'), 3) if 'similar' in o else '—'} |")
    if res.get("ocr_pipeline"):
        L += ["", "Bảng trên: **so với cache** (dựng trên T4) — chỉ mô tả. Bảng dưới: **luật R-OCR v2** (cùng tiến trình OCR, cùng L4).", "",
              "| Tập | Ảnh | Giao diện = đọc thẳng | Khác |", "|---|---|---|---|"]
        for d, o in res["ocr_pipeline"].items():
            L.append(f"| {d} | {o['n_images']} | {_f(100 * o['identical'], 1)}% | {o['n_diff']} |")
    L += ["", f"➜ **R-OCR: {res['verdicts']['R-OCR']}** (trùng hệt ≥ {_f(100 * OCR_IDENTICAL_MIN, 0)}% từng tập)", ""]
    for d, o in res["ocr"].items():
        for x in o["diffs"][:3]:
            L.append(f"- {d} `{x['image']}` (giống {_f(x.get('similarity'), 3)}): trực tiếp {x['live'][:6]} · cache {(x['cache'] or [])[:6]}")
    L += ["", "## 2. R-LIVE — câu trả lời qua đường giao diện = run eval W11?", "",
          "| Tập | Câu (bỏ có/không · bỏ ảnh xoay) | Trùng hệt | EM trực tiếp | EM W11 | Δ |", "|---|---|---|---|---|---|"]
    for d, a in res["answers"].items():
        L.append(f"| {d} | {a['n']} ({a['skipped_yesno']} · {a.get('skipped_rotated', 0)}) | {_f(100 * a['identical'], 1)}% | "
                 f"{_f(a['em_live'])} | {_f(a['em_w11'])} | {_f(a['em_delta'])} |")
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
    L += ["", f"Mức tin cậy trên mẫu: {res['levels']} · câu có ảnh đi nguyên file: {res['n_untouched_images']}"
          + (f" · ảnh xoay EXIF: {res['n_rotated_records']} câu · ảnh thu nhỏ: {res['n_resized_records']} câu"
             if "n_rotated_records" in res else ""), ""]
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
                   ocr_device: str, out: str, config: str, version: str = DEFAULT_VERSION) -> Dict[str, Any]:
    from src.eval.metrics import exact_match
    from src.prototype.app import load_ocr_selection, load_ui_settings, make_real_backend
    qs = read_questions_csv(questions_csv)
    os.makedirs(out, exist_ok=True)
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
        recs = run_items(qs, backend.ask, os.path.join(out, "records.jsonl"), os.path.join(out, "_tmp"),
                         max_side=VERSIONS[version]["max_side"])
    finally:
        client.close()
    for r in recs:
        r["em"] = exact_match(r["prediction"], r["answers"]) if r.get("answers") else None
    graded = [r for r in recs if r["em"] is not None]
    L = [f"# {VERSIONS[version]['title']} — ảnh thật của nhóm (chỉ mô tả, {VERSIONS[version]['live']})", "",
         f"{len(recs)} câu · {len({r['image'] for r in recs})} ảnh"
         + (f" · EM trên {len(graded)} câu có đáp án: {_f(100 * sum(r['em'] for r in graded) / len(graded), 1)}%" if graded else ""), "",
         "| # | Ảnh | Câu hỏi | Trả lời | Đáp án | Mức | Chế độ | Dòng OCR | Ảnh gốc → dùng | Tổng (s) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(recs, 1):
        L.append(f"| {i} | {r['image']} | {r['question']} | {r['prediction']} | {' / '.join(r['answers'] or []) or '—'} | "
                 f"{r['level']} | {r['mode']} | {r['n_ocr_lines']} | {r['img_size'][0]}×{r['img_size'][1]}"
                 f"{'' if r['image_untouched'] else ' (xoay/thu nhỏ)'} | {_f(r['sec_total'])} |")
    md = "\n".join(L) + "\n"
    _write_json(os.path.join(out, "realphotos_summary.json"), {"live": VERSIONS[version]["live"], "version": version,
                                                             "startup": start, "records": recs,
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
    # (6) W14 v2: ngưỡng ảnh, độ giống OCR, bỏ ảnh xoay khỏi R-LIVE, phán quyết 2 mức
    assert VERSIONS["v1"]["max_side"] == 2048 and VERSIONS["v2"]["max_side"] == 6000 and DEFAULT_VERSION == "v2"
    from src.prototype.app import MAX_SIDE, MAX_SIDE_W13
    assert (MAX_SIDE, MAX_SIDE_W13) == (VERSIONS["v2"]["max_side"], VERSIONS["v1"]["max_side"])
    p_big = os.path.join(tmp, "big.jpg")
    Image.new("RGB", (3000, 100), "white").save(p_big)
    it_big = [{"dataset": "vitextvqa_official", "question_id": "b1", "image": "big.jpg", "image_path": p_big,
               "question": "tên quán?", "answers": ["x"]}]
    seen = []
    r1 = run_items(it_big, lambda path, q: (seen.append(path), ask(path, q))[1], os.path.join(tmp, "v1.jsonl"),
                   os.path.join(tmp, "t"), max_side=VERSIONS["v1"]["max_side"])
    r2 = run_items(it_big, lambda path, q: (seen.append(path), ask(path, q))[1], os.path.join(tmp, "v2.jsonl"),
                   os.path.join(tmp, "t"), max_side=VERSIONS["v2"]["max_side"])
    assert r1[0]["resized"] and not r1[0]["image_untouched"] and seen[0] != p_big
    assert not r2[0]["resized"] and r2[0]["image_untouched"] and seen[1] == p_big
    assert abs(text_similarity(["Ironpot"], ["Ironipor"]) - 0.8) < 0.01 and text_similarity([], []) == 1.0
    assert text_similarity(["A"], []) == 0.0
    recs_v2 = [dict(r, cache_key=f"im{i}.jpg", rotated=(i == 0)) for i, r in enumerate(recs)]
    ct = {f"im{i}.jpg": ["ABC"] for i in range(8)}
    ct["im1.jpg"] = ["ABD"]                               # lệch nhẹ (giống 0,67) -> dưới 0,9
    o2 = ocr_agreement(recs_v2, ct, 0.9)
    assert o2["n_diff"] == 1 and o2["n_below_sim"] == 1 and abs(o2["similar"] - 7 / 8) < 1e-9
    an3 = answer_agreement(recs_v2, ref, exclude_rotated=True)
    assert an3["skipped_rotated"] == 0 and an3["skipped_yesno"] == 4             # ảnh xoay (t0) là câu có/không
    recs_rot = [dict(r, rotated=(r["question_id"] == "t1")) for r in recs]
    an4 = answer_agreement(recs_rot, {**ref, "t1": "xanh"}, exclude_rotated=True)
    assert an4["skipped_rotated"] == 1 and an4["n"] == 3 and an4["n_mismatch"] == 0
    oc = {"vivqa": {"identical": 1.0}, "vitextvqa_official": {"identical": 0.96}}        # lệch cache (khác GPU)
    pipe = {"vivqa": {"identical": 1.0}, "vitextvqa_official": {"identical": 0.99}}      # giao diện = đọc thẳng
    assert verdicts(oc, ga)["R-OCR"] == "KHÔNG ĐẠT" and verdicts(oc, ga, pipe)["R-OCR"] == "ĐẠT"
    assert verdicts(oc, ga, {**pipe, "vivqa": {"identical": 0.97}})["R-OCR"] == "KHÔNG ĐẠT"
    md2 = report({**res, "version": "v2", "ocr": {"vitextvqa_official": o2},
                  "ocr_pipeline": {"vitextvqa_official": ocr_agreement(recs_v2, {f"im{i}.jpg": ["ABC"] for i in range(8)})},
                  "verdicts": verdicts({"vitextvqa_official": o2}, {"vivqa": an}, {"vitextvqa_official": {"identical": 1.0}}),
                  "n_rotated_records": 1, "n_resized_records": 0})
    assert "W14 bước 3" in md2 and "Độ giống" in md2 and "đọc thẳng" in md2 and "R-OCR: ĐẠT" in md2
    print("✅ selftest prototype_live OK (mẫu tất định, đường giao diện + resume, R-OCR, R-LIVE bỏ câu có/không, "
          "latency, CSV ảnh thật, báo cáo, v2: ngưỡng 6000 px, độ giống OCR, bỏ ảnh xoay, R-OCR theo đường giao diện)")


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
    ap.add_argument("--version", default=DEFAULT_VERSION, choices=sorted(VERSIONS),
                    help="v2 (W14, mặc định) | v1 (tái lập đúng run W13)")
    a = ap.parse_args()
    V = VERSIONS[a.version]
    exp_root = a.exp_root or os.path.join(a.drive_root, "experiments")
    if a.selftest:
        _selftest()
    elif a.run:
        if a.n_images != N_IMAGES:
            raise SystemExit(f"--n_images khác {N_IMAGES} = không phải run của luật")
        run_live(a.drive_root, a.adapter, a.expect_adapter_sha, a.ocr_device, exp_root, a.config, version=a.version)
    elif a.analyze:
        analyze(exp_root, a.drive_root, a.out or os.path.join(exp_root, V["out"]), version=a.version)
    elif a.realphotos:
        run_realphotos(a.photos_dir, a.questions_csv, a.adapter, a.expect_adapter_sha, a.ocr_device,
                       a.out or os.path.join(exp_root, V["real"]), a.config, version=a.version)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
