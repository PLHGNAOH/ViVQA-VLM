"""
error_analysis.py — Error analysis của HAI MÔ HÌNH CHÍNH THỨC (W12): B3-full (ViTextVQA) và C3 (ViVQA).

Vấn đề: đã có EM theo loại câu hỏi (W11) và audit tay 200 lỗi của B2 (W07, mô hình cũ), nhưng chưa biết
lỗi của mô hình chính thức đến từ đâu: OCR không đưa được đáp án vào prompt, hay model có đáp án trong OCR mà
vẫn chọn sai. Luật ghi trước: docs/W12/README.md, mục "Error analysis". CHỈ MÔ TẢ, không có phán quyết.

Phân tích (mọi định nghĩa cố định trước khi chạy):
  ViTextVQA full test (10.028 câu) — B1, B2, B3-full:
    A1  kiểu lỗi (answer_format.error_bucket: tone / truncated / extended / near_miss / other) của từng run
    A2  nguồn lỗi của B3-full × kiểu lỗi × loại câu hỏi (qt-v1). Trạng thái OCR của câu (văn bản OCR ĐÚNG phần
        đã vào prompt: VietOCR · max_chars 600 · min_conf 0):
          has_answer        đáp án gold là một cụm từ trong văn bản OCR
          wrong_diacritics  chỉ khớp khi bỏ dấu cả hai phía (OCR đọc đúng chữ, sai dấu)
          answer_missing    OCR có chữ nhưng không chứa đáp án
          no_text           ảnh không đọc ra chữ nào
        Nguồn lỗi (câu B3-full sai EM):
          ocr_upstream      trạng thái OCR ≠ has_answer -> OCR không đưa đáp án vào prompt
          model_wrong_span  has_answer và câu trả lời là một cụm trong OCR -> chép SAI đoạn
          model_outside_ocr has_answer và câu trả lời không nằm trong OCR -> bỏ qua OCR / tự sinh
    A3  chuyển trạng thái B2 -> B3-full (ghép cặp): sửa được / làm hỏng / cùng đúng / cùng sai, theo trạng thái OCR;
        với câu bị làm hỏng: % câu trả lời nằm trong OCR (proxy "OCR làm phân tâm")
  ViVQA test (3.001 câu) — B1, C1, C3:
    V1  kiểu lỗi từng run; kiểu lỗi C3 × loại câu hỏi (4 loại có sẵn)
    V2  10 cặp (gold -> dự đoán) sai nhiều nhất của C3 theo loại câu hỏi
    V3  tách theo ảnh đã thấy / chưa thấy lúc train (stratify_by_overlap, train + val), B1 vs C3
  Ứng viên ví dụ định tính (bước 6): chọn tất định (seed 42), tối đa 5 câu / nhóm.

Chạy (local, đọc Drive qua Google Drive for desktop):
    python -m src.eval.error_analysis --exp_root experiments \
        --vitext_test "G:/My Drive/ViVQA-VLM/data/vitextvqa_official/test.json" \
        --vitext_ocr "G:/My Drive/ViVQA-VLM/data/vitextvqa_official/ocr/paddledet_vietocr.json" \
        --vivqa_dir "G:/My Drive/ViVQA-VLM/data/vivqa" --out experiments/W12_error_analysis
    python -m src.eval.error_analysis --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.ocr import lines_to_texts, load_ocr_cache  # noqa: E402
from src.eval.answer_format import BUCKETS, error_bucket  # noqa: E402
from src.eval.metrics import exact_match  # noqa: E402
from src.eval.ocr_coverage import contains_phrase, norm, strip_diacritics  # noqa: E402
from src.eval.question_types import QT_VERSION, classify, merged_types, rules_sha256  # noqa: E402
from src.eval.stratify_by_overlap import stratify  # noqa: E402

SEED = 42
N_EXAMPLES = 5
OCR_MAX_CHARS, OCR_MIN_CONF = 600, 0.0            # = thiết lập OCR chốt W08 (ocr_selection.json)
OCR_STATUS = ("has_answer", "wrong_diacritics", "answer_missing", "no_text")
SOURCES = ("ocr_upstream", "model_wrong_span", "model_outside_ocr")
TRANSITIONS = ("fixed", "broken", "both_right", "both_wrong")
VIVQA_TYPES = ("object", "number", "color", "location")
RUNS = {
    "vitext": {"B1": "W07_B1_zeroshot_qwen25vl_vitextvqa_testfull_bf16_L4_seed42",
               "B2": "W07_B2_eval_qlora_r16_ep1_vitextvqa_testfull_bf16_L4_seed42",
               "B3full": "W10_B3full_eval_qlora_ocr_vietocr_v1_r16_full_ep1_vitextvqa_testfull_bf16_L4_seed42"},
    "vivqa": {"B1": "W07_B1_zeroshot_qwen25vl_vivqa_test_bf16_L4_seed42",
              "C1": "W07_B2_eval_qlora_r16_ep1_vivqa_test_bf16_L4_seed42",
              "C3": "W09_C3_eval_qlora_r16_vivqa_test_bf16_L4_seed42"},
}
TEST2000_B1 = "W07_B1_zeroshot_qwen25vl_vitextvqa_test2000_bf16_L4_seed42"   # để gộp loại câu hỏi như W11


# =============================================================================
# Tiện ích
# =============================================================================
def _load(path: str) -> Any:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for b in iter(lambda: fp.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _records(path: str) -> List[Dict[str, Any]]:
    raw = _load(path)
    if isinstance(raw, dict):
        return raw["data"] if isinstance(raw.get("data"), list) else list(raw.values())
    return raw


def _pct(c: Counter, keys, n: Optional[int] = None) -> Dict[str, float]:
    n = n if n is not None else sum(c.values())
    return {k: round(100 * c.get(k, 0) / n, 1) if n else 0.0 for k in keys}


def is_right(r: Dict[str, Any]) -> bool:
    return bool(exact_match(r["prediction"], r["answers"]))


def ocr_text_of(entry: Dict[str, Any]) -> str:
    """Văn bản OCR ĐÚNG phần đã vào prompt (cắt 600 ký tự, giữ thứ tự đọc), đã chuẩn hoá để tìm cụm từ."""
    texts, _ = lines_to_texts(entry, OCR_MAX_CHARS, OCR_MIN_CONF)
    return norm(" | ".join(texts))


def ocr_status(ocr_text: str, answers: List[str]) -> str:
    golds = [norm(a) for a in answers if norm(a)]
    if any(contains_phrase(ocr_text, g) for g in golds):
        return "has_answer"
    if not ocr_text:
        return "no_text"
    if any(contains_phrase(strip_diacritics(ocr_text), strip_diacritics(g)) for g in golds):
        return "wrong_diacritics"
    return "answer_missing"


def pred_in_ocr(ocr_text: str, pred: str) -> bool:
    p = norm(pred)
    return bool(p) and contains_phrase(ocr_text, p)


def error_source(status: str, pred_inside: bool) -> str:
    if status != "has_answer":
        return "ocr_upstream"
    return "model_wrong_span" if pred_inside else "model_outside_ocr"


def transition(a_right: bool, b_right: bool) -> str:
    if a_right and b_right:
        return "both_right"
    if b_right:
        return "fixed"
    if a_right:
        return "broken"
    return "both_wrong"


def bucket_table(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    errs = [r for r in recs if not is_right(r)]
    c = Counter(error_bucket(r["prediction"], r["answers"]) for r in errs)
    return {"n": len(recs), "n_errors": len(errs), "error_rate_pct": round(100 * len(errs) / max(len(recs), 1), 2),
            "counts": {b: c.get(b, 0) for b in BUCKETS}, "pct_of_errors": _pct(c, BUCKETS)}


def _check_aligned(runs: Dict[str, List[Dict[str, Any]]], n_expect: Optional[int] = None) -> List[str]:
    ids = None
    for k, rr in runs.items():
        q = [str(r["question_id"]) for r in rr]
        if ids is None:
            ids = q
        elif q != ids:
            raise ValueError(f"{k}: thứ tự question_id khác run đầu -> DỪNG, báo Claude")
    if n_expect is not None and len(ids) != n_expect:
        raise ValueError(f"cần {n_expect} câu, có {len(ids)}")
    return ids


def _pick(rng_seed: int, items: List[Dict[str, Any]], k: int = N_EXAMPLES) -> List[Dict[str, Any]]:
    items = sorted(items, key=lambda x: str(x.get("qid", x.get("question_id"))))
    return random.Random(rng_seed).sample(items, min(k, len(items)))


# =============================================================================
# ViTextVQA
# =============================================================================
def analyze_vitext(runs: Dict[str, List[Dict[str, Any]]], test: List[Dict[str, Any]], cache: Dict[str, Any],
                   tmap: Dict[str, str]) -> Dict[str, Any]:
    ids = _check_aligned(runs)
    by_id = {str(t["question_id"]): t for t in test}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} câu dự đoán không có trong file test, vd {missing[:3]}")
    imgs = cache["images"]
    rows = []
    for k, qid in enumerate(ids):
        t = by_id[qid]
        key = os.path.basename(str(t["image"]))
        if key not in imgs:
            raise KeyError(f"ảnh {key} không có trong OCR cache")
        txt = ocr_text_of(imgs[key])
        b2, b3 = runs["B2"][k], runs["B3full"][k]
        st = ocr_status(txt, b3["answers"])
        inside = pred_in_ocr(txt, b3["prediction"])
        rows.append({"qid": qid, "image": key, "question": b3["question"], "answers": b3["answers"],
                     "type": tmap[classify(b3["question"])], "ocr_status": st,
                     "b1": runs["B1"][k]["prediction"], "b2": b2["prediction"], "b3": b3["prediction"],
                     "b2_right": is_right(b2), "b3_right": is_right(b3), "b3_in_ocr": inside,
                     "b2_in_ocr": pred_in_ocr(txt, b2["prediction"]),
                     "b3_bucket": None if is_right(b3) else error_bucket(b3["prediction"], b3["answers"]),
                     "source": None if is_right(b3) else error_source(st, inside)})
    out: Dict[str, Any] = {"n": len(rows), "buckets": {k: bucket_table(v) for k, v in runs.items()}}
    st_all = Counter(r["ocr_status"] for r in rows)
    out["ocr_status_all_questions"] = {"counts": dict(st_all), "pct": _pct(st_all, OCR_STATUS)}
    errs = [r for r in rows if not r["b3_right"]]
    src = Counter(r["source"] for r in errs)
    out["b3_sources"] = {"n_errors": len(errs), "counts": {s: src.get(s, 0) for s in SOURCES}, "pct": _pct(src, SOURCES),
                         "ocr_upstream_detail": _pct(Counter(r["ocr_status"] for r in errs if r["source"] == "ocr_upstream"),
                                                     OCR_STATUS[1:])}
    out["b3_source_x_bucket"] = {s: {b: sum(1 for r in errs if r["source"] == s and r["b3_bucket"] == b) for b in BUCKETS}
                                 for s in SOURCES}
    types = sorted({r["type"] for r in rows})
    out["b3_by_type"] = {}
    for ty in types:
        rr = [r for r in rows if r["type"] == ty]
        ee = [r for r in rr if not r["b3_right"]]
        out["b3_by_type"][ty] = {"n": len(rr), "em_pct": round(100 * (len(rr) - len(ee)) / len(rr), 1),
                                 "has_answer_pct": round(100 * sum(r["ocr_status"] == "has_answer" for r in rr) / len(rr), 1),
                                 "n_errors": len(ee), "source_pct": _pct(Counter(r["source"] for r in ee), SOURCES)}
    tr = Counter(transition(r["b2_right"], r["b3_right"]) for r in rows)
    out["b2_to_b3"] = {"all": {t: tr.get(t, 0) for t in TRANSITIONS}, "by_ocr_status": {}}
    for st in OCR_STATUS:
        rr = [r for r in rows if r["ocr_status"] == st]
        c = Counter(transition(r["b2_right"], r["b3_right"]) for r in rr)
        out["b2_to_b3"]["by_ocr_status"][st] = {"n": len(rr), **{t: c.get(t, 0) for t in TRANSITIONS}}
    broken = [r for r in rows if r["b2_right"] and not r["b3_right"]]
    fixed = [r for r in rows if r["b3_right"] and not r["b2_right"]]
    out["b2_to_b3"]["broken_b3_answer_in_ocr_pct"] = round(100 * sum(r["b3_in_ocr"] for r in broken) / max(len(broken), 1), 1)
    out["b2_to_b3"]["fixed_gold_in_ocr_pct"] = round(100 * sum(r["ocr_status"] == "has_answer" for r in fixed) / max(len(fixed), 1), 1)
    out["examples"] = {
        "ocr_fixed": _pick(SEED + 1, [r for r in fixed if r["ocr_status"] == "has_answer"]),
        "ocr_distraction": _pick(SEED + 2, [r for r in broken if r["b3_in_ocr"]]),
        "ocr_missing": _pick(SEED + 3, [r for r in errs if r["ocr_status"] == "answer_missing"]),
        "ocr_wrong_diacritics": _pick(SEED + 4, [r for r in errs if r["ocr_status"] == "wrong_diacritics"]),
        "model_wrong_span": _pick(SEED + 5, [r for r in errs if r["source"] == "model_wrong_span"]),
        "tone": _pick(SEED + 6, [r for r in errs if r["b3_bucket"] == "tone"]),
    }
    return out


# =============================================================================
# ViVQA
# =============================================================================
def analyze_vivqa(runs: Dict[str, List[Dict[str, Any]]], test: List[Dict[str, Any]],
                  train_val: List[Dict[str, Any]]) -> Dict[str, Any]:
    _check_aligned(runs)
    c3 = runs["C3"]
    out: Dict[str, Any] = {"n": len(c3), "buckets": {k: bucket_table(v) for k, v in runs.items()}}
    out["c3_bucket_by_type"] = {}
    out["c3_top_confusions"] = {}
    for ty in VIVQA_TYPES:
        rr = [r for r in c3 if r.get("question_type") == ty]
        out["c3_bucket_by_type"][ty] = bucket_table(rr)
        pairs = Counter((norm(r["answers"][0]), norm(r["prediction"])) for r in rr if not is_right(r))
        out["c3_top_confusions"][ty] = [{"gold": g, "pred": p, "n": n} for (g, p), n in pairs.most_common(10)]
    s = stratify(train_val, test, runs["B1"], c3, label_a="B1", label_b="C3")
    out["seen_unseen_B1_vs_C3"] = {k: v for k, v in s["subsets"].items() if k in ("all", "image_seen", "image_unseen", "clean")}
    by_id = {str(t["question_id"]): t for t in test}
    errs = [{"qid": str(r["question_id"]), "image": by_id[str(r["question_id"])].get("image"), "question": r["question"],
             "answers": r["answers"], "type": r.get("question_type"), "b1": b1["prediction"], "c3": r["prediction"],
             "bucket": error_bucket(r["prediction"], r["answers"])}
            for r, b1 in zip(c3, runs["B1"]) if not is_right(r)]
    out["examples"] = {f"c3_{ty}": _pick(SEED + 10 + i, [e for e in errs if e["type"] == ty and e["bucket"] == "other"])
                       for i, ty in enumerate(VIVQA_TYPES)}
    return out


# =============================================================================
# Báo cáo
# =============================================================================
def report(s: Dict[str, Any]) -> str:
    vt, vv = s["vitext"], s["vivqa"]
    L = ["# Error analysis — mô hình chính thức (W12, chỉ mô tả)", "",
         f"qt-v1 rules sha256 `{s['qt_rules_sha256'][:12]}` · OCR: VietOCR, max_chars {OCR_MAX_CHARS}, min_conf {OCR_MIN_CONF} · "
         f"seed {SEED}", "", f"## ViTextVQA full test ({vt['n']} câu)", "",
         "### A1. Kiểu lỗi (% số câu sai)", "", "| Run | Tỉ lệ sai | " + " | ".join(BUCKETS) + " |",
         "|---|---|" + "---|" * len(BUCKETS)]
    for k, b in vt["buckets"].items():
        L.append(f"| {k} | {b['error_rate_pct']:.1f}% | " + " | ".join(f"{b['pct_of_errors'][x]:.1f}" for x in BUCKETS) + " |")
    o = vt["ocr_status_all_questions"]["pct"]
    L += ["", "### A2. Nguồn lỗi của B3-full", "",
          "Trạng thái OCR trên mọi câu: " + " · ".join(f"{k} {o[k]:.1f}%" for k in OCR_STATUS), "",
          f"Trên {vt['b3_sources']['n_errors']} câu B3-full sai: " +
          " · ".join(f"**{k}** {vt['b3_sources']['pct'][k]:.1f}%" for k in SOURCES) +
          " (ocr_upstream gồm: " + ", ".join(f"{k} {v:.1f}%" for k, v in vt["b3_sources"]["ocr_upstream_detail"].items()) + ")", "",
          "| Nguồn lỗi | " + " | ".join(BUCKETS) + " |", "|---|" + "---|" * len(BUCKETS)]
    for src, row in vt["b3_source_x_bucket"].items():
        L.append(f"| {src} | " + " | ".join(str(row[b]) for b in BUCKETS) + " |")
    L += ["", "| Loại (qt-v1) | n | EM B3-full | % câu có đáp án trong OCR | " + " | ".join(SOURCES) + " |",
          "|---|---|---|---|" + "---|" * len(SOURCES)]
    for ty, v in vt["b3_by_type"].items():
        L.append(f"| {ty} | {v['n']} | {v['em_pct']:.1f} | {v['has_answer_pct']:.1f} | " +
                 " | ".join(f"{v['source_pct'][x]:.1f}" for x in SOURCES) + " |")
    t = vt["b2_to_b3"]
    L += ["", "### A3. B2 → B3-full (ghép cặp, số câu)", "", "| Trạng thái OCR | n | " + " | ".join(TRANSITIONS) + " |",
          "|---|---|" + "---|" * len(TRANSITIONS),
          f"| tất cả | {vt['n']} | " + " | ".join(str(t['all'][x]) for x in TRANSITIONS) + " |"]
    for st, v in t["by_ocr_status"].items():
        L.append(f"| {st} | {v['n']} | " + " | ".join(str(v[x]) for x in TRANSITIONS) + " |")
    L += ["", f"Câu được sửa có đáp án trong OCR: {t['fixed_gold_in_ocr_pct']:.1f}% · câu bị làm hỏng mà câu trả lời B3-full "
              f"nằm trong OCR (OCR làm phân tâm): {t['broken_b3_answer_in_ocr_pct']:.1f}%", "",
          f"## ViVQA test ({vv['n']} câu)", "", "### V1. Kiểu lỗi", "",
          "| Run / loại | n | Tỉ lệ sai | " + " | ".join(BUCKETS) + " |", "|---|---|---|" + "---|" * len(BUCKETS)]
    for k, b in list(vv["buckets"].items()) + [(f"C3 · {ty}", vv["c3_bucket_by_type"][ty]) for ty in VIVQA_TYPES]:
        L.append(f"| {k} | {b['n']} | {b['error_rate_pct']:.1f}% | " + " | ".join(f"{b['pct_of_errors'][x]:.1f}" for x in BUCKETS) + " |")
    L += ["", "### V2. Cặp nhầm nhiều nhất của C3 (gold → dự đoán, số câu)", ""]
    for ty in VIVQA_TYPES:
        L.append(f"- **{ty}**: " + "; ".join(f"{c['gold']} → {c['pred']} ({c['n']})" for c in vv["c3_top_confusions"][ty][:8]))
    L += ["", "### V3. Ảnh đã thấy / chưa thấy lúc train (train + val)", "",
          "| Tập con | n | % test | EM B1 | EM C3 | ΔEM [CI95] |", "|---|---|---|---|---|---|"]
    for k, v in vv["seen_unseen_B1_vs_C3"].items():
        if v.get("n"):
            lo, hi = v["delta_EM_ci95"]
            L.append(f"| {k} | {v['n']} | {v['pct_of_test']} | {100 * v['EM_B1']:.2f} | {100 * v['EM_C3']:.2f} | "
                     f"{100 * v['delta_EM']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] |")
    return "\n".join(L) + "\n"


def run_all(exp_root: str, vitext_test: str, vitext_ocr: str, vivqa_dir: str) -> Dict[str, Any]:
    preds = {ds: {k: _load(os.path.join(exp_root, r, "predictions.json")) for k, r in rr.items()} for ds, rr in RUNS.items()}
    meta_b3 = _load(os.path.join(exp_root, RUNS["vitext"]["B3full"], "metrics.json"))["meta"]
    meta_c3 = _load(os.path.join(exp_root, RUNS["vivqa"]["C3"], "metrics.json"))["meta"]
    vivqa_test = os.path.join(vivqa_dir, "test.json")
    for path, meta in ((vitext_test, meta_b3), (vivqa_test, meta_c3)):
        if _sha256(path) != meta["data_file_sha256"]:
            raise ValueError(f"{path} khác file test của run gốc (sha256) -> DỪNG, báo Claude")
    o = meta_b3.get("ocr") or {}
    if (o.get("max_chars"), o.get("min_conf"), o.get("template")) != (OCR_MAX_CHARS, OCR_MIN_CONF, "v1"):
        raise ValueError(f"thiết lập OCR của B3-full khác hằng số trong script: {o}")
    t2000 = _load(os.path.join(exp_root, TEST2000_B1, "predictions.json"))
    tmap = merged_types(Counter(classify(r["question"]) for r in t2000))
    _check_aligned(preds["vitext"], 10028)
    _check_aligned(preds["vivqa"], 3001)
    tv = []
    for name in ("train", "val"):
        p = os.path.join(vivqa_dir, f"{name}.json")
        if os.path.exists(p):
            tv += _records(p)
    return {"qt_version": QT_VERSION, "qt_rules_sha256": rules_sha256(), "type_map": tmap, "runs": RUNS,
            "inputs_sha256": {"vitext_test": meta_b3["data_file_sha256"], "vivqa_test": meta_c3["data_file_sha256"],
                              "vitext_ocr": _sha256(vitext_ocr)},
            "vitext": analyze_vitext(preds["vitext"], _records(vitext_test), load_ocr_cache(vitext_ocr), tmap),
            "vivqa": analyze_vivqa(preds["vivqa"], _records(vivqa_test), tv)}


# =============================================================================
# Kiểm thử offline
# =============================================================================
def _selftest() -> None:
    # (a) trạng thái OCR + nguồn lỗi
    assert ocr_status(norm("Phở Sưa | 0909 123 456"), ["0909 123 456"]) == "has_answer"
    assert ocr_status(norm("Pho Sua"), ["phở sưa"]) == "wrong_diacritics"
    assert ocr_status(norm("Bánh mì"), ["phở"]) == "answer_missing"
    assert ocr_status("", ["phở"]) == "no_text"
    assert ocr_status(norm("bán phở"), ["an"]) == "answer_missing", "không khớp nửa từ"
    assert pred_in_ocr(norm("Phở Sưa | Bánh mì"), "bánh mì") and not pred_in_ocr(norm("Phở"), "")
    assert error_source("has_answer", True) == "model_wrong_span" and error_source("has_answer", False) == "model_outside_ocr"
    assert error_source("no_text", True) == "ocr_upstream"
    assert [transition(a, b) for a, b in ((0, 1), (1, 0), (1, 1), (0, 0))] == list(TRANSITIONS)
    print("ocr status + source OK")
    # (b) ViTextVQA trên dữ liệu giả
    cache = {"images": {"a.jpg": {"lines": [{"text": "Phở Sưa", "conf": 0.9}, {"text": "0909 123", "conf": 0.9}]},
                        "b.jpg": {"lines": []},
                        "c.jpg": {"lines": [{"text": "Pho Sua", "conf": 0.9}]}}}
    test = [{"question_id": "q1", "image": "a.jpg"}, {"question_id": "q2", "image": "a.jpg"},
            {"question_id": "q3", "image": "b.jpg"}, {"question_id": "q4", "image": "c.jpg"}]
    gold = {"q1": ["phở sưa"], "q2": ["0909 123"], "q3": ["chè"], "q4": ["phở sưa"]}
    qs = {"q1": "quán này tên là gì ?", "q2": "số điện thoại là gì ?", "q3": "quán bán gì ?", "q4": "tên quán là gì ?"}

    def mk(pred):
        return [{"question_id": q, "question": qs[q], "answers": gold[q], "prediction": pred[q], "question_type": "unknown"}
                for q in ("q1", "q2", "q3", "q4")]
    runs = {"B1": mk({"q1": "x", "q2": "x", "q3": "x", "q4": "x"}),
            "B2": mk({"q1": "phở sưa", "q2": "x", "q3": "x", "q4": "x"}),
            "B3full": mk({"q1": "0909 123", "q2": "0909 123", "q3": "bún", "q4": "pho sua"})}
    tmap = merged_types(Counter(classify(q) for q in qs.values()), min_n=1)
    v = analyze_vitext(runs, test, cache, tmap)
    assert v["b3_sources"]["counts"] == {"ocr_upstream": 2, "model_wrong_span": 1, "model_outside_ocr": 0}, v["b3_sources"]
    assert v["b2_to_b3"]["all"] == {"fixed": 1, "broken": 1, "both_right": 0, "both_wrong": 2}
    assert v["b2_to_b3"]["broken_b3_answer_in_ocr_pct"] == 100.0 and v["b2_to_b3"]["fixed_gold_in_ocr_pct"] == 100.0
    assert v["buckets"]["B3full"]["counts"]["tone"] == 1
    assert v["examples"]["ocr_distraction"][0]["qid"] == "q1" and v["examples"]["ocr_fixed"][0]["qid"] == "q2"
    try:
        analyze_vitext({**runs, "B2": runs["B2"][::-1]}, test, cache, tmap)
        raise AssertionError("không chặn lệch thứ tự")
    except ValueError:
        pass
    print("vitext OK")
    # (c) ViVQA trên dữ liệu giả
    tv = [{"question_id": "tr1", "question": "màu của xe là gì", "answers": ["đỏ"], "image": "i1.jpg", "question_type": "color"}]
    te = [{"question_id": "t1", "question": "màu của áo là gì", "answers": ["xanh"], "image": "i1.jpg", "question_type": "color"},
          {"question_id": "t2", "question": "có bao nhiêu con mèo", "answers": ["2"], "image": "i2.jpg", "question_type": "number"}]
    mkv = lambda p: [{**t, "prediction": x} for t, x in zip(te, p)]
    vv = analyze_vivqa({"B1": mkv(["đỏ", "3"]), "C1": mkv(["xanh", "3"]), "C3": mkv(["đỏ", "2"])}, te, tv)
    assert vv["c3_top_confusions"]["color"] == [{"gold": "xanh", "pred": "đỏ", "n": 1}]
    assert vv["seen_unseen_B1_vs_C3"]["image_seen"]["n"] == 1 and vv["seen_unseen_B1_vs_C3"]["image_unseen"]["n"] == 1
    assert "## ViVQA" in report({"qt_rules_sha256": "x" * 12, "vitext": v, "vivqa": vv})
    print("vivqa OK")
    print("error_analysis selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(description="Error analysis mô hình chính thức (W12, chỉ mô tả)")
    ap.add_argument("--exp_root", default=os.path.join(REPO_ROOT, "experiments"))
    ap.add_argument("--vitext_test")
    ap.add_argument("--vitext_ocr")
    ap.add_argument("--vivqa_dir")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")       # console Windows cp1252 không in được tiếng Việt
    except Exception:
        pass
    if a.selftest:
        _selftest()
        return 0
    if not (a.vitext_test and a.vitext_ocr and a.vivqa_dir and a.out):
        ap.error("cần --vitext_test --vitext_ocr --vivqa_dir --out (hoặc --selftest)")
    s = run_all(a.exp_root, a.vitext_test, a.vitext_ocr, a.vivqa_dir)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "error_analysis_summary.json"), "w", encoding="utf-8", newline="\n") as fp:
        json.dump(s, fp, ensure_ascii=False, indent=1)
    md = report(s)
    with open(os.path.join(a.out, "error_analysis_summary.md"), "w", encoding="utf-8", newline="\n") as fp:
        fp.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
