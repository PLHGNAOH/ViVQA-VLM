"""
shared_adapter.py — Áp LUẬT ghi trước (docs/W11/README.md, "Luật adapter dùng chung — SH") cho adapter dùng chung W11.

Vấn đề: prototype cần trả lời được cả câu hỏi thường (ViVQA) lẫn câu hỏi đọc chữ (ViTextVQA). Hai adapter chuyên biệt
buộc phải có bộ chọn adapter. Câu hỏi: MỘT adapter (SH) train trộn hai dataset (phương án S2: mọi câu dùng prompt `ocr`)
có KÉM adapter chuyên biệt cùng điều kiện không?

So sánh chính (ghép cặp theo question_id), biên không kém δ = 1,5 điểm EM:
    ViVQA test (3.001)      : C1 (QLoRA ViVQA, 1 epoch, W07)          -> SH
    ViTextVQA test2000      : B3 (QLoRA + OCR, tập con 8k, seed 42, W08) -> SH
    mỗi dataset: cận dưới CI95 ΔEM > −1,5 -> KHÔNG KÉM (thêm ΔEM > 0 và McNemar p < 0,05 -> KHÔNG KÉM — TỐT HƠN)
                 cận trên  CI95 ΔEM < −1,5 -> KÉM · còn lại -> CHƯA KẾT LUẬN
    chung: cả hai KHÔNG KÉM -> ADAPTER DÙNG CHUNG ĐẠT, ngược lại -> KHÔNG ĐẠT
Chỉ báo cáo: tách ViVQA theo việc prompt có dòng OCR hay không (nhóm không có dòng OCR: đầu vào SH == C1 -> chênh lệch
chỉ do trộn dữ liệu), so với mô hình chính thức C3 / B3-full (khác điều kiện), chẩn đoán "chép chữ OCR" trên ViVQA, chi phí.

Kiểm hợp lệ TRƯỚC khi kết luận (sai -> dừng): train_log SH (dữ liệu = đúng file C1 + đúng file B3, LoRA chỉ LLM 144 lớp,
7.372.800 tham số, siêu tham số = C1 = B3, phần OCR ViTextVQA trùng dấu vân tay B3) và 2 run eval (cùng file test,
cùng OCR với B3 trên test2000, adapter thuộc run train SH).

Chạy:
    python -m src.eval.shared_adapter --exp_root <drive>/experiments --sh_train <run> --sh_vivqa <run> --sh_vitext <run> \
        --vivqa_test <drive>/data/vivqa/test.json --vivqa_ocr_cache <drive>/data/vivqa/ocr/paddledet_vietocr.json \
        --out <dir>/shared_adapter_summary.json
Kiểm thử offline:
    python -m src.eval.shared_adapter --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.eval.compare_runs import load_predictions, per_sample_scores  # noqa: E402
from src.eval.factorial_2x2 import paired_effect  # noqa: E402
from src.eval.lora_ablation import em_diff_ci95  # noqa: E402
from src.eval.metrics import normalize_vi  # noqa: E402

ALPHA = 0.05
MARGIN_PTS = 1.5
VERDICTS = ("KHÔNG KÉM — TỐT HƠN", "KHÔNG KÉM", "KÉM", "CHƯA KẾT LUẬN")
OVERALL = ("ADAPTER DÙNG CHUNG ĐẠT", "ADAPTER DÙNG CHUNG KHÔNG ĐẠT")

# Run mốc (thư mục trong experiments/) — đúng luật ghi trước
REF = {
    "C1_train": "W07_B2_qlora_r16_qwen25vl_vivqa_ep1_bf16_L4_seed42",
    "B3_train": "W08_B3_qlora_ocr_vietocr_v1_r16_qwen25vl_vitextvqa_sub8k_ep1_bf16_L4_seed42",
    "C1": "W07_B2_eval_qlora_r16_ep1_vivqa_test_bf16_L4_seed42",
    "B3": "W08_B3_eval_qlora_ocr_vietocr_v1_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed42",
    "C3": "W09_C3_eval_qlora_r16_vivqa_test_bf16_L4_seed42",
    "B3full": "W10_B3full_eval_qlora_ocr_vietocr_v1_r16_full_ep1_vitextvqa_testfull_bf16_L4_seed42",
}
SAME_TRAINING = ("learning_rate", "epochs", "per_device_batch_size", "gradient_accumulation_steps",
                 "warmup_ratio", "weight_decay", "gradient_checkpointing")
SAME_MODEL = ("model_id", "max_pixels", "min_pixels", "torch_dtype", "quantization", "seed")
SAME_OCR = ("backend", "engine", "template", "max_chars", "min_conf")


# =============================================================================
# Luật
# =============================================================================
def verdict_dataset(em_diff: float, p_value: float, ci_pts: Tuple[float, float],
                    margin: float = MARGIN_PTS, alpha: float = ALPHA) -> str:
    lo, hi = ci_pts
    if lo > -margin:
        return VERDICTS[0] if (em_diff > 0 and p_value < alpha) else VERDICTS[1]
    if hi < -margin:
        return VERDICTS[2]
    return VERDICTS[3]


def verdict_overall(per_ds: Dict[str, str]) -> str:
    return OVERALL[0] if all(v.startswith("KHÔNG KÉM") for v in per_ds.values()) else OVERALL[1]


# =============================================================================
# Kiểm hợp lệ
# =============================================================================
def _load_meta(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["meta"]


def check_sh_train(sh: Dict[str, Any], c1: Dict[str, Any], b3: Dict[str, Any]) -> List[str]:
    """Trả danh sách lỗi (rỗng = hợp lệ). sh/c1/b3 = meta của train_log.json."""
    err: List[str] = []
    parts = {p["dataset"]: p for p in (sh.get("data_parts") or [])}
    if set(parts) != {"vivqa", "vitextvqa_official"}:
        err.append(f"data_parts phải gồm vivqa + vitextvqa_official, có {sorted(parts)}")
    else:
        for ds, ref, lab in (("vivqa", c1, "C1"), ("vitextvqa_official", b3, "B3")):
            p = parts[ds]
            if p["data_file_sha256"] != ref["data_file_sha256"]:
                err.append(f"{ds}: file train khác file của {lab}")
            if p["num_samples"] != ref["num_samples"] or p["num_dropped"] != 0:
                err.append(f"{ds}: {p['num_samples']} mẫu (loại {p['num_dropped']}) ≠ {lab} {ref['num_samples']}")
    if sh.get("num_samples") != c1["num_samples"] + b3["num_samples"] or sh.get("num_dropped") != 0:
        err.append(f"num_samples {sh.get('num_samples')} (loại {sh.get('num_dropped')}) ≠ "
                   f"{c1['num_samples']} + {b3['num_samples']}")
    for k in SAME_MODEL:
        for ref, lab in ((c1, "C1"), (b3, "B3")):
            if sh.get(k) != ref.get(k):
                err.append(f"{k}: SH {sh.get(k)} ≠ {lab} {ref.get(k)}")
    for k in SAME_TRAINING:
        for ref, lab in ((c1, "C1"), (b3, "B3")):
            if sh["training"].get(k) != ref["training"].get(k):
                err.append(f"training.{k}: SH {sh['training'].get(k)} ≠ {lab} {ref['training'].get(k)}")
    if sh.get("lora") != b3.get("lora") or sh.get("lora") != c1.get("lora"):
        err.append(f"lora khác mốc: {sh.get('lora')}")
    if (sh.get("lora_audit") or {}).get("by_tower") != {"llm": 144}:
        err.append(f"lora_audit.by_tower = {(sh.get('lora_audit') or {}).get('by_tower')} (cần {{'llm': 144}})")
    if sh.get("trainable_params") != 7_372_800:
        err.append(f"trainable_params = {sh.get('trainable_params')} (cần 7372800)")
    if sh.get("prompting_mode") != "ocr":
        err.append(f"prompting_mode = {sh.get('prompting_mode')} (S2 cần 'ocr')")
    o, ob = sh.get("ocr") or {}, b3.get("ocr") or {}
    for k in SAME_OCR:
        if o.get(k) != ob.get(k):
            err.append(f"ocr.{k}: SH {o.get(k)} ≠ B3 {ob.get(k)}")
    if (o.get("subset_sha256_by_dataset") or {}).get("vitextvqa_official") != ob.get("subset_sha256"):
        err.append("phần OCR ViTextVQA của SH KHÁC dấu vân tay OCR của B3 (không cùng đầu vào)")
    return err


def check_sh_eval(sh: Dict[str, Any], ref: Dict[str, Any], sh_train_run: str, lab: str,
                  ocr_ref: Optional[Dict[str, Any]] = None) -> List[str]:
    """sh/ref = meta của metrics.json. ocr_ref: meta eval của B3 (test2000) để bắt cùng đầu vào OCR."""
    err: List[str] = []
    for k in ("data_file_sha256", "max_new_tokens", "decoding", "model_id", "max_pixels", "min_pixels", "torch_dtype"):
        if sh.get(k) != ref.get(k):
            err.append(f"[{lab}] {k}: SH {sh.get(k)} ≠ mốc {ref.get(k)}")
    if sh.get("prompting_mode") != "ocr":
        err.append(f"[{lab}] prompting_mode = {sh.get('prompting_mode')} (cần 'ocr')")
    path = (sh.get("adapter") or {}).get("path") or ""
    if f"/{sh_train_run}/" not in path + "/":
        err.append(f"[{lab}] adapter {path} không thuộc run train {sh_train_run}")
    if ocr_ref is not None:
        o, r = sh.get("ocr") or {}, ocr_ref.get("ocr") or {}
        for k in SAME_OCR + ("subset_sha256",):
            if o.get(k) != r.get(k):
                err.append(f"[{lab}] ocr.{k}: SH {o.get(k)} ≠ B3 {r.get(k)}")
    return err


# =============================================================================
# Phân tích
# =============================================================================
def align(ref: List[Dict[str, Any]], other: List[Dict[str, Any]], lab: str,
          subset: bool = False) -> List[Dict[str, Any]]:
    """Sắp `other` theo thứ tự question_id của `ref`. subset=True: other được phép có NHIỀU câu hơn (vd full test)."""
    by_id = {r["question_id"]: r for r in other}
    if not subset and len(by_id) != len(ref):
        raise ValueError(f"{lab}: {len(by_id)} câu ≠ mốc {len(ref)} câu")
    miss = [r["question_id"] for r in ref if r["question_id"] not in by_id]
    if miss:
        raise ValueError(f"{lab}: thiếu {len(miss)} câu của mốc, vd {miss[:3]}")
    return [by_id[r["question_id"]] for r in ref]


def effect(a: List[Dict[str, float]], b: List[Dict[str, float]]) -> Dict[str, Any]:
    """Ghép cặp A -> B: EM, ΔEM (điểm %) + CI95, McNemar p, ΔANLS + CI95."""
    e = paired_effect(a, b)
    lo, hi = em_diff_ci95(len(a), e["n_only_a"], e["n_only_b"])
    return {**e, "n": len(a), "em_diff_pts": 100 * e["em_diff"], "em_diff_ci95_pts": [100 * lo, 100 * hi],
            "anls_a": sum(x["anls"] for x in a) / len(a), "anls_b": sum(x["anls"] for x in b) / len(b)}


def ocr_flags(test_file: str, cache_path: str, ocr_meta: Dict[str, Any]) -> Dict[str, List[str]]:
    """question_id -> các dòng OCR ĐÚNG như lúc eval SH (kiểm bằng dấu vân tay phần cache)."""
    from src.data.ocr import attach_ocr, load_ocr_cache
    from src.data.vivqa_dataset import load_vivqa
    samples = load_vivqa(test_file, image_dir="")
    st = attach_ocr(samples, load_ocr_cache(cache_path), ocr_meta["max_chars"], ocr_meta["min_conf"])
    if st["subset_sha256"] != ocr_meta["subset_sha256"]:
        raise ValueError("OCR dựng lại KHÁC OCR lúc eval SH (dấu vân tay lệch) -> sai cache/sai file test")
    return {s["question_id"]: s["ocr_texts"] for s in samples}


def copies_ocr(pred: str, ocr_texts: List[str]) -> bool:
    """Dự đoán (đã chuẩn hoá) xuất hiện NGUYÊN CỤM TỪ trong văn bản OCR."""
    p = normalize_vi(pred)
    if not p or not ocr_texts:
        return False
    o = " ".join(normalize_vi(t) for t in ocr_texts)
    return re.search(rf"(?<!\w){re.escape(p)}(?!\w)", o) is not None


def ocr_copy_rate(recs: List[Dict[str, Any]], scores: List[Dict[str, float]],
                  ocr: Dict[str, List[str]]) -> Dict[str, Any]:
    """Trong các câu prompt CÓ dòng OCR: tỉ lệ câu sai (EM = 0) mà dự đoán là cụm chữ chép từ OCR."""
    idx = [i for i, r in enumerate(recs) if ocr.get(r["question_id"])]
    wrong_copy = sum(1 for i in idx if scores[i]["exact_match"] != 1.0 and copies_ocr(recs[i]["prediction"],
                                                                                      ocr[recs[i]["question_id"]]))
    return {"n_with_ocr": len(idx), "n_wrong_copied_from_ocr": wrong_copy,
            "pct_wrong_copied_from_ocr": 100 * wrong_copy / len(idx) if idx else None}


def analyze(p: Dict[str, List[Dict[str, Any]]], ocr: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """p: C1, SHv (SH trên ViVQA), B3, SHt (SH trên test2000); tuỳ chọn C3, B3full (chỉ mô tả)."""
    out: Dict[str, Any] = {"rule": {"margin_pts": MARGIN_PTS, "alpha": ALPHA,
                                    "doc": "docs/W11/README.md — Luật adapter dùng chung — SH"}}
    sc = {"C1": per_sample_scores(p["C1"]), "SHv": per_sample_scores(align(p["C1"], p["SHv"], "SH ViVQA")),
          "B3": per_sample_scores(p["B3"]), "SHt": per_sample_scores(align(p["B3"], p["SHt"], "SH test2000"))}
    main = {"vivqa": {"ref": "C1", **effect(sc["C1"], sc["SHv"])},
            "vitextvqa_test2000": {"ref": "B3", **effect(sc["B3"], sc["SHt"])}}
    for v in main.values():
        v["verdict"] = verdict_dataset(v["em_diff"], v["p_value"], tuple(v["em_diff_ci95_pts"]))
    out["main"] = main
    out["verdict"] = verdict_overall({k: v["verdict"] for k, v in main.items()})
    desc = {}
    if p.get("C3"):
        desc["vivqa_C3_to_SH"] = effect(per_sample_scores(align(p["C1"], p["C3"], "C3")), sc["SHv"])
    if p.get("B3full"):
        desc["vitextvqa_test2000_B3full_to_SH"] = effect(
            per_sample_scores(align(p["B3"], p["B3full"], "B3-full", subset=True)), sc["SHt"])
    out["descriptive_vs_official"] = desc
    if ocr is not None:
        shv = align(p["C1"], p["SHv"], "SH ViVQA")
        strata = {}
        for name, want in (("no_ocr_line", False), ("with_ocr_line", True)):
            idx = [i for i, r in enumerate(p["C1"]) if bool(ocr.get(r["question_id"])) == want]
            if idx:
                strata[name] = effect([sc["C1"][i] for i in idx], [sc["SHv"][i] for i in idx])
        out["vivqa_by_ocr_line"] = strata
        out["vivqa_ocr_copy"] = {"C1": ocr_copy_rate(p["C1"], sc["C1"], ocr),
                                 "SH": ocr_copy_rate(shv, sc["SHv"], ocr)}
    return out


def cost(sh: Dict[str, Any], c1: Dict[str, Any], b3: Dict[str, Any]) -> Dict[str, Any]:
    h = lambda m: round(m["cost"]["train_runtime_sec"] / 3600, 2)  # noqa: E731
    return {"SH_train_h": h(sh), "C1_plus_B3_train_h": round(h(c1) + h(b3), 2), "SH_sec_per_step": sh["cost"]["sec_per_step"],
            "SH_peak_vram_gb": sh["cost"]["peak_vram_gb"], "SH_steps": sh["training"].get("total_steps"),
            "SH_resumed_from": sh.get("resumed_from"), "n_adapters_prototype_if_pass": 1, "n_adapters_otherwise": 2}


def report(s: Dict[str, Any]) -> str:
    L = []
    for ds, v in s["main"].items():
        L.append(f"{ds:20s} {v['ref']} {100 * v['em_a']:.2f} -> SH {100 * v['em_b']:.2f} | ΔEM {v['em_diff_pts']:+.2f} "
                 f"CI95 [{v['em_diff_ci95_pts'][0]:+.2f}; {v['em_diff_ci95_pts'][1]:+.2f}] | chỉ {v['ref']} đúng "
                 f"{v['n_only_a']} / chỉ SH đúng {v['n_only_b']} | p = {v['p_value']:.3g} | ΔANLS {100 * v['anls_diff']:+.2f} "
                 f"[{100 * v['anls_ci95'][0]:+.2f}; {100 * v['anls_ci95'][1]:+.2f}] -> {v['verdict']}")
    L.append(f"➜ KẾT LUẬN THEO LUẬT (δ = {MARGIN_PTS} điểm EM): {s['verdict']}")
    for k, v in (s.get("vivqa_by_ocr_line") or {}).items():
        L.append(f"   ViVQA · {k:14s} n={v['n']:5d} | C1 {100 * v['em_a']:.2f} -> SH {100 * v['em_b']:.2f} | ΔEM "
                 f"{v['em_diff_pts']:+.2f} [{v['em_diff_ci95_pts'][0]:+.2f}; {v['em_diff_ci95_pts'][1]:+.2f}] (chỉ báo cáo)")
    for k, v in (s.get("vivqa_ocr_copy") or {}).items():
        if v["pct_wrong_copied_from_ocr"] is not None:
            L.append(f"   chép chữ OCR (ViVQA, câu có dòng OCR) · {k}: {v['n_wrong_copied_from_ocr']}/{v['n_with_ocr']} "
                     f"= {v['pct_wrong_copied_from_ocr']:.2f}% câu sai")
    for k, v in (s.get("descriptive_vs_official") or {}).items():
        L.append(f"   (mô tả, khác điều kiện) {k}: {100 * v['em_a']:.2f} -> {100 * v['em_b']:.2f} | ΔEM {v['em_diff_pts']:+.2f} "
                 f"[{v['em_diff_ci95_pts'][0]:+.2f}; {v['em_diff_ci95_pts'][1]:+.2f}]")
    return "\n".join(L)


# =============================================================================
# Selftest
# =============================================================================
def _selftest() -> None:
    # luật
    assert verdict_dataset(0.004, 0.6, (-1.0, +1.8)) == VERDICTS[1]
    assert verdict_dataset(0.020, 0.001, (+0.9, +3.1)) == VERDICTS[0]
    assert verdict_dataset(0.004, 0.2, (-0.6, +1.4)) == VERDICTS[1]          # tăng nhưng không có ý nghĩa
    assert verdict_dataset(-0.030, 1e-5, (-4.0, -2.0)) == VERDICTS[2]
    assert verdict_dataset(-0.012, 0.04, (-2.2, -0.2)) == VERDICTS[3]        # CI chứa −1,5
    assert verdict_dataset(-0.015, 0.01, (-1.5, -0.4)) == VERDICTS[3]        # cận dưới đúng −1,5 -> chưa đủ
    assert verdict_overall({"a": VERDICTS[1], "b": VERDICTS[0]}) == OVERALL[0]
    assert verdict_overall({"a": VERDICTS[1], "b": VERDICTS[3]}) == OVERALL[1]
    # chép OCR: nguyên cụm từ, không khớp giữa chữ
    assert copies_ocr("Pepsi", ["Uống PEPSI mát lạnh"]) and not copies_ocr("pep", ["PEPSI"])
    assert not copies_ocr("đỏ", []) and copies_ocr("màu đỏ", ["Đỏ"])
    # kiểm train_log
    base = {"num_samples": 10799, "num_dropped": 0, "data_file_sha256": "sv", "model_id": "m", "max_pixels": 1,
            "min_pixels": 0, "torch_dtype": "bfloat16", "quantization": {"q": 4}, "seed": 42,
            "training": {k: 1 for k in SAME_TRAINING}, "lora": {"r": 16, "lora_alpha": 32},
            "lora_audit": {"by_tower": {"llm": 144}}, "trainable_params": 7372800, "prompting_mode": "zero_shot", "ocr": None}
    c1 = dict(base)
    b3 = {**base, "num_samples": 8000, "data_file_sha256": "st", "prompting_mode": "ocr",
          "ocr": {"backend": "v", "engine": {"e": 1}, "template": "v1", "max_chars": 600, "min_conf": 0.0, "subset_sha256": "F"}}
    sh = {**base, "num_samples": 18799, "prompting_mode": "ocr",
          "data_parts": [{"dataset": "vivqa", "data_file_sha256": "sv", "num_samples": 10799, "num_dropped": 0},
                         {"dataset": "vitextvqa_official", "data_file_sha256": "st", "num_samples": 8000, "num_dropped": 0}],
          "ocr": {**{k: b3["ocr"][k] for k in SAME_OCR}, "subset_sha256_by_dataset": {"vitextvqa_official": "F", "vivqa": "G"}}}
    assert check_sh_train(sh, c1, b3) == [], check_sh_train(sh, c1, b3)
    for bad in ({**sh, "ocr": {**sh["ocr"], "subset_sha256_by_dataset": {"vitextvqa_official": "X"}}},
                {**sh, "training": {**sh["training"], "learning_rate": 9}},
                {**sh, "trainable_params": 1},
                {**sh, "data_parts": sh["data_parts"][:1]},
                {**sh, "num_samples": 18000}):
        assert check_sh_train(bad, c1, b3), f"không bắt lỗi {bad}"
    # kiểm eval
    ev = {"data_file_sha256": "t", "max_new_tokens": 32, "decoding": "g", "model_id": "m", "max_pixels": 1, "min_pixels": 0,
          "torch_dtype": "bf16", "prompting_mode": "ocr", "adapter": {"path": "/d/adapters/RUN/epoch_1"},
          "ocr": {**b3["ocr"], "subset_sha256": "E"}}
    assert check_sh_eval(ev, ev, "RUN", "x", ocr_ref=ev) == []
    assert check_sh_eval({**ev, "adapter": {"path": "/d/adapters/KHAC/epoch_1"}}, ev, "RUN", "x")
    assert check_sh_eval({**ev, "ocr": {**ev["ocr"], "subset_sha256": "Z"}}, ev, "RUN", "x", ocr_ref=ev)
    # phân tích trên dữ liệu giả: SH = C1 trên ViVQA, SH hơn B3 rõ trên test2000
    mk = lambda qs, preds: [{"question_id": q, "prediction": p_, "answers": ["a"]} for q, p_ in zip(qs, preds)]  # noqa: E731
    qv = [f"v{i}" for i in range(400)]
    qt = [f"t{i}" for i in range(400)]
    C1 = mk(qv, ["a" if i % 2 else "b" for i in range(400)])
    SHv = list(reversed(mk(qv, ["a" if i % 2 else "b" for i in range(400)])))     # thứ tự khác -> align
    B3 = mk(qt, ["a" if i % 2 else "b" for i in range(400)])
    SHt = mk(qt, ["a" if (i % 2 or i % 8 == 0) else "b" for i in range(400)])     # +50 câu đúng
    ocr = {q: (["a"] if i < 100 else []) for i, q in enumerate(qv)}
    s = analyze({"C1": C1, "SHv": SHv, "B3": B3, "SHt": SHt, "C3": C1,
                 "B3full": B3 + mk(["extra"], ["a"])}, ocr)
    assert s["main"]["vivqa"]["verdict"] == VERDICTS[1] and s["main"]["vivqa"]["em_diff_ci95_pts"] == [0.0, 0.0]
    assert s["main"]["vitextvqa_test2000"]["verdict"] == VERDICTS[0] and s["verdict"] == OVERALL[0]
    assert s["vivqa_by_ocr_line"]["with_ocr_line"]["n"] == 100 and s["vivqa_by_ocr_line"]["no_ocr_line"]["n"] == 300
    assert s["vivqa_ocr_copy"]["C1"]["n_with_ocr"] == 100
    assert "KẾT LUẬN THEO LUẬT" in report(s)
    try:
        analyze({"C1": C1, "SHv": SHv[:-1], "B3": B3, "SHt": SHt})
        raise AssertionError("không chặn thiếu câu")
    except ValueError:
        pass
    print("shared adapter rules OK")
    print("SELFTEST OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_root", default=os.path.join(REPO_ROOT, "experiments"))
    ap.add_argument("--sh_train")
    ap.add_argument("--sh_vivqa")
    ap.add_argument("--sh_vitext")
    ap.add_argument("--vivqa_test", help="data/vivqa/test.json (để dựng lại dòng OCR, tách nhóm)")
    ap.add_argument("--vivqa_ocr_cache", help="data/vivqa/ocr/paddledet_vietocr.json")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if a.selftest:
        _selftest()
        return 0
    if not all([a.sh_train, a.sh_vivqa, a.sh_vitext, a.out]):
        ap.error("cần --sh_train --sh_vivqa --sh_vitext --out")
    E = lambda run: os.path.join(a.exp_root, run)  # noqa: E731
    sh_t = _load_meta(os.path.join(E(a.sh_train), "train_log.json"))
    c1_t = _load_meta(os.path.join(E(REF["C1_train"]), "train_log.json"))
    b3_t = _load_meta(os.path.join(E(REF["B3_train"]), "train_log.json"))
    m = {k: _load_meta(os.path.join(E(r), "metrics.json")) for k, r in
         (("SHv", a.sh_vivqa), ("SHt", a.sh_vitext), ("C1", REF["C1"]), ("B3", REF["B3"]))}
    errs = check_sh_train(sh_t, c1_t, b3_t) + check_sh_eval(m["SHv"], m["C1"], a.sh_train, "ViVQA") + \
        check_sh_eval(m["SHt"], m["B3"], a.sh_train, "test2000", ocr_ref=m["B3"])
    if errs:
        print("❌ RUN SH KHÔNG HỢP LỆ theo luật W11 -> không kết luận, báo Claude:\n  - " + "\n  - ".join(errs))
        return 1
    print("✔ kiểm hợp lệ: train_log SH + 2 run eval khớp luật W11")
    preds = {"C1": load_predictions(E(REF["C1"])), "SHv": load_predictions(E(a.sh_vivqa)),
             "B3": load_predictions(E(REF["B3"])), "SHt": load_predictions(E(a.sh_vitext))}
    for k in ("C3", "B3full"):
        if os.path.isfile(os.path.join(E(REF[k]), "predictions.json")):
            preds[k] = load_predictions(E(REF[k]))
    ocr = ocr_flags(a.vivqa_test, a.vivqa_ocr_cache, m["SHv"]["ocr"]) if (a.vivqa_test and a.vivqa_ocr_cache) else None
    s = analyze(preds, ocr)
    s.update({"runs": {"SH_train": a.sh_train, "SH_vivqa": a.sh_vivqa, "SH_vitext": a.sh_vitext, **REF},
              "cost": cost(sh_t, c1_t, b3_t), "validity_checks": "passed"})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=2)
    print(report(s))
    print(f"   chi phí: SH {s['cost']['SH_train_h']} h train (C1 + B3 = {s['cost']['C1_plus_B3_train_h']} h) | "
          f"{s['cost']['SH_sec_per_step']} s/step | VRAM {s['cost']['SH_peak_vram_gb']} GB")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
