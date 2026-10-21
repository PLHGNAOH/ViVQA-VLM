"""
full_eval_table.py — Bảng đánh giá ĐẦY ĐỦ (W11, bước 2): mọi run chính × 2 dataset, chỉ đọc file đã có, KHÔNG cần GPU.

Vấn đề: số liệu nằm rải ở ~30 thư mục experiments/ (W07–W10), mỗi README tuần chỉ chép một phần. Viết Results
(R6) và slide từ các bản chép tay dễ lệch số. Script này là NGUỒN DUY NHẤT cho bảng kết quả chính:
    1. đọc metrics.json của từng run eval (EM, VQA-Acc, ANLS, token-F1, s/câu, GPU);
    2. TÍNH LẠI EM/VQA-Acc/ANLS/F1 từ predictions.json bằng src/eval/metrics.py và DỪNG nếu lệch metrics.json
       (bắt lỗi chép nhầm thư mục / file bị ghi đè);
    3. lần ngược adapter -> run train -> train_log.json để lấy chi phí (giờ train, VRAM đỉnh, tham số train, MB adapter);
    4. kiểm quy ước VQA-Acc: đếm số câu có > 1 đáp án gold. Bằng 0 ở cả 2 dataset -> VQA-Acc ≡ EM (ghi vào thesis);
    5. độ ổn định theo seed (B2-sub, B3: seed 42/43/44 trên test2000).

Đầu ra (thư mục --out, mặc định experiments/W11_full_eval_table/):
    full_eval_table.json  — toàn bộ số + kiểm tra (máy đọc)
    full_eval_table.csv   — 1 dòng / run (dán vào Excel / thesis)
    full_eval_table.md    — các bảng đã định dạng (đọc nhanh, chép vào R6)

Chạy (máy local hoặc Colab, chỉ CPU, ≈ 2–4 phút vì tính lại metric trên ~60.000 câu):
    python -m src.eval.full_eval_table
    python -m src.eval.full_eval_table --exp_root experiments --out experiments/W11_full_eval_table
Kiểm thử offline:
    python -m src.eval.full_eval_table --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import tempfile
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.eval.metrics import evaluate_predictions  # noqa: E402

METRICS = ("exact_match", "vqa_accuracy", "anls", "token_f1")
TOL = 1e-9                     # metrics.json phải khớp tính lại tới sai số làm tròn float

# --------------------------------------------------------------------------- #
# DANH SÁCH RUN CHÍNH (sửa ở ĐÂY khi có run mới — vd adapter chung W11)
#   id      : tên dùng trong thesis/slide (đánh số chính thức sau Review 2)
#   section : bảng nào trong file .md
#   eval    : thư mục run eval trong experiments/
#   train   : thư mục run train (None = không train: zero-shot / few-shot / model tham chiếu)
#   ref     : id của dòng mốc để tính ΔEM trong cùng bảng (None = không tính)
# Ghi chú tên thư mục: ablation A1 (module) nằm ở W10_A2_*, A2 (rank) ở W10_A3_* — xem bảng ánh xạ docs/W10/README.md.
# --------------------------------------------------------------------------- #
T = {  # thư mục train (viết tắt cho gọn)
    "B2_vt": "W07_B2_qlora_r16_qwen25vl_vitextvqa_ep1_bf16_L4_seed42",
    "B2_vv": "W07_B2_qlora_r16_qwen25vl_vivqa_ep1_bf16_L4_seed42",
    "C3_vv": "W09_B2_qlora_r16_qwen25vl_vivqa_ep2_bf16_L4_seed42",
    "B2sub": "W08_B2sub_qlora_r16_qwen25vl_vitextvqa_sub8k_ep1_bf16_L4_seed42",
    "B3": "W08_B3_qlora_ocr_vietocr_v1_r16_qwen25vl_vitextvqa_sub8k_ep1_bf16_L4_seed42",
    "A1": "W10_A2_qlora_ocr_vietocr_v1_alllinear_r4_qwen25vl_vitextvqa_sub8k_ep1_bf16_L4_seed42",
    "A2": "W10_A3_qlora_ocr_vietocr_v1_r64_qwen25vl_vitextvqa_sub8k_ep1_bf16_L4_seed42",
    "B3full": "W10_B3full_qlora_ocr_vietocr_v1_r16_qwen25vl_vitextvqa_full_ep1_bf16_L4_seed42",
}

SECTIONS = {
    "vitextvqa_testfull": "ViTextVQA — full test (10,028 q) · main result",
    "vitextvqa_test2000": "ViTextVQA — test2000 (2,000 q, seed 42) · ablations",
    "vivqa_test": "ViVQA — test (3,001 q)",
}

RUNS: List[Dict[str, Any]] = [
    # ---- ViTextVQA full test (bảng kết quả chính) ----
    dict(id="B1", section="vitextvqa_testfull", ref=None, train=None,
         method="Zero-shot Qwen2.5-VL-3B (4-bit)",
         eval="W07_B1_zeroshot_qwen25vl_vitextvqa_testfull_bf16_L4_seed42"),
    dict(id="B2-sub", section="vitextvqa_testfull", ref="B1", train=T["B2sub"],
         method="QLoRA, 23% train (8k), no OCR",
         eval="W09_B2sub_eval_qlora_r16_sub8k_ep1_vitextvqa_testfull_bf16_L4_seed42"),
    dict(id="B3", section="vitextvqa_testfull", ref="B1", train=T["B3"],
         method="OCR-aware QLoRA, 23% train (8k)",
         eval="W09_B3_eval_qlora_ocr_vietocr_v1_r16_sub8k_ep1_vitextvqa_testfull_bf16_L4_seed42"),
    dict(id="B2", section="vitextvqa_testfull", ref="B1", train=T["B2_vt"],
         method="QLoRA, full train, no OCR",
         eval="W07_B2_eval_qlora_r16_ep1_vitextvqa_testfull_bf16_L4_seed42"),
    dict(id="B3-full", section="vitextvqa_testfull", ref="B1", train=T["B3full"],
         method="OCR-aware QLoRA, full train (proposed)",
         eval="W10_B3full_eval_qlora_ocr_vietocr_v1_r16_full_ep1_vitextvqa_testfull_bf16_L4_seed42"),
    # ---- ViTextVQA test2000 (ablation) ----
    dict(id="B1", section="vitextvqa_test2000", ref=None, train=None,
         method="Zero-shot",
         eval="W07_B1_zeroshot_qwen25vl_vitextvqa_test2000_bf16_L4_seed42"),
    dict(id="FS5", section="vitextvqa_test2000", ref="B1", train=None,
         method="5-shot text-only format examples",
         eval="W07_FS5_fewshot_qwen25vl_vitextvqa_test2000_bf16_L4_seed42"),
    dict(id="B1+OCR", section="vitextvqa_test2000", ref="B1", train=None,
         method="Zero-shot + OCR line at inference",
         eval="W08_B1ocr_vietocr_v1_qwen25vl_vitextvqa_test2000_bf16_L4_seed42"),
    dict(id="R1b", section="vitextvqa_test2000", ref="B1", train=None,
         method="Reference: zero-shot Qwen2-VL-7B (B1 config)",
         eval="W07_R1b_vitextvqa_test2000_qwen2vl7b_b1cfg_L4_seed42"),
    dict(id="B2", section="vitextvqa_test2000", ref="B1", train=T["B2_vt"],
         method="QLoRA, full train, no OCR",
         eval="W07_B2_eval_qlora_r16_ep1_vitextvqa_test2000_bf16_L4_seed42"),
    dict(id="B2+OCRinf", section="vitextvqa_test2000", ref="B1", train=T["B2_vt"],
         method="B2 adapter + OCR line at inference only",
         eval="W08_B2ocr_vietocr_v1_qlora_r16_ep1_vitextvqa_test2000_bf16_L4_seed42"),
    dict(id="B2-sub", section="vitextvqa_test2000", ref="B1", train=T["B2sub"],
         method="QLoRA, 8k subset, no OCR",
         eval="W08_B2sub_eval_qlora_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed42"),
    dict(id="B2-sub+OCRinf", section="vitextvqa_test2000", ref="B1", train=T["B2sub"],
         method="B2-sub adapter + OCR line at inference only",
         eval="W09_B2subocr_eval_qlora_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed42"),
    dict(id="B3", section="vitextvqa_test2000", ref="B1", train=T["B3"],
         method="OCR-aware QLoRA, 8k subset (attn r16)",
         eval="W08_B3_eval_qlora_ocr_vietocr_v1_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed42"),
    dict(id="B3-noOCRinf", section="vitextvqa_test2000", ref="B1", train=T["B3"],
         method="B3 adapter, OCR removed at inference",
         eval="W09_B3noocr_eval_qlora_ocr_vietocr_v1_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed42"),
    dict(id="A1 (all-linear r4)", section="vitextvqa_test2000", ref="B1", train=T["A1"],
         method="Ablation A1: LoRA on all LLM linear layers, r4 (param-matched)",
         eval="W10_A2_eval_qlora_ocr_vietocr_v1_alllinear_r4_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed42"),
    dict(id="A2 (r64)", section="vitextvqa_test2000", ref="B1", train=T["A2"],
         method="Ablation A2: attention LoRA, r64",
         eval="W10_A3_eval_qlora_ocr_vietocr_v1_r64_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed42"),
    # ---- ViVQA test ----
    dict(id="B1", section="vivqa_test", ref=None, train=None,
         method="Zero-shot",
         eval="W07_B1_zeroshot_qwen25vl_vivqa_test_bf16_L4_seed42"),
    dict(id="FS5", section="vivqa_test", ref="B1", train=None,
         method="5-shot text-only format examples",
         eval="W07_FS5_fewshot_qwen25vl_vivqa_test_bf16_L4_seed42"),
    dict(id="C1", section="vivqa_test", ref="B1", train=T["B2_vv"],
         method="QLoRA, 1 epoch",
         eval="W07_B2_eval_qlora_r16_ep1_vivqa_test_bf16_L4_seed42"),
    dict(id="C3", section="vivqa_test", ref="B1", train=T["C3_vv"],
         method="QLoRA, 2 epochs (selected on val-clean)",
         eval="W09_C3_eval_qlora_r16_vivqa_test_bf16_L4_seed42"),
]

# Độ ổn định theo seed (test2000): id -> các thư mục eval seed 42/43/44
SEED_GROUPS = {
    "B2-sub": ["W08_B2sub_eval_qlora_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed42",
               "W09_B2sub_eval_qlora_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed43",
               "W09_B2sub_eval_qlora_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed44"],
    "B3": ["W08_B3_eval_qlora_ocr_vietocr_v1_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed42",
           "W09_B3_eval_qlora_ocr_vietocr_v1_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed43",
           "W09_B3_eval_qlora_ocr_vietocr_v1_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed44"],
}


# =============================================================================
# Đọc + kiểm
# =============================================================================
def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def recompute(records: List[Dict[str, Any]]) -> Dict[str, float]:
    return evaluate_predictions([r.get("prediction", "") for r in records],
                                [r.get("answers", []) for r in records], metric_names=list(METRICS))


def read_eval(exp_root: str, eval_dir: str) -> Dict[str, Any]:
    """Số của 1 run eval + kiểm metrics.json == tính lại từ predictions.json."""
    d = os.path.join(exp_root, eval_dir)
    mj = _load_json(os.path.join(d, "metrics.json"))
    preds = _load_json(os.path.join(d, "predictions.json"))
    stored, meta = mj["metrics"], mj.get("meta", {})
    again = recompute(preds)
    for m in METRICS:
        if m in stored and abs(stored[m] - again[m]) > TOL:
            raise ValueError(f"{eval_dir}: {m} trong metrics.json = {stored[m]} nhưng tính lại = {again[m]} -> DỪNG, báo Claude")
    if meta.get("num_samples") not in (None, len(preds)):
        raise ValueError(f"{eval_dir}: num_samples {meta.get('num_samples')} != {len(preds)} câu trong predictions.json")
    n_multi = sum(1 for r in preds if len(r.get("answers", [])) > 1)
    em_eq_vqa = all(abs(evaluate_predictions([r["prediction"]], [r["answers"]], ["exact_match", "vqa_accuracy"])
                        ["exact_match"] - evaluate_predictions([r["prediction"]], [r["answers"]],
                                                                ["exact_match", "vqa_accuracy"])["vqa_accuracy"]) < TOL
                    for r in preds) if n_multi == 0 else None
    gpus = (meta.get("hardware") or {}).get("gpus_seen") or []
    by_type = {}
    types = sorted({r.get("question_type") or "unknown" for r in preds})
    if types != ["unknown"]:          # chỉ dataset có nhãn loại câu hỏi sẵn (ViVQA); ViTextVQA = W11 bước 6
        for t in types:
            sub = [r for r in preds if (r.get("question_type") or "unknown") == t]
            v = recompute(sub)
            by_type[t] = {"n": len(sub), "exact_match": v["exact_match"], "anls": v["anls"]}
    return {"n": len(preds), **{m: again[m] for m in METRICS}, "by_type": by_type,
            "sec_per_q": meta.get("sec_per_sample"), "infer_h": (meta.get("inference_time_sec") or 0) / 3600 or None,
            "gpu": ",".join(gpus) or None, "prompting_mode": meta.get("prompting_mode"),
            "adapter_mb": (meta.get("adapter") or {}).get("weights_mb"),
            "adapter_path": (meta.get("adapter") or {}).get("path"),
            "n_multi_gold": n_multi, "vqa_acc_equals_em": em_eq_vqa,
            "qids": [r["question_id"] for r in preds]}


def _targets_label(t: Any) -> Optional[str]:
    """['q_proj','k_proj',..] -> 'q+k+v+o'; regex preset all_linear_llm (W10) -> 'all-linear (LLM)'."""
    if isinstance(t, list):
        return "+".join(x.replace("_proj", "") for x in t)
    if isinstance(t, str) and "mlp" in t and "self_attn" in t:
        return "all-linear (LLM)"
    return t


def read_train(exp_root: str, train_dir: Optional[str]) -> Dict[str, Any]:
    if not train_dir:
        return {}
    m = _load_json(os.path.join(exp_root, train_dir, "train_log.json"))["meta"]
    c, tr, lora = m.get("cost") or {}, m.get("training") or {}, m.get("lora") or {}
    return {"train_n": m.get("num_samples"), "epochs": tr.get("epochs"), "steps": c.get("steps"),
            "train_h": (c.get("train_runtime_sec") or 0) / 3600 or None, "sec_per_step": c.get("sec_per_step"),
            "train_vram_gb": c.get("peak_vram_gb"), "trainable_params": m.get("trainable_params"),
            "trainable_pct": m.get("trainable_pct"), "lora_r": lora.get("r"),
            "lora_targets": _targets_label(lora.get("target_modules")),
            "train_prompting": m.get("prompting_mode"), "train_gpu": (m.get("hardware") or {}).get("gpu")}


def check_adapter_link(row: Dict[str, Any]) -> None:
    """Adapter mà run eval nạp phải thuộc đúng run train khai trong RUNS (bắt nhầm dòng)."""
    if row["train_dir"] and row.get("adapter_path") and f"/{row['train_dir']}/" not in row["adapter_path"] + "/":
        raise ValueError(f"{row['eval_dir']}: adapter {row['adapter_path']} không thuộc run train {row['train_dir']}")
    if row["train_dir"] is None and row.get("adapter_path"):
        raise ValueError(f"{row['eval_dir']}: khai là không train nhưng có adapter {row['adapter_path']}")


def build(exp_root: str, runs: List[Dict[str, Any]] = RUNS,
          seed_groups: Dict[str, List[str]] = SEED_GROUPS) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for spec in runs:
        ev = read_eval(exp_root, spec["eval"])
        row = {"section": spec["section"], "id": spec["id"], "method": spec["method"], "eval_dir": spec["eval"],
               "train_dir": spec["train"], "ref": spec["ref"], **ev, **read_train(exp_root, spec["train"])}
        check_adapter_link(row)
        rows.append(row)
    # cùng bảng -> cùng tập câu hỏi (ghép cặp được)
    for sec in SECTIONS:
        q = [r["qids"] for r in rows if r["section"] == sec]
        if q and any(sorted(x) != sorted(q[0]) for x in q):
            raise ValueError(f"Bảng {sec}: các run không cùng tập câu hỏi")
    for r in rows:
        if r["ref"]:
            base = next(x for x in rows if x["section"] == r["section"] and x["id"] == r["ref"])
            r["d_em_vs_ref"] = 100 * (r["exact_match"] - base["exact_match"])
            r["d_anls_vs_ref"] = 100 * (r["anls"] - base["anls"])
        r.pop("qids")
    seeds = {}
    for gid, dirs in seed_groups.items():
        vals = [read_eval(exp_root, d) for d in dirs]
        seeds[gid] = {"runs": dirs, "em": [v["exact_match"] for v in vals], "anls": [v["anls"] for v in vals],
                      "em_mean": statistics.mean(v["exact_match"] for v in vals),
                      "em_std": statistics.stdev(v["exact_match"] for v in vals) if len(vals) > 1 else 0.0,
                      "anls_mean": statistics.mean(v["anls"] for v in vals),
                      "anls_std": statistics.stdev(v["anls"] for v in vals) if len(vals) > 1 else 0.0}
    by_ds = {}
    for r in rows:
        ds = r["section"].split("_")[0]
        by_ds.setdefault(ds, {"n_multi_gold": 0, "all_vqa_acc_equals_em": True})
        by_ds[ds]["n_multi_gold"] += r["n_multi_gold"]
        by_ds[ds]["all_vqa_acc_equals_em"] &= bool(r["vqa_acc_equals_em"])
    return {"what": "Bảng đánh giá đầy đủ W11 — tính lại từ predictions.json, khớp metrics.json (sai số ≤ 1e-9)",
            "rows": rows, "seed_stability": seeds, "vqa_acc_convention": by_ds}


# =============================================================================
# Xuất file
# =============================================================================
def _f(x: Any, nd: int = 2, pct: bool = False, sign: bool = False) -> str:
    if x is None or x == "":
        return "—"
    if isinstance(x, (int, float)):
        v = 100 * x if pct else x
        return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return str(x)


def to_markdown(res: Dict[str, Any]) -> str:
    out = ["# Full evaluation table (W11)", "",
           "Generated by `src/eval/full_eval_table.py` from `experiments/*/predictions.json` "
           "(recomputed; equals `metrics.json`). Metrics in %. Δ = vs. the zero-shot row (B1) of the same table.", ""]
    for sec, title in SECTIONS.items():
        rows = [r for r in res["rows"] if r["section"] == sec]
        if not rows:
            continue
        out += [f"## {title}", "",
                "| ID | Method | EM | VQA-Acc | ANLS | Token-F1 | ΔEM | ΔANLS |",
                "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            out.append(f"| {r['id']} | {r['method']} | {_f(r['exact_match'], pct=True)} | {_f(r['vqa_accuracy'], pct=True)} | "
                       f"{_f(r['anls'], pct=True)} | {_f(r['token_f1'], pct=True)} | {_f(r.get('d_em_vs_ref'), sign=True)} | "
                       f"{_f(r.get('d_anls_vs_ref'), sign=True)} |")
        out.append("")
    out += ["## Cost (training: from train_log.json; inference: from metrics.json)", "",
            "| Table | ID | Train q | Epochs | LoRA | Trainable params | Train h | s/step | Train VRAM GB | Adapter MB | Infer s/q | GPU |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        lora = f"{r.get('lora_targets')} r{r.get('lora_r')}" if r.get("lora_r") else "—"
        tp = f"{r['trainable_params']:,} ({r['trainable_pct']}%)" if r.get("trainable_params") else "—"
        out.append(f"| {r['section']} | {r['id']} | {_f(r.get('train_n'), 0)} | {_f(r.get('epochs'), 0)} | {lora} | {tp} | "
                   f"{_f(r.get('train_h'))} | {_f(r.get('sec_per_step'))} | {_f(r.get('train_vram_gb'))} | "
                   f"{_f(r.get('adapter_mb'))} | {_f(r.get('sec_per_q'), 3)} | {r.get('gpu') or '—'} |")
    typed = [r for r in res["rows"] if r.get("by_type")]
    if typed:
        types = sorted({t for r in typed for t in r["by_type"]})
        out += ["", "## Per question type (dataset labels; EM %, n in header)", "",
                "| Table | ID | " + " | ".join(f"{t} (n={typed[0]['by_type'][t]['n']})" for t in types) + " |",
                "|---|---|" + "---|" * len(types)]
        for r in typed:
            out.append(f"| {r['section']} | {r['id']} | " + " | ".join(_f(r["by_type"][t]["exact_match"], pct=True)
                                                                       for t in types) + " |")
    out += ["", "## Seed stability (ViTextVQA test2000, seeds 42/43/44)", "",
            "| ID | EM per seed | EM mean ± std | ANLS mean ± std |", "|---|---|---|---|"]
    for gid, s in res["seed_stability"].items():
        out.append(f"| {gid} | {' / '.join(_f(v, pct=True) for v in s['em'])} | {_f(s['em_mean'], pct=True)} ± "
                   f"{_f(s['em_std'], pct=True)} | {_f(s['anls_mean'], pct=True)} ± {_f(s['anls_std'], pct=True)} |")
    out += ["", "## VQA-Acc convention", ""]
    for ds, v in res["vqa_acc_convention"].items():
        out.append(f"- **{ds}**: questions with > 1 gold answer = {v['n_multi_gold']} → VQA-Acc identical to EM on every "
                   f"question: {'yes' if v['all_vqa_acc_equals_em'] else 'NO'}.")
    out += ["", "Both datasets provide a single gold answer per question, so the soft VQA accuracy "
            "min(#matches/3, 1) reduces to exact match; we report it for completeness and state this in the thesis.", ""]
    return "\n".join(out)


CSV_COLS = ["section", "id", "method", "n", "exact_match", "vqa_accuracy", "anls", "token_f1", "d_em_vs_ref",
            "d_anls_vs_ref", "train_n", "epochs", "lora_targets", "lora_r", "trainable_params", "trainable_pct",
            "train_h", "sec_per_step", "train_vram_gb", "adapter_mb", "sec_per_q", "gpu", "prompting_mode",
            "eval_dir", "train_dir"]


def write_all(res: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "full_eval_table.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "full_eval_table.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in res["rows"]:
            w.writerow({k: r.get(k) for k in CSV_COLS})
    with open(os.path.join(out_dir, "full_eval_table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(to_markdown(res))


# =============================================================================
# Selftest (dữ liệu giả, không cần experiments/)
# =============================================================================
def _selftest() -> None:
    with tempfile.TemporaryDirectory() as td:
        def mk_eval(name, preds, adapter=None):
            os.makedirs(os.path.join(td, name))
            recs = [{"question_id": f"q{i}", "question": "?", "prediction": p, "answers": [g]} for i, (p, g) in enumerate(preds)]
            json.dump(recs, open(os.path.join(td, name, "predictions.json"), "w", encoding="utf-8"), ensure_ascii=False)
            met = recompute(recs)
            meta = {"num_samples": len(recs), "sec_per_sample": 0.5, "hardware": {"gpus_seen": ["NVIDIA L4"]}}
            if adapter:
                meta["adapter"] = {"path": f"/d/adapters/{adapter}/epoch_1", "weights_mb": 29.53}
            json.dump({"metrics": met, "meta": meta}, open(os.path.join(td, name, "metrics.json"), "w"))
        def mk_train(name):
            os.makedirs(os.path.join(td, name))
            json.dump({"meta": {"num_samples": 8, "trainable_params": 7372800, "trainable_pct": 0.196,
                                "training": {"epochs": 1.0}, "lora": {"r": 16, "target_modules": ["q_proj", "v_proj"]},
                                "cost": {"train_runtime_sec": 7200, "steps": 1, "sec_per_step": 3.0, "peak_vram_gb": 6.6},
                                "hardware": {"gpu": "NVIDIA L4"}}}, open(os.path.join(td, name, "train_log.json"), "w"))
        gold = ["màu đỏ", "hai", "phòng bếp", "con mèo"]
        mk_eval("e_b1", list(zip(["đỏ", "ba", "bếp", "chó"], gold)))
        mk_eval("e_ft", list(zip(["Đỏ", "2", "phòng bếp", "chó"], gold)), adapter="t_ft")
        mk_train("t_ft")
        runs = [dict(id="B1", section="vivqa_test", ref=None, train=None, method="zs", eval="e_b1"),
                dict(id="C1", section="vivqa_test", ref="B1", train="t_ft", method="ft", eval="e_ft")]
        res = build(td, runs, {"C1": ["e_ft", "e_ft"]})
        r = {x["id"]: x for x in res["rows"]}
        assert abs(r["B1"]["exact_match"] - 0.25) < 1e-12 and abs(r["C1"]["exact_match"] - 0.75) < 1e-12, r
        assert abs(r["C1"]["d_em_vs_ref"] - 50.0) < 1e-9
        assert r["C1"]["train_h"] == 2.0 and r["C1"]["lora_targets"] == "q+v"
        assert res["vqa_acc_convention"]["vivqa"] == {"n_multi_gold": 0, "all_vqa_acc_equals_em": True}
        # metrics.json bị sửa -> phải dừng
        p = os.path.join(td, "e_ft", "metrics.json")
        mj = json.load(open(p)); mj["metrics"]["exact_match"] = 0.9; json.dump(mj, open(p, "w"))
        try:
            build(td, runs, {})
            raise AssertionError("không bắt được metrics.json lệch")
        except ValueError:
            pass
        # adapter thuộc run train khác -> phải dừng
        mj["metrics"]["exact_match"] = 0.75; json.dump(mj, open(p, "w"))
        runs[1]["train"] = "t_other"; mk_train("t_other")
        try:
            build(td, runs, {})
            raise AssertionError("không bắt được adapter sai run train")
        except ValueError:
            pass
        runs[1]["train"] = "t_ft"
        write_all(build(td, runs, {}), os.path.join(td, "out"))
        assert "| C1 | ft | 75.00 |" in open(os.path.join(td, "out", "full_eval_table.md"), encoding="utf-8").read()
    print("✅ full_eval_table selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_root", default=os.path.join(REPO_ROOT, "experiments"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "experiments", "W11_full_eval_table"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    try:                                   # console Windows (cp1252) không in được Δ / tiếng Việt
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if a.selftest:
        _selftest()
        return 0
    res = build(a.exp_root)
    write_all(res, a.out)
    print(to_markdown(res))
    print(f"\n✅ {len(res['rows'])} dòng, mọi metrics.json khớp tính lại -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
