"""
lora_ablation.py — Áp LUẬT KẾT LUẬN ghi trước (docs/W10/README.md) cho ablation cấu hình LoRA (H3, H4).

Vấn đề: B3 (QLoRA + OCR) dùng LoRA r = 16 chỉ trên 4 lớp attention (q, k, v, o) của LLM — cấu hình
chọn từ W04, chưa từng kiểm. Hội đồng sẽ hỏi: "Sao không gắn LoRA vào cả MLP như QLoRA paper khuyên?"
và "r = 16 đã đủ chưa?". Hai giả thuyết (docs/W02/Hypotheses.md):
  H3 — LoRA trên TẤT CẢ lớp tuyến tính của LLM tốt hơn attention-only ở CÙNG ngân sách tham số.
  H4 — Hiệu năng bão hoà ở r ≤ 16; r = 64 không cải thiện quá 1 điểm.

Thiết kế (mọi thứ khác y hệt B3 seed 42 W08: tập con 8.000 câu, OCR VietOCR · v1 · 600 · 0, lr 2e-4,
1 epoch, batch hiệu dụng 16, bf16, 4-bit NF4, L4, eval test2000 greedy):
  B3 (mốc) : attention q,k,v,o · r 16 · α 32   ·  7.372.800 tham số
  A2 (H3)  : q,k,v,o + gate,up,down (chỉ LLM) · r 4 · α 8  ·  7.483.392 tham số (+1,5% — cùng ngân sách)
  A3 (H4)  : attention q,k,v,o · r 64 · α 128  · 29.491.200 tham số (×4)
  α = 2r ở mọi run -> hệ số scale α/r = 2 không đổi.

Đầu vào: paired_B3_vs_A2.json và paired_B3_vs_A3.json do compare_runs.py tạo (A = B3, B = biến thể),
         train_log.json của 3 run (chi phí, số tham số, LoRA audit).
Đầu ra : bảng + kết luận H3/H4 theo luật + CI95 của ΔEM (bootstrap ghép cặp từ bảng chéo).

Chạy:
    python -m src.eval.lora_ablation --paired A2=<dir>/paired_B3_vs_A2.json A3=<dir>/paired_B3_vs_A3.json \
        --train_log B3=<run>/train_log.json A2=<run>/train_log.json A3=<run>/train_log.json \
        --out <dir>/lora_ablation_summary.json
Kiểm thử offline:
    python -m src.eval.lora_ablation --selftest
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

ALPHA = 0.05
H4_MARGIN_PTS = 1.0          # H4: "r = 64 không cải thiện > 1 điểm EM"
N_BOOT, BOOT_SEED = 10_000, 42

H3_VERDICTS = ("ỦNG HỘ H3: ALL-LINEAR TỐT HƠN ATTENTION-ONLY (cùng ngân sách)",
               "BÁC BỎ H3: ATTENTION-ONLY TỐT HƠN",
               "H3 KHÔNG ĐƯỢC ỦNG HỘ: KHÔNG KHÁC BIỆT CÓ Ý NGHĨA")
H4_VERDICTS = ("BÁC BỎ H4: r = 64 TỐT HƠN r = 16 QUÁ 1 ĐIỂM (chưa bão hoà)",
               "ỦNG HỘ H4: r = 64 KHÔNG KHÁC BIỆT CÓ Ý NGHĨA (bão hoà ở r = 16)",
               "ỦNG HỘ H4: r = 64 KÉM HƠN r = 16",
               "ỦNG HỘ H4: r = 64 tăng có ý nghĩa nhưng ≤ 1 điểm")


def em_diff_ci95(n: int, n_only_a: int, n_only_b: int, n_boot: int = N_BOOT,
                 seed: int = BOOT_SEED) -> Tuple[float, float]:
    """CI95 (percentile) của ΔEM = EM_B − EM_A, bootstrap ghép cặp.

    Chênh lệch từng câu chỉ nhận 3 giá trị: +1 (chỉ B đúng), −1 (chỉ A đúng), 0 (còn lại) -> lấy mẫu lại
    n câu tương đương rút multinomial trên 3 ô của bảng chéo. Tái lập được (numpy default_rng(seed)).
    """
    import numpy as np
    if n <= 0:
        raise ValueError("n phải > 0")
    p = np.array([n_only_b, n_only_a, n - n_only_a - n_only_b], dtype=float) / n
    draws = np.random.default_rng(seed).multinomial(n, p, size=n_boot)
    diffs = np.sort((draws[:, 0] - draws[:, 1]) / n)
    return float(diffs[int(round(0.025 * (n_boot - 1)))]), float(diffs[int(round(0.975 * (n_boot - 1)))])


def verdict_h3(p: Dict[str, Any], alpha: float = ALPHA) -> str:
    """p: paired B3 (A) -> A2 (B)."""
    if p["p_value"] < alpha and p["em_b"] > p["em_a"]:
        return H3_VERDICTS[0]
    if p["p_value"] < alpha and p["em_b"] < p["em_a"]:
        return H3_VERDICTS[1]
    return H3_VERDICTS[2]


def verdict_h4(p: Dict[str, Any], alpha: float = ALPHA, margin_pts: float = H4_MARGIN_PTS) -> str:
    """p: paired B3 (A, r 16) -> A3 (B, r 64)."""
    d_pts = 100 * (p["em_b"] - p["em_a"])
    if p["p_value"] >= alpha:
        return H4_VERDICTS[1]
    if d_pts < 0:
        return H4_VERDICTS[2]
    return H4_VERDICTS[0] if d_pts > margin_pts else H4_VERDICTS[3]


def h4_strength(ci_pts: Tuple[float, float], margin_pts: float = H4_MARGIN_PTS) -> str:
    """Độ mạnh của "ủng hộ H4": CI95 ΔEM nằm hẳn dưới +1 điểm -> mạnh; còn chạm +1 -> yếu (độ nhạy chưa đủ)."""
    return ("mạnh: cận trên CI95 ΔEM < +1 điểm" if ci_pts[1] < margin_pts else
            "yếu: CI95 ΔEM còn chứa giá trị > +1 điểm (test2000 chưa đủ nhạy để loại trừ)")


def cost_row(label: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    lora, c = meta["lora"], meta["cost"]
    tm = lora["target_modules"]
    audit = meta.get("lora_audit") or {}
    return {"run": label, "target_modules": tm if isinstance(tm, str) else ",".join(tm),
            "r": lora["r"], "alpha": lora["lora_alpha"], "trainable_params": meta["trainable_params"],
            "trainable_pct": meta["trainable_pct"], "lora_modules": audit.get("n_modules"),
            "lora_by_tower": audit.get("by_tower"), "train_h": round(c["train_runtime_sec"] / 3600, 2),
            "sec_per_step": c["sec_per_step"], "peak_vram_gb": c["peak_vram_gb"], "final_loss": c["final_loss"],
            "seed": meta["seed"], "resumed_from": meta.get("resumed_from")}


def check_same_recipe(ref: Dict[str, Any], var: Dict[str, Any], label: str) -> None:
    """Biến thể chỉ được khác B3 ở mục lora (r, alpha, target_modules). Khác chỗ khác -> dừng."""
    keys = ("data_file_sha256", "num_samples", "prompting_mode", "max_pixels", "torch_dtype", "seed", "model_id")
    diff = {k: (ref.get(k), var.get(k)) for k in keys if ref.get(k) != var.get(k)}
    for k in ("learning_rate", "epochs", "per_device_batch_size", "gradient_accumulation_steps", "warmup_ratio"):
        if ref["training"].get(k) != var["training"].get(k):
            diff[f"training.{k}"] = (ref["training"].get(k), var["training"].get(k))
    o_ref, o_var = ref.get("ocr") or {}, var.get("ocr") or {}
    for k in ("backend", "template", "max_chars", "min_conf"):
        if o_ref.get(k) != o_var.get(k):
            diff[f"ocr.{k}"] = (o_ref.get(k), o_var.get(k))
    for k in ("lora_dropout", "bias"):
        if ref["lora"].get(k) != var["lora"].get(k):
            diff[f"lora.{k}"] = (ref["lora"].get(k), var["lora"].get(k))
    if diff:
        raise ValueError(f"{label} khác B3 ngoài cấu hình LoRA -> không phải ablation sạch: {diff}")


def summarize(paired: Dict[str, Dict[str, Any]], metas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    for lab in ("A2", "A3"):
        if lab not in paired:
            raise ValueError(f"Thiếu paired {lab}")
        if lab in metas and "B3" in metas:
            check_same_recipe(metas["B3"], metas[lab], lab)
    if paired["A2"]["n"] != paired["A3"]["n"] or abs(paired["A2"]["em_a"] - paired["A3"]["em_a"]) > 1e-12:
        raise ValueError("Hai so sánh không cùng mốc B3 / cùng tập câu hỏi")
    rows, out = [], {}
    for lab, fn in (("A2", verdict_h3), ("A3", verdict_h4)):
        p = paired[lab]
        lo, hi = em_diff_ci95(p["n"], p["n_only_a"], p["n_only_b"])
        ci_pts = (100 * lo, 100 * hi)
        rows.append({"so_sánh": f"B3 -> {lab}", "n": p["n"], "EM_B3": p["em_a"], f"EM_{lab}": p["em_b"],
                     "ΔEM_điểm": round(100 * (p["em_b"] - p["em_a"]), 2),
                     "ΔEM_CI95_điểm": [round(ci_pts[0], 2), round(ci_pts[1], 2)],
                     "chỉ_B3_đúng": p["n_only_a"], f"chỉ_{lab}_đúng": p["n_only_b"], "p_McNemar": p["p_value"],
                     "ΔANLS": round(p["anls_diff"], 4), "ΔANLS_CI95": [round(x, 4) for x in p["anls_ci95"]]})
        out[lab] = {"verdict": fn(p), "em_diff_ci95_pts": ci_pts}
    out["A3"]["strength"] = h4_strength(out["A3"]["em_diff_ci95_pts"]) if out["A3"]["verdict"] != H4_VERDICTS[0] else None
    costs = [cost_row(lab, metas[lab]) for lab in ("B3", "A2", "A3") if lab in metas]
    return {"what": "W10 ablation cấu hình LoRA trên công thức B3 (H3 vị trí gắn, H4 rank), test2000, seed 42",
            "rule": "docs/W10/README.md mục Luật ablation LoRA", "table": rows, "costs": costs,
            "verdict_H3": out["A2"]["verdict"], "verdict_H4": out["A3"]["verdict"],
            "H4_strength": out["A3"]["strength"],
            "limits": ["1 seed huấn luyện (42); std EM của B3 qua 3 seed = 0,23 điểm (W09)",
                       "test2000: McNemar chỉ phát hiện được |ΔEM| khoảng ≥ 1,7 điểm; 'không khác biệt' = nhỏ hơn mức đo được"]}


def _selftest() -> None:
    base = {"n": 2000, "em_a": 0.4925, "anls_a": 0.70, "anls_diff": 0.0, "anls_ci95": [-0.01, 0.01]}
    # H3
    assert verdict_h3({**base, "em_b": 0.52, "p_value": 1e-4}) == H3_VERDICTS[0]
    assert verdict_h3({**base, "em_b": 0.47, "p_value": 1e-3}) == H3_VERDICTS[1]
    assert verdict_h3({**base, "em_b": 0.50, "p_value": 0.2}) == H3_VERDICTS[2]
    assert verdict_h3({**base, "em_b": 0.48, "p_value": 0.06}) == H3_VERDICTS[2]
    # H4
    assert verdict_h4({**base, "em_b": 0.5125, "p_value": 1e-3}) == H4_VERDICTS[0]    # +2 điểm
    assert verdict_h4({**base, "em_b": 0.4990, "p_value": 0.3}) == H4_VERDICTS[1]
    assert verdict_h4({**base, "em_b": 0.4700, "p_value": 1e-3}) == H4_VERDICTS[2]
    assert verdict_h4({**base, "em_b": 0.5005, "p_value": 0.04}) == H4_VERDICTS[3]    # +0,8 điểm
    # CI95 ΔEM: không có câu lật -> [0, 0]; đối xứng quanh ΔEM thật; tái lập
    assert em_diff_ci95(2000, 0, 0) == (0.0, 0.0)
    lo, hi = em_diff_ci95(2000, 81, 155)                     # W08 B2-sub -> B3: ΔEM thật +3,70 điểm
    assert lo < 0.037 < hi and 0.02 < lo and hi < 0.055, (lo, hi)
    assert em_diff_ci95(2000, 81, 155) == (lo, hi)
    assert "mạnh" in h4_strength((-1.5, 0.8)) and "yếu" in h4_strength((-0.9, 1.6))
    # Kiểm "chỉ khác LoRA"
    meta = {"data_file_sha256": "x", "num_samples": 8000, "prompting_mode": "ocr", "max_pixels": 401408,
            "torch_dtype": "bfloat16", "seed": 42, "model_id": "m",
            "training": {"learning_rate": 2e-4, "epochs": 1.0, "per_device_batch_size": 2,
                         "gradient_accumulation_steps": 8, "warmup_ratio": 0.03},
            "ocr": {"backend": "paddledet_vietocr", "template": "v1", "max_chars": 600, "min_conf": 0.0},
            "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "target_modules": ["q_proj"]}}
    check_same_recipe(meta, {**meta, "lora": {**meta["lora"], "r": 64, "lora_alpha": 128}}, "A3")
    for bad in [{**meta, "seed": 43}, {**meta, "training": {**meta["training"], "learning_rate": 1e-4}},
                {**meta, "ocr": {**meta["ocr"], "template": "v2"}}, {**meta, "lora": {**meta["lora"], "lora_dropout": 0.1}}]:
        try:
            check_same_recipe(meta, bad, "X")
            raise AssertionError(f"không chặn {bad}")
        except ValueError:
            pass
    # summarize: cùng mốc B3
    pa2 = {**base, "em_b": 0.50, "p_value": 0.2, "n_only_a": 100, "n_only_b": 115}
    pa3 = {**base, "em_b": 0.49, "p_value": 0.8, "n_only_a": 90, "n_only_b": 85}
    s = summarize({"A2": pa2, "A3": pa3}, {})
    assert s["verdict_H3"] == H3_VERDICTS[2] and s["verdict_H4"] == H4_VERDICTS[1] and s["H4_strength"]
    try:
        summarize({"A2": pa2, "A3": {**pa3, "em_a": 0.4555}}, {})
        raise AssertionError("không chặn khác mốc B3")
    except ValueError:
        pass
    print("lora ablation rules OK")
    print("SELFTEST OK")


def _parse_kv(items: List[str]) -> Dict[str, str]:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not v:
            raise ValueError(f"Cần dạng NHÃN=đường_dẫn, nhận {it!r}")
        out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Kết luận ablation LoRA (H3, H4) theo luật ghi trước.")
    ap.add_argument("--paired", nargs="+", help="A2=paired_B3_vs_A2.json A3=paired_B3_vs_A3.json")
    ap.add_argument("--train_log", nargs="*", default=[], help="B3=... A2=... A3=... (train_log.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    paired = {}
    for k, path in _parse_kv(a.paired).items():
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        d.pop("only_b_question_ids", None)
        paired[k] = d
    metas = {}
    for k, path in _parse_kv(a.train_log).items():
        with open(path, encoding="utf-8") as fh:
            metas[k] = json.load(fh)["meta"]
    s = summarize(paired, metas)
    for r in s["table"]:
        print(r)
    for c in s["costs"]:
        print(c)
    print(f"➜ H3 (A2 all-linear r4 vs B3 attention r16, cùng ngân sách): {s['verdict_H3']}")
    print(f"➜ H4 (A3 attention r64 vs B3 r16): {s['verdict_H4']}" + (f" — {s['H4_strength']}" if s["H4_strength"] else ""))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(s, fh, ensure_ascii=False, indent=2)
        print("->", a.out)


if __name__ == "__main__":
    main()
