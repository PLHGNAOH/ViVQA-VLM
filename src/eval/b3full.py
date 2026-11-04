"""
b3full.py — Áp LUẬT KẾT LUẬN ghi trước (docs/W10/README.md, "Luật B3-full") cho OCR-aware QLoRA trên TOÀN BỘ train.

Vấn đề: W09 cho thấy B3 (QLoRA + OCR, 8.000 câu = 23% train) thắng B2-sub (cùng 8.000 câu, không OCR) có ý nghĩa,
nhưng vẫn THẤP hơn B2 (QLoRA không OCR, 100% train) 1,16 điểm EM trên full test. Câu hỏi còn mở: khi ĐỦ dữ liệu,
train cùng OCR có còn giúp không? Run B3-full = công thức B3 y hệt, chỉ đổi tập con 8k -> toàn bộ train.

Đầu vào: predictions.json (full test 10.028 câu) của 4 run, cùng tập câu hỏi:
    B2sub (23%, không OCR) · B3 (23%, OCR) · B2 (100%, không OCR, W07) · B3full (100%, OCR, W10)
Đầu ra:
    1. So sánh chính B2 -> B3full (McNemar exact trên EM, ΔANLS + CI95 bootstrap) + kết luận theo luật.
    2. B3full so với ViTextBLIP-2 (Table 6, Nguyen et al., ESWA 2025): "cao hơn" chỉ khi cận dưới CI95 > số công bố.
    3. CHỈ BÁO CÁO: bảng 2×2 lượng dữ liệu × OCR, 4 hiệu ứng đơn và tương tác theo câu
       d_i = (B3full_i − B2_i) − (B3_i − B2sub_i)  (lợi ích của OCR thay đổi bao nhiêu khi tăng dữ liệu 23% -> 100%).

Chạy:
    python -m src.eval.b3full --b2sub <run> --b3 <run> --b2 <run> --b3full <run> --out <dir>/b3full_summary.json
Kiểm thử offline:
    python -m src.eval.b3full --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.eval.compare_runs import (BOOTSTRAP_SEED, N_BOOTSTRAP, load_predictions,  # noqa: E402
                                   paired_bootstrap_ci, per_sample_scores)
from src.eval.factorial_2x2 import paired_effect  # noqa: E402

ALPHA = 0.05
RUNS = ("B2sub", "B3", "B2", "B3full")
DESC = {"B2sub": "23% train · không OCR", "B3": "23% train · OCR",
        "B2": "100% train · không OCR (W07)", "B3full": "100% train · OCR (W10)"}
EFFECTS = {  # tên: (A, B, ý nghĩa)
    "ocr_at_23pct": ("B2sub", "B3", "OCR-aware, 23% dữ liệu (W09)"),
    "ocr_at_100pct": ("B2", "B3full", "OCR-aware, 100% dữ liệu — SO SÁNH CHÍNH"),
    "data_without_ocr": ("B2sub", "B2", "thêm dữ liệu 23% -> 100%, không OCR"),
    "data_with_ocr": ("B3", "B3full", "thêm dữ liệu 23% -> 100%, có OCR"),
}
VERDICTS = ("OCR-AWARE VƯỢT B2 KHI ĐỦ DỮ LIỆU", "OCR-AWARE KÉM B2 KHI ĐỦ DỮ LIỆU", "NGANG B2 KHI ĐỦ DỮ LIỆU")


def verdict_main(e: Dict[str, Any], alpha: float = ALPHA) -> str:
    """e = hiệu ứng B2 -> B3full (paired_effect)."""
    if e["p_value"] < alpha and e["em_diff"] > 0:
        return VERDICTS[0]
    if e["p_value"] < alpha and e["em_diff"] < 0:
        return VERDICTS[1]
    return VERDICTS[2]


def align(records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Sắp 4 run theo cùng thứ tự question_id. Khác tập câu hỏi -> lỗi (không ghép cặp được)."""
    ref = [r["question_id"] for r in records[RUNS[0]]]
    if len(set(ref)) != len(ref):
        raise ValueError("question_id trùng lặp")
    out = {}
    for k in RUNS:
        by_id = {r["question_id"]: r for r in records[k]}
        if set(by_id) != set(ref) or len(by_id) != len(records[k]):
            raise ValueError(f"{k} không cùng tập câu hỏi với {RUNS[0]}")
        out[k] = [by_id[q] for q in ref]
    return out


def analyze(records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    recs = align(records)
    sc = {k: per_sample_scores(recs[k]) for k in RUNS}
    n = len(sc[RUNS[0]])
    cells = {k: {"desc": DESC[k], "em": sum(s["exact_match"] for s in sc[k]) / n,
                 "anls": sum(s["anls"] for s in sc[k]) / n} for k in RUNS}
    effects = {name: {"from": a, "to": b, "meaning": m, **paired_effect(sc[a], sc[b])}
               for name, (a, b, m) in EFFECTS.items()}
    inter = {}
    for metric in ("exact_match", "anls"):
        d = [(s4[metric] - s3[metric]) - (s2[metric] - s1[metric])
             for s1, s2, s3, s4 in zip(sc["B2sub"], sc["B3"], sc["B2"], sc["B3full"])]
        lo, hi = paired_bootstrap_ci(d, N_BOOTSTRAP, BOOTSTRAP_SEED)
        inter[metric] = {"mean": sum(d) / n, "ci95": [lo, hi],
                         "definition": "(B3full − B2) − (B3 − B2sub), trung bình theo câu"}
    main = effects["ocr_at_100pct"]
    return {"n": n, "cells": cells, "effects": effects, "interaction_report_only": inter,
            "verdict_main": verdict_main(main),
            "rule": (f"B2 -> B3full: EM tăng & p<0,05 -> '{VERDICTS[0]}'; EM giảm & p<0,05 -> '{VERDICTS[1]}'; "
                     f"còn lại -> '{VERDICTS[2]}'. Bảng 2×2 + tương tác: CHỈ BÁO CÁO."),
            "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "method": "paired percentile"}}


def print_summary(s: Dict[str, Any]) -> None:
    c = s["cells"]
    print(f"n = {s['n']} (full test)")
    print(f"{'':16s}{'không OCR':>16s}{'có OCR':>16s}")
    for lab, (a, b) in (("23% train", ("B2sub", "B3")), ("100% train", ("B2", "B3full"))):
        print(f"{lab:16s}{'EM ' + format(100 * c[a]['em'], '.2f'):>16s}{'EM ' + format(100 * c[b]['em'], '.2f'):>16s}")
    for e in s["effects"].values():
        print(f"{e['meaning']:42s} {e['from']}->{e['to']} EM {100 * e['em_a']:.2f}->{100 * e['em_b']:.2f} "
              f"({100 * e['em_diff']:+.2f}đ) | {e['n_only_a']}/{e['n_only_b']} p={e['p_value']:.3g} | "
              f"ΔANLS {e['anls_diff']:+.4f} [{e['anls_ci95'][0]:+.4f}; {e['anls_ci95'][1]:+.4f}]")
    for m, it in s["interaction_report_only"].items():
        print(f"Tương tác dữ liệu × OCR ({m}, chỉ báo cáo): {it['mean']:+.4f} CI95 [{it['ci95'][0]:+.4f}; {it['ci95'][1]:+.4f}]")
    print(f"➜ KẾT LUẬN THEO LUẬT (B2 -> B3full): {s['verdict_main']}")


def _selftest() -> None:
    def recs(correct: List[int]) -> List[Dict[str, Any]]:
        return [{"question_id": f"q{i}", "prediction": "đúng" if c else "sai", "answers": ["đúng"]}
                for i, c in enumerate(correct)]
    n = 400
    base = [1 if i % 2 == 0 else 0 for i in range(n)]                    # 50%
    plus = [1 if (i % 2 == 0 or i % 10 == 1) else 0 for i in range(n)]     # +10 điểm (chỉ thêm câu đúng)
    minus = [1 if (i % 2 == 0 and i % 10 != 0) else 0 for i in range(n)]  # −10 điểm
    same = list(base)
    # thắng / thua / ngang có ý nghĩa
    s = analyze({"B2sub": recs(base), "B3": recs(plus), "B2": recs(base), "B3full": recs(plus)})
    assert s["verdict_main"] == VERDICTS[0], s["verdict_main"]
    assert abs(s["interaction_report_only"]["exact_match"]["mean"]) < 1e-12           # cùng lợi ích ở 2 mức dữ liệu
    assert analyze({"B2sub": recs(base), "B3": recs(plus), "B2": recs(base), "B3full": recs(minus)})["verdict_main"] == VERDICTS[1]
    s = analyze({"B2sub": recs(base), "B3": recs(plus), "B2": recs(base), "B3full": recs(same)})
    assert s["verdict_main"] == VERDICTS[2]
    assert abs(s["interaction_report_only"]["exact_match"]["mean"] + 0.10) < 1e-12    # lợi ích OCR mất đi khi đủ dữ liệu
    assert s["effects"]["ocr_at_23pct"]["n_only_b"] == 40 and s["effects"]["ocr_at_23pct"]["n_only_a"] == 0
    # thứ tự câu hỏi khác nhau vẫn ghép đúng; khác tập câu hỏi -> lỗi
    shuffled = list(reversed(recs(plus)))
    assert analyze({"B2sub": recs(base), "B3": shuffled, "B2": recs(base), "B3full": recs(plus)})["cells"]["B3"]["em"] == 0.6
    try:
        analyze({"B2sub": recs(base), "B3": recs(plus)[:-1], "B2": recs(base), "B3full": recs(plus)})
        raise AssertionError("không chặn khác tập câu hỏi")
    except ValueError:
        pass
    print("b3full rules OK")
    print("SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kết luận B3-full theo luật ghi trước (W10).")
    for k in RUNS:
        ap.add_argument(f"--{k.lower()}", help=f"thư mục run eval full test: {DESC[k]}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    records = {k: load_predictions(getattr(a, k.lower())) for k in RUNS}
    s = analyze(records)
    s["runs"] = {k: getattr(a, k.lower()).replace(os.sep, "/") for k in RUNS}
    print_summary(s)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(s, fh, ensure_ascii=False, indent=2)
        print("->", a.out)


if __name__ == "__main__":
    main()
