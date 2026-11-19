"""
factorial_2x2.py — Thiết kế 2×2 "OCR lúc train × OCR lúc test" trên cùng một tập câu hỏi.

Vấn đề: W08–W09 có 3 ô: B2-sub (train không OCR, test không OCR), B3 bỏ OCR (train có, test không),
B3 (train có, test có). Thiếu ô "train không OCR, test có OCR" nên chưa nói được lợi ích của B3 đến từ
việc TRAIN cùng OCR hay chỉ cần CÓ OCR lúc test. Script này nhận đủ 4 ô và trả lời bằng kiểm định ghép cặp.

Ký hiệu ô (train, test):  c00 = (không, không)   c01 = (không, có)
                          c10 = (có,    không)   c11 = (có,    có)

Đầu ra:
  - EM / ANLS của 4 ô;
  - 4 "hiệu ứng đơn" (McNemar exact trên EM + ΔANLS CI95 bootstrap ghép cặp, như compare_runs.py);
  - tương tác (difference-in-differences) theo từng câu: d_i = (y11 - y10) - (y01 - y00),
    trung bình + CI95 bootstrap ghép cặp (10.000 lần, seed 42), cho cả EM và ANLS;
  - kết luận theo luật ghi trước trong docs/W09/README.md:
      K = so sánh c01 -> c11 (train cùng OCR có thêm lợi ích khi lúc test ĐÃ có OCR không?)
      * K: EM tăng & p < 0,05 & cận dưới CI95 tương tác EM > 0 -> "TƯƠNG TÁC: CẦN OCR Ở CẢ TRAIN LẪN TEST"
      * K: EM tăng & p < 0,05 & CI95 tương tác EM chứa 0       -> "CỘNG GỘP: TRAIN CÙNG OCR CÓ LỢI RIÊNG"
      * K: p >= 0,05                                           -> "OCR LÚC TEST LÀ ĐỦ"
      * K: EM giảm & p < 0,05                                  -> "TRAIN CÙNG OCR KÉM HƠN"

Chạy:
    python -m src.eval.factorial_2x2 --c00 <run B2-sub> --c01 <run B2-sub+OCR> \
        --c10 <run B3 bỏ OCR> --c11 <run B3> --out <dir>/factorial_summary.json

Kiểm thử offline:
    python -m src.eval.factorial_2x2 --selftest
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
                                   mcnemar_exact_p, paired_bootstrap_ci, per_sample_scores)

CELLS = ("c00", "c01", "c10", "c11")
CELL_DESC = {"c00": "train không OCR · test không OCR", "c01": "train không OCR · test có OCR",
             "c10": "train có OCR · test không OCR", "c11": "train có OCR · test có OCR"}
EFFECTS = {  # tên: (ô A, ô B, ý nghĩa)
    "test_ocr_given_train_no": ("c00", "c01", "OCR lúc test, model train KHÔNG OCR"),
    "test_ocr_given_train_yes": ("c10", "c11", "OCR lúc test, model train CÓ OCR"),
    "train_ocr_given_test_no": ("c00", "c10", "train cùng OCR, test KHÔNG OCR"),
    "train_ocr_given_test_yes": ("c01", "c11", "train cùng OCR, test CÓ OCR (so sánh K)"),
}
VERDICTS = ("TƯƠNG TÁC: CẦN OCR Ở CẢ TRAIN LẪN TEST", "CỘNG GỘP: TRAIN CÙNG OCR CÓ LỢI RIÊNG",
            "OCR LÚC TEST LÀ ĐỦ", "TRAIN CÙNG OCR KÉM HƠN")
ALPHA = 0.05


def paired_effect(sa: List[Dict[str, float]], sb: List[Dict[str, float]]) -> Dict[str, Any]:
    n = len(sa)
    only_a = sum(1 for a, b in zip(sa, sb) if a["exact_match"] == 1.0 and b["exact_match"] != 1.0)
    only_b = sum(1 for a, b in zip(sa, sb) if b["exact_match"] == 1.0 and a["exact_match"] != 1.0)
    em_a = sum(a["exact_match"] for a in sa) / n
    em_b = sum(b["exact_match"] for b in sb) / n
    diffs = [b["anls"] - a["anls"] for a, b in zip(sa, sb)]
    return {"em_a": em_a, "em_b": em_b, "em_diff": em_b - em_a, "n_only_a": only_a, "n_only_b": only_b,
            "p_value": mcnemar_exact_p(only_a, only_b), "anls_diff": sum(diffs) / n,
            "anls_ci95": list(paired_bootstrap_ci(diffs, N_BOOTSTRAP, BOOTSTRAP_SEED))}


def interaction(sc: Dict[str, List[Dict[str, float]]], metric: str) -> Dict[str, Any]:
    d = [(s11[metric] - s10[metric]) - (s01[metric] - s00[metric])
         for s00, s01, s10, s11 in zip(sc["c00"], sc["c01"], sc["c10"], sc["c11"])]
    lo, hi = paired_bootstrap_ci(d, N_BOOTSTRAP, BOOTSTRAP_SEED)
    return {"metric": metric, "mean": sum(d) / len(d), "ci95": [lo, hi],
            "definition": "(c11 - c10) - (c01 - c00), trung bình theo câu"}


def decide(k: Dict[str, Any], inter_em: Dict[str, Any], alpha: float = ALPHA) -> str:
    if k["p_value"] >= alpha:
        return VERDICTS[2]
    if k["em_diff"] < 0:
        return VERDICTS[3]
    return VERDICTS[0] if inter_em["ci95"][0] > 0 else VERDICTS[1]


def analyze(records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    ids = {c: [r["question_id"] for r in records[c]] for c in CELLS}
    if any(ids[c] != ids["c00"] for c in CELLS):
        raise ValueError("4 ô phải cùng question_id, cùng thứ tự — không ghép cặp được")
    if not ids["c00"]:
        raise ValueError("Không có câu nào")
    sc = {c: per_sample_scores(records[c]) for c in CELLS}
    n = len(ids["c00"])
    cells = {c: {"desc": CELL_DESC[c], "em": sum(s["exact_match"] for s in sc[c]) / n,
                 "anls": sum(s["anls"] for s in sc[c]) / n} for c in CELLS}
    effects = {name: {"from": a, "to": b, "meaning": m, **paired_effect(sc[a], sc[b])}
               for name, (a, b, m) in EFFECTS.items()}
    inter = {"em": interaction(sc, "exact_match"), "anls": interaction(sc, "anls")}
    return {"n": n, "cells": cells, "effects": effects, "interaction": inter,
            "rule": ("K = train_ocr_given_test_yes (c01 -> c11). EM tăng & p<0,05 & cận dưới CI95 tương tác EM > 0 -> "
                     f"'{VERDICTS[0]}'; EM tăng & p<0,05 & CI95 chứa 0 -> '{VERDICTS[1]}'; p>=0,05 -> '{VERDICTS[2]}'; "
                     f"EM giảm & p<0,05 -> '{VERDICTS[3]}'"),
            "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "method": "paired percentile"},
            "verdict": decide(effects["train_ocr_given_test_yes"], inter["em"])}


def print_summary(s: Dict[str, Any]) -> None:
    c = s["cells"]
    print(f"n = {s['n']}")
    print(f"{'':22s}{'test không OCR':>20s}{'test có OCR':>20s}")
    for t, (a, b) in (("train không OCR", ("c00", "c01")), ("train có OCR", ("c10", "c11"))):
        print(f"{t:22s}{'EM ' + format(c[a]['em'], '.4f'):>20s}{'EM ' + format(c[b]['em'], '.4f'):>20s}")
    for name, e in s["effects"].items():
        print(f"{e['meaning']:45s} {e['from']}->{e['to']} EM {e['em_a']:.4f}->{e['em_b']:.4f} "
              f"({100 * e['em_diff']:+.2f}đ) | {e['n_only_a']}/{e['n_only_b']} p={e['p_value']:.3g} | "
              f"ΔANLS {e['anls_diff']:+.4f} [{e['anls_ci95'][0]:+.4f}; {e['anls_ci95'][1]:+.4f}]")
    for m, it in s["interaction"].items():
        print(f"Tương tác {m.upper():4s}: {it['mean']:+.4f} CI95 [{it['ci95'][0]:+.4f}; {it['ci95'][1]:+.4f}]")
    print(f"➜ KẾT LUẬN 2×2 THEO LUẬT: {s['verdict']}")


def _selftest() -> None:
    def recs(correct: List[int]) -> List[Dict[str, Any]]:
        return [{"question_id": f"q{i}", "prediction": "đúng" if ok else "sai", "answers": ["đúng"]}
                for i, ok in enumerate(correct)]
    n = 400
    base = [1 if i < 180 else 0 for i in range(n)]                       # c00: 180 đúng
    plus = [1 if i < 260 else 0 for i in range(n)]                       # 260 đúng
    # (1) chỉ ô c11 tăng -> tương tác + K có ý nghĩa
    s = analyze({"c00": recs(base), "c01": recs(base), "c10": recs(base), "c11": recs(plus)})
    assert s["verdict"] == VERDICTS[0], s["verdict"]
    assert abs(s["interaction"]["em"]["mean"] - 80 / n) < 1e-12 and s["interaction"]["em"]["ci95"][0] > 0
    # (2) OCR lúc test giúp như nhau cho cả hai model -> K không ý nghĩa -> "OCR lúc test là đủ"
    s = analyze({"c00": recs(base), "c01": recs(plus), "c10": recs(base), "c11": recs(plus)})
    assert s["verdict"] == VERDICTS[2] and s["effects"]["train_ocr_given_test_yes"]["p_value"] == 1.0
    # (3) hai yếu tố cộng gộp: train OCR +40 ở cả hai cột, test OCR +40 ở cả hai hàng -> tương tác 0
    mid = [1 if i < 220 else 0 for i in range(n)]
    s = analyze({"c00": recs(base), "c01": recs(mid), "c10": recs(mid), "c11": recs(plus)})
    assert s["verdict"] == VERDICTS[1] and abs(s["interaction"]["em"]["mean"]) < 1e-12
    # (4) train cùng OCR làm kém khi có OCR
    s = analyze({"c00": recs(base), "c01": recs(plus), "c10": recs(base), "c11": recs(base)})
    assert s["verdict"] == VERDICTS[3]
    # (5) lệch question_id -> lỗi
    bad = recs(base)[::-1]
    try:
        analyze({"c00": recs(base), "c01": bad, "c10": recs(base), "c11": recs(base)})
        raise AssertionError("không chặn lệch question_id")
    except ValueError:
        pass
    print("SELFTEST OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for c in CELLS:
        ap.add_argument(f"--{c}", help=f"Thư mục run ô {c} ({CELL_DESC[c]}), chứa predictions.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return 0
    if not all(getattr(a, c) for c in CELLS):
        ap.error("cần đủ --c00 --c01 --c10 --c11")
    s = analyze({c: load_predictions(getattr(a, c)) for c in CELLS})
    s["runs"] = {c: getattr(a, c).replace(os.sep, "/") for c in CELLS}
    print_summary(s)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(s, fh, ensure_ascii=False, indent=2)
        print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
