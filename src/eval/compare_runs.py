"""
compare_runs.py — So sánh BẮT CẶP hai run trên cùng một tập mẫu.

- Metric từng mẫu (exact_match, anls) tính bằng src.eval.metrics.evaluate_predictions
  trên list 1 phần tử (dùng lại logic chuẩn hoá sẵn có).
- Bảng chéo 2x2 theo exact_match + kiểm định McNemar chính xác (nhị thức, p=0.5).
- Chênh lệch ANLS trung bình (B - A) kèm CI95 bằng bootstrap bắt cặp.

Chạy:
    python src/eval/compare_runs.py \
        --run_a experiments/<RUN_A> --run_b experiments/<RUN_B> \
        --label_a A --label_b B \
        --out experiments/<DIR>/paired_A_vs_B.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.eval.metrics import evaluate_predictions  # noqa: E402

ANLS_THRESHOLD = 0.5
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42


def load_predictions(run_dir: str) -> List[Dict[str, Any]]:
    with open(os.path.join(run_dir, "predictions.json"), encoding="utf-8") as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        raise ValueError(f"Kỳ vọng list trong {run_dir}/predictions.json, nhận {type(records).__name__}")
    return records


def per_sample_scores(records: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """[{exact_match, anls}] cho từng mẫu, theo đúng thứ tự records."""
    return [
        evaluate_predictions(
            predictions=[r.get("prediction", "")],
            references=[r.get("answers", [])],
            metric_names=["exact_match", "anls"],
            anls_threshold=ANLS_THRESHOLD,
        )
        for r in records
    ]


def mcnemar_exact_p(b: int, c: int) -> float:
    """p-value hai phía của McNemar chính xác: Binomial(n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(diffs: List[float], n_boot: int, seed: int) -> Tuple[float, float]:
    """CI95 (percentile) của trung bình chênh lệch, lấy mẫu lại theo cặp."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = means[int(round(0.025 * (n_boot - 1)))]
    hi = means[int(round(0.975 * (n_boot - 1)))]
    return lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_a", required=True, help="Thư mục run A (chứa predictions.json).")
    parser.add_argument("--run_b", required=True, help="Thư mục run B (chứa predictions.json).")
    parser.add_argument("--label_a", default="A", help="Nhãn hiển thị của run A.")
    parser.add_argument("--label_b", default="B", help="Nhãn hiển thị của run B.")
    parser.add_argument("--out", required=True, help="Đường dẫn file JSON đầu ra.")
    args = parser.parse_args()

    recs_a = load_predictions(args.run_a)
    recs_b = load_predictions(args.run_b)
    ids_a = [r["question_id"] for r in recs_a]
    ids_b = [r["question_id"] for r in recs_b]
    if ids_a != ids_b:
        raise ValueError(
            f"question_id của 2 run không giống hệt/cùng thứ tự (n_a={len(ids_a)}, n_b={len(ids_b)}, "
            f"khác tập: {len(set(ids_a) ^ set(ids_b))}) — không so sánh bắt cặp được."
        )
    n = len(ids_a)
    if n == 0:
        raise ValueError("Không có mẫu nào để so sánh.")

    sc_a = per_sample_scores(recs_a)
    sc_b = per_sample_scores(recs_b)

    n_both_correct = n_only_a = n_only_b = n_both_wrong = 0
    only_b_ids: List[str] = []
    for qid, sa, sb in zip(ids_a, sc_a, sc_b):
        ca, cb = sa["exact_match"] == 1.0, sb["exact_match"] == 1.0
        if ca and cb:
            n_both_correct += 1
        elif ca:
            n_only_a += 1
        elif cb:
            n_only_b += 1
            only_b_ids.append(qid)
        else:
            n_both_wrong += 1

    b, c = n_only_a, n_only_b
    p_value = mcnemar_exact_p(b, c)

    em_a = sum(s["exact_match"] for s in sc_a) / n
    em_b = sum(s["exact_match"] for s in sc_b) / n
    anls_a = sum(s["anls"] for s in sc_a) / n
    anls_b = sum(s["anls"] for s in sc_b) / n
    diffs = [sb["anls"] - sa["anls"] for sa, sb in zip(sc_a, sc_b)]
    anls_diff = sum(diffs) / n
    ci_lo, ci_hi = paired_bootstrap_ci(diffs, N_BOOTSTRAP, BOOTSTRAP_SEED)

    result = {
        "run_a": args.run_a.replace(os.sep, "/"),
        "run_b": args.run_b.replace(os.sep, "/"),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "n": n,
        "n_both_correct": n_both_correct,
        "n_only_a": n_only_a,
        "n_only_b": n_only_b,
        "n_both_wrong": n_both_wrong,
        "b": b,
        "c": c,
        "p_value": p_value,
        "mcnemar_test": "exact binomial, two-sided, n=b+c, p=0.5",
        "em_a": em_a,
        "em_b": em_b,
        "em_diff": em_b - em_a,
        "anls_threshold": ANLS_THRESHOLD,
        "anls_a": anls_a,
        "anls_b": anls_b,
        "anls_diff": anls_diff,
        "anls_ci95": [ci_lo, ci_hi],
        "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "method": "paired percentile"},
        "only_b_question_ids": only_b_ids,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    la, lb = args.label_a, args.label_b
    w = len(la) + 7
    cw = len(lb) + 8
    print(f"n = {n}   (hàng = {la}, cột = {lb}, theo exact_match)")
    print(f"{'':<{w}}{lb + ' đúng':>{cw}}{lb + ' sai':>{cw}}")
    print(f"{la + ' đúng':<{w}}{n_both_correct:>{cw}}{n_only_a:>{cw}}")
    print(f"{la + ' sai':<{w}}{n_only_b:>{cw}}{n_both_wrong:>{cw}}")
    print(f"b (chỉ {la} đúng) = {b}   c (chỉ {lb} đúng) = {c}")
    print(f"McNemar exact p-value = {p_value:.4f}")
    print(f"EM:   {la}={em_a:.4f}  {lb}={em_b:.4f}  diff={em_b - em_a:+.4f}")
    print(f"ANLS: {la}={anls_a:.4f}  {lb}={anls_b:.4f}  diff={anls_diff:+.4f}  "
          f"CI95=[{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
