"""
probe_breakdown.py — Tách metric của dtype probe (fp16 vs fp32) theo 2 nhóm.

- failed_group : các câu có question_id nằm trong failed_ids.json (câu hỏng ở fp16).
- control_group: các câu còn lại của tập probe.

Metric từng nhóm tính bằng src.eval.metrics.evaluate_predictions (không viết lại).
Ngoài ra đếm:
- n_recovered: câu trong failed_group SAI ở fp16 -> ĐÚNG ở fp32 (theo exact_match).
- n_regressed: câu trong control_group ĐÚNG ở fp16 -> SAI ở fp32.

Chạy:
    python src/eval/probe_breakdown.py \
        --fp16_dir experiments/W06_probe_fp16_qwen25vl_vitextvqa73_seed42 \
        --fp32_dir experiments/W06_probe_fp32_qwen25vl_vitextvqa73_seed42 \
        --failed_ids experiments/W06_dtype_probe_seed42/failed_ids.json \
        --out experiments/W06_dtype_probe_seed42/probe_breakdown.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Set

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.eval.metrics import evaluate_predictions  # noqa: E402

METRIC_NAMES = ["exact_match", "vqa_accuracy", "anls"]
ANLS_THRESHOLD = 0.5


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_predictions(run_dir: str) -> List[Dict[str, Any]]:
    records = load_json(os.path.join(run_dir, "predictions.json"))
    if not isinstance(records, list):
        raise ValueError(f"Kỳ vọng list trong {run_dir}/predictions.json, nhận {type(records).__name__}")
    return records


def group_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Metric trung bình trên 1 nhóm bản ghi."""
    return evaluate_predictions(
        predictions=[r.get("prediction", "") for r in records],
        references=[r.get("answers", []) for r in records],
        metric_names=METRIC_NAMES,
        anls_threshold=ANLS_THRESHOLD,
    )


def per_sample_em(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """{question_id: exact_match} cho từng mẫu."""
    return {
        r["question_id"]: evaluate_predictions(
            predictions=[r.get("prediction", "")],
            references=[r.get("answers", [])],
            metric_names=["exact_match"],
            anls_threshold=ANLS_THRESHOLD,
        )["exact_match"]
        for r in records
    }


def split_groups(records: List[Dict[str, Any]], failed: Set[str]):
    failed_group = [r for r in records if r["question_id"] in failed]
    control_group = [r for r in records if r["question_id"] not in failed]
    return failed_group, control_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp16_dir", required=True, help="Thư mục run fp16 (chứa predictions.json).")
    parser.add_argument("--fp32_dir", required=True, help="Thư mục run fp32 (chứa predictions.json).")
    parser.add_argument("--failed_ids", required=True, help="Đường dẫn failed_ids.json.")
    parser.add_argument("--out", required=True, help="Đường dẫn file JSON đầu ra.")
    args = parser.parse_args()

    failed: Set[str] = set(load_json(args.failed_ids)["question_ids"])

    runs = {"fp16": load_predictions(args.fp16_dir), "fp32": load_predictions(args.fp32_dir)}
    ids16 = {r["question_id"] for r in runs["fp16"]}
    ids32 = {r["question_id"] for r in runs["fp32"]}
    if ids16 != ids32:
        raise ValueError(f"question_id của 2 run không khớp (chỉ fp16: {len(ids16 - ids32)}, "
                         f"chỉ fp32: {len(ids32 - ids16)})")

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for name, records in runs.items():
        fg, cg = split_groups(records, failed)
        groups[name] = {"failed_group": fg, "control_group": cg}
        metrics[name] = {"failed_group": group_metrics(fg), "control_group": group_metrics(cg)}

    n_failed = len(groups["fp16"]["failed_group"])
    n_control = len(groups["fp16"]["control_group"])

    em16 = per_sample_em(runs["fp16"])
    em32 = per_sample_em(runs["fp32"])
    recovered_ids = sorted(q for q in ids16 if q in failed and em16[q] == 0.0 and em32[q] == 1.0)
    regressed_ids = sorted(q for q in ids16 if q not in failed and em16[q] == 1.0 and em32[q] == 0.0)

    result = {
        "fp16_dir": args.fp16_dir.replace(os.sep, "/"),
        "fp32_dir": args.fp32_dir.replace(os.sep, "/"),
        "failed_ids_file": args.failed_ids.replace(os.sep, "/"),
        "metric_names": METRIC_NAMES,
        "anls_threshold": ANLS_THRESHOLD,
        "n_failed": n_failed,
        "n_control": n_control,
        "metrics": metrics,
        "n_recovered": len(recovered_ids),
        "recovered_ids": recovered_ids,
        "n_regressed": len(regressed_ids),
        "regressed_ids": regressed_ids,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"{'run':<5} {'group':<14} {'n':>3} {'EM':>7} {'VQA':>7} {'ANLS':>7}")
    for name in ("fp16", "fp32"):
        for g in ("failed_group", "control_group"):
            m = metrics[name][g]
            n = len(groups[name][g])
            print(f"{name:<5} {g:<14} {n:>3} {m['exact_match']:>7.4f} "
                  f"{m['vqa_accuracy']:>7.4f} {m['anls']:>7.4f}")
    print(f"n_recovered = {len(recovered_ids)}")
    print(f"n_regressed = {len(regressed_ids)}  {regressed_ids}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
