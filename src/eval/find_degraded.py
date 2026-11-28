"""
find_degraded.py — Tìm các câu trả lời "hỏng" (degraded) trong predictions.json.

Output hỏng = sau .strip() chỉ gồm DUY NHẤT một ký tự lặp lại, dài >= 4
(vd "!!!!!!!!"). Đây là dấu hiệu model sinh lỗi số học (thường do dtype fp16),
không phải trả lời sai thông thường.

Chạy:
    python src/eval/find_degraded.py \
        --predictions experiments/<RUN>/predictions.json \
        --out experiments/<RUN_PROBE>/failed_ids.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List

MIN_REPEAT_LEN = 4


def is_degraded(pred: str) -> bool:
    """True nếu pred (sau strip) là 1 ký tự lặp lại, dài >= MIN_REPEAT_LEN."""
    if pred is None:
        return False
    s = str(pred).strip()
    return len(s) >= MIN_REPEAT_LEN and len(set(s)) == 1


def sha256_file(path: str) -> str:
    """Tính sha256 của file (đọc theo khối)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_degraded(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lọc các bản ghi có prediction hỏng, chỉ giữ các trường cần cho phân tích."""
    return [
        {
            "question_id": r.get("question_id"),
            "question": r.get("question"),
            "prediction": r.get("prediction"),
            "answers": r.get("answers"),
        }
        for r in records
        if is_degraded(r.get("prediction", ""))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="Đường dẫn predictions.json.")
    parser.add_argument("--out", required=True, help="Đường dẫn file JSON đầu ra.")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        raise ValueError(f"Kỳ vọng list trong {args.predictions}, nhận {type(records).__name__}")

    degraded = find_degraded(records)
    result = {
        "source_predictions": os.path.relpath(args.predictions).replace(os.sep, "/"),
        "source_sha256": sha256_file(args.predictions),
        "num_total": len(records),
        "num_degraded": len(degraded),
        "question_ids": [r["question_id"] for r in degraded],
        "records": degraded,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"num_total    = {result['num_total']}")
    print(f"num_degraded = {result['num_degraded']}")
    print(f"first_ids    = {result['question_ids'][:3]}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
