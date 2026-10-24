"""
check_split_overlap.py — Kiểm tra rò rỉ dữ liệu (data leakage) giữa train/dev và test.

Vấn đề: B2 (QLoRA 1 epoch) tăng EM ViVQA từ 0,22 lên 0,74. Mức tăng lớn như vậy phải loại trừ
khả năng test "lọt" vào train trước khi tin. Script đo, KHÔNG sửa dữ liệu:
    1. Ảnh trùng giữa các split (cùng tên file ảnh).
    2. Câu hỏi trùng nguyên văn (sau chuẩn hoá) giữa train và test, và cặp (câu hỏi, đáp án) trùng.
    3. Độ phủ đáp án: bao nhiêu % câu test có đáp án chuẩn đã xuất hiện trong tập đáp án train
       (đo mức "từ vựng đóng" của dataset — giải thích vì sao học định dạng đáp án lại tăng mạnh).
    4. Top đáp án train vs test.

Chạy (máy local hoặc Colab):
    python -m src.data.check_split_overlap --data_dir "G:/My Drive/ViVQA-VLM/data/vivqa" \
        --splits train val test --out experiments/W07_data_checks/vivqa_split_overlap.json
    python -m src.data.check_split_overlap --data_dir "G:/My Drive/ViVQA-VLM/data/vitextvqa_official" \
        --splits train dev test --out experiments/W07_data_checks/vitextvqa_split_overlap.json
Kiểm thử: python -m src.data.check_split_overlap --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.eval.metrics import normalize_vi  # noqa: E402


def _load(path: str) -> List[Dict[str, Any]]:
    raw = json.load(open(path, encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw["data"] if isinstance(raw.get("data"), list) else list(raw.values())
    return raw


def _answers(r: Dict[str, Any]) -> List[str]:
    a = r.get("answers")
    if a is None:
        a = [r.get("answer", "")]
    return [str(x) for x in (a if isinstance(a, list) else [a]) if str(x).strip()]


def _qnorm(q: str) -> str:
    return " ".join(normalize_vi(q).split())


def analyze(splits: Dict[str, List[Dict[str, Any]]], test_name: str = "test") -> Dict[str, Any]:
    imgs = {k: {os.path.basename(str(r.get("image", ""))) for r in v} for k, v in splits.items()}
    test = splits[test_name]
    out: Dict[str, Any] = {"sizes": {k: len(v) for k, v in splits.items()},
                           "num_images": {k: len(v) for k, v in imgs.items()}}
    out["image_overlap_with_test"] = {k: len(imgs[k] & imgs[test_name]) for k in splits if k != test_name}
    test_q = Counter(_qnorm(r["question"]) for r in test)
    test_qa = Counter((_qnorm(r["question"]), normalize_vi(_answers(r)[0])) for r in test if _answers(r))
    test_img_q = {(os.path.basename(str(r.get("image", ""))), _qnorm(r["question"])) for r in test}
    out["question_overlap_with_test"] = {}
    for k, v in splits.items():
        if k == test_name:
            continue
        qs = {_qnorm(r["question"]) for r in v}
        qas = {(_qnorm(r["question"]), normalize_vi(_answers(r)[0])) for r in v if _answers(r)}
        img_q = {(os.path.basename(str(r.get("image", ""))), _qnorm(r["question"])) for r in v}
        out["question_overlap_with_test"][k] = {
            "test_questions_seen_verbatim": sum(c for q, c in test_q.items() if q in qs),
            "test_question_answer_pairs_seen": sum(c for qa, c in test_qa.items() if qa in qas),
            "same_image_and_question": len(img_q & test_img_q),
        }
    train_name = next(k for k in splits if k != test_name)
    train_ans = {normalize_vi(a) for r in splits[train_name] for a in _answers(r)}
    covered = sum(1 for r in test if _answers(r) and normalize_vi(_answers(r)[0]) in train_ans)
    out["test_answer_in_train_vocab"] = {"train_split": train_name, "covered": covered, "total": len(test),
                                         "pct": round(100 * covered / max(len(test), 1), 2),
                                         "train_unique_answers": len(train_ans)}
    for k in (train_name, test_name):
        c = Counter(normalize_vi(_answers(r)[0]) for r in splits[k] if _answers(r))
        out[f"top20_answers_{k}"] = c.most_common(20)
    return out


def _selftest() -> None:
    tr = [{"question": "màu của con mèo là gì", "answers": ["màu đen"], "image": "a.jpg"},
          {"question": "cái gì trên bàn", "answers": ["cái cốc"], "image": "b.jpg"}]
    te = [{"question": "Màu của con mèo là gì?", "answers": ["đen"], "image": "c.jpg"},
          {"question": "cái gì ở đâu", "answers": ["con chó"], "image": "b.jpg"}]
    r = analyze({"train": tr, "test": te})
    assert r["image_overlap_with_test"]["train"] == 1
    q = r["question_overlap_with_test"]["train"]
    assert q["test_questions_seen_verbatim"] == 1 and q["test_question_answer_pairs_seen"] == 1, q
    assert q["same_image_and_question"] == 0
    assert r["test_answer_in_train_vocab"]["covered"] == 1     # "đen" (sau bỏ "màu ") có trong train
    print("SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kiểm tra rò rỉ train/dev -> test (chỉ đọc).")
    ap.add_argument("--data_dir")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Tên file split (không đuôi .json); split CUỐI là test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    splits = {s: _load(os.path.join(a.data_dir, f"{s}.json")) for s in a.splits}
    res = analyze(splits, test_name=a.splits[-1])
    res["data_dir"] = a.data_dir.replace("\\", "/")
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("top20")}, ensure_ascii=False, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("Ghi ->", a.out)


if __name__ == "__main__":
    main()
