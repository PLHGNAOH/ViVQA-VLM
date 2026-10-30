"""
stratify_by_overlap.py — Tách kết quả B1/B2 theo mức "đã thấy trong train" của từng câu test.

Vấn đề: kiểm tra rò rỉ (W07) cho thấy ở ViVQA, 1.274/2.789 ảnh test (46%) cũng có trong train,
57 cặp (ảnh, câu hỏi) test có y hệt trong train, và 99,4% đáp án test thuộc 333 đáp án của train.
Đây là đặc điểm của split chính thức (nhóm không tự chia), nhưng nó có thể thổi phồng điểm của
model đã fine-tune. Script này đo mức thổi phồng đó, không sửa dữ liệu:

  Tập con (theo câu test):
    all                : toàn bộ test
    image_seen         : ảnh của câu có xuất hiện trong train
    image_unseen       : ảnh chưa từng có trong train
    qa_seen            : cặp (câu hỏi, đáp án) đã có nguyên văn trong train
    clean              : ảnh chưa thấy VÀ cặp (câu hỏi, đáp án) chưa thấy  <- ước lượng "sạch"
  Mỗi tập con: n, EM/ANLS của A và B, kiểm định McNemar chính xác, CI95 bootstrap của ΔEM.
  Kèm mốc "prior": đoán đáp án phổ biến nhất của train theo loại câu hỏi (không nhìn ảnh, không đọc câu hỏi)
  -> cho biết bao nhiêu điểm có được chỉ nhờ phân bố đáp án.

Chạy (local, đọc dữ liệu trên Drive):
    python -m src.eval.stratify_by_overlap --data_dir "G:/My Drive/ViVQA-VLM/data/vivqa" \
        --run_a experiments/W07_B1_zeroshot_qwen25vl_vivqa_test_bf16_L4_seed42 \
        --run_b experiments/W07_B2_eval_qlora_r16_ep1_vivqa_test_bf16_L4_seed42 \
        --label_a B1 --label_b B2 --out experiments/W07_B2_summary_L4_seed42/stratified_B1_vs_B2_vivqa.json
Kiểm thử: python -m src.eval.stratify_by_overlap --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.eval.metrics import anls, exact_match, normalize_vi  # noqa: E402
from src.eval.compare_runs import mcnemar_exact_p  # noqa: E402

SUBSETS = ["all", "image_seen", "image_unseen", "qa_seen", "clean"]


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


def _img(r: Dict[str, Any]) -> str:
    return os.path.basename(str(r.get("image", "")))


def _q(r: Dict[str, Any]) -> str:
    return " ".join(normalize_vi(r["question"]).split())


def align(test: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ghép dự đoán với bản ghi test theo question_id; nếu id không khớp thì theo thứ tự + kiểm câu hỏi."""
    by_id = {str(r.get("question_id")): r for r in test}
    if all(str(p["question_id"]) in by_id for p in preds):
        return [by_id[str(p["question_id"])] for p in preds]
    if len(test) != len(preds):
        raise ValueError(f"Không ghép được: {len(test)} câu test vs {len(preds)} dự đoán")
    for t, p in zip(test, preds):
        if _q(t) != _q(p):
            raise ValueError(f"Thứ tự lệch: {t['question']!r} vs {p['question']!r}")
    return list(test)


def subset_masks(train: List[Dict[str, Any]], test_aligned: List[Dict[str, Any]]) -> Dict[str, List[bool]]:
    imgs = {_img(r) for r in train}
    qa = {(_q(r), normalize_vi(_answers(r)[0])) for r in train if _answers(r)}
    img_seen = [_img(t) in imgs for t in test_aligned]
    qa_seen = [bool(_answers(t)) and (_q(t), normalize_vi(_answers(t)[0])) in qa for t in test_aligned]
    return {"all": [True] * len(test_aligned), "image_seen": img_seen,
            "image_unseen": [not s for s in img_seen], "qa_seen": qa_seen,
            "clean": [(not i) and (not q) for i, q in zip(img_seen, qa_seen)]}


def prior_predictions(train: List[Dict[str, Any]], test_aligned: List[Dict[str, Any]]) -> List[str]:
    """Đáp án phổ biến nhất của train theo question_type (không nhìn ảnh/câu hỏi)."""
    by_type: Dict[str, Counter] = defaultdict(Counter)
    for r in train:
        if _answers(r):
            by_type[r.get("question_type", "unknown")][normalize_vi(_answers(r)[0])] += 1
    overall = Counter()
    for c in by_type.values():
        overall.update(c)
    out = []
    for t in test_aligned:
        c = by_type.get(t.get("question_type", "unknown")) or overall
        out.append(c.most_common(1)[0][0] if c else "")
    return out


def _boot_ci(diffs: List[float], n_boot: int = 5000, seed: int = 42):
    if not diffs:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return [means[int(0.025 * (n_boot - 1))], means[int(0.975 * (n_boot - 1))]]


def stratify(train, test, preds_a, preds_b, label_a="A", label_b="B") -> Dict[str, Any]:
    t_al = align(test, preds_a)
    if [str(p["question_id"]) for p in preds_a] != [str(p["question_id"]) for p in preds_b]:
        pb = {str(p["question_id"]): p for p in preds_b}
        preds_b = [pb[str(p["question_id"])] for p in preds_a]
    masks = subset_masks(train, t_al)
    prior = prior_predictions(train, t_al)
    em_a = [exact_match(p["prediction"], p["answers"]) for p in preds_a]
    em_b = [exact_match(p["prediction"], p["answers"]) for p in preds_b]
    an_a = [anls(p["prediction"], p["answers"]) for p in preds_a]
    an_b = [anls(p["prediction"], p["answers"]) for p in preds_b]
    em_p = [exact_match(pr, p["answers"]) for pr, p in zip(prior, preds_a)]
    res = {"labels": [label_a, label_b], "subsets": {}}
    for name in SUBSETS:
        idx = [i for i, m in enumerate(masks[name]) if m]
        n = len(idx)
        if n == 0:
            res["subsets"][name] = {"n": 0}
            continue
        b = sum(1 for i in idx if em_a[i] and not em_b[i])
        c = sum(1 for i in idx if em_b[i] and not em_a[i])
        mean = lambda xs: sum(xs[i] for i in idx) / n
        res["subsets"][name] = {
            "n": n, "pct_of_test": round(100 * n / len(t_al), 2),
            f"EM_{label_a}": mean(em_a), f"EM_{label_b}": mean(em_b), "EM_prior": mean(em_p),
            f"ANLS_{label_a}": mean(an_a), f"ANLS_{label_b}": mean(an_b),
            "delta_EM": mean(em_b) - mean(em_a),
            "delta_EM_ci95": _boot_ci([em_b[i] - em_a[i] for i in idx]),
            f"only_{label_a}_correct": b, f"only_{label_b}_correct": c, "mcnemar_p": mcnemar_exact_p(b, c),
        }
    res["prior_rule"] = "đáp án train phổ biến nhất theo question_type (không nhìn ảnh, không đọc câu hỏi)"
    return res


def _selftest() -> None:
    train = [{"question": "màu gì", "answers": ["đỏ"], "image": "a.jpg", "question_type": "color"},
             {"question": "màu gì", "answers": ["đỏ"], "image": "b.jpg", "question_type": "color"},
             {"question": "cái gì", "answers": ["chó"], "image": "c.jpg", "question_type": "object"}]
    test = [{"question_id": "t0", "question": "màu gì", "answers": ["đỏ"], "image": "a.jpg", "question_type": "color"},
            {"question_id": "t1", "question": "con gì", "answers": ["mèo"], "image": "z.jpg", "question_type": "object"},
            {"question_id": "t2", "question": "màu nào", "answers": ["xanh"], "image": "y.jpg", "question_type": "color"}]
    pa = [{"question_id": f"t{i}", "question": t["question"], "answers": t["answers"], "prediction": "sai"}
          for i, t in enumerate(test)]
    pb = [dict(p, prediction=t["answers"][0]) for p, t in zip(pa, test)]
    r = stratify(train, test, pa, pb, "B1", "B2")
    s = r["subsets"]
    assert s["all"]["n"] == 3 and s["image_seen"]["n"] == 1 and s["qa_seen"]["n"] == 1
    assert s["clean"]["n"] == 2 and s["clean"]["EM_B2"] == 1.0 and s["clean"]["EM_B1"] == 0.0
    assert abs(s["all"]["EM_prior"] - 1 / 3) < 1e-9       # prior đoán "đỏ" cho color -> đúng t0
    assert s["all"]["only_B2_correct"] == 3
    print("SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tách kết quả theo mức trùng với train (chỉ đọc).")
    ap.add_argument("--data_dir")
    ap.add_argument("--train", default="train")
    ap.add_argument("--test", default="test")
    ap.add_argument("--run_a")
    ap.add_argument("--run_b")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    train = _load(os.path.join(a.data_dir, f"{a.train}.json"))
    test = _load(os.path.join(a.data_dir, f"{a.test}.json"))
    pa = json.load(open(os.path.join(a.run_a, "predictions.json"), encoding="utf-8"))
    pb = json.load(open(os.path.join(a.run_b, "predictions.json"), encoding="utf-8"))
    res = stratify(train, test, pa, pb, a.label_a, a.label_b)
    res.update({"data_dir": a.data_dir.replace("\\", "/"), "run_a": a.run_a.replace("\\", "/"),
                "run_b": a.run_b.replace("\\", "/")})
    la, lb = a.label_a, a.label_b
    print(f"{'tập con':<14}{'n':>6}{'%test':>7}{'EM_' + la:>9}{'EM_' + lb:>9}{'EM_prior':>10}{'ΔEM':>9}  CI95            McNemar p")
    for k, s in res["subsets"].items():
        if s["n"]:
            lo, hi = s["delta_EM_ci95"]
            print(f"{k:<14}{s['n']:>6}{s['pct_of_test']:>7}{s['EM_' + la]:>9.4f}{s['EM_' + lb]:>9.4f}"
                  f"{s['EM_prior']:>10.4f}{s['delta_EM']:>+9.4f}  [{lo:+.3f}; {hi:+.3f}]  {s['mcnemar_p']:.2g}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("Ghi ->", a.out)


if __name__ == "__main__":
    main()
