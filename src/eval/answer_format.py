"""
answer_format.py — Tách lỗi "ĐỊNH DẠNG câu trả lời" khỏi lỗi "ĐỌC/HIỂU SAI".

Vấn đề: EM chấm sai cả khi model đọc đúng chữ nhưng nói dài ("Lịch tết năm 2021" vs gold "2021").
W07 R1 cho thấy Qwen2-VL-7B kém Qwen2.5-VL-3B 12 điểm EM chủ yếu vì trả lời dài. Script này đo,
trên cùng tập dự đoán:
    words          số từ trung bình của câu trả lời (sau normalize_vi) và % câu > 3 từ
    exact_match    EM như pipeline
    containment    % câu mà một đáp án gold (đã chuẩn hoá) nằm NGUYÊN CỤM TỪ trong câu trả lời
    contain_not_em % câu "đọc ra đúng đáp án nhưng EM vẫn sai" = phần mất vì định dạng
So ghép cặp containment giữa 2 run: McNemar exact (cùng hàm với compare_runs).

Giới hạn (ghi trong báo cáo): containment là chỉ số CHẨN ĐOÁN, rộng tay — câu trả lời rất dài dễ
"chứa" đáp án ngắn một cách tình cờ; không dùng thay EM/ANLS.

Chạy:   python -m src.eval.answer_format --run B1=experiments/<run_b1> --run B2=experiments/<run_b2> \
            [--pair B1,B2] [--out experiments/<dir>/answer_format.json]
Kiểm thử: python -m src.eval.answer_format --selftest
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
from src.eval.metrics import exact_match, normalize_vi  # noqa: E402
from src.eval.compare_runs import mcnemar_exact_p  # noqa: E402


def n_words(text: str) -> int:
    return len(normalize_vi(text).split())


def contains_gold(pred: str, answers: List[str]) -> bool:
    """True nếu một đáp án gold (chuẩn hoá) xuất hiện nguyên cụm từ (khớp ranh giới từ) trong câu trả lời."""
    p = f" {normalize_vi(pred)} "
    return any(f" {g} " in p for g in (normalize_vi(a) for a in answers) if g)


def per_sample(records: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    em = [exact_match(r["prediction"], r["answers"]) for r in records]
    ct = [1.0 if contains_gold(r["prediction"], r["answers"]) else 0.0 for r in records]
    return {"em": em, "contain": ct, "words": [n_words(r["prediction"]) for r in records]}


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    s = per_sample(records)
    n = len(records)
    return {"n": n,
            "mean_words": round(sum(s["words"]) / n, 2),
            "pct_gt3_words": round(100 * sum(w > 3 for w in s["words"]) / n, 2),
            "EM_pct": round(100 * sum(s["em"]) / n, 2),
            "containment_pct": round(100 * sum(s["contain"]) / n, 2),
            "contain_not_em_pct": round(100 * sum(c and not e for c, e in zip(s["contain"], s["em"])) / n, 2)}


def gold_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    w = [n_words(r["answers"][0]) for r in records]
    return {"mean_words": round(sum(w) / len(w), 2), "pct_gt3_words": round(100 * sum(x > 3 for x in w) / len(w), 2)}


def paired_containment(rec_a: List[Dict[str, Any]], rec_b: List[Dict[str, Any]]) -> Dict[str, Any]:
    if [r["question_id"] for r in rec_a] != [r["question_id"] for r in rec_b]:
        raise ValueError("hai run không cùng danh sách question_id theo đúng thứ tự")
    ca, cb = per_sample(rec_a)["contain"], per_sample(rec_b)["contain"]
    b = sum(1 for x, y in zip(ca, cb) if x and not y)
    c = sum(1 for x, y in zip(ca, cb) if y and not x)
    return {"only_a": b, "only_b": c, "mcnemar_p": mcnemar_exact_p(b, c),
            "delta_containment_pct": round(100 * (sum(cb) - sum(ca)) / len(ca), 2)}


def _selftest() -> None:
    assert contains_gold("Lịch tết năm 2021", ["2021"]) and not contains_gold("20215", ["2021"])
    assert contains_gold('Quán phở này tên là "Phở Sưa".', ["sưa"])
    assert not contains_gold("Phở Sữa", ["sưa"]), "giữ dấu thanh: sữa ≠ sưa"
    assert not contains_gold("con mèo", [""]) and n_words("Màu đỏ.") == 1   # 'màu ' bị bỏ ở đầu chuỗi
    recs_a = [{"question_id": "1", "prediction": "Lịch tết năm 2021", "answers": ["2021"]},
              {"question_id": "2", "prediction": "chó", "answers": ["mèo"]},
              {"question_id": "3", "prediction": "hà nội", "answers": ["Hà Nội"]}]
    recs_b = [dict(r, prediction=r["answers"][0]) for r in recs_a]
    sa, sb = summarize(recs_a), summarize(recs_b)
    assert sa["EM_pct"] == 33.33 and sa["containment_pct"] == 66.67 and sa["contain_not_em_pct"] == 33.33, sa
    assert sb["EM_pct"] == 100.0 and sb["contain_not_em_pct"] == 0.0
    pc = paired_containment(recs_a, recs_b)
    assert pc["only_a"] == 0 and pc["only_b"] == 1 and pc["delta_containment_pct"] == 33.33, pc
    assert gold_summary(recs_a)["mean_words"] == 1.33
    print("SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chẩn đoán định dạng câu trả lời (containment).")
    ap.add_argument("--run", action="append", default=[], help="NHÃN=thư mục run (lặp lại được)")
    ap.add_argument("--pair", action="append", default=[], help="NHÃN_A,NHÃN_B để so ghép cặp containment")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    runs = dict(x.split("=", 1) for x in a.run)
    recs = {k: json.load(open(os.path.join(v, "predictions.json"), encoding="utf-8")) for k, v in runs.items()}
    first = next(iter(recs.values()))
    res = {"gold": gold_summary(first), "runs": {k: summarize(v) for k, v in recs.items()},
           "pairs": {p: paired_containment(recs[p.split(",")[0]], recs[p.split(",")[1]]) for p in a.pair},
           "paths": {k: v.replace("\\", "/") for k, v in runs.items()},
           "note": "containment là chỉ số chẩn đoán (post-hoc), không thay EM/ANLS"}
    print(f"{'run':<8}{'EM':>7}{'chứa gold':>11}{'chứa≠EM':>9}{'từ TB':>7}{'>3 từ %':>9}")
    print(f"{'gold':<8}{'':>7}{'':>11}{'':>9}{res['gold']['mean_words']:>7}{res['gold']['pct_gt3_words']:>9}")
    for k, s in res["runs"].items():
        print(f"{k:<8}{s['EM_pct']:>7}{s['containment_pct']:>11}{s['contain_not_em_pct']:>9}"
              f"{s['mean_words']:>7}{s['pct_gt3_words']:>9}")
    for p, r in res["pairs"].items():
        print(f"{p}: chứa gold chỉ A {r['only_a']} | chỉ B {r['only_b']} | Δ {r['delta_containment_pct']:+} điểm | "
              f"McNemar p = {r['mcnemar_p']:.2g}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("Ghi ->", a.out)


if __name__ == "__main__":
    main()
