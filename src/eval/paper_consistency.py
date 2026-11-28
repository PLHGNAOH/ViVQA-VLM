"""
paper_consistency.py — Đặt kết quả của nhóm cạnh số CÔNG BỐ của paper ViTextVQA, có tính sai số mẫu.

Vấn đề: so một con số của nhóm với một con số trong paper bằng mắt ("28,8 gần 28,0 -> khớp") là không
đủ: con số của paper cũng là một ƯỚC LƯỢNG, có sai số. Bảng 7 của paper ViTextVQA chỉ chấm trên 100 câu
test chọn ngẫu nhiên, nên sai số chuẩn của EM khoảng 4,5 điểm (khoảng tin cậy 95% rộng khoảng ±9 điểm).
Script này làm phép so sánh có kiểm soát sai số, với luật CHỐT TRƯỚC khi chạy:

  (1) consistency_vs_paper — R1 (Qwen2-VL-7B zero-shot) so với Bảng 7 (n=100):
      z = (ours - paper) / sqrt(s^2/100 + s^2/n_ours), s = độ lệch chuẩn từng câu của run nhóm.
      |z| <= 1,96 -> "NHẤT QUÁN với paper" (không bác bỏ được là cùng một mức); ngược lại -> "LỆCH".
      Ghi kèm "độ lệch nhỏ nhất phát hiện được" (MDE) để không đọc "nhất quán" thành "chứng minh giống nhau".
  (2) claim_vs_published — B1/B2 full test (10.028 câu) so với ViTextBLIP-2 (Bảng 6, full test):
      CI95 bootstrap của nhóm nằm TRỌN trên số công bố -> "CAO HƠN"; trọn dưới -> "THẤP HƠN";
      còn lại -> "KHÔNG PHÂN BIỆT ĐƯỢC". Số công bố coi như cố định (paper không công bố sai số).

Giới hạn (phải ghi trong báo cáo): paper KHÔNG công bố prompt, checkpoint, cách chuẩn hoá khi tính EM/F1.
F1 của nhóm là token-F1 trên chuỗi đã chuẩn hoá bằng normalize_vi (NFC, chữ thường, bỏ dấu câu, giữ dấu
thanh). Khác cách chuẩn hoá có thể dịch chuyển vài điểm -> mọi kết luận ở đây là "trong phạm vi sai số mẫu,
với giả định cách chấm tương đương".

Nguồn: Nguyen Q.V. et al., "ViTextVQA: A Large-Scale Visual Question Answering Dataset for Evaluating
Vietnamese Text Comprehension in Images", Expert Systems with Applications (2025), arXiv:2404.10652v5 —
Bảng 6 (ViTextBLIP-2, full test) và Bảng 7 (100 câu test ngẫu nhiên, zero-shot).

Chạy:  python -m src.eval.paper_consistency --run experiments/<run> --mode table7
       python -m src.eval.paper_consistency --run experiments/<run> --mode vitextblip2
Kiểm thử: python -m src.eval.paper_consistency --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.eval.metrics import exact_match, token_f1  # noqa: E402

PAPER_SOURCE = ("Nguyen et al., ViTextVQA, Expert Systems with Applications 2025, arXiv:2404.10652v5")
TABLE7_QWENVL7B = {"what": "Table 7, dòng 'QwenVL-7b', zero-shot, 100 câu test ngẫu nhiên (prompt/checkpoint không công bố)",
                   "em": 28.00, "f1": 53.80, "n": 100}
TABLE6_VITEXTBLIP2 = {"what": "Table 6, ViTextBLIP-2 (SOTA của paper), full test",
                      "em": 25.48, "f1": 53.95,
                      "note": "Table 10/12 cùng paper ghi 24,83 EM / 52,51 F1 cho mô hình đầy đủ; paper không giải thích"}
Z95 = 1.959964


def per_sample_scores(records: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """EM và token-F1 từng câu (0..1), dùng đúng metric của pipeline nhóm."""
    return {"em": [exact_match(r["prediction"], r["answers"]) for r in records],
            "f1": [token_f1(r["prediction"], r["answers"]) for r in records]}


def _mean_sd(xs: List[float]):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var)


def consistency_vs_paper(scores: List[float], paper_pct: float, paper_n: int) -> Dict[str, Any]:
    """Luật (1). scores: điểm từng câu 0..1 của nhóm; paper_pct: số công bố (%) trên paper_n câu."""
    n = len(scores)
    m, s = _mean_sd(scores)
    se_diff = s * math.sqrt(1.0 / paper_n + 1.0 / n)
    diff = 100 * m - paper_pct
    z = diff / (100 * se_diff) if se_diff > 0 else (0.0 if diff == 0 else math.inf)
    half_paper = 100 * Z95 * s / math.sqrt(paper_n)
    return {"ours_pct": round(100 * m, 2), "paper_pct": paper_pct, "n_ours": n, "n_paper": paper_n,
            "diff_points": round(diff, 2), "z": round(z, 3),
            "paper_ci95_pct": [round(paper_pct - half_paper, 2), round(paper_pct + half_paper, 2)],
            "min_detectable_diff_points": round(100 * Z95 * se_diff, 2),
            "verdict": "NHẤT QUÁN với paper" if abs(z) <= Z95 else "LỆCH so với paper"}


def bootstrap_ci(scores: List[float], n_boot: int = 10_000, seed: int = 42) -> List[float]:
    """CI95 percentile của trung bình (0..1). Dùng numpy (Colab có sẵn)."""
    import numpy as np
    a = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(0, n_boot, 500):                       # theo khối: đỡ tốn RAM với 10.028 câu
        k = min(500, n_boot - i)
        means[i:i + k] = a[rng.integers(0, len(a), size=(k, len(a)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def claim_vs_published(scores: List[float], published_pct: float, n_boot: int = 10_000) -> Dict[str, Any]:
    """Luật (2)."""
    m = sum(scores) / len(scores)
    lo, hi = bootstrap_ci(scores, n_boot=n_boot)
    if 100 * lo > published_pct:
        verdict = "CAO HƠN số công bố"
    elif 100 * hi < published_pct:
        verdict = "THẤP HƠN số công bố"
    else:
        verdict = "KHÔNG PHÂN BIỆT ĐƯỢC"
    return {"ours_pct": round(100 * m, 2), "ours_ci95_pct": [round(100 * lo, 2), round(100 * hi, 2)],
            "published_pct": published_pct, "diff_points": round(100 * m - published_pct, 2), "n": len(scores),
            "verdict": verdict}


def table7_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    sc = per_sample_scores(records)
    em = consistency_vs_paper(sc["em"], TABLE7_QWENVL7B["em"], TABLE7_QWENVL7B["n"])
    f1 = consistency_vs_paper(sc["f1"], TABLE7_QWENVL7B["f1"], TABLE7_QWENVL7B["n"])
    both = em["verdict"].startswith("NHẤT QUÁN") and f1["verdict"].startswith("NHẤT QUÁN")
    return {"reference": TABLE7_QWENVL7B, "source": PAPER_SOURCE, "EM": em, "F1": f1,
            "rule": "nhất quán nếu |z| <= 1,96 cho CẢ EM và F1 (z tính cả sai số 100 câu của paper)",
            "verdict": "NHẤT QUÁN với Table 7 (trong phạm vi sai số 100 câu)" if both else "LỆCH so với Table 7"}


def vitextblip2_report(records: List[Dict[str, Any]], n_boot: int = 10_000) -> Dict[str, Any]:
    sc = per_sample_scores(records)
    return {"reference": TABLE6_VITEXTBLIP2, "source": PAPER_SOURCE,
            "EM": claim_vs_published(sc["em"], TABLE6_VITEXTBLIP2["em"], n_boot),
            "F1": claim_vs_published(sc["f1"], TABLE6_VITEXTBLIP2["f1"], n_boot),
            "rule": "CAO HƠN nếu cận dưới CI95 bootstrap > số công bố (xét riêng EM và F1)"}


def _selftest() -> None:
    # (1) 2.000 câu EM = 28,8% so với paper 28,0% trên 100 câu -> nhất quán; MDE khoảng 9 điểm
    sc = [1.0] * 576 + [0.0] * 1424
    r = consistency_vs_paper(sc, 28.0, 100)
    assert r["ours_pct"] == 28.8 and r["verdict"].startswith("NHẤT QUÁN"), r
    assert 8.5 < r["min_detectable_diff_points"] < 9.5, r
    assert r["paper_ci95_pct"][0] < 20 and r["paper_ci95_pct"][1] > 36, r
    # EM 45% -> lệch 17 điểm -> LỆCH
    assert consistency_vs_paper([1.0] * 900 + [0.0] * 1100, 28.0, 100)["verdict"].startswith("LỆCH")
    # (2) 10.028 câu EM 30% vs 25,48 -> CAO HƠN; 25,6% -> không phân biệt được; 20% -> THẤP HƠN
    for k, want in [(3008, "CAO HƠN"), (2567, "KHÔNG PHÂN BIỆT"), (2006, "THẤP HƠN")]:
        got = claim_vs_published([1.0] * k + [0.0] * (10028 - k), 25.48, n_boot=2000)["verdict"]
        assert got.startswith(want), (k, got)
    # báo cáo đầy đủ chạy được trên bản ghi dự đoán
    recs = [{"prediction": "Hà Nội", "answers": ["hà nội"]}, {"prediction": "chợ lớn", "answers": ["chợ"]},
            {"prediction": "x", "answers": ["y"]}] * 40
    t7 = table7_report(recs)
    assert t7["EM"]["ours_pct"] == 33.33 and abs(t7["F1"]["ours_pct"] - 55.56) < 0.01, t7
    assert set(vitextblip2_report(recs, n_boot=500)) >= {"EM", "F1", "rule"}
    print("SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="So kết quả nhóm với số công bố của paper ViTextVQA, có tính sai số.")
    ap.add_argument("--run", help="Thư mục run có predictions.json")
    ap.add_argument("--mode", choices=["table7", "vitextblip2"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.run or not a.mode:
        ap.error("cần --run và --mode (trừ khi --selftest)")
    recs = json.load(open(os.path.join(a.run, "predictions.json"), encoding="utf-8"))
    res = table7_report(recs) if a.mode == "table7" else vitextblip2_report(recs)
    res["run"] = a.run.replace("\\", "/")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("Ghi ->", a.out)


if __name__ == "__main__":
    main()
