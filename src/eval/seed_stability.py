"""
seed_stability.py — Gộp kết quả so sánh ghép cặp (compare_runs.py) qua nhiều seed huấn luyện
và áp LUẬT KẾT LUẬN ghi trước trong docs/W09/README.md.

Vấn đề: kết quả chính W08 (B3 > B2-sub, p = 1,7e-6) mới có 1 seed (42). Một kết quả 1 seed có thể
do "may" (khởi tạo LoRA, thứ tự batch). Script này trả lời: lợi ích có giữ được khi đổi seed không?

Đầu vào: mỗi seed một file paired_*.json do compare_runs.py tạo (A = mốc so sánh, B = run cải tiến).
Đầu ra: bảng từng seed + mean ± std (độ lệch chuẩn mẫu, ddof = 1) + kết luận theo luật.

Luật (ghi TRƯỚC khi chạy seed 43/44 — docs/W09/README.md):
  - CẢI THIỆN ỔN ĐỊNH            : đủ số seed yêu cầu, MỌI seed có EM_B > EM_A và McNemar p < 0,05.
  - HƯỚNG NHẤT QUÁN, CHƯA ỔN ĐỊNH: đủ seed, mọi seed EM_B > EM_A nhưng có seed p >= 0,05.
  - KHÔNG ỔN ĐỊNH                : đủ seed, có seed EM_B <= EM_A.
  - CHƯA ĐỦ SEED                 : chưa đủ số seed yêu cầu (chỉ in bảng tạm, KHÔNG kết luận).

Chạy:
    python -m src.eval.seed_stability \
        --paired 42=<dir>/paired_seed42.json 43=<dir>/paired_seed43.json 44=<dir>/paired_seed44.json \
        --required_seeds 42 43 44 --out <dir>/seed_stability_summary.json

Kiểm thử offline:
    python -m src.eval.seed_stability --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional

ALPHA = 0.05
VERDICTS = ("CẢI THIỆN ỔN ĐỊNH", "HƯỚNG NHẤT QUÁN, CHƯA ỔN ĐỊNH", "KHÔNG ỔN ĐỊNH", "CHƯA ĐỦ SEED")


def mean_std(xs: List[float]) -> Dict[str, Optional[float]]:
    """Trung bình + độ lệch chuẩn mẫu (ddof = 1). 1 giá trị -> std = None (không ước lượng được)."""
    n = len(xs)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else None
    return {"mean": m, "std": sd, "n": n}


def verdict(per_seed: List[Dict[str, Any]], required_seeds: List[int], alpha: float = ALPHA) -> str:
    have = {r["seed"] for r in per_seed}
    if not set(required_seeds) <= have:
        return VERDICTS[3]
    rows = [r for r in per_seed if r["seed"] in set(required_seeds)]
    if any(r["em_b"] <= r["em_a"] for r in rows):
        return VERDICTS[2]
    if all(r["p_value"] < alpha for r in rows):
        return VERDICTS[0]
    return VERDICTS[1]


def summarize(paired: Dict[int, Dict[str, Any]], required_seeds: List[int], alpha: float = ALPHA) -> Dict[str, Any]:
    """paired: {seed: nội dung paired_*.json}. Kiểm cùng tập câu hỏi (n) giữa các seed."""
    if not paired:
        raise ValueError("Không có seed nào")
    ns = {int(s): p["n"] for s, p in paired.items()}
    if len(set(ns.values())) != 1:
        raise ValueError(f"Các seed đánh giá trên số câu khác nhau: {ns}")
    rows = []
    for s in sorted(paired):
        p = paired[s]
        rows.append({"seed": int(s), "n": p["n"], "em_a": p["em_a"], "em_b": p["em_b"],
                     "em_diff": p["em_b"] - p["em_a"], "n_only_a": p["n_only_a"], "n_only_b": p["n_only_b"],
                     "p_value": p["p_value"], "anls_a": p["anls_a"], "anls_b": p["anls_b"],
                     "anls_diff": p["anls_diff"], "anls_ci95": p["anls_ci95"],
                     "run_a": p.get("run_a"), "run_b": p.get("run_b")})
    agg = {k: mean_std([r[k] for r in rows]) for k in ("em_a", "em_b", "em_diff", "anls_a", "anls_b", "anls_diff")}
    return {"label_a": next(iter(paired.values())).get("label_a", "A"),
            "label_b": next(iter(paired.values())).get("label_b", "B"),
            "required_seeds": list(required_seeds), "alpha": alpha,
            "rule": ("CẢI THIỆN ỔN ĐỊNH nếu mọi seed EM_B > EM_A và McNemar exact p < alpha; "
                     "HƯỚNG NHẤT QUÁN, CHƯA ỔN ĐỊNH nếu mọi seed EM_B > EM_A nhưng có seed p >= alpha; "
                     "KHÔNG ỔN ĐỊNH nếu có seed EM_B <= EM_A; CHƯA ĐỦ SEED nếu thiếu seed"),
            "per_seed": rows, "aggregate_mean_std_ddof1": agg,
            "verdict": verdict(rows, required_seeds, alpha)}


def _fmt(ms: Dict[str, Optional[float]], pct: bool = False) -> str:
    k = 100 if pct else 1
    if ms["mean"] is None:
        return "—"
    sd = "n/a" if ms["std"] is None else f"{ms['std'] * k:.2f}" if pct else f"{ms['std']:.4f}"
    return f"{ms['mean'] * k:.2f} ± {sd}" if pct else f"{ms['mean']:.4f} ± {sd}"


def print_summary(s: Dict[str, Any]) -> None:
    la, lb = s["label_a"], s["label_b"]
    print(f"{'seed':>5} | {'EM ' + la:>12} | {'EM ' + lb:>10} | {'ΔEM':>7} | {'chỉ A/chỉ B':>11} | {'p':>9} | ΔANLS (CI95)")
    for r in s["per_seed"]:
        print(f"{r['seed']:>5} | {r['em_a']:>12.4f} | {r['em_b']:>10.4f} | {100 * r['em_diff']:>+6.2f}đ | "
              f"{r['n_only_a']:>5}/{r['n_only_b']:<5} | {r['p_value']:>9.3g} | {r['anls_diff']:+.4f} "
              f"[{r['anls_ci95'][0]:+.4f}; {r['anls_ci95'][1]:+.4f}]")
    a = s["aggregate_mean_std_ddof1"]
    print(f"mean ± std (điểm EM): {la} {_fmt(a['em_a'], True)} | {lb} {_fmt(a['em_b'], True)} | "
          f"Δ {_fmt(a['em_diff'], True)}")
    print(f"mean ± std (ANLS)   : {la} {_fmt(a['anls_a'])} | {lb} {_fmt(a['anls_b'])} | Δ {_fmt(a['anls_diff'])}")
    print(f"➜ KẾT LUẬN THEO LUẬT ({'/'.join(map(str, s['required_seeds']))}): {s['verdict']}")


def _selftest() -> None:
    def P(n, ea, eb, a, b, p, da=0.01):
        return {"n": n, "em_a": ea, "em_b": eb, "n_only_a": a, "n_only_b": b, "p_value": p,
                "anls_a": 0.68, "anls_b": 0.68 + da, "anls_diff": da, "anls_ci95": [da - 0.01, da + 0.01],
                "label_a": "B2sub", "label_b": "B3"}
    req = [42, 43, 44]
    # đủ 3 seed, cùng chiều, đều p < 0,05 -> ổn định
    s = summarize({42: P(2000, .4555, .4925, 81, 155, 1.7e-6), 43: P(2000, .46, .49, 90, 150, 1e-4),
                   44: P(2000, .45, .48, 95, 155, 2e-4)}, req)
    assert s["verdict"] == VERDICTS[0], s["verdict"]
    assert abs(s["aggregate_mean_std_ddof1"]["em_diff"]["mean"] - (0.037 + 0.03 + 0.03) / 3) < 1e-9
    # cùng chiều nhưng 1 seed p >= 0,05
    s = summarize({42: P(2000, .45, .49, 81, 155, 1e-6), 43: P(2000, .46, .47, 120, 140, 0.24),
                   44: P(2000, .45, .48, 95, 155, 2e-4)}, req)
    assert s["verdict"] == VERDICTS[1]
    # 1 seed đảo chiều -> không ổn định
    s = summarize({42: P(2000, .45, .49, 81, 155, 1e-6), 43: P(2000, .47, .46, 140, 120, 0.24),
                   44: P(2000, .45, .48, 95, 155, 2e-4)}, req)
    assert s["verdict"] == VERDICTS[2]
    # thiếu seed -> chưa kết luận; 1 seed -> std None
    s = summarize({42: P(2000, .45, .49, 81, 155, 1e-6)}, req)
    assert s["verdict"] == VERDICTS[3] and s["aggregate_mean_std_ddof1"]["em_b"]["std"] is None
    # khác số câu giữa các seed -> lỗi
    try:
        summarize({42: P(2000, .45, .49, 81, 155, 1e-6), 43: P(1999, .45, .49, 81, 155, 1e-6)}, req)
        raise AssertionError("không chặn n khác nhau")
    except ValueError:
        pass
    # std mẫu ddof=1: [1, 3] -> 1,4142
    assert abs(mean_std([1.0, 3.0])["std"] - math.sqrt(2)) < 1e-12
    print("SELFTEST OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paired", nargs="*", default=[], help="seed=đường_dẫn_paired.json (nhiều cặp)")
    ap.add_argument("--required_seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return 0
    if not a.paired:
        ap.error("cần --paired seed=path ...")
    paired = {}
    for item in a.paired:
        seed, path = item.split("=", 1)
        with open(path, encoding="utf-8") as fh:
            paired[int(seed)] = json.load(fh)
    s = summarize(paired, a.required_seeds)
    print_summary(s)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(s, fh, ensure_ascii=False, indent=2)
        print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
