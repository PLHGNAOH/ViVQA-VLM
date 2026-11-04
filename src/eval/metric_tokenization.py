"""
metric_tokenization.py — W15 (góp ý Review 2, G2 + G3): chuẩn hoá văn bản và tách từ tiếng Việt khi chấm điểm.

VẤN ĐỀ (hội đồng Review 2, 29/10/2026):
  G2 — phải gọi đúng tên từng kỹ thuật chuẩn hoá và có số liệu (mỗi bước ảnh hưởng bao nhiêu câu).
  G3 — tiếng Việt có từ ghép ("thương mại điện tử"), tiếng Anh chủ yếu từ đơn: metric tính trên đơn vị nào,
       tách từ thế nào, có làm đổi kết luận không.

CÁCH LÀM (chỉ đọc predictions.json đã có trong experiments/, không cần GPU):
  1) Thang chuẩn hoá cộng dồn S0 → S6 (S6 = metrics.normalize_vi, dùng cho mọi số đã báo cáo):
       S0 chuỗi thô (chỉ strip) · S1 + Unicode NFC · S2 + chữ thường · S3 + regex bỏ dấu câu [^\\w\\s]
       · S4 + gộp khoảng trắng \\s+ · S5 + bỏ tiền tố "màu " · S6 + số đứng một mình chữ → chữ số.
     Với mỗi run: EM ở từng bậc; với gold: % đáp án bị bậc đó thay đổi.
  2) Đơn vị tính:
       EM   so CẢ CHUỖI sau chuẩn hoá -> tách từ không làm đổi EM (cùng chuỗi -> cùng cách tách).
       ANLS so theo KÝ TỰ (Levenshtein) -> không phụ thuộc tách từ.
       token-F1 (metric phụ) đếm theo ÂM TIẾT (tách bằng khoảng trắng) -> có phụ thuộc.
     Nên đo lại token-F1 theo TỪ: tách từ tiếng Việt bằng pyvi (ViTokenizer, CRF; "thương_mại điện_tử")
     rồi tính F1 trên từ. So: điểm từng run, chênh lệch giữa các run (CI95 bootstrap ghép cặp, seed 42),
     và thứ hạng các run có đổi không.

Chạy:  python -m src.eval.metric_tokenization --run      (ghi experiments/W15_metric_tokenization/)
Kiểm thử: python -m src.eval.metric_tokenization --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Dict, List, Tuple

from src.eval.metrics import (_canonical_number, _strip_type_prefix, _token_f1_single, exact_match,
                              normalize_vi, token_f1)

VERSION = "metric-tokenization-v1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_REL = "experiments/W15_metric_tokenization"
N_BOOT = 2000
SEED = 42

# Model chính thức (configs/official_models.json) — run trong experiments/
RUNS: Dict[str, Dict[str, str]] = {
    "vivqa_test": {
        "B1": "W07_B1_zeroshot_qwen25vl_vivqa_test_bf16_L4_seed42",
        "C3": "W09_C3_eval_qlora_r16_vivqa_test_bf16_L4_seed42",
    },
    "vitextvqa_testfull": {
        "B1": "W07_B1_zeroshot_qwen25vl_vitextvqa_testfull_bf16_L4_seed42",
        "B2": "W07_B2_eval_qlora_r16_ep1_vitextvqa_testfull_bf16_L4_seed42",
        "B3-full": "W10_B3full_eval_qlora_ocr_vietocr_v1_r16_full_ep1_vitextvqa_testfull_bf16_L4_seed42",
    },
}
PAIRS = {"vivqa_test": [("B1", "C3")], "vitextvqa_testfull": [("B1", "B2"), ("B2", "B3-full"), ("B1", "B3-full")]}

STEPS: List[Tuple[str, str, Callable[[str], str]]] = [
    ("S0", "chuỗi thô (strip)", lambda t: t.strip()),
    ("S1", "+ Unicode NFC", lambda t: unicodedata.normalize("NFC", t)),
    ("S2", "+ chữ thường (str.lower)", lambda t: t.lower().strip()),
    ("S3", "+ regex bỏ dấu câu [^\\w\\s] (giữ dấu thanh)", lambda t: re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)),
    ("S4", "+ gộp khoảng trắng \\s+", lambda t: re.sub(r"\s+", " ", t).strip()),
    ("S5", "+ bỏ tiền tố 'màu ' ở đầu", _strip_type_prefix),
    ("S6", "+ số đứng một mình: chữ → chữ số (= normalize_vi)", _canonical_number),
]


# --------------------------------------------------------------------------- #
# Hàm thuần
# --------------------------------------------------------------------------- #
def ladder(text: str) -> List[str]:
    """Chuỗi sau từng bậc S0..S6 (cộng dồn)."""
    out, t = [], "" if text is None else str(text)
    for _, _, f in STEPS:
        t = f(t)
        out.append(t)
    return out


def em_at_step(pred: str, gts: List[str], k: int) -> float:
    p = ladder(pred)[k]
    return 1.0 if p in {ladder(g)[k] for g in gts} else 0.0


def make_word_seg():
    from pyvi import ViTokenizer

    cache: Dict[str, List[str]] = {}

    def seg(norm_text: str) -> List[str]:
        if norm_text not in cache:
            cache[norm_text] = ViTokenizer.tokenize(norm_text).split() if norm_text else []
        return cache[norm_text]
    return seg


def word_f1(pred: str, gts: List[str], seg: Callable[[str], List[str]]) -> float:
    """token-F1 trên TỪ (pyvi), cùng normalize_vi; từ ghép là một token ('điện_thoại')."""
    p = " ".join(seg(normalize_vi(pred)))
    return max((_token_f1_single(p, " ".join(seg(normalize_vi(g)))) for g in gts), default=0.0)


def paired_boot(a: List[float], b: List[float], n_boot: int = N_BOOT, seed: int = SEED) -> Dict[str, float]:
    """Chênh b − a (điểm %) + CI95 bootstrap ghép cặp (percentile, numpy default_rng(seed))."""
    import numpy as np
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.sort(np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]))
    return {"diff_pts": round(100 * float(d.mean()), 2),
            "ci95_pts": [round(100 * float(means[int(0.025 * n_boot)]), 2),
                         round(100 * float(means[int(0.975 * n_boot) - 1]), 2)]}


def analyse(records_by_run: Dict[str, List[dict]], pairs: List[Tuple[str, str]], seg) -> Dict[str, object]:
    first = next(iter(records_by_run.values()))
    golds = [r["answers"] for r in first]
    for name, recs in records_by_run.items():
        if [r["question_id"] for r in recs] != [r["question_id"] for r in first]:
            raise ValueError(f"{name}: thứ tự câu khác run đầu")
    # gold: mỗi bậc thay đổi bao nhiêu đáp án (so với bậc trước)
    gold_flat = [g for gs in golds for g in gs]
    lad = [ladder(g) for g in gold_flat]
    step_changes = []
    for k, (sid, desc, _) in enumerate(STEPS):
        if k == 0:
            n_ch = sum(1 for g, L in zip(gold_flat, lad) if L[0] != g)
        else:
            n_ch = sum(1 for L in lad if L[k] != L[k - 1])
        step_changes.append({"step": sid, "what": desc, "n_gold_changed": n_ch,
                             "pct_gold_changed": round(100 * n_ch / max(len(gold_flat), 1), 2)})
    n_not_nfc = sum(1 for g in gold_flat if unicodedata.normalize("NFC", g) != g)
    # compound words in gold
    norm_g = [normalize_vi(g) for g in gold_flat]
    syl = [len(t.split()) for t in norm_g]
    words = [seg(t) for t in norm_g]
    n_compound = sum(1 for w in words if any("_" in x for x in w))
    out = {"n_questions": len(first), "n_gold_answers": len(gold_flat),
           "gold_not_nfc": n_not_nfc, "gold_step_changes": step_changes,
           "gold_mean_syllables": round(sum(syl) / max(len(syl), 1), 2),
           "gold_mean_words": round(sum(len(w) for w in words) / max(len(words), 1), 2),
           "pct_gold_with_compound_word": round(100 * n_compound / max(len(words), 1), 2),
           "gold_top_compounds": Counter(x for w in words for x in w if "_" in x).most_common(15),
           "runs": {}, "pairs": []}
    per_run_f1 = {}
    for name, recs in records_by_run.items():
        preds = [r["prediction"] for r in recs]
        ems = [[em_at_step(p, g, k) for p, g in zip(preds, golds)] for k in range(len(STEPS))]
        em_ref = [exact_match(p, g) for p, g in zip(preds, golds)]
        assert ems[-1] == em_ref, f"{name}: S6 phải trùng metrics.exact_match"
        f1_syl = [token_f1(p, g) for p, g in zip(preds, golds)]
        f1_word = [word_f1(p, g, seg) for p, g in zip(preds, golds)]
        em_word = [1.0 if " ".join(seg(normalize_vi(p))) in {" ".join(seg(normalize_vi(x))) for x in g} else 0.0
                   for p, g in zip(preds, golds)]
        per_run_f1[name] = (f1_syl, f1_word)
        out["runs"][name] = {
            "em_by_step_pct": {STEPS[k][0]: round(100 * sum(e) / len(e), 2) for k, e in enumerate(ems)},
            "em_pct": round(100 * sum(em_ref) / len(em_ref), 2),
            "em_after_word_seg_pct": round(100 * sum(em_word) / len(em_word), 2),
            "token_f1_syllable_pct": round(100 * sum(f1_syl) / len(f1_syl), 2),
            "token_f1_word_pct": round(100 * sum(f1_word) / len(f1_word), 2),
        }
    for a, b in pairs:
        out["pairs"].append({"a": a, "b": b,
                             "token_f1_syllable": paired_boot(per_run_f1[a][0], per_run_f1[b][0]),
                             "token_f1_word": paired_boot(per_run_f1[a][1], per_run_f1[b][1])})
    rk_s = sorted(out["runs"], key=lambda n: out["runs"][n]["token_f1_syllable_pct"])
    rk_w = sorted(out["runs"], key=lambda n: out["runs"][n]["token_f1_word_pct"])
    out["ranking_syllable"], out["ranking_word"], out["ranking_unchanged"] = rk_s, rk_w, rk_s == rk_w
    return out


def render_md(summary: Dict[str, object]) -> str:
    L = [f"# Chuẩn hoá + tách từ khi chấm điểm ({VERSION})", ""]
    for ds, s in summary["datasets"].items():
        L += [f"## {ds} — {s['n_questions']:,} câu, {s['n_gold_answers']:,} đáp án gold", "",
              "| Bậc | Kỹ thuật | % gold bị đổi | " + " | ".join(f"EM {n}" for n in s["runs"]) + " |",
              "|---|---|---|" + "---|" * len(s["runs"])]
        for k, st in enumerate(s["gold_step_changes"]):
            L.append(f"| {st['step']} | {st['what']} | {st['pct_gold_changed']} | " +
                     " | ".join(f"{r['em_by_step_pct'][st['step']]}" for r in s["runs"].values()) + " |")
        L += ["", f"Gold chưa ở dạng NFC: {s['gold_not_nfc']} · TB {s['gold_mean_syllables']} âm tiết / "
                  f"{s['gold_mean_words']} từ mỗi đáp án · {s['pct_gold_with_compound_word']}% đáp án có từ ghép (pyvi)", "",
              "| Run | EM | EM sau tách từ | token-F1 âm tiết | token-F1 từ (pyvi) |", "|---|---|---|---|---|"]
        for n, r in s["runs"].items():
            L.append(f"| {n} | {r['em_pct']} | {r['em_after_word_seg_pct']} | {r['token_f1_syllable_pct']} | {r['token_f1_word_pct']} |")
        L += ["", "| Cặp | ΔF1 âm tiết [CI95] | ΔF1 từ [CI95] |", "|---|---|---|"]
        for p in s["pairs"]:
            a, b = p["token_f1_syllable"], p["token_f1_word"]
            L.append(f"| {p['a']} → {p['b']} | {a['diff_pts']:+} [{a['ci95_pts'][0]:+}; {a['ci95_pts'][1]:+}] | "
                     f"{b['diff_pts']:+} [{b['ci95_pts'][0]:+}; {b['ci95_pts'][1]:+}] |")
        L += ["", f"Thứ hạng theo F1 âm tiết: {' < '.join(s['ranking_syllable'])} · theo F1 từ: "
                  f"{' < '.join(s['ranking_word'])} → {'KHÔNG đổi' if s['ranking_unchanged'] else 'ĐỔI'}", ""]
    return "\n".join(L)


def run(repo_root: str = REPO_ROOT, out_rel: str = OUT_REL) -> Dict[str, object]:
    import pyvi
    seg = make_word_seg()
    res = {}
    for ds, runs in RUNS.items():
        recs = {}
        for name, rd in runs.items():
            with open(os.path.join(repo_root, "experiments", rd, "predictions.json"), encoding="utf-8") as fp:
                recs[name] = json.load(fp)
        res[ds] = analyse(recs, PAIRS[ds], seg)
        res[ds]["run_dirs"] = runs
        print(f"== {ds}: " + " | ".join(f"{n} EM {r['em_pct']} F1s {r['token_f1_syllable_pct']} F1w {r['token_f1_word_pct']}"
                                         for n, r in res[ds]["runs"].items()), flush=True)
    summary = {"version": VERSION, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "word_segmenter": f"pyvi {getattr(pyvi, '__version__', '0.1.1')} ViTokenizer (CRF)",
               "bootstrap": {"n": N_BOOT, "seed": SEED, "method": "paired percentile, mean of per-question diff"},
               "datasets": res}
    out = os.path.join(repo_root, out_rel)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8", newline="\n") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as fp:
        fp.write(render_md(summary))
    return summary


def _selftest() -> None:
    L = ladder("  Màu  ĐỎ! ")
    assert L[0] == "Màu  ĐỎ!" and L[2] == "màu  đỏ!" and L[3] == "màu  đỏ " and L[4] == "màu đỏ" and L[5] == "đỏ"
    assert ladder("Mười Một")[-1] == "11" and ladder("năm 2021")[-1] == "năm 2021"
    for t in ["Bảo tàng Lịch sử Quốc gia.", "màu đỏ", "Hai", "cool-air", "é"]:   # 'é' dạng tách rời
        assert ladder(t)[-1] == normalize_vi(t), t
    assert em_at_step("Đỏ", ["đỏ"], 1) == 0.0 and em_at_step("Đỏ", ["đỏ"], 2) == 1.0
    seg = lambda t: {"cửa hàng điện thoại": ["cửa_hàng", "điện_thoại"], "điện thoại": ["điện_thoại"]}.get(t, t.split())  # noqa: E731
    assert abs(token_f1("điện thoại", ["cửa hàng điện thoại"]) - 2 * 1 * 0.5 / 1.5) < 1e-9          # âm tiết: P=1, R=2/4
    assert abs(word_f1("điện thoại", ["cửa hàng điện thoại"], seg) - 2 * 1 * 0.5 / 1.5) < 1e-9     # từ: P=1, R=1/2
    pb = paired_boot([0, 0, 1, 1], [1, 1, 1, 1], n_boot=200)
    assert pb["diff_pts"] == 50.0 and pb["ci95_pts"][0] <= 50.0 <= pb["ci95_pts"][1]
    recs = {"A": [{"question_id": "1", "prediction": "Đỏ", "answers": ["đỏ"]},
                  {"question_id": "2", "prediction": "cửa hàng", "answers": ["cửa hàng điện thoại"]}],
            "B": [{"question_id": "1", "prediction": "đỏ", "answers": ["đỏ"]},
                  {"question_id": "2", "prediction": "cửa hàng điện thoại", "answers": ["cửa hàng điện thoại"]}]}
    s = analyse(recs, [("A", "B")], seg)
    assert s["runs"]["A"]["em_by_step_pct"]["S1"] == 0.0 and s["runs"]["A"]["em_pct"] == 50.0
    assert s["runs"]["B"]["em_pct"] == 100.0 and s["ranking_unchanged"]
    assert "| S6 |" in render_md({"datasets": {"x": s}})
    print("metric_tokenization SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    if a.run:
        run()
    if not (a.selftest or a.run):
        ap.print_help()


if __name__ == "__main__":
    main()
