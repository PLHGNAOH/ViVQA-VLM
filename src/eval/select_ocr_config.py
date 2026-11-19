"""
select_ocr_config.py — Áp dụng ĐÚNG luật chọn cấu hình OCR đã đặt trước (docs/W08/README.md, 13/10/2026).

Luật (chỉ dùng dev300, không nhìn test):
  1. Backend: so EM của B1 + OCR (template v1) giữa 2 backend bằng McNemar exact.
       p < 0,05  -> chọn backend EM cao hơn.
       p >= 0,05 -> chọn backend có answer_in_ocr_pct cao hơn (ocr_coverage.json); vẫn hoà -> backend nhanh hơn.
  2. Template trên backend đã chọn: v2 CHỈ khi EM(v2) > EM(v1) và p < 0,05; còn lại giữ v1.
     (Tiêu chí phụ answer_in_ocr và tốc độ giống hệt nhau giữa 2 template của cùng backend -> hoà = v1.)
  3. max_chars: chỉ cảnh báo nếu answer_in_ocr_pct − answer_in_ocr_capped_pct > 5 điểm (cắt làm mất đáp án).
Ngoài luật (chỉ báo cáo): mỗi run OCR so với B1 zero-shot (OCR có giúp B1 trên dev không).

Chạy:
  python -m src.eval.select_ocr_config --b1 <run B1 dev300> \
      --run paddleocr:v1=<dir> --run paddledet_vietocr:v1=<dir> --run paddleocr:v2=<dir> --run paddledet_vietocr:v2=<dir> \
      --coverage <ocr_coverage.json> --out <dir>/ocr_selection.json
Kiểm thử offline:
  python -m src.eval.select_ocr_config --selftest
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

from src.eval.compare_runs import load_predictions, mcnemar_exact_p, per_sample_scores

ALPHA = 0.05
MAX_CHARS_LOSS_POINTS = 5.0


def paired(dir_a: str, dir_b: str) -> Dict[str, Any]:
    """So ghép cặp EM (McNemar exact) + ANLS trung bình của 2 run trên cùng câu hỏi."""
    ra, rb = load_predictions(dir_a), load_predictions(dir_b)
    if [r["question_id"] for r in ra] != [r["question_id"] for r in rb]:
        raise ValueError(f"{dir_a} và {dir_b} không cùng danh sách câu hỏi")
    sa, sb = per_sample_scores(ra), per_sample_scores(rb)
    n = len(sa)
    only_a = sum(1 for x, y in zip(sa, sb) if x["exact_match"] == 1.0 and y["exact_match"] != 1.0)
    only_b = sum(1 for x, y in zip(sa, sb) if y["exact_match"] == 1.0 and x["exact_match"] != 1.0)
    return {"n": n, "em_a": sum(x["exact_match"] for x in sa) / n, "em_b": sum(y["exact_match"] for y in sb) / n,
            "anls_a": sum(x["anls"] for x in sa) / n, "anls_b": sum(y["anls"] for y in sb) / n,
            "only_a": only_a, "only_b": only_b, "p": mcnemar_exact_p(only_a, only_b)}


def _sec_per_sample(run_dir: str) -> float:
    with open(os.path.join(run_dir, "metrics.json"), encoding="utf-8") as fp:
        return json.load(fp)["meta"]["sec_per_sample"]


def select(b1: str, runs: Dict[Tuple[str, str], str], coverage: Dict[str, Any]) -> Dict[str, Any]:
    cov = {b["backend"]: b for b in coverage["backends"]}
    backends = sorted({k[0] for k in runs})
    if len(backends) != 2 or any((b, "v1") not in runs for b in backends):
        raise ValueError("cần đúng 2 backend, mỗi backend có run template v1")
    a, b = backends
    p1 = paired(runs[(a, "v1")], runs[(b, "v1")])
    log: List[str] = []
    if p1["p"] < ALPHA and p1["em_a"] != p1["em_b"]:
        backend = a if p1["em_a"] > p1["em_b"] else b
        log.append(f"Luật 1: EM v1 {a}={p1['em_a']:.4f} vs {b}={p1['em_b']:.4f}, McNemar p={p1['p']:.3g} < 0,05 -> {backend}")
    else:
        ca, cb = cov[a]["answer_in_ocr_pct"], cov[b]["answer_in_ocr_pct"]
        log.append(f"Luật 1: EM không phân biệt được (p={p1['p']:.3g}) -> xét answer_in_ocr: {a}={ca} vs {b}={cb}")
        if ca != cb:
            backend = a if ca > cb else b
        else:
            sa, sb = cov[a]["sec_per_image_cache"], cov[b]["sec_per_image_cache"]
            backend = a if sa <= sb else b
            log.append(f"         hoà answer_in_ocr -> nhanh hơn: {a}={sa} s/ảnh vs {b}={sb} s/ảnh")
        log.append(f"         -> {backend}")
    template, p2 = "v1", None
    if (backend, "v2") in runs:
        p2 = paired(runs[(backend, "v1")], runs[(backend, "v2")])
        if p2["p"] < ALPHA and p2["em_b"] > p2["em_a"]:
            template = "v2"
        log.append(f"Luật 2: {backend} v1={p2['em_a']:.4f} vs v2={p2['em_b']:.4f}, p={p2['p']:.3g} -> {template}")
    else:
        log.append("Luật 2: không có run v2 -> giữ v1")
    c = cov[backend]
    loss = round(c["answer_in_ocr_pct"] - c["answer_in_ocr_capped_pct"], 2)
    log.append(f"Luật 3: answer_in_ocr mất do cắt 600 ký tự = {loss} điểm "
               + ("> 5 -> CẦN xem lại max_chars" if loss > MAX_CHARS_LOSS_POINTS else "<= 5 -> giữ max_chars 600, min_conf 0"))
    vs_b1 = {f"{k[0]}:{k[1]}": {**paired(b1, d), "sec_per_sample": _sec_per_sample(d)} for k, d in sorted(runs.items())}
    return {"rule_source": "docs/W08/README.md — Luật chọn cấu hình OCR (13/10/2026)", "split": "dev300",
            "decision": {"backend": backend, "template": template, "max_chars": 600, "min_conf": 0.0,
                         "max_chars_needs_review": loss > MAX_CHARS_LOSS_POINTS},
            "log": log, "backend_test_v1": {"a": a, "b": b, **p1}, "template_test": p2,
            "b1_sec_per_sample": _sec_per_sample(b1),
            "vs_b1_zero_shot_info_only": vs_b1}


def _selftest() -> None:
    import tempfile

    def mk(td, name, preds, sec=1.0):
        d = os.path.join(td, name)
        os.makedirs(d)
        recs = [{"question_id": f"q{i}", "prediction": p, "answers": ["đúng"], "question_type": "unknown"}
                for i, p in enumerate(preds)]
        json.dump(recs, open(os.path.join(d, "predictions.json"), "w", encoding="utf-8"), ensure_ascii=False)
        json.dump({"meta": {"sec_per_sample": sec}}, open(os.path.join(d, "metrics.json"), "w"))
        return d

    cov = {"backends": [{"backend": "A", "answer_in_ocr_pct": 27.0, "answer_in_ocr_capped_pct": 27.0, "sec_per_image_cache": 0.3},
                        {"backend": "B", "answer_in_ocr_pct": 37.7, "answer_in_ocr_capped_pct": 30.0, "sec_per_image_cache": 0.9}]}
    with tempfile.TemporaryDirectory() as td:
        n = 40
        b1 = mk(td, "b1", ["sai"] * n)
        # (1) B tốt hơn rõ rệt (12 câu chỉ B đúng) -> chọn B theo EM; v2 tốt hơn không có ý nghĩa -> giữ v1
        runs = {("A", "v1"): mk(td, "a1", ["sai"] * n), ("B", "v1"): mk(td, "b1v1", ["đúng"] * 12 + ["sai"] * 28),
                ("A", "v2"): mk(td, "a2", ["sai"] * n), ("B", "v2"): mk(td, "b2", ["đúng"] * 14 + ["sai"] * 26)}
        r = select(b1, runs, cov)
        assert r["decision"]["backend"] == "B" and r["decision"]["template"] == "v1", r["log"]
        assert r["decision"]["max_chars_needs_review"] is True             # B mất 7,7 điểm do cắt
        # (2) EM hoà (p lớn) -> answer_in_ocr quyết: B
        runs2 = {("A", "v1"): mk(td, "a1x", ["đúng"] * 5 + ["sai"] * 35), ("B", "v1"): mk(td, "b1x", ["đúng"] * 6 + ["sai"] * 34)}
        r2 = select(b1, runs2, cov)
        assert r2["decision"]["backend"] == "B" and r2["template_test"] is None and "answer_in_ocr" in r2["log"][0]
        # (3) A tốt hơn rõ rệt dù answer_in_ocr thấp hơn -> EM thắng; v2 tốt hơn rõ rệt -> v2
        runs3 = {("A", "v1"): mk(td, "a1y", ["đúng"] * 12 + ["sai"] * 28), ("B", "v1"): mk(td, "b1y", ["sai"] * n),
                 ("A", "v2"): mk(td, "a2y", ["đúng"] * 30 + ["sai"] * 10)}
        r3 = select(b1, runs3, cov)
        assert r3["decision"] == {"backend": "A", "template": "v2", "max_chars": 600, "min_conf": 0.0,
                                  "max_chars_needs_review": False}, r3
        assert r3["vs_b1_zero_shot_info_only"]["A:v2"]["only_b"] == 30
        try:
            select(b1, {("A", "v2"): runs3[("A", "v2")], ("B", "v1"): runs3[("B", "v1")]}, cov)
            raise AssertionError("không chặn thiếu run v1")
        except ValueError:
            pass
    print("select_ocr_config SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chọn backend + template OCR theo luật đặt trước (dev300)")
    ap.add_argument("--b1", help="Thư mục run B1 zero-shot trên dev300")
    ap.add_argument("--run", action="append", default=[], help="backend:template=thư_mục_run (lặp lại)")
    ap.add_argument("--coverage", help="ocr_coverage.json của dev300")
    ap.add_argument("--out", help="File JSON kết quả")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not (a.b1 and a.run and a.coverage and a.out):
        ap.error("cần --b1, --run, --coverage, --out")
    runs = {}
    for item in a.run:
        key, d = item.split("=", 1)
        backend, tpl = key.split(":")
        runs[(backend, tpl)] = d
    with open(a.coverage, encoding="utf-8") as fp:
        cov = json.load(fp)
    res = select(a.b1, runs, cov)
    for line in res["log"]:
        print(line)
    print("\nOCR có giúp B1 trên dev300 không? (chỉ báo cáo, không thuộc luật chọn)")
    print(f"  B1 zero-shot: {res['b1_sec_per_sample']} s/câu")
    for k, v in res["vs_b1_zero_shot_info_only"].items():
        print(f"  {k:22s} EM {v['em_a']:.4f} -> {v['em_b']:.4f} | ANLS {v['anls_a']:.4f} -> {v['anls_b']:.4f} | "
              f"chỉ B1 đúng {v['only_a']}, chỉ OCR đúng {v['only_b']}, p={v['p']:.3g} | {v['sec_per_sample']} s/câu")
    d = res["decision"]
    print(f"\n➜ CHỐT: backend={d['backend']} | template={d['template']} | max_chars={d['max_chars']} | min_conf={d['min_conf']}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print("->", a.out)


if __name__ == "__main__":
    main()
