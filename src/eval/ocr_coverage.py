"""
ocr_coverage.py — Đo chất lượng OCR cache KHÔNG cần nhãn chữ thật (W08, bước 4).

VẤN ĐỀ: ViTextVQA không có nhãn OCR (vị trí + nội dung chữ thật), nên không tính được CER/WER
của PaddleOCR hay VietOCR. Nhưng điều ta cần biết là: OCR có ĐƯA ĐƯỢC ĐÁP ÁN vào prompt không?

CHỈ SỐ THAY THẾ (proxy), tính trên câu hỏi, so với đáp án gold:
  answer_in_ocr        đáp án gold (chuẩn hoá) xuất hiện NGUYÊN CỤM TỪ trong văn bản OCR của ảnh
                       -> trần trên (upper bound) cho lợi ích của OCR-enhanced prompting.
  answer_in_ocr_capped như trên nhưng sau khi cắt max_chars (đúng phần thật sự vào prompt)
                       -> chênh lệch = phần mất do cắt.
  answer_in_ocr_nodiac như trên nhưng BỎ DẤU cả hai phía
                       -> chênh với answer_in_ocr = phần OCR "đọc đúng chữ, sai dấu thanh".
  token_recall         tỉ lệ từ của đáp án có mặt trong tập từ OCR (mềm hơn, bắt trường hợp đọc gần đúng).
So hai backend trên CÙNG câu hỏi: McNemar exact cho answer_in_ocr (ghép cặp).

LƯU Ý: đây chỉ là chỉ số CHẨN ĐOÁN. Luật chọn backend chính thức dùng EM của B1 + OCR trên dev300
(docs/W08/README.md). Chuẩn hoá ở đây KHÔNG map số / bỏ tiền tố "màu" như normalize_vi của EM,
vì ta tìm đáp án bên trong một đoạn văn bản dài, không so hai chuỗi ngắn.

Chạy:
  python -m src.eval.ocr_coverage --data <dev_subset300_seed42.json> \
      --caches <ocr/paddleocr.json> <ocr/paddledet_vietocr.json> --out <dir>/ocr_coverage.json
Kiểm thử offline:
  python -m src.eval.ocr_coverage --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

from src.data.ocr import OCR_DEFAULTS, lines_to_texts, load_ocr_cache
from src.eval.compare_runs import mcnemar_exact_p


def norm(text: str) -> str:
    """NFC, chữ thường, dấu câu -> khoảng trắng, gộp khoảng trắng. GIỮ dấu thanh."""
    text = unicodedata.normalize("NFC", str(text or "")).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt (kể cả đ -> d)."""
    t = unicodedata.normalize("NFD", text).replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", "".join(ch for ch in t if unicodedata.category(ch) != "Mn"))


def contains_phrase(hay: str, needle: str) -> bool:
    """needle là một CỤM TỪ trọn vẹn trong hay (không khớp nửa từ: 'an' không khớp trong 'bán')."""
    return bool(needle) and f" {needle} " in f" {hay} "


def _answers(r: Dict[str, Any]) -> List[str]:
    a = r.get("answers") or r.get("answer") or []
    return [str(x) for x in (a if isinstance(a, list) else [a]) if str(x).strip()]


def coverage(records: List[Dict[str, Any]], cache: Dict[str, Any], max_chars: Optional[int]) -> Dict[str, Any]:
    imgs = cache["images"]
    per_q: List[int] = []
    capped = nodiac = 0
    recalls: List[float] = []
    missing = sorted({str(r["image"]) for r in records if str(r["image"]) not in imgs})
    if missing:
        raise KeyError(f"{len(missing)} ảnh chưa có trong cache {cache.get('backend')}, vd {missing[:3]}")
    for r in records:
        e = imgs[str(r["image"])]
        full = norm(" ".join(ln["text"] for ln in e.get("lines", [])))
        cap = norm(" ".join(lines_to_texts(e, max_chars)[0]))
        golds = [norm(a) for a in _answers(r)]
        hit = any(contains_phrase(full, g) for g in golds)
        per_q.append(int(hit))
        capped += any(contains_phrase(cap, g) for g in golds)
        nodiac += any(contains_phrase(strip_diacritics(full), strip_diacritics(g)) for g in golds)
        toks = set(full.split())
        recalls.append(max((sum(t in toks for t in g.split()) / len(g.split()) for g in golds if g), default=0.0))
    uniq = sorted({str(r["image"]) for r in records})
    lines = [imgs[k].get("lines", []) for k in uniq]
    confs = [ln["conf"] for ls in lines for ln in ls if ln.get("conf") is not None]
    n = max(len(records), 1)
    t = cache.get("timing") or {}
    return {
        "backend": cache.get("backend"), "engine": cache.get("engine_info"),
        "n_questions": len(records), "n_images": len(uniq),
        "pct_images_with_text": round(100 * sum(bool(ls) for ls in lines) / max(len(uniq), 1), 2),
        "n_ocr_errors": sum(1 for k in uniq if imgs[k].get("error")),
        "mean_lines_per_image": round(sum(len(ls) for ls in lines) / max(len(uniq), 1), 2),
        "mean_chars_per_image": round(sum(len(ln["text"]) for ls in lines for ln in ls) / max(len(uniq), 1), 1),
        "mean_conf": round(sum(confs) / len(confs), 4) if confs else None,
        "answer_in_ocr_pct": round(100 * sum(per_q) / n, 2),
        "answer_in_ocr_capped_pct": round(100 * capped / n, 2),
        "answer_in_ocr_nodiac_pct": round(100 * nodiac / n, 2),
        "token_recall_mean": round(sum(recalls) / n, 4),
        "max_chars": max_chars,
        "sec_per_image_cache": round(t.get("sec", 0.0) / t["images"], 3) if t.get("images") else None,
        "_hits": per_q,
    }


def compare(records: List[Dict[str, Any]], cache_paths: List[str], max_chars: Optional[int]) -> Dict[str, Any]:
    res = [coverage(records, load_ocr_cache(p), max_chars) for p in cache_paths]
    out: Dict[str, Any] = {"what": "Chỉ số thay thế chất lượng OCR (không cần nhãn chữ thật): đáp án gold có nằm trong văn bản OCR không",
                           "rule_note": "CHẨN ĐOÁN, không phải luật chọn backend (luật: EM của B1+OCR trên dev300)",
                           "backends": []}
    for p, r in zip(cache_paths, res):
        out["backends"].append({"cache_file": os.path.basename(p), **{k: v for k, v in r.items() if k != "_hits"}})
    if len(res) == 2:
        a, b = res[0]["_hits"], res[1]["_hits"]
        only_a = sum(x and not y for x, y in zip(a, b))
        only_b = sum(y and not x for x, y in zip(a, b))
        out["paired_answer_in_ocr"] = {"a": res[0]["backend"], "b": res[1]["backend"], "n": len(a),
                                       "only_a": only_a, "only_b": only_b,
                                       "mcnemar_p": mcnemar_exact_p(only_a, only_b),
                                       "mcnemar_test": "exact binomial, two-sided"}
    return out


def _print_table(out: Dict[str, Any]) -> None:
    cols = [("backend", "backend"), ("n_images", "ảnh"), ("pct_images_with_text", "% ảnh có chữ"),
            ("mean_lines_per_image", "dòng/ảnh"), ("mean_conf", "conf TB"), ("answer_in_ocr_pct", "đáp án∈OCR %"),
            ("answer_in_ocr_capped_pct", "…sau cắt %"), ("answer_in_ocr_nodiac_pct", "…bỏ dấu %"),
            ("token_recall_mean", "token recall"), ("sec_per_image_cache", "s/ảnh")]
    print(" | ".join(h for _, h in cols))
    for b in out["backends"]:
        print(" | ".join(str(b.get(k)) for k, _ in cols))
    if "paired_answer_in_ocr" in out:
        p = out["paired_answer_in_ocr"]
        print(f"Ghép cặp: chỉ {p['a']} có đáp án = {p['only_a']} | chỉ {p['b']} = {p['only_b']} | McNemar p = {p['mcnemar_p']:.3g}")


def _selftest() -> None:
    import tempfile
    assert norm(" Cấm  đỗ-xe! ") == "cấm đỗ xe"
    assert strip_diacritics("Đường Lê Lợi") == "Duong Le Loi"
    assert contains_phrase("bán bánh mì", "bánh mì") and not contains_phrase("bán bánh mì", "an")
    from src.data.ocr import CACHE_FORMAT, save_ocr_cache
    recs = [{"question_id": "q1", "image": "1.jpg", "answers": ["Vinmart"]},
            {"question_id": "q2", "image": "2.jpg", "answers": ["hà nội"]},
            {"question_id": "q3", "image": "3.jpg", "answers": ["khai trương"]},
            {"question_id": "q4", "image": "4.jpg", "answers": ["50"]}]

    def cache(backend, d):
        return {"format": CACHE_FORMAT, "backend": backend, "engine_info": {}, "timing": {"images": 4, "sec": 2.0},
                "images": {k: {"lines": [{"text": t, "conf": 0.9} for t in v]} for k, v in d.items()}}
    a = cache("A", {"1.jpg": ["VINMART", "siêu thị"], "2.jpg": ["Ha Noi"], "3.jpg": ["x" * 700, "khai trương"], "4.jpg": []})
    b = cache("B", {"1.jpg": ["vinmart"], "2.jpg": ["Hà Nội"], "3.jpg": ["khai trương"], "4.jpg": ["giảm 50%"]})
    ra = coverage(recs, a, 600)
    assert ra["answer_in_ocr_pct"] == 50.0, ra            # q1 (hoa/thường) + q3; q2 sai dấu; q4 không có chữ
    assert ra["answer_in_ocr_capped_pct"] == 25.0, ra     # q3 bị cắt mất (dòng 700 ký tự đứng trước)
    assert ra["answer_in_ocr_nodiac_pct"] == 75.0, ra     # bỏ dấu -> q2 khớp
    assert ra["pct_images_with_text"] == 75.0 and ra["sec_per_image_cache"] == 0.5
    with tempfile.TemporaryDirectory() as td:
        pa, pb = os.path.join(td, "A.json"), os.path.join(td, "B.json")
        save_ocr_cache(a, pa)
        save_ocr_cache(b, pb)
        out = compare(recs, [pa, pb], 600)
        p = out["paired_answer_in_ocr"]
        assert (p["only_a"], p["only_b"]) == (0, 2) and out["backends"][1]["answer_in_ocr_pct"] == 100.0, out
        assert "_hits" not in out["backends"][0]
        try:
            coverage(recs + [{"question_id": "q5", "image": "9.jpg", "answers": ["a"]}], a, 600)
            raise AssertionError("không chặn ảnh thiếu trong cache")
        except KeyError:
            pass
    print("ocr_coverage SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chỉ số thay thế chất lượng OCR cache (đáp án có trong OCR không)")
    ap.add_argument("--data", nargs="+", help="File dữ liệu (json) — gộp lại")
    ap.add_argument("--caches", nargs="+", help="1 hoặc 2 OCR cache để so")
    ap.add_argument("--max_chars", type=int, default=OCR_DEFAULTS["max_chars"])
    ap.add_argument("--out", default=None, help="Ghi kết quả JSON ra file này")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.data or not a.caches:
        ap.error("cần --data và --caches")
    recs: List[Dict[str, Any]] = []
    for f in a.data:
        with open(f, encoding="utf-8") as fp:
            recs += json.load(fp)
    out = compare(recs, a.caches, a.max_chars)
    out["data_files"] = [os.path.basename(f) for f in a.data]
    _print_table(out)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fp:
            json.dump(out, fp, ensure_ascii=False, indent=2)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
