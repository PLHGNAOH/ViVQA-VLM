"""
demo_examples.py — W14 bước 4: chọn ví dụ cho giao diện / video demo bằng LUẬT CỐ ĐỊNH (không chọn tay).

Vấn đề: ví dụ demo chọn tay dễ thành "chỉ khoe câu đúng". Ở đây ví dụ được rút TỰ ĐỘNG từ kết quả chạy thật
của prototype (records.jsonl của run R-LIVE v2, thứ tự mẫu seed 42), theo từng ô sau, lấy câu ĐẦU TIÊN thoả:

  ô                      điều kiện
  chu_trong_anh (2 câu)  ViTextVQA · có dòng OCR · trả lời đúng (EM) · mức "Cao" · ảnh không xoay
  anh_xoay_exif          ViTextVQA · ảnh có cờ xoay EXIF (giao diện tự xoay) · có dòng OCR
  co_khong               ViVQA · câu có/không (yesno-detector-v1)
  dem                    ViVQA · câu có "bao nhiêu"
  mau_sac                ViVQA · câu có "màu"
  tra_loi_sai            ViTextVQA · trả lời SAI (EM = 0) · mức "Thấp"  -> cho thấy giới hạn + cảnh báo
  anh_that_*             mọi dòng của cau_hoi.csv ảnh thật (nếu có)

Ô nào không có câu thoả thì bỏ trống (ghi rõ trong bảng). Kết quả: examples.json (cho --examples_json của giao diện)
+ demo_examples.md (bảng: câu hỏi, đáp án, trả lời, mức — dùng khi quay video).

Chạy:   python -m src.prototype.demo_examples --records <run>/records.jsonl --image_roots vivqa=<dir> vitextvqa_official=<dir>
            [--real_csv <csv> --real_dir <dir>] --out <dir>
Kiểm:   python -m src.prototype.demo_examples --selftest
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

SLOTS: List[Tuple[str, int, str]] = [
    ("chu_trong_anh", 2, "Đọc chữ trong ảnh (ViTextVQA): đúng, mức Cao"),
    ("anh_xoay_exif", 1, "Ảnh điện thoại có cờ xoay EXIF: giao diện tự xoay rồi mới OCR"),
    ("co_khong", 1, "Câu có/không (đi đường riêng, ngưỡng τ*)"),
    ("dem", 1, "Câu đếm (bao nhiêu)"),
    ("mau_sac", 1, "Câu về màu sắc"),
    ("tra_loi_sai", 1, "Trả lời sai, mức Thấp: giới hạn của hệ thống"),
]


def _em(r: Dict[str, Any]) -> Optional[float]:
    from src.eval.metrics import exact_match
    return exact_match(r["prediction"], r["answers"]) if r.get("answers") else None


def _conds() -> Dict[str, Callable[[Dict[str, Any]], bool]]:
    from src.prototype.core import is_yes_no_question
    return {
        "chu_trong_anh": lambda r: (r["dataset"] == "vitextvqa_official" and r.get("n_ocr_lines", 0) > 0
                                    and _em(r) == 1.0 and r.get("level") == "Cao" and not r.get("rotated")),
        "anh_xoay_exif": lambda r: (r["dataset"] == "vitextvqa_official" and bool(r.get("rotated"))
                                    and r.get("n_ocr_lines", 0) > 0),
        "co_khong": lambda r: r["dataset"] == "vivqa" and is_yes_no_question(r["question"]),
        "dem": lambda r: r["dataset"] == "vivqa" and "bao nhiêu" in r["question"].lower(),
        "mau_sac": lambda r: r["dataset"] == "vivqa" and "màu" in r["question"].lower(),
        "tra_loi_sai": lambda r: (r["dataset"] == "vitextvqa_official" and _em(r) == 0.0 and r.get("level") == "Thấp"),
    }


def pick(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Theo thứ tự records (thứ tự mẫu seed 42), mỗi ô lấy câu đầu thoả, không lặp ảnh giữa các ô."""
    conds = _conds()
    used_imgs, out = set(), []
    for slot, k, label in SLOTS:
        got = []
        for r in records:
            key = (r["dataset"], r["image"])
            if key in used_imgs or not conds[slot](r):
                continue
            got.append(r)
            used_imgs.add(key)
            if len(got) == k:
                break
        for r in got:
            out.append({"slot": slot, "label": label, "dataset": r["dataset"], "image": r["image"],
                        "question": r["question"], "answers": r.get("answers"), "prediction": r["prediction"],
                        "level": r.get("level"), "mode": r.get("mode"), "rotated": bool(r.get("rotated"))})
        if len(got) < k:
            out.append({"slot": slot, "label": label, "missing": k - len(got)})
    return out


def real_examples(real_csv: Optional[str], real_dir: Optional[str]) -> List[Dict[str, Any]]:
    if not real_csv or not os.path.exists(real_csv):
        return []
    from src.eval.prototype_live import read_questions_csv
    return [{"slot": f"anh_that_{i:02d}", "label": "Ảnh thật của nhóm", "dataset": "real",
             "image": q["image"], "image_path": os.path.join(real_dir or "", q["image"]),
             "question": q["question"], "answers": q.get("answers")}
            for i, q in enumerate(read_questions_csv(real_csv), 1)]


def build(records_path: str, image_roots: Dict[str, str], out_dir: str,
          real_csv: Optional[str] = None, real_dir: Optional[str] = None) -> Dict[str, Any]:
    recs = [json.loads(l) for l in open(records_path, encoding="utf-8") if l.strip()]
    chosen = pick(recs) + real_examples(real_csv, real_dir)
    ui = []
    for c in chosen:
        if c.get("missing"):
            continue
        p = c.get("image_path") or os.path.join(image_roots[c["dataset"]], c["image"])
        if os.path.exists(p):
            ui.append({"image": p, "question": c["question"]})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "examples.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(ui, f, ensure_ascii=False, indent=1)
    L = ["# W14 bước 4 — ví dụ demo (chọn bằng luật cố định, `src/prototype/demo_examples.py`)", "",
         "| # | Ô | Tập | Ảnh | Câu hỏi | Đáp án | Trả lời (chạy thật) | Mức |", "|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(chosen, 1):
        if c.get("missing"):
            L.append(f"| {i} | {c['slot']} | — | — | *(không có câu thoả, thiếu {c['missing']})* | | | |")
            continue
        L.append(f"| {i} | {c['slot']} | {c['dataset']} | {c['image']}{' (xoay EXIF)' if c.get('rotated') else ''} | "
                 f"{c['question']} | {' / '.join(c.get('answers') or []) or '—'} | {c.get('prediction', '(trả lời khi demo)')} | "
                 f"{c.get('level') or '—'} |")
    md = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "demo_examples.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    with open(os.path.join(out_dir, "demo_examples.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(chosen, f, ensure_ascii=False, indent=1)
    print(md)
    return {"n_ui": len(ui), "chosen": chosen}


def _selftest() -> None:
    import tempfile
    def rec(ds, im, q, pred, ans, lvl="Cao", n_ocr=2, rot=False):
        return {"dataset": ds, "image": im, "question": q, "prediction": pred, "answers": ans, "level": lvl,
                "n_ocr_lines": n_ocr, "rotated": rot, "mode": "open"}
    recs = [rec("vitextvqa_official", "a.jpg", "tên quán?", "phở", ["phở"]),
            rec("vitextvqa_official", "a.jpg", "số nhà?", "12", ["12"]),            # cùng ảnh -> không lấy lần 2
            rec("vitextvqa_official", "b.jpg", "biển ghi gì?", "cấm đỗ", ["cấm đỗ"], lvl="Trung bình"),
            rec("vitextvqa_official", "c.jpg", "giá?", "10k", ["10k"]),
            rec("vitextvqa_official", "d.jpg", "tên?", "x", ["y"], rot=True),
            rec("vivqa", "e.jpg", "trong ảnh có con mèo không?", "không", ["không"], n_ocr=0),
            rec("vivqa", "f.jpg", "có bao nhiêu con chó?", "hai", ["hai"], n_ocr=0),
            rec("vitextvqa_official", "g.jpg", "hãng nào?", "abc", ["xyz"], lvl="Thấp")]
    got = pick(recs)
    slots = [g["slot"] for g in got]
    assert slots == ["chu_trong_anh", "chu_trong_anh", "anh_xoay_exif", "co_khong", "dem", "mau_sac", "tra_loi_sai"], slots
    assert [g.get("image") for g in got[:2]] == ["a.jpg", "c.jpg"]                  # b.jpg mức Trung bình -> bỏ
    assert got[5].get("missing") == 1 and got[6]["image"] == "g.jpg" and got[2]["rotated"]
    assert pick(list(recs)) == got                                                   # tất định
    with tempfile.TemporaryDirectory() as td:
        from PIL import Image
        for ds in ("vivqa", "vitextvqa_official"):
            os.makedirs(os.path.join(td, ds))
        for r in recs:
            Image.new("RGB", (8, 8)).save(os.path.join(td, r["dataset"], r["image"]))
        rp = os.path.join(td, "records.jsonl")
        with open(rp, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n")
        real = os.path.join(td, "real")
        os.makedirs(real)
        Image.new("RGB", (8, 8)).save(os.path.join(real, "p1.jpg"))
        csvp = os.path.join(td, "cau_hoi.csv")
        with open(csvp, "w", encoding="utf-8") as f:
            f.write("anh,cau_hoi,dap_an\np1.jpg,quán tên gì?,phở\nkhong_co.jpg,x?,\n")
        res = build(rp, {"vivqa": os.path.join(td, "vivqa"), "vitextvqa_official": os.path.join(td, "vitextvqa_official")},
                    os.path.join(td, "out"), csvp, real)
        ui = json.load(open(os.path.join(td, "out", "examples.json"), encoding="utf-8"))
        assert res["n_ui"] == len(ui) == 7 and ui[-1]["image"].endswith("p1.jpg")    # 6 câu dataset + 1 ảnh thật có file
        md = open(os.path.join(td, "out", "demo_examples.md"), encoding="utf-8").read()
        assert "không có câu thoả" in md and "(xoay EXIF)" in md
    print("✅ selftest demo_examples OK (luật cố định, không lặp ảnh, ô trống ghi rõ, ảnh thật, file cho giao diện)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--records")
    ap.add_argument("--image_roots", nargs="+", default=[], help="tập=thư_mục_ảnh, vd vivqa=/content/vivqa_images")
    ap.add_argument("--real_csv")
    ap.add_argument("--real_dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.records or not a.out:
        ap.error("cần --records và --out")
    roots = dict(x.split("=", 1) for x in a.image_roots)
    build(a.records, roots, a.out, a.real_csv, a.real_dir)


if __name__ == "__main__":
    main()
