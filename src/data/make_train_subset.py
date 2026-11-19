"""
make_train_subset.py — Tập con TRAIN cố định cho B3 (QLoRA + OCR) và B2-sub (QLoRA, không OCR).

VÌ SAO (quyết định 12/10/2026, docs/W08/README.md):
    1 epoch QLoRA trên toàn bộ ViTextVQA train (35.159 câu) mất ~12 giờ L4 -> không đủ ngân sách
    Colab để train thêm B3. Train CẢ B3 và B2-sub trên CÙNG một tập con -> hai run chỉ khác đúng
    yếu tố OCR, so sánh vẫn công bằng.

VÌ SAO CHỌN THEO ẢNH (không chọn ngẫu nhiên từng câu hỏi):
    - Mọi câu hỏi của một ảnh đi cùng nhau -> không có ảnh "nửa trong nửa ngoài" tập con.
    - Ít ảnh hơn nhiều (~3 câu/ảnh) -> OCR cache cho tập con rẻ hơn ~3 lần.
    Cách làm: sắp tên ảnh, xáo bằng seed, lấy dần từng ảnh (kèm toàn bộ câu hỏi) tới khi
    đủ >= n_questions câu. Thứ tự bản ghi trong file ra giữ như file gốc.

Chạy (máy local, Drive for desktop; hoặc Colab):
    python -m src.data.make_train_subset --drive_root "G:/My Drive/ViVQA-VLM" \
        --dataset vitextvqa_official --n_questions 8000 --seed 42
    -> <drive>/data/vitextvqa_official/train_subset8000_byimage_seed42.json (+ _ids.json)
    -> experiments/W08_data_subsets/train_subset8000_byimage_seed42_manifest.json (vào git)
Kiểm thử offline:
    python -m src.data.make_train_subset --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIRS = {"vivqa": "data/vivqa", "vitextvqa_official": "data/vitextvqa_official"}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: str, obj: Any, indent: Optional[int] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def subset_by_image(records: List[Dict[str, Any]], n_questions: int, seed: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Chọn ảnh ngẫu nhiên (cố định theo seed) tới khi đủ >= n_questions câu. Trả (bản ghi, danh sách ảnh đã sắp)."""
    per_img = Counter(str(r["image"]) for r in records)
    if n_questions >= len(records):
        raise ValueError(f"n_questions={n_questions} >= số câu train ({len(records)}) -> không cần tập con")
    images = sorted(per_img)
    random.Random(seed).shuffle(images)
    chosen, total = set(), 0
    for im in images:
        if total >= n_questions:
            break
        chosen.add(im)
        total += per_img[im]
    out = [r for r in records if str(r["image"]) in chosen]
    return out, sorted(chosen)


def make(drive_root: str, dataset: str, n_questions: int, seed: int,
         manifest_dir: str = os.path.join(REPO_ROOT, "experiments", "W08_data_subsets")) -> Dict[str, Any]:
    src = os.path.join(drive_root, DATASET_DIRS[dataset], "train.json")
    with open(src, encoding="utf-8") as fp:
        records = json.load(fp)
    sub, imgs = subset_by_image(records, n_questions, seed)
    name = f"train_subset{n_questions}_byimage_seed{seed}"
    out = os.path.join(drive_root, DATASET_DIRS[dataset], f"{name}.json")
    ids_path = os.path.join(drive_root, DATASET_DIRS[dataset], f"{name}_ids.json")
    _write_json(out, sub)
    _write_json(ids_path, sorted(str(r["question_id"]) for r in sub), indent=0)
    manifest = {
        "what": "Tập con TRAIN cố định cho B3 (QLoRA + OCR) và B2-sub (QLoRA, không OCR) — chỉ khác yếu tố OCR",
        "why": "1 epoch full ViTextVQA train ≈ 12 h L4; quyết định 12/10/2026 (docs/W08/README.md)",
        "dataset": dataset, "method": "theo ảnh: sắp tên ảnh, xáo bằng seed, lấy dần tới khi >= n_questions câu",
        "seed": seed, "n_questions_target": n_questions,
        "n_questions": len(sub), "n_images": len(imgs),
        "source_file": os.path.relpath(src, drive_root).replace(os.sep, "/"), "source_sha256": _sha256(src),
        "source_n_questions": len(records), "source_n_images": len({str(r["image"]) for r in records}),
        "output_file": os.path.relpath(out, drive_root).replace(os.sep, "/"), "output_sha256": _sha256(out),
        "ids_file": os.path.relpath(ids_path, drive_root).replace(os.sep, "/"),
        "pct_of_train_questions": round(100 * len(sub) / len(records), 2),
    }
    _write_json(os.path.join(manifest_dir, f"{name}_manifest.json"), manifest, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def _selftest() -> None:
    import tempfile
    recs = []
    for i in range(50):                                   # 50 ảnh, 1..4 câu/ảnh
        for j in range(1 + i % 4):
            recs.append({"question_id": f"q{i}_{j}", "image": f"{i}.jpg", "question": "?", "answers": ["a"]})
    a, ia = subset_by_image(recs, 30, 42)
    b, ib = subset_by_image(recs, 30, 42)
    assert a == b and ia == ib, "phải tất định"
    assert ia != subset_by_image(recs, 30, 7)[1], "seed khác -> tập khác"
    assert len(a) >= 30 and len(a) - 30 < 4, len(a)        # vượt ngưỡng không quá số câu của 1 ảnh
    imgs = {r["image"] for r in a}
    assert all(sum(r["image"] == im for r in recs) == sum(r["image"] == im for r in a) for im in imgs), "ảnh bị tách"
    assert [r["question_id"] for r in a] == [r["question_id"] for r in recs if r["image"] in imgs], "đổi thứ tự"
    try:
        subset_by_image(recs, len(recs), 42)
        raise AssertionError("không chặn n >= toàn bộ")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "data/vitextvqa_official")
        os.makedirs(d)
        _write_json(os.path.join(d, "train.json"), recs)
        m = make(td, "vitextvqa_official", 30, 42, manifest_dir=os.path.join(td, "man"))
        assert m["n_questions"] == len(a) and m["n_images"] == len(ia)
        assert os.path.exists(os.path.join(td, "man", "train_subset30_byimage_seed42_manifest.json"))
        m2 = make(td, "vitextvqa_official", 30, 42, manifest_dir=os.path.join(td, "man"))
        assert m2["output_sha256"] == m["output_sha256"], "chạy lại phải ra đúng file cũ"
    print("make_train_subset SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tập con train cố định (chọn theo ảnh) cho B3 / B2-sub")
    ap.add_argument("--drive_root", default="/content/drive/MyDrive/ViVQA-VLM")
    ap.add_argument("--dataset", default="vitextvqa_official", choices=sorted(DATASET_DIRS))
    ap.add_argument("--n_questions", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    make(a.drive_root, a.dataset, a.n_questions, a.seed)


if __name__ == "__main__":
    main()
