"""
prepare_vitextvqa.py — Join annotation COCO-style của ViTextVQA thành list phẳng.

ViTextVQA (mirror HF, file test.json) tách:
    images:      [{id, filename, ...}, ...]
    annotations: [{id, image_id, question, answers, ...}, ...]

Loader / vòng infer cần 1 record / 1 câu hỏi. Module này làm đúng bước join
đã chạy ở W04 (notebook cell B2), để notebook chỉ gọi hàm.

Không tải ảnh / không giải zip ở đây — phần đó giữ trên Colab (nặng, một lần).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def find_image_root(root: str) -> str:
    """Thư mục thật chứa .jpg (zip HF đôi khi lồng thêm 1 lớp folder)."""
    for dirpath, _, filenames in os.walk(root):
        if any(name.lower().endswith(".jpg") for name in filenames):
            return dirpath
    return root


def flatten_coco_style(coco: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Join images ↔ annotations → list phẳng.

    Mỗi phần tử:
        question_id, question, answers, image (tên file), question_type="unknown"
    Bỏ annotation nếu image_id không có trong images.
    """
    images = coco.get("images") or []
    annotations = coco.get("annotations") or []
    id2file = {im["id"]: im["filename"] for im in images}

    flat: List[Dict[str, Any]] = []
    for ann in annotations:
        filename = id2file.get(ann["image_id"])
        if filename is None:
            continue
        flat.append({
            "question_id": ann["id"],
            "question": ann["question"],
            "answers": ann["answers"],
            "image": filename,
            "question_type": "unknown",
        })
    return flat


def flatten_coco_file(
    test_json_path: str,
    out_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Đọc test.json COCO-style, flatten, tùy chọn ghi JSON (một mảng)."""
    with open(test_json_path, encoding="utf-8") as fp:
        coco = json.load(fp)
    if not isinstance(coco, dict):
        raise ValueError(f"Kỳ vọng dict COCO-style, nhận {type(coco).__name__}: {test_json_path}")
    flat = flatten_coco_style(coco)
    if out_path:
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(flat, fp, ensure_ascii=False)
    return flat


if __name__ == "__main__":
    import tempfile

    demo = {
        "images": [{"id": 7, "filename": "foo.jpg"}],
        "annotations": [
            {"id": 29, "image_id": 7, "question": "chữ gì?", "answers": ["cool air"]},
            {"id": 99, "image_id": 999, "question": "mồ côi", "answers": ["x"]},
        ],
    }
    with tempfile.TemporaryDirectory() as td:
        img_nested = os.path.join(td, "images", "inner")
        os.makedirs(img_nested)
        open(os.path.join(img_nested, "foo.jpg"), "wb").close()
        src = os.path.join(td, "test.json")
        dst = os.path.join(td, "test_flat.json")
        with open(src, "w", encoding="utf-8") as fp:
            json.dump(demo, fp, ensure_ascii=False)
        root = find_image_root(os.path.join(td, "images"))
        assert os.path.basename(root) == "inner", root
        flat = flatten_coco_file(src, out_path=dst)
        assert len(flat) == 1, flat
        assert flat[0]["question_id"] == 29
        assert flat[0]["image"] == "foo.jpg"
        loaded = json.loads(open(dst, encoding="utf-8").read())
        assert loaded == flat
        print("OK n=", len(flat), "qid=", flat[0]["question_id"])
