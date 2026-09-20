"""
vivqa_dataset.py — Nạp & CHUẨN HÓA dữ liệu ViVQA / ViTextVQA về một schema thống nhất.

VÌ SAO CẦN MODULE NÀY:
    ViVQA (Tran et al., PACLIC 2021) và ViTextVQA có định dạng file KHÁC NHAU
    (tên trường, cách trỏ ảnh...). Thay vì để code eval/train phụ thuộc từng định
    dạng, ta chuẩn hóa hết về 1 "sample chuẩn":

        {
          "question_id": str,
          "question":    str,
          "answers":     List[str],   # LUÔN là list (metric cần list đáp án đúng)
          "image_path":  str,         # đường dẫn ảnh ĐẦY ĐỦ, đã sẵn sàng mở
          "question_type": str        # phục vụ error analysis; "unknown" nếu thiếu
        }

    Nhờ vậy, đổi dataset chỉ cần đổi 'field_map' trong config — không sửa code eval.

LƯU Ý QUAN TRỌNG:
    Đây KHÔNG phải dataset ViVQA-X của repo cũ. Đây là ViVQA/ViTextVQA mà PDF yêu cầu.
    Bước 3 sẽ mở rộng loader này để đính kèm OCR cache cho ViTextVQA.
"""
from __future__ import annotations
import os
import json
from typing import List, Dict, Optional, Any


# Ánh xạ tên trường mặc định. Ghi đè qua config nếu file dataset dùng tên khác.
DEFAULT_FIELD_MAP = {
    "question_id": "question_id",
    "question": "question",
    "answer": "answer",        # trường đáp án đơn (string)
    "answers": "answers",      # trường đáp án nhiều (list) — ưu tiên nếu có
    "image": "image",          # tên/đường dẫn ảnh
    "question_type": "question_type",
}


def _extract_answers(item: Dict[str, Any], fmap: Dict[str, str]) -> List[str]:
    """
    Rút danh sách đáp án đúng từ 1 bản ghi, chấp nhận nhiều định dạng:
        - 'answers': ["a", "b"]                    (list string)
        - 'answers': [{"answer": "a"}, ...]        (list dict kiểu VQA v2)
        - 'answer':  "a"                            (string đơn)
    Trả về list string (đã bỏ rỗng). Không chuẩn hóa chữ ở đây — để metric lo.
    """
    out: List[str] = []
    raw_answers = item.get(fmap["answers"])
    if isinstance(raw_answers, list):
        for a in raw_answers:
            if isinstance(a, dict):
                a = a.get("answer", "")
            if isinstance(a, str) and a.strip():
                out.append(a.strip())
    # Bổ sung/dự phòng: trường đáp án đơn
    single = item.get(fmap["answer"])
    if isinstance(single, str) and single.strip():
        out.append(single.strip())
    return out or [""]   # luôn trả ít nhất 1 phần tử để metric không lỗi chia 0


def _resolve_image_path(item: Dict[str, Any], fmap: Dict[str, str],
                        image_dir: str, image_pattern: Optional[str]) -> str:
    """
    Dựng đường dẫn ảnh đầy đủ.
        - Nếu 'image' đã là tên file (có đuôi ảnh) -> ghép với image_dir.
        - Nếu chỉ là image_id (số) và có image_pattern -> format theo mẫu.
          Ví dụ ViVQA dùng ảnh COCO: image_pattern = "COCO_val2014_{:012d}.jpg".
    """
    val = item.get(fmap["image"], "")
    val_str = str(val)
    has_ext = val_str.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    if not has_ext and image_pattern:
        try:
            val_str = image_pattern.format(int(val))
        except (ValueError, TypeError):
            val_str = image_pattern.format(val)
    return os.path.join(image_dir, val_str)


def load_vivqa(
    data_path: str,
    image_dir: str,
    field_map: Optional[Dict[str, str]] = None,
    image_pattern: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Nạp file JSON dataset và trả list 'sample chuẩn'.

    Args:
        data_path: đường dẫn file JSON (list các bản ghi, hoặc {"data": [...]},
                   hoặc {id: record}).
        image_dir: thư mục chứa ảnh.
        field_map: ghi đè DEFAULT_FIELD_MAP nếu tên trường khác.
        image_pattern: mẫu tên ảnh khi dataset chỉ cho image_id (vd COCO).
        max_samples: giới hạn số mẫu (để smoke-test nhanh); None = tất cả.

    Returns:
        List[dict] theo schema chuẩn (xem docstring đầu file).
    """
    fmap = {**DEFAULT_FIELD_MAP, **(field_map or {})}

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Chuẩn hóa về list bản ghi bất kể định dạng gốc
    if isinstance(raw, dict):
        records = raw["data"] if isinstance(raw.get("data"), list) else list(raw.values())
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("Cấu trúc JSON không hỗ trợ (cần list hoặc dict).")

    samples: List[Dict[str, Any]] = []
    for idx, item in enumerate(records):
        if max_samples is not None and idx >= max_samples:
            break
        samples.append({
            "question_id": str(item.get(fmap["question_id"], idx)),
            "question": str(item.get(fmap["question"], "")).strip(),
            "answers": _extract_answers(item, fmap),
            "image_path": _resolve_image_path(item, fmap, image_dir, image_pattern),
            "question_type": str(item.get(fmap["question_type"], "unknown")),
        })
    return samples


# --- Smoke test với dữ liệu giả (không cần dataset thật) ---
if __name__ == "__main__":
    import tempfile
    demo = [
        {"question_id": "1", "question": "Con vật trong ảnh là gì?",
         "answer": "con mèo", "image": "000000001.jpg", "question_type": "what"},
        {"question_id": "2", "question": "Có mấy người?",
         "answers": ["2", "hai"], "image": 262284, "question_type": "counting"},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(demo, tf, ensure_ascii=False)
        tmp = tf.name
    out = load_vivqa(tmp, image_dir="data/vivqa/images",
                     image_pattern="COCO_val2014_{:012d}.jpg")
    for s in out:
        print(s)
