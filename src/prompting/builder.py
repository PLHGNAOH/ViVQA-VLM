"""
builder.py — Xây dựng prompt đa phương thức cho VLM.

VÌ SAO MODULE NÀY QUAN TRỌNG VỚI CAPSTONE:
    PDF bắt buộc ablation study gồm: zero-shot vs prompt-optimized vs LoRA;
    with/without OCR; with/without RAG. Ý tưởng thiết kế cốt lõi: TẤT CẢ các
    ablation ở tầng input này chỉ khác nhau ở PROMPT — cùng model, cùng seed,
    cùng split. Nhờ đó so sánh "under identical conditions" như PDF đòi hỏi.

    Model không đổi -> chỉ đổi `mode` -> ra một dòng trong bảng ablation.

Luồng dùng:
    text = build_prompt(question, mode="ocr", ocr_texts=[...])
    messages = to_chat_messages(image, text)          # định dạng chat của Qwen2.5-VL
    inputs = processor.apply_chat_template(messages, ...)
"""
from __future__ import annotations
from typing import List, Optional


# Các chế độ prompt hợp lệ (khớp với mục prompting.mode trong config)
VALID_MODES = {"zero_shot", "prompt_optimized", "ocr", "rag", "ocr_rag"}

# Câu chỉ dẫn hệ thống: ép model trả lời NGẮN, bằng tiếng Việt -> hợp metric EM/VQA-Acc.
# (Trả lời lan man sẽ trượt EM dù ý đúng — nên ràng buộc độ dài ngay từ prompt.)
_SYSTEM_VI = (
    "Bạn là trợ lý hỏi đáp hình ảnh tiếng Việt. "
    "Hãy trả lời thật NGẮN GỌN và CHÍNH XÁC bằng tiếng Việt, "
    "chỉ đưa ra đáp án, không giải thích dài dòng."
)


def build_prompt(
    question: str,
    mode: str = "zero_shot",
    ocr_texts: Optional[List[str]] = None,
    retrieved_context: Optional[str] = None,
) -> str:
    """
    Ghép phần TEXT của prompt theo chế độ ablation.

    Args:
        question: câu hỏi tiếng Việt.
        mode: một trong VALID_MODES.
            - zero_shot      : chỉ câu hỏi (baseline thô).
            - prompt_optimized: thêm chỉ dẫn định dạng, vẫn không dùng OCR/RAG.
            - ocr            : chèn văn bản OCR đọc được trong ảnh (cho ViTextVQA).
            - rag            : chèn ngữ cảnh truy hồi (retrieval).
            - ocr_rag        : dùng cả hai.
        ocr_texts: list chuỗi do PaddleOCR/VietOCR trích (Bước 3 sẽ cache sẵn).
        retrieved_context: đoạn văn bản do module RAG cung cấp.

    Returns:
        Chuỗi prompt hoàn chỉnh (chưa gắn ảnh).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode='{mode}' không hợp lệ. Chọn trong {VALID_MODES}.")

    parts: List[str] = []

    # 1) Chèn bằng chứng OCR (chế độ ocr / ocr_rag) — đặt TRƯỚC câu hỏi để model
    #    coi đây là ngữ cảnh khi đọc chữ trong ảnh. Đây chính là "OCR-enhanced prompting".
    if mode in ("ocr", "ocr_rag") and ocr_texts:
        joined = " | ".join(t.strip() for t in ocr_texts if t and t.strip())
        if joined:
            parts.append(f"Văn bản đọc được trong ảnh: {joined}")

    # 2) Chèn ngữ cảnh truy hồi (chế độ rag / ocr_rag)
    if mode in ("rag", "ocr_rag") and retrieved_context:
        parts.append(f"Ngữ cảnh tham khảo: {retrieved_context.strip()}")

    # 3) Câu hỏi + chỉ dẫn định dạng.
    if mode == "zero_shot":
        parts.append(f"Câu hỏi: {question}")
    else:
        # prompt_optimized và các chế độ có augmentation đều thêm ràng buộc đầu ra.
        parts.append(f"Câu hỏi: {question}\nTrả lời ngắn gọn bằng tiếng Việt:")

    return "\n".join(parts)


def to_chat_messages(image, prompt_text: str, system: str = _SYSTEM_VI) -> list:
    """
    Bọc (ảnh, prompt_text) vào định dạng "messages" của Qwen2.5-VL.

    'image' có thể là PIL.Image hoặc đường dẫn/URL — processor của Qwen chấp nhận cả hai.
    Với backbone khác (BLIP-2/Florence-2) cấu trúc messages sẽ khác; tách hàm riêng
    ở đây để chỉ cần sửa 1 chỗ khi đổi model.
    """
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


# --- Smoke test: `python -m src.prompting.builder` ---
if __name__ == "__main__":
    q = "Trên tấm biển ghi chữ gì?"
    for mode in ["zero_shot", "prompt_optimized", "ocr", "ocr_rag"]:
        print(f"\n===== mode = {mode} =====")
        print(build_prompt(q, mode=mode,
                           ocr_texts=["CẤM ĐỖ XE", "24/24"],
                           retrieved_context="Biển P.131 cấm đỗ xe."))
