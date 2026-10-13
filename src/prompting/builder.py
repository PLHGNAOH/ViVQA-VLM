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
VALID_MODES = {"zero_shot", "prompt_optimized", "ocr", "rag", "ocr_rag", "few_shot"}
OCR_MODES = {"ocr", "ocr_rag"}

# W08 OCR-enhanced prompting: mẫu dòng chèn văn bản OCR (chọn trên dev, ghi vào run_signature).
#   v1 : trung tính.
#   v2 : báo trước OCR có thể sai -> model không chép mù khi OCR đọc nhầm dấu/chữ.
OCR_TEMPLATES = {
    "v1": "Văn bản đọc được trong ảnh: {ocr}",
    "v2": "Văn bản OCR trích từ ảnh (có thể sai chính tả hoặc thiếu dấu): {ocr}",
}
OCR_JOINER = " | "

# W07 few-shot control: tiêu đề khối ví dụ (CHỈ chữ, không kèm ảnh của ví dụ) — mục đích duy nhất là
# cho model thấy ĐỊNH DẠNG đáp án của dataset (ngắn, đúng chữ), để tách "học định dạng" khỏi QLoRA.
FEW_SHOT_HEADER = "Một số ví dụ về cách viết đáp án (không kèm ảnh của các ví dụ):"

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
    examples: Optional[List[dict]] = None,
    ocr_template: str = "v1",
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
        examples: (chỉ mode few_shot) list {"question", "answer"} lấy từ TRAIN — ví dụ định dạng đáp án.
        ocr_template: khoá trong OCR_TEMPLATES (chỉ dùng ở mode ocr / ocr_rag).

    Returns:
        Chuỗi prompt hoàn chỉnh (chưa gắn ảnh).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode='{mode}' không hợp lệ. Chọn trong {VALID_MODES}.")

    parts: List[str] = []

    # 0) few_shot: khối ví dụ (chỉ chữ) + câu hỏi theo đúng khuôn của ví dụ.
    if mode == "few_shot":
        if not examples:
            raise ValueError("mode='few_shot' cần examples (list {question, answer} từ train)")
        parts.append(FEW_SHOT_HEADER)
        for e in examples:
            parts.append(f"Câu hỏi: {e['question']}\nTrả lời: {e['answer']}")
        parts.append(f"Câu hỏi: {question}\nTrả lời:")
        return "\n".join(parts)

    # 1) Chèn bằng chứng OCR (chế độ ocr / ocr_rag) — đặt TRƯỚC câu hỏi để model
    #    coi đây là ngữ cảnh khi đọc chữ trong ảnh. Đây chính là "OCR-enhanced prompting".
    #    Ảnh không có chữ (OCR rỗng) -> KHÔNG chèn gì -> prompt trùng hệt zero-shot.
    if mode in OCR_MODES and ocr_texts:
        if ocr_template not in OCR_TEMPLATES:
            raise ValueError(f"ocr_template='{ocr_template}' không hợp lệ. Chọn trong {sorted(OCR_TEMPLATES)}.")
        joined = OCR_JOINER.join(t.strip() for t in ocr_texts if t and t.strip())
        if joined:
            parts.append(OCR_TEMPLATES[ocr_template].format(ocr=joined))

    # 2) Chèn ngữ cảnh truy hồi (chế độ rag / ocr_rag)
    if mode in ("rag", "ocr_rag") and retrieved_context:
        parts.append(f"Ngữ cảnh tham khảo: {retrieved_context.strip()}")

    # 3) Câu hỏi + chỉ dẫn định dạng.
    #    W08: mode "ocr" dùng ĐÚNG dòng câu hỏi của zero_shot -> ablation "có / không OCR" chỉ khác
    #    đúng MỘT dòng OCR. (Bản trước W08 thêm cả chỉ dẫn "Trả lời ngắn gọn..." -> lẫn 2 biến.)
    if mode in ("zero_shot", "ocr"):
        parts.append(f"Câu hỏi: {question}")
    else:
        # prompt_optimized và các chế độ có augmentation đều thêm ràng buộc đầu ra.
        parts.append(f"Câu hỏi: {question}\nTrả lời ngắn gọn bằng tiếng Việt:")

    return "\n".join(parts)


def prompt_for_sample(sample: dict, prompting: dict) -> str:
    """
    Prompt của MỘT mẫu theo mục `prompting` của config — hàm DUY NHẤT dùng cho cả train
    (train_qlora collate) lẫn eval (run_eval) -> prompt lúc train trùng hệt lúc eval.

    Mode OCR: mẫu phải có khoá "ocr_texts" (gắn bởi src.data.ocr.attach_ocr, có thể là list rỗng).
    Thiếu khoá -> báo lỗi, KHÔNG âm thầm rơi về prompt zero-shot.
    """
    mode = prompting["mode"]
    if mode in OCR_MODES and "ocr_texts" not in sample:
        raise ValueError(f"mode='{mode}' nhưng mẫu {sample.get('question_id')} chưa có 'ocr_texts' "
                         f"(chưa gắn OCR cache?)")
    return build_prompt(sample["question"], mode=mode, ocr_texts=sample.get("ocr_texts"),
                        examples=prompting.get("few_shot_examples"),
                        ocr_template=(prompting.get("ocr") or {}).get("template", "v1"))


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


def _selftest() -> None:
    q = "Trên tấm biển ghi chữ gì?"
    zs = build_prompt(q)
    assert zs == "Câu hỏi: Trên tấm biển ghi chữ gì?", zs                     # B1/B2 KHÔNG đổi
    o = build_prompt(q, mode="ocr", ocr_texts=[" CẤM ĐỖ XE ", "", "24/24"])
    assert o == "Văn bản đọc được trong ảnh: CẤM ĐỖ XE | 24/24\n" + zs, o     # khác zero-shot đúng 1 dòng
    assert build_prompt(q, mode="ocr", ocr_texts=[]) == zs                    # ảnh không chữ -> = zero-shot
    assert build_prompt(q, mode="ocr", ocr_texts=None) == zs
    assert build_prompt(q, mode="ocr", ocr_texts=["A"], ocr_template="v2").startswith("Văn bản OCR trích từ ảnh")
    try:
        build_prompt(q, mode="ocr", ocr_texts=["A"], ocr_template="v9")
        raise AssertionError("không chặn template lạ")
    except ValueError:
        pass
    # prompt_optimized giữ nguyên hành vi cũ
    assert build_prompt(q, mode="prompt_optimized").endswith("\nTrả lời ngắn gọn bằng tiếng Việt:")
    # prompt_for_sample: train và eval dùng chung; mode OCR thiếu ocr_texts -> lỗi
    s = {"question_id": "t1", "question": q}
    assert prompt_for_sample(s, {"mode": "zero_shot"}) == zs
    assert prompt_for_sample({**s, "ocr_texts": ["A"]}, {"mode": "ocr", "ocr": {"template": "v1"}}) == \
        "Văn bản đọc được trong ảnh: A\n" + zs
    try:
        prompt_for_sample(s, {"mode": "ocr"})
        raise AssertionError("không chặn mẫu thiếu ocr_texts")
    except ValueError:
        pass
    print("builder SELFTEST OK")


# --- Smoke test: `python -m src.prompting.builder` ---
if __name__ == "__main__":
    _selftest()
    q = "Trên tấm biển ghi chữ gì?"
    for mode in ["zero_shot", "prompt_optimized", "ocr", "ocr_rag"]:
        print(f"\n===== mode = {mode} =====")
        print(build_prompt(q, mode=mode,
                           ocr_texts=["CẤM ĐỖ XE", "24/24"],
                           retrieved_context="Biển P.131 cấm đỗ xe."))
