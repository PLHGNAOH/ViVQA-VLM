"""
run_eval.py — Vòng ĐÁNH GIÁ dùng chung cho baseline & mọi ablation.

Một hàm `run_evaluation(...)` duy nhất phục vụ tất cả các dòng trong bảng ablation
(zero-shot, +OCR, +RAG, LoRA-adapted): chỉ khác nhau ở `mode` (prompt) và ở chỗ
model có adapter hay không. Nhờ vậy MỌI run dùng CÙNG code eval -> so sánh công bằng.

Đầu ra: file JSON experiment-log chứa metric + metadata (split, seed, hardware,
thời gian inference) — đúng "Evaluation protocol" mà PDF bắt buộc.

Thiết kế để chạy trên Colab/Kaggle (GPU). Hàm sinh câu trả lời (`generate_answer`)
viết riêng cho Qwen2.5-VL; đổi backbone thì thay hàm này.
"""
from __future__ import annotations
import os
import json
import time
import platform
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.prompting.builder import build_prompt, to_chat_messages
from src.eval.metrics import evaluate_predictions


# =============================================================================
# 1) SINH CÂU TRẢ LỜI (đặc thù Qwen2.5-VL)
# =============================================================================
def generate_answer(model, processor, image_path: str, prompt_text: str,
                    max_new_tokens: int = 32) -> str:
    """
    Chạy 1 lượt inference Qwen2.5-VL cho (ảnh, prompt) và trả câu trả lời (string).

    Các bước theo đúng quy trình chuẩn của Qwen2.5-VL:
        1. Bọc thành 'messages' (system + user[image,text]).
        2. apply_chat_template -> chuỗi prompt có token đặc biệt.
        3. process_vision_info -> tách phần ảnh cho processor.
        4. model.generate -> cắt bỏ phần token đầu vào -> decode phần sinh mới.
    """
    from qwen_vl_utils import process_vision_info  # tiện ích chính thức của Qwen-VL
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = to_chat_messages(image, prompt_text)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Cắt bỏ phần prompt, chỉ giữ token model sinh ra
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    answer = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return answer.strip()


# =============================================================================
# 2) VÒNG ĐÁNH GIÁ CHÍNH
# =============================================================================
def run_evaluation(
    model,
    processor,
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    ocr_lookup: Optional[Dict[str, List[str]]] = None,
    generate_fn=generate_answer,
    verbose_every: int = 50,
) -> Dict[str, Any]:
    """
    Duyệt toàn bộ tập test, sinh câu trả lời, tính metric, đóng gói kết quả + log.

    Args:
        model, processor: từ src.models.vlm_loader.
        samples: list sample chuẩn từ src.data.vivqa_dataset.load_vivqa().
        cfg: dict config (đọc từ configs/*.yaml).
        ocr_lookup: (tùy chọn) {question_id hoặc image: [ocr_texts]} cho chế độ OCR.
                    Bước 3 sẽ cung cấp; zero-shot không cần.
        generate_fn: hàm sinh câu trả lời (mặc định Qwen2.5-VL). Cho phép thay bằng
                     hàm giả khi test offline không GPU.
        verbose_every: in tiến độ mỗi N mẫu.

    Returns:
        dict {"metrics": {...}, "meta": {...}, "predictions": [...]}.
    """
    mode = cfg["prompting"]["mode"]
    max_new_tokens = cfg["prompting"].get("max_new_tokens", 32)
    anls_threshold = cfg["eval"].get("anls_threshold", 0.5)
    metric_names = cfg["eval"].get("primary_metrics",
                                   ["exact_match", "vqa_accuracy", "anls"])

    preds: List[str] = []
    refs: List[List[str]] = []
    qtypes: List[str] = []
    records: List[Dict[str, Any]] = []   # lưu chi tiết từng mẫu -> phục vụ error analysis

    t0 = time.time()
    for i, s in enumerate(samples):
        # Lấy OCR cho mẫu này (nếu chế độ cần và có cache)
        ocr_texts = None
        if ocr_lookup is not None:
            ocr_texts = ocr_lookup.get(s["question_id"]) or ocr_lookup.get(
                os.path.basename(s["image_path"]))

        prompt_text = build_prompt(s["question"], mode=mode, ocr_texts=ocr_texts)
        pred = generate_fn(model, processor, s["image_path"], prompt_text,
                           max_new_tokens=max_new_tokens)

        preds.append(pred)
        refs.append(s["answers"])
        qtypes.append(s["question_type"])
        records.append({
            "question_id": s["question_id"],
            "question": s["question"],
            "prediction": pred,
            "answers": s["answers"],
            "question_type": s["question_type"],
        })

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  [{i + 1}/{len(samples)}] q='{s['question'][:40]}' -> '{pred[:40]}'")

    elapsed = time.time() - t0

    # Tính metric (tổng thể + tách theo question type)
    metrics = evaluate_predictions(
        preds, refs, metric_names=metric_names,
        anls_threshold=anls_threshold, question_types=qtypes,
    )

    # Metadata cho reproducibility (PDF bắt buộc: split, seed, hardware, time)
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        gpu = "unknown"

    meta = {
        "run_name": cfg["run"]["name"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": cfg["data"]["dataset_name"],
        "split": os.path.basename(cfg["data"].get("test_path", "")),
        "prompting_mode": mode,
        "model_id": cfg["model"]["model_id"],
        "quantization_4bit": cfg["quantization"]["enabled"],
        "seed": cfg["run"]["seed"],
        "num_samples": len(samples),
        "hardware": {"gpu": gpu, "platform": platform.platform()},
        "inference_time_sec": round(elapsed, 2),
        "sec_per_sample": round(elapsed / max(len(samples), 1), 3),
    }

    return {"metrics": metrics, "meta": meta, "predictions": records}


# =============================================================================
# 3) LƯU KẾT QUẢ
# =============================================================================
def save_results(result: Dict[str, Any], output_dir: str) -> str:
    """
    Ghi kết quả ra 2 file trong output_dir/<run_name>/:
        - metrics.json     : metric + meta (gọn, để tổng hợp bảng).
        - predictions.json : chi tiết từng mẫu (để error/hallucination analysis).
    Trả về đường dẫn thư mục.
    """
    run_dir = os.path.join(output_dir, result["meta"]["run_name"])
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": result["metrics"], "meta": result["meta"]},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(result["predictions"], f, ensure_ascii=False, indent=2)

    return run_dir


# --- Smoke test OFFLINE: dùng generate_fn giả, không cần GPU/model ---
if __name__ == "__main__":
    fake_cfg = {
        "run": {"name": "smoke_test", "seed": 42},
        "model": {"model_id": "fake", },
        "quantization": {"enabled": True},
        "data": {"dataset_name": "vivqa", "test_path": "x/test.json"},
        "prompting": {"mode": "zero_shot", "max_new_tokens": 32},
        "eval": {"primary_metrics": ["exact_match", "vqa_accuracy", "anls"],
                 "anls_threshold": 0.5},
    }
    fake_samples = [
        {"question_id": "1", "question": "Con vật gì?", "answers": ["con mèo", "mèo"],
         "image_path": "n/a", "question_type": "what"},
        {"question_id": "2", "question": "Mấy người?", "answers": ["hai", "2"],
         "image_path": "n/a", "question_type": "counting"},
    ]
    # Hàm sinh giả: cứ trả đáp án đúng đầu tiên -> kiểm tra pipeline chạy trơn tru
    def fake_gen(model, processor, image_path, prompt_text, max_new_tokens=32):
        return "con mèo" if "vật" in prompt_text else "ba"

    res = run_evaluation(None, None, fake_samples, fake_cfg, generate_fn=fake_gen,
                         verbose_every=0)
    print("METRICS:", json.dumps(res["metrics"], ensure_ascii=False, indent=2))
    print("META   :", json.dumps(res["meta"], ensure_ascii=False, indent=2))
