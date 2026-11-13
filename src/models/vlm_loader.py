"""
vlm_loader.py — Nạp baseline VLM và gắn PEFT (LoRA/QLoRA).

Đây là module THỂ HIỆN RÀNG BUỘC CỨNG của đề tài: "PEFT-only, no full fine-tuning".
Toàn bộ trọng số gốc của VLM bị ĐÓNG BĂNG; chỉ có LoRA adapter (~0.3–1% tham số)
được huấn luyện. Hàm `load_vlm_for_training` in ra số tham số trainable để bạn
DÁN THẲNG vào báo cáo làm bằng chứng tuân thủ.

Hai điểm vào chính:
    - load_vlm_for_inference(cfg): nạp model gốc (zero-shot / eval baseline).
    - load_vlm_for_training(cfg):  nạp model + gắn LoRA (giai đoạn adaptation).

Ghi chú về đa backbone:
    PDF cho chọn 1 trong BLIP-2 / Florence-2 / Qwen2.5-VL. Code này mặc định
    Qwen2.5-VL (khuyến nghị: mạnh scene-text tiếng Việt, có bản 3B nhẹ). Muốn đổi
    backbone, chỉ cần thêm nhánh trong `_load_base_model` và chỉnh `lora.target_modules`
    trong config cho khớp tên lớp attention của kiến trúc đó.
"""
from __future__ import annotations
from typing import Dict, Tuple, Any
import torch


# =============================================================================
# Tiện ích: đổi chuỗi dtype trong config -> torch.dtype
# =============================================================================
def _resolve_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.float16)


# =============================================================================
# 1) Cấu hình lượng tử hóa 4-bit (QLoRA)
# =============================================================================
def _build_quant_config(cfg: Dict[str, Any]):
    """
    Dựng BitsAndBytesConfig từ mục `quantization` của config.
    Trả về None nếu tắt lượng tử hóa (khi đó LoRA chạy trên fp16 -> cần VRAM nhiều hơn).
    """
    q = cfg.get("quantization", {})
    if not q.get("enabled", False):
        return None

    from transformers import BitsAndBytesConfig  # import trễ để lỗi bitsandbytes không chặn cả file
    return BitsAndBytesConfig(
        load_in_4bit=q.get("load_in_4bit", True),
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=_resolve_dtype(q.get("bnb_4bit_compute_dtype", "bfloat16")),
    )


# =============================================================================
# 2) Nạp model gốc + processor theo backbone
# =============================================================================
def _load_base_model(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Nạp checkpoint pretrained + processor tương ứng.
    'processor' là bộ tiền xử lý ĐA PHƯƠNG THỨC của VLM (ảnh + text) — thay thế
    hoàn toàn cho việc tự-xây-vocab + tokenize thủ công của repo LSTM cũ.
    """
    m = cfg["model"]
    backbone = m.get("backbone", "qwen2_5_vl")
    quant_config = _build_quant_config(cfg)
    dtype = _resolve_dtype(m.get("torch_dtype", "bfloat16"))

    if backbone == "qwen2_5_vl":
        # Cần transformers >= 4.49. Nếu ImportError -> nâng cấp transformers.
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            m["model_id"],
            quantization_config=quant_config,   # None nếu không QLoRA
            torch_dtype=dtype,
            device_map="auto",                   # tự trải model lên GPU khả dụng
        )
        # min/max_pixels: khống chế độ phân giải ảnh -> kiểm soát VRAM & tốc độ
        processor = AutoProcessor.from_pretrained(
            m["model_id"],
            min_pixels=m.get("min_pixels"),
            max_pixels=m.get("max_pixels"),
        )
        return model, processor

    # ----- Chỗ mở rộng cho backbone khác (team SV1 có thể thêm) -----
    # if backbone == "blip2":
    #     from transformers import Blip2ForConditionalGeneration, Blip2Processor
    #     ... target_modules LoRA cho BLIP-2 thường là ["q", "v"] của lớp attention ...
    # if backbone == "florence2":
    #     ... Florence-2 cần trust_remote_code=True ...
    raise NotImplementedError(
        f"Backbone '{backbone}' chưa được hỗ trợ. Hãy thêm nhánh trong _load_base_model()."
    )


# =============================================================================
# 3) API công khai
# =============================================================================
def load_vlm_for_inference(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Nạp VLM ở chế độ SUY LUẬN (không LoRA).
    Dùng cho: reproduce baseline zero-shot, và eval mọi cấu hình prompting.
    """
    model, processor = _load_base_model(cfg)
    model.eval()
    return model, processor


def load_vlm_for_training(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Nạp VLM + GẮN LoRA adapter cho giai đoạn adaptation (PEFT).

    Quy trình:
        1. Nạp model gốc (đã lượng tử hóa 4-bit nếu bật QLoRA).
        2. prepare_model_for_kbit_training: bật gradient checkpointing, ổn định
           layernorm ở fp32... — bước chuẩn bị bắt buộc trước khi LoRA lên model 4-bit.
        3. get_peft_model: chèn LoRA vào các lớp attention chỉ định.
        4. In số tham số trainable (bằng chứng KHÔNG full fine-tuning).
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model, processor = _load_base_model(cfg)

    if cfg.get("quantization", {}).get("enabled", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg["training"].get("gradient_checkpointing", True),
        )

    lcfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        bias=lcfg.get("bias", "none"),
        task_type="CAUSAL_LM",                          # VLM sinh text -> causal LM
        target_modules=lcfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)

    # === DÒNG QUAN TRỌNG cho báo cáo ===
    # Kỳ vọng in ra: trainable params ~ vài triệu / tổng ~ vài tỉ  => <1%.
    model.print_trainable_parameters()

    return model, processor


# --- Kiểm tra cấu hình mà KHÔNG tải model (đỡ tốn mạng/VRAM) ---
if __name__ == "__main__":
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/qwen_lora.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("Backbone     :", cfg["model"]["backbone"])
    print("Model ID     :", cfg["model"]["model_id"])
    print("QLoRA 4-bit  :", cfg["quantization"]["enabled"])
    print("LoRA target  :", cfg["lora"]["target_modules"])
    print("=> Config hợp lệ. Chạy load_vlm_for_training(cfg) trên Colab để nạp thật.")
