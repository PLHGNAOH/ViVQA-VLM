# `src/` — Kiến trúc code ViVQA-VLM (Bước 1: Skeleton)

> Tài liệu này dành cho **Leader giải thích cho các thành viên**. Nó mô tả *tại sao*
> code được tổ chức như vậy và mỗi module khớp yêu cầu nào trong PDF capstone.

## Nguyên tắc thiết kế cốt lõi

**Tách bạch 3 trục để chạy ablation "cùng điều kiện" như PDF đòi hỏi:**

1. **Model** (`models/vlm_loader.py`) — nạp VLM + gắn LoRA. *Cố định* giữa các ablation.
2. **Input/Prompt** (`prompting/builder.py`) — mọi biến thể OCR/RAG/prompt nằm ở đây.
3. **Đo lường** (`eval/metrics.py`) — EM / VQA-Acc / ANLS, độc lập model.

Nhờ tách vậy, một dòng trong bảng ablation = đổi `mode` trong config, **không sửa code model**.

## Bản đồ module → yêu cầu PDF

| File | Vai trò | Thỏa yêu cầu PDF |
|------|---------|------------------|
| `models/vlm_loader.py` | Nạp Qwen2.5-VL 4-bit + LoRA; in % param trainable | Baseline Requirement + **Training Constraint (PEFT-only)** |
| `eval/metrics.py` | EM, VQA Accuracy, ANLS (+ tách theo question type) | **Primary metrics** + Error analysis |
| `prompting/builder.py` | Prompt zero-shot / prompt-opt / +OCR / +RAG | **Ablation study** (OCR/RAG) |
| `configs/qwen_lora.yaml` | Mọi siêu tham số 1 run | **Reproducibility package** (config + seed + log) |

## Trạng thái

**Bước 1 (skeleton) — xong:**
- [x] `models/vlm_loader.py` — loader QLoRA
- [x] `eval/metrics.py` — 3 metric chính (test offline OK)
- [x] `prompting/builder.py` — prompt builder (test offline OK)
- [x] `configs/qwen_lora.yaml` — config trung tâm

**Bước 2 (zero-shot baseline) — xong:**
- [x] `data/vivqa_dataset.py` — loader ViVQA/ViTextVQA → schema chuẩn (test offline OK)
- [x] `eval/run_eval.py` — runner eval + experiment-log (test offline OK)
- [x] `notebooks/01_zeroshot_baseline_vivqa.ipynb` — notebook Colab chạy thật

**Bước 3 (OCR cache) — xong:**
- [x] `data/ocr.py` — engine PaddleOCR/VietOCR + cache có resume (test offline OK)
- [x] mục `data.ocr` trong `configs/qwen_lora.yaml`

**Còn lại:**
- [ ] `train/train_lora.py` — vòng huấn luyện PEFT (LoRA/QLoRA)
- [ ] `app/gradio_app.py` — web prototype (tuần 14–15)

## Chạy ablation OCR (Bước 3)

```bash
# 1) Cài (trên Colab/Kaggle GPU):  pip install paddlepaddle-gpu paddleocr vietocr
# 2) OCR toàn bộ ảnh -> ghi data/vivqa/ocr_cache.json (chỉ chạy 1 lần)
python -m src.data.ocr --config configs/qwen_lora.yaml --split test
# (test logic offline không cần cài gì:  --backend mock)
```

Trong notebook, để có dòng ablation **+OCR**, chỉ cần:
```python
from src.data.ocr import load_ocr_cache
ocr_lookup = load_ocr_cache(cfg['data']['ocr_cache'])
cfg['prompting']['mode'] = 'ocr'
result = run_evaluation(model, processor, samples, cfg, ocr_lookup=ocr_lookup)
```

## Chạy thử offline (không cần GPU)

```bash
# Kiểm tra logic metric (EM/VQA-Acc/ANLS)
python -m src.eval.metrics

# Xem prompt sinh ra ở từng chế độ ablation
python -m src.prompting.builder

# Kiểm tra config hợp lệ (không tải model)
python -m src.models.vlm_loader configs/qwen_lora.yaml
```

## Phụ thuộc (cài trên Colab/Kaggle ở Bước 2)

```
torch  transformers>=4.49  peft  bitsandbytes  accelerate  pyyaml  qwen-vl-utils
```
> `bitsandbytes` chỉ chạy trên Linux + GPU (Colab/Kaggle) — KHÔNG cài được trên máy CPU/Windows.
> Vì vậy 3 lệnh smoke-test ở trên được thiết kế để chạy được **không cần** các gói nặng này.
