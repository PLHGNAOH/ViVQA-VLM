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
- [x] `notebooks/01_zeroshot_baseline.ipynb` — notebook Colab chạy thật

**Bước 3 (OCR cache) — xong:**
- [x] `data/ocr.py` — engine PaddleOCR/VietOCR + cache có resume (test offline OK)
- [x] mục `data.ocr` trong `configs/qwen_lora.yaml`

**Còn lại:**
- [ ] `train/train_lora.py` — vòng huấn luyện PEFT (LoRA/QLoRA)
- [ ] `app/gradio_app.py` — web prototype (tuần 14–15)

## Chạy ablation OCR (W08)

OCR chạy ở **phiên Colab riêng**, một lần, ra cache `<drive>/data/<dataset>/ocr/<backend>.json`.
Mọi run (eval lẫn train) chỉ tra cache -> cùng đầu vào OCR cho baseline và cải tiến.

```bash
# 1) Phiên OCR riêng: pip install paddlepaddle(-gpu) paddleocr   (PaddleOCR 3.x)
#    VietOCR (backend paddledet_vietocr): pip install --no-deps vietocr && pip install gdown einops
python -m src.data.ocr --dataset vitextvqa_official --backend paddleocr --device gpu \
    --files dev_subset300_seed42.json --limit_images 100     # đo tốc độ trước, rồi bỏ --limit_images
# 2) Eval có OCR (phiên VLM bình thường, không cần thư viện OCR):
python -m src.eval.run_baseline --dataset vitextvqa_official --run_name <run> --mode ocr --ocr_backend paddleocr
# 3) Train có OCR (B3):
python -m src.adaptation.train_qlora --dataset vitextvqa_official --data_file <tập con> --run_name <run> \
    --epochs 1 --mode ocr --ocr_backend paddleocr
```

Prompt chế độ `ocr` = prompt zero-shot + đúng **một** dòng OCR đặt trước câu hỏi (ảnh không có chữ -> trùng
hệt zero-shot). Train và eval dựng prompt bằng cùng một hàm `src.prompting.builder.prompt_for_sample`.
Thiết lập OCR (backend, model OCR, template, `max_chars`, `min_conf`, sha256 phần cache dùng) nằm trong
`run_signature.json` -> đổi bất kỳ thứ gì cũng là run mới.

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
