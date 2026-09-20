# ViVQA-VLM

**Efficient Vietnamese Visual Question Answering using Vision-Language Models** — Đồ án tốt nghiệp (Capstone, ĐH FPT).

Hệ thống nhận **ảnh** + **câu hỏi tiếng Việt** → dùng **Vision-Language Model** (baseline reproduce: BLIP-2 / Florence-2 / Qwen2.5-VL) thích nghi **chỉ bằng PEFT (LoRA / QLoRA 4/8-bit) — không full fine-tuning**, kết hợp OCR (PaddleOCR / VietOCR) và tùy chọn RAG → trả **câu trả lời tiếng Việt**. Benchmark **ViVQA** (general) + **ViTextVQA** (scene-text); đánh giá bằng **EM · VQA Accuracy · ANLS**, kèm **ablation**, **error analysis** và **hallucination analysis**. Mọi run log split, seed, hardware, hyperparameters, thời gian train/inference để đảm bảo reproducibility.

## 🎯 Deliverable cuối cùng

1. **Code demo** — prototype web: upload ảnh, hỏi tiếng Việt, hiển thị câu trả lời + confidence / OCR evidence (`src/prototype/`).
2. **Slides** — bộ trình bày cho Review 1 / Review 2 / Faculty Review / Final Defense (`docs/04_slides/`).
3. **Báo cáo (Overleaf)** — thesis LaTeX viết trên Overleaf, đồng bộ bản `.tex` + hình/bảng về `docs/03_report/` (`docs/03_report/README.md`).

## 📁 Cây thư mục (tóm tắt)

```text
ViVQA-VLM/
├── README.md                 # file này
├── requirements.txt          # pin dependencies
├── docs/
│   ├── 00_admin/             # đăng ký đề tài, form GVHD, timeline
│   ├── 01_proposal/          # capstone proposal (docx/pdf)
│   ├── 02_references/        # PDF paper + Reading_Notes + báo cáo tham chiếu
│   ├── 03_report/            # thesis LaTeX (đồng bộ Overleaf) + R1..R7
│   ├── 04_slides/            # slide theo milestone (review1/review2/faculty_review/final_defense)
│   ├── literature/           # paper reviews (SV1-4) + Literature Matrix
│   ├── notes/ · templates/   # ghi chú & template nhóm
│   ├── W01/ · W02/           # deliverable theo tuần (W02: literature + dataset)
│   ├── master_instruction.md
│   └── MIGRATION_REPORT.md
├── src/
│   ├── data/                 # dataset loaders, OCR (vivqa_dataset.py, ocr.py)
│   ├── models/               # VLM loader
│   ├── prompting/            # prompt builder
│   ├── baseline/             # baseline reproduce
│   ├── adaptation/           # PEFT (LoRA/QLoRA)
│   ├── eval/                 # metrics, run_eval
│   └── prototype/            # web demo
├── notebooks/                # notebook thử nghiệm (zero-shot baseline)
├── configs/                  # cấu hình train/eval (qwen_lora.yaml, ...)
├── data/                     # raw/processed/ocr/splits + ViVQA-X/
├── adapters/                 # LoRA/QLoRA adapter weights
├── experiments/              # run dir Wxx_<tên>
└── _archive/                 # tài liệu/khảo sát cũ, code khóa Agentic-RAG (tham khảo)
```

## ⚙️ Setup

```bash
# Python 3.10+ (khuyến nghị dùng venv riêng)
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Stack: Python · PyTorch · HuggingFace `transformers` / `peft` / `bitsandbytes` · PaddleOCR / VietOCR · Gradio.
Môi trường mục tiêu: **Google Colab Pro/Premium** hoặc **GPU server trường**.

## 👥 Phân công (4 SV)

| SV  | Mảng                       | Trách nhiệm chính |
|-----|----------------------------|-------------------|
| SV1 | VLM / Baseline             | Reproduce baseline (BLIP-2 / Florence-2 / Qwen2.5-VL) |
| SV2 | Dataset / OCR              | ViVQA, ViTextVQA, PaddleOCR / VietOCR, split + seed |
| SV3 | PEFT / Training            | LoRA, QLoRA, prompting, multimodal prompting, RAG |
| SV4 | Evaluation / Deployment    | EM / VQA-Acc / ANLS, ablation, hallucination, prototype |

## 🔒 Ràng buộc cứng (PEFT-only)

- **Chỉ** LoRA / QLoRA (4-bit hoặc 8-bit) khi fine-tune. **Không** full fine-tuning.
- Cải tiến phải so với baseline đã reproduce dưới **cùng split · cùng seed · cùng hardware protocol**.
- Chạy được trên Colab Pro/Premium hoặc GPU trường.

## 📌 Bắt đầu từ đâu

1. Đọc `docs/master_instruction.md`.
2. Tuần 2: `docs/W02/` (index + `WEEK2_STATUS.md`). Tuần 1: `docs/W01/`. Scaffold W03–W15: `_archive/docs_weekly_legacy_2026-09-16/`.
3. Đặt dataset vào `data/`; viết code vào `src/{data,baseline,adaptation,eval,prototype}`.
