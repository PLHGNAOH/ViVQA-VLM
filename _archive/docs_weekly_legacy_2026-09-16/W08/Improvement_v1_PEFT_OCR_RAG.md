# Improvement v1 — PEFT / OCR / RAG

## 1. Thành phần v1

| Module | Bật? | Chi tiết triển khai | Path |
|--------|------|---------------------|------|
| LoRA / QLoRA | | rank `_` · 4/8-bit `_` | `src/adaptation/` · `adapters/` |
| OCR prompt | | engine `_` · template `_` | `src/data/` · `data/ocr/` |
| RAG | | k `_` · index `_` | `src/adaptation/` |
| Prompt-opt | | | |

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.


## 2. So sánh fair vs baseline W05

Cùng split `_` · seed `_` · decode `_` · eval script `_`.

| Run ID | EM | VQA-Acc | ANLS | Train time | Infer time |
|--------|----|---------|------|------------|------------|
| W05 baseline | | | | | |
| W08 v1 | | | | | |

## 3. Log & config

- Experiment folder: `experiments/W08_<ten>/`
- Config: `configs/`
- Seed: `_`
- Hardware: `_`

## 4. Sự cố & xử lý

OOM / loss nan / OCR empty / retrieval miss: _…_

## 5. Việc để W09

_…_
