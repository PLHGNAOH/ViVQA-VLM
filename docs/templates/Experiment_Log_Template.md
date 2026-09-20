# Experiment Log Template

Mỗi run: copy thành `experiments/Wxx_<tên>/README.md` (và/hoặc một hàng trong bảng tuần).

## Run card

| Trường | Giá trị |
|--------|---------|
| **Run ID** | `Wxx_<short_name>` |
| **Model / Baseline** | BLIP-2 / Florence-2 / Qwen2.5-VL · hf id `_` |
| **Method** | LoRA / QLoRA 4-bit / QLoRA 8-bit / OCR / RAG / prompt / zero-shot |
| **Dataset** | ViVQA / ViTextVQA / both |
| **Split** | train / val / test · file `data/splits/` |
| **Seed** | |
| **Hardware** | Colab Pro/Premium / GPU server trường · GPU `_` · VRAM `_` |
| **Hyperparameters** | lr `_` · rank `_` · alpha `_` · dropout `_` · bs `_` · max_new_tokens `_` · quant `_` |
| **Exact Match (EM)** | |
| **VQA Accuracy** | |
| **ANLS** | |
| **BLEU / CIDEr (phụ)** | |
| **Train time** | |
| **Infer time** | |
| **Notes** | |

## Fair comparison

So với run: `_` · cùng split/seed/decode? Có / Không (nếu không — **không claim SOTA nội bộ**).

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.


## Artifacts

| Artifact | Path |
|----------|------|
| Config | `configs/` |
| Adapter | `adapters/` |
| Predictions | _local_ |
| Git commit | _nếu có git_ |
