# Improvement Final (frozen method)

## 1. Method cuối (freeze)

Mô tả 1–2 đoạn pipeline suy diễn và (nếu có) PEFT train.

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.


## 2. Hyperparameters cuối

| Key | Value |
|-----|-------|
| Model backbone | |
| LoRA rank / alpha / targets | |
| Quant | 4-bit / 8-bit / none |
| LR / steps | |
| Prompt template | |
| OCR | |
| RAG k | |
| Seed | |
| Split | |

## 3. Khác v1

_…_

## 4. Checkpoint

| | Path (local, không commit file lớn) |
|--|--------------------------------------|
| Adapter | `adapters/` |
| Config | `configs/` |
| Tokenizer notes | |

## 5. Không được làm sau freeze

- [ ] Đổi test split
- [ ] Full FT
- [ ] Tune trên test
