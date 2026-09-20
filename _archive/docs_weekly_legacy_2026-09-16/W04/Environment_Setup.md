# Environment Setup

## 1. Runtime

| Mục | Giá trị |
|-----|---------|
| Nền tảng | Colab Pro / Colab Premium / GPU server trường / local |
| OS | _…_ |
| GPU | _name_ |
| VRAM | _…_ GB |
| CUDA / Driver | _…_ |
| Python | _…_ |
| PyTorch | _…_ |
| Transformers / PEFT / bitsandbytes | _…_ |

## 2. Repo layout dùng cho code mới

`src/data` · `src/baseline` · `src/adaptation` · `src/eval` · `src/prototype`

(Code cũ từ đề tài trước nằm trong `src/production-agentic-rag-course/` — **không xóa**; không dùng làm baseline VQA.)

## 3. Checklist cài đặt

- [ ] Torch thấy GPU
- [ ] Load tokenizer/model thử (subset)
- [ ] bitsandbytes 4-bit (nếu QLoRA)
- [ ] PaddleOCR hoặc VietOCR import được
- [ ] Ghi lệnh setup vào đây (copy-paste được)

## 4. Data paths

| Path | Nội dung |
|------|----------|
| `data/raw/` | ViVQA, ViTextVQA gốc |
| `data/processed/` | normalize |
| `data/ocr/` | text OCR |
| `data/splits/` | train/val/test + seed |

## 5. Lệnh chạy thử (điền)

```text
# inference smoke test — KHÔNG full FT
```

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.

