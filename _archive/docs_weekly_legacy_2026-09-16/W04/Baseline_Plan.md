# Baseline Plan — reproduce đúng 1 VLM

## 1. Quyết định (chọn đúng 1)

- [ ] BLIP-2
- [ ] Florence-2
- [ ] Qwen2.5-VL

**Model id / revision:** `_`  
**Lý do (VRAM, code, giấy phép, tiếng Việt):** `_`

## 2. Protocol reproduce

| Hạng mục | Quyết định |
|----------|------------|
| Mode | Zero-shot / official checkpoint (ưu tiên trước PEFT) |
| Dataset | ViVQA / ViTextVQA / cả hai |
| Split + seed | `_` |
| Image size / processor | `_` |
| Decoding | greedy / beam `_` · max_new_tokens `_` |
| Batch size | `_` |
| Quantization inference | none / 8-bit / 4-bit |
| Eval metrics | EM, VQA-Acc, ANLS |
| Subset nếu GPU yếu | N=`_` · lý do `_` |

## 3. Việc **không** làm ở baseline

- Không LoRA trừ khi paper gốc bắt buộc (mặc định: zero-shot/official)
- **Không full fine-tuning**
- Không đổi test split để khớp paper

## 4. Tiêu chí "reproduce xong" (W05)

- [ ] Lệnh chạy ghi trong `Baseline_Reproduce.md`
- [ ] Experiment log 1 run
- [ ] Ba metric chính (hoặc ghi N/A + lý do nếu subset)

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.

