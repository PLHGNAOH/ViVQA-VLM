# CONVENTIONS — ViVQA-VLM

> Chốt W03 (2026-09-17). Bổ sung khi nhóm thống nhất thêm quy tắc.
> Không thay ràng buộc Master Instruction §4 / §7.

## 1. Run naming

Mọi thư mục run dưới `experiments/` và mọi log/adapter gắn với một thí nghiệm dùng:

```text
W0X_<method>_<model>_seed42
```

| Thành phần | Quy tắc | Ví dụ |
|------------|---------|--------|
| `W0X` | Tuần *bắt đầu* hoặc tuần *công bố kết quả chính* của run (2 chữ số) | `W08` |
| `<method>` | snake_case, không dấu: `zeroshot`, `prompt`, `qlora`, `lora`, `ocr`, `rag`, `qlora_ocr` | `qlora` |
| `<model>` | token ngắn, không version dài: `qwen3b`, `qwen7b`, `florence2`, `blip2` | `qwen3b` |
| `seed42` | seed mặc định **42**. Seed khác: `seed7`, `seed123` (không bỏ tiền tố `seed`) | `seed42` |

**Ví dụ đầy đủ**

| Run | Thư mục |
|-----|---------|
| QLoRA Qwen2.5-VL-3B, seed 42, tuần 8 | `experiments/W08_qlora_qwen3b_seed42/` |
| Zero-shot cùng model, tuần 5 | `experiments/W05_zeroshot_qwen3b_seed42/` |
| Format-prompt (không train) | `experiments/W05_prompt_qwen3b_seed42/` |
| OCR-enhanced + QLoRA | `experiments/W08_qlora_ocr_qwen3b_seed42/` |
| Backup Florence-2 zero-shot | `experiments/W05_zeroshot_florence2_seed42/` |

Adapter weights (nếu có): `adapters/W0X_<method>_<model>_seed42/` — cùng stem, không tự đặt tên khác.

Config (nếu tạo **mới**): ưu tiên `configs/W0X_<method>_<model>_seed42.yaml`. **Không** đổi tên `configs/*.json` hiện có (ràng buộc W03).

## 2. Seed

- Mặc định **42** (đã dùng cho dataset stats W02).
- Kết quả chính (H1): 3 seed — ghi rõ từng run (`seed42`, cộng 2 seed phụ khi chạy).
- Mọi JSON/log phải chứa trường `"seed"`.

## 3. Không làm

- Không full fine-tuning.
- Không đặt tên run kiểu `exp1`, `final_v2`, `test` — không truy được tuần/method/model/seed.
