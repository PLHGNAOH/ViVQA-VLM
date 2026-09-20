# Ablation Study

Mỗi claim "cải thiện nhờ X" cần một hàng ablation.

## Setup chung

Split `_` · seed `_` · decode `_` · N `_` · hardware `_`

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.


## Bảng ablation

| ID | Setting | EM | VQA-Acc | ANLS | Δ vs baseline | Run ID |
|----|---------|----|---------|------|---------------|--------|
| A0 | Baseline zero-shot | | | | 0 | |
| A1 | + prompt-opt | | | | | |
| A2 | + OCR prompt | | | | | |
| A3 | + RAG | | | | | |
| A4 | + LoRA/QLoRA | | | | | |
| A5 | Best combination (frozen W09) | | | | | |

Chỉ điền nhánh **đã chạy**. Không bịa số.

## Kết luận ablation

Thành phần đóng góp lớn nhất: `_`  
Thành phần không giúp / hại: `_`
