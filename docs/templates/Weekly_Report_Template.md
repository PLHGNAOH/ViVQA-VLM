# Weekly Report Template — ViVQA-VLM

Sao chép sang `docs/weekly/Wxx/Wxx_Team_Report.md` mỗi tuần (các file tuần đã prefill).

## Metadata

| Trường | Nội dung |
|--------|----------|
| **Tuần** | Wxx — _tên objective_ |
| **Nhóm** | ViVQA-VLM (4 SV) |
| **GVHD** | Assoc.Prof. Đặng Ngọc Minh Đức |
| **Ngày** | YYYY-MM-DD |

## Mục tiêu

_1 đoạn. Bám Master Instruction và README tuần._

## Đã làm (per-SV)

| SV | Vai trò | Đã làm | Evidence |
|----|---------|--------|----------|
| SV1 | VLM / Baseline | | `src/baseline/` |
| SV2 | Dataset / OCR | | `data/` · `src/data/` |
| SV3 | PEFT / Training | | `src/adaptation/` · `adapters/` |
| SV4 | Evaluation / Deployment | | `src/eval/` · `src/prototype/` |

## Kết quả / số liệu

| Mục | Giá trị |
|-----|---------|
| Dataset split + seed | |
| Model / baseline | BLIP-2 / Florence-2 / Qwen2.5-VL |
| Method | LoRA / QLoRA / OCR / RAG / prompt / zero-shot |
| **Exact Match (EM)** | |
| **VQA Accuracy** | |
| **ANLS** | |
| BLEU / CIDEr (phụ) | |
| Train time | |
| Inference time | |
| Hardware | Colab Pro/Premium / GPU trường |

## Blocker

| Blocker | Impact | Owner | Next step |
|---------|--------|-------|-----------|
| | | | |

## Kế hoạch tuần sau

- [ ] SV1:
- [ ] SV2:
- [ ] SV3:
- [ ] SV4:

## Link experiment log

- Template: `docs/templates/Experiment_Log_Template.md`
- Folder: `experiments/Wxx_<tên>/`

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.

