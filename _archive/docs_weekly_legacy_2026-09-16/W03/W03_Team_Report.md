# W03 — Team Weekly Report

> Điền theo `docs/templates/Weekly_Report_Template.md`. Không để trống các mục số liệu nếu tuần có thí nghiệm.

| Trường | Nội dung |
|--------|----------|
| **Tuần** | W03 — REVIEW 1 |
| **Nhóm** | ViVQA-VLM (4 SV) |
| **GVHD** | Assoc.Prof. Đặng Ngọc Minh Đức |
| **Ngày nộp** | _YYYY-MM-DD_ |

## 1. Mục tiêu tuần

**Milestone Review 1:** trình bày bài toán, related work, gap, RQ, hướng phương pháp (PEFT-only), kế hoạch 15 tuần, phân công, và tiêu chí đánh giá. Chưa yêu cầu số liệu baseline đầy đủ.

## 2. Đã làm (per-SV)

| SV | Vai trò | Công việc đã hoàn thành | Evidence (file / commit / log) |
|----|---------|-------------------------|--------------------------------|
| SV1 | VLM / Baseline | _…_ | `src/baseline/` · `docs/weekly/W03/` |
| SV2 | Dataset / OCR | _…_ | `data/` · `src/data/` |
| SV3 | PEFT / Training | _…_ | `src/adaptation/` · `experiments/` · `configs/` |
| SV4 | Evaluation / Deployment | _…_ | `src/eval/` · `src/prototype/` |

## 3. Kết quả / số liệu

| Mục | Giá trị | Ghi chú |
|-----|---------|---------|
| Dataset / split | ViVQA / ViTextVQA · split `_` · seed `_` | Bắt buộc nếu đã freeze split |
| Baseline model | BLIP-2 / Florence-2 / Qwen2.5-VL / _chưa chọn_ | Reproduce **đúng 1** |
| Method | Zero-shot / prompt / LoRA / QLoRA / OCR / RAG | PEFT-only nếu có train |
| Exact Match (EM) | _…_ | Metric chính |
| VQA Accuracy | _…_ | Metric chính |
| ANLS | _…_ | Metric chính (đặc biệt ViTextVQA) |
| BLEU / CIDEr | _…_ | **Phụ**, không dùng làm metric chính |
| Train time | _…_ | Nếu có PEFT |
| Inference time | _…_ | Batch size / hardware |
| Hardware | Colab Pro/Premium / GPU server trường · GPU `_` · VRAM `_` | |

## 4. Blocker

| Blocker | Impact | Owner | Hướng xử lý |
|---------|--------|-------|-------------|
| _…_ | High/Med/Low | SVx | _…_ |

## 5. Kế hoạch tuần sau (W04)

- [ ] SV1: _…_
- [ ] SV2: _…_
- [ ] SV3: _…_
- [ ] SV4: _…_

## 6. Link experiment log

- Template: `docs/templates/Experiment_Log_Template.md`
- Run folder: `experiments/W03_<ten_run>/` (nếu tuần này có chạy)
- Config: `configs/` (nếu có)

## 7. Ràng buộc nhắc lại

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.

