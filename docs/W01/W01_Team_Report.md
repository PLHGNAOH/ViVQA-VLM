# W01 Team Report — Research Orientation & Planning

> Điền theo `docs/templates/Weekly_Report_Template.md`. Không để trống các mục số liệu nếu tuần có thí nghiệm.

## Metadata

| Trường | Nội dung |
|--------|----------|
| **Tuần** | W01 — Research orientation & planning |
| **Nhóm** | ViVQA-VLM (4 SV) |
| **GVHD** | Assoc.Prof. Đặng Ngọc Minh Đức |
| **Ngày nộp** | 2026-09-14 |

## 1. Mục tiêu tuần

Định hướng nghiên cứu và lập kế hoạch 15 tuần: hiểu bài toán Vietnamese VQA với VLM, ràng buộc **PEFT-only**, phân công SV1–SV4, mỗi SV đọc ≥ 3 paper theo mảng (10 câu hỏi + "liên quan gì đến ViVQA-VLM"), dựng khung docs, và chốt milestone Review 1 / Review 2 / Faculty Review / Final Defense.

## 2. Đã làm (per-SV)

| SV | Vai trò | Công việc đã hoàn thành | Evidence (file / commit / log) |
|----|---------|-------------------------|--------------------------------|
| SV1 | VLM / Baseline | Review 3 paper (BLIP-2, Florence-2, Qwen2.5-VL) đủ 10 câu; lập bảng so sánh 3 baseline, đề xuất Qwen2.5-VL-3B làm baseline chính | `docs/W01/SV1_Paper_Review.md`; `Literature_Matrix.md` (weekly #1–#3 lúc W01; canonical L1/L5/L6) |
| SV2 | Dataset / OCR | Review 3 paper (ViVQA, ViTextVQA, TextVQA/LoRRA) đủ 10 câu; bảng thống kê nhanh 2 dataset; đề xuất PaddleOCR + đối chứng VietOCR | `docs/W01/SV2_Paper_Review.md`; `Literature_Matrix.md` (weekly #4–#5; canonical L3/L4) |
| SV3 | PEFT / Training | Review 4 paper (LoRA, QLoRA, VL-Adapter, LLaVA-1.5) đủ 10 câu; đề xuất recipe QLoRA, 9 ablation, 8 giả thuyết PEFT; điền 4 hàng Literature Matrix | Canonical review: `docs/02_references/Reading_Notes/SV3_Paper_Review.md` (W01 = pointer); `Hypotheses.md` H1–H8; `Candidate_Methods.md` |
| SV4 | Evaluation / Deployment | Review 3 paper (POPE, VQAv2, Survey TPAMI) đủ 10 câu; chốt định nghĩa EM/VQA-Acc/ANLS/POPE-vi; giao thức chuẩn hoá câu trả lời tiếng Việt | `docs/W01/SV4_Paper_Review.md`; `Literature_Matrix.md` (weekly #8; canonical L9) |

## 3. Team deliverables

- [x] Cấu trúc thư mục `ViVQA-VLM/` + khung docs `docs/weekly/W01/`
- [x] `Project_Plan_15_Weeks.md` — 4 milestone đã **gắn ngày dự kiến** (W03/W07/W13/W15)
- [x] Paper review cá nhân SV1–SV4 (mỗi người ≥ 3 paper, đủ 10 câu + mục liên quan)
- [x] `Research_Orientation.md` — bài toán, scope, gap sơ bộ, risk, evaluation (mục 7 đã trả lời)
- [x] Deliverable synthesis (Literature_Matrix, Research_Gap, RQ, Hypotheses, Candidate_Methods, Dataset_Analysis) — **thuộc W02**, bản nháp đã có tại `docs/weekly/W02/`
- [x] `ViVQA-VLM_Week1_Progress.pptx` (deck tiến độ, 12 slide)

## 4. Kết quả / số liệu

| Mục | Giá trị | Ghi chú |
|-----|---------|---------|
| Dataset / split | ViVQA / ViTextVQA · split `chốt W02` · seed `42` | Số liệu chi tiết → W02 |
| Baseline model | **Qwen2.5-VL-3B** (đề xuất) · BLIP-2 đối chứng · Florence-2 ablation | Reproduce **đúng 1**, chốt W03 |
| Method | Đề xuất: QLoRA NF4 all-linear + format-prompt tiếng Việt + OCR-prompting | PEFT-only |
| Exact Match (EM) | _—_ (đo từ W05) | Metric chính |
| VQA Accuracy | _—_ (đo từ W05) | Metric chính |
| ANLS | _—_ (đo từ W05, ViTextVQA) | Metric chính scene-text |
| BLEU / CIDEr | _—_ | **Phụ**, không dùng làm metric chính |
| Train time | _—_ (đo pilot W04) | Nếu có PEFT |
| Inference time | _—_ (đo W05) | Batch size / hardware |
| Hardware | Colab Pro/Premium / GPU trường | GPU/VRAM ghi khi chạy |

> Tuần 1 **không có thí nghiệm chính thức** (đúng phạm vi orientation); các ô metric để "—" và sẽ đo từ W04–W05.

## 5. Blocker

| Blocker | Impact | Owner | Hướng xử lý |
|---------|--------|-------|-------------|
| Chưa tải PDF/dataset về local | Low | Cả nhóm | Tải dataset + PDF vào W02; đối chiếu số liệu "≈" trong review |
| Chưa cấp phát GPU trường / hạn mức Colab | Medium | SV3 | Đăng ký Colab Pro/Premium; queue GPU trường cho pilot W04 |

## 6. Kế hoạch tuần sau (W02)

- [ ] SV1: hoàn tất so sánh baseline (bảng số) BLIP-2 / Florence-2 / Qwen2.5-VL; hỗ trợ chốt baseline
- [ ] SV2: tải ViVQA, ViTextVQA; thống kê thật (#ảnh/#QA/loại câu hỏi); thử PaddleOCR/VietOCR trên 50 ảnh (CER/WER)
- [ ] SV3: đọc bổ sung DoRA / LLaMA-Adapter V2; viết notebook pilot QLoRA (Qwen2.5-VL-3B, ~200 mẫu) cho `experiments/W01_Pilot/`
- [ ] SV4: hiện thực `src/eval/` (EM, VQA-Acc, ANLS, POPE-vi) + script chuẩn hoá câu trả lời tiếng Việt

## 7. Link experiment log

- Template: `docs/templates/Experiment_Log_Template.md`
- Run folder: `experiments/W01_Pilot/` (pilot QLoRA dự kiến W04)
- Config: `configs/` (pin phiên bản ở W04)

## 8. Ràng buộc nhắc lại

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.
