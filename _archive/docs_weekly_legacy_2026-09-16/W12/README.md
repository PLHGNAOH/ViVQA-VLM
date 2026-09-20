# W12 — Analysis

## 1. Objective

Hoàn thành **ablation**, **error analysis**, **hallucination analysis** — bắt buộc trước Faculty Review (W13). Mỗi claim 'cải thiện nhờ X' phải có ablation.

## 2. Deliverable / Milestone

Ablation_Study.md · Error_Analysis.md · Hallucination_Analysis.md.

File đặc thù tuần này: `Ablation_Study.md, Error_Analysis.md, Hallucination_Analysis.md` · báo cáo nhóm: `W12_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Case study failure: model nhìn sai object / bỏ qua region.
- [ ] So sánh qualitative baseline vs method (5–10 ảnh).

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Error theo loại câu hỏi; scene-text + diacritics + code-switching.
- [ ] Liên hệ lỗi OCR với ANLS.

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Ablation: zero-shot vs prompt vs LoRA/QLoRA vs OCR vs RAG (các nhánh đã triển khai).
- [ ] Bảng Δ metric từng thành phần.

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Hallucination: object vs text; protocol (POPE-style hoặc rubric nhóm).
- [ ] Tổng hợp figure/table cho thesis R6/R7.

## 4. Checklist

- [ ] Ablation cover các thành phần method đã claim.
- [ ] Error analysis có phân rã + ví dụ ảnh tiếng Việt.
- [ ] Hallucination analysis có định nghĩa + tỉ lệ / ví dụ.
- [ ] Không đổi test labels để 'làm đẹp số'.
- [ ] Nộp W12_Team_Report.md.

## 5. Definition of Done

Đủ 3 phân tích bắt buộc; sẵn sàng gói Faculty Review.

## 6. Ràng buộc (nhắc lại)

Tuần này trọng tâm không phải train. Khi có huấn luyện ở tuần khác: **CHỈ PEFT (LoRA/QLoRA 4/8-bit), KHÔNG full fine-tuning.** Cải tiến luôn so baseline cùng split/seed/hardware protocol. Chi tiết: `docs/Master_Instruction.md` Mục 4 và Mục 7.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
