# W03 — REVIEW 1

> **MILESTONE BẮT BUỘC: REVIEW 1**

## 1. Objective

**Milestone Review 1:** trình bày bài toán, related work, gap, RQ, hướng phương pháp (PEFT-only), kế hoạch 15 tuần, phân công, và tiêu chí đánh giá. Chưa yêu cầu số liệu baseline đầy đủ.

## 2. Deliverable / Milestone

**REVIEW 1** — slides outline, checklist, team report, package tài liệu W01–W02.

File đặc thù tuần này: `Review1_Slides_Outline.md, Review1_Checklist.md` · báo cáo nhóm: `W03_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Slide: VLM là gì + vì sao chọn 1 baseline (chưa cần số).
- [ ] Q&A kỹ thuật kiến trúc nếu hội đồng hỏi.

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Slide: ViVQA vs ViTextVQA, OCR, thách thức tiếng Việt.
- [ ] Chuẩn bị 3–5 sample ảnh+câu hỏi minh họa.

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Slide: PEFT-only, LoRA/QLoRA, vì sao không full FT.
- [ ] Nêu candidate improvement (PEFT / OCR-prompt / RAG) ở mức ý tưởng.

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Slide: metric chính + reproducibility (seed/split/hardware/time).
- [ ] Tổng hợp checklist Review 1; điều phối thời gian trình bày.

## 4. Checklist

- [ ] Review1_Slides_Outline.md đủ phần bắt buộc (problem, gap, method hướng, plan, eval).
- [ ] Review1_Checklist.md tick trước giờ review.
- [ ] Mỗi SV biết mình trình bày mục nào (không 'cả nhóm nói chung').
- [ ] Nhắc ràng buộc PEFT-only trên slide ràng buộc.
- [ ] Nộp W03_Team_Report.md.

## 5. Definition of Done

Review 1 hoàn thành: hội đồng nắm bài toán, gap, hướng PEFT-only, kế hoạch và phân công; nhóm ghi nhận feedback.

## 6. Ràng buộc (nhắc lại)

Tuần này trọng tâm không phải train. Khi có huấn luyện ở tuần khác: **CHỈ PEFT (LoRA/QLoRA 4/8-bit), KHÔNG full fine-tuning.** Cải tiến luôn so baseline cùng split/seed/hardware protocol. Chi tiết: `docs/Master_Instruction.md` Mục 4 và Mục 7.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
