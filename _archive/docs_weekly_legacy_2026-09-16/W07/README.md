# W07 — REVIEW 2

> **MILESTONE BẮT BUỘC: REVIEW 2**

## 1. Objective

**Milestone Review 2:** chứng minh baseline đã reproduce, trình bày proposed method (PEFT-only), và **preliminary results** (dù subset). Chốt thí nghiệm sẽ chạy W08–W12.

## 2. Deliverable / Milestone

**REVIEW 2** — slides outline, Preliminary_Results.md, Review2_Checklist.md.

File đặc thù tuần này: `Review2_Slides_Outline.md, Preliminary_Results.md, Review2_Checklist.md` · báo cáo nhóm: `W07_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Slide: kết quả baseline + hạn chế (tiếng Việt, scene-text, hallucination).
- [ ] Defend lựa chọn đúng 1 baseline.

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Slide: thống kê data/OCR; sample lỗi OCR.
- [ ] Xác nhận split freeze không đổi sau Review 2.

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Slide: proposed method (LoRA/QLoRA và/hoặc OCR-prompt và/hoặc RAG).
- [ ] VRAM budget + lịch train W08–W09.

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Slide: preliminary metrics + experiment nào sẽ trình Faculty Review.
- [ ] Điều phối Q&A; ghi phút feedback.

## 4. Checklist

- [ ] Baseline số liệu có trong Preliminary_Results.md.
- [ ] Proposed method rõ, PEFT-only.
- [ ] Review2_Checklist.md hoàn tất trước giờ review.
- [ ] Kế hoạch W08–W12 gắn ablation + error + hallucination.
- [ ] Nộp W07_Team_Report.md.

## 5. Definition of Done

Review 2: baseline có số, method có, preliminary results có; nhóm biết thí nghiệm còn thiếu trước W13.

## 6. Ràng buộc (nhắc lại)

Tuần này trọng tâm không phải train. Khi có huấn luyện ở tuần khác: **CHỈ PEFT (LoRA/QLoRA 4/8-bit), KHÔNG full fine-tuning.** Cải tiến luôn so baseline cùng split/seed/hardware protocol. Chi tiết: `docs/Master_Instruction.md` Mục 4 và Mục 7.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
