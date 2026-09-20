# W15 — Final Defense

> **MILESTONE BẮT BUỘC: FINAL DEFENSE**

## 1. Objective

**Milestone Final Defense:** hoàn thiện slides, thesis, reproducibility package (code, LoRA/QLoRA adapters, configs, inference script, README setup/run, experiment log).

## 2. Deliverable / Milestone

**FINAL DEFENSE** — Final_Defense_Slides_Outline.md · Reproducibility_Package_Checklist.md · thesis + demo.

File đặc thù tuần này: `Final_Defense_Slides_Outline.md, Reproducibility_Package_Checklist.md` · báo cáo nhóm: `W15_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Slide kiến trúc + baseline; rehearsal Q&A.
- [ ] Kiểm tra inference script sạch.

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Slide data/OCR; kiểm tra hướng dẫn tải dataset (không vi phạm licence).
- [ ] Đóng gói sample (nếu được phép) cho demo.

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Đóng gói adapter + configs; README train PEFT.
- [ ] Xác nhận không có full FT checkpoint.

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Tổng rehearsal; checklist reproducibility.
- [ ] Slides kết quả + limitation + ethics (CLO7).

## 4. Checklist

- [ ] Reproducibility_Package_Checklist.md đủ 6 nhóm hạng mục.
- [ ] Slides cover 4 trụ DoD: Knowledge, Research, Rigor, Delivery.
- [ ] Số liệu khớp log; nêu hạn chế trung thực.
- [ ] Nộp W15_Team_Report.md.

## 5. Definition of Done

Bảo vệ: có cải tiến kiểm chứng vs baseline (PEFT-only), ablation+error+hallucination, prototype, package tái lập.

## 6. Ràng buộc (nhắc lại)

Tuần này trọng tâm không phải train. Khi có huấn luyện ở tuần khác: **CHỈ PEFT (LoRA/QLoRA 4/8-bit), KHÔNG full fine-tuning.** Cải tiến luôn so baseline cùng split/seed/hardware protocol. Chi tiết: `docs/Master_Instruction.md` Mục 4 và Mục 7.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
