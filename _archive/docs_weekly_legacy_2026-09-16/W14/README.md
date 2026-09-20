# W14 — Prototype / Thesis

## 1. Objective

Hoàn thiện prototype web (upload ảnh tiếng Việt, hỏi tiếng Việt, trả lời, hiện confidence và OCR/evidence nếu có) và đẩy thesis draft (R1, R3–R7).

## 2. Deliverable / Milestone

Prototype_Notes.md · Thesis_Draft_Status.md · demo chạy được trên môi trường nhóm.

File đặc thù tuần này: `Prototype_Notes.md, Thesis_Draft_Status.md` · báo cáo nhóm: `W14_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Gắn model/adapter vào backend inference prototype.
- [ ] Giới hạn độ dài / timeout cho demo.

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Pipeline OCR realtime (hoặc cache) trên ảnh upload.
- [ ] Cảnh báo privacy (ảnh mặt, giấy tờ) trên UI.

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Load LoRA/QLoRA adapter; fallback zero-shot nếu OOM.
- [ ] Ghi hướng dẫn GPU demo.

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] UI: ảnh + câu hỏi + answer + confidence + OCR/evidence.
- [ ] Cập nhật thesis draft status; reproducibility README.

## 4. Checklist

- [ ] Prototype thỏa mục 11 Master Instruction.
- [ ] Thesis_Draft_Status.md: từng chương % hoàn thành.
- [ ] Không commit weights lớn; ghi path `adapters/`.
- [ ] Nộp W14_Team_Report.md.

## 5. Definition of Done

Demo chạy được trên 1 luồng happy-path; thesis có bản draft đủ chương.

## 6. Ràng buộc (nhắc lại)

Tuần này trọng tâm không phải train. Khi có huấn luyện ở tuần khác: **CHỈ PEFT (LoRA/QLoRA 4/8-bit), KHÔNG full fine-tuning.** Cải tiến luôn so baseline cùng split/seed/hardware protocol. Chi tiết: `docs/Master_Instruction.md` Mục 4 và Mục 7.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
