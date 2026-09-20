# W13 — FACULTY REVIEW

> **MILESTONE BẮT BUỘC: FACULTY REVIEW**

## 1. Objective

**Milestone hội đồng cấp Khoa:** phần lớn nghiên cứu phải xong — methodology, experiments, main results, ablation, error analysis, hallucination. Prototype có thể chưa hoàn thiện (W14).

## 2. Deliverable / Milestone

**FACULTY REVIEW** — Faculty_Review_Package.md · Faculty_Review_Checklist.md · slides (outline).

File đặc thù tuần này: `Faculty_Review_Package.md, Faculty_Review_Checklist.md` · báo cáo nhóm: `W13_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Trình bày baseline + kiến trúc + qualitative.
- [ ] Q&A VLM.

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Trình bày data/OCR + error liên quan tiếng Việt.
- [ ] Q&A dataset licence / split.

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Trình bày PEFT method + ablation (cải thiện nhờ gì).
- [ ] Q&A: khẳng định không full FT.

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Trình bày metrics + hallucination + reproducibility.
- [ ] Ghép package; timekeeper.

## 4. Checklist

- [ ] Package gồm: problem, gap, method, main table, ablation, error, hallucination, limitations.
- [ ] Faculty_Review_Checklist.md tick hết mục bắt buộc.
- [ ] Số liệu khớp experiment log.
- [ ] PEFT-only nêu rõ trên slide ràng buộc.
- [ ] Nộp W13_Team_Report.md (kèm phút feedback hội đồng).

## 5. Definition of Done

Faculty Review hoàn thành; nhóm có list revision cho W14–W15.

## 6. Ràng buộc (nhắc lại)

Tuần này trọng tâm không phải train. Khi có huấn luyện ở tuần khác: **CHỈ PEFT (LoRA/QLoRA 4/8-bit), KHÔNG full fine-tuning.** Cải tiến luôn so baseline cùng split/seed/hardware protocol. Chi tiết: `docs/Master_Instruction.md` Mục 4 và Mục 7.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
