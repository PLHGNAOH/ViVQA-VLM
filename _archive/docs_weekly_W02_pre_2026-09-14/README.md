# W02 — Literature & dataset

## 1. Objective

Xây literature matrix, phân tích dataset, chốt research gap (có evidence), RQ, hypotheses, và candidate methods. Mỗi SV nộp paper review theo 10 câu hỏi (bắt buộc: What can we use for our project?).

## 2. Deliverable / Milestone

Literature_Matrix · Dataset_Analysis · Research_Gap · Research_Questions · Hypotheses · Candidate_Methods · SV1–SV4 Paper Review.

File đặc thù tuần này: `Literature_Matrix.md, Dataset_Analysis.md, Research_Gap.md, Research_Questions.md, Hypotheses.md, Candidate_Methods.md, SV1–SV4_Paper_Review.md` · báo cáo nhóm: `W02_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Review paper VLM/baseline (BLIP-2 / Florence-2 / Qwen2.5-VL / LLaVA) → `SV1_Paper_Review.md`.
- [ ] Điền cột method/backbone trong Literature_Matrix.md.

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Review ViVQA, ViTextVQA (+ paper OCR tiếng Việt nếu cần) → `SV2_Paper_Review.md`.
- [ ] Hoàn thiện Dataset_Analysis.md (thống kê, loại câu hỏi, thách thức diacritics / scene-text).

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Review LoRA, QLoRA, PEFT-for-VLM, RAG-VQA → `SV3_Paper_Review.md`.
- [ ] Điền Candidate_Methods.md (feasibility dưới PEFT-only + GPU hạn chế).

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Review metric VQA/ANLS + POPE/hallucination → `SV4_Paper_Review.md`.
- [ ] Đề xuất protocol log + tiêu chí so sánh fair vs baseline.

## 4. Checklist

- [ ] Mỗi paper review có đủ 10 câu, gồm *What can we learn/use for our project?*.
- [ ] Research gap gắn evidence (citation), không copy nguyên đề cương.
- [ ] RQ và hypotheses đo được bằng EM / VQA-Acc / ANLS và/hoặc hallucination rate.
- [ ] Candidate methods loại trừ full fine-tuning.
- [ ] Nộp W02_Team_Report.md.

## 5. Definition of Done

Có gap có citation, RQ/hypothesis rõ, method ứng viên khả thi PEFT-only, và 4 paper review cá nhân.

## 6. Ràng buộc (nhắc lại)

Tuần này trọng tâm không phải train. Khi có huấn luyện ở tuần khác: **CHỈ PEFT (LoRA/QLoRA 4/8-bit), KHÔNG full fine-tuning.** Cải tiến luôn so baseline cùng split/seed/hardware protocol. Chi tiết: `docs/Master_Instruction.md` Mục 4 và Mục 7.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
