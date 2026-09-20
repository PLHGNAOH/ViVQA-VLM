# W06 — Baseline → Method

## 1. Objective

Chuyển từ baseline đã reproduce sang **methodology** chính thức (nguồn cho `03_Report/R4_Methodology.md`): pipeline, giả thuyết, thiết kế PEFT/OCR/RAG, ablation dự kiến.

## 2. Deliverable / Milestone

Methodology.md (đồng bộ ý với R4) · cập nhật RQ/hypothesis nếu Review 1 yêu cầu.

File đặc thù tuần này: `Methodology.md (nguồn cho `03_Report/R4_Methodology.md`)` · báo cáo nhóm: `W06_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Vẽ pipeline: image+question → VLM baseline → (optional PEFT/OCR/RAG) → answer.
- [ ] Xác định module nào freeze / nào LoRA (target modules).

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Chốt OCR trong method: luôn / chỉ ViTextVQA / fallback.
- [ ] Mô tả normalize câu trả lời (lowercase, dấu, tokenizer eval).

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Thiết kế LoRA rank, alpha, dropout, 4/8-bit; hoặc lý do chọn prompt/RAG trước PEFT.
- [ ] Kế hoạch ablation: zero-shot vs prompt vs LoRA vs OCR vs RAG.

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Protocol so sánh fair (cùng split/seed/decode).
- [ ] Định nghĩa hallucination labels (object / text) cho W12.

## 4. Checklist

- [ ] Methodology.md có đủ: data, baseline, improvement ≥ 1, eval, ablation plan.
- [ ] Cải tiến xuất phát từ gap W02, không 'thêm vì vui'.
- [ ] PEFT-only tường minh; không full FT.
- [ ] Đã copy/ý chính sang `03_Report/R4_Methodology.md` (không xoá nội dung cũ nếu đã có — file này là file mới).
- [ ] Nộp W06_Team_Report.md.

## 5. Definition of Done

Hội đồng (Review 2) có thể đọc Methodology và hiểu sẽ thí nghiệm gì, so với baseline ra sao.

## 6. Ràng buộc (nhắc lại)

Tuần này trọng tâm không phải train. Khi có huấn luyện ở tuần khác: **CHỈ PEFT (LoRA/QLoRA 4/8-bit), KHÔNG full fine-tuning.** Cải tiến luôn so baseline cùng split/seed/hardware protocol. Chi tiết: `docs/Master_Instruction.md` Mục 4 và Mục 7.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
