# W09 — Improvement final

## 1. Objective

Chốt **cải tiến cuối** (hyperparameter, kết hợp OCR/RAG/PEFT). Freeze method trước khi chạy full eval W10–W11. Mọi tuyên bố cải thiện phải so baseline cùng điều kiện.

## 2. Deliverable / Milestone

Improvement_Final.md · adapter/config cuối · không đổi split/seed.

File đặc thù tuần này: `Improvement_Final.md` · báo cáo nhóm: `W09_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Freeze inference recipe (decode, image preprocess) cho method cuối.
- [ ] Document khác biệt vs v1.

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Freeze OCR/RAG recipe (engine, prompt template, k).
- [ ] Kiểm tra không leak test vào train/index.

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Train/chọn checkpoint cuối (PEFT only); lưu adapter id.
- [ ] Ghi hyperparameters cuối vào Improvement_Final.md.

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Smoke eval val; không được 'tune trên test'.
- [ ] Chuẩn bị bảng main results (trống số test — điền W10/W11).

## 4. Checklist

- [ ] Method cuối mô tả đủ để người khác chạy lại.
- [ ] PEFT-only; adapter không phải full weights.
- [ ] Split/seed không đổi so với baseline.
- [ ] Nộp W09_Team_Report.md.

## 5. Definition of Done

Cải tiến đã freeze; sẵn sàng experiments chính và evaluation đầy đủ.

## 6. Ràng buộc PEFT-only (tuần có training / chạy model)

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
