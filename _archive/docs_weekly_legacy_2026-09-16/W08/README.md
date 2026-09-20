# W08 — Improvement v1

## 1. Objective

Triển khai **cải tiến bản 1** (PEFT và/hoặc OCR-enhanced prompting và/hoặc RAG) trên **cùng điều kiện** baseline. Mục tiêu: có run so sánh được, chưa cần tối ưu hết.

## 2. Deliverable / Milestone

Improvement_v1_PEFT_OCR_RAG.md · adapters (nếu LoRA) trong `adapters/` (không commit file lớn) · experiment log W08.

File đặc thù tuần này: `Improvement_v1_PEFT_OCR_RAG.md` · báo cáo nhóm: `W08_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Tích hợp adapter / prompt vào pipeline inference baseline.
- [ ] Kiểm tra không regress input format so với W05.

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Nếu OCR-prompt: chèn OCR text vào prompt; log OCR engine + version.
- [ ] Nếu RAG: index caption/OCR; ghi retrieval k.

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Train LoRA/QLoRA (4/8-bit) **hoặc** hoàn thiện prompt/RAG v1 nếu chọn non-weight method trước.
- [ ] Log: lr, rank, steps, train time, loss curve (ghi file log, không cần plot binary).

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] So sánh v1 vs baseline trên cùng split/seed: EM, VQA-Acc, ANLS.
- [ ] Ghi infer time (adapter vs baseline).

## 4. Checklist

- [ ] Có ít nhất 1 run cải tiến vs baseline, điều kiện y hệt.
- [ ] Không full fine-tuning.
- [ ] Improvement_v1_PEFT_OCR_RAG.md mô tả thay đổi và kết quả.
- [ ] Experiment log đầy đủ.
- [ ] Nộp W08_Team_Report.md.

## 5. Definition of Done

Có evidence số rằng v1 đã chạy và so được với baseline (tăng, hòa, hoặc giảm — đều phải báo cáo trung thực).

## 6. Ràng buộc PEFT-only (tuần có training / chạy model)

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
