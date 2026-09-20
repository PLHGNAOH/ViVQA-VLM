# W05 — Baseline reproduce

## 1. Objective

Reproduce **đúng 1** VLM baseline trên split đã freeze. Ưu tiên zero-shot / official checkpoint (chưa cải tiến). Log đủ split, seed, hardware, time, EM / VQA-Acc / ANLS.

## 2. Deliverable / Milestone

Baseline_Reproduce.md · experiment log `experiments/W05_baseline_<model>/` · số liệu sơ bộ trên val (hoặc subset nếu GPU hạn chế — phải ghi rõ).

File đặc thù tuần này: `Baseline_Reproduce.md` · báo cáo nhóm: `W05_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Chạy inference baseline; ghi checkpoint id / revision Hugging Face.
- [ ] Ghi failure mode kiến trúc (tokenizer tiếng Việt, max length, image size).

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Đảm bảo dataloader đúng split; thống kê coverage OCR trên subset eval.
- [ ] Tag question type trên tập eval (yes/no, count, object, text, reasoning).

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Hỗ trợ quantization cho inference nếu OOM; **không** full FT.
- [ ] Lưu config run (dtype, batch, max_new_tokens).

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Tính EM, VQA-Acc, ANLS; điền Experiment_Log.
- [ ] Lưu 10 case đúng / 10 case sai (để W12).

## 4. Checklist

- [ ] Có số liệu baseline (dù là subset — phải nêu N và lý do).
- [ ] Log đủ: split, seed, hardware, infer time; train time = N/A nếu zero-shot.
- [ ] Không có full fine-tuning.
- [ ] Baseline_Reproduce.md mô tả lệnh chạy và lệch so với paper gốc (nếu có).
- [ ] Nộp W05_Team_Report.md.

## 5. Definition of Done

Có mốc baseline tái lập được (cùng điều kiện sẽ dùng cho mọi cải tiến sau này).

## 6. Ràng buộc PEFT-only (tuần có training / chạy model)

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
