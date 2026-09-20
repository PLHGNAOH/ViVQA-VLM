# W10 — Experiments

## 1. Objective

Chạy thí nghiệm chính (main results): baseline vs method cuối trên đúng split. Log mọi run. Có thể chạy nốt ablation nhẹ nếu GPU cho phép — ablation đầy đủ là W12.

## 2. Deliverable / Milestone

Experiments_Main_Results.md · logs trong `experiments/W10_*/`.

File đặc thù tuần này: `Experiments_Main_Results.md` · báo cáo nhóm: `W10_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Chạy/verify baseline main (không đổi checkpoint so với W05 protocol).
- [ ] Ghi version code inference.

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Cung cấp test/val loader chính thức; thống kê N per question type.
- [ ] OCR cache không đổi giữa các run so sánh.

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Chạy method cuối; không train thêm trên test.
- [ ] Nếu còn slot GPU: 1 ablation phụ (ví dụ LoRA rank).

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Tổng hợp bảng EM / VQA-Acc / ANLS (+ phụ BLEU/CIDEr).
- [ ] Ghi train/infer time và hardware từng run.

## 4. Checklist

- [ ] Bảng main results có baseline và proposed method.
- [ ] Mỗi dòng bảng map sang 1 experiment log.
- [ ] PEFT-only cho mọi run có train.
- [ ] Nộp W10_Team_Report.md.

## 5. Definition of Done

Có bảng kết quả chính tái lập được, đủ để viết R6.

## 6. Ràng buộc PEFT-only (tuần có training / chạy model)

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
