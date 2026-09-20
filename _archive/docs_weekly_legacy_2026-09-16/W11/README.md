# W11 — Full evaluation

## 1. Objective

Đánh giá đầy đủ metric chính (+ phụ), theo dataset (ViVQA vs ViTextVQA) và theo loại câu hỏi. Chuẩn bị dữ liệu cho ablation / error / hallucination (W12).

## 2. Deliverable / Milestone

Evaluation_Full_Metrics.md · pred files (mô tả path, không commit file lớn nếu cần thì để local).

File đặc thù tuần này: `Evaluation_Full_Metrics.md` · báo cáo nhóm: `W11_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Phân rã lỗi theo model behaviour (vision vs language vs alignment).
- [ ] Ghi chú truncation / multilingual tokenizer.

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Bảng metric theo question type và theo dataset.
- [ ] Đối chiếu lỗi OCR vs lỗi VQA.

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Xác nhận không có run 'lén' full FT.
- [ ] Liệt kê configs dùng cho eval.

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Hoàn thiện EM, VQA-Acc, ANLS; optional human eval mẫu nhỏ.
- [ ] Export danh sách failure cho W12.

## 4. Checklist

- [ ] Evaluation_Full_Metrics.md có bảng đủ metric chính.
- [ ] Có breakdown dataset và question type (dù N nhỏ phải ghi).
- [ ] BLEU/CIDEr nếu có thì gắn nhãn phụ.
- [ ] Nộp W11_Team_Report.md.

## 5. Definition of Done

Bộ số liệu đánh giá đầy đủ, sẵn sàng phân tích W12 và Faculty Review.

## 6. Ràng buộc (nhắc lại)

Tuần này trọng tâm không phải train. Khi có huấn luyện ở tuần khác: **CHỈ PEFT (LoRA/QLoRA 4/8-bit), KHÔNG full fine-tuning.** Cải tiến luôn so baseline cùng split/seed/hardware protocol. Chi tiết: `docs/Master_Instruction.md` Mục 4 và Mục 7.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
