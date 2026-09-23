# W01 — Orientation & planning

## 1. Objective

Định hướng nghiên cứu và lập kế hoạch 15 tuần: hiểu bài toán Vietnamese VQA, ràng buộc PEFT-only, phân công SV1–SV4, và chốt milestone Review 1 / Review 2 / Faculty Review / Final Defense.

## 2. Deliverable / Milestone

W01 Team Report · Project_Plan_15_Weeks.md · Research_Orientation.md · phân công cá nhân rõ ràng.

File đặc thù tuần này: `Project_Plan_15_Weeks.md, Research_Orientation.md` · báo cáo nhóm: `W01_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Tóm tắt kiến trúc VLM (vision encoder, projector/Q-Former, LM) và 3 ứng viên baseline: BLIP-2, Florence-2, Qwen2.5-VL.
- [ ] Đề xuất 1 baseline sẽ reproduce (kèm lý do VRAM / giấy phép / code có sẵn).

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Rà soát ViVQA và ViTextVQA (licence, kích thước, split gốc, đặc thù scene-text).
- [ ] Phác thảo pipeline OCR (PaddleOCR vs VietOCR) và nơi lưu `data/raw`, `data/ocr`.

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Đọc LoRA/QLoRA; liệt kê ràng buộc: không full FT, 4/8-bit, Colab Pro/Premium hoặc GPU trường.
- [ ] Phác thảo không gian phương pháp: PEFT · OCR-prompt · RAG · prompt-opt.

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Chốt metric chính: EM, VQA Accuracy, ANLS; BLEU/CIDEr chỉ phụ.
- [ ] Phác thảo experiment log (seed, split, hardware, thời gian) và yêu cầu prototype (upload ảnh + hỏi tiếng Việt).

## 4. Checklist

- [ ] Cả nhóm đọc `docs/Master_Instruction.md` (Mục 4 ràng buộc + Mục 7 evaluation).
- [ ] Đã điền Project_Plan_15_Weeks.md (gantt logic, owner từng deliverable).
- [ ] Đã điền Research_Orientation.md (bài toán, gap sơ bộ, rủi ro ≥ 5).
- [ ] Phân công SV1–SV4 không chồng lấn trách nhiệm kỹ thuật.
- [ ] Nộp W01_Team_Report.md.

## 5. Definition of Done

Nhóm giải thích được bài toán, ràng buộc PEFT-only, 4 milestone, và mỗi SV có task cá nhân + deliverable tuần 1.

## 6. Ràng buộc (nhắc lại)

Tuần này trọng tâm không phải train. Khi có huấn luyện ở tuần khác: **CHỈ PEFT (LoRA/QLoRA 4/8-bit), KHÔNG full fine-tuning.** Cải tiến luôn so baseline cùng split/seed/hardware protocol. Chi tiết: `docs/master_instruction.md` Mục 4 và Mục 7.

## Liên kết

- Master: [`docs/master_instruction.md`](../master_instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../templates/Experiment_Log_Template.md)
