# W04 — Baseline setup

## 1. Objective

Dựng môi trường (Colab Pro/Premium hoặc GPU trường), chốt 1 baseline, freeze split, và viết Baseline_Plan (bước reproduce, hyperparameters dự kiến, ngân sách VRAM).

## 2. Deliverable / Milestone

Environment_Setup.md · Baseline_Plan.md · môi trường chạy thử inference 1 batch (nếu GPU sẵn).

File đặc thù tuần này: `Environment_Setup.md, Baseline_Plan.md` · báo cáo nhóm: `W04_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Chốt đúng 1 baseline; clone/hf model card; ghi version transformers/peft/torch.
- [ ] Viết Baseline_Plan.md (zero-shot trước, PEFT sau — không full FT).

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Tải / đăng ký dataset vào `data/raw/` (không commit ảnh lớn).
- [ ] Viết spec split → `data/splits/` (seed ghi rõ); thử OCR 1 subset.

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Setup bitsandbytes / peft / accelerate; kiểm tra QLoRA 4-bit fit VRAM.
- [ ] Soạn `configs/` naming cho run sau (chưa cần yaml nếu chưa chạy).

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Script khung log metric (EM/VQA-Acc/ANLS) trong `src/eval/`.
- [ ] Ghi hardware inventory (GPU name, VRAM, driver) vào Environment_Setup.md.

## 4. Checklist

- [ ] Environment_Setup.md: OS/Python/CUDA/GPU/Colab hay server.
- [ ] Baseline_Plan.md chọn đúng 1 model + lý do + kế hoạch reproduce.
- [ ] Split seed đã quyết định (kể cả 'dùng split gốc của dataset').
- [ ] PEFT-only được ghi trong plan (cấm full FT).
- [ ] Nộp W04_Team_Report.md.

## 5. Definition of Done

Môi trường tái lập được; baseline đã chọn; split/seed có chủ; plan reproduce không vi phạm PEFT-only.

## 6. Ràng buộc PEFT-only (tuần có training / chạy model)

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.

## Liên kết

- Master: [`docs/Master_Instruction.md`](../../Master_Instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../../templates/Experiment_Log_Template.md)
