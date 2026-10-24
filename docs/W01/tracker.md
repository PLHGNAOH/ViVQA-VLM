# Week 1 — Research orientation & planning

> Milestone: none · Ngày: 2026-09-14 · Owner: cả nhóm (SV1–SV4)

## 🎯 Mục tiêu tuần

Hiểu bài toán Vietnamese VQA + ràng buộc PEFT-only; mỗi SV đọc ≥ 3 paper (10 câu hỏi + "liên quan gì đến đề tài"); dựng khung docs; hoàn thiện kế hoạch 15 tuần với 4 milestone (W03/W07/W13/W15) có ngày dự kiến; nộp W01 Team Report + deck tiến độ.

## ✅ Task list

- [x] (SV1) Review BLIP-2, Florence-2, Qwen2.5-VL; bảng so sánh 3 baseline; đề xuất baseline chính → `SV1_Paper_Review.md`
- [x] (SV2) Review ViVQA, ViTextVQA, TextVQA/LoRRA; bảng thống kê dataset; đề xuất OCR → `SV2_Paper_Review.md`
- [x] (SV3) Review LoRA, QLoRA, VL-Adapter, LLaVA-1.5; recipe QLoRA + ablation + hypotheses → `SV3_Paper_Review.md`
- [x] (SV4) Review POPE, VQAv2, Survey TPAMI; chốt EM/VQA-Acc/ANLS/POPE-vi + chuẩn hoá tiếng Việt → `SV4_Paper_Review.md`
- [x] (Cả nhóm) Kế hoạch 15 tuần gắn ngày milestone → `Project_Plan_15_Weeks.md`
- [x] (Cả nhóm) Research Orientation (bài toán, gap, RQ định hướng, evaluation) → `Research_Orientation.md`
- [x] (Cả nhóm) W01 Team Report + deck tiến độ → `W01_Team_Report.md`, `ViVQA-VLM_Week1_Progress.pptx`

## 📝 Meeting notes

- 2026-09-14: Thống nhất **PEFT-only** (LoRA/QLoRA 4/8-bit), KHÔNG full fine-tuning.
- Đề xuất **Qwen2.5-VL-3B** làm baseline chính (đa ngữ + OCR + fit Colab); chốt chính thức ở W03.
- Gap trọng tâm: **G3 — PEFT cho VLM tiếng Việt**; G2 (scene-text/OCR) & G4 (hallucination) làm trục ablation.
- Deliverable synthesis (Literature_Matrix, gap, RQ, hypotheses, methods, dataset analysis) hoàn thiện ở **W02**.

## 📦 Deliverables

| Deliverable | File | Trạng thái |
|-------------|------|-----------|
| Paper review ×4 | `SV1..SV4_Paper_Review.md` | ✅ Xong |
| Kế hoạch 15 tuần | `Project_Plan_15_Weeks.md` | ✅ Xong (ngày dự kiến) |
| Research Orientation | `Research_Orientation.md` | ✅ Xong |
| Team Report W01 | `W01_Team_Report.md` | ✅ Xong |
| Progress deck | `ViVQA-VLM_Week1_Progress.pptx` | ✅ Xong (12 slide) |
