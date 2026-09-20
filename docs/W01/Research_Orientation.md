# Research Orientation — ViVQA-VLM

> Điền trong W01. Không code nặng tuần này.

## 1. Bài toán

```text
Vietnamese Image + Vietnamese Question → VLM → Vietnamese Answer
```

- **Vì sao khó:** VLM pretrain thiên về tiếng Anh; scene-text; dấu thanh; code-switching; hallucination.
- **Vì sao đáng làm:** thích nghi VLM cho tiếng Việt dưới **ngân sách tham số** (PEFT-only) là đóng góp đo được.

## 2. Phạm vi (in / out)

| In scope | Out of scope |
|----------|----------------|
| 1 baseline reproduce (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL) | Full fine-tuning |
| ≥ 1 cải tiến: LoRA/QLoRA / OCR-prompt / RAG / prompt-opt | Train model từ scratch |
| ViVQA + ViTextVQA | Dataset nội bộ không licence |
| EM, VQA-Acc, ANLS + ablation + error + hallucination | Lấy BLEU làm metric chính |
| Prototype web | Production SLA / app store |

## 3. Research gap (sơ bộ — chốt evidence ở W02)

| ID | Gap | Khả thi Capstone? | Evidence cần đọc |
|----|-----|-------------------|------------------|
| G1 | English-centric pretraining | Có | BLIP-2, Qwen2.5-VL, surveys |
| G2 | Vietnamese VQA / scene-text yếu | Có | ViVQA, ViTextVQA |
| G3 | PEFT cho VLM tiếng Việt chưa bài bản | **Ưu tiên** | LoRA, QLoRA |
| G4 | Hallucination | Có (phân tích) | POPE |

## 4. Hướng phương pháp (ưu tiên — chốt phương pháp cuối ở W06)

- [x] QLoRA 4-bit trên baseline đã chọn (Qwen2.5-VL-3B đề xuất) — **trọng tâm**
- [x] OCR-enhanced prompting (PaddleOCR / VietOCR) — cho scene-text
- [x] Prompt optimization (format-prompt tiếng Việt, không train) — làm sớm ở W05
- [ ] Retrieval-augmented VQA — tùy chọn nếu còn thời gian (W09+)

## 5. Evaluation orientation

Metric chính: **Exact Match · VQA Accuracy · ANLS**. BLEU/CIDEr phụ. Mọi run log: split, seed, hardware, hyperparameters, train time, infer time.

## 6. Ethics (CLO7)

Privacy ảnh upload · bias English-centric · hallucination/reliability · nêu confidence và giới hạn trên prototype.

## 7. Câu hỏi định hướng (trả lời ngắn)

1. **Đề tài giải quyết gì?** Thích nghi một VLM mã nguồn mở cho **Vietnamese VQA** (ảnh tiếng Việt + câu hỏi tiếng Việt → câu trả lời tiếng Việt) **dưới ngân sách tham số** (không full fine-tuning), và kiểm chứng ≥ 1 cải tiến so với baseline.
2. **Vì sao quan trọng với NLP tiếng Việt?** Tiếng Việt ít tài nguyên; VLM hiện thiên English-centric. Adapt hiệu quả (PEFT) là hướng thực tế, đo được, còn ít nghiên cứu — mở ra ứng dụng đọc-hiểu ảnh tiếng Việt (scene-text, biển hiệu, tài liệu).
3. **VLM là gì (1 đoạn)?** Mô hình đa phương thức nối một **vision encoder** (trích đặc trưng ảnh) với một **language model** qua một **vision-language interface** (Q-Former/MLP/merger), cho phép suy luận đồng thời trên ảnh và văn bản: `Ảnh → Vision Encoder → Visual Representation → VL Interface → LLM → Câu trả lời`.
4. **Thách thức Vietnamese VQA?** Pretrain English-centric; **scene-text** (đọc chữ trong ảnh) cần OCR tốt; **dấu thanh (diacritics)** & chuẩn hoá Unicode; **code-switching** Anh–Việt; **hallucination**; tài nguyên GPU hạn chế.
5. **Nghiên cứu trước đã làm gì?** VLM nền tảng (BLIP-2, Florence-2, Qwen2.5-VL, LLaVA); benchmark tiếng Việt (ViVQA general, ViTextVQA scene-text); PEFT (LoRA, QLoRA, VL-Adapter); đánh giá hallucination (POPE). Nhưng **chưa kiểm chứng PEFT cho VLM trên tiếng Việt** (đặc biệt scene-text có dấu).
6. **Gap nào nhóm chọn?** **G3 — PEFT cho VLM tiếng Việt** làm trọng tâm; **G2 (scene-text + OCR)** và **G4 (hallucination)** làm trục phân tích/ablation.
7. **Sẽ đánh giá bằng gì?** Metric chính **EM · VQA-Accuracy · ANLS** (ANLS cho ViTextVQA); phân tích **hallucination (POPE-vi F1)**; BLEU/CIDEr chỉ phụ. Mọi run log split/seed/hardware/thời gian train+infer.
8. **15 tuần chạy thế nào?** → xem `Project_Plan_15_Weeks.md` (milestone W03 Review 1 · W07 Review 2 · W13 Faculty · W15 Final Defense, đã gắn ngày dự kiến).
