# Hypotheses — ViVQA-VLM

> Owner: cả nhóm (SV3 soạn các giả thuyết PEFT) · Cập nhật: _YYYY-MM-DD_
> Mỗi giả thuyết phải **falsifiable**: có điều kiện bác bỏ rõ, có metric, có thí nghiệm kiểm chứng.

## 1. Giả thuyết chính

| # | Hypothesis | RQ | Cơ sở từ literature | Cách kiểm chứng | Điều kiện bác bỏ | Owner |
|---|------------|----|---------------------|-----------------|------------------|-------|
| H1 | QLoRA (NF4, 4-bit) trên VLM baseline cải thiện EM/VQA-Acc trên ViVQA **≥ +5 điểm** so với zero-shot cùng prompt. | RQ2 | LoRA ≈ full FT (Hu 2022); LLaVA-1.5 LoRA ≈ full FT; QLoRA NF4 ≈ 16-bit (Dettmers 2023) | W08–W10: train QLoRA trên train split, eval test split, 3 seed | Δ < +2 điểm hoặc không ổn định qua seed | SV3 |
| H2 | QLoRA 4-bit đạt **≥ 97%** hiệu năng của LoRA 16-bit (cùng r, cùng targets) trong khi giảm VRAM đỉnh ≥ 40%. | RQ2c | QLoRA Table (NF4+DQ ≈ BF16 trong ±1 điểm) | W09: chạy cặp 4-bit / 16-bit trên A100, log VRAM + thời gian | Chênh > 3% hoặc VRAM không giảm | SV3 |
| H3 | LoRA trên **tất cả linear layer** của LLM tốt hơn attention-only ở cùng ngân sách tham số. | RQ2a | QLoRA §4 (Figure: all-linear cần thiết để khớp full FT) | W08 ablation A2 | Attention-only ≥ all-linear | SV3 |
| H4 | Hiệu năng bão hòa ở **r ≤ 16**; tăng r lên 64 không cải thiện > 1 điểm. | RQ2b | LoRA §7.2 (r=1 đủ cho Wq+Wv) | W09 ablation A3 | r=64 hơn r=16 > 1 điểm nhất quán | SV3 |
| H5 | PEFT adaptation **không làm tăng** hallucination (POPE-vi F1 không giảm) so với baseline zero-shot. | RQ4 | LLaVA-1.5 Model Zoo: LoRA POPE 86.4–86.7 > full FT 85.9 | W12 với SV4 | POPE-vi giảm > 2 điểm F1 | SV3, SV4 |
| H6 | OCR-enhanced prompting tăng ANLS trên ViTextVQA ≥ +3 điểm, và cộng hưởng (không triệt tiêu) với QLoRA. | RQ3 | LLaVA-1.5 (OCR-VQA/TextCaps data giúp TextVQA); ViTextVQA paper | W08–W11 ablation A7 | Δ ANLS < +1 hoặc âm khi kết hợp | SV2, SV3 |
| H7 | Response-format prompt tiếng Việt ("Trả lời bằng một từ hoặc cụm từ ngắn.") tăng EM zero-shot ≥ +3 điểm mà không cần huấn luyện. | RQ2e | LLaVA-1.5 Table 2 (format prompt: MME 1197 → 1323.8) | W05 khi reproduce baseline | Δ EM < +1 | SV3, SV4 |
| H8 | Adapter chung cho ViVQA + ViTextVQA ≥ adapter riêng trên tập nhỏ hơn (ViTextVQA). | RQ2d | VL-Adapter: Single Adapter NLVR2 74.2 vs Multiple 69.8 | W09 ablation A5 | Adapter riêng hơn > 1 điểm | SV3 |

## 2. Giả thuyết do SV1 / SV2 / SV4 bổ sung

| # | Hypothesis | RQ | Cơ sở | Kiểm chứng | Bác bỏ | Owner |
|---|------------|----|-------|------------|--------|-------|
| H9 | _…_ (baseline: ví dụ Qwen2.5-VL zero-shot > BLIP-2 zero-shot trên ViVQA do tokenizer đa ngữ) | RQ1 | | | | SV1 |
| H10 | _…_ (dataset/OCR) | RQ3 | | | | SV2 |
| H11 | _…_ (evaluation) | RQ4 | | | | SV4 |

## 3. Checklist
- [ ] Mỗi H có điều kiện bác bỏ định lượng
- [ ] Mỗi H gắn ≥ 1 ablation trong `Candidate_Methods.md`
- [ ] Nhóm chốt H1 là giả thuyết trọng tâm cho Review 1 (W03)
