# Candidate Methods — ViVQA-VLM

> **[W3-PREP — does NOT block Week-2 100%]** Finalize method at W6.
> Constraint: **PEFT-only** (LoRA/QLoRA). No full fine-tuning. Mọi cải tiến so với **đúng 1** baseline đã reproduce.
> Owner: SV3 (PEFT) · SV1 (baseline) · SV2 (OCR) · SV4 (eval/prototype) · Cập nhật: _YYYY-MM-DD_
>
> Scaffold note (not a second table): the English W02 scaffold marked **Qwen2.5-VL-3B** as recommended primary. The weekly table below lists Qwen2.5-VL 3B/7B with VRAM estimates; **Trạng thái** is still `_…_`.
> TODO (W3): chốt đúng 1 baseline; confirm GPU + pin versions (`transformers` / `accelerate` / `bitsandbytes` / `peft`).
> Full rationale + starting config: `docs/literature/SV3_Paper_Review.md` §5 / §5.2.

## 1. Baseline ứng viên (SV1 chọn ĐÚNG 1 ở W03–W04)

| Model | Kích cỡ | Interface | Ưu | Nhược | VRAM 4-bit (ước lượng) | Trạng thái |
|-------|---------|-----------|----|-------|------------------------|-----------|
| BLIP-2 | ~3.9B (OPT-2.7B) / ~12B (FlanT5-XXL) | Q-Former | Nhẹ, nhiều tài liệu | LLM tiếng Anh, yếu scene-text | | _…_ |
| Florence-2 | 0.23B / 0.77B | seq2seq, task prompt | Rất nhẹ, OCR nội tại | Tiếng Việt hạn chế, câu trả lời ngắn | | _…_ |
| Qwen2.5-VL | 3B / 7B | ViT + merger + Qwen2.5 LLM | Đa ngữ, OCR mạnh, dynamic resolution | VRAM (7B) | 3B ≈ 2–2.5 GB, 7B ≈ 4.5–5.5 GB trọng số | _…_ |

## 2. Danh mục cải tiến (theo Master Instruction 4.2) — xếp ưu tiên

| Ưu tiên | Phương pháp | Mô tả ngắn | Xuất phát từ paper | Cần train? | Owner | Tuần |
|---------|-------------|------------|--------------------|-----------|-------|------|
| 1 | **QLoRA fine-tuning** (NF4 + DQ, LoRA all-linear trên LLM, freeze ViT) | Cải tiến trọng tâm | Hu 2022; Dettmers 2023; Liu 2024 | Có | SV3 | W08–W09 |
| 2 | **Response-format prompt tiếng Việt** | Prompt optimization, không train | Liu 2024 | Không | SV3, SV4 | W05 |
| 3 | **OCR-enhanced prompting** (PaddleOCR / VietOCR → chèn text vào prompt) | Multimodal prompt engineering | ViTextVQA; Liu 2024 | Không | SV2, SV3 | W08 |
| 4 | **Adapter chung đa dataset** (ViVQA + ViTextVQA) | Lightweight VLM adaptation | Sung 2022 | Có | SV3 | W09 |
| 5 | Retrieval-Augmented VQA (RAG) — few-shot ví dụ tương tự qua CLIP embedding | Tùy chọn nếu còn thời gian | — | Không | SV3 | W09+ |
| — | Soft prompt-tuning | **Loại** — yếu trên V&L (Sung 2022: 59.0 vs 77.6) | Sung 2022 | — | — | — |

## 3. Recipe PEFT xuất phát (chi tiết trong `SV3_Paper_Review.md` §5.2)

| Tham số | Giá trị xuất phát | Ablation |
|---------|-------------------|----------|
| Quantization | NF4 + double quant, compute BF16 (FP16 trên T4) | 4-bit / 8-bit / 16-bit |
| LoRA r / α / dropout | 16 / 32 / 0.05 | r ∈ {8,16,32,64} |
| Target modules | q,k,v,o,gate,up,down (all-linear LLM) | attention-only |
| Vision tower | Freeze | + LoRA trên ViT (ViTextVQA) |
| Projector / merger | Freeze (ablation: mở, LR 2e-5) | mở / đóng |
| LR / optimizer | 2e-4, paged AdamW 32-bit, max_grad_norm 0.3 | 1e-4 |
| Epoch / batch | 1–3 / 2 × grad-accum 8 | — |
| Seed | 42 (+ 2 seed phụ cho kết quả chính) | — |

## 4. Ma trận ablation (bắt buộc — Master Instruction 7.3)

| # | Ablation | H | Owner | Tuần |
|---|----------|---|-------|------|
| A1 | Zero-shot vs prompt-optimized vs LoRA/QLoRA | H1, H7 | SV3 | W08–W10 |
| A2 | Target modules attention-only vs all-linear | H3 | SV3 | W08 |
| A3 | Rank sweep | H4 | SV3 | W09 |
| A4 | 4-bit vs 8-bit vs 16-bit LoRA (VRAM, thời gian) | H2 | SV3 | W09 |
| A5 | Adapter chung vs riêng | H8 | SV3 | W09 |
| A6 | Có vs không OCR-enhanced prompting | H6 | SV2, SV3 | W08–W11 |
| A7 | Có vs không retrieval augmentation (nếu làm) | — | SV3 | W09+ |
| A8 | Backbone khác (nếu tài nguyên cho phép) | — | SV1 | W10 |
| A9 | Hallucination trước / sau PEFT | H5 | SV4, SV3 | W12 |

## 5. Công cụ / thư viện

| Thành phần | Thư viện | Phiên bản (pin ở W04) |
|------------|----------|------------------------|
| Model | `transformers`, `accelerate` | |
| PEFT | `peft` (LoRA), `bitsandbytes` (NF4/8-bit) | |
| Train | `trl` SFTTrainer hoặc HF `Trainer` | |
| OCR | `paddleocr`, `vietocr` | |
| Eval | `src/eval/` (EM, VQA-Acc, ANLS, POPE-vi) | |
| Prototype | Gradio / Streamlit | |

## 6. Checklist
- [ ] Baseline chốt (SV1) — ngày: ____
- [ ] Recipe PEFT chốt (SV3) — ngày: ____
- [ ] Ma trận ablation gắn với `Hypotheses.md`
