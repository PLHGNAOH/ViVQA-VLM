# Literature Matrix — ViVOA-VLM

> Mỗi hàng = 1 paper. Cột **"What can we use for our project?"** là bắt buộc. Mỗi SV điền ≥ 3 hàng từ file `SVx_Paper_Review.md` của mình. Cập nhật: _YYYY-MM-DD_

## Bảng tổng

| # | Author / Year / Venue | Problem | Method | Dataset / Metric | Kết quả chính | Limitation | **What can we use for our project?** | SV |
|---|------------------------|---------|--------|------------------|---------------|------------|--------------------------------------|----|
| 1 | Li et al. / 2023 / ICML | Vision-language alignment | BLIP-2 (Q-Former) | _…_ | _…_ | _…_ | Ứng viên baseline | SV1 |
| 2 | Xiao et al. / 2024 / CVPR | Unified vision tasks | Florence-2 | _…_ | _…_ | _…_ | Ứng viên baseline | SV1 |
| 3 | Bai et al. / 2025 / TR | Open VLM | Qwen2.5-VL | _…_ | _…_ | _…_ | Ứng viên baseline | SV1 |
| 4 | Tran et al. / 2021 / PACLIC | Vietnamese VQA | ViVQA benchmark | ViVQA / _…_ | _…_ | _…_ | Dataset general | SV2 |
| 5 | Nguyen et al. / 2024 (ESWA 2025) | Scene-text VQA tiếng Việt | ViTextVQA | ViTextVQA / ANLS | _…_ | _…_ | Dataset scene-text + ANLS | SV2 |
| 6 | Hu et al. / 2022 / ICLR | Full FT LLM quá tốn bộ nhớ/lưu trữ | LoRA: ΔW = BA hạng thấp, gộp được | GLUE, E2E NLG, WikiSQL, MNLI, SAMSum / Acc, BLEU, ROUGE | GPT-3 175B: LoRA 4.7M tham số ≈/＞ full FT; 10.000× ít tham số, 0 độ trễ | Chỉ LLM tiếng Anh; chọn module heuristic; vẫn cần W₀ FP16 | **Kỹ thuật bắt buộc**; r nhỏ trên nhiều module; gộp trọng số cho demo; rank sweep | SV3 |
| 7 | Dettmers et al. / 2023 / NeurIPS | Fine-tune 33–65B trên 1 GPU | QLoRA: NF4 + Double Quant + Paged Optimizer, LoRA all-linear | MMLU, Vicuna (GPT-4/human Elo), GLUE, SuperNI | Guanaco-65B 99.3% ChatGPT, 24h/1 GPU 48GB; NF4 ≈ 16-bit | Chưa so full-FT 65B; chậm hơn 16-bit; tiếng Anh, thuần văn bản | **Fit Colab**; cấu hình NF4+DQ; all-linear targets; data quality > size | SV3 |
| 8 | Li et al. / 2023 / EMNLP | Object hallucination | POPE | _…_ | _…_ | _…_ | Rubric hallucination | SV4 |
| 9 | Sung et al. / 2022 / CVPR | PEFT cho V&L (VQA, caption) | VL-Adapter: adapter chia sẻ đa task, Compacter, Hyperformer, LoRA, prompt | VQAv2, GQA, NLVR2, COCO / Acc, CIDEr, % params | Single Adapter 4.18% ≈ full FT (77.4 vs 77.6); prompt-tuning yếu (59.0) | Mô hình nhỏ, 224 px, tiếng Anh, không OCR | Freeze vision encoder; adapter chung ViVQA+ViTextVQA; bỏ soft-prompt | SV3 |
| 10 | Liu et al. / 2024 / CVPR | Recipe VLM mạnh, rẻ, tái lập | LLaVA-1.5 (MLP connector, 336 px, VQA data + format prompt); LoRA r=128 | 12 benchmark incl. VQAv2, TextVQA, POPE, MME | SOTA 11/12; LoRA ≈ full FT ở 7B/13B (POPE cao hơn) | Tiếng Anh; 8×A100; hallucination còn; LoRA chỉ ở Model Zoo | Format prompt tiếng Việt; 2 LR (LoRA/projector); độ phân giải; POPE-vi | SV3 |
| 11 | _…_ | | | | | | | SV1 |
| 12 | _…_ | | | | | | | SV2 |
| 13 | _…_ | | | | | | | SV4 |

## Ma trận Gap × Paper (đánh ✓ nếu paper cung cấp evidence cho gap)

| Gap | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 |
|-----|----|----|----|----|----|----|----|----|----|-----|
| G1 English-centric pretraining | | | | | | ✓ | ✓ | | ✓ | ✓ |
| G2 Vietnamese VQA yếu, scene-text | | | | | | | | | ✓ | ✓ |
| G3 PEFT cho VLM tiếng Việt chưa có | | | | | | ✓ | ✓ | | ✓ | ✓ |
| G4 Hallucination | | | | | | | | | | ✓ |

## Ghi chú
- Reading note chi tiết: `docs/W01/SVx_Paper_Review.md` (10 câu hỏi + "liên quan gì đến ViVOA-VLM").
- Hàng #6, #7, #9, #10 do SV3 điền từ `SV3_Paper_Review.md`.
