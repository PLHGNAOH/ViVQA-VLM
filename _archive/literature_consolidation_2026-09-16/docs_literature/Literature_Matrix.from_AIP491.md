# Literature Matrix

> Mỗi hàng = 1 paper. Cột bắt buộc: What can we use for our project?

| # | Author / Year / Venue | Problem | Method | Dataset / Metric | Kết quả chính | Limitation | **What can we use for our project?** | SV |
|---|------------------------|---------|--------|------------------|---------------|-------------|--------------------------------------|-----|
| 1 | Li et al. / 2023 / ICML | Vision-language alignment | BLIP-2 (Q-Former) | _…_ | _…_ | English-centric, compute | Ứng viên **baseline**; không full FT | SV1 |
| 2 | Xiao et al. / 2024 / CVPR | Unified vision tasks | Florence-2 | _…_ | _…_ | _…_ | Ứng viên baseline | SV1 |
| 3 | Bai et al. / 2025 / TR | Open VLM | Qwen2.5-VL | _…_ | _…_ | VRAM | Ứng viên baseline | SV1 |
| 4 | Tran et al. / 2021 / PACLIC | Vietnamese VQA | ViVQA benchmark | ViVQA | _…_ | Quy mô | Dataset general | SV2 |
| 5 | Nguyen et al. / 2024 | Scene-text VQA vi | ViTextVQA | ViTextVQA / ANLS | _…_ | OCR noise | Dataset scene-text + ANLS | SV2 |
| 6 | Hu et al. / 2022 | PEFT | LoRA | _…_ | _…_ | Rank tuning | **Bắt buộc** nếu train | SV3 |
| 7 | Dettmers et al. / 2023 | PEFT + quant | QLoRA | _…_ | _…_ | 4-bit tradeoff | Fit Colab / GPU trường | SV3 |
| 8 | Li et al. / 2023 / EMNLP | Object hallucination | POPE | _…_ | _…_ | Không phải VQA vi | Rubric hallucination | SV4 |
| 9 | _…_ | | | | | | | |
| 10 | _…_ | | | | | | | |

## Ghi chú đọc

- File note chi tiết: `02_References/Reading_Notes/` (template 10 câu hỏi).
- Review cá nhân tuần này: `SV1_Paper_Review.md` … `SV4_Paper_Review.md`.
