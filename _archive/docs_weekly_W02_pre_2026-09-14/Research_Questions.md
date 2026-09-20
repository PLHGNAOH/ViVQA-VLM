# Research Questions — ViVQA-VLM

> Owner: cả nhóm · Cập nhật: _YYYY-MM-DD_
> Mỗi RQ phải: gắn với 1 gap trong `Research_Gap.md`, có metric đo được, có SV chịu trách nhiệm.

## 1. Research question chính

| # | Research Question | Gap | Metric | Dataset | Owner |
|---|-------------------|-----|--------|---------|-------|
| RQ1 | Một VLM mã nguồn mở (BLIP-2 / Florence-2 / Qwen2.5-VL) đạt hiệu năng zero-shot thế nào trên Vietnamese VQA (ViVQA, ViTextVQA)? | G1, G2 | EM, VQA-Acc, ANLS | ViVQA, ViTextVQA | SV1, SV4 |
| RQ2 | PEFT (LoRA / QLoRA 4-bit) trên VLM đó cải thiện Vietnamese VQA bao nhiêu so với zero-shot và prompt-optimized, **không full fine-tuning**? | G3 | ΔEM, ΔVQA-Acc, ΔANLS; % tham số; VRAM; thời gian train | ViVQA, ViTextVQA | SV3 |
| RQ3 | OCR-enhanced prompting (PaddleOCR / VietOCR) có cải thiện scene-text VQA tiếng Việt không, và tương tác thế nào với PEFT? | G2 | ANLS, EM theo loại "text reading" | ViTextVQA | SV2, SV3 |
| RQ4 | PEFT adaptation ảnh hưởng thế nào đến hallucination (object / text) trên ảnh tiếng Việt? | G4 | POPE-vi F1, tỉ lệ hallucination theo loại | Subset thủ công | SV4, SV3 |

## 2. Research question phụ (ablation)

| # | Sub-RQ | Xuất phát từ | Owner |
|---|--------|--------------|-------|
| RQ2a | LoRA target modules: attention-only vs all-linear? | QLoRA (Dettmers 2023), LoRA (Hu 2022) | SV3 |
| RQ2b | Rank r ∈ {8,16,32,64} — hiệu năng bão hòa ở đâu? | LoRA §7 | SV3 |
| RQ2c | 4-bit NF4 vs 8-bit vs 16-bit LoRA: trade-off chất lượng / VRAM / thời gian? | QLoRA | SV3 |
| RQ2d | Adapter chung cho ViVQA + ViTextVQA vs adapter riêng? | VL-Adapter (Sung 2022) | SV3 |
| RQ2e | Response-format prompt tiếng Việt cải thiện EM bao nhiêu (trước khi train)? | LLaVA-1.5 (Liu 2024) | SV3, SV4 |
| RQ3a | Độ phân giải ảnh × OCR-prompting trên ViTextVQA? | LLaVA-1.5 §5.2 | SV2, SV3 |
| RQ4a | Hallucination phân rã theo loại câu hỏi (yes/no, counting, text, reasoning)? | Master Instruction 7.4 | SV4 |

## 3. Ràng buộc chung cho mọi RQ
- Cùng baseline đã reproduce, cùng split, cùng seed, cùng hardware protocol.
- Không full fine-tuning.
- Báo cáo: dataset split, seed, hardware, thời gian inference **và** thời gian PEFT fine-tuning.

## 4. Checklist
- [ ] Mỗi RQ có gap, metric, owner
- [ ] RQ2 (PEFT) là RQ trọng tâm — nhóm xác nhận
- [ ] Đối chiếu với `Hypotheses.md`
