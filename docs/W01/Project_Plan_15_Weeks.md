# Project Plan — 15 tuần (ViVQA-VLM)

> Owner: cả nhóm · Cập nhật: 2026-09-14 · Nguồn ràng buộc: Master Instruction (mục 4, 7, 9)
> Ngày là **dự kiến** (mốc bắt đầu tuần, thứ Hai); nhóm điều chỉnh theo lịch khoa nhưng **không đổi thứ tự milestone**.

## 1. Mục tiêu & đóng góp

- **Bài toán:** ảnh tiếng Việt + câu hỏi tiếng Việt → VLM → câu trả lời tiếng Việt.
- **Đóng góp bắt buộc:** kiểm chứng bằng thực nghiệm ≥ 1 cải tiến (PEFT / OCR-prompt / RAG / prompt-opt) so với **đúng 1** baseline đã reproduce, **không full fine-tuning**, có ablation + error analysis + hallucination analysis.

## 2. Phân công cố định

| SV | Vai trò | Deliverable cá nhân xuyên suốt |
|----|---------|--------------------------------|
| SV1 | VLM / Baseline | Reproduce 1 trong BLIP-2 / Florence-2 / Qwen2.5-VL; `src/baseline/` |
| SV2 | Dataset / OCR | ViVQA, ViTextVQA, PaddleOCR/VietOCR, split + seed; `data/`, `src/data/` |
| SV3 | PEFT / Training | LoRA/QLoRA 4/8-bit, prompting, RAG; `src/adaptation/`, `adapters/`, `configs/` |
| SV4 | Evaluation / Deployment | EM, VQA-Acc, ANLS, hallucination, prototype; `src/eval/`, `src/prototype/` |

## 3. Lịch 15 tuần

| Tuần | Objective | Deliverable | Milestone | Owner chính | Ngày dự kiến (bắt đầu) |
|------|-----------|-------------|-----------|-------------|--------------|
| W01 | Orientation & planning | Plan, paper review cá nhân (SV1–SV4), khung docs | | Cả nhóm | ___ |
| W02 | Literature & dataset | Literature matrix, dataset analysis, gap, RQ, hypothesis | | Cả nhóm | ___ |
| **W03** | Review 1 | Slides + checklist; chốt baseline | **REVIEW 1** | SV4 điều phối | **___** |
| W04 | Baseline setup | Env (Colab/GPU trường), pin phiên bản, baseline zero-shot chạy được; **pilot QLoRA trên 200 mẫu** (`experiments/W01_Pilot/`) | | SV1 + SV3 | ___ |
| W05 | Baseline reproduce | Số liệu zero-shot EM/VQA-Acc/ANLS trên test split; format-prompt (H7) | | SV1 + SV4 | ___ |
| W06 | Method | Methodology (Report 4): recipe PEFT, ablation plan | | SV3 + SV1 | ___ |
| **W07** | Review 2 | Baseline + method + preliminary results | **REVIEW 2** | SV4 điều phối | **___** |
| W08 | Improvement v1 | QLoRA v1 (A1, A2); OCR-prompting v1 (A6) | | SV3, SV2 | ___ |
| W09 | Improvement final | Rank sweep (A3), 4/8/16-bit (A4), adapter chung (A5); method freeze | | SV3 | ___ |
| W10 | Experiments | Kết quả chính, 3 seed | | SV4 + SV3 | ___ |
| W11 | Full evaluation | Full metrics + thời gian train/infer + VRAM | | SV4 | ___ |
| W12 | Analysis | Ablation + error analysis theo loại câu hỏi + hallucination (A9) | | Cả nhóm | ___ |
| **W13** | Faculty Review | Package nghiên cứu gần hoàn chỉnh | **FACULTY REVIEW** | SV4 điều phối | **___** |
| W14 | Prototype / Thesis | Web demo (upload ảnh, hỏi tiếng Việt, confidence, OCR evidence) + draft thesis | | SV4 + cả nhóm | ___ |
| **W15** | Final | Defense + reproducibility package (code, adapter, config, log, README) | **FINAL DEFENSE** | Cả nhóm | **___** |

## 4. Đường găng

```text
W01–W02 literature/data  →  W03 Review 1 (chốt baseline + gap)
W04–W05 baseline         →  W06 method  →  W07 Review 2
W08–W09 PEFT/OCR/RAG     →  W10–W11 eval  →  W12 analysis  →  W13 Faculty
W14 prototype/thesis     →  W15 Final Defense
```

Không đảo milestone. Không cải tiến trước khi có baseline protocol.

## 5. Ngân sách tính toán

| Hạng mục | Giá trị dự kiến |
|----------|-----------------|
| Môi trường | Colab Pro/Premium (T4 16 GB / L4 24 GB / A100 40 GB) hoặc GPU server trường |
| Baseline (1 model) | **Qwen2.5-VL-3B** (đề xuất) · BLIP-2 đối chứng · Florence-2 ablation (chốt W03) |
| PEFT | QLoRA NF4, LoRA r=16 (sweep 8–64), all-linear |
| Thời gian train dự kiến / epoch | _đo ở pilot W04_ |
| Thời gian infer / sample | _đo ở W05_ |
| VRAM đỉnh | _đo ở pilot W04_ |

> **Ràng buộc cứng (PEFT-only):** CHỈ LoRA hoặc QLoRA (4-bit / 8-bit). KHÔNG full fine-tuning. Cùng split, cùng seed, cùng hardware protocol với baseline.

## 6. Rủi ro (≥ 5)

| Risk | Impact | Mitigation | Backup | Owner |
|------|--------|-----------|--------|-------|
| Model quá lớn / OOM | High | QLoRA 4-bit, giảm max_pixels, batch 1 | Baseline nhỏ hơn (3B) | SV1 + SV3 |
| GPU không đủ / Colab ngắt | High | Checkpoint mỗi N step; Colab Premium; queue GPU trường | Subset eval có ghi N | SV3 |
| QLoRA train chậm | Medium | Subset train, 8-bit/BF16 trên A100 | Prompt-opt không train | SV3 |
| Dataset lỗi / licence | Medium | Chỉ dùng split công bố, preprocessing kỹ | Subset thủ công có ghi | SV2 |
| Baseline khó reproduce | High | Đổi 1 trong 3 ứng viên | Zero-shot + paper metric | SV1 |
| OCR kém | Medium | Đổi PaddleOCR ↔ VietOCR | Loại câu hỏi không text-in-image | SV2 |
| Train không hội tụ / quên tiếng Việt | High | Giảm LR, thêm text-only vi, kiểm tra tokenizer | Prompting + RAG | SV3 |
| Demo khó deploy | Medium | Gradio trên Colab | Video backup | SV4 |

## 7. Definition of Done của plan
- [x] 4 milestone gắn ngày dự kiến (W03 ___ · W07 ___ · W13 ___ · W15 ___)
- [x] Mỗi SV có deliverable cá nhân theo tuần (mục 2 + cột Owner chính)
- [x] PEFT-only được cả nhóm xác nhận (2026-09-14)
- [ ] Pilot experiment W04 định nghĩa trong `experiments/W01_Pilot/README.md` (thực hiện ở W04)
