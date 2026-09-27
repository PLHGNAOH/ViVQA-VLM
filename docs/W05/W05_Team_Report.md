# W05 — Team Weekly Report

Nguồn: `docs/W05/W05_Baseline_Report.md` + log trong `experiments/W05_*` + lịch sử commit. Khung theo `docs/templates/Weekly_Report_Template.md`. Không có số liệu ngoài log. Ô chưa có nguồn → `TODO (nhóm bổ sung)`.

## Metadata

| Trường | Nội dung |
|--------|----------|
| **Tuần** | W05 — Baseline reproduce hoàn chỉnh |
| **Nhóm** | ViVQA-VLM (4 SV) |
| **GVHD** | Assoc.Prof. Đặng Ngọc Minh Đức · Phạm Minh Trí |
| **Thời gian** | 2026-09-22 → 2026-09-27 |

## Mục tiêu

Baseline zero-shot **chính thức, tái lập được** trên cả ViVQA và ViTextVQA (master §9: W5 = baseline reproduce xong), với dữ liệu đúng nguồn chính thức, metric và config chốt trước khi chạy test.

## Đã làm

1. **Dữ liệu ViVQA gốc:** phát hiện dữ liệu trên máy là ViVQA-X (dataset khác) → lấy ViVQA từ repo gốc, viết `src/data/prepare_vivqa.py` (tải 10.328 ảnh COCO, cắt val 10% seed 42).
2. **Dữ liệu ViTextVQA chính thức:** xin quyền kho gated `minhquan6203/ViTextVQA`, phát hiện `test.json` chỉ là mẫu nộp Kaggle (dùng `test_gt.json`), viết lại `src/data/prepare_vitextvqa.py` (kiểm tra tự động, tập con 2.000 câu cố định).
3. **Metric chặt:** sửa `src/eval/metrics.py` (chữ thường thật sự, bỏ tiền tố `màu `, đổi số chỉ khi cả chuỗi là một số, không bảng đồng nghĩa), có test.
4. **Config chuẩn:** `max_pixels = 512·28·28`, float16 cho T4 (commit `37da80a`).
5. **Script chạy chung** `src/eval/run_baseline.py`: chạy theo lô, resume, chống trộn kết quả, ghi log tái lập — đáp ứng mục "inference script" của reproducibility package.
6. **Chạy baseline chính thức** trên cả hai dataset + error analysis (xem Baseline Report §6–§8).
7. **Dọn dẹp:** repo sạch (không file chưa commit, không nhánh thừa), Drive gọn (gom dữ liệu thừa, archive notebook W4), `.gitignore` chặn file cá nhân khỏi repo public.

### Per-SV

| SV | Vai trò | Đã làm | Evidence |
|----|---------|--------|----------|
| SV1 | VLM / Baseline | TODO (nhóm bổ sung) | |
| SV2 | Dataset / OCR | TODO (nhóm bổ sung) | |
| SV3 | PEFT / Training | Thực hiện các mục 1–7 ở trên | commit `37da80a` → `10cc915` |
| SV4 | Evaluation / Deployment | TODO (nhóm bổ sung) | |

### Definition of Done

| DoD item | Status | Evidence |
|----------|:------:|----------|
| Model chạy được trên Colab (4-bit NF4, T4) | ✅ | `metrics.json` meta (GPU, version) |
| Dữ liệu ViVQA đúng nguồn, đủ ảnh | ✅ | `data/vivqa/prep_report.json` — 10.328/10.328 ảnh |
| Dữ liệu ViTextVQA chính thức, đủ ảnh, không dùng nhầm mẫu Kaggle | ✅ | `data/vitextvqa_official/prep_report.json` — thiếu 0 ảnh |
| Metric EM / VQA-Acc / ANLS có test, chốt trước test | ✅ | `src/eval/metrics.py` |
| Baseline ViVQA test đầy đủ | ✅ | `experiments/W05_zeroshot_qwen25vl_vivqa_test_seed42/` |
| Baseline ViTextVQA (tập con 2.000) | ✅ | `experiments/W05_zeroshot_qwen25vl_vitextvqa_test2000_seed42/` |
| Log tái lập (split, seed, GPU, version, config, commit, thời gian) | ✅ | `meta` trong mỗi `metrics.json` |
| Error analysis sơ bộ | ✅ | Baseline Report §6–§7 |
| Baseline được GVHD xác nhận | ❌ | xem Blocker B1 |

## Kết quả / số liệu

| Mục | ViVQA | ViTextVQA |
|-----|-------|-----------|
| Tập đánh giá | test đầy đủ, 3.001 câu | tập con test 2.000 câu, seed 42 |
| Model / baseline | Qwen2.5-VL-3B-Instruct, zero-shot, 4-bit NF4 | như bên trái |
| **Exact Match (EM)** | **0.225** | **0.285** |
| **VQA Accuracy** | 0.225 | 0.285 |
| **ANLS** | **0.288** | **0.468** |
| BLEU / CIDEr (phụ) | không dùng (đáp án ngắn) | không dùng |
| Train time | không train | không train |
| Inference time | 2.532 s (0,84 s/mẫu) | 2.702 s (1,35 s/mẫu) |
| Hardware | Colab Tesla T4 15 GB | Colab Tesla T4 15 GB |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Owner | Bước tiếp |
|---|--------|-----|-------|-----------|
| B1 | Chưa có xác nhận của GVHD cho lựa chọn baseline Qwen2.5-VL-3B (mở từ W4) | Cao | TODO | Trình thầy kèm kết quả W5 |
| B2 | 23 output hỏng `!!!!` ở ViTextVQA — nghi tràn số fp16; rủi ro cho train QLoRA trên T4 | Cao | SV3 | Chạy lại 23 câu với compute fp32 (W6) |
| B3 | Số ViTextVQA giảm ~10 điểm so với W4 — nghi do `max_pixels` 512 và 4-bit | Trung bình | TODO | Ablation trên dev: 512 vs 1024 token, 4-bit vs fp16 (W6) |
| B4 | ViTextVQA không có nhãn loại câu hỏi → chưa phân tích lỗi theo loại được | Trung bình | TODO | Gán loại câu hỏi (heuristic hoặc tay trên mẫu) |
| B5 | Val ViVQA là tự cắt (bản gốc không có) | Thấp | — | Đã ghi rõ trong protocol (seed 42, 10%) |

## Kế hoạch tuần sau (W06 — Baseline → Method, Report 4)

- [ ] Kiểm chứng vấn đề fp16 (B2) — **ưu tiên**, vì ảnh hưởng trực tiếp việc train QLoRA
- [ ] Ablation độ phân giải / lượng tử hoá trên dev (B3), quyết định config cuối
- [ ] Prompt optimization trên VAL/dev (siết trả lời ngắn, tiếng Việt, không lặp câu hỏi)
- [ ] Dựng pipeline OCR (PaddleOCR / VietOCR) cho ViTextVQA
- [ ] Gán loại câu hỏi ViTextVQA (B4)
- [ ] Trình GVHD xác nhận baseline (B1)
- [ ] Viết Methodology (Report 4) — bản tiếng Việt trước
- [ ] Chuẩn bị dàn ý Review 2 (W7): baseline + phương pháp đề xuất + kết quả sơ bộ

Phân công per-SV: TODO (nhóm thống nhất).

## Link experiment log

- `experiments/W05_zeroshot_qwen25vl_vivqa_test_seed42/` (`metrics.json`, `predictions.json`)
- `experiments/W05_zeroshot_qwen25vl_vitextvqa_test2000_seed42/` (`metrics.json`, `predictions.json`)
- Reproduce: xem `W05_Baseline_Report.md` §11
