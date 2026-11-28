# Week 5 — Baseline reproduce hoàn chỉnh

Index tuần 5 của ViVQA-VLM. Mục tiêu W5: **baseline zero-shot chính thức, tái lập được** trên cả hai dataset của đề tài (ViVQA + ViTextVQA), dữ liệu đúng nguồn chính thức, metric và config chốt trước khi chạy test. *Chưa train* (QLoRA để W8–W9).

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Bản tiếng Anh cho thesis (R4 §2–3, §6 và R6 §2) sẽ chuyển từ bản này sau khi nhóm duyệt xong.

## Deliverables tuần này

| File | Vai trò |
|------|---------|
| [W05_Baseline_Report.md](W05_Baseline_Report.md) | Báo cáo baseline chi tiết: số liệu, protocol, dữ liệu, error analysis, so sánh W4↔W5, vấn đề mở |
| [W05_Team_Report.md](W05_Team_Report.md) | Báo cáo tuần theo khuôn nhóm (per-SV, DoD, blocker, kế hoạch W6) |

## Kết quả chính (zero-shot, Qwen2.5-VL-3B-Instruct, 4-bit NF4)

| Dataset | Tập đánh giá | EM | VQA-Acc | ANLS |
|---------|--------------|:--:|:-------:|:----:|
| ViVQA | test đầy đủ, 3.001 câu | **0.225** | 0.225 | **0.288** |
| ViTextVQA (chính thức) | tập con test 2.000 câu, seed 42 | **0.285** | 0.285 | **0.468** |

⚠️ Không so sánh với số W4 (EM 38.3% / ANLS 55.9%): khác cấu hình ở 4 điểm — xem `W05_Baseline_Report.md` §8.

## Artifacts

- Kết quả: `../../experiments/W05_zeroshot_qwen25vl_vivqa_test_seed42/`, `../../experiments/W05_zeroshot_qwen25vl_vitextvqa_test2000_seed42/` (`metrics.json` + `predictions.json`)
- Script: `../../src/data/prepare_vivqa.py`, `../../src/data/prepare_vitextvqa.py`, `../../src/eval/run_baseline.py`, `../../src/eval/metrics.py`
- Config chuẩn: `../../configs/qwen_lora.yaml` (commit `37da80a`)
- Notebook: `../../notebooks/01_baseline_zeroshot_qwen25vl.ipynb` (ViVQA), `../../notebooks/02_run_baseline_qwen25vl.ipynb` (runner chung)

## Trạng thái

✅ Baseline đã reproduce trên cả hai dataset, log tái lập đầy đủ. Việc còn mở xem cuối `W05_Team_Report.md`.
