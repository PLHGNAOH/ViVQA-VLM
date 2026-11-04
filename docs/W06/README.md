# Week 6 — Baseline → Method (Report 4)

Index tuần 6 của ViVQA-VLM. Mục tiêu W6: **chẩn đoán và chốt những gì ảnh hưởng tới tính công bằng của mọi so sánh về sau**, dựng đường ray huấn luyện QLoRA, và viết Methodology (R4). Nguyên tắc: *chẩn đoán dtype → ablation cấu hình → đóng băng cấu hình → mới huấn luyện*.

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Deliverables tuần này

| File | Vai trò |
|------|---------|
| [W06_Team_Report.md](W06_Team_Report.md) | Báo cáo tuần: việc đã làm, số liệu, blocker, kế hoạch W7 |
| [../03_report/R4_Methodology.md](../03_report/R4_Methodology.md) | **Deliverable chính** — Methodology bản tiếng Việt (7 mục + phụ lục quyết định còn mở) |

## Ba phát hiện chính

| # | Phát hiện | Con số then chốt |
|---|-----------|------------------|
| 1 | **Tràn số fp16** là nguyên nhân 23 output suy biến `!!!!` của baseline | fp16: 23 câu hỏng → fp32: 0; đối chứng 50 câu không thoái lui câu nào |
| 2 | **Độ phân giải là đánh đổi**, không phải cải thiện đơn thuần | 1.280 vs 512 token: ANLS +0,039 (CI95 [+0,012; +0,066]), EM +0,027 (McNemar p = 0,134), chậm 1,46× |
| 3 | **T4 bão hoà tính toán ở fp32** khi huấn luyện | 4,93 s/mẫu; tăng batch 2 → 6 không nhanh hơn; ~14,77 giờ/epoch ViVQA |

Bằng chứng PEFT-only: **7.372.800 / 3.761.995.776 = 0,196%** tham số được huấn luyện, khớp chính xác LoRA r=16 trên q/k/v/o × 36 lớp decoder ngôn ngữ.

## Artifacts

- **Dtype probe:** `../../experiments/W06_dtype_probe_seed42/` (`failed_ids.json`, `probe_summary.json`, `probe_breakdown.json`)
- **Ablation độ phân giải:** `../../experiments/W06_ablation_config_dev300_seed42/` (`ablation_summary.json`, `paired_A1_vs_A2.json`)
- **Smoke run QLoRA:** `../../experiments/W06_smoke_qlora_seed42/smoke_summary.json` + `train_log.json` của từng run
- **Script mới:** `../../src/adaptation/train_qlora.py`, `../../src/eval/find_degraded.py`, `../../src/eval/probe_breakdown.py`, `../../src/eval/compare_runs.py`
- **Notebook:** `../../notebooks/03_ablation_config_qwen25vl.ipynb`, `../../notebooks/04_qlora_qwen25vl.ipynb`

## Trạng thái

✅ Phần thực nghiệm cấu hình và R4 bản tiếng Việt hoàn tất. ⏳ Năm quyết định cấu hình được dời có chủ đích sang W7 (đo trên GPU L4) — xem cuối `W06_Team_Report.md`.
