# Week 4 — Reproduce Baseline (khởi động)

Index tuần 4 của ViVQA-VLM. Mục tiêu W4: hiểu kiến trúc VLM · dựng data pipeline · **baseline zero-shot đo được** trên ViTextVQA. *Chưa train* (QLoRA để sau).

## Deliverables tuần này
- `W04_Team_Report.md` — báo cáo tuần (tóm tắt + trạng thái).
- `W04_Baseline_Report.md` — báo cáo baseline chi tiết (số, protocol, phát hiện, failure cases, lever cải tiến → W5).
- (xuyên tuần) `../Glossary.md` — sổ tay thuật ngữ.
- (xuyên tuần) `../Review_W01-W04_and_Defense_Prep.md` — ôn tập W1–W4 + luyện bảo vệ.

## Kết quả chính
- Baseline: **Qwen2.5-VL-3B-Instruct**, zero-shot.
- **EM 38.3% / ANLS 55.9%** (ViTextVQA mirror, test, n=300, seed 42).

## Artifacts (log thí nghiệm)
- `../../experiments/W04_zeroshot_qwen25vl_seed42/` — `env.json`, `zeroshot_sample.json`, `zeroshot_vitextvqa_n20.json`, `zeroshot_vitextvqa_n300_seed42.json`.
- `../../notebooks/01_zeroshot_baseline.ipynb` — notebook chạy (runner).

## Trạng thái
✅ Đủ điều kiện sang W5 (baseline đo được + tái lập). Việc còn mở xem cuối `W04_Team_Report.md`.
