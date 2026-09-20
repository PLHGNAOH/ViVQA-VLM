# W02 — Literature & dataset (index)

Tuần 2: **Literature matrix + Dataset analysis**. Canonical dir: `docs/W02/`. Không phải tuần train.

## Deliverable

| File | Vai trò |
|------|---------|
| [Literature_Matrix.md](../02_references/Literature_Matrix.md) | canonical sống tại `docs/02_references/Literature_Matrix.md` (W02 chỉ còn pointer) |
| [Dataset_Analysis.md](Dataset_Analysis.md) | thống kê thật từ JSON, seed 42 |
| [Research_Gap.md](Research_Gap.md) | G1–G4 evidence-linked, focus G3 |
| [Research_Questions.md](Research_Questions.md) | RQ1–RQ4 + sub-RQ (W3-PREP, hợp nhất từ weekly) |
| [Hypotheses.md](Hypotheses.md) | H1–H8 (W3-PREP, hợp nhất từ weekly) |
| [Candidate_Methods.md](Candidate_Methods.md) | baseline table, recipe NF4, A1–A9 |
| [W02_Team_Report.md](W02_Team_Report.md) | báo cáo tuần (từ WEEK2_STATUS + template) |
| [WEEK2_STATUS.md](WEEK2_STATUS.md) | self-audit DoD, 90% |

## Stats & script

- Output: `experiments/W02_dataset_stats/` (`vivqa` / `vitextvqa` / `vivqa_x` stats + `samples/` 6 loại câu hỏi × 3 dataset).
- Script: `src/data/analyze_dataset.py`

```
python src/data/analyze_dataset.py
```

Run of record: 2026-09-14, seed 42. Optional: `--data_root data`, `--top_k 30`, `--only vivqa vitextvqa`.
