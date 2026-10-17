# W02 dataset stats

Descriptive stats for ViVQA, ViTextVQA, and related ViVQA-X. Produced by `src/data/analyze_dataset.py` (NFC, seed **42**, no fabricated numbers; failed loads write `PENDING_RUN`).

```
python src/data/analyze_dataset.py
```

Outputs: `{vivqa,vitextvqa,vivqa_x}_stats.json` and `samples/<dataset>__<qtype>.json` (this directory). Optional: `--data_root data`, `--top_k 30`, `--only vivqa vitextvqa`. Run of record: 2026-09-14.
