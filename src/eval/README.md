# src/eval — EM, VQA Accuracy, ANLS, hallucination

Metric chính: Exact Match, VQA Accuracy, ANLS. BLEU/CIDEr phụ.

Mọi số liệu phải gắn run log: split, seed, hardware, time.

## Script (W07)

| Script | Việc | Selftest |
|---|---|---|
| `run_baseline.py` | Suy luận theo lô, resume được, chữ ký run. `--adapter` (B2), `--variant qwen2vl7b_paper/qwen2vl7b_b1cfg` (R1/R1b, Qwen2-VL-7B), `--data_file` (vd full test ViTextVQA) | `--selftest` |
| `compare_runs.py` | So ghép cặp 2 run: McNemar exact (EM) + CI95 bootstrap (ANLS) | — |
| `paper_consistency.py` | Đối chiếu số công bố của paper ViTextVQA có tính sai số mẫu: Table 7 (100 câu) và ViTextBLIP-2 (Table 6, full test) | `--selftest` |
| `stratify_by_overlap.py` | Tách kết quả theo mức trùng train (rò rỉ) | `--selftest` |
| `run_textvqa_r0.py` | R0: tái lập TextVQA val của Qwen2.5-VL-3B (kiểm chứng pipeline) | `--selftest` |
