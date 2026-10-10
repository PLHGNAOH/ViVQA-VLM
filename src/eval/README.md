# src/eval — EM, VQA Accuracy, ANLS, hallucination

Metric chính: Exact Match, VQA Accuracy, ANLS. BLEU/CIDEr phụ.

Mọi số liệu phải gắn run log: split, seed, hardware, time.

## Script (W07)

| Script | Việc | Selftest |
|---|---|---|
| `run_baseline.py` | Suy luận theo lô, resume được, chữ ký run. `--adapter` (B2), `--variant qwen2vl7b_paper/qwen2vl7b_b1cfg` (R1/R1b, Qwen2-VL-7B), `--data_file` (vd full test ViTextVQA), `--few_shot_k 5` (đối chứng few-shot, ví dụ chỉ chữ từ train) | `--selftest` |
| `compare_runs.py` | So ghép cặp 2 run: McNemar exact (EM) + CI95 bootstrap (ANLS) | — |
| `paper_consistency.py` | Đối chiếu số công bố của paper ViTextVQA có tính sai số mẫu: Table 7 (100 câu) và ViTextBLIP-2 (Table 6, full test) | `--selftest` |
| `answer_format.py` | Chẩn đoán định dạng: số từ, % chứa đáp án gold, % chứa gold nhưng EM sai (post-hoc) | `--selftest` |
| `stratify_by_overlap.py` | Tách kết quả theo mức trùng train (rò rỉ) | `--selftest` |
| `run_textvqa_r0.py` | R0: tái lập TextVQA val của Qwen2.5-VL-3B (kiểm chứng pipeline) | `--selftest` |

Duyệt tay (W07): `python -m src.data.make_audit_sheet` tạo `audit/W07/W07_audit_400.xlsx` + ảnh trên Drive (data_audit 200 dòng train, error_audit 200 câu B2 sai).
