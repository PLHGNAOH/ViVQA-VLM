# Token budget — Qwen/Qwen2.5-VL-3B-Instruct (token-budget-v1)

Context window (max_position_embeddings): **128,000** token · KV cache 36,864 byte/token (bf16) · token ảnh: min 200,704 px = 256, max 401,408 px = 512 · max_new_tokens 32

| Tập | Mode | Câu | Token ảnh p50 / max | % ảnh bị chặn 512 | Câu hỏi p50 / max | Dòng OCR p50 / max (khi có) | Input p50 / p95 / max | Chuỗi dài nhất | % context | KV cache dài nhất |
|---|---|---|---|---|---|---|---|---|---|---|
| vivqa_test | zero_shot | 3,001 | 345 / 506 | 1.5 | 11 / 32 | — | 425 / 532 / 587 | 619 | 0.484 | 21.76 MB |
| vivqa_test | ocr | 3,001 | 345 / 506 | 1.5 | 11 / 32 | 13 / 428 | 438 / 552 / 888 | 920 | 0.719 | 32.34 MB |
| vivqa_train | zero_shot | 10,799 | 345 / 506 | 1.56 | 11 / 37 | — | 425 / 530 / 598 | 630 | 0.492 | 22.15 MB |
| vivqa_train | ocr | 10,799 | 345 / 506 | 1.56 | 11 / 37 | 14 / 401 | 438 / 552 / 887 | 919 | 0.718 | 32.31 MB |
| vitextvqa_test | zero_shot | 10,028 | 480 / 512 | 57.69 | 11 / 40 | — | 554 / 578 / 598 | 630 | 0.492 | 22.15 MB |
| vitextvqa_test | ocr | 10,028 | 480 / 512 | 57.69 | 11 / 40 | 76 / 473 | 610 / 810 / 1026 | 1,058 | 0.827 | 37.2 MB |
| vitextvqa_train | zero_shot | 35,159 | 480 / 512 | 59.8 | 11 / 51 | — | 554 / 578 / 613 | 645 | 0.504 | 22.68 MB |
| vitextvqa_train | ocr | 35,159 | 480 / 512 | 59.8 | 11 / 51 | 76 / 462 | 613 / 815 / 1012 | 1,044 | 0.816 | 36.7 MB |

Nếu không chặn ở 512 token (độ phân giải gốc qua smart_resize của qwen_vl_utils):

| Tập | Token ảnh gốc p50 / p95 / max | Nếu chặn 1.280: p50 / max |
|---|---|---|
| vivqa_test | 345 / 460 / 529 | 345 / 529 |
| vivqa_train | 345 / 460 / 529 | 345 / 529 |
| vitextvqa_test | 609 / 15552 / 16335 | 609 / 1280 |
| vitextvqa_train | 625 / 15552 / 16335 | 625 / 1280 |

Kiểm chứng bằng đường processor đầy đủ: 1200 câu, lệch 0 câu (ĐẠT).
