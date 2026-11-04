# W15 · Deck bảo vệ cuối + hoàn thiện code/demo + bắt đầu thesis

Bắt đầu: 02/11/2026 (tuần thực 9). Mốc thật phía trước: hội đồng cấp Khoa = tuần thực 13 (30/11–06/12/2026), bảo vệ cuối = tuần thực 15 (14–20/12/2026).

## Đầu vào của tuần
- Deck Review 2 đúng bản đã trình bày ngày 29/10/2026 (xuất từ Canva, TV + EN, 27 slide, notes khôi phục): `docs/04_slides/review2/`.
- Góp ý Review 2 (từ bản ghi âm): `docs/W15/Review2_Feedback.md`.
- Colab: 207,21 CU (02/11/2026, sau gia hạn).

## Thứ tự công việc (Hoàng chọn)
1. Deck hoàn chỉnh từ đầu tới cuối cho bảo vệ: **bản tiếng Việt trước**, rồi bản tiếng Anh từ bản tiếng Việt đã sửa xong.
   - Sửa Part A (Review 1) và Part B (Review 2) theo góp ý G1–G8.
   - Thêm Part C: hallucination (POPE-vi), phân tích lỗi, EXIF + độ nhạy, adapter dùng chung, prototype v2.1 (ảnh chụp thật), video demo rút gọn.
   - Thêm Part D: kết luận, đóng góp, giới hạn, hướng phát triển; phụ lục cho hỏi đáp.
2. Code và demo chạy trơn tru (notebook 28, video dự phòng, gói tái lập).
3. Bắt đầu thesis bản tiếng Việt.

## Trạng thái
| Việc | Trạng thái |
|---|---|
| Đặt deck Review 2 (bản trình bày) vào repo | xong khi commit setup W15 |
| Danh sách góp ý Review 2 | dùng bản chép lời (Hoàng duyệt 04/11: làm theo bản nghe được, ưu tiên G1 token) |
| Checklist AI_Capstone_Review123 sheet Review2 + mail | chưa |
| Deck TV hoàn chỉnh | nháp 1 (50 slide) chờ Hoàng duyệt |
| Deck EN | chưa |
| Code góp ý Review 2 (G1 token, G2–G3 chuẩn hoá/tách từ) | xong — G1 chạy notebook 29 ngày 04/11 (CPU, 16,5 phút) |
| Code + demo cuối | chưa |
| Thesis TV | chưa |

## Code trả lời góp ý Review 2

| Góp ý | Script | Chạy ở đâu | Kết quả |
|---|---|---|---|
| G1 — số token ảnh / câu hỏi / OCR so với sức chứa model | `src/eval/token_budget.py` + `notebooks/29_token_budget_qwen25vl.ipynb` | Colab **CPU** | `experiments/W15_token_budget/` (Cursor chép từ Drive) |
| G5 — vì sao 512 token ảnh, vì sao bf16 | (không chạy mới) | — | đã có: `experiments/W06_ablation_config_dev300_seed42/`, `experiments/W07_config_freeze_L4_seed42/` |
| G2 — tên kỹ thuật chuẩn hoá + định lượng | `src/eval/metric_tokenization.py` | máy bất kỳ (đọc `predictions.json`) | `experiments/W15_metric_tokenization/` |
| G3 — từ ghép / tách từ khi tính metric | `src/eval/metric_tokenization.py` (pyvi) | như trên | như trên |

**G1 — cách đo:** token ảnh theo đúng đường suy luận (`qwen_vl_utils.fetch_image` → image processor của Qwen, min 200.704 px / max 401.408 px = 256 / 512 token, công thức (H/14)·(W/14)/4); token chữ bằng tokenizer Qwen trên chuỗi chat template thật; token đáp án như `train_qlora`. 150 câu ngẫu nhiên mỗi tập × mode (seed 42) chạy lại bằng nguyên `processor(text, images)`; lệch thì script tự đo lại bằng processor. Context window và KV cache đọc từ `config.json` (revision `66285546…` như W07). Đây là phép **đo mô tả**, không có giả thuyết hay luật quyết định.

**G5 — bằng chứng có sẵn (không chạy thêm):**
- W06 (dev300, zero-shot, fp32, T4): 512 token EM 33,3 / ANLS 46,5 · 1.280 token EM 36,0 / ANLS 50,4; McNemar p = 0,13 (EM không khác biệt), ΔANLS +3,9 [+1,2; +6,6]; suy luận chậm 1,46×.
- W07 (L4, bf16): train ViTextVQA 1.280 token chậm 1,91× (11,1 → 21,2 giờ/epoch), VRAM 6,71 → 10,84 GB → chọn 512 theo luật (chọn 1.280 nếu ≤ 1,5×). ViVQA: chỉ 1% ảnh vượt 512 token (ảnh COCO nhỏ).
- bf16 vs fp16 vs fp32 (probe 73 câu, L4): fp16 tràn số 23/73 câu hỏng; bf16 0 câu hỏng, nhanh hơn fp32 1,71×, thoái lui 1/50 câu đối chứng.

**G2–G3 — kết quả** (`experiments/W15_metric_tokenization/summary.md`):
- Chuẩn hoá áp **đối xứng** cho dự đoán và gold, giữ dấu thanh. Bậc có tác dụng lớn nhất: ViTextVQA = regex bỏ dấu câu + gộp khoảng trắng (25,7% gold đổi: gold viết tách dấu câu kiểu "0983 . 651 888"); ViVQA = bỏ tiền tố "màu " (20,9% gold) và số chữ → chữ số (14,8%). Model QLoRA gần như không phụ thuộc chuẩn hoá (C3: EM 76,64 ở mọi bậc; B3-full 51,80 → 53,35); zero-shot phụ thuộc nhiều (B1 ViVQA 1,70 → 22,09).
- 41,8% đáp án gold ViTextVQA và 20,9% ViVQA có từ ghép (pyvi). EM không đổi khi tách từ (so cả chuỗi); ANLS theo ký tự. token-F1 theo từ thấp hơn theo âm tiết 1–3 điểm nhưng **mọi chênh lệch giữ nguyên**: B2 → B3-full +2,57 [+2,11; +3,02] (âm tiết) vs +2,60 [+2,12; +3,06] (từ); thứ hạng B1 < B2 < B3-full không đổi.

**G1 — kết quả** (`experiments/W15_token_budget/summary.md`, notebook 29, 04/11/2026; kiểm chứng 1.200 câu bằng đường processor đầy đủ: lệch 0):

| Tập (prompt) | Token ảnh p50 / max | % ảnh bị chặn ở 512 | Dòng OCR p50 / max (khi có) | Input p50 / p95 / max | Chuỗi dài nhất | % context |
|---|---|---|---|---|---|---|
| ViVQA test (không OCR, C3) | 345 / 506 | 1,5 | — | 425 / 532 / 587 | 619 | 0,48 |
| ViTextVQA test (không OCR, B2) | 480 / 512 | 57,7 | — | 554 / 578 / 598 | 630 | 0,49 |
| ViTextVQA test (OCR, B3-full) | 480 / 512 | 57,7 | 76 / 473 | 610 / 810 / 1.026 | 1.058 | 0,83 |
| ViTextVQA train (OCR, B3-full) | 480 / 512 | 59,8 | 76 / 462 | 613 / 815 / 1.012 | 1.044 | 0,82 |

- Context window 128.000 token (config.json, revision `66285546…`); KV cache 36.864 byte/token (36 lớp × 2 KV-head × head_dim 128 × K,V × bf16). Chuỗi dài nhất của nhóm = 0,83% sức chứa, KV cache 37 MB.
- Token ảnh chiếm 73–87% input (trung vị). Nếu không chặn, ảnh ViTextVQA cần trung vị 609, p95 15.552, tối đa 16.335 token ảnh → chuỗi ≈ 16,9 nghìn token = 13% sức chứa: giới hạn thật là VRAM / thời gian train (1.280 token đã chậm 1,91×, W07), không phải context window.
- ViVQA: 14,5% ảnh được nâng lên mức tối thiểu 256 token; chỉ 1,5% chạm 512. Dòng OCR có ở 44% câu ViVQA (chữ trên ảnh COCO, trung vị 13 token) và 99,7% câu ViTextVQA.
- Đáp án train ViTextVQA: trung vị 6 token, p99 28, dài nhất 64 → giới hạn sinh 32 token gần như không cắt đáp án (< 1% đáp án train dài hơn 28 token).
