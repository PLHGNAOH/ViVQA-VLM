# Week 8 — Improvement (OCR-enhanced prompting)

Index tuần 8 (tuần ảo) của ViVQA-VLM. Mục tiêu W8: **nối OCR-enhanced prompting vào pipeline và chạy ablation "có / không OCR"** trên ViTextVQA, đồng thời chạy QLoRA ViVQA nhiều epoch có chọn epoch trên val. Hồ sơ Review 2 (Team Report W07 + README W07) phải xong trước tuần thực 7 (19–25/10).

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Vì sao chọn OCR-enhanced prompting

Error audit W07 (200 lỗi EM của B2): trên ViTextVQA, nhóm lỗi đọc chữ chiếm phần lớn — sai OCR 28%, nhầm vùng chữ 23%, dấu thanh 8% (nguồn: `../W07/audit/W07_error_audit_final.json`). Đây là gap G2 (scene-text tiếng Việt). Ablation "with vs. without OCR-enhanced prompting" cũng là yêu cầu bắt buộc của đề cương (mục 7.3).

Rủi ro đã biết: Qwen2.5-VL vốn tự đọc chữ tốt (R0 TextVQA val VQA-Acc 79,60), nên text OCR ngoài có thể gây nhiễu. Kết quả âm vẫn hợp lệ nếu ablation đầy đủ.

## Kế hoạch tuần

| # | Việc | Trạng thái |
|---|------|-----------|
| 1 | Dọn tồn W07: commit `thuHoiNhanh` v2, `compute_error_audit.py` ghi JSON với `newline="\n"`, tạo `docs/W08/` | ✅ `5d161c4` |
| 2 | Hồ sơ Review 2: `docs/W07/W07_Team_Report.md` + `docs/W07/README.md` (bản TV); checklist Review2 điền SAU buổi review | ✅ `3f736eb` |
| 3 | Nối OCR vào `src/eval/run_baseline.py` và `src/adaptation/train_qlora.py` (`--mode ocr` đọc OCR cache) + selftest; script tập con train | ✅ |
| 4 | Tạo OCR cache ViTextVQA: notebook `10_ocr_cache_vitextvqa.ipynb` — Phần A: pilot 100 ảnh + dev300 × 2 backend + chỉ số answer-in-OCR; Phần B (sau bước 5): test2000 + tập con train | ✅ Phần A (13/10, T4, Paddle GPU) · ⏳ Phần B |
| 5 | Chọn backend OCR (`paddleocr` vs `paddledet_vietocr`) + mẫu prompt trên dev300: notebook `11_ocr_select_dev300_qwen25vl.ipynb` + `src/eval/select_ocr_config.py` (áp luật tự động) | ⏳ code xong, chờ chạy Colab L4 |
| 6 | Ablation OCR chỉ ở inference trên test2000: B1+OCR, B2+OCR | ⏳ |
| 7 | QLoRA ViVQA 2 epoch, eval val sau mỗi epoch để chọn epoch | ⏳ |
| 8 | Đóng tuần: tổng kết + bàn giao W08 | ⏳ |

## Quyết định đã chốt

| Ngày | Quyết định | Lý do |
|------|-----------|-------|
| 12/10/2026 | **B3 (QLoRA + OCR) train trên tập con cố định của ViTextVQA train (≈ 8.000 mẫu, seed 42)**; train lại **B2-sub** (không OCR) trên đúng tập con đó để so sánh cùng điều kiện. Không train B3 trên full train. | 1 epoch full ViTextVQA tốn 43.156 s ≈ 12,0 h trên L4 (`train_log.json` của B2) → full B3 ≈ 60–70 CU, vượt ngân sách Colab còn lại. Tập con giữ so sánh công bằng (chỉ khác yếu tố OCR) với ≈ 30 CU. |
| 13/10/2026 | Tập con **chọn theo ảnh** (`src/data/make_train_subset.py`): xáo tên ảnh bằng seed 42, lấy dần tới khi ≥ 8.000 câu | Mọi câu hỏi của một ảnh đi cùng nhau; ~3 câu/ảnh nên OCR cache cho tập con rẻ hơn ~3 lần. Số câu/ảnh thật + sha256: `../../experiments/W08_data_subsets/` |
| 13/10/2026 | Prompt `ocr` = prompt zero-shot + **đúng một dòng OCR** đặt trước câu hỏi; ảnh không có chữ → trùng hệt zero-shot | Bản trước W08 thêm cả chỉ dẫn "Trả lời ngắn gọn…" → ablation lẫn 2 biến. Train và eval dùng chung `prompt_for_sample` |
| 13/10/2026 | OCR chạy ở **phiên Colab riêng** với PaddleOCR 3.x; VietOCR cài `--no-deps` | `pip install paddleocr` chỉ còn bản 3.x (API khác 2.x). Gói VietOCR trên PyPI khoá `pillow==10.2.0`, sẽ hạ cấp Pillow của môi trường chạy VLM |

## Luật chọn cấu hình OCR (đặt TRƯỚC khi có số liệu — 13/10/2026)

Chọn trên **dev300** (không nhìn test), với B1 (zero-shot, cấu hình chốt W07) + OCR:

1. **Backend:** chọn backend có EM(B1 + OCR) cao hơn trên dev300. Nếu McNemar exact p ≥ 0,05 (không phân biệt được), chọn backend có `answer_in_ocr_pct` cao hơn (`experiments/W08_ocr_dev300_coverage/`). Nếu vẫn hoà, chọn backend nhanh hơn (s/ảnh).
2. **Template** (`v1` trung tính / `v2` báo OCR có thể sai): cùng luật như trên với backend đã chọn. Hoà → giữ `v1` (đơn giản hơn).
3. `max_chars = 600`, `min_conf = 0` giữ mặc định. Chỉ đổi nếu chênh `answer_in_ocr_pct` − `answer_in_ocr_capped_pct` > 5 điểm (cắt làm mất đáp án).
4. Cấu hình chọn xong thì **đóng băng** cho B1+OCR / B2+OCR trên test2000 và B3 (train). Không chỉnh thêm sau khi nhìn test.

`answer_in_ocr` (đáp án gold nằm nguyên cụm từ trong văn bản OCR) chỉ là **chẩn đoán / tiêu chí phụ**: ViTextVQA không có nhãn chữ thật nên không tính được CER của OCR.

## Kết quả bước 4 — OCR dev300 (266 ảnh, 300 câu)

Nguồn: `../../experiments/W08_ocr_dev300_coverage/ocr_coverage.json`. Môi trường: Colab T4, paddleocr 3.7.0, paddlepaddle 3.3.1 (GPU).

| Backend | Model đọc | % ảnh có chữ | dòng/ảnh | conf TB | đáp án ∈ OCR (giữ dấu) | đáp án ∈ OCR (bỏ dấu) | token recall | s/ảnh |
|---|---|---|---|---|---|---|---|---|
| `paddleocr` | PP-OCRv6_medium_rec (Latin) | 99,62 | 13,34 | 0,889 | 27,0% | **57,67%** | 0,490 | **0,34** |
| `paddledet_vietocr` | VietOCR vgg_transformer | 100,0 | 12,87 | 0,758 | **37,67%** | 48,67% | **0,573** | 0,95 |

Ghép cặp (đáp án ∈ OCR, giữ dấu): chỉ `paddleocr` có đáp án 28 câu, chỉ `paddledet_vietocr` có 60 câu, McNemar exact p = 0,0008. Cắt ở 600 ký tự không làm mất đáp án ở cả 2 backend.

**Phát hiện:** mô hình đọc Latin của PaddleOCR **làm rơi ký tự có dấu tiếng Việt** ("Kiểu Thẩm" → "Kiu Thm", "KIẾN THỨC" → "IENTHUC"): chênh giữ dấu / bỏ dấu tới 30,7 điểm. VietOCR giữ dấu tốt hơn (chênh 11,0 điểm) nhưng đọc sót mặt chữ nhiều hơn. Đây là bằng chứng trực tiếp cho gap G2 (OCR có sẵn chưa xử lý tốt dấu thanh tiếng Việt), khớp nhóm lỗi "sai dấu thanh" 8% trong error audit W07. Lựa chọn backend vẫn theo luật EM ở bước 5.

**Sự cố đã xử lý:** `paddlepaddle-gpu` (CUDA 12.6) ghi đè thư viện NCCL của torch (CUDA 13) → `import torch` hỏng (`undefined symbol: ncclCommWindowDeregister`). Notebook 10 mục 2 chụp và khôi phục thư viện `nvidia-*` của torch, kiểm torch + Paddle GPU ở tiến trình riêng, CPU fallback giữ torch nguyên (commit `833abd8`).

## Số liệu ngân sách (đo từ log W07, L4, bf16, 4-bit NF4)

| Hạng mục | Thời gian | Nguồn |
|----------|-----------|-------|
| QLoRA 1 epoch ViVQA (10.799 mẫu) | 10.917 s ≈ 3,03 h | `experiments/W07_B2_qlora_r16_qwen25vl_vivqa_ep1_bf16_L4_seed42/train_log.json` |
| QLoRA 1 epoch ViTextVQA (35.159 mẫu) | 43.156 s ≈ 12,0 h | `experiments/W07_B2_qlora_r16_qwen25vl_vitextvqa_ep1_bf16_L4_seed42/train_log.json` |
| Eval B2 ViTextVQA | ≈ 1,0 s/mẫu | `experiments/W07_B2_eval_qlora_r16_ep1_vitextvqa_test2000_bf16_L4_seed42/metrics.json` |
| Eval B1 ViVQA | ≈ 0,51 s/mẫu | `experiments/W07_B1_zeroshot_qwen25vl_vivqa_test_bf16_L4_seed42/metrics.json` |

## Artifacts

_(điền khi có)_

## Trạng thái

⏳ Đang làm.
