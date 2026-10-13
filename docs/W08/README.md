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
| 4 | Tạo OCR cache ViTextVQA: notebook `10_ocr_cache_vitextvqa.ipynb` — Phần A: pilot 100 ảnh + dev300 × 2 backend + chỉ số answer-in-OCR; Phần B (sau bước 5): test2000 + tập con train | ✅ 13/10 — Phần A (dev300, 2 backend) + Phần B (VietOCR: test2000 1.486 ảnh + train 2.670 ảnh, 0 lỗi, 0,97 s/ảnh) |
| 5 | Chọn backend OCR (`paddleocr` vs `paddledet_vietocr`) + mẫu prompt trên dev300: notebook `11_ocr_select_dev300_qwen25vl.ipynb` + `src/eval/select_ocr_config.py` (áp luật tự động) | ✅ 13/10 — chốt **VietOCR · v1** |
| 6 | Ablation OCR chỉ ở inference trên test2000: B1+OCR, B2+OCR — notebook `12_ocr_ablation_test2000_qwen25vl.ipynb` | ⏳ code xong, chờ chạy Colab L4 |
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

## Kết quả bước 5 — chọn cấu hình OCR trên dev300 (L4, bf16, 4-bit, seed 42)

Nguồn: `../../experiments/W08_ocr_select_dev300_L4_seed42/ocr_selection.json` + 5 run `W08_B1*_dev300_*`.

| Run | EM | ANLS | token-F1 (phụ) | Chỉ B1 đúng / chỉ OCR đúng | McNemar p (so B1) | s/câu |
|---|---|---|---|---|---|---|
| B1 zero-shot | 0,3267 | 0,4615 | 0,5327 | — | — | 0,859 |
| B1 + VietOCR · v1 **(chốt)** | 0,3267 | 0,4600 | 0,5385 | 16 / 16 | 1 | 0,878 |
| B1 + VietOCR · v2 | 0,3233 | 0,4696 | 0,5434 | 17 / 16 | 1 | 0,867 |
| B1 + PaddleOCR · v1 | 0,3200 | 0,4606 | 0,5384 | 20 / 18 | 0,871 | 0,853 |
| B1 + PaddleOCR · v2 | 0,3067 | 0,4700 | 0,5410 | 23 / 17 | 0,43 | 0,858 |

**Áp luật (đúng như ghi trước):** Luật 1: EM v1 hai backend không phân biệt được (p = 0,856) → xét answer-in-OCR: VietOCR 37,67% > PaddleOCR 27,0% → **VietOCR**. Luật 2: v1 = 0,3267 vs v2 = 0,3233, p = 1 → **v1**. Luật 3: cắt 600 ký tự mất 0 điểm → giữ `max_chars 600`, `min_conf 0`. Cấu hình này **đóng băng** cho test2000 và B3.

**Phát hiện:** chèn OCR vào prompt lúc suy luận **không cải thiện B1 zero-shot** (EM giữ nguyên, 16 câu được sửa đúng nhưng 16 câu bị làm sai). Phù hợp rủi ro đã nêu: Qwen2.5-VL vốn tự đọc chữ tốt (R0 TextVQA 79,60). Đây là kết quả âm hợp lệ cho ablation "có / không OCR". Phép thử chính vẫn là **B3 (train cùng OCR) so với B2-sub**: model zero-shot chưa được dạy dùng dòng OCR; train cùng OCR có thể dạy model dùng OCR để sửa dấu. **Phân tích lật (dev300, VietOCR · v1 so với B1):** 167/300 câu trả lời **đổi** khi thêm OCR → model **có dùng** dòng OCR, không bỏ qua. 16 câu OCR sửa đúng, chủ yếu khi model tự đọc sai: số điện thoại (`0914.414.227` → `0949.002.339`), tên riêng (`Lê Xuân Khanh` → `Lê Xuân Khoa`), dấu (`Dây nổ` → `Dây nịt`). 16 câu OCR làm sai do model **chép nguyên lỗi OCR** (`CHĂN` → `CHÂN`, `Thấm` → `Thắm`, `Số` → `Sô`), **dán nguyên chuỗi OCR** (`Thăng Long | 0919467868 | …`) hoặc trả lời dài hơn. 31 câu chỉ khác nhau ở dấu. → Model zero-shot dùng OCR **không phân biệt** (không biết khi nào tin ảnh, khi nào tin OCR, không biết trả lời ngắn) — đúng thứ train cùng OCR (B3) có thể dạy.

## Luật kết luận bước 6 và B3 (đặt TRƯỚC khi chạy test — 13/10/2026)

Cấu hình OCR dùng cho mọi run dưới đây = cấu hình đóng băng ở bước 5 (VietOCR · v1 · 600 · 0). Tất cả trên **test2000** (seed 42), L4, bf16, 4-bit, cùng prompt zero-shot ± đúng 1 dòng OCR.

| So sánh | Vai trò | "Cải thiện" khi |
|---|---|---|
| B1+OCR vs B1 | Ablation bắt buộc "có / không OCR" (zero-shot) | EM tăng **và** McNemar exact p < 0,05 |
| B2+OCR vs B2 | Chẩn đoán: B2 chưa từng thấy dòng OCR khi train (lệch train/test) | như trên — **không** dùng làm kết quả cải tiến |
| **B3 vs B2-sub** | **Phép thử chính:** cùng tập con train 8.000 câu (`train_subset8000_byimage_seed42`), cùng siêu tham số QLoRA (r 16, 1 epoch), chỉ khác B3 có dòng OCR khi train **và** khi test | EM tăng **và** McNemar exact p < 0,05 |
| B3 vs B2 (full train) | Chỉ báo cáo: tập con 8.000 câu + OCR có bù được 77% dữ liệu train bị bỏ không | — (không kết luận cải tiến) |

ANLS luôn báo cáo kèm CI95 bootstrap ghép cặp (10.000 lần, seed 42). Kết quả âm (p ≥ 0,05) vẫn được báo cáo đầy đủ trong bảng ablation.

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
