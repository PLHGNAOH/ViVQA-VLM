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
| 6 | Ablation OCR chỉ ở inference trên test2000: B1+OCR, B2+OCR — notebook `12_ocr_ablation_test2000_qwen25vl.ipynb` | ✅ 13/10 — không khác biệt có ý nghĩa (cả B1 lẫn B2) |
| 7 | **B3 (QLoRA + OCR) vs B2-sub (QLoRA, không OCR)** trên tập con 8.000 câu, eval test2000 — notebook `13_qlora_ocr_b3_vs_b2sub_qwen25vl.ipynb` (thay QLoRA ViVQA 2 epoch, dời sang W9) | ✅ 14/10 — **B3 cải thiện có ý nghĩa** (EM +3,70 điểm, p = 1,7 × 10⁻⁶) |
| 8 | Đóng tuần: tổng kết + bàn giao W08 | ⏳ |

## Quyết định đã chốt

| Ngày | Quyết định | Lý do |
|------|-----------|-------|
| 12/10/2026 | **B3 (QLoRA + OCR) train trên tập con cố định của ViTextVQA train (≈ 8.000 mẫu, seed 42)**; train lại **B2-sub** (không OCR) trên đúng tập con đó để so sánh cùng điều kiện. Không train B3 trên full train. | 1 epoch full ViTextVQA tốn 43.156 s ≈ 12,0 h trên L4 (`train_log.json` của B2) → full B3 ≈ 60–70 CU, vượt ngân sách Colab còn lại. Tập con giữ so sánh công bằng (chỉ khác yếu tố OCR) với ≈ 30 CU. |
| 13/10/2026 | Tập con **chọn theo ảnh** (`src/data/make_train_subset.py`): xáo tên ảnh bằng seed 42, lấy dần tới khi ≥ 8.000 câu | Mọi câu hỏi của một ảnh đi cùng nhau; ~3 câu/ảnh nên OCR cache cho tập con rẻ hơn ~3 lần. Số câu/ảnh thật + sha256: `../../experiments/W08_data_subsets/` |
| 13/10/2026 | Prompt `ocr` = prompt zero-shot + **đúng một dòng OCR** đặt trước câu hỏi; ảnh không có chữ → trùng hệt zero-shot | Bản trước W08 thêm cả chỉ dẫn "Trả lời ngắn gọn…" → ablation lẫn 2 biến. Train và eval dùng chung `prompt_for_sample` |
| 13/10/2026 | GPU còn lại của W8 dùng cho **B3 + B2-sub** (phép thử chính của hướng OCR); **QLoRA ViVQA 2 epoch dời sang W9** | B3 vs B2-sub trả lời câu hỏi nghiên cứu chính; ViVQA 2 epoch chỉ tinh chỉnh B2, không liên quan OCR. Colab còn 146,75 CU (13/10, 20:27) |
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

## Kết quả bước 6 — ablation "có / không OCR" trên test2000 (L4, bf16, 4-bit, seed 42)

Nguồn: `../../experiments/W08_ocr_ablation_test2000_L4_seed42/` (`ocr_ablation_summary.json`, 2 file `paired_*.json`) + 2 run `W08_B1ocr_…_test2000_…`, `W08_B2ocr_…_test2000_…`. Cấu hình OCR đóng băng từ bước 5 (VietOCR · v1 · 600 · 0); 99,85% câu có chữ OCR, trung bình 144 ký tự, 3,55% bị cắt.

| Run | EM | ANLS | token-F1 (phụ) | s/câu |
|---|---|---|---|---|
| B1 zero-shot (W07) | 0,2880 | 0,4753 | 0,5377 | 0,862 |
| B1 + OCR | 0,2935 | 0,4802 | 0,5462 | 0,906 |
| B2 QLoRA 1 epoch (W07) | 0,4970 | 0,7191 | 0,7101 | 1,011 |
| B2 + OCR (chỉ lúc suy luận) | 0,5070 | 0,7214 | 0,7124 | 0,997 |

| So sánh | Chỉ base đúng / chỉ +OCR đúng | McNemar p | ΔANLS (CI95) | Kết luận theo luật |
|---|---|---|---|---|
| B1 → B1 + OCR | 132 / 143 | 0,547 | +0,0050 [−0,0097; +0,0203] | Không khác biệt có ý nghĩa |
| B2 → B2 + OCR | 121 / 141 | 0,24 | +0,0023 [−0,0102; +0,0148] | Không khác biệt có ý nghĩa |

**Diễn giải:**
- Ablation bắt buộc "with / without OCR-enhanced prompting" ở **suy luận** cho **kết quả âm**, nhất quán với dev300. Báo cáo đầy đủ như một dòng ablation.
- Tín hiệu (chưa có ý nghĩa thống kê, **không** dùng làm kết luận): B2 — dù chưa từng thấy dòng OCR khi train — nhích nhiều hơn B1 (+1,00 so với +0,55 điểm EM). Gợi ý mô hình đã fine-tune khai thác OCR tốt hơn zero-shot → củng cố lý do kiểm định B3 (train cùng OCR) so với B2-sub.
- **Độ nhạy (tính bằng McNemar exact, giả định ~260 câu lật như bước 6):** ngưỡng để đạt p < 0,05 là chênh ≈ 34 câu = **1,7 điểm EM**; xác suất phát hiện (mô phỏng 2.000 lần) khi chênh thật là 1,5 / 2,0 / 2,3 / 2,5 điểm lần lượt ≈ 42% / 67% / 79% / 85%. → B3 vs B2-sub chỉ phát hiện ổn định nếu lợi ích thật ≳ 2,3 điểm; kết quả âm khi lợi ích nhỏ hơn không có nghĩa là "OCR vô dụng".

## Kết quả B3 vs B2-sub — phép thử chính (test2000, L4, bf16, 4-bit, seed 42)

Nguồn: `../../experiments/W08_B3_vs_B2sub_test2000_L4_seed42/` (`b3_vs_b2sub_summary.json` + 3 file `paired_*.json`), train log `W08_B2sub_qlora_…`, `W08_B3_qlora_ocr_…`, eval `W08_B2sub_eval_…`, `W08_B3_eval_…`. Notebook 13 chạy một mạch 13–14/10 (không resume).

| Run | Train | OCR | EM | ANLS | token-F1 (phụ) | s/câu |
|---|---|---|---|---|---|---|
| B1 zero-shot | — | không | 0,2880 | 0,4753 | 0,5377 | 0,862 |
| B2 (W07) | full 35.159 câu | không | 0,4970 | 0,7191 | 0,7101 | 1,011 |
| B2-sub | 8.000 câu (2.670 ảnh) | không | 0,4555 | 0,6848 | 0,6692 | 1,021 |
| **B3** | **cùng 8.000 câu** | **VietOCR · v1 (train + test)** | **0,4925** | **0,7038** | **0,6981** | 1,001 |

| So sánh | Chỉ A đúng / chỉ B đúng | McNemar p | ΔANLS (CI95) | Vai trò |
|---|---|---|---|---|
| **B2-sub → B3** | **81 / 155** | **1,67 × 10⁻⁶** | **+0,0191 [+0,0073; +0,0309]** | **Phép thử chính → CẢI THIỆN** (EM +3,70 điểm, vượt ngưỡng độ nhạy ~2,3 điểm) |
| B2 → B3 | 156 / 147 | 0,646 | −0,0153 [−0,0288; −0,0017] | Chỉ báo cáo: EM không phân biệt được; ANLS B3 thấp hơn chút |
| B2 → B2-sub | 154 / 71 | 3,24 × 10⁻⁸ | −0,0343 [−0,0470; −0,0224] | Chỉ báo cáo: cắt 77% dữ liệu train làm giảm 4,15 điểm EM |

| Chi phí train | Mẫu | Step | Giờ train | s/step | VRAM đỉnh | Loss cuối | Tham số train |
|---|---|---|---|---|---|---|---|
| B2 (W07) | 35.159 | 2.198 | 11,99 | 19,63 | 6,88 GB | 0,618 | 0,196% |
| B2-sub | 8.000 | 500 | 2,76 | 19,89 | 6,85 GB | 0,716 | 0,196% |
| B3 | 8.000 | 500 | 2,98 | 21,48 | 8,25 GB | 0,661 | 0,196% |

**Kết luận theo luật đặt trước:** train QLoRA **cùng** OCR (B3) cải thiện có ý nghĩa so với cùng cấu hình **không** OCR (B2-sub): EM 0,4555 → 0,4925, p = 1,7 × 10⁻⁶; ANLS +0,019 (CI95 không chứa 0).

**Diễn giải:**
- Khác bước 6 (OCR chỉ lúc suy luận: không cải thiện), OCR **có tác dụng khi được train cùng** — khớp phân tích dev300: model zero-shot dùng OCR không phân biệt; train cùng OCR dạy model khi nào tin OCR.
- **Hiệu quả:** với 23% dữ liệu và 25% thời gian train (2,98 / 11,99 giờ), B3 đạt EM ngang B2 full (0,4925 vs 0,4970, p = 0,65). OCR bù ≈ 89% phần EM mất do cắt dữ liệu: (0,4925 − 0,4555) / (0,4970 − 0,4555) ≈ 0,89. Chi phí thêm: +8% s/step, +1,4 GB VRAM.

**Giới hạn (phải nêu trong báo cáo):**
1. Một seed (42), một tập con — chưa đo dao động theo seed.
2. Đánh giá trên test2000, chưa trên full test 10.028 câu.
3. ANLS B3 vẫn thấp hơn B2 full (−0,015, CI95 không chứa 0) → B3 "ngang" B2 ở EM, không phải mọi metric.
4. B3 dùng OCR cả train lẫn test (đúng thiết kế); chưa đo B3 khi bỏ OCR lúc test (model có phụ thuộc OCR không).

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
