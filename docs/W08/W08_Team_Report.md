# W08 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W08 — Improvement: OCR-enhanced prompting |
| Thời gian thực | 12–14/10/2026 (tuần thực 6) |
| Mốc kế tiếp | **Review 2** — tuần thực 7 (19–25/10/2026); ngày cụ thể bộ môn báo qua email trong tuần đó |
| Phần cứng | Colab Pro: NVIDIA L4 (VLM, train/eval) · Tesla T4 (OCR) |
| Commit cuối tuần | `89c7d5e` (xem mục "Dấu vết commit") |

## Mục tiêu

Error audit W07 cho thấy trên ViTextVQA, lỗi đọc chữ chiếm phần lớn lỗi của B2 (sai OCR 28%, sai vùng 23%, sai dấu 8%). Đề cương bắt buộc ablation "with / without OCR-enhanced prompting". Tuần này:

1. Nối OCR vào pipeline (eval và train) sao cho ablation **chỉ khác đúng một dòng OCR** trong prompt.
2. Tạo OCR cache một lần, tái lập được (ghi rõ engine, phiên bản, model).
3. Chọn cấu hình OCR trên **dev**, theo luật **ghi trước khi có số liệu**.
4. Ablation trên test2000: OCR chỉ lúc suy luận (B1+OCR, B2+OCR).
5. **Phép thử chính:** QLoRA train cùng OCR (B3) so với QLoRA không OCR (B2-sub), cùng tập con train.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | Dọn tồn W07 + hồ sơ Review 2 (W07 Team Report, README) | ✅ `5d161c4`, `3f736eb` |
| 2 | OCR vào pipeline: `--mode ocr` cho `run_baseline` và `train_qlora`; hàm dựng prompt dùng chung `prompt_for_sample` | ✅ prompt `ocr` = zero-shot + đúng 1 dòng OCR; train = eval (selftest) |
| 3 | `ocr.py` viết lại theo PaddleOCR 3.7 (bản `pip` hiện có); cache v2 có conf + box + nguồn gốc | ✅ ảnh thiếu OCR → báo lỗi; thiết lập OCR nằm trong chữ ký run |
| 4 | Tập con train cố định **theo ảnh**: 8.000 câu / 2.670 ảnh, seed 42 | ✅ tất định (manifest có sha256) |
| 5 | Notebook 10 — OCR cache: dev300 × 2 backend, rồi test2000 + train bằng backend đã chọn | ✅ 4.422 ảnh VietOCR, 0 lỗi, 0 NaN; sự cố NCCL đã xử lý |
| 6 | Chỉ số thay thế khi không có nhãn chữ: đáp án có trong OCR không (`ocr_coverage.py`) | ✅ phát hiện PaddleOCR làm rơi dấu tiếng Việt |
| 7 | Notebook 11 — chọn backend × template trên dev300 bằng luật tự động (`select_ocr_config.py`) | ✅ chốt **VietOCR · v1 · 600 ký tự · min_conf 0** |
| 8 | Notebook 12 — ablation OCR lúc suy luận trên test2000 | ✅ không khác biệt có ý nghĩa (B1 lẫn B2) |
| 9 | Notebook 13 — **B3 vs B2-sub** (train cùng / không OCR, cùng tập con) | ✅ **B3 cải thiện có ý nghĩa: EM +3,70 điểm, p = 1,7 × 10⁻⁶** |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| Dataset / OCR | OCR cache (5), chỉ số answer-in-OCR (6), tập con train (4) |
| PEFT / Training | `--mode ocr` khi train (2), B2-sub + B3 (9) |
| Evaluation | Chọn cấu hình theo luật (7), ablation test2000 (8), McNemar + bootstrap, phân tích độ nhạy |
| VLM / Baseline | Prompt OCR sạch, B1 dev300 trên cấu hình chốt W07 |

### Definition of Done

- [x] Ablation "có / không OCR" có kiểm định, luật kết luận ghi trước khi chạy test
- [x] Cấu hình OCR chọn trên dev, đóng băng trước khi chạy test
- [x] OCR cache tái lập được (engine, phiên bản, model, sha256 trong biên bản)
- [x] Phép thử chính chỉ khác đúng yếu tố OCR (chữ ký train B2-sub và B3 chỉ khác khoá `prompting.ocr`)
- [x] Mọi số liệu truy được về file trong `experiments/`
- [ ] Độ ổn định theo seed (mới 1 seed) → **W9**
- [ ] B3 trên full test 10.028 câu, B3 khi bỏ OCR lúc test → **W9**
- [ ] QLoRA ViVQA 2 epoch (dời từ W8) → **W9**

## Kết quả / số liệu

### OCR trên dev300 (266 ảnh, 300 câu) — chỉ số thay thế

| Backend | Đáp án ∈ OCR (giữ dấu) | Đáp án ∈ OCR (bỏ dấu) | s/ảnh (T4) |
|---|---|---|---|
| PaddleOCR (PP-OCRv6, Latin) | 27,0% | **57,7%** | **0,34** |
| PaddleOCR dò + **VietOCR** đọc | **37,7%** | 48,7% | 0,95 |

McNemar (giữ dấu): 28 / 60 câu, p = 0,0008. PaddleOCR làm rơi ký tự có dấu ("Kiểu Thẩm" → "Kiu Thm"). Đây là bằng chứng cho gap G2: OCR có sẵn chưa xử lý tốt dấu thanh tiếng Việt.

### Chọn cấu hình trên dev300 (B1 + OCR)

| Run | EM | ANLS | Lật so với B1 (B1 đúng / OCR đúng) |
|---|---|---|---|
| B1 zero-shot | 0,3267 | 0,4615 | — |
| B1 + VietOCR · v1 **(chốt)** | 0,3267 | 0,4600 | 16 / 16, p = 1 |
| B1 + VietOCR · v2 | 0,3233 | 0,4696 | 17 / 16 |
| B1 + PaddleOCR · v1 | 0,3200 | 0,4606 | 20 / 18 |
| B1 + PaddleOCR · v2 | 0,3067 | 0,4700 | 23 / 17 |

Luật: EM 2 backend hoà (p = 0,856) → xét answer-in-OCR → VietOCR; v2 không hơn v1 → giữ v1.

### Ablation OCR lúc suy luận — test2000

| So sánh | EM | Lật | p | ΔANLS (CI95) | Kết luận |
|---|---|---|---|---|---|
| B1 → B1 + OCR | 0,2880 → 0,2935 | 132 / 143 | 0,547 | +0,005 [−0,010; +0,020] | Không khác biệt |
| B2 → B2 + OCR | 0,4970 → 0,5070 | 121 / 141 | 0,24 | +0,002 [−0,010; +0,015] | Không khác biệt |

Phân tích dev300: OCR làm đổi 167/300 câu trả lời. Model **có dùng** OCR, nhưng dùng không phân biệt (sửa đúng số điện thoại, tên riêng; nhưng cũng chép nguyên lỗi OCR và dán cả chuỗi OCR).

### Phép thử chính — B3 vs B2-sub (test2000)

| Run | Train | OCR | EM | ANLS | Giờ train | VRAM đỉnh |
|---|---|---|---|---|---|---|
| B2 (W07) | 35.159 câu | không | 0,4970 | 0,7191 | 11,99 | 6,88 GB |
| B2-sub | 8.000 câu | không | 0,4555 | 0,6848 | 2,76 | 6,85 GB |
| **B3** | **cùng 8.000 câu** | **có (train + test)** | **0,4925** | **0,7038** | 2,98 | 8,25 GB |

| So sánh | Lật | p | ΔANLS (CI95) | Vai trò |
|---|---|---|---|---|
| **B2-sub → B3** | **81 / 155** | **1,7 × 10⁻⁶** | **+0,019 [+0,007; +0,031]** | **Phép thử chính → CẢI THIỆN** |
| B2 → B3 | 156 / 147 | 0,646 | −0,015 [−0,029; −0,002] | Báo cáo: EM ngang nhau, ANLS B3 thấp hơn chút |
| B2 → B2-sub | 154 / 71 | 3,2 × 10⁻⁸ | −0,034 [−0,047; −0,022] | Báo cáo: cắt 77% dữ liệu mất 4,15 điểm EM |

**Ý nghĩa:** OCR không giúp khi chỉ chèn lúc suy luận, nhưng **giúp rõ khi được train cùng**. Với 23% dữ liệu và 25% thời gian train, B3 đạt EM ngang B2 full; OCR bù khoảng 89% phần EM mất đi do cắt dữ liệu. Chi phí thêm: +8% s/step, +1,4 GB VRAM.

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|--------|-----|-------------|
| B1 | Kết quả B3 mới **1 seed** | **Cao** (hội đồng sẽ hỏi) | W9: chạy thêm seed cho B2-sub và B3 (khoảng 6 h L4 mỗi seed) |
| B2 | Mới đánh giá trên test2000 | Trung bình | W9: B3 trên full test 10.028 câu (khoảng 2,8 h) để đặt cạnh B2 full test và ViTextBLIP-2 |
| B3 | Chưa biết B3 có **phụ thuộc OCR** không | Trung bình | W9: eval B3 khi **bỏ** dòng OCR lúc test (khoảng 35 phút) |
| B4 | ANLS B3 vẫn thấp hơn B2 full | Thấp | Ghi giới hạn; thử B3 với nhiều dữ liệu hơn nếu còn ngân sách |
| B5 | PaddleOCR GPU (CUDA 12) ghi đè NCCL của torch (CUDA 13) | Đã xử lý | Notebook 10 mục 2 chụp + khôi phục thư viện `nvidia-*` của torch (`833abd8`) |

**Ngân sách Colab:** còn 133 CU (14/10); L4 đo thật **1,54 CU/giờ** → khoảng 86 giờ L4, đủ cho phần còn lại của đồ án (ước tính cần 40–45 giờ).

## Kế hoạch tuần sau (W09)

1. **Củng cố B3:** thêm seed cho B2-sub và B3; B3 trên full test; B3 khi bỏ OCR lúc test.
2. QLoRA ViVQA 2 epoch, chọn epoch trên val (dời từ W8).
3. Sau buổi Review 2: điền checklist AI_Capstone_Review123 (sheet Review2), gửi mail, upload Drive.
4. Bắt đầu Results (R6) bản tiếng Việt cho phần OCR, sau khi số liệu đã ổn định.

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `5d161c4` | Bắt đầu W8: `thuHoiNhanh` v2, audit JSON LF, index `docs/W08/` |
| `3f736eb` | W07 Team Report + README (Review 2); `.gitignore` file admin |
| `4c71df2` | Pipeline OCR: PaddleOCR 3.x, cache v2, `--mode ocr`, `prompt_for_sample`, tập con train theo ảnh |
| `c31700e` | Notebook 10, `ocr_coverage.py`, luật chọn cấu hình OCR ghi trước |
| `833abd8` | Sửa notebook 10: bảo vệ NCCL của torch khi cài Paddle GPU |
| `dc76997` | Kết quả OCR dev300 + notebook 11 + `select_ocr_config.py` |
| `f63f346` | Kết quả chọn cấu hình (VietOCR · v1) |
| `50b794c` | Biên bản OCR cache + luật bước 6 và B3 ghi trước + notebook 12 |
| `c80374e` | Kết quả ablation OCR lúc suy luận (test2000) |
| `6b69ff2` | Quyết định A (B3 vs B2-sub) + notebook 13 |
| `89c7d5e` | **Kết quả chính: B3 vs B2-sub** |

## Link experiment log

- OCR: `experiments/W08_ocr_dev300_coverage/`, `experiments/W08_ocr_cache/ocr_cache_manifest.json`
- Tập con train: `experiments/W08_data_subsets/train_subset8000_byimage_seed42_manifest.json`
- Chọn cấu hình: `experiments/W08_ocr_select_dev300_L4_seed42/` + 5 run `W08_B1*_dev300_*`
- Ablation test2000: `experiments/W08_ocr_ablation_test2000_L4_seed42/` + `W08_B1ocr_…_test2000_…`, `W08_B2ocr_…_test2000_…`
- B3 vs B2-sub: `experiments/W08_B3_vs_B2sub_test2000_L4_seed42/` + train `W08_B2sub_qlora_…`, `W08_B3_qlora_ocr_…`, `W08_smoke_B3_…` + eval `W08_B2sub_eval_…`, `W08_B3_eval_…`
- Adapter (không lên git, trên Drive): `adapters/W08_B2sub_qlora_…/epoch_1`, `adapters/W08_B3_qlora_ocr_…/epoch_1`
