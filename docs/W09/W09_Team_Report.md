# W09 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W09 — Củng cố B3 (QLoRA nhận biết OCR) + Review 2 |
| Thời gian thực | 15–17/10/2026 (tuần thực 6) |
| Mốc kế tiếp | **Review 2** — tuần thực 7 (19–25/10/2026); ngày cụ thể bộ môn báo qua email trong tuần đó |
| Phần cứng | Colab Pro: NVIDIA L4 (train/eval) · Tesla T4 (OCR) |
| Commit cuối tuần | commit đóng tuần sau `7701fd3` (xem mục "Dấu vết commit") |

## Mục tiêu

Kết quả chính W08 — B3 (QLoRA train cùng OCR) thắng B2-sub (cùng QLoRA, cùng 8.000 câu train, không OCR) — còn 4 lỗ hổng mà hội đồng sẽ hỏi. Tuần này lấp cả 4, rồi đưa phương pháp lên slide Review 2:

1. **Mới 1 seed** → chạy thêm seed 43, 44, kết luận theo luật ghi trước.
2. **Chưa biết B3 có dựa vào OCR không** → eval B3 khi bỏ OCR lúc test.
3. **Chưa tách được** "train cùng OCR" khỏi "có OCR lúc test" → hoàn chỉnh bảng 2×2.
4. **Mới test2000** → chạy lại trên full test 10.028 câu (tập paper dùng).
5. Việc dời từ W08: QLoRA ViVQA 2 epoch, **chọn epoch không chạm test**.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | `--seed` cho `train_qlora.py` (seed nằm trong chữ ký run; mặc định 42 giữ nguyên chữ ký W07/W08); `seed_stability.py`; notebook 14 | ✅ `c69794a` |
| 2 | Phần A: B3 seed 42 **bỏ OCR** lúc test | ✅ **B3 phụ thuộc OCR** (−4,25 EM, p = 3,8 × 10⁻⁹); bỏ OCR thì **ngang B2-sub** về EM |
| 3 | Seed 43, 44 cho B2-sub và B3 (tập con train giữ nguyên) | ✅ **Cải thiện ổn định**: 3/3 seed, +3,57 ± 0,76 EM |
| 4 | `factorial_2x2.py` + notebook 15 Phần A: ô "train không OCR, test có OCR" | ✅ **Tương tác**: cần OCR ở cả train lẫn test (+4,9 EM, CI95 [+3,2; +6,6]) |
| 5 | Notebook 10 Phần C: OCR thêm 1.867 ảnh full test (VietOCR, cùng engine W08) | ✅ 0 lỗi; dấu vân tay 4.422 ảnh cũ giữ nguyên (kiểm độc lập từ Drive) |
| 6 | Notebook 15 Phần B: B3 + B2-sub trên **full test 10.028 câu** | ✅ **Tái lập**: +2,54 EM, p = 1,1 × 10⁻¹⁴; trùng 100% với run test2000 trên 2.000 câu chung |
| 7 | Notebook 16: QLoRA ViVQA lịch 2 epoch, chọn epoch trên **val sạch** (ảnh không trùng train) | ✅ chọn C3; test 0,7378 → **0,7664** (+2,86, p = 4,6 × 10⁻⁹) |
| 8 | Slide Review 2: thêm Part 06 (4 slide), sửa slide 1, 2, 3, 13, 18, 19; script sinh từ ghi chú slide | ✅ 20 slide, bản EN + TV |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| Dataset / OCR | Mở rộng OCR cache sang full test (5); val sạch ViVQA (7) |
| PEFT / Training | `--seed` (1); seed 43/44 (3); QLoRA ViVQA 2 epoch (7) |
| Evaluation | Phần A (2), bảng 2×2 + tương tác (4), full test + so ViTextBLIP-2 (6), luật chọn epoch + phân tầng rò rỉ (7) |
| VLM / Baseline | B2 ViVQA chính thức đổi sang C3 (7) |

### Definition of Done

- [x] Độ ổn định theo seed: 3 seed, luật ghi trước, mean ± std
- [x] B3 khi bỏ OCR lúc test (phụ thuộc OCR) — luật ghi trước
- [x] Ablation 2×2 OCR lúc train × lúc test, có kiểm định tương tác
- [x] B3 trên full test 10.028 câu, kiểm tái lập ≥ 99%
- [x] QLoRA ViVQA 2 epoch, chọn epoch trên val sạch, test chạm một lần
- [x] Slide Review 2 có phương pháp đề xuất + kết quả sơ bộ
- [x] Mọi số liệu truy được về file trong `experiments/`; mọi luật kết luận commit lên git **trước** khi có số
- [ ] Checklist AI_Capstone_Review123 (sheet Review2) → **sau buổi Review 2**

## Kết quả / số liệu

### 1. Độ ổn định theo seed (test2000)

| Seed | EM B2-sub | EM B3 | ΔEM | p |
|---|---|---|---|---|
| 42 | 0,4555 | 0,4925 | +3,70 | 1,7 × 10⁻⁶ |
| 43 | 0,4475 | 0,4900 | +4,25 | 4,0 × 10⁻⁸ |
| 44 | 0,4605 | 0,4880 | +2,75 | 1,4 × 10⁻⁴ |
| **mean ± std** | 45,45 ± 0,66 | 49,02 ± 0,23 | **+3,57 ± 0,76** | |

ANLS: +0,0155 ± 0,0040 (seed 44 cận dưới CI95 sát 0). Kết luận theo luật: **CẢI THIỆN ỔN ĐỊNH**.

### 2. Bảng 2×2 — OCR lúc train × OCR lúc test (test2000, seed 42)

| EM | Test không OCR | Test có OCR |
|---|---|---|
| Train không OCR | 0,4555 | 0,4490 |
| Train có OCR | 0,4500 | **0,4925** |

Chỉ OCR lúc test: −0,65 (p = 0,49). Chỉ OCR lúc train: −0,55 (p = 0,35). Cả hai: +4,35 (p = 2,2 × 10⁻⁸). Tương tác **+4,9 EM**, CI95 [+3,2; +6,6]. → Đóng góp là **dạy model dùng OCR** (OCR-aware fine-tuning), không phải "thêm OCR vào prompt".

### 3. Full test 10.028 câu

| Run | Dữ liệu train | EM | ANLS | F1 (phụ) |
|---|---|---|---|---|
| B1 zero-shot | — | 29,12 | 47,60 | 53,89 |
| B2 (W07) | 100% | 50,26 | 72,02 | 71,16 |
| B2-sub | 23% | 46,56 | 68,90 | 67,83 |
| **B3** | 23% + OCR | **49,10** | **70,47** | **70,03** |
| ViTextBLIP-2 (paper Table 6) | 100% | 25,48 | — | 53,95 |

- B2-sub → B3: **+2,54 EM** (417 / 672, p = 1,1 × 10⁻¹⁴), ANLS +0,0157 [+0,0106; +0,0208] → **tái lập**.
- B3 cao hơn ViTextBLIP-2 cả EM và F1 (cận dưới CI95 48,11 / 69,30).
- **Đính chính W08:** B3 **thấp hơn** B2 dữ liệu đầy đủ 1,16 EM (p = 0,0025). Phát biểu đúng: với 23% dữ liệu và 25% thời gian train, OCR bù ≈ **69%** khoảng cách EM do cắt dữ liệu.

### 4. ViVQA — chọn epoch

| Ứng viên | EM val sạch (688 câu) |
|---|---|
| C1 · B2 W07, 1 epoch | 0,7311 |
| C2 · lịch 2 epoch, sau epoch 1 | 0,7151 (p = 0,15 so với C1) |
| **C3 · lịch 2 epoch, sau epoch 2** | **0,7703** (p = 1,4 × 10⁻⁴) |

Test: 0,7378 → **0,7664** (+2,86, p = 4,6 × 10⁻⁹). Kiểm sau: mức tăng đều giữa ảnh đã thấy (+2,9) và chưa thấy (+2,9), trên tập sạch +3,2 (p = 1,5 × 10⁻⁴) → **không phải nhớ ảnh**. B2 ViVQA chính thức = C3.

### 5. Đối chiếu giả thuyết

| Giả thuyết | Trạng thái | Lý do |
|---|---|---|
| H6 · OCR prompting tăng ANLS ≥ +3 và cộng hưởng với QLoRA | **Ủng hộ một phần** | Cộng hưởng mạnh (tương tác +4,9 EM); nhưng ANLS +1,55 ± 0,40 (< +3, > ngưỡng bác bỏ +1) |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|--------|-----|-------------|
| B1 | B3 vẫn thấp hơn B2 dữ liệu đầy đủ (−1,16 EM, full test) | Trung bình | W10: train B3 trên toàn bộ train ViTextVQA (≈ 13 h L4, ≈ 20 CU) — câu hỏi: OCR-aware có vượt B2 khi đủ dữ liệu không |
| B2 | Lợi ích ANLS nhỏ hơn EM (H6 một phần) | Thấp | Ghi giới hạn; phân tích lỗi B3 theo nhóm (dấu thanh, lệch ký tự) ở W11 |
| B3 | B3 phụ thuộc OCR: OCR hỏng → lùi về mức B2-sub | Thấp | Prototype: chạy OCR trước, hiển thị văn bản OCR, cảnh báo khi OCR rỗng |
| B4 | Colab mất kết nối tab / runtime bị xoá nhầm (15/10) | Đã xử lý | Notebook resume được từ checkpoint; notebook 15, 16 tự trả máy; quy tắc: không mở menu runtime khi đang train |
| B5 | Selftest crash trên console Windows (cp1252) | Thấp | Chạy với `PYTHONIOENCODING=utf-8`; sửa gốc (`sys.stdout.reconfigure`) khi tiện |

**Ngân sách Colab (ước tính):** W09 dùng ≈ 49 CU (notebook 14 ≈ 26, notebook 15 ≈ 11, notebook 16 ≈ 11, OCR T4 ≈ 2) → còn ≈ **84 CU ≈ 55 giờ L4** cho W10–W15 (kiểm lại số thật trong Colab → Resources).

## Kế hoạch tuần sau (W10)

1. **Review 2** (tuần thực 7): tập nói slide 20 trang (≈ 22–26 phút); sau buổi review điền checklist AI_Capstone_Review123 sheet Review2, gửi mail, upload Drive.
2. B3 trên toàn bộ train ViTextVQA (đóng blocker B1).
3. Ablation cấu hình LoRA: rank và module gắn LoRA (H3, H4).
4. Bắt đầu Results (R6) bản tiếng Việt cho phần OCR — số liệu W09 đã ổn định.

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `c69794a` | Bước 0: `--seed`, `seed_stability.py`, notebook 14, luật Phần A + luật seed ghi trước |
| `7697750` | Kết quả Phần A (B3 bỏ OCR) |
| `701f87f` | Bước 1: luật 2×2 + luật full test ghi trước; `factorial_2x2.py`; notebook 15; notebook 10 Phần C |
| `c9894f5` | Kết quả 3 seed; notebook 15 tự trả máy |
| `ab70867` | Biên bản OCR full test + kiểm độc lập |
| `f871872` | Bước 3: luật chọn epoch ViVQA ghi trước; notebook 16 |
| `1777b08` | Kết quả bảng 2×2 |
| `c3c3937` | Kết quả full test + đính chính W08 |
| `7701fd3` | Kết quả chọn epoch ViVQA + phân tầng rò rỉ |
| *(commit đóng tuần)* | Slide Review 2 (20 slide) + README + Team Report W09 |

## Link experiment log

- Phần A: `experiments/W09_B3_ocr_dependency_test2000_L4_seed42/`
- Seed: `experiments/W09_B3_seed_stability_test2000_L4/` + các run `W09_B2sub_*_seed{43,44}`, `W09_B3_*_seed{43,44}`
- Bảng 2×2: `experiments/W09_ocr_factorial_2x2_test2000_L4_seed42/`
- OCR full test: `experiments/W09_ocr_cache_fulltest/`
- Full test: `experiments/W09_fulltest_B3_vs_B2sub_L4_seed42/`
- ViVQA chọn epoch: `experiments/W09_vivqa_epoch_select_L4_seed42/`
- Adapter (không lên git, trên Drive): `adapters/W09_*`
