# W11 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W11 — Bảng đánh giá đầy đủ + adapter dùng chung hai dataset + phân rã theo loại câu hỏi |
| Thời gian thực | 21–22/10/2026 (tuần thực 7) |
| Mốc kế tiếp | **Review 2 — thứ Năm 29/10/2026** (tuần thực 8) |
| Phần cứng | Colab Pro: NVIDIA L4 (train/eval) · Tesla T4 (OCR) |
| Commit cuối tuần | commit đóng tuần sau `c406196` (xem mục "Dấu vết commit") |

## Mục tiêu

Sau W10, mô hình chính thức đã chốt (ViVQA: C3; ViTextVQA: B3-full). W11 trả lời ba câu hỏi còn lại trước Review 2:

1. **Mọi số liệu có nhất quán không?** Gom mọi run chính của hai dataset vào một bảng duy nhất, tính lại từ `predictions.json` thay vì chép tay từ README các tuần.
2. **Một adapter có phục vụ được cả hai loại câu hỏi không?** Prototype không biết câu hỏi "thuộc dataset nào", nên cần một adapter dùng chung (SH) mà không kém adapter chuyên biệt. Đây là điều đã hứa trên slide Review 2.
3. **OCR giúp loại câu hỏi nào?** ViTextVQA không có nhãn loại câu hỏi, trong khi master yêu cầu error analysis theo loại.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | Luật ghi trước: SH (phương án S2, δ = 1,5 EM), quy ước VQA-Acc, luật bộ phân loại câu hỏi | ✅ `324caeb` — commit **trước** khi có số |
| 2 | `src/eval/full_eval_table.py`: bảng đầy đủ (EM · VQA-Acc · ANLS · F1 · chi phí · ổn định seed · loại câu hỏi ViVQA) | ✅ 23 dòng, 23/23 khớp `metrics.json`; phát hiện **VQA-Acc ≡ EM** (0 câu có > 1 đáp án gold) |
| 3 | OCR ảnh ViVQA (notebook 19) | ✅ lần 3: 9.770 ảnh, **0 lỗi**, 0,11 s/ảnh, engine trùng cache ViTextVQA. Lần 1–2 hỏng do cài Paddle (xem Blocker đã xử lý) |
| 4 | `train_qlora.py --mix`, `shared_adapter.py`, notebook 20 | ✅ `5076b76` — selftest: phần ViTextVQA trong run trộn giữ đúng dấu vân tay OCR của B3 |
| 5 | SH: train + eval 2 dataset + kết luận theo luật | ✅ **ADAPTER DÙNG CHUNG ĐẠT** (6,07 h train L4) |
| 6 | Bộ phân loại `qt-v1` (9 loại) + 150 nhãn tay mù + bảng EM/ANLS theo loại | ✅ đóng băng `9b441f1` trước khi chạy test; accuracy 80,7% |
| 7 | Deck cộng dồn Review 1 + Review 2 (44 slide, có lời nói) | ✅ **bản nháp**; Hoàng còn đọc và sửa Phần B |
| 8 | Tập nói Review 2 | ➜ chuyển W12 (trước 29/10) |
| 9 | Đóng tuần: README, Team Report, R6 (mục 3, 4.5, 5), dọn Downloads + Drive | ✅ |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| VLM / Baseline | Bảng đầy đủ: B1 và R1b trong cùng giao thức (2); slide kiến trúc baseline chuyển sang Existing System (7) |
| Dataset / OCR | OCR ảnh ViVQA + fail-fast `ocr.py` (3); gán nhãn tay 150 câu (6) |
| PEFT / Training | `--mix` + train SH (4, 5) |
| Evaluation / Deployment | `full_eval_table.py` (2); luật + kết luận SH (1, 5); `question_types.py` + bảng theo loại (6); R6 (9). SH là adapter cho prototype |

### Definition of Done

- [x] Một bảng duy nhất cho mọi run chính, tính lại từ predictions và đối chiếu với `metrics.json`
- [x] SH so với mốc **cùng điều kiện** (C1, B3): cùng file train, cùng siêu tham số, cùng OCR; luật ghi trước, ghép cặp trên cùng câu
- [x] Điều kiện hợp lệ của SH kiểm tự động trước khi kết luận
- [x] Bộ phân loại câu hỏi đóng băng trước khi chạy test, kiểm bằng nhãn tay mù, gắn cờ loại kém tin cậy
- [x] Mọi số truy được về file trong `experiments/`; sai lệch so với luật ghi trước được ghi lại (câu stt 97)
- [x] Deck có kết quả W10 + W11
- [ ] Tập nói Review 2 → W12
- [x] Dọn Downloads + Drive

## Kết quả / số liệu

### 1. Bảng chính (nguồn: `experiments/W11_full_eval_table/`)

| Dataset · tập test | Zero-shot B1 | QLoRA không OCR | **Đề xuất** | ΔEM so với B1 |
|---|---|---|---|---|
| ViTextVQA · full test (10.028) | EM 29,12 · ANLS 47,60 | B2: 50,26 · 72,02 | **B3-full: 53,35 · 73,89** | +24,23 |
| ViVQA · test (3.001) | EM 22,09 · ANLS 28,61 | C1: 73,78 · 75,66 | **C3: 76,64 · 78,03** | +54,55 |

VQA-Acc trùng EM ở từng câu vì cả hai dataset chỉ có một đáp án gold mỗi câu; cột VQA-Acc vẫn báo kèm chú thích.

### 2. Adapter dùng chung SH (ViVQA 10.799 + ViTextVQA 8.000 câu, prompt OCR cho mọi câu, 1 epoch, seed 42)

| Dataset | Mốc → SH (EM) | ΔEM (CI95) | McNemar p | Kết luận (δ = 1,5) |
|---|---|---|---|---|
| ViVQA test (3.001) | C1 73,78 → 73,51 | −0,27 [−1,13; +0,60] | 0,60 | KHÔNG KÉM |
| ViTextVQA test2000 | B3 49,25 → 49,50 | +0,25 [−0,95; +1,45] | 0,75 | KHÔNG KÉM |

- → **ADAPTER DÙNG CHUNG ĐẠT**: prototype dùng 1 adapter (29,5 MB) thay vì 2. Bảng kết quả chính vẫn dùng C3 và B3-full.
- Dòng OCR vô nghĩa trên ảnh ViVQA (44% câu, đáp án gold trong OCR 0,0%) **không làm hại**: ΔEM +0,15 [−1,05; +1,36]; chép chữ OCR vào đáp án sai 0,30% (C1: 0,23%).
- Chi phí: train 6,07 h ≈ C1 + B3 (6,01 h); VRAM đỉnh 8,15 GB; suy luận như adapter chuyên biệt.

### 3. EM theo loại câu hỏi — ViTextVQA full test (chỉ mô tả)

| | source | contact | number_time | person | location | name ⚠ | text_content ⚠ | product_service | other ⚠ |
|---|---|---|---|---|---|---|---|---|---|
| B2 | 57,2 | 55,5 | 55,7 | 54,8 | 44,0 | 55,7 | 48,6 | 42,5 | 46,1 |
| B3-full | 58,1 | 64,1 | 58,8 | 61,2 | 47,0 | 57,6 | 51,1 | 44,1 | 49,6 |
| Δ | +0,9 | **+8,6** | +3,1 | +6,4 | +3,0 | +1,9 | +2,5 | +1,6 | +3,5 |

⚠ = precision < 80% so với nhãn tay (bộ luật không sửa sau khi thấy test). OCR giúp nhiều nhất khi phải chép đúng chuỗi (số điện thoại, website, tên người); loại khó nhất là câu phải suy ra từ chữ (`product_service`, `location`).

### 4. Đối chiếu giả thuyết

| Mục | Trạng thái sau W11 | Lý do |
|---|---|---|
| H1 · QLoRA ≥ +5 EM | Ủng hộ (không đổi) | B1 → B2 +21,1 EM (ViTextVQA), +51,7 (ViVQA, C1) |
| H2 · OCR ≥ +3 ANLS, cộng hưởng với QLoRA | Ủng hộ một phần (không đổi) | Bảng theo loại cho thấy lợi ích tập trung ở loại chép chuỗi |
| H3 · prompt định dạng tiếng Việt | Không ủng hộ (không đổi) | |
| H4 · QLoRA không tăng hallucination | Chờ W12 | Đã có đầu vào: tỉ lệ chép OCR nhiễu 0,30% |
| Adapter dùng chung (mục tuỳ chọn) | **Đạt** | Không kém cả hai mốc |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|---|---|---|
| B1 | Deck Phần B là bản nháp Claude soạn, chưa theo cách nói của Hoàng; chưa tập nói | **Cao** (review 29/10) | W12, trước 29/10: Hoàng nắm Phần A → đọc Phần B, ghi yêu cầu sửa → tập nói toàn bộ, bấm giờ |
| B2 | H4 (hallucination) chưa đo | Trung bình | W12: kiểu POPE trên B1 / B2 / B3-full |
| B3 | `name`, `text_content`, `other` của `qt-v1` kém tin cậy (precision 45–70%) | Thấp | Chỉ dùng mô tả, ghi giới hạn; không sửa luật sau khi thấy test |
| B4 | SH thấp hơn mô hình chính thức 3–4 EM (bản rẻ) | Thấp | SH-full ≈ 19 h L4 ≈ 29 CU — chưa quyết, cân nhắc sau Review 2 |
| B5 | `docs/W02/Hypotheses.md` còn đánh số cũ | Thấp | Cập nhật khi viết R4 Methodology |
| B6 | Ngân sách Colab | Trung bình | Còn ≈ 37 CU (ước tính); đọc số thật đầu W12 |

**Đã xử lý trong tuần:** notebook 19 lần 1–2 tải wheel Paddle GPU bị timeout nên chuyển CPU; Paddle 3.3 CPU + oneDNN lỗi với mọi ảnh. Mục kiểm chặn được trước khi run nào dùng cache lỗi. Sửa: `ocr.py` dừng ngay nếu 20 ảnh đầu đều lỗi, tắt oneDNN trên CPU; notebook bắt buộc 0 lỗi ở bước thử, tải lại Paddle GPU 3 lần, ghim `paddlepaddle-gpu==3.3.1`.

## Kế hoạch tuần sau (W12 · Hallucination + error analysis + Review 2)

1. **Review 2 (ưu tiên số 1, trước 29/10):** Hoàng nắm Phần A, đọc Phần B và ghi yêu cầu sửa; Claude sửa; tập nói toàn bộ, bấm giờ; chuẩn bị câu hỏi phản biện.
2. **H4 — hallucination kiểu POPE tiếng Việt** trên B1 / B2 / B3-full: luật ghi trước, sinh câu hỏi có/không về vật thể có / không có trong ảnh.
3. **Error analysis B3-full theo loại câu hỏi và kiểu lỗi** (đọc chữ sai, sai vùng, sai dấu thanh, gold nhiễu), lấy mẫu theo loại; ví dụ định tính cho R6 mục 7.
4. Cập nhật `docs/W02/Hypotheses.md` theo đánh số mới.

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `324caeb` | Luật ghi trước (README W11), `full_eval_table.py` + kết quả, notebook 19 |
| `5076b76` | `train_qlora.py --mix`, `shared_adapter.py`, notebook 20 |
| `401a09d` | Sửa OCR lần 1: fail-fast, tắt oneDNN trên CPU, cách ly cache lỗi |
| `72286c4` | Notebook 19: tải lại Paddle GPU, ghim 3.3.1 |
| `9b441f1` | Đóng băng `qt-v1` + phiếu 150 câu + hướng dẫn gán nhãn |
| `14783da` | 150 nhãn tay, kiểm nhãn, bảng theo loại |
| `fadb7b5` | Run SH (train + 2 eval), kết luận SH, biên bản OCR ViVQA |
| `c406196` | README phần SH, bảng đầy đủ 23 dòng |
| *(commit đóng tuần)* | Deck `ViVQA-VLM_Review1-2_KLTN.pptx` + script, builder cũ vào `archive_W10_builder/`, README + Team Report W11, R6 mục 3 / 4.5 / 5 |

## Link experiment log

- Bảng đầy đủ: `experiments/W11_full_eval_table/`
- OCR ViVQA: `experiments/W11_ocr_cache_vivqa/`
- SH: `experiments/W11_SH_qlora_*` (train), `experiments/W11_SH_eval_*` (2 eval), `experiments/W11_shared_adapter_L4_seed42/`
- Loại câu hỏi: `experiments/W11_question_types/`, `docs/W11/qt_labels_150.csv`
- Adapter (không lên git, trên Drive): `adapters/W11_SH_*`
