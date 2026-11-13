# W12 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W12 — Hallucination (H4, POPE-vi) + error analysis của hai mô hình chính thức |
| Thời gian thực | 23–24/10/2026 (tuần thực 7) |
| Mốc kế tiếp | Review 2 (thứ Năm 29/10/2026, làm ở phiên riêng) · W13 = prototype web |
| Phần cứng | Colab Pro: NVIDIA L4 (POPE-vi) · CPU (sinh câu hỏi) · máy local (error analysis, audit) |
| Commit cuối tuần | commit đóng tuần sau `3f3ce27` (xem mục "Dấu vết commit") |

## Mục tiêu

Sau W11, hai mô hình chính thức đã chốt (ViVQA: C3; ViTextVQA: B3-full) và prototype dùng adapter chung SH. W12 trả lời hai câu hỏi bắt buộc của master ("W12 · Analysis"):

1. **QLoRA có làm mô hình "bịa" vật thể nhiều hơn không?** Đây là giả thuyết H4 (H8 cũ ở W02), đo kiểu POPE bằng tiếng Việt.
2. **Lỗi còn lại của mô hình chính thức đến từ đâu?** OCR không đưa được đáp án vào prompt, model chọn sai, hay lỗi thật ra đúng ý?

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | Luật H4 ghi trước + từ điển 80 lớp COCO tiếng Việt + 9 nhóm dễ nhầm | ✅ `9209170`, commit **trước** khi sinh câu hỏi |
| 2 | `src/eval/pope_vi.py` + notebook 21 (Phần A CPU / Phần B L4) | ✅ selftest đạt trên cả Windows lẫn Colab |
| 3 | Sinh file câu hỏi POPE-vi, khoá vào git | ✅ `b09aff4`: 4.985 câu duy nhất, 500 ảnh test chưa thấy lúc train, 3.000 lượt chấm / cách chọn |
| 4 | Chạy POPE-vi 5 cấu hình (B1, B1-ocr, C3, B3-full, SH) | ✅ ≈ 3 h 20 phút L4; kết quả `a945ac5` |
| 5 | Error analysis tự động (luật trước `a523f35`, kết quả `9e15817`) | ✅ nguồn lỗi B3-full, B2 → B3-full ghép cặp, C3 theo loại, ảnh đã thấy / chưa thấy |
| 6 | Audit tay mù 150 lỗi (100 B3-full + 50 C3) | ✅ công cụ `de947a1`, mẫu khoá `e3e6168`, SV4 gán, kết quả `3f3ce27` |
| 7 | R6 §5–§7 bản nháp tiếng Việt, 12 ví dụ định tính | ✅ `2785d40` + đóng tuần |
| 8 | Bảng đối chiếu đánh số giả thuyết (`docs/W02/Hypotheses.md`) | ✅ `2785d40` |
| 9 | Đóng tuần: README, Team Report, R6 §5.6, dọn Downloads + Drive | ✅ |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| VLM / Baseline | B1 và B1-ocr làm mốc zero-shot cho POPE-vi (4) |
| Dataset / OCR | Từ điển COCO tiếng Việt + nhóm dễ nhầm (1); phát hiện 49% ảnh ViVQA test trùng train/val, khớp W07 (3); trạng thái OCR từng câu (5) |
| PEFT / Training | Ảnh hưởng của QLoRA lên hallucination và lên khả năng trả lời có/không (4) |
| Evaluation / Deployment | `pope_vi.py`, `error_analysis.py`, `audit_w12.py` (2, 5, 6); SV4 gán nhãn audit (6); R6 §5–§7 (7); rủi ro thiên lệch "có" của SH cho prototype |

### Definition of Done

- [x] Luật H4 và file câu hỏi khoá trên git **trước** khi chạy model; Phần B tự kiểm sha256 file câu hỏi trùng bản commit
- [x] Mỗi adapter so với zero-shot **cùng prompt** (C3 ↔ B1; B3-full, SH ↔ B1-ocr); adapter kiểm bằng dấu vân tay trọng số
- [x] Câu không đọc được tính sai theo cách nghiêm; cấu hình > 5% không đọc được bị loại khỏi kết luận
- [x] Phán quyết H4 giữ đúng luật; mọi phân tích sau đều ghi nhãn "không thuộc luật"
- [x] Error analysis: định nghĩa commit trước khi chạy; số tự kiểm khớp kết quả chính (705 − 395 = +3,09 EM)
- [x] Audit tay: mẫu khoá trước, gán mù, cùng bộ nhãn W07
- [x] Mọi số truy được về `experiments/W12_*`
- [x] Dọn Downloads + Drive

## Kết quả / số liệu

### 1. H4 — POPE-vi (nguồn: `experiments/W12_pope_vi_L4_seed42/`)

| Cấu hình | F1 random | F1 popular | F1 adversarial | Tỉ lệ báo sai adversarial | yes-ratio |
|---|---|---|---|---|---|
| B1 (zero-shot) | 84,60 | 83,17 | 81,19 | 10,9% | 40–43% |
| B1-ocr | 83,39 | 81,99 | 80,59 | 8,9% | 38–41% |
| C3 | — (91,43% không đọc được) | — | — | — | — |
| **B3-full** | **88,89** | **85,65** | **83,06** | 22,4% | 48–55% |
| SH | 77,70 | 72,83 | 71,17 | 78,7% | 77–89% |

- **Theo luật: H4 MỘT PHẦN.** B3-full KHÔNG TĂNG HALLUCINATION; C3 CHƯA KẾT LUẬN (lệch định dạng); SH TĂNG HALLUCINATION (không tính vào H4).
- **Phân tích sau:**
  - F1 tăng ở B3-full là nhờ recall, trong khi **tỉ lệ báo sai tăng 2,5 lần** (8,9% → 22,4% ở adversarial). Vì vậy F1 không đủ để kết luận về hallucination.
  - C3 trả lời câu có/không bằng danh từ: quên kỹ năng tuân theo chỉ dẫn.
  - B3-full chen tiếng Anh "no" ở 17,7% câu.
  - Train không có đáp án "có" nào, nên giả thuyết "SH học lệch từ dữ liệu" bị bác.

### 2. Error analysis tự động (nguồn: `experiments/W12_error_analysis/`)

- **QLoRA sửa lỗi trả lời dài dòng:** lỗi `extended` giảm từ 25,7% xuống 13,6% số lỗi (ViTextVQA) và từ 21,3% xuống 3,9% (ViVQA).
- **B3-full:** 81% lỗi nằm ở câu mà OCR không chứa nguyên văn đáp án (cận trên; xem mục 3). Khi OCR có đáp án, lỗi chủ yếu là **cắt sai ranh giới** (383/492). 11,5% câu có OCR đọc đúng chữ nhưng sai dấu.
- **OCR B2 → B3-full:** sửa 705, làm hỏng 395 câu (+3,09 EM). Tỉ lệ sửa/hỏng là 326 : 116 ở câu OCR có đáp án. OCR làm phân tâm khoảng 1,4% số câu.
- **C3:** lỗi `number` toàn bộ là đếm lệch; `color` nhầm màu gần nhau; `object` nhiều cặp đồng nghĩa. Ảnh trùng train chỉ làm phồng khoảng 2 điểm (77,66 so với 75,67).

### 3. Audit tay mù 150 lỗi (nguồn: `experiments/W12_error_audit/`)

| | B3-full | C3 |
|---|---|---|
| % lỗi EM mà đúng ý | 55,0 [45,2; 64,4] | 64,0 [50,1; 75,9] |
| EM → EM theo người đọc | 53,35 → **79,01** | 76,64 → **91,59** |
| Hallucination trên mọi câu trả lời | 2,33% | 3,74% |

Kiểu lỗi lớn nhất của B3-full là định dạng / độ dài (40%). Lỗi đọc chữ thật chiếm 36% (≈ 17 điểm EM, trần thô cho việc nâng cấp OCR). Với C3, 28% lỗi là gold sai / nhiễu.

### 4. Đối chiếu giả thuyết

| Mục | Trạng thái sau W12 | Lý do |
|---|---|---|
| H1 · QLoRA ≥ +5 EM | Ủng hộ (không đổi) | B1 → B2 +21,1 EM (ViTextVQA), +51,7 (ViVQA, C1) |
| H2 · OCR ≥ +3 ANLS, cộng hưởng với QLoRA | Ủng hộ một phần (không đổi) | Error analysis W12: OCR sửa 705 / hỏng 395 câu |
| H3 · prompt định dạng tiếng Việt | Không ủng hộ (không đổi) | |
| **H4 · QLoRA không tăng hallucination** | **Một phần** (theo luật) | B3-full không tăng theo F1; C3 không đo được; tỉ lệ báo sai tăng (phân tích sau) |
| Adapter dùng chung SH | Đạt ở W11; **rủi ro mới**: thiên lệch "có" | Xử lý ở W13 trước khi dùng cho prototype |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|---|---|---|
| B1 | SH (adapter prototype) thiên lệch "có": báo sai 55–79% | **Cao** (prototype) | Đầu W13: chẩn đoán (xác suất "có"/"không"), rồi sửa bằng ngưỡng lúc suy luận hoặc train lại có câu có/không cân bằng; kiểm lại không-kém + POPE-vi |
| B2 | F1 không bắt được việc tăng báo sai | Trung bình | Báo kèm tỉ lệ báo sai + yes-ratio; ghi vào Discussion (R7) như giới hạn metric |
| B3 | C3 quên cách trả lời có/không | Trung bình | Ghi là cái giá của fine-tune chuyên biệt; prototype dùng adapter chung nên không chặn |
| B4 | B3-full chen tiếng Anh "no" (17,7%) | Thấp | W13: xử lý trong prototype và báo cáo nếu có hậu xử lý |
| B5 | EM đánh giá thấp (EM theo người đọc 79 / 92) | Thấp | Báo cả EM theo người đọc ở R6/R7; không thay metric chính |
| B6 | Audit một người gán (không có κ W12) | Thấp | Ghi giới hạn; κ bộ nhãn đo ở W07 (0,62) |

## Kế hoạch tuần sau (W13 · Prototype web)

1. **Sửa SH trước khi dựng prototype** (B1): chẩn đoán rồi chọn cách sửa, có số kèm theo. Chạy lại POPE-vi và kiểm không-kém cho bản sửa.
2. **Prototype web hoàn chỉnh** theo master §11: upload ảnh, câu hỏi tiếng Việt, câu trả lời, confidence, văn bản OCR; 1 adapter; chạy trên Colab L4.
3. Xử lý code-switching trong prototype (B4).
4. Bản tiếng Anh R6 §5–§7 để dành W14 (viết report sau khi thí nghiệm chốt).

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `9209170` | Luật H4 (README W12), từ điển `configs/pope_vi_objects.json`, `src/eval/pope_vi.py`, notebook 21 |
| `b09aff4` | Điểm dừng A: khoá file câu hỏi POPE-vi v1 (sha256 `53d04eb2…`) |
| `a523f35` | Luật error analysis + `src/eval/error_analysis.py`; đính chính ghi chú trùng ảnh (W07: 46%, tính cả val: 49%); bỏ quyết định "không train thêm" |
| `9e15817` | Kết quả error analysis tự động |
| `de947a1` | Công cụ audit tay `src/eval/audit_w12.py` |
| `e3e6168` | Khoá danh sách 150 câu audit (seed 42) |
| `14a83c1` | Hướng dẫn gán nhãn, người gán = SV4 |
| `a945ac5` | Kết quả POPE-vi 5 cấu hình + README kết quả H4 |
| `2785d40` | Phân tích bổ sung H4, R6 §5–§7, đối chiếu đánh số giả thuyết |
| `3f3ce27` | Nhãn audit SV4 + kết quả audit |
| *(commit đóng tuần)* | R6 §5.6, README + Team Report W12 |

## Link experiment log

- POPE-vi: `experiments/W12_pope_vi_questions/` (file câu hỏi), `experiments/W12_POPE_*` (5 run), `experiments/W12_pope_vi_L4_seed42/` (phán quyết)
- Error analysis: `experiments/W12_error_analysis/`
- Audit tay: `experiments/W12_error_audit/`, `docs/W12/audit/` (manifest, hướng dẫn, nhãn)
- Notebook: `notebooks/21_pope_vi_hallucination_qwen25vl.ipynb`
