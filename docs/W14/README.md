# W14 · Hoàn thiện code: dữ liệu + prototype + demo

> Tuần ảo W14 (bắt đầu 01/11/2026, tuần thực 9). Hội đồng cấp Khoa là **tuần thực 13 (30/11–06/12)**.
> **Phạm vi (Hoàng, 01/11/2026):** tuần này chỉ làm phần **code**: xử lý dữ liệu, prototype web, bản demo hoàn chỉnh. Thesis, slide và sửa theo góp ý Review 2 **dời sang tuần sau**. Deck đang lưu trên Canva: khi nào làm slide, Hoàng tải về rồi Claude cập nhật và lưu đúng chỗ.
> Colab đầu tuần: đọc ở **Colab → Resources** khi mở notebook 25 và ghi vào đây.

## Việc trong tuần

| # | Việc | Ước tính | Trạng thái |
|---|---|---|---|
| 1 | **R-EXIF**: kiểm cờ xoay EXIF có làm hỏng OCR cache ViTextVQA không — luật (README này) + `src/data/exif_audit.py` + notebook `25_exif_ocr_audit_vitextvqa.ipynb` (T4) | 3 h (≈ 1 h T4) | ✅ 01/11 — **L1 XÁC NHẬN · L2 > 1% · L3 ≤ 5%** |
| 2 | Theo L2: sửa `ocr.py` + cache mới (notebook `26_ocr_cache_exiffix_vitextvqa.ipynb`, T4) + suy luận lại B3-full full test (notebook `27_b3full_exiffix_testfull_qwen25vl.ipynb`, L4). Không train lại (L3) | ≈ 5 h (≈ 0,4 h T4 + 3 h L4) | ⏳ code xong, chờ chạy |
| 3 | Prototype: sửa đường ảnh (OCR trên ảnh độ phân giải gốc, xử lý EXIF thống nhất với eval) + kiểm lại R-OCR / R-LIVE (luật ghi trước) | 3 h (≈ 1 h L4) | ⏳ |
| 4 | Demo hoàn chỉnh: bộ ví dụ chọn sẵn, kịch bản demo, quay video dự phòng | 2,5 h | ⏳ |
| 5 | Gói tái lập phần code: ghim phiên bản thư viện, script suy luận dòng lệnh, README chạy lại, manifest adapter | 3 h | ⏳ |
| 6 | Đóng tuần: README + Team Report, dọn Downloads + `G:\My Drive\ViVQA-VLM` | 1,5 h | ⏳ |
| | **Tổng** | **≈ 18 h** | |

## R-EXIF — luật ghi trước (commit TRƯỚC khi chạy notebook 25)

**Vấn đề (W13, bước 8).** Ở `1849.jpg` và `14277.jpg` (4000×3000), cache OCR chứa chữ không liên quan, còn OCR trên ảnh đã xoay / thu nhỏ đọc đúng biển hiệu. Nếu lỗi có hệ thống thì OCR mà B3, B3-full và SH dùng lúc train lẫn lúc test bị hỏng ở một phần dữ liệu.

**Giả thuyết cơ chế.** `PaddleDetVietOCREngine.read` đưa **đường dẫn** ảnh cho Paddle dò hộp chữ. Paddle đọc bằng OpenCV, mà `cv2.imread` mặc định **xoay** ảnh theo cờ EXIF Orientation. VietOCR lại cắt hộp từ ảnh PIL, mà PIL **không xoay**. Ảnh có Orientation ≠ 1 thì hộp ở toạ độ ảnh đã xoay nhưng bị cắt trên ảnh chưa xoay. Kết quả là cắt nhầm vùng và đọc ra chữ rác.
- Đã kiểm trên ảnh giả 400×300 có cờ xoay 90° (01/11): OpenCV 5.0 đọc ra 300×400, PIL 12.3 đọc ra 400×300.
- **Chưa kiểm** Paddle có thật sự đọc bằng OpenCV hay không. Bước B kiểm điều này trên dữ liệu thật.

**Bước A (chỉ đọc header, CPU).** Quét mọi ảnh của ViTextVQA (train / dev / test full) và ViVQA (train / test). Đếm ảnh có Orientation ≠ 1 và số câu hỏi dùng các ảnh đó.

**Bước B (T4).** Lấy mẫu cố định, seed 42:
- ≤ 300 ảnh có cờ xoay (ưu tiên ảnh test);
- 200 ảnh test **đối chứng** không có cờ;
- luôn gồm `1849.jpg` và `14277.jpg`.

Mỗi ảnh OCR 3 kiểu, cùng engine với cache (kiểm sha256 cache = biên bản W10 `e7ce4f76d98c…` và engine trùng trước khi chạy):
- `asis`: file gốc, đúng như cách dựng cache;
- `fixed`: ảnh đã xoay theo EXIF, lưu PNG không còn cờ;
- `small`: như `fixed` nhưng thu cạnh dài về 2.048 px (chỉ ảnh > 2.048 px; chỉ mô tả).

Độ giống chuỗi = `difflib` ratio trên văn bản OCR đã chuẩn hoá (`ocr_coverage.norm`); cả hai rỗng = 1, một bên rỗng = 0. **"Lệch"** = độ giống giữa cache và `fixed` < 0,5.

**Luật:**

| Luật | Điều kiện | Hệ quả |
|---|---|---|
| **L1** cơ chế | ≥ 80% ảnh có cờ xoay trong mẫu bị lệch **và** < 5% ảnh đối chứng bị lệch | **XÁC NHẬN** / **KHÔNG XÁC NHẬN**. Không có ảnh có cờ xoay → "không có ảnh để kiểm" |
| **L2** test | Chỉ khi L1 xác nhận. Tỉ lệ câu **full test ViTextVQA** (10.028) có ảnh cờ xoay | **> 1%** → sửa `ocr.py`, OCR lại các ảnh đó vào **file cache mới** (cache cũ giữ nguyên), suy luận lại B3-full trên full test, báo cả hai số. Số chính thức vẫn là bản cũ nếu không train lại (L3). **≤ 1%** → giữ số chính thức, ghi giới hạn, báo thêm EM khi bỏ các câu này |
| **L3** train | Chỉ khi L1 xác nhận. Tỉ lệ câu **train ViTextVQA** (35.159) có ảnh cờ xoay | **> 5%** → train lại B3-full với cache đã sửa (≈ 13 h L4). Số chính thức ViTextVQA chuyển sang bản train lại (bản cũ học trên OCR lỗi); bản cũ giữ trong bảng. **≤ 5%** → không train lại |

**L1 không xác nhận** thì giữ nguyên số, và nguyên nhân ở 2 ảnh nghi vấn được tìm tiếp trong kiểu `small` (giả thuyết "OCR kém trên ảnh độ phân giải rất cao").

**Chỉ mô tả (không có luật):** % câu test có đáp án gold nằm trong OCR (cache vs `asis` / `fixed` / `small`); độ lệch `asis` vs cache của nhóm đối chứng (sai khác giữa hai lần chạy); ảnh ViVQA có cờ xoay (SH, prototype).

**Không đổi** cache, code OCR hay số đã báo cáo trong bước 1. Mọi sửa chữa là bước 2, có commit riêng.

## Kết quả R-EXIF (notebook 25, 01/11/2026, T4): **L1 XÁC NHẬN · L2 > 1% → sửa · L3 ≤ 5% → không train lại**

Nguồn: `../../experiments/W14_exif_audit_vitextvqa/` (`exif_scan.json`, `exif_reocr.json`, `exif_audit_summary.json`). Commit lúc chạy `a2e20b1`.

**Bước A — ảnh có cờ xoay (Orientation ≠ 1):**

| Tập | Câu bị ảnh hưởng | Ảnh | Hướng |
|---|---|---|---|
| ViTextVQA train | 450 / 35.159 (**1,28%**) | 124 / 11.733 | 6: 118 · 8: 5 · 3: 1 |
| ViTextVQA dev | 74 / 5.155 (1,44%) | 19 / 1.676 | 6: 19 |
| ViTextVQA test full | 126 / 10.028 (**1,256%**) | 41 / 3.353 | 6: 41 |
| ViVQA train / test | 0 / 10.799 · 0 / 3.001 | 0 / 8.255 · 0 / 2.789 | — |

**Bước B — OCR lại 365 ảnh (165 ảnh có cờ xoay = toàn bộ ảnh train + test có cờ; 200 ảnh đối chứng):**

| Nhóm | Lệch cache (kiểu `fixed`) | Lệch cache (kiểu `asis`) |
|---|---|---|
| Có cờ xoay | **164 / 165 = 99,4%** [96,7; 99,9] | 0 / 165 |
| Đối chứng | **0 / 200 = 0%** [0; 1,9] | 0 / 200 |

- Kiểu `asis` trùng cache 100% ở cả hai nhóm → lỗi **tất định** do cách đọc ảnh, không phải do khác GPU. Cơ chế "OpenCV xoay, PIL không xoay" được xác nhận.
- Đáp án gold nằm trong OCR, 126 câu test ảnh có cờ xoay: cache **0,0%** → ảnh đã xoay **41,3%** (đối chứng: 29,2%, không đổi).
- `1849.jpg`, `14277.jpg` đều có cờ xoay 6. Cache là chữ rác (`0310000…`), ảnh đã xoay đọc ra biển hiệu tiếng Việt → giả thuyết (c) của W13 bước 8 đúng.
- ViVQA không có ảnh nào có cờ xoay → C3, SH trên ViVQA và POPE-vi **không** bị ảnh hưởng.

## R-EXIF bước 2 — luật ghi trước (commit TRƯỚC notebook 26 và 27)

**Sửa code (`src/data/ocr.py`, hàm `oriented_image`):**
- Ảnh có cờ xoay: xoay ảnh theo EXIF, lưu PNG tạm không còn cờ; Paddle dò và VietOCR cắt trên **cùng** ảnh đã xoay.
- Ảnh không có cờ: giữ nguyên đường cũ (Paddle đọc file gốc) → OCR của mọi ảnh không có cờ giữ nguyên như W08–W10.
- `engine_info` có thêm khoá `exif`.

**Phần A — cache mới (notebook 26, T4):** chép cache W10 sang `data/vitextvqa_official/ocr/paddledet_vietocr_exiffix.json` và OCR lại mọi ảnh có cờ xoay trong cache. Cache W10 giữ nguyên. **Hợp lệ** khi đủ cả 4 điều kiện:
- V1: 0 ảnh lỗi;
- V2: mọi ảnh không có cờ xoay trùng từng dòng cache W10;
- V3: ≥ 95% ảnh mẫu bước B có văn bản (160 ký tự đầu) trùng kiểu `fixed` của notebook 25;
- V4: engine trùng cache W10 ở `paddleocr` / `det_model` / `rec_model` / `vietocr_config` / `lang` và có khoá `exif`.

**Phần B — suy luận lại (notebook 27, L4):** B3-full (adapter W10, cấu hình OCR W08 VietOCR · v1 · 600 · 0) trên full test 10.028 câu với cache mới.
- **Hợp lệ (V5):** ≥ 99,5% câu có ảnh không cờ xoay cho dự đoán **trùng** run W10. Không đạt thì không dùng run.
- **Báo cáo (L2), cả hai số:**
  - full test và riêng 126 câu: EM, ANLS, ΔEM (McNemar chính xác, bootstrap CI95 10.000 lần, seed 42), ΔANLS (bootstrap CI95);
  - B2 → B3-full cũ / mới (McNemar).
- **Số chính thức (theo L3):** ViTextVQA vẫn là **B3-full W10** (EM 53,35 · ANLS 73,89), vì adapter không train lại. Bản mới là dòng độ nhạy "B3-full + OCR sửa EXIF" trong bảng kết quả và phần giới hạn.
- **Chỉ mô tả:** 126 câu với **ảnh đã xoay** + OCR đã sửa (đường ảnh của prototype) so với ảnh chưa xoay; EM của 9.902 câu không bị ảnh hưởng.

**Không chạy lại** các run OCR khác (B3 W08/W09 và 3 seed, A2/A3, SH test2000). Chúng dùng chung cache W10 nên chịu cùng lỗi ở 1,256% câu test và 1,28% câu train. So sánh giữa các run này vẫn công bằng (cùng lỗi ở cả hai phía), nhưng điều này phải ghi vào phần giới hạn.

**Prototype:** giao diện đã xoay ảnh có cờ EXIF và đưa PNG không còn cờ cho tiến trình OCR, nên OCR trực tiếp của prototype vốn đã đúng. Với bản sửa, `ocr_worker` cũng tự xoay nếu nhận ảnh còn cờ.

## File mới (bước 1, 01/11/2026)

| File | Nội dung | Kiểm |
|---|---|---|
| `src/data/exif_audit.py` | `--scan` / `--reocr` / `--analyze`; luật R-EXIF ở hằng `RULE` (trùng README này) | `--selftest`: đọc cờ EXIF, đếm theo split, ảnh sửa xoay đúng + không còn cờ, mẫu cố định gồm ảnh bắt buộc, vòng OCR lại với engine giả, 4 nhánh luật |
| `notebooks/25_exif_ocr_audit_vitextvqa.ipynb` | Run all trên T4: minh hoạ cơ chế → bước A → cài Paddle/VietOCR (như notebook 19) → kiểm sha256 cache + engine → bước B → bước C → khối gửi Claude (mục 8) → tự trả máy | — |

Kết quả ghi vào Drive `experiments/W14_exif_audit_vitextvqa/` (`exif_scan.json`, `exif_reocr.json`, `exif_audit_summary.json`, 2 log), sau đó Cursor chép lên git.

## File mới / đổi (bước 2, 01/11/2026)

| File | Nội dung | Kiểm |
|---|---|---|
| `src/data/ocr.py` | `exif_orientation_of`, `oriented_image`; `PaddleDetVietOCREngine.read` dò + cắt trên cùng ảnh đã xoay (chỉ ảnh có cờ); `engine_info["exif"]` | `--selftest` thêm `ocr exif fix OK` (ảnh có cờ → 300×400 + PNG không cờ; ảnh thường → đường cũ) |
| `src/data/exif_audit.py` | `--fix_cache` (V1–V4, biên bản), `--make_subset` (126 câu + ảnh đã xoay), `--compare` (V5, McNemar, bootstrap) | `--selftest` thêm `exif_audit step2 OK` |
| `src/eval/run_baseline.py` | tham số `--image_dir` (không vào `run_signature`) | `--selftest` OK |
| `notebooks/26_ocr_cache_exiffix_vitextvqa.ipynb` | T4: kiểm phán quyết notebook 25 → cài Paddle (bắt buộc GPU) → dựng cache mới → khối gửi Claude → tự trả máy | — |
| `notebooks/27_b3full_exiffix_testfull_qwen25vl.ipynb` | L4: kiểm biên bản + adapter + 2 run tham chiếu → eval full test → 126 câu ảnh xoay → so ghép cặp → khối gửi Claude → tự trả máy | — |
| `experiments/W14_exif_audit_vitextvqa/` | kết quả notebook 25 (chép từ Drive) | — |
