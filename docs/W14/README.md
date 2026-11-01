# W14 · Hoàn thiện code: dữ liệu + prototype + demo

> Tuần ảo W14 (bắt đầu 01/11/2026, tuần thực 9). Hội đồng cấp Khoa là **tuần thực 13 (30/11–06/12)**.
> **Phạm vi (Hoàng, 01/11/2026):** tuần này chỉ làm phần **code**: xử lý dữ liệu, prototype web, bản demo hoàn chỉnh. Thesis, slide và sửa theo góp ý Review 2 **dời sang tuần sau**. Deck đang lưu trên Canva: khi nào làm slide, Hoàng tải về rồi Claude cập nhật và lưu đúng chỗ.
> Colab đầu tuần: đọc ở **Colab → Resources** khi mở notebook 25 và ghi vào đây.

## Việc trong tuần

| # | Việc | Ước tính | Trạng thái |
|---|---|---|---|
| 1 | **R-EXIF**: kiểm cờ xoay EXIF có làm hỏng OCR cache ViTextVQA không — luật (README này) + `src/data/exif_audit.py` + notebook `25_exif_ocr_audit_vitextvqa.ipynb` (T4) | 3 h (≈ 1 h T4) | ⏳ code xong, chờ chạy |
| 2 | Làm theo phán quyết R-EXIF: chỉ ghi giới hạn · hoặc sửa `ocr.py` + OCR lại ảnh lỗi + suy luận lại B3-full · hoặc thêm train lại B3-full | 0,5 h / ≈ 5 h / ≈ 20 h | ⏳ |
| 3 | Prototype: sửa đường ảnh (OCR trên ảnh độ phân giải gốc, xử lý EXIF thống nhất với eval) + kiểm lại R-OCR / R-LIVE (luật ghi trước) | 3 h (≈ 1 h L4) | ⏳ |
| 4 | Demo hoàn chỉnh: bộ ví dụ chọn sẵn, kịch bản demo, quay video dự phòng | 2,5 h | ⏳ |
| 5 | Gói tái lập phần code: ghim phiên bản thư viện, script suy luận dòng lệnh, README chạy lại, manifest adapter | 3 h | ⏳ |
| 6 | Đóng tuần: README + Team Report, dọn Downloads + `G:\My Drive\ViVQA-VLM` | 1,5 h | ⏳ |
| | **Tổng** (theo nhánh của bước 2) | **≈ 13,5 h / ≈ 18 h / ≈ 33 h** | |

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

## File mới (bước 1, 01/11/2026)

| File | Nội dung | Kiểm |
|---|---|---|
| `src/data/exif_audit.py` | `--scan` / `--reocr` / `--analyze`; luật R-EXIF ở hằng `RULE` (trùng README này) | `--selftest`: đọc cờ EXIF, đếm theo split, ảnh sửa xoay đúng + không còn cờ, mẫu cố định gồm ảnh bắt buộc, vòng OCR lại với engine giả, 4 nhánh luật |
| `notebooks/25_exif_ocr_audit_vitextvqa.ipynb` | Run all trên T4: minh hoạ cơ chế → bước A → cài Paddle/VietOCR (như notebook 19) → kiểm sha256 cache + engine → bước B → bước C → khối gửi Claude (mục 8) → tự trả máy | — |

Kết quả ghi vào Drive `experiments/W14_exif_audit_vitextvqa/` (`exif_scan.json`, `exif_reocr.json`, `exif_audit_summary.json`, 2 log), sau đó Cursor chép lên git.
