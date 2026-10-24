# Hướng dẫn gán nhãn — Audit lỗi W12 (ViVQA-VLM)

**Người gán:** Văn Quang Trường (SV4 · Evaluation / Hallucination) · **Thời gian:** khoảng 1 giờ 40 phút (≈ 40 giây/dòng), chia nhiều lượt được.

## 1. Việc này để làm gì?

Hai mô hình chính thức của nhóm trả lời sai EM ở một số câu: **B3-full** (QLoRA + OCR, ViTextVQA) và **C3** (QLoRA, ViVQA). Máy chỉ so chuỗi nên không biết:
- câu nào **sai EM nhưng thật ra đúng ý**,
- câu nào **gold sai**,
- câu nào mô hình **bịa** ra thứ không có trong ảnh (hallucination).

Chỉ người xem ảnh mới trả lời được. Kết quả dùng cho phần Error analysis và Hallucination analysis của báo cáo. Bộ nhãn **giống hệt audit W07** mà Trường đã làm.

## 2. File nhận được

`W12_audit_150_cho_Truong.zip` gồm:
- `W12_audit_150.xlsx`: phiếu 150 dòng (100 câu ViTextVQA + 50 câu ViVQA);
- thư mục `images/`: ảnh của từng câu;
- file hướng dẫn này.

## 3. Các bước

1. **Giải nén toàn bộ zip** ra một thư mục, ví dụ `Desktop\W12_audit`. Không mở xlsx khi còn nằm trong zip, vì link ảnh sẽ không chạy.
2. Mở `W12_audit_150.xlsx` bằng **Microsoft Excel** và đọc sheet `huong_dan`.
3. Sang sheet `error_audit`. Mỗi dòng:
   - bấm link ở cột `anh` để mở ảnh;
   - đọc `question`, `gold` (đáp án chuẩn của dataset; nếu có nhiều đáp án thì ngăn bằng `||`) và `du_doan` (câu trả lời của mô hình);
   - chọn 3 ô màu vàng bằng dropdown: `loai_loi`, `dung_y`, `hallucination`. Ghi chú vào `ghi_chu` khi cần.
4. Bấm **Ctrl+S** thường xuyên. Khi xong, **đổi tên file** thành `W12_audit_150_Truong.xlsx` và gửi lại cho Hoàng (chỉ cần file xlsx, không cần gửi ảnh).

## 4. Ba cột cần điền

| Cột | Điền gì |
|---|---|
| `loai_loi` | **Một** nguyên nhân chính (danh sách ở mục 5) |
| `dung_y` | **Có** nếu bạn chấp nhận câu trả lời là đúng ý (dù EM chấm sai) · **Không** nếu sai thật |
| `hallucination` | **Có** nếu câu trả lời nói một thứ **không hề có** trong ảnh (vật / chữ / số không xuất hiện) · Đọc nhầm một chữ **có thật** trong ảnh thì là **Không** |

## 5. Bộ nhãn `loai_loi` và ví dụ

| Gặp trường hợp | `loai_loi` | `dung_y` | `hallucination` |
|---|---|---|---|
| Gold "2021", trả lời "năm 2021" | Định dạng / độ dài (ý đúng) | Có | Không |
| Gold "máy tính", trả lời "laptop", ảnh đúng là laptop | Đồng nghĩa / khác từ (ý đúng) | Có | Không |
| Ảnh ghi "Phở Sưa", trả lời "Phở Sứa" | Sai dấu thanh | Không | Không |
| Ảnh ghi "0909 123 456", trả lời "0909 128 456" | Đọc chữ sai (OCR) | Không | Không |
| Trả lời một tên / số không hề có trong ảnh | Sai vùng / không có trong ảnh (hallucination) | Không | **Có** |
| Đọc đúng một dòng chữ có thật nhưng không phải dòng được hỏi | Sai vùng / không có trong ảnh (hallucination) | Không | Không |
| Trả lời bằng tiếng Anh khi gold là tiếng Việt | Code-switching (tiếng Anh) | tuỳ ý nghĩa | Không |
| Ảnh có 4 con mèo, trả lời "ba" | Đếm sai | Không | Không |
| Câu trả lời đúng với ảnh, gold mới sai | Gold sai / nhiễu | Có | Không |
| Không rơi vào nhãn nào / không chắc | Khác (+ ghi chú) | theo ý bạn | theo ý bạn |

Lưu ý: nhãn "Sai vùng / không có trong ảnh" gộp cả *nhìn sai chỗ* lẫn *bịa*. Cột `hallucination` mới là nơi tách riêng trường hợp **bịa**.

## 6. Quy tắc (quan trọng)

- **Gán mù:** chỉ dựa vào ảnh, câu hỏi, gold và câu trả lời trong phiếu. Không tra cứu kết quả hay file phân tích nào khác của nhóm, và không hỏi người khác câu này "nên" là nhãn gì.
- Mỗi dòng chọn **đúng 1** `loai_loi`. Dòng khó thì chọn **Khác** và ghi chú, **không đoán**.
- **Không** sửa, xoá, sắp xếp lại, hay thêm dòng/cột ở các cột không tô vàng. Thứ tự dòng được kiểm bằng máy khi chấm.
- Điền đủ cả 150 dòng × 3 cột. Ô trống sẽ bị máy báo lỗi khi chấm.
