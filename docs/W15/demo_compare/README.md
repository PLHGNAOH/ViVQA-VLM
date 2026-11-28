# Luật chọn ví dụ demo so sánh (chốt 28/11/2026, trước khi chạy)

Luật này ghi **trước** khi xem câu trả lời trên laptop. Sau khi có kết quả: ô nào không có item đạt thì để trống và ghi rõ. **Không sửa luật** để lấp ô.

## Ứng viên chào hàng

Một item là ứng viên khi có **đủ** các điều kiện sau:

- `sh_stable`: ba câu trả lời của mô hình nhóm (ba lần gọi) giống hệt nhau.
- Mô hình nhóm đúng: EM = 1. Ảnh dataset dùng đáp án trong file test. Ảnh thật dùng đáp án trong `cau_hoi.csv` (nhiều cách viết tách bằng `|`).
- Mô hình gốc sai: EM của B1 = 0.
- Với **ViTextVQA**, thêm: EM của gốc + OCR (B1-OCR) = 0.
- Mức tin cậy của mô hình nhóm khác "Thấp".
- Tổng thời gian ≤ 8 giây.
- Với **ViVQA**, ảnh không nằm trong tập train.

Lỗi sinh (không có câu trả lời) **không** tính là EM = 0. Item đó không phải ứng viên.

## Cách tính (cùng lúc với luật, trước khi chạy)

- EM = `src.eval.metrics.exact_match` (1.0 hoặc 0.0).
- Mức tin cậy lấy ở lượt mô hình nhóm đầu tiên.
- Tổng giây = `sec_total` của lượt mô hình nhóm đầu (gồm OCR và mô hình nhóm) + giây mô hình của B1 + giây mô hình của B1-OCR. Ba lần lặp chỉ để kiểm `sh_stable`, không cộng ba lần vào tổng.
- ViVQA trùng train: tên file ảnh hoặc `img_id` có trong `train.json`.
- "Ảnh thật có chữ": nguồn là ảnh thật và lượt OCR trả về ít nhất một dòng.
- "Câu có/không": `is_yes_no_question` trong `src.prototype.core`.

Khớp câu tham chiếu Colab (B1 / B1-OCR / SH) **không** nằm trong luật chọn. Lệch vì laptop đọc OCR trên CPU so với cache Colab chỉ được ghi lại.

## Thứ tự ô — item đầu tiên đạt luật

Đi lần lượt các ô. Trong mỗi ô lấy **item đầu tiên** đạt luật, theo đúng thứ tự liệt kê. Item đã chọn cho ô trước thì ô sau bỏ qua.

1. **anh_that_chu** — ảnh thật có chữ. Thứ tự: các dòng `cau_hoi.csv` của 20 ảnh thật (ảnh thật đứng trước toàn bộ câu dataset).
2. **so_lieu** — `test_6734`, `test_9245`, `test_11314`.
3. **dau_thanh** — `test_85`.
4. **goc_tra_loi_tieng_anh** — `test_1315`, `test_5081`, `test_631`, `test_9472`.
5. **goc_bia** — `test_10580`.
6. **day_du_2x2** — item ViTextVQA đạt luật đầu tiên còn chưa dùng ở ô khác. Thứ tự id ViTextVQA: `test_85`, `test_6734`, `test_9245`, `test_11314`, `test_10580`, `test_15656`, `test_6945`, `test_10736`, `test_5081`, `test_9472`, `test_3037`, `test_11785`.
7. **co_khong** — ảnh thật, câu có/không. Cùng thứ tự dòng trong `cau_hoi.csv`.

Ô không có item đạt → để trống, ghi rõ trong bảng kết quả.

## Bảng thứ tự demo sau khi chạy (28/11/2026)

Luật phía trên không sửa. Ô trống vì không có item nào đạt đủ điều kiện (phần lớn vì tổng giây > 8; trung bình đo được 12,86 s).

| Thứ tự | Ô | Item | Câu hỏi |
|---|---|---|---|
| 1 | anh_that_chu | trống — không có item đạt luật | |
| 2 | so_lieu | trống — không có item đạt luật | |
| 3 | dau_thanh | trống — không có item đạt luật | |
| 4 | goc_tra_loi_tieng_anh | `vivqa_test_1315` | cái gì kiểm tra tảng đá trong bao vây của nó |
| 5 | goc_bia | trống — không có item đạt luật | |
| 6 | day_du_2x2 | trống — không có item đạt luật | |
| 7 | co_khong | trống — không có item đạt luật | |
