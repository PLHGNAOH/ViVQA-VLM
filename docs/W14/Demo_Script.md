# Kịch bản video demo ViVQA-VLM (W14, bước 4)

> Video dự phòng cho buổi bảo vệ: nếu Colab hoặc mạng hỏng lúc trình bày thì chiếu video này.
> Độ dài mục tiêu **3–4 phút**. Quay trên máy tính, khi notebook 28 mục 8 đang mở giao diện (30 phút).
> Ví dụ dùng trong video là các ví dụ do `src/prototype/demo_examples.py` chọn **bằng luật cố định**, kể cả một câu trả lời **sai**. Bảng ví dụ thật nằm ở `experiments/W14_prototype_live_v2_L4_seed42/demo/demo_examples.md`.

## Cách quay (Windows 11)

1. Mở link `https://….gradio.live` (notebook 28, mục 8) trong Chrome, phóng to cửa sổ, zoom trang 90–100%.
2. **Che thanh địa chỉ**: bấm F11 (toàn màn hình), vì link công khai ai có cũng dùng được.
3. Quay bằng **Snipping Tool**: Win + Shift + R → kéo chọn vùng cửa sổ Chrome → Start. Hoặc dùng Xbox Game Bar: Win + Alt + R để bắt đầu / dừng.
4. Nói tiếng Việt theo lời thoại bên dưới. Nếu không ghi âm được thì quay không tiếng; lời thoại để đọc lúc chiếu.
5. Lưu file `ViVQA-VLM_demo_W14.mp4` vào `G:\My Drive\ViVQA-VLM\demo\`. Video **không** đưa lên GitHub (dung lượng lớn).
6. Quay xong, nếu còn thời gian thì quay thêm một bản dự phòng thứ hai.

## Lời thoại theo cảnh

| # | Cảnh (thao tác) | Lời thoại gợi ý | ~Giây |
|---|---|---|---|
| 1 | Màn hình giao diện trống | "Đây là prototype ViVQA-VLM: hỏi đáp hình ảnh tiếng Việt. Hệ thống gồm Qwen2.5-VL-3B lượng tử hoá 4-bit, một adapter QLoRA dùng chung cho ViVQA và ViTextVQA, cùng OCR PaddleOCR + VietOCR để đọc chữ trong ảnh. Tất cả chạy trên một GPU L4." | 20 |
| 2 | Chỉ lần lượt các vùng | "Bên trái: tải ảnh và nhập câu hỏi. Bên phải: câu trả lời, **mức tin cậy** Cao / Trung bình / Thấp, và **bằng chứng OCR**: các vùng chữ được đánh số theo thứ tự đọc, dòng nào khớp câu trả lời thì tô xanh." | 20 |
| 3 | Bấm ví dụ `chu_trong_anh` #1 → Trả lời | "Câu hỏi cần đọc chữ trên biển hiệu. Model trả lời đúng, mức Cao. Ở khung OCR ta thấy dòng chữ chứa câu trả lời, tức là model dùng đúng phần chữ đọc được." | 30 |
| 4 | Ví dụ `chu_trong_anh` #2 | "Một ví dụ chữ trong ảnh khác. Đây là điểm mạnh chính của đề tài: QLoRA nhận biết OCR giúp tăng 3,1 điểm EM trên ViTextVQA so với QLoRA thường." | 20 |
| 5 | Bấm ví dụ `dem` (ảnh COCO), **xoá câu hỏi**, gõ tay: *"Trong ảnh có con mèo không?"* (chọn một vật **không** có trong ảnh). Ô `co_khong` trống vì ViVQA test không có câu có/không | "Câu có/không đi đường riêng — câu này tôi gõ tay. Ở W12 nhóm phát hiện adapter dùng chung trả lời 'có' quá nhiều. Prototype dùng ngưỡng τ* chọn trên tập hiệu chỉnh riêng, nên câu hỏi về vật không có trong ảnh được trả lời 'không'." | 25 |
| 6 | Ví dụ `dem` và `mau_sac` (nếu có) | "Hệ thống cũng trả lời câu đếm và câu về màu sắc trên ảnh đời thường của ViVQA." | 20 |
| 7 | Ví dụ `anh_xoay_exif` | "Ảnh điện thoại chụp dọc có cờ xoay EXIF. Giao diện xoay ảnh đúng chiều rồi mới đọc chữ. Tuần 14 nhóm phát hiện và sửa đúng lỗi này trong dữ liệu đánh giá." | 20 |
| 8 | Ví dụ `tra_loi_sai` | "Đây là một câu trả lời **sai**. Hệ thống báo mức tin cậy **Thấp** và hiện cảnh báo. Giao diện không hiện con số xác suất, vì model tự tin quá mức (ECE 0,198), nên chỉ hiện mức để người dùng không tin nhầm." | 25 |
| 9 | Tải 1 ảnh thật của nhóm (ô ví dụ ảnh thật) | "Ảnh thật do nhóm tự chụp, không có trong dataset." | 20 |
| 10 | Mở "Chi tiết kỹ thuật", rồi chỉ dòng disclaimer | "Mỗi câu mất khoảng 1 giây (p50) trên L4. Ảnh và câu hỏi không được lưu lại. Câu trả lời do AI sinh ra có thể sai, nên không dùng cho quyết định quan trọng." | 20 |

**Tổng ≈ 3 phút 40 giây.**

## Kiểm trước khi quay

- [ ] Xem trước ảnh từng ví dụ (bảng mục 6b). Ví dụ nào có nội dung không phù hợp để chiếu trước hội đồng thì **bỏ qua cảnh đó** và ghi lý do ở cuối file này. Không thay bằng câu đúng khác, để giữ tinh thần "không chọn tay".

- [ ] Notebook 28 mục 7 đã in khối gửi Claude (kết quả R-OCR v2 / R-LIVE v2 đã có).
- [ ] Bảng `demo_examples.md` đã in ở mục 6b; biết trước câu nào đúng, câu nào sai.
- [ ] Thanh địa chỉ đã che (F11).
- [ ] Còn ít nhất 15 phút trước khi giao diện tự tắt (30 phút từ lúc link hiện ra).

## Câu hội đồng có thể hỏi khi xem video

- "Ví dụ có phải chọn toàn câu đúng không?" → Không. Ví dụ do luật cố định chọn từ mẫu seed 42, gồm một câu sai mức Thấp. Luật nằm trong `demo_examples.py`.
- "Demo có đúng là hệ thống đã đánh giá không?" → Có: R-PAR 100% (W13), R-LIVE (W13 + W14 v2) so câu trả lời qua giao diện với run eval.
- "Nếu người dùng tải ảnh rất lớn thì sao?" → Ảnh lớn hơn 6.000 px được thu nhỏ trước khi xử lý; giao diện ghi rõ trong cảnh báo.
