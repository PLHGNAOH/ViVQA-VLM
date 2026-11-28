# Hướng dẫn gán nhãn loại câu hỏi ViTextVQA (W11, bước 6)

**Mục đích:** kiểm xem bộ phân loại theo luật `src/eval/question_types.py` (phiên bản `qt-v1`, đóng băng ngày 21/10/2026) gán loại câu hỏi đúng đến đâu. Hoàng tự gán nhãn tay **150 câu test2000** (chọn ngẫu nhiên, seed 42), sau đó máy so nhãn tay với nhãn máy.

## Quy tắc khi gán nhãn

1. Chỉ đọc **câu hỏi**. Phiếu không có ảnh, đáp án, dự đoán của model hay nhãn máy, và **đừng tìm xem** những thứ đó. Gán nhãn "mù" như vậy thì phép kiểm mới khách quan.
2. Chọn loại mô tả đúng nhất **câu trả lời mà câu hỏi đòi**, không phải chủ đề của ảnh.
3. Nếu câu hỏi hợp với **hai loại**, chọn loại **đứng trước** trong bảng dưới.
   - Ví dụ: "số điện thoại của cửa hàng tên gì" hợp cả `contact` lẫn `name` → chọn `contact`.
4. Không chắc thì vẫn chọn một loại, rồi ghi lý do vào cột `note`.
5. Ghi đúng **tên loại** trong cột `label`. File Excel có sẵn danh sách thả xuống.

## 9 loại (thứ tự ưu tiên từ trên xuống)

| # | Nhãn | Câu hỏi đòi gì | Ví dụ (lấy từ **dev**, không phải test) |
|---|---|---|---|
| 1 | `source` | nguồn của ảnh: báo, kênh truyền hình, trang web (thường là watermark) | "bức ảnh này lấy nguồn từ đâu ?" · "bức hình này của báo nào ?" |
| 2 | `contact` | số điện thoại, hotline, fax, zalo, website, email | "số điện thoại của quán cắt tóc này là gì ?" · "website của siêu thị là gì ?" |
| 3 | `number_time` | một con số hoặc mốc thời gian: giá, số lượng, số nhà, biển số, mã số, ngày/năm/giờ, lần thứ, phần trăm | "dưa hấu có giá bao nhiêu ?" · "hình ảnh được chụp vào năm nào ?" |
| 4 | `person` | tên **người**: tác giả, người dịch, chủ biên, "ai" | "tác giả của cuốn sách này là ai ?" |
| 5 | `location` | địa chỉ, nơi chốn, tỉnh/quận/đường nào, đi đến đâu, hướng nào | "địa chỉ quán chè hẻm là gì ?" · "sở y tế này thuộc tỉnh nào ?" |
| 6 | `name` | **tên riêng** của cửa hàng, sản phẩm, sách, công ty, thương hiệu, tổ chức | "cửa hàng này có tên là gì ?" · "kệ sữa trong hình là của hãng nào ?" |
| 7 | `text_content` | đọc lại một **dòng chữ / nội dung / khẩu hiệu / thông báo** | "dòng chữ ghi trên bảng đen là gì ?" · "biển này yêu cầu điều gì ?" |
| 8 | `product_service` | nơi đó **bán gì / làm dịch vụ gì / là loại cửa hàng gì** | "cửa hàng này bán cái gì ?" · "trend smile là cửa hàng gì ?" |
| 9 | `other` | còn lại: sự kiện, mô tả, suy luận, "như thế nào", "mục đích gì" | "cuốn sách này được viết nhằm mục đích gì ?" · "đây là chương trình gì ?" |

## Sau khi gán xong

Lưu file Excel rồi gửi lại cho Claude. Claude sẽ chuyển file thành `docs/W11/qt_labels_150.csv` và chạy `--score`:
- độ chính xác tổng;
- theo từng loại: **precision** (trong các câu máy gán loại T, bao nhiêu % Hoàng cũng gán T) và recall.

Loại nào có precision < 80% (và có ít nhất 5 câu trong mẫu) thì bị gắn cờ **"nhãn kém tin cậy"** trong bảng EM/ANLS theo loại.
