# Khoá luận trên Overleaf

Bản khoá luận CHÍNH được viết trên Overleaf (project riêng của Hoàng, tạo từ template của mentor, ngày 10/11/2026).
Google Doc "ViVQA-VLM Thesis (TV)" chỉ còn là bản lưu trữ của Chương I–II, không sửa tiếp.

## Cấu trúc (theo template của mentor: 6 chương)
| Template | Nội dung | Trạng thái 11/11/2026 |
|---|---|---|
| Chapter 1 | Giới thiệu | Đã viết (VI) |
| Chapter 2 | Các nghiên cứu liên quan | Đã viết (VI) |
| Chapter 3 | Dữ liệu và phương pháp | Đã viết (VI) |
| Chapter 4 | Kết quả | Đã viết (VI) |
| Chapter 5 | Thảo luận | Đã viết (VI) |
| Chapter 6 | Kết luận | Đã viết (VI) |
| Appendix A | Kế hoạch quản lý dự án | Đã viết (VI) |
| Appendix B | Gói tái lập | Đã viết (VI) |
- Tóm tắt (Abstract) đã viết (VI). Trang Dedication đang tắt trong thesis.tex (template còn chữ mẫu).
- Hình: Chapter3/Figs (4 hình khổ ngang, sidewaysfigure, cần gói rotating đã bật trong preamble), Chapter4/Figs (2 biểu đồ PDF).
- Snapshot hiện tại: v3 (11/11/2026): tắt microtype expansion, ảnh Ch3 2600 px, thêm 3 dòng %\includeonly; biên dịch nhanh hơn trên Overleaf gói miễn phí.

## Ngôn ngữ
- Viết tiếng Việt trước. Preamble/preamble.tex: `\thesisvitrue` = tiếng Việt, `\thesisvifalse` = tiếng Anh.
- Bản tiếng Anh đã có của các phần đã duyệt nằm trong source_vi/_EN_draft/ (không được biên dịch).

## Thư mục này
- source_vi/ = bản VI v3 (11/11/2026): tắt microtype expansion, ảnh Ch3 2600 px, thêm 3 dòng %\includeonly.
- *_preview.pdf: bản PDF biên dịch thử tại thời điểm snapshot.
- Cập nhật snapshot: tải source zip mới từ Overleaf, giải nén đè source_vi/, commit.
- Template gốc của mentor: G:\My Drive\ViVQA-VLM\thesis\template_original\ (không đưa lên repo).

## Biên dịch trên Overleaf gói miễn phí
- Lần đầu biên dịch lần lượt từng dòng `\includeonly` (bỏ `%` ở đúng một dòng mỗi lần).
- Sau đó comment lại cả 3 dòng để biên dịch bản đầy đủ.
- Nếu vẫn hết giờ thì dùng preview PDF trong thư mục này.

## Viết trên máy (Cursor + MiKTeX)
- `Ctrl+S` = build Nhanh (1 vòng pdflatex).
- Sidebar TeX → Build with recipe → Full khi thêm `\cite`/`\label`/chương mới.
- `Ctrl+Alt+J` và Ctrl+click để nhảy giữa source và PDF.
- Chốt mốc = chép `thesis.pdf` thành preview có ngày rồi commit.
- Overleaf gói miễn phí giới hạn 10 giây nên không còn dùng để build bản đầy đủ.
