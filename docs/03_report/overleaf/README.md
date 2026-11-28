# Khoá luận trên Overleaf

Bản khoá luận CHÍNH được viết trên Overleaf (project riêng của Hoàng, tạo từ template của mentor, ngày 10/11/2026).
Google Doc "ViVQA-VLM Thesis (TV)" chỉ còn là bản lưu trữ của Chương I–II, không sửa tiếp.

## Cấu trúc (từ 27/11/2026: 8 chương theo thứ tự Report 1–7 của AIP491)
Giữ nguyên lớp LaTeX và 6 chương nghiên cứu của template mentor, chèn thêm Chương 2 (PMP) và Chương 5 (System Design) để
thứ tự khớp `docs/02_references/templates/Template_AIP491_CP_StudentsGuide.docx` (Final Report = bản final của Report #1–#7).

| Thư mục | Chương | Report AIP491 |
|---|---|---|
| Chapter1 | Giới thiệu / Introduction (§1.10 có bảng đối chiếu Report ↔ chương) | Report 1 |
| Chapter2 | Kế hoạch quản lý dự án / Project management plan (từ Phụ lục A cũ + §1.10–1.11 cũ; thêm quản lý thay đổi, đóng dự án) | Report 2 |
| Chapter3 | Nghiên cứu liên quan và hệ thống hiện có / Related work and existing systems | Report 3 |
| Chapter4 | Phương pháp / Methodology (Chương 3 cũ, bỏ §3.8) | Report 4 |
| Chapter5 | Thiết kế và hiện thực hệ thống / System design and implementation (mới; gồm §3.8 và §4.8 cũ) | Report 5 |
| Chapter6 | Kết quả / Results (Chương 4 cũ, bỏ §4.8) | Report 6 |
| Chapter7 | Thảo luận / Discussion | Report 6 |
| Chapter8 | Kết luận / Conclusion (+ mục Nhìn lại quá trình thực hiện) | Report 7 |
| Appendix1 | Phụ lục A: Gói tái lập / Reproducibility package (Phụ lục B cũ) | — |
- Hình mới: Chapter5/Figs/sd_architecture.pdf (vẽ bằng TikZ, mã nguồn ở Chapter5/Figs/src/, biên dịch riêng bằng pdflatex) và sd_interface.png (ảnh giao diện ở chế độ backend giả, từ docs/W13/figures/ui_mock_desktop.png).
- Nhãn LaTeX cũ được giữ (sec:intro-team, sec:intro-pm, sec:met-proto, sec:res-proto...) nên mọi \ref tự cập nhật số.

## Ngôn ngữ
- Viết tiếng Việt trước. Preamble/preamble.tex: `\thesisvitrue` = tiếng Việt, `\thesisvifalse` = tiếng Anh.

## Thư mục này
- source_vi/ = bản VI v3 (11/11/2026): tắt microtype expansion, ảnh Ch3 2600 px, thêm 3 dòng %\includeonly.
- ViVQA-VLM_Thesis_VI.pdf: bản PDF đầy đủ MỚI NHẤT của luận văn (chỉ giữ 1 file; bản cũ xem trong lịch sử git).
- Cập nhật snapshot: tải source zip mới từ Overleaf, giải nén đè source_vi/, commit.
- Template gốc của mentor: G:\My Drive\ViVQA-VLM\thesis\template_original\ (không đưa lên repo).

## Bản tiếng Anh (source_en)
- Dịch từ `source_vi` @ `de78d9e`; preamble `\thesisvifalse`.
- 7 hình vẽ lại bằng tiếng Anh; build giống `source_vi` (recipe Full).
- PDF: `ViVQA-VLM_Thesis_EN.pdf` (ghi đè sau mỗi Full build từ `source_en/thesis.pdf`).
- Sửa nội dung một bản thì sửa cả bản kia.

## Biên dịch trên Overleaf gói miễn phí
- Lần đầu biên dịch lần lượt từng dòng `\includeonly` (bỏ `%` ở đúng một dòng mỗi lần).
- Sau đó comment lại cả 3 dòng để biên dịch bản đầy đủ.
- Nếu vẫn hết giờ thì dùng ViVQA-VLM_Thesis_VI.pdf trong thư mục này.

## Viết trên máy (Cursor + MiKTeX)
- Trích dẫn hiện [?] hoặc thiếu Tài liệu tham khảo: sidebar TeX → Clean up auxiliary files, rồi Build with recipe → Full. (Nguyên nhân thường gặp: .aux dở dang do lượt build trước bị ngắt.)
- `Ctrl+S` = build Nhanh (1 vòng pdflatex).
- Sidebar TeX → Build with recipe → Full khi thêm `\cite`/`\label`/chương mới.
- `Ctrl+Alt+J` và Ctrl+click để nhảy giữa source và PDF.
- Sau mỗi lần sửa: build Full rồi GHI ĐÈ `thesis.pdf` lên `../ViVQA-VLM_Thesis_VI.pdf` và commit. Không tạo bản PDF có ngày.
- Overleaf gói miễn phí giới hạn 10 giây nên không còn dùng để build bản đầy đủ.
