# Review 2 — Góp ý của hội đồng (29/10/2026)

Nguồn: bản ghi âm buổi Review 2 (17 phút 58 giây, Hoàng ghi). File ghi âm và bản chép lời thô lưu trên Drive `G:\My Drive\ViVQA-VLM\reviews\` (không đưa lên git vì repo public và có giọng giảng viên).
Cách chép lời: nhận dạng giọng nói tự động (sherpa-onnx zipformer tiếng Việt + Whisper turbo), Claude đọc lại và gom ý. Chất lượng âm thanh phần giảng viên nói thấp, nên các mục đánh dấu **(cần nghe lại)** phải được Hoàng nghe lại đúng mốc thời gian trước khi điền checklist.

## Phần A — Góp ý Review 1 (Hoàng nhắc lại đầu buổi, 00:52)
| # | Góp ý Review 1 | Đã xử lý trong deck Review 2 |
|---|---|---|
| A1 | Trình bày rõ input → output của bài toán | Slide 3 |
| A2 | Vẽ được pipeline của nhóm | Slide 10 (pipeline 8 bước, khối A/B/C) |
| A3 | Nói rõ sản phẩm đầu ra | Slide 11 |

## Phần B — Góp ý Review 2 (giảng viên phản biện, 09:04 → 17:38)
| # | Mốc | Slide bị góp ý | Nội dung góp ý (diễn đạt lại) | Hướng xử lý cho deck cuối + thesis |
|---|---|---|---|---|
| G1 | 09:04–09:44 | 15 Kiến trúc baseline | Phải làm rõ **số lượng token**: ảnh, câu hỏi (và OCR) mỗi phần cần bao nhiêu token, **sức chứa** (context window) của model là bao nhiêu, có chạy nổi không. (cần nghe lại: ví dụ con số giảng viên đưa ra) | Thêm "ngân sách token" đo trên dữ liệu thật: token ảnh (256–512), token prompt + câu hỏi, token dòng OCR (max 600 ký tự), token sinh (≤ 32) — phân phối p50/p95/max so với context window ghi trong `config.json` của Qwen2.5-VL-3B |
| G2 | 11:27–12:05 | 18 Xử lý dữ liệu | Phải **nói đúng tên kỹ thuật và đúng định lượng** (regex, chuẩn hoá khoảng trắng, bỏ ký tự đặc biệt…), không mô tả theo cách tự nghĩ ra | Bảng các bước `normalize_vi`: tên kỹ thuật (Unicode NFC, lowercase, regex bỏ dấu câu, gộp khoảng trắng) + số câu trả lời bị thay đổi ở từng bước (đếm trên dữ liệu) |
| G3 | 12:12–14:21 | 19 Giao thức đánh giá | **Chưa trình bày phương pháp tính metric.** Tiếng Việt có **từ ghép** ("thương mại điện tử"), tiếng Anh chủ yếu từ đơn → tách từ thế nào, tính toán thế nào, so sánh với các nghiên cứu tiếng Anh ra sao; có thể dùng công cụ/tri thức có sẵn để hiểu từ | Slide công thức EM, ANLS, token-F1 (cấp âm tiết) + giải thích: EM so cả chuỗi, ANLS theo ký tự → không phụ thuộc cách tách từ; token-F1 phụ thuộc → đo lại F1 sau tách từ tiếng Việt (vd. underthesea / pyvi) để cho thấy thứ hạng B1 < B2 < B3-full không đổi |
| G4 | 14:23–14:58 | 14, 15 (Existing System) | Mới chỉ nói overview; chưa làm rõ **các nghiên cứu trước đi theo lộ trình nào, khó khăn của họ là gì**; kết quả **cao hơn SOTA thì phải đưa bằng chứng vì sao làm tốt hơn** | Slide hệ thống hiện có: ViVQA baseline, ViTextBLIP-2 (cách làm, điểm yếu); slide "vì sao cao hơn": so cùng test, cùng EM + F1 (53,4 / 73,7 so với 25,5 / 54,0), nêu khác biệt chuẩn hoá đáp án là giới hạn; đưa P2–P3 phụ lục vào phần chính |
| G5 | 15:04–15:30 | 16, 19 (cấu hình) | Cấu hình bf16 / 4-bit, **512 token ảnh: tại sao 512 mà không phải số khác**, tiếng Việt khác ở đâu → phải giải thích rõ (cần nghe lại 15:04–15:18) | Bằng chứng đã có: R0 (bf16, đủ token) 79,6 vs R0b (4-bit, 512) 75,2, VRAM 10,9 → 2,5 GB. Có thể bổ sung ablation số token ảnh lúc suy luận (256 / 512 / 1024) — chờ Hoàng quyết |
| G6 | 15:30–16:00 | 9 Dữ liệu | Dữ liệu chủ yếu lấy có sẵn; **chưa đánh giá dữ liệu còn thiếu gì, nhóm bổ sung gì** (hoặc chứng minh đã đủ) | Slide chất lượng dữ liệu: gold nhiễu 12,5% (ViVQA 19%), trùng ảnh ViVQA train/test, ViVQA không có câu có/không, EXIF 1,26% câu test; phần nhóm bổ sung: lớp OCR (15.352 ảnh), val sạch 688 câu, bộ POPE-vi 4.985 câu, sửa EXIF |
| G7 | 16:00–17:17 | chung | Các khẳng định phải **có cơ sở (paper) và đo lường được**, ví dụ xử lý dấu thanh phải đo ra con số (cần nghe lại 16:05–16:55) | Slide "vấn đề tiếng Việt — đã đo": PaddleOCR 27,0% vs VietOCR 37,7% giữ đúng dấu (p = 0,0008); 9,0% lỗi B2 là sai dấu; 11,5% câu OCR đúng chữ sai dấu; code-switching "no" 17,7% (POPE-vi) |
| G8 | 17:28–17:38 | — | **Hướng nghiên cứu được ghi nhận** (đi đúng hướng) | Giữ nguyên hướng: một đóng góp = QLoRA nhận biết OCR |

## Ghi chú
- Các đoạn 03:23–04:16 có câu hỏi xen của giảng viên (hỏi về phần quay/chạy live) — không phải góp ý nội dung.
- Thứ tự ưu tiên khi sửa deck: G3 → G1 → G2 → G4 → G7 → G6 → G5 (G3, G1 là hai chỗ giảng viên nói lâu nhất).
- Sau khi Hoàng nghe lại các mục "cần nghe lại", cập nhật file này rồi mới điền checklist AI_Capstone_Review123 (sheet Review2).
