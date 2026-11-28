# W15 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Nguồn số liệu: các thư mục `experiments/` ghi ở từng mục và luận văn `docs/03_report/overleaf/`.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W15 — Xử lý góp ý Review 2, luận văn VI + EN, demo chạy trên laptop, deck bảo vệ, gói nộp EduNext |
| Thời gian thực | 02/11 – 23/11/2026 (tuần thực 9–12) |
| Mốc kế tiếp | Hội đồng cấp Khoa: 16:00 thứ Hai 30/11/2026, P.211 Campus CNC · Nộp EduNext: 10:00 08/12 → 10:00 12/12/2026 · Bảo vệ lần 1: 19–21/12/2026 |
| Phần cứng | Colab CPU (notebook 29) · laptop RTX 3050 Ti Laptop 4 GB, i5-12500H, RAM 16 GB (demo, ảnh thật) · máy local + MiKTeX (build luận văn) |
| CU Colab | 207,21 (02/11/2026, sau gia hạn); tuần này chỉ dùng CPU. Đọc số thật ở Colab → Resources |
| Commit cuối tuần | xem mục "Dấu vết commit" |

## Mục tiêu

1. Trả lời đủ 8 góp ý Review 2 (G1–G8) bằng số đo, không chỉ bằng lời.
2. Viết xong luận văn bản tiếng Việt, rồi bản tiếng Anh (bản nộp).
3. Demo chạy **offline trên laptop 4 GB VRAM**, cho kết quả trùng bản đã đánh giá trên Colab.
4. Có deck bảo vệ hoàn chỉnh (VI + EN, có speaker notes) và gói nộp EduNext.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | Tổng hợp góp ý Review 2 từ bản ghi âm buổi 29/10 | `docs/W15/Review2_Feedback.md`: 8 góp ý G1–G8 + 3 góp ý Review 1 nhắc lại, kèm hướng xử lý |
| 2 | G1 — đo số token ảnh / câu hỏi / OCR và sức chứa model (`src/eval/token_budget.py`, notebook 29, Colab CPU, 04/11) | Chuỗi dài nhất 1.058 token = **0,83%** context 128.000; kiểm lại 1.200 câu bằng đường processor đầy đủ: lệch 0 |
| 3 | G2–G3 — tên kỹ thuật chuẩn hoá, từ ghép khi tính metric (`src/eval/metric_tokenization.py`) | Chuẩn hoá đối xứng, giữ dấu thanh; tách từ không đổi EM, không đổi thứ hạng hay chênh lệch giữa các run |
| 4 | G5 — vì sao 512 token ảnh, vì sao bf16 | Dùng bằng chứng có sẵn W06/W07 (không chạy thêm) |
| 5 | Luận văn tiếng Việt: Overleaf v1 (10/11) → v2 đủ Chương 1–6, Tóm tắt, Phụ lục A–B (11/11) → v3 build local bằng MiKTeX | `ViVQA-VLM_Thesis_VI.pdf` (110 trang) |
| 6 | Luận văn tiếng Anh `source_en` dịch từ bản VI (15/11); 7 hình vẽ lại bằng tiếng Anh | `ViVQA-VLM_Thesis_EN.pdf` (118 trang) — bản nộp |
| 7 | Ca minh hoạ "chữ OCR không liên quan" (§5.8, VI + EN, 15/11) và phân tích lật đáp án trên ViVQA | SH so với C1 trên 1.328 câu ViVQA có OCR: **+0,15 EM, p = 0,90** (không khác biệt) |
| 8 | Prototype chạy local trên Windows (13–19/11): Qwen 4-bit trên GPU + OCR trên CPU, chế độ offline, `demo.cmd` một lệnh, OCR chạy nền + cache, khởi động song song, `.\demo safe`, giao diện sáng cho máy chiếu, ẩn bảng ví dụ | Câu trả lời trùng bản Colab 20/20; xem số đo ở mục Kết quả |
| 9 | Ảnh thật trên laptop + chọn bộ ảnh demo (16–19/11) | 14/20 đúng; bộ demo v3 gồm 13 ảnh, đã kiểm không nằm trong tập train; thư mục `demo_images/` chỉ để local (gitignore) |
| 10 | Deck bảo vệ: draft 2 (10/11), draft 3 (11/11) → 3 phương án deck độc lập, mỗi phương án có bản VI và EN, speaker notes 100% (19/11); dọn `docs/04_slides` còn link Canva + 6 file | Đối chiếu tự động: mọi con số trên slide đều có trong luận văn hoặc `experiments/`; đã sửa 3 lỗi diễn đạt (mục 81% là cận trên, tên run B3-full, H1 ghi rõ C1 một epoch) |
| 11 | Sửa 2 chỗ trong luận văn (VI + EN, 23/11) | Qwen2-VL-7B zero-shot ghi rõ 2 cấu hình (cấu hình bài báo tham chiếu 16,7 EM / cấu hình B1 16,2 EM); Bảng 4.9 cột kích thước adapter đổi đơn vị đúng là MiB |
| 12 | Gói nộp EduNext `submission_edunext/` (local, gitignore, 23/11) | 4 mục khớp 4 thư mục EduNext; `Source_Code.zip` (git archive, đã quét không lọt ảnh thật hay token), `Adapter_SH.zip` (sha256 khớp), `Progress_Reports.zip` |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| VLM / Baseline | G1 token budget; G5 cấu hình 512 token / bf16; Qwen2-VL-7B hai cấu hình chạy |
| Dataset / OCR | G6 chất lượng dữ liệu; ca OCR không liên quan; bộ ảnh demo không trùng train |
| PEFT / Training | Bảng 4.9 (ablation LoRA) đơn vị MiB; adapter SH đóng gói kèm sha256 |
| Evaluation / Deployment | G2–G3 chuẩn hoá và tách từ; prototype local; đo độ trễ; deck và gói nộp |

### Definition of Done

| Tiêu chí | Trạng thái |
|---|---|
| G1–G8 đều có slide trả lời, ánh xạ G → số slide | Đạt |
| Luận văn VI + EN build đầy đủ, không lỗi tham chiếu | Đạt |
| Demo offline trên laptop trùng bản đã đánh giá | Đạt (20/20) |
| Deck VI + EN có speaker notes trên mọi slide | Đạt |
| Gói nộp đủ 4 mục EduNext | Đạt (chờ chốt 1 deck) |

## Kết quả / số liệu

### 1. G1 — ngân sách token (nguồn `experiments/W15_token_budget/`)

| Tập (prompt) | Token ảnh p50 / max | Input p50 / p95 / max | Chuỗi dài nhất | % context |
|---|---|---|---|---|
| ViVQA test (không OCR, C3) | 345 / 506 | 425 / 532 / 587 | 619 | 0,48 |
| ViTextVQA test (không OCR, B2) | 480 / 512 | 554 / 578 / 598 | 630 | 0,49 |
| ViTextVQA test (OCR, B3-full) | 480 / 512 | 610 / 810 / 1.026 | 1.058 | 0,83 |

- Context window 128.000 token; KV cache 36.864 byte/token → chuỗi dài nhất chỉ cần khoảng 37 MB.
- Token ảnh chiếm 73–87% input. Giới hạn thật là VRAM và thời gian train (1.280 token ảnh chậm 1,91× ở W07), không phải context window.

### 2. G2–G3 — chuẩn hoá và tách từ (nguồn `experiments/W15_metric_tokenization/`)

- Bậc chuẩn hoá tác dụng lớn nhất: ViTextVQA là regex bỏ dấu câu + gộp khoảng trắng (25,7% gold đổi); ViVQA là bỏ tiền tố "màu " (20,9%) và số chữ → chữ số (14,8%).
- Model QLoRA gần như không phụ thuộc chuẩn hoá (C3: 76,64 EM ở mọi bậc); zero-shot phụ thuộc nhiều (B1 ViVQA 1,70 → 22,09).
- Từ ghép: 41,8% gold ViTextVQA, 20,9% gold ViVQA. Tách từ không đổi EM; token-F1 theo từ thấp hơn 1–3 điểm nhưng chênh lệch giữ nguyên (B2 → B3-full: +2,57 theo âm tiết, +2,60 theo từ).

### 3. Prototype trên laptop (nguồn `experiments/W15_REAL_local_sh/`, `src/prototype/README.md`)

| Chỉ số | Giá trị |
|---|---|
| Trùng câu trả lời với bản Colab | 20/20 |
| Ảnh thật đúng | 14/20 (có/không 5/5, câu mở 9/15) |
| Thời gian mỗi câu khi OCR đã chạy nền | ≈ 0,93 s |
| Hỏi ngay sau khi tải ảnh | ≈ 7,39 s |
| Khởi động tới khi sẵn sàng | ≈ 27 s (song song) · 47,9 s (tuần tự, `.\demo safe`) |
| VRAM đỉnh / RAM đỉnh | ≈ 2,40 GB / 13,5 trên 15,7 GB |

### 4. Đối chiếu deck với luận văn

- 3 phương án deck: 57 / 35 / 51 slide, cặp VI–EN cùng số slide, notes 100%.
- Quét tự động mọi con số trên slide: đều có trong luận văn hoặc `experiments/` (chỉ "59.000 câu" là làm tròn của 58.987).

## Kiến thức rút ra trong tuần

- Context window không phải nút cổ chai: chuỗi dài nhất chỉ 0,83% sức chứa. Lý do chặn 512 token ảnh là chi phí train và VRAM.
- Báo cáo EM phải kèm bậc chuẩn hoá: zero-shot thay đổi tới 20 điểm tuỳ chuẩn hoá, QLoRA thì gần như không đổi.
- "81% lỗi khi OCR không chứa đáp án" chỉ là **cận trên**; duyệt tay cho thấy lỗi đọc chữ thật khoảng 36% số lỗi, tức khoảng 17 điểm EM.
- Chữ OCR không liên quan có thể làm sai câu không cần đọc chữ, nhưng mức tin cậy Thấp cảnh báo được, và trên toàn tập không làm giảm EM (+0,15, p = 0,90).
- Cùng một model, hai cấu hình chạy cho kết quả khác nhau (Qwen2-VL-7B 16,7 / 16,2): luận văn phải ghi rõ cấu hình đi kèm mọi con số.
- Đơn vị phải ghi đúng: MiB (chia 1024²) khác MB (chia 10⁶).

## Câu hỏi hội đồng có thể hỏi (phần W15)

1. "Model nhận bao nhiêu token, có vượt sức chứa không?" → Dài nhất 1.058 token = 0,83% của 128.000; token ảnh chiếm 73–87%.
2. "Tiếng Việt có từ ghép, tính metric thế nào?" → EM so cả chuỗi nên không bị ảnh hưởng; token-F1 tính cả hai cách, thứ hạng không đổi.
3. "Demo trên laptop có giống kết quả đánh giá không?" → Cùng adapter, cùng prompt builder, trùng 20/20 câu.
4. "Vì sao chọn 512 token ảnh?" → 1.280 token không tăng EM có ý nghĩa (p = 0,13) mà train chậm 1,91×, VRAM 6,71 → 10,84 GB.

## Blocker / vấn đề mở

| # | Vấn đề | Hướng xử lý |
|---|---|---|
| B1 | Trang EduNext hiện trạng thái "Expired" dù cổng nộp chưa mở (08/12) | Hỏi thư ký bộ môn trước 08/12; nếu 10:00 08/12 vẫn khoá thì báo ngay |
| B2 | Chưa chốt 1 trong 3 phương án deck | Nhóm chọn, xin GVHD duyệt trước buổi 30/11 |
| B3 | File góp ý Review 1–2 đã nộp nằm trên Drive tổng của lớp, chưa có trong gói Progress Reports | Tải về gói hoặc bỏ mục này (đã có `Review2_Feedback.md` và các deck Review) |
| B4 | `Review2_Feedback.md` còn vài mục đánh dấu "cần nghe lại" | Nghe lại bản ghi âm và cập nhật trước buổi 30/11 |

## Kế hoạch tuần sau

1. Chốt deck, tập trình bày, review hội đồng cấp Khoa 30/11.
2. Sửa luận văn và deck theo góp ý buổi 30/11.
3. Rà soát gói `submission_edunext/`, đổi tên file theo quy định EduNext, nộp trong 08–12/12 (nhắm xong trước 10/12).
4. Chuẩn bị bảo vệ lần 1 (19–21/12).

## Dấu vết commit

| Ngày | Commit | Nội dung |
|---|---|---|
| 10/11 | `36860d0` | Snapshot luận văn Overleaf VI v1, deck draft 2 |
| 11/11 | `3de8cb5`, `953d810` | Luận văn VI v2–v3 (Chương 1–6), deck draft 3 |
| 11/11 | `327351d`, `5de54dc` | Build luận văn local (MiKTeX), trang ngang cho 4 hình rộng, bookmark PDF |
| 13/11 | `81d23c2`, `de78d9e` | Demo local Windows (GPU 4-bit + OCR CPU), chế độ offline |
| 15/11 | `a77d600` | Luận văn bản tiếng Anh `source_en` |
| 15/11 | `b88549c` → `62469cc` | `demo.cmd`, OCR chạy nền + cache, đo độ trễ, giao diện sáng |
| 15/11 | `b0f1dfa` → `0cc0471` | Phân tích OCR lật đáp án ViVQA; ca §5.8 trong luận văn |
| 16/11 | `90c736b` → `7932686` | Ảnh thật trên laptop, chọn bộ demo, khởi động song song, `.\demo safe` |
| 19/11 | `98001f6` → `7b45ff6` | Ẩn bảng ví dụ, `demo_images/` local, bộ demo v3 (13 ảnh) |
| 19/11 | `a67c591` | `docs/04_slides` còn link Canva + 6 file deck |
| 23/11 | `0851bb2` | Bỏ qua thư mục gói nộp local `submission_edunext/` |
| 23/11 | (commit đóng tuần) | Báo cáo W15, deck đã sửa, 2 chỗ sửa luận văn, `.gradio/` vào gitignore |

## Link experiment log

- `experiments/W15_token_budget/summary.md`
- `experiments/W15_metric_tokenization/summary.md`
- `experiments/W15_REAL_local_sh/`
- `experiments/W06_ablation_config_dev300_seed42/`, `experiments/W07_config_freeze_L4_seed42/` (bằng chứng G5)
- `experiments/W07_R1_summary_L4_seed42/r1_summary.json` (Qwen2-VL-7B, 2 cấu hình: R1 theo bài báo, R1b theo B1)
- `experiments/W10_lora_ablation_test2000_L4_seed42/lora_ablation_summary.json` (kích thước adapter, MiB)
