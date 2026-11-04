# W14 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W14 — Hoàn thiện code: dữ liệu + prototype + demo (thesis, slide, góp ý Review 2 dời sang tuần sau theo quyết định của Hoàng, 01/11) |
| Thời gian thực | 01–02/11/2026 (tuần thực 8–9) |
| Mốc kế tiếp | Hội đồng cấp Khoa, tuần thực 13 (30/11–06/12) |
| Phần cứng | Colab Pro: NVIDIA T4 (notebook 25, 26), NVIDIA L4 (notebook 27, 28 × 2) · máy local (selftest, Cursor, ffmpeg) |
| CU Colab | 17,38 (đầu tuần, 31/10) → khoảng 7 (ước tính: ≈ 1,5 h T4 + ≈ 5,2 h L4). Đọc số thật ở Colab → Resources. Gói gia hạn 02/11: tiền đã trừ ngày 1/11, CU chưa thấy cộng lúc 11:00 ngày 2/11 → kiểm lại hết ngày 3/11 |
| Commit cuối tuần | commit đóng tuần sau `86cd466` (xem mục "Dấu vết commit") |

## Mục tiêu

Hoàn thiện **toàn bộ phần code** trước khi viết thesis và làm slide:
1. Xác minh và xử lý giả thuyết **lỗi EXIF trong OCR cache** (W13, blocker B1), vì nó có thể ảnh hưởng số của đóng góp chính.
2. Đưa prototype về **đúng hệ thống đã đánh giá** ở cả bước OCR. R-OCR của W13 không đạt.
3. Có **video demo dự phòng**.
4. Có **gói tái lập** đúng master §10.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | **R-EXIF** (luật ghi trước): quét cờ xoay EXIF + OCR lại mẫu cố định theo 3 kiểu (`src/data/exif_audit.py`, notebook 25, T4) | **L1 XÁC NHẬN** (164/165 ảnh cờ xoay lệch cache, 0/200 đối chứng; kiểu `asis` trùng cache 100% → lỗi tất định) · **L2:** test 1,256% > 1% → phải sửa · **L3:** train 1,28% ≤ 5% → không train lại |
| 2 | Sửa `src/data/ocr.py` (`oriented_image`), dựng cache mới (notebook 26, T4), suy luận lại B3-full full test (notebook 27, L4) | V1–V5 **hợp lệ** (167 ảnh OCR lại, 15.185 ảnh giữ nguyên, 9.902 / 9.902 dự đoán trùng). B3-full **53,35 → 53,63 EM** (độ nhạy); 126 câu bị lỗi 26,98 → 49,21 |
| 3 | Prototype v2: ngưỡng thu nhỏ 2.048 → 6.000 px, R-OCR v2 / R-LIVE v2 (luật ghi trước), notebook 28 (L4) | **R-OCR v2 ĐẠT · R-LIVE v2 ĐẠT** (100% / 100%); p50 / p95 1,00 / 3,58 s; VRAM 4,78 GB; ảnh thật EM 14 / 20 |
| 4 | Demo: ví dụ chọn bằng luật cố định (`src/prototype/demo_examples.py`), kịch bản `Demo_Script.md`, video lần 1 → lộ lỗi ảnh bằng chứng hiện chậm → sửa `prototype-app-v2.1` → video lần 2 | Video 10 phút 28 giây trên Drive `demo/` (1080p 664 MB + 720p 17 MB) |
| 5 | Gói tái lập: `docs/REPRODUCE.md`, `requirements-colab.txt`, `configs/official_models.json` (`src/eval/repro_manifest.py`), `src/prototype/infer.py` | 6 model chính thức; **4 / 4 adapter trên Drive trùng sha256**; phiên bản thư viện khớp log; selftest chạy được trên cả Linux và Windows |
| 6 | Đóng tuần: README W14 + gốc, Team Report, cập nhật `docs/W13/Prototype_Guide.md`, dọn Downloads + Drive | ✅ |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| Dataset / OCR | R-EXIF; sửa `ocr.py`; cache `_exiffix`; dấu vết dịch máy "tía" trong ViVQA |
| VLM / Baseline | Kiểm độ tất định: suy luận lại cho dự đoán trùng 100% trên cùng loại GPU |
| PEFT / Training | Quyết định không train lại (L3); adapter chính thức + sha256 |
| Evaluation / Deployment | Dòng độ nhạy B3-full; prototype v2 / v2.1; R-OCR v2 / R-LIVE v2; demo, video; gói tái lập |

### Definition of Done

- [x] Mọi luật (R-EXIF L1–L3, bước 2 V1–V5, R-OCR v2 / R-LIVE v2, chọn ví dụ demo) commit **trước** khi chạy notebook tương ứng (`a2e20b1`, `cded8d9`, `5e48695`)
- [x] Kết quả W13 (R-OCR KHÔNG ĐẠT) giữ nguyên trong báo cáo; luật v2 được ghi rõ là luật mới, kèm lý do đổi
- [x] Số chính thức không bị thay sau khi đã thấy kết quả; bản sửa lỗi báo thành dòng độ nhạy, đúng luật L3
- [x] Cache OCR cũ giữ nguyên; cache mới là file riêng, có biên bản sha256
- [x] Claude tính lại độc lập từ dữ liệu thô (EM 53,35 → 53,63; 5 / 33 câu lật; 9.902 câu trùng) và kiểm từng commit từ bản clone mới
- [x] Video demo dự phòng có trên Drive; không lên GitHub
- [x] Gói tái lập đủ 6 thành phần của master §10; adapter kiểm sha256 trên Drive
- [x] Dọn Downloads + Drive dự án (prompt đóng tuần)

## Kết quả / số liệu

### 1. R-EXIF (nguồn `experiments/W14_exif_audit_vitextvqa/`)

| Tập | Câu có ảnh cờ xoay | Ảnh |
|---|---|---|
| ViTextVQA train | 450 / 35.159 (1,28%) | 124 / 11.733 |
| ViTextVQA dev | 74 / 5.155 (1,44%) | 19 / 1.676 |
| ViTextVQA test full | 126 / 10.028 (1,256%) | 41 / 3.353 |
| ViVQA train / test | 0 | 0 |

- **Lệch cache (kiểu `fixed`):** 164 / 165 ảnh cờ xoay (99,4% [96,7; 99,9]); 0 / 200 ảnh đối chứng (0% [0; 1,9]).
- **Đáp án nằm trong OCR (126 câu):** cache 0,0% → ảnh đã xoay 41,3%.
- **Cơ chế:** OpenCV (Paddle dò hộp chữ) xoay ảnh theo EXIF, còn PIL (VietOCR cắt hộp) không xoay.
- **Phát hiện phụ:** W13 ghi "ảnh dataset đều nhỏ hơn 2.048 px" là sai. Thực tế 1.418 / 16.762 ảnh ViTextVQA lớn hơn (lớn nhất 5.770 px).

### 2. B3-full với OCR đã sửa (nguồn `experiments/W14_exif_fix/`)

| | B3-full W10 (chính thức) | + OCR sửa EXIF (độ nhạy) | Δ [CI95] |
|---|---|---|---|
| Full test EM | 53,35 | 53,63 | +0,28 [+0,17; +0,41], McNemar 33 / 5, p = 4,3 × 10⁻⁶ |
| Full test ANLS | 73,89 | 74,07 | +0,18 [+0,09; +0,28] |
| 126 câu ảnh cờ xoay, EM | 26,98 | 49,21 | +22,22 [+13,49; +30,95] |
| B2 → B3-full, EM | +3,09 (p = 6,7 × 10⁻²¹) | +3,37 (p = 4,5 × 10⁻²⁴) | |

**Chỉ mô tả:** đưa thêm ảnh đã xoay cho model (đường ảnh của prototype) trên 126 câu: ANLS +9,67 [+3,42; +16,17], EM +4,76 (p = 0,26).

### 3. Prototype v2 (nguồn `experiments/W14_prototype_live_v2_L4_seed42/`)

| Tập | R-OCR v2 | So cache T4 (mô tả) | R-LIVE v2 | EM giao diện / W11 |
|---|---|---|---|---|
| ViVQA | 100% | 100% | 100% (211 câu) | 75,83 / 75,83 |
| ViTextVQA | 100% | 95,5% (9 ảnh lệch số học GPU) | 100% (259 câu) | 50,19 / 50,19 |

- **Tốc độ (L4):** tổng p50 / p95 = 1,00 / 3,58 s; OCR 0,32 / 2,10 s; model 0,65 / 1,62 s.
- **Tài nguyên:** khởi động 61 s; VRAM 2,61 GB (tiến trình chính) / 4,78 GB (tổng).
- **v2.1:** ảnh bằng chứng OCR thu về 1.280 px JPEG khi hiển thị. Chỉ đổi phần hiển thị; model và OCR không đổi.

### 4. Gói tái lập (nguồn `configs/official_models.json`, `docs/W14/official_models.md`)

| id | Vai trò | Kết quả (EM %) | Adapter |
|---|---|---|---|
| r0 | tái lập baseline | TextVQA val VQA-Acc 79,60 (công bố 79,3) | — |
| b1 | baseline zero-shot | ViVQA 22,09 · ViTextVQA 29,12 | — |
| b2_vitextvqa | bước thích nghi | 50,26 | `2e7b17d18b32…` |
| c3_vivqa | chính thức ViVQA | 76,64 | `3ceff2f7f961…` (epoch 2) |
| b3full | chính thức ViTextVQA | 53,35 (độ nhạy 53,63) | `c1eb9c2b4baf…` |
| sh | prototype | ViVQA 73,51 · ViTextVQA test2000 49,50 | `c9791e961aaa…` |

Môi trường (W14 freeze, L4): torch 2.11.0+cu130 · transformers 5.17.0 · peft 0.21.2 · bitsandbytes 0.50.2 · accelerate 1.15.0 · qwen-vl-utils 0.0.14 · gradio 6.31.0 · paddlepaddle 3.3.1 · paddleocr 3.7.0 · Pillow 11.3.0 · Python 3.13.16.

### 5. Đối chiếu giả thuyết

| Mục | Trạng thái sau W14 | Lý do |
|---|---|---|
| H1 · QLoRA ≥ +5 EM | Ủng hộ (không đổi) | W14 không chạy lại B1 / B2 |
| H2 · OCR + QLoRA ≥ +3 ANLS | **Một phần** (không đổi) | ΔANLS so với B2: +1,87 (chính thức) / +2,05 (độ nhạy), vẫn < +3 |
| H3 · prompt định dạng tiếng Việt | Không ủng hộ (không đổi) | — |
| H4 · QLoRA không tăng hallucination | Một phần (không đổi) | POPE-vi dùng ảnh ViVQA, không có ảnh cờ xoay → không bị lỗi EXIF |
| Prototype đúng hệ thống đã đánh giá | **Đạt hoàn toàn** | R-PAR (W13) + R-OCR v2 + R-LIVE v2 (W14) |

## Kiến thức rút ra trong tuần

1. **Siêu dữ liệu ảnh (EXIF) có thể làm hỏng pipeline một cách âm thầm.** Hai thư viện đọc cùng một file mà nhìn thấy hai ảnh khác chiều: OpenCV xoay theo cờ, PIL thì không. Bài học: mọi bước trong pipeline phải đọc ảnh theo **cùng một cách**. Khi nghi ngờ, kiểm bằng ảnh giả có cờ xoay.
2. **Nhóm đối chứng tách được nguyên nhân.** Nếu chỉ OCR lại ảnh có cờ xoay thì không phân biệt được lỗi EXIF với sai khác giữa hai lần chạy. Thêm 200 ảnh đối chứng (0% lệch) và kiểu `asis` (trùng cache 100%) thì kết luận "lỗi do EXIF, tất định" mới chắc.
3. **Số chính thức và dòng độ nhạy là hai thứ khác nhau.** Luật L3 ghi trước nói rõ khi nào được thay số chính thức. Sửa lỗi xong vẫn giữ số cũ, báo số mới bên cạnh. Hội đồng thấy được cả hai, và không ai có thể nói nhóm "chọn số đẹp".
4. **Kiểm độ tất định (V5) là phép kiểm rẻ mà mạnh.** 9.902 câu không bị ảnh hưởng ra dự đoán trùng 100%. Điều này chứng minh mọi khác biệt chỉ đến từ OCR đã sửa, và pipeline tái lập được.
5. **Kiểm đúng đối tượng.** R-OCR W13 so OCR trên L4 với cache dựng trên T4, nên trộn hai nguồn lệch. R-OCR v2 so giao diện với đọc thẳng file trên cùng máy, tức kiểm đúng phần giao diện chịu trách nhiệm. Phần lệch do GPU vẫn được báo cáo dưới dạng mô tả.
6. **Ảnh để xử lý và ảnh để hiển thị nên tách riêng.** Gửi ảnh 20 MB cho trình duyệt làm giao diện hiện bằng chứng cũ. Thu nhỏ bản hiển thị không ảnh hưởng kết quả.
7. **Video demo phát hiện được lỗi mà selftest bỏ sót:** ảnh bằng chứng hiện chậm, đường truyền của link share chập chờn. Lỗi trải nghiệm phải kiểm bằng cách dùng thật.
8. **Gói tái lập cần được tạo tự động.** Biên bản model sinh từ log, có sha256 adapter, thư viện ghim theo `pip freeze`. Gõ tay thì dễ lệch.
9. **Code phải chạy được trên nhiều hệ điều hành.** Đường dẫn ghép bằng `/` chạy đúng trên Colab nhưng hỏng selftest trên Windows. Phải dùng `os.path.join` cho từng thành phần.

## Câu hỏi hội đồng có thể hỏi (phần W14)

| Câu hỏi | Ý trả lời |
|---|---|
| Phát hiện lỗi dữ liệu muộn như vậy, kết quả còn tin được không? | Lỗi chỉ ở 1,26% câu test. Luật ghi trước quyết định cách xử lý. Bản sửa chỉ đổi +0,28 EM, và kết luận B3-full vượt B2 còn mạnh hơn (+3,37 EM) |
| Vì sao không thay số chính thức bằng số đã sửa? | Luật L3 (ghi trước khi biết kết quả): adapter không train lại, nên giữ số W10. Số sửa được báo minh bạch bên cạnh |
| Lỗi EXIF có ảnh hưởng các run OCR khác (B3, SH, ablation) không? | Có, ở cùng khoảng 1,3% câu và ở cả hai phía của mọi so sánh, nên các so sánh vẫn công bằng. Phần giới hạn có ghi rõ |
| Đổi luật R-OCR sau khi trượt có phải gian lận không? | Kết quả W13 giữ nguyên là KHÔNG ĐẠT. Luật v2 kiểm một câu hỏi khác (giao diện có biến đổi dữ liệu không), có lý do, commit trước khi chạy. Phần lệch do GPU vẫn báo cáo |
| Demo có đúng là hệ thống trong khoá luận? | R-PAR 100% (W13), R-OCR v2 và R-LIVE v2 100% / 100% (W14): cùng ảnh, cùng OCR, cùng câu trả lời với run eval |
| Ví dụ trong video có phải chọn toàn câu đúng? | Không. Ví dụ do luật cố định chọn từ mẫu seed 42, gồm một câu sai mức Thấp (`226.jpg`) |
| Người khác chạy lại thế nào? | `docs/REPRODUCE.md`: chạy notebook theo bảng, notebook tự kiểm sha256; adapter kiểm bằng `repro_manifest --check_adapters`; thư viện theo `requirements-colab.txt` |
| Demo cần phần cứng gì, nhanh tới đâu? | Một L4, VRAM khoảng 4,8 GB; p50 khoảng 1 s, p95 khoảng 3,6 s mỗi câu, kể cả OCR |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|---|---|---|
| B1 | CU Colab của kỳ gia hạn 02/11 chưa thấy cộng | Trung bình (cần GPU khi demo trực tiếp) | Kiểm hết ngày 3/11 (đơn hàng Google Play, trang CU). Ngày 4/11 báo Claude để soạn nội dung liên hệ |
| B2 | Video demo dài 10 phút 28 giây, không lời thoại | Thấp | Khi làm slide: cắt bản rút gọn khoảng 4 phút, có thể chèn phụ đề |
| B3 | Link `gradio.live` thỉnh thoảng lỗi đường truyền | Thấp | Bấm lại / mở link mới; có video dự phòng; hướng dẫn trong `Prototype_Guide.md` |
| B4 | Model nhận ảnh chưa xoay ở câu có cờ EXIF trong eval (mọi model) | Thấp | Ghi giới hạn; prototype đã xoay ảnh |

## Kế hoạch tuần sau (theo phạm vi Hoàng chọn)

1. **Góp ý Review 2:** từ bản ghi âm buổi 29/10, lập danh sách góp ý, điền checklist AI_Capstone_Review123 (sheet Review2), gửi mail, upload.
2. **Deck cộng dồn cho hội đồng cấp Khoa** (tuần thực 13):
   - sửa Part A / B theo góp ý Review 2;
   - thêm Part C: EXIF + độ nhạy, prototype v2.1 với ảnh chụp giao diện thật, video rút gọn.
3. **Thesis bản tiếng Việt** (R1–R7), có phần Prototype và giới hạn của W13–W14; bản tiếng Anh làm sau.
4. **Kiểm CU** (mục B1) ngày 4/11.

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `a2e20b1` | Bước 1 (ghi trước): luật R-EXIF, `exif_audit.py`, notebook 25 |
| `cded8d9` | Kết quả bước 1 + bước 2 (ghi trước): `oriented_image`, `--fix_cache` / `--compare`, `run_baseline --image_dir`, notebook 26 / 27 |
| `361431a` | Kết quả bước 2: cache đã sửa + B3-full suy luận lại (V1–V5 hợp lệ) |
| `5e48695` | Bước 3–4 (ghi trước): `prototype-app-v2`, `prototype-live-v2`, `demo_examples.py`, `Demo_Script.md`, notebook 28 |
| `e5225e0` | Kết quả bước 3: R-OCR v2 / R-LIVE v2 ĐẠT |
| `b146cff` | `prototype-app-v2.1`: ảnh bằng chứng thu nhỏ khi hiển thị (sau video lần 1) |
| `86cd466` | Bước 5: gói tái lập |
| *(commit đóng tuần)* | Team Report, README W14 + gốc, `Prototype_Guide.md` cập nhật |

## Link experiment log

- R-EXIF: `experiments/W14_exif_audit_vitextvqa/`
- Sửa cache + B3-full: `experiments/W14_exif_fix/`, `experiments/W14_B3full_eval_exiffix_*`, `experiments/W14_B3full_eval_exiffix_rotimg_*`
- Prototype v2: `experiments/W14_prototype_live_v2_L4_seed42/` (kèm `demo/`), `experiments/W14_LIVE_sh_ocrlive_v2_sample200img_bf16_L4_seed42/`, `experiments/W14_REAL_sh_realphotos_v2_bf16_L4/`
- Môi trường: `experiments/W14_env_freeze_L4.txt`
- Notebook: `notebooks/25_exif_ocr_audit_vitextvqa.ipynb`, `26_ocr_cache_exiffix_vitextvqa.ipynb`, `27_b3full_exiffix_testfull_qwen25vl.ipynb`, `28_prototype_v2_live_demo_sh_qwen25vl.ipynb`
- Tái lập: `docs/REPRODUCE.md`, `configs/official_models.json`, `docs/W14/official_models.md`
- Video (Drive, không lên git): `demo/ViVQA-VLM_demo_W14.mp4`, `demo/ViVQA-VLM_demo_W14_1080p.mp4`
