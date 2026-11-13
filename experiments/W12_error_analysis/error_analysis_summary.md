# Error analysis — mô hình chính thức (W12, chỉ mô tả)

qt-v1 rules sha256 `92c21a7a8e23` · OCR: VietOCR, max_chars 600, min_conf 0.0 · seed 42

## ViTextVQA full test (10028 câu)

### A1. Kiểu lỗi (% số câu sai)

| Run | Tỉ lệ sai | tone | truncated | extended | near_miss | other |
|---|---|---|---|---|---|---|
| B1 | 70.9% | 3.3 | 5.0 | 25.7 | 8.0 | 57.9 |
| B2 | 49.7% | 9.0 | 13.2 | 13.6 | 16.8 | 47.4 |
| B3full | 46.6% | 9.6 | 15.5 | 14.5 | 16.5 | 43.9 |

### A2. Nguồn lỗi của B3-full

Trạng thái OCR trên mọi câu: has_answer 33.1% · wrong_diacritics 11.5% · answer_missing 55.1% · no_text 0.3%

Trên 4678 câu B3-full sai: **ocr_upstream** 81.0% · **model_wrong_span** 10.5% · **model_outside_ocr** 8.5% (ocr_upstream gồm: wrong_diacritics 14.6%, answer_missing 84.8%, no_text 0.6%)

| Nguồn lỗi | tone | truncated | extended | near_miss | other |
|---|---|---|---|---|---|
| ocr_upstream | 393 | 537 | 360 | 682 | 1816 |
| model_wrong_span | 2 | 190 | 193 | 11 | 96 |
| model_outside_ocr | 55 | 0 | 123 | 77 | 143 |

| Loại (qt-v1) | n | EM B3-full | % câu có đáp án trong OCR | ocr_upstream | model_wrong_span | model_outside_ocr |
|---|---|---|---|---|---|---|
| contact | 641 | 64.1 | 30.1 | 90.9 | 3.9 | 5.2 |
| location | 969 | 47.0 | 30.1 | 84.4 | 10.3 | 5.3 |
| name | 1971 | 57.6 | 34.2 | 80.4 | 9.9 | 9.7 |
| number_time | 1041 | 58.8 | 47.4 | 72.5 | 15.4 | 12.1 |
| other | 2404 | 49.6 | 34.0 | 77.8 | 14.2 | 8.0 |
| person | 420 | 61.2 | 40.0 | 82.2 | 8.6 | 9.2 |
| product_service | 1144 | 44.1 | 25.6 | 81.7 | 10.2 | 8.1 |
| source | 661 | 58.1 | 30.7 | 83.8 | 5.1 | 11.2 |
| text_content | 777 | 51.1 | 24.1 | 87.6 | 4.2 | 8.2 |

### A3. B2 → B3-full (ghép cặp, số câu)

| Trạng thái OCR | n | fixed | broken | both_right | both_wrong |
|---|---|---|---|---|---|
| tất cả | 10028 | 705 | 395 | 4645 | 4283 |
| has_answer | 3321 | 326 | 116 | 2105 | 774 |
| wrong_diacritics | 1149 | 90 | 68 | 506 | 485 |
| answer_missing | 5526 | 289 | 211 | 2023 | 3003 |
| no_text | 32 | 0 | 0 | 11 | 21 |

Câu được sửa có đáp án trong OCR: 46.2% · câu bị làm hỏng mà câu trả lời B3-full nằm trong OCR (OCR làm phân tâm): 34.9%

## ViVQA test (3001 câu)

### V1. Kiểu lỗi

| Run / loại | n | Tỉ lệ sai | tone | truncated | extended | near_miss | other |
|---|---|---|---|---|---|---|---|
| B1 | 3001 | 77.9% | 0.0 | 3.6 | 21.3 | 6.4 | 68.6 |
| C1 | 3001 | 26.2% | 0.5 | 2.8 | 3.2 | 15.1 | 78.4 |
| C3 | 3001 | 23.4% | 0.0 | 2.3 | 3.9 | 16.0 | 77.9 |
| C3 · object | 1247 | 23.3% | 0.0 | 3.8 | 3.4 | 2.4 | 90.4 |
| C3 · number | 444 | 23.0% | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| C3 · color | 625 | 15.7% | 0.0 | 0.0 | 0.0 | 1.0 | 99.0 |
| C3 · location | 685 | 30.7% | 0.0 | 2.4 | 8.1 | 1.0 | 88.6 |

### V2. Cặp nhầm nhiều nhất của C3 (gold → dự đoán, số câu)

- **object**: tủ lạnh → tủ đá (6); xe cộ → xe tải (5); xe đạp → xe máy (4); tòa nhà → tòa tháp (4); máy tính → laptop (4); lá → cây (3); hải âu → con chim (3); laptop → máy vi tính (3)
- **number**: bốn → ba (11); ba → bốn (10); một → hai (10); ba → hai (9); năm → bốn (8); bốn → hai (7); hai → ba (6); bốn → năm (5)
- **color**: màu nâu → màu đen (5); màu đỏ → màu cam (5); màu đen → màu xanh dương (5); màu xám → màu trắng (4); màu đỏ → màu tía (4); màu nâu → màu vàng (4); màu vàng → màu nâu (4); màu trắng → màu nâu (4)
- **location**: phòng → phòng ngủ (6); nhà vệ sinh → phòng tắm (5); phòng → phòng tắm (5); vườn bách thú → chuồng (4); ô cửa → phòng (3); vòi hoa sen → phòng tắm (3); thùng chứa → cái nồi (3); nhà ở → phòng (3)

### V3. Ảnh đã thấy / chưa thấy lúc train (train + val)

| Tập con | n | % test | EM B1 | EM C3 | ΔEM [CI95] |
|---|---|---|---|---|---|
| all | 3001 | 100.0 | 22.09 | 76.64 | +54.55 [+52.68, +56.41] |
| image_seen | 1464 | 48.78 | 12.77 | 77.66 | +64.89 [+62.36, +67.35] |
| image_unseen | 1537 | 51.22 | 30.97 | 75.67 | +44.70 [+41.96, +47.36] |
| clean | 1118 | 37.25 | 26.39 | 71.74 | +45.35 [+42.31, +48.48] |
