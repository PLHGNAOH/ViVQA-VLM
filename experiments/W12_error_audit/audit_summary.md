# Audit tay lỗi mô hình chính thức (W12, bước 4b)

| | B3-full (ViTextVQA) | C3 (ViVQA) | B2 W07 ViTextVQA | B2 W07 ViVQA |
|---|---|---|---|---|
| n lỗi duyệt | 100 | 50 | 100 | 100 |
| % đúng ý [CI95] | 55.0 [45.2, 64.4] | 64.0 [50.1, 75.9] | 46.0 | 62.0 |
| % hallucination [CI95] | 5.0 [2.2, 11.2] | 16.0 [8.3, 28.5] | 9.0 | 5.0 |
| EM → EM theo người đọc [CI95] | 53.35 → 79.01 [74.44, 83.39] | 76.64 → 91.59 [88.34, 94.37] | | |
| Hallucination / mọi câu trả lời | 2.33% | 3.74% | | |

## Kiểu lỗi — ViTextVQA · B3-full

- Định dạng / độ dài (ý đúng): 40.0% [30.9, 49.8]
- Đọc chữ sai (OCR): 29.0% [21.0, 38.5]
- Sai vùng / không có trong ảnh (hallucination): 15.0% [9.3, 23.3]
- Sai dấu thanh: 7.0% [3.4, 13.7]
- Gold sai / nhiễu: 5.0% [2.2, 11.2]
- Đồng nghĩa / khác từ (ý đúng): 4.0% [1.6, 9.8]

## Kiểu lỗi — ViVQA · C3

- Đồng nghĩa / khác từ (ý đúng): 36.0% [24.1, 49.9]
- Gold sai / nhiễu: 28.0% [17.5, 41.7]
- Sai vùng / không có trong ảnh (hallucination): 28.0% [17.5, 41.7]
- Đếm sai: 6.0% [2.1, 16.2]
- Khác: 2.0% [0.4, 10.5]

## Nhãn tay × nguồn lỗi tự động (B3-full)

- **ocr_upstream**: Đọc chữ sai (OCR) 27; Định dạng / độ dài (ý đúng) 26; Sai vùng / không có trong ảnh (hallucination) 11; Sai dấu thanh 7; Gold sai / nhiễu 5; Đồng nghĩa / khác từ (ý đúng) 2
- **model_outside_ocr**: Định dạng / độ dài (ý đúng) 3; Đọc chữ sai (OCR) 2; Sai vùng / không có trong ảnh (hallucination) 2; Đồng nghĩa / khác từ (ý đúng) 1
- **model_wrong_span**: Định dạng / độ dài (ý đúng) 11; Sai vùng / không có trong ảnh (hallucination) 2; Đồng nghĩa / khác từ (ý đúng) 1
