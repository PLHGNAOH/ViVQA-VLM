# Dataset Analysis — ViVQA & ViTextVQA

> Owner: **SV2** (Dataset / OCR) · Hỗ trợ: SV4 (metric), SV3 (định dạng dữ liệu fine-tune) · Cập nhật: _YYYY-MM-DD_

## 1. Tổng quan dataset

| Dataset | Nguồn / Paper | Loại | #Ảnh | #QA (train / val / test) | Ngôn ngữ | Licence | Link tải |
|---------|---------------|------|------|--------------------------|----------|---------|----------|
| ViVQA | Tran et al., PACLIC 2021 | General VQA | | | vi | | |
| ViTextVQA | Nguyen et al., 2024/2025 | Scene-text VQA | | | vi | | |

## 2. Thống kê mô tả (điền sau khi tải)

| Chỉ số | ViVQA | ViTextVQA |
|--------|-------|-----------|
| Độ dài câu hỏi trung bình (token) | | |
| Độ dài câu trả lời trung bình (token) | | |
| % câu trả lời 1 từ / ≤ 3 từ | | |
| Số câu trả lời duy nhất | | |
| Kích cỡ ảnh phổ biến (px) | | |
| % ảnh có chữ (ước lượng bằng OCR) | | |

## 3. Phân loại câu hỏi (mẫu ≥ 200 câu / dataset — cơ sở cho error analysis W12)

| Loại | Định nghĩa | Ví dụ | % ViVQA | % ViTextVQA |
|------|------------|-------|---------|-------------|
| Yes/No | | | | |
| Counting | | | | |
| Object recognition | | | | |
| Text reading | | | | |
| Reasoning | | | | |
| Khác | | | | |

## 4. Thách thức đặc thù tiếng Việt (kèm ví dụ thực từ dataset)

| Thách thức | Ví dụ | Ảnh hưởng đến metric | Xử lý đề xuất |
|------------|-------|----------------------|---------------|
| Dấu thanh / Unicode (NFC vs NFD) | | EM sai dù đúng nghĩa | Chuẩn hóa NFC |
| Code-switching Anh–Việt | | | |
| Từ đồng nghĩa / cách viết số | | | |
| Lỗi dịch máy trong dataset (nếu có) | | | |
| Chữ trong ảnh nghiêng / mờ / có dấu | | ANLS | OCR đa engine |

## 5. OCR cho ViTextVQA

| Engine | Phiên bản | Hỗ trợ dấu | Tốc độ (ảnh/s, GPU) | CER/WER trên 50 ảnh mẫu | Quyết định |
|--------|-----------|------------|---------------------|-------------------------|-----------|
| PaddleOCR | | | | | |
| VietOCR | | | | | |

Định dạng lưu OCR: `data/ocr/<dataset>/<image_id>.json` gồm `text`, `bbox`, `confidence`.

## 6. Split cố định & seed

| Dataset | Train | Val | Test | Seed | Ghi chú |
|---------|-------|-----|------|------|---------|
| ViVQA | | | | 42 | Dùng split gốc nếu có |
| ViTextVQA | | | | 42 | |

## 7. Định dạng dữ liệu cho fine-tune (thống nhất với SV3)

```json
{
  "id": "vivqa_000001",
  "image": "data/processed/vivqa/000001.jpg",
  "question": "Có bao nhiêu người trong ảnh?",
  "answer": "hai",
  "answers": ["hai", "2"],
  "question_type": "counting",
  "ocr_text": "",
  "source": "ViVQA",
  "split": "train"
}
```

## 8. Rủi ro dữ liệu & mitigation

| Rủi ro | Impact | Mitigation | Owner |
|--------|--------|-----------|-------|
| Link tải hỏng / licence hạn chế | High | Liên hệ tác giả; dùng subset công bố | SV2 |
| Câu trả lời nhiễu | Medium | Bộ lọc + kiểm tra thủ công mẫu | SV2 |
| OCR kém trên ảnh tiếng Việt | Medium | Đổi engine, ensemble | SV2 |

## 9. Checklist
- [ ] Tải và kiểm tra checksum 2 dataset
- [ ] Thống kê mục 2–3
- [ ] Chọn OCR engine, chạy toàn bộ ViTextVQA
- [ ] Cố định split + seed, commit `data/splits/`
