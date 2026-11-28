# SV2 Paper Review — Dataset / OCR

> **Đề tài:** ViVOA-VLM — Efficient Vietnamese Visual Question Answering using Vision-Language Models
> **Vai trò:** SV2 — Dataset / OCR (ViVQA, ViTextVQA, scene-text, PaddleOCR / VietOCR, thách thức tiếng Việt)
> **Tuần:** W01 · **Ngày:** _YYYY-MM-DD_
> **Yêu cầu:** ≥ 3 paper, ưu tiên: paper ViVQA (Tran et al., PACLIC 2021), paper ViTextVQA (Nguyen et al., 2024/2025), 1 paper về OCR tiếng Việt hoặc scene-text VQA (TextVQA / ST-VQA / OCR-VQA). Mỗi paper trả lời **đủ 10 câu hỏi** + mục **"Paper này liên quan gì đến đề tài ViVOA-VLM?"**. Cấm chỉ tóm tắt abstract.

## 0. Danh sách paper đã chọn & lý do

| # | Paper | Venue / Năm | Nhóm | Vì sao chọn |
|---|-------|-------------|------|-------------|
| P1 | ViVQA (Tran et al.) | PACLIC 2021 | Benchmark general VQA tiếng Việt | _…_ |
| P2 | ViTextVQA (Nguyen et al.) | arXiv 2024 / ESWA 2025 | Benchmark scene-text VQA tiếng Việt | _…_ |
| P3 | _(OCR / scene-text)_ TextVQA (Singh et al., CVPR 2019) hoặc PaddleOCR / VietOCR | | OCR & ANLS | _…_ |

---

## P1. _Tên paper_

| Trường | Giá trị |
|--------|---------|
| Title | |
| Author(s) | |
| Year / Venue | |
| Data / Code | |
| Reviewer | SV2 |

### 1. What problem does the paper solve?
_…_

### 2. Why is the problem important?
_…_

### 3. What is the research gap?
_…_

### 4. What is the proposed method?
_…_ (với paper dataset: quy trình thu thập, dịch/annotate, kiểm định chất lượng)

### 5. What is the key technical idea?
_…_

### 6. What dataset is used?
_…_ (số ảnh, số QA, split train/val/test, phân bố loại câu hỏi, độ dài câu trả lời)

### 7. What metrics are used?
_…_ (EM, VQA-Acc, ANLS, F1, BLEU/CIDEr…)

### 8. What are the main results?
_…_ (baseline nào, số liệu nào, human performance?)

### 9. What are the limitations?
_…_ (quy mô, nhiễu dịch máy, licence, thiếu text-in-image, thiếu split cố định…)

### 10. What can we learn/use for our project? **(BẮT BUỘC)**
_…_

> ### Paper này liên quan gì đến đề tài ViVOA-VLM?
> _…_ (Dataset này là general hay scene-text? Split nào sẽ được cố định cho nhóm? OCR nào cần chạy? Thách thức tiếng Việt nào — dấu thanh, code-switching — xuất hiện?)

---

## P2. _Tên paper_

_(lặp lại đúng cấu trúc P1)_

---

## P3. _Tên paper_

_(lặp lại đúng cấu trúc P1)_

---

## Tổng hợp cho nhóm

### Bảng thống kê nhanh dataset (chuyển sang `Dataset_Analysis.md`)

| Dataset | #Ảnh | #QA | Split | Loại câu hỏi | Metric chính | Licence | Link |
|---------|------|-----|-------|--------------|--------------|---------|------|
| ViVQA | | | | | EM / VQA-Acc | | |
| ViTextVQA | | | | | ANLS | | |

### Lựa chọn OCR

| Công cụ | Hỗ trợ dấu tiếng Việt | Tốc độ | Độ chính xác (ước lượng trên 50 ảnh mẫu) | Ghi chú |
|---------|----------------------|--------|-------------------------------------------|---------|
| PaddleOCR | | | | |
| VietOCR | | | | |

### Hàng đóng góp vào `Literature_Matrix.md`
- [ ] Đã chép ≥ 3 hàng (SV2)
