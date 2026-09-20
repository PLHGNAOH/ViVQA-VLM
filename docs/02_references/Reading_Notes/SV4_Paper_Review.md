# SV4 Paper Review — Evaluation / Deployment

> **Đề tài:** ViVOA-VLM — Efficient Vietnamese Visual Question Answering using Vision-Language Models
> **Vai trò:** SV4 — Evaluation / Deployment (EM, VQA-Acc, ANLS, hallucination, efficient inference, prototype web)
> **Tuần:** W01 · **Ngày:** _YYYY-MM-DD_
> **Yêu cầu:** ≥ 3 paper, ưu tiên: 1 paper hallucination (POPE — Li et al., EMNLP 2023), 1 paper về metric VQA/ANLS (VQAv2 — Goyal et al., 2017; ST-VQA — Biten et al., 2019), 1 paper về efficient inference / evaluation VLM (survey Zhang et al., TPAMI 2024 hoặc MME/MMBench). Mỗi paper trả lời **đủ 10 câu hỏi** + mục **"Paper này liên quan gì đến đề tài ViVOA-VLM?"**. Cấm chỉ tóm tắt abstract.

## 0. Danh sách paper đã chọn & lý do

| # | Paper | Venue / Năm | Nhóm | Vì sao chọn |
|---|-------|-------------|------|-------------|
| P1 | POPE — Evaluating Object Hallucination in LVLMs (Li et al.) | EMNLP 2023 | Hallucination | _…_ |
| P2 | _(metric)_ Making the V in VQA Matter (Goyal et al.) / Scene Text VQA — ANLS (Biten et al.) | CVPR 2017 / ICCV 2019 | Metric | _…_ |
| P3 | _(survey / benchmark)_ VLMs for Vision Tasks: A Survey (Zhang et al.) | TPAMI 2024 | Evaluation & deployment | _…_ |

---

## P1. _Tên paper_

| Trường | Giá trị |
|--------|---------|
| Title | |
| Author(s) | |
| Year / Venue | |
| Code | |
| Reviewer | SV4 |

### 1. What problem does the paper solve?
_…_

### 2. Why is the problem important?
_…_

### 3. What is the research gap?
_…_

### 4. What is the proposed method?
_…_

### 5. What is the key technical idea?
_…_

### 6. What dataset is used?
_…_

### 7. What metrics are used?
_…_ (định nghĩa chính xác công thức EM / VQA-Acc (min(#người đồng ý/3, 1)) / ANLS (ngưỡng 0.5) / POPE F1)

### 8. What are the main results?
_…_

### 9. What are the limitations?
_…_

### 10. What can we learn/use for our project? **(BẮT BUỘC)**
_…_

> ### Paper này liên quan gì đến đề tài ViVOA-VLM?
> _…_ (Metric/rubric nào sẽ được cài trong `src/eval/`? Hallucination probing tiếng Việt xây thế nào? Ảnh hưởng đến prototype: confidence, OCR evidence?)

---

## P2. _Tên paper_

_(lặp lại đúng cấu trúc P1)_

---

## P3. _Tên paper_

_(lặp lại đúng cấu trúc P1)_

---

## Tổng hợp cho nhóm

### Giao thức đánh giá đề xuất (chuyển sang `Candidate_Methods.md` / `Project_Plan_15_Weeks.md`)

| Metric | Công thức / chuẩn hóa | Dataset áp dụng | Thư viện / script |
|--------|-----------------------|-----------------|-------------------|
| Exact Match | | ViVQA, ViTextVQA | `src/eval/` |
| VQA Accuracy | | ViVQA | |
| ANLS | | ViTextVQA | |
| Hallucination (POPE-vi) | | Subset thủ công | |
| BLEU / CIDEr (phụ) | | Câu trả lời mở | |

### Chuẩn hóa câu trả lời tiếng Việt trước khi chấm
- [ ] lowercase, bỏ dấu câu cuối, chuẩn hóa Unicode NFC (dấu thanh), số ↔ chữ số
- [ ] _…_

### Hàng đóng góp vào `Literature_Matrix.md`
- [ ] Đã chép ≥ 3 hàng (SV4)
