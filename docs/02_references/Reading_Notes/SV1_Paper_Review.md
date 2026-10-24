# SV1 Paper Review — VLM / Baseline

> **Đề tài:** ViVOA-VLM — Efficient Vietnamese Visual Question Answering using Vision-Language Models
> **Vai trò:** SV1 — VLM / Baseline (BLIP-2, LLaVA, Florence-2, Qwen2.5-VL)
> **Tuần:** W01 · **Ngày:** _YYYY-MM-DD_
> **Yêu cầu:** ≥ 3 paper, ưu tiên: 1 paper kiến trúc VLM nền tảng (BLIP-2 / LLaVA), 1–2 paper baseline ứng viên (Florence-2, Qwen2.5-VL). Mỗi paper trả lời **đủ 10 câu hỏi** + mục **"Paper này liên quan gì đến đề tài ViVOA-VLM?"**. Cấm chỉ tóm tắt abstract.

## 0. Danh sách paper đã chọn & lý do

| # | Paper | Venue / Năm | Nhóm | Vì sao chọn |
|---|-------|-------------|------|-------------|
| P1 | BLIP-2 (Li et al.) | ICML 2023 | Kiến trúc VLM (Q-Former) | _…_ |
| P2 | Florence-2 (Xiao et al.) | CVPR 2024 | Baseline ứng viên | _…_ |
| P3 | Qwen2.5-VL Technical Report (Bai et al.) | 2025 | Baseline ứng viên | _…_ |
| P4 | _(tùy chọn)_ LLaVA — Visual Instruction Tuning (Liu et al.) | NeurIPS 2023 | Kiến trúc VLM | _…_ |

---

## P1. _Tên paper_

| Trường | Giá trị |
|--------|---------|
| Title | |
| Author(s) | |
| Year / Venue | |
| Code / Weights | |
| Reviewer | SV1 |

### 1. What problem does the paper solve?
_…_

### 2. Why is the problem important?
_…_

### 3. What is the research gap?
_…_

### 4. What is the proposed method?
_…_ (vẽ/giải thích pipeline: Vision Encoder → Visual Representation → Vision-Language Interface → LLM → Answer)

### 5. What is the key technical idea?
_…_

### 6. What dataset is used?
_…_

### 7. What metrics are used?
_…_

### 8. What are the main results?
_…_ (số liệu cụ thể từ bảng chính)

### 9. What are the limitations?
_…_ (đặc biệt: English-centric, VRAM, licence, scene-text)

### 10. What can we learn/use for our project? **(BẮT BUỘC)**
_…_

> ### Paper này liên quan gì đến đề tài ViVOA-VLM?
> _…_ (Có phải baseline ứng viên? Reproduce được trên Colab Pro/Premium hay GPU trường không? Điểm nào của kiến trúc SV3 sẽ gắn LoRA/QLoRA vào?)

---

## P2. _Tên paper_

_(lặp lại đúng cấu trúc P1)_

---

## P3. _Tên paper_

_(lặp lại đúng cấu trúc P1)_

---

## Tổng hợp cho nhóm

### So sánh 3 baseline ứng viên (điền để nhóm chọn ĐÚNG 1 baseline ở W03–W04)

| Tiêu chí | BLIP-2 | Florence-2 | Qwen2.5-VL |
|----------|--------|------------|------------|
| Kích cỡ (tham số) | | | |
| Kiến trúc interface (Q-Former / seq2seq / merger) | | | |
| Hỗ trợ tiếng Việt (tokenizer, pretrain) | | | |
| Scene-text / OCR nội tại | | | |
| VRAM inference (FP16 / 4-bit) | | | |
| Có code + checkpoint HF | | | |
| Licence | | | |
| Đề xuất | | | |

### Hàng đóng góp vào `Literature_Matrix.md`
- [ ] Đã chép ≥ 3 hàng (SV1)

### Follow-up experiment idea (W04–W05)
_…_
