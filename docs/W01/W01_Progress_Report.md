# Báo cáo tiến độ — Tuần 1 (W01)

> Dự án: **ViVQA-VLM** — Efficient Vietnamese Visual Question Answering using Vision-Language Models
> Phạm vi: `docs/weekly/`. Ngày lập báo cáo: 2026-09-14.
> Trạng thái: **W01 đã hoàn thiện ~100%** và thư mục đã được dọn sạch (xem mục 5–6).

---

## 1. Cấu trúc W01 sau khi dọn dẹp

W01 giờ **chỉ còn deliverable Tuần 1** (10 file + báo cáo này):

| File | Vai trò |
|------|---------|
| `README.md` | Hướng dẫn tuần |
| `Research_Orientation.md` | Định hướng nghiên cứu (đã điền đủ, gồm 7 câu hỏi định hướng) |
| `Project_Plan_15_Weeks.md` | Kế hoạch 15 tuần, 4 milestone đã gắn ngày |
| `SV1_Paper_Review.md` | Review VLM/baseline (BLIP-2, Florence-2, Qwen2.5-VL) |
| `SV2_Paper_Review.md` | Review dataset/OCR (ViVQA, ViTextVQA, TextVQA) |
| `SV3_Paper_Review.md` | Review PEFT (LoRA, QLoRA, VL-Adapter, LLaVA-1.5) |
| `SV4_Paper_Review.md` | Review eval/hallucination (POPE, VQAv2, Survey) |
| `W01_Team_Report.md` | Báo cáo nhóm tuần 1 |
| `tracker.md` | Tracker mục tiêu/task/deliverable |
| `ViVQA-VLM_Week1_Progress.pptx` | Deck tiến độ (12 slide) |
| `W01_Progress_Report.md` | Báo cáo tiến độ này |

**Deliverable synthesis (thuộc W2)** đã được chuyển và **hợp nhất vào `docs/weekly/W02/`**: `Literature_Matrix.md`, `Dataset_Analysis.md`, `Research_Gap.md`, `Research_Questions.md`, `Hypotheses.md`, `Candidate_Methods.md` (bản đầy đủ từ W01 thay cho template rỗng của W02; thư mục trung gian `W02/from_W01/` đã xoá).

---

## 2. Rubric tiến độ W1

Trọng số chuẩn hoá về 100% (đã loại synthesis docs W2). Cột "Mức điền" ước lượng theo nội dung thật.

| Hạng mục | Trọng số | Mức điền (%) | Đóng góp |
|---|---|---|---|
| Khung docs + cấu trúc thư mục | 12% | 100% | 12.0% |
| Project Plan 15 tuần (.md) | 18% | 100% | 18.0% |
| Research Orientation | 12% | 100% | 12.0% |
| Paper review cá nhân ×4 (SV1–SV4) | 40% | 100% | 40.0% |
| Team Report W01 | 12% | 100% | 12.0% |
| Progress deck (pptx) | 6% | 100% | 6.0% |
| **TỔNG** | **100%** | — | **= 100%** |

> ### 👉 Tiến độ W1 = **100%**

**Căn cứ:**
- **Khung docs 100%** — README đầy đủ; cấu trúc W01…W15 đủ; `tracker.md` đã điền mục tiêu/task/deliverable.
- **Project Plan 100%** — phân công, lịch 15 tuần, **4 milestone đã gắn ngày dự kiến**, đường găng, ngân sách, ≥ 5 rủi ro (có owner), DoD đã tick (trừ pilot W04 là việc tương lai).
- **Research Orientation 100%** — mục 1–6 đầy đủ; **7 câu hỏi định hướng (mục 7) đã trả lời**.
- **Paper review ×4 = 100%** — cả SV1/SV2/SV3/SV4 đều ≥ 3 paper, đủ 10 câu + mục "liên quan gì đến đề tài" + bảng tổng hợp (mục 3).
- **Team Report 100%** — GVHD, ngày, 4 hàng per-SV có evidence, deliverables, kết quả, blocker, kế hoạch W02.
- **Deck 100%** — `ViVQA-VLM_Week1_Progress.pptx`, 12 slide.

---

## 3. Chi tiết paper review theo từng SV

| SV | Trạng thái | Số paper | Đủ 10 câu hỏi? | Có mục "liên quan đến đề tài"? | Ghi chú |
|----|-----------|----------|----------------|-------------------------------|---------|
| **SV1** | ✅ Hoàn tất | 3 (BLIP-2, Florence-2, Qwen2.5-VL) | ✅ | ✅ | Bảng so sánh 3 baseline; đề xuất **Qwen2.5-VL-3B** làm baseline chính |
| **SV2** | ✅ Hoàn tất | 3 (ViVQA, ViTextVQA, TextVQA/LoRRA) | ✅ | ✅ | Bảng thống kê dataset; đề xuất **PaddleOCR + đối chứng VietOCR** (CER đo W02) |
| **SV3** | ✅ Hoàn tất | 4 (LoRA, QLoRA, VL-Adapter, LLaVA-1.5) | ✅ | ✅ | Recipe QLoRA, 9 ablation, 8 giả thuyết |
| **SV4** | ✅ Hoàn tất | 3 (POPE, VQAv2, Survey TPAMI) | ✅ | ✅ | Chốt EM/VQA-Acc/ANLS/POPE-vi + chuẩn hoá câu trả lời tiếng Việt |

---

## 4. Việc chuyển sang W02+ (không tính vào tiến độ W1)

Những mục sau **thuộc W02 trở đi** (cần chạy thực nghiệm/tải dữ liệu — không bịa số ở W1):
- Số liệu thật dataset (ViVQA/ViTextVQA: #ảnh, #QA, phân bố loại câu hỏi) — **W02** (`Dataset_Analysis.md`).
- CER/WER PaddleOCR vs VietOCR trên 50 ảnh mẫu — **W02**.
- Zero-shot EM/VQA-Acc/ANLS của baseline; pilot QLoRA 200 mẫu (`experiments/W01_Pilot/README.md`) — **W04–W05**.
- Hợp nhất nội dung 6 file synthesis W01↔W02 (đã đặt bản đầy đủ tại `docs/weekly/W02/`; nhóm rà soát lại ở W02).

---

## 5. Xử lý lỗi tồn đọng (đã khắc phục)

- **Sai tên "ViVOA-VLM" → "ViVQA-VLM":** đã sửa toàn bộ ở W01 (`W01_Team_Report.md`, `Project_Plan_15_Weeks.md`, `SV1/SV2/SV4_Paper_Review.md`) và ở 5 file đã chuyển sang W02.
- **Milestone chưa gắn ngày:** đã thêm **Ngày dự kiến** cho cả 15 tuần; 4 milestone: **W03 ___ · W07 ___ · W13 ___ · W15 ___**; DoD đã tick.
- **`tracker.md` rỗng:** đã điền mục tiêu tuần, task theo SV, meeting notes, deliverables.
- **Ô nguồn dataset trống:** đã điền bảng thống kê nhanh (theo số công bố của paper) trong `SV2_Paper_Review.md`; số đo chi tiết để ở W02 (`Dataset_Analysis.md`).

---

## 6. Dọn dẹp file trùng lặp / thừa (đã thực hiện — được người dùng phê duyệt)

**Đã XOÁ vĩnh viễn (repo không dùng git):**
- `docs/weekly/W01/Project_Plan_15_Weeks.from_AIP491.md` — bản trùng, giữ bản `.md` đầy đủ hơn.
- `docs/weekly/W01/W01_Team_Report.from_AIP491.md` — bản trùng, giữ bản `.md` đầy đủ hơn.
- `docs/weekly/W01/W01_Research_Orientation.txt` — đề bài/hướng dẫn Tuần 1, trùng vai trò với `README.md` + `docs/Master_Instruction.md`.
- `docs/weekly/W02/{Literature_Matrix, Dataset_Analysis, Research_Gap, Research_Questions, Hypotheses, Candidate_Methods}.md` — 6 template rỗng của W02, thay bằng bản đầy đủ chuyển từ W01.

**Đã HỢP NHẤT:** 6 file synthesis (bản đầy đủ) → `docs/weekly/W02/`; xoá thư mục trung gian `docs/weekly/W02/from_W01/`.

> Không còn cặp file trùng lặp nào trong W01. Mọi thao tác chỉ trong `docs/weekly/`.
