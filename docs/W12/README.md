# W12 · Hallucination + Error analysis

> Tuần ảo W12 (bắt đầu 23/10/2026, tuần thực 7). Mốc master: "W12 · Analysis — ablation + error + hallucination".
> Review 2 (29/10) làm ở phiên chat riêng, **không** thuộc phạm vi README này.
> Colab đầu tuần (23/10 10:15, đọc ở Resources): **34,96 CU** (W11 ước tính ≈ 37 → lệch −2).

## Việc trong tuần

| # | Việc | Ước tính | Trạng thái |
|---|---|---|---|
| 1 | Khởi động: README W12 + luật H4 ghi trước + từ điển `configs/pope_vi_objects.json` | 0,5 h | ✅ |
| 2 | Code `src/eval/pope_vi.py` (sinh câu hỏi, chấm, phán quyết) + selftest + notebook 21; commit luật TRƯỚC khi chạy | 1,5 h | ⏳ |
| 3 | Colab L4: chạy POPE-vi 5 cấu hình | ≈ 4,5 h GPU (≈ 7 CU) | ☐ |
| 4 | Error analysis C3 (ViVQA, 4 loại có sẵn) + B3-full (ViTextVQA, `qt-v1`), luật phân loại lỗi chốt trước, không cần GPU | 2,5 h | ☐ |
| 5 | Kết luận H4 theo luật + ghi README | 1 h | ☐ |
| 6 | Ví dụ định tính 6–8 failure case | 1,5 h | ☐ |
| 7 | R6 §5–§7 bản tiếng Việt | 2 h | ☐ |
| 8 | Sửa đánh số cũ trong `docs/W02/Hypotheses.md` (H8 cũ = H4 mới) | 0,3 h | ☐ |
| 9 | Đóng tuần: Team Report, dọn Downloads + Drive, commit + push | 1 h | ☐ |

## Quyết định

| Ngày | Quyết định | Lý do |
|---|---|---|
| 23/10/2026 | **H4 đo theo kiểu POPE (Li et al., EMNLP 2023) bằng tiếng Việt: POPE-vi** trên ảnh test ViVQA (ảnh COCO 2014), nhãn vật thể lấy từ annotation COCO `instances_train2014/val2014` | ViVQA là ảnh COCO nên có nhãn vật thể thật, dựng được đúng 3 cách chọn câu "không" của POPE. ViTextVQA không có nhãn vật thể |
| 23/10/2026 | **5 cấu hình**: B1-plain, B1-ocr, C3, B3-full, SH | C3 và B3-full là hai adapter chính thức, SH là adapter của prototype. B1-ocr để so B3-full/SH với zero-shot **cùng prompt**, tách hiệu ứng QLoRA khỏi hiệu ứng dòng OCR |
| 23/10/2026 | **Quy mô**: 500 ảnh × (3 câu "có" dùng chung + 3 câu "không" cho mỗi cách random/popular/adversarial) = **6.000 câu / cấu hình** | Đúng khuôn POPE (500 ảnh, 50% có / 50% không mỗi cách chọn). Câu "có" dùng chung cho 3 cách nên chênh giữa 3 cách chỉ do câu "không"; tiết kiệm 1/3 GPU so với 9.000 câu. Cặp (ảnh, vật) trùng giữa các cách chọn chỉ hỏi **1 lần** (greedy ⇒ cùng prompt cùng đáp án) nên số câu chạy thật < 6.000; mỗi cách chọn vẫn chấm đủ 3.000 lượt (bổ sung 23/10, trước khi chạy) |
| 23/10/2026 | Ảnh chỉ lấy từ ViVQA test **không** xuất hiện trong ViVQA train | ViVQA train và test có ảnh dùng chung; C3/SH đã thấy ảnh train lúc huấn luyện |
| 23/10/2026 | Không train thêm run nào từ W12 trở đi | Ngân sách 34,96 CU chỉ đủ cho đánh giá + prototype + demo (xem mục Ngân sách) |

## H4 — luật ghi trước (chốt trước khi chạy; commit trước notebook 21)

**Giả thuyết H4** (H8 cũ ở W02): QLoRA không làm tăng hallucination vật thể so với zero-shot. Điều kiện bác bỏ kế thừa W02: F1 POPE-vi giảm > 2 điểm.

**Câu hỏi.**
- Ảnh: ảnh ViVQA test có annotation COCO, ≥ 3 lớp vật thể khác nhau, không nằm trong ViVQA train. Chọn 500 ảnh, seed 42.
- Câu "có": 3 lớp có trong ảnh, chọn ngẫu nhiên (seed 42).
- Câu "không" (3 lớp không có trong ảnh, loại các lớp cùng nhóm dễ nhầm trong `confusable_groups`):
  - **random**: chọn ngẫu nhiên.
  - **popular**: các lớp xuất hiện nhiều nhất trong toàn bộ ảnh của pool, lấy theo thứ tự.
  - **adversarial**: các lớp đồng xuất hiện nhiều nhất với các lớp có trong ảnh, lấy theo thứ tự.
- Mẫu câu: `Trong ảnh có {tên tiếng Việt} không? Trả lời "có" hoặc "không".` Tên lấy từ `configs/pope_vi_objects.json` (`pope-vi-objects-v1`).
- File câu hỏi sinh một lần, ghi sha256 vào README trước khi chạy model. Thủ tục: notebook 21 Phần A (CPU) sinh file → ⏸ Điểm dừng A → chép file vào `experiments/W12_pope_vi_questions/` + ghi sha256 + commit → Phần B (L4) kiểm sha256 trùng bản commit rồi mới chạy model.
- sha256 file câu hỏi `pope_vi_questions_v1.json`: _(điền ở Điểm dừng A)_

**Chạy.** Qwen2.5-VL-3B-Instruct 4-bit NF4, bf16, L4, greedy, `max_new_tokens` 32 (giữ đúng config chuẩn `configs/qwen_lora.yaml`, không thêm biến; sửa 23/10 trước khi chạy), seed 42. Chạy bằng `src.eval.run_baseline --dataset vivqa --data_file <file câu hỏi>` (theo lô, resume, chữ ký run). B1-plain và C3: prompt thường. B1-ocr, B3-full, SH: prompt `ocr` (VietOCR · v1 · 600 · 0, cache W11 `experiments/W11_ocr_cache_vivqa`). Adapter kiểm bằng `adapter_fingerprint` (khớp W10/W11).

**Đọc câu trả lời** (NFC, chữ thường, bỏ dấu câu; sửa 23/10 trước khi chạy): từ **đầu tiên** là "có"/"yes" → **có**; là "không"/"no" → **không**; nếu từ đầu không phải hai loại đó mà câu chứa "không có"/"không thấy" → **không**; còn lại → **không đọc được**. Câu không đọc được tính là **sai** theo cách nghiêm: ở câu "có" là FN, ở câu "không" là FP (không thể giúp adapter trông tốt hơn).

**Metric** (dương = "có"), cho từng cách chọn: Accuracy, Precision, Recall, **F1 (chính)**, yes-ratio, tỉ lệ không đọc được.

**So sánh** (ghép cặp cùng câu hỏi):
- C3 so với B1-plain.
- B3-full và SH so với B1-ocr.
- ΔF1 = F1(adapter) − F1(mốc). CI95 bằng bootstrap theo ảnh (mỗi ảnh lấy cả 6 câu của nó), 10.000 lần, seed 42.

**Phán quyết cho từng adapter:**
- **KHÔNG TĂNG HALLUCINATION**: cận dưới CI95 của ΔF1 > −2,0 ở **cả 3** cách chọn.
- **TĂNG HALLUCINATION**: cận trên CI95 < −2,0 ở **ít nhất 1** cách chọn.
- Còn lại: **CHƯA KẾT LUẬN**.

**H4 tổng:** **ỦNG HỘ** nếu C3 và B3-full đều KHÔNG TĂNG; **BÁC BỎ** nếu một trong hai TĂNG; còn lại **MỘT PHẦN**. SH chấm cùng luật, báo riêng như kiểm tra an toàn cho prototype (không tính vào H4).

**Kiểm hợp lệ** (kiểm trước khi đọc kết quả):
1. Mỗi cấu hình đủ 6.000 dự đoán, đúng sha256 file câu hỏi.
2. Fingerprint adapter khớp.
3. Tỉ lệ không đọc được ≤ 5% ở mỗi cấu hình. Nếu vượt, cấu hình đó không hợp lệ cho H4: báo như một phát hiện về lệch định dạng (format shift) và phán quyết của adapter đó = CHƯA KẾT LUẬN.

**Chỉ mô tả, không có luật:** B1-ocr so với B1-plain (dòng OCR có gây hallucination ở zero-shot không); yes-ratio theo adapter.

**Giới hạn đã biết:** annotation COCO không đầy đủ (vật có thật nhưng không được gán nhãn sẽ thành câu "không" sai); Qwen2.5-VL có thể đã thấy ảnh COCO khi pretrain. Cả 5 cấu hình dùng chung một backbone nên hai giới hạn này ảnh hưởng như nhau.

## Ngân sách

| Việc | GPU | CU (L4 ≈ 1,54 CU/h, T4 ≈ 1,2 CU/h) |
|---|---|---|
| POPE-vi 5 cấu hình × ≈ 5.000–5.900 câu sau khi bỏ trùng (≈ 0,57 s/câu như C3/SH trên ViVQA test) | ≈ 4,5 h L4 | ≈ 7 |
| Error analysis, ví dụ định tính | — | 0 |
| **Còn lại dự kiến cuối W12** | | **≈ 28** |

## File mới (bước 2)

| File | Nội dung | Kiểm |
|---|---|---|
| `src/eval/pope_vi.py` | `--build` sinh câu hỏi POPE-vi từ annotation COCO 2014 (ảnh ViVQA test chưa thấy lúc train, ≥ 3 lớp, 500 ảnh, seed 42, loại lớp dễ nhầm, bỏ câu trùng; file có khoá `data` nên `run_baseline --data_file` đọc thẳng) · `--score` kiểm hợp lệ (đủ câu, sha256, mode, adapter trùng run gốc, OCR cùng SH), metric theo cách chọn, bootstrap theo ảnh, phán quyết theo luật | `--selftest`: đọc câu trả lời (14 ca), build tất định, loại ảnh train, loại lớp dễ nhầm, phán quyết 3 nhánh + H4, bắt lỗi adapter / sha256 / mode / OCR, `load_vivqa` đọc được file câu hỏi |
| `notebooks/21_pope_vi_hallucination_qwen25vl.ipynb` | Phần A (CPU): tải annotation COCO về `/content` (không lên Drive), sinh file câu hỏi, ⏸ Điểm dừng A · Phần B (L4): kiểm OCR cache = biên bản W11, adapter = run gốc, chạy 5 cấu hình, chấm, tự trả máy | Cú pháp mọi cell; mỗi code cell có markdown mô tả + kỳ vọng |
