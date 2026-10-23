# W12 · Hallucination + Error analysis

> Tuần ảo W12 (bắt đầu 23/10/2026, tuần thực 7). Mốc master: "W12 · Analysis — ablation + error + hallucination".
> Review 2 (29/10) làm ở phiên chat riêng, **không** thuộc phạm vi README này.
> Colab đầu tuần (23/10 10:15, đọc ở Resources): **34,96 CU** (W11 ước tính ≈ 37 → lệch −2).

## Việc trong tuần

| # | Việc | Ước tính | Trạng thái |
|---|---|---|---|
| 1 | Khởi động: README W12 + luật H4 ghi trước + từ điển `configs/pope_vi_objects.json` | 0,5 h | ✅ |
| 2 | Code `src/eval/pope_vi.py` (sinh câu hỏi, chấm, phán quyết) + selftest + notebook 21; commit luật TRƯỚC khi chạy | 1,5 h | ✅ |
| 3 | Colab L4: chạy POPE-vi 5 cấu hình | ≈ 4,5 h GPU (≈ 7 CU) | ✅ (23/10, ≈ 3 h 20 phút L4) |
| 4 | Error analysis C3 (ViVQA, 4 loại có sẵn) + B3-full (ViTextVQA, `qt-v1`), luật phân loại lỗi chốt trước, không cần GPU | 2,5 h | ⏳ (4a tự động ✅ · 4b audit tay: chờ SV4) |
| 5 | Kết luận H4 theo luật + ghi README | 1 h | ✅ |
| 6 | Ví dụ định tính 6–8 failure case | 1,5 h | ✅ (12 ví dụ, R6 §7) |
| 7 | R6 §5–§7 bản tiếng Việt | 2 h | ✅ (bản nháp; §5.6 chờ audit) |
| 8 | Sửa đánh số cũ trong `docs/W02/Hypotheses.md` (H8 cũ = H4 mới) | 0,3 h | ✅ (bảng đối chiếu, không đổi số cũ) |
| 9 | Đóng tuần: Team Report, dọn Downloads + Drive, commit + push | 1 h | ☐ |

## Quyết định

| Ngày | Quyết định | Lý do |
|---|---|---|
| 23/10/2026 | **H4 đo theo kiểu POPE (Li et al., EMNLP 2023) bằng tiếng Việt: POPE-vi** trên ảnh test ViVQA (ảnh COCO 2014), nhãn vật thể lấy từ annotation COCO `instances_train2014/val2014` | ViVQA là ảnh COCO nên có nhãn vật thể thật, dựng được đúng 3 cách chọn câu "không" của POPE. ViTextVQA không có nhãn vật thể |
| 23/10/2026 | **5 cấu hình**: B1-plain, B1-ocr, C3, B3-full, SH | C3 và B3-full là hai adapter chính thức, SH là adapter của prototype. B1-ocr để so B3-full/SH với zero-shot **cùng prompt**, tách hiệu ứng QLoRA khỏi hiệu ứng dòng OCR |
| 23/10/2026 | **Quy mô**: 500 ảnh × (3 câu "có" dùng chung + 3 câu "không" cho mỗi cách random/popular/adversarial) = **6.000 câu / cấu hình** | Đúng khuôn POPE (500 ảnh, 50% có / 50% không mỗi cách chọn). Câu "có" dùng chung cho 3 cách nên chênh giữa 3 cách chỉ do câu "không"; tiết kiệm 1/3 GPU so với 9.000 câu. Cặp (ảnh, vật) trùng giữa các cách chọn chỉ hỏi **1 lần** (greedy ⇒ cùng prompt cùng đáp án) nên số câu chạy thật < 6.000; mỗi cách chọn vẫn chấm đủ 3.000 lượt (bổ sung 23/10, trước khi chạy) |
| 23/10/2026 | Ảnh chỉ lấy từ ViVQA test **không** xuất hiện trong ViVQA train | ViVQA train và test có ảnh dùng chung; C3/SH đã thấy ảnh train lúc huấn luyện |
| 23/10/2026 | ~~Không train thêm run nào từ W12 trở đi~~ → **bỏ** (cùng ngày): CU được cộng thêm khi gia hạn Google AI Pro (02/11); run mới được phép nếu có lý do khoa học | Ngân sách không còn là ràng buộc; hiệu quả (giờ GPU, VRAM) vẫn báo đầy đủ vì là luận điểm của đề tài |

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
- sha256 file câu hỏi `pope_vi_questions_v1.json`: `53d04eb23ad41ecee372d6dee01201578de75d992d33dd2c5b94d4aa328928fd` (sinh 23/10/2026 trên Colab CPU, commit trước khi chạy model)

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

## Điểm dừng A — file câu hỏi đã chốt (23/10/2026)

| Mục | Giá trị |
|---|---|
| File | `experiments/W12_pope_vi_questions/pope_vi_questions_v1.json` (bản Drive trùng byte) |
| Câu duy nhất | **4.985** (500 ảnh; mỗi cách chọn random / popular / adversarial đủ 3.000 lượt chấm; bỏ trùng tiết kiệm 1.015 câu) |
| Pool ảnh | 2.789 ảnh ViVQA test → 1.413 ảnh **không** có trong train/val → 618 ảnh có ≥ 3 lớp COCO → chọn 500 (seed 42) |
| popular top 10 | person, chair, dining table, bottle, cup, bowl, car, handbag, book, couch |
| Thời gian dự kiến Phần B | 5 × 4.985 câu × ≈ 0,57 s ≈ 4,2 h L4 ≈ 6,5 CU |

**Trùng ảnh train–test ViVQA (đối chiếu kiểm tra rò rỉ W07, `src/eval/stratify_by_overlap.py`):** W07 đã đo 1.274 / 2.789 ảnh test (46%) có trong train; tính cả val là 1.376 / 2.789 (**49%**) (khác câu hỏi, cùng ảnh). Không phải phát hiện mới; ghi cho R3/R6 và câu hỏi của thầy Trí về nhiễu dataset. Điểm ViVQA test vì vậy có phần ảnh model đã thấy lúc train. POPE-vi chỉ dùng ảnh chưa thấy nên không bị ảnh hưởng. Error analysis W12 nên tách EM ViVQA test theo nhóm ảnh đã thấy / chưa thấy (chỉ mô tả).

## Error analysis — luật ghi trước (23/10/2026, commit trước khi chạy)

**Mục tiêu:** biết lỗi của hai mô hình chính thức (B3-full, C3) đến từ đâu. **Chỉ mô tả**, không phán quyết, không dùng để chọn mô hình. Script `src/eval/error_analysis.py` (định nghĩa cố định trong code, có selftest). Không tốn GPU: chỉ đọc predictions đã có.

**Đầu vào:** ViTextVQA full test 10.028 câu (B1, B2, B3-full) · ViVQA test 3.001 câu (B1, C1, C3). Script kiểm sha256 file test trùng run gốc và thiết lập OCR của B3-full = VietOCR · v1 · 600 · 0.

**Định nghĩa:**
- Kiểu lỗi (câu sai EM): dùng nguyên `answer_format.error_bucket` (W07), thứ tự ưu tiên tone → truncated → extended → near_miss → other.
- Trạng thái OCR của câu ViTextVQA (văn bản OCR đúng phần đã vào prompt, cắt 600 ký tự): `has_answer` (đáp án là một cụm trong OCR) · `wrong_diacritics` (chỉ khớp khi bỏ dấu) · `answer_missing` (OCR có chữ nhưng không có đáp án) · `no_text`.
- Nguồn lỗi B3-full: `ocr_upstream` (OCR không đưa đáp án vào prompt) · `model_wrong_span` (OCR có đáp án, model chép sai đoạn OCR) · `model_outside_ocr` (OCR có đáp án, câu trả lời không nằm trong OCR).
- B2 → B3-full ghép cặp: fixed / broken / both_right / both_wrong, tách theo trạng thái OCR; % câu bị làm hỏng mà câu trả lời nằm trong OCR (proxy "OCR làm phân tâm").
- ViVQA: kiểu lỗi C3 × 4 loại câu hỏi; 10 cặp (gold → dự đoán) sai nhiều nhất theo loại; EM tách theo ảnh đã thấy / chưa thấy (train + val), B1 vs C3.
- Loại câu hỏi ViTextVQA: `qt-v1` (đóng băng W11), gộp loại như W11.
- Ứng viên ví dụ định tính (bước 6): chọn tất định seed 42, tối đa 5 câu / nhóm.

**Audit tay (bước 4b, sau khi có số tự động):** 100 lỗi B3-full (full test) + 50 lỗi C3, chọn ngẫu nhiên seed 42 trong các câu sai EM; gán theo đúng bộ nhãn W07 (F S O T H C N G K + "đúng ý" + "hallucination"); gán **mù** (không xem kiểu lỗi / nguồn lỗi tự động, không xem dự đoán của run khác). So với audit W07 của B2 (chỉ mô tả).

**Người gán (23/10/2026):** Văn Quang Trường (SV4 · Evaluation/Hallucination), một người gán, gán mù (chưa xem kết quả error analysis tự động). Hướng dẫn: `docs/W12/audit/HUONG_DAN_GAN_NHAN_W12.md`. Phiếu gốc: `audit/W12/W12_audit_150.xlsx` trên Drive (danh sách 150 câu đã khoá ở `docs/W12/audit/audit_manifest.json`, commit `e3e6168`). Giới hạn: một người gán nên không tính được κ trong W12; κ của bộ nhãn này đo ở W07 (Huân–Trường 0,62).

## Kết quả H4 — POPE-vi (23/10/2026)

Nguồn: `experiments/W12_pope_vi_L4_seed42/pope_vi_summary.md` (bảng đầy đủ 5 cấu hình × 3 cách chọn). 5 run: `experiments/W12_POPE_*`.

| So sánh (luật ghi trước) | ΔF1 random | ΔF1 popular | ΔF1 adversarial | Phán quyết |
|---|---|---|---|---|
| C3 vs B1-plain | — | — | — | **CHƯA KẾT LUẬN** (91,43% câu không đọc được > 5% → không hợp lệ) |
| B3-full vs B1-ocr | +5,50 [+4,22; +6,80] | +3,65 [+2,29; +5,02] | +2,47 [+1,08; +3,88] | **KHÔNG TĂNG HALLUCINATION** |
| SH vs B1-ocr (prototype) | −5,69 [−7,30; −4,06] | −9,16 [−10,73; −7,57] | −9,42 [−10,98; −7,82] | **TĂNG HALLUCINATION** |

**➜ H4 theo luật: MỘT PHẦN.** Không đổi phán quyết sau khi xem số.

**Phân tích sau (KHÔNG thuộc luật, chỉ mô tả):**
- Tỉ lệ báo sai (câu "không" bị trả lời "có", trên 1.500 câu "không" mỗi cách chọn; tính từ Precision/Recall trong bảng): B1-ocr 2,7 / 5,7 / 8,9% · **B3-full 8,7 / 16,1 / 22,4%** · SH 55,4 / 72,4 / 78,7% (random / popular / adversarial). F1 của B3-full tăng vì recall tăng (73,5 → 86,9%) nhưng precision giảm → **F1 (chọn từ W02 theo LLaVA) không bắt được việc B3-full nói "có" sai nhiều hơn**. Ghi vào Discussion như giới hạn của metric.
- C3 gần như không trả lời "có/không" (lệch định dạng sau QLoRA 2 epoch trên ViVQA — dataset không có câu có/không). Cần xem C3 trả lời gì thay thế.
- SH có thiên lệch "có" mạnh (yes-ratio 77–89%) → rủi ro cho prototype W13. Đang kiểm giả thuyết: dữ liệu có/không trong ViTextVQA train lệch về "có".
- B1-ocr vs B1-plain: dòng OCR làm F1 zero-shot giảm nhẹ (−0,6 đến −1,2 điểm), chủ yếu qua recall.

### H4 — phân tích bổ sung + quyết định (23/10/2026, KHÔNG thuộc luật)

- **C3 trả lời câu có/không bằng danh từ** ("bức ảnh" 525, "cái ghế" 434; chỉ 372 / 4.985 câu là "không") → quên kỹ năng tuân theo chỉ dẫn sau QLoRA 2 epoch trên ViVQA (không có câu có/không).
- **B3-full code-switching:** trả lời tiếng Anh "no" ở 881 / 4.985 câu (17,7%), gần như luôn đúng (802 ở câu "không").
- **Giả thuyết "SH thiên lệch 'có' do dữ liệu có/không lệch" bị bác:** đáp án "có" = 0 trong ViTextVQA train (35.159), tập con 8k (8.000) và ViVQA train (10.799); câu hỏi dạng có/không chỉ 18 / 5 / 7. Nguyên nhân thiên lệch của SH chưa rõ.
- **Quyết định (Hoàng, 23/10):** sửa SH để sang **đầu W13** (chẩn đoán rồi mới sửa, vì sửa SH là việc chuẩn bị prototype). W12 ghi giới hạn này trong R6 §6.
- Viết đầy đủ ở `docs/03_report/R6_Results.md` §6.
