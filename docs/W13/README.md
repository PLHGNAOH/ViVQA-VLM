# W13 · Prototype web

> Tuần ảo W13 (bắt đầu 26/10/2026, tuần thực 8). Mốc master: "W14 · Prototype + thesis" được kéo lên sớm. Hội đồng cấp Khoa là **tuần thực 13 (30/11–06/12)**, không phải tuần ảo này.
> Review 2 (thứ Năm 29/10) làm ở phiên chat riêng, **không** thuộc phạm vi README này.
> Colab đầu tuần: đọc ở **Colab → Resources** khi mở notebook 22 và ghi vào đây.

## Vấn đề trước khi dựng giao diện

| # | Vấn đề | Vì sao chặn prototype |
|---|---|---|
| P1 | **SH thiên lệch "có"** (W12 POPE-vi: ΔF1 −5,7 / −9,2 / −9,4; yes-ratio 77–89%; báo sai 55–79%) | Prototype dùng SH. Câu "Trong ảnh có X không?" gần như luôn được trả lời "có", tức là hallucination lộ ngay khi demo |
| P2 | **Confidence chưa có định nghĩa và chưa được kiểm** | Hiển thị một con số không khớp tỉ lệ đúng thật là gây hiểu lầm (master §13 Responsible AI) |
| P3 | **Lệch pipeline giữa demo và eval** | B3/SH phụ thuộc OCR (W09: bỏ OCR lúc test −4,25 EM). Prototype dựng prompt/OCR khác đi thì số đã báo cáo không còn đúng cho demo |
| P4 | **Paddle GPU xung đột torch** (W08: NCCL bị ghi đè) | Prototype phải chạy OCR và VLM trong **cùng một runtime**, khác với eval (OCR tính trước thành cache) |

## Việc trong tuần

| # | Việc | Ước tính | Trạng thái |
|---|---|---|---|
| 0 | README W13 + luật R-YB ghi trước | 1,5 h | ✅ (commit cùng code bước 1) |
| 1 | Chẩn đoán thiên lệch "có": `src/eval/yesno_calib.py` + notebook 22 (logit token đầu, AUC, tập calibration) | 2,5 h (≈ 2–3 h L4) | ⏳ |
| 2 | Fix A: ngưỡng τ lúc suy luận (chọn trên calibration, áp POPE-vi 1 lần) — cùng notebook 22 | (trong bước 1) | ⏳ |
| 3 | *(chỉ khi R-YB KHÔNG ĐẠT)* Fix B: train lại SH có câu có/không cân bằng | ≈ 10 h (≈ 7 h L4) | — |
| 4 | Confidence: định nghĩa + ECE + 3 mức Cao / Trung bình / Thấp (luật R-CONF chốt trước khi chạy) | 2,5 h (≈ 1 h L4) | ⏳ |
| 5 | Backend `src/prototype/` (dùng lại `prompt_for_sample`, `ocr.py`, adapter SH 4-bit; OCR det chạy CPU) | 4 h | ⏳ |
| 6 | Kiểm khớp pipeline (parity) prototype ↔ eval, 200 + 200 câu (luật R-PAR) | 1,5 h | ⏳ |
| 7 | Giao diện Gradio (ảnh, câu hỏi, câu trả lời, confidence, OCR + box, cảnh báo, disclaimer) | 3 h | ⏳ |
| 8 | Đo hiệu năng (latency p50/p95 tách OCR / VLM, VRAM, thời gian khởi động) + thử 15–20 ảnh thật | 2 h | ⏳ |
| 9 | Tài liệu: hướng dẫn chạy prototype, Team Report W13, video demo dự phòng | 2 h | ⏳ |
| 10 | Đóng tuần: commit + push, dọn Downloads + `G:\My Drive\ViVQA-VLM` | 1 h | ⏳ |
| | **Tổng** (không Fix B / có Fix B) | **≈ 21,5 h / ≈ 31,5 h** | |

## Quyết định

| Ngày | Quyết định | Lý do |
|---|---|---|
| 23/10/2026 (W12) | Sửa thiên lệch "có" của SH ở **đầu W13**, chẩn đoán trước rồi mới sửa | Đây là việc chuẩn bị prototype; W12 đã bác giả thuyết "dữ liệu có/không bị lệch" (0 đáp án "có" trong mọi tập train) |
| 26/10/2026 | **Chẩn đoán bằng logit token đầu** (1 forward, không sinh chữ) thay vì chạy lại greedy | Rẻ hơn khoảng 3 lần. Cho biết thiên lệch là *lệch ngưỡng* (AUC còn cao → dời ngưỡng sửa được) hay *mất phân biệt* (AUC thấp → phải train lại) |
| 26/10/2026 | **Fix A trước Fix B**: ngưỡng τ lúc suy luận, không train | Không tốn train, không đụng trọng số nên không đổi kết quả EM/ANLS đã báo cáo của SH. Chỉ áp cho câu có/không. Vẫn trong ràng buộc PEFT (không train gì thêm) |
| 26/10/2026 | Tập calibration = **ảnh ViVQA test chưa thấy lúc train, không thuộc 500 ảnh POPE-vi** | Chọn τ trên chính POPE-vi là "học vẹt" tập đánh giá. Ảnh train thì SH đã thấy lúc huấn luyện, nên biên độ có thể khác ảnh mới người dùng tải lên. Phần ảnh test còn lại có sẵn OCR cache W11 và annotation COCO |
| 26/10/2026 | Thêm **POPE-vi mẫu câu 2** `Có {vật} trong ảnh không?` (không có chỉ dẫn trả lời), chỉ mô tả | Người dùng prototype không gõ "Trả lời có hoặc không"; cần biết τ có còn đúng khi đổi cách hỏi |

## R-YB — luật ghi trước cho Fix A (commit TRƯỚC notebook 22)

**Câu hỏi:** dời ngưỡng quyết định "có/không" của SH (không train lại) có đưa hallucination vật thể về mức không kém zero-shot không?

**Đo** (`src/eval/yesno_calib.py`, `first-token-logsumexp-v1`):
- Một lượt forward, **cùng input với greedy** (`prompt_for_sample` + `to_chat_messages` + processor, 4-bit NF4, bf16, 512 token ảnh, L4). Lấy log-softmax tại bước sinh **đầu tiên**.
- `s_yes = logsumexp` trên token của {"có", "Có", "CÓ", "yes", "Yes"}, `s_no = logsumexp` trên token của {"không", "Không", "KHÔNG", "no", "No"}. **Chỉ lấy từ mã hoá thành đúng 1 token**: từ nhiều token có token đầu là tiền tố chung, sẽ lẫn xác suất của câu trả lời khác. Danh sách từ thật sự dùng được ghi vào `meta.json` từng run.
- `margin = s_yes − s_no`. Quyết định "có" ⇔ `margin > τ`.
- Cấu hình: **SH** (đối tượng), **B1-ocr** (mốc, cùng prompt ocr như luật H4), **B3-full** (chỉ mô tả). OCR: VietOCR · v1 · 600 · 0 (cache W11). Adapter kiểm bằng dấu vân tay trọng số = run greedy W12.

**Ba tập câu hỏi:**

| Tập | Nội dung | Vai trò |
|---|---|---|
| `calib` (`yesno_calib_v1.json`) | Ảnh ViVQA test chưa thấy lúc train, **không** thuộc 500 ảnh POPE-vi, ≥ 1 lớp COCO, tối đa 600 ảnh, seed 42. Mỗi ảnh: k = min(3, số lớp) câu "có" + k câu "không" cho mỗi cách chọn random / popular / adversarial (cùng luật chọn và loại lớp dễ nhầm như POPE-vi), cùng mẫu câu POPE-vi, bỏ câu trùng | **Chỉ để chọn τ** |
| `popevi` (`pope_vi_questions_v1.json`, sha `53d04eb2…`, W12) | Bộ câu W12 đã chốt | **Đánh giá theo luật, 1 lần** |
| `popevi_t2` (`pope_vi_questions_t2_v1.json`) | Như `popevi`, câu hỏi đổi thành `Có {vật} trong ảnh không?` | Chỉ mô tả |

File `calib` và `t2` sinh **tất định** từ code + seed trong commit này. sha256 ghi vào summary; Cursor chép file vào `experiments/W13_yesno_questions/` ở bước đóng.

**Chọn τ** (cho từng cấu hình, trên `calib`; mỗi câu duy nhất tính 1 lần): `τ* = argmax balanced accuracy = ½(TPR + TNR)` trên lưới −15…+15 bước 0,05; hoà thì chọn |τ| nhỏ nhất, rồi τ nhỏ hơn. Dùng balanced accuracy (không dùng F1) vì F1 thưởng recall, tức là chính cái lệch "có" cần sửa (bài học W12: F1 che tỉ lệ báo sai của B3-full).

**Kiểm hợp lệ** (kiểm trước khi đọc kết quả, sai thì **KHÔNG HỢP LỆ**, không kết luận):
1. Mỗi run đủ câu, đúng sha256 file câu hỏi, mode ocr, đúng `first-token-logsumexp-v1`.
2. Adapter trùng run greedy W12. OCR cùng thiết lập SH W12.
3. Trên POPE-vi, quyết định logit ở τ = 0 **trùng greedy W12 ≥ 98%** (tính trên các câu greedy đọc được) với **SH và B1-ocr**. Điều này chứng minh logit token đầu đo đúng cái greedy đang làm.

**Phán quyết R-YB** (SH với τ* của SH, so với **B1-ocr greedy W12** — cùng mốc với luật H4, ΔF1 ghép cặp, CI95 bootstrap theo ảnh, 10.000 lần, seed 42):
- **ĐẠT** nếu đủ cả hai điều kiện ở **cả 3** cách chọn:
  - cận dưới CI95 của ΔF1 > −2,0 (giống mức "KHÔNG TĂNG HALLUCINATION" của H4);
  - yes-ratio ≤ 60%.
- **KHÔNG ĐẠT** nếu hợp lệ nhưng thiếu một trong hai → sang Fix B (bước 3).
- Báo kèm (không thuộc luật): tỉ lệ báo sai FP / (FP + TN) của SH-sửa và B1-ocr.

**Chỉ mô tả, không có luật:**
- AUC của margin theo cách chọn (POPE-vi, t2, calib) cho 3 cấu hình. AUC SH ≈ B1-ocr nghĩa là lệch ngưỡng; AUC SH thấp hơn rõ nghĩa là mất phân biệt.
- SH ở τ = 0 (kiểm khớp với số greedy W12); B1-ocr và B3-full với τ* của chính chúng.
- Mẫu câu 2: SH với τ*, SH / B1-ocr / B3-full ở τ = 0.
- Tỉ lệ câu có token top-1 là token có/không (model có trả lời đúng định dạng không), thời gian / câu, VRAM.

**Dùng trong prototype nếu ĐẠT:** τ* áp cho câu được bộ nhận diện câu có/không đánh dấu (định nghĩa ở bước 5, có thống kê số câu ViVQA / ViTextVQA test bị đánh dấu). Câu khác vẫn đi greedy như eval. EM/ANLS chính thức của SH **không đổi** vì trọng số và greedy giữ nguyên. Hậu xử lý ("no" → "không") chỉ áp ở prototype và sẽ được báo cáo riêng.

**Giới hạn đã biết:** kế thừa giới hạn POPE-vi W12 (annotation COCO thiếu; backbone có thể đã thấy ảnh COCO khi pretrain). Calibration cùng mẫu câu POPE-vi nên τ có thể lệch với cách hỏi khác; t2 đo mức lệch này.

## Fix B (chỉ khi R-YB KHÔNG ĐẠT) — phác thảo, luật chi tiết chốt trước khi train

Train lại SH (cùng công thức S2 W11) có thêm câu có/không cân bằng 50/50 từ **ảnh ViVQA train** (annotation COCO). Mẫu câu khác POPE-vi, và không dùng ảnh POPE-vi lẫn ảnh calibration. Kiểm lại:
1. Không kém SH hiện tại trên ViVQA test và ViTextVQA test2000 (δ = 1,5 EM như W11).
2. POPE-vi ở τ = 0 theo cùng tiêu chí R-YB.

## R-CONF, R-PAR — bản nháp (chốt trước khi chạy bước 4 / bước 6)

- **R-CONF (confidence):** `conf = exp(trung bình log-prob các token trả lời)` (greedy). Câu có/không dùng `sigmoid(margin − τ*)`. Chia 3 mức Cao / Trung bình / Thấp bằng ngưỡng chọn **trên val** (ViVQA val sạch 688, ViTextVQA dev). Báo ECE (15 bin) + reliability diagram trên test. ĐẠT nếu EM test giảm **đơn điệu** Cao > Trung bình > Thấp. ECE cao thì giao diện ghi "độ tin cậy tương đối", không bao giờ ghi "% đúng".
- **R-PAR (khớp pipeline):** backend prototype chạy 200 câu ViVQA test + 200 câu ViTextVQA test2000 (seed 42), so với `predictions.json` của run eval SH. ĐẠT nếu ≥ 99% câu trả lời trùng hệt và mọi câu lệch đều giải thích được (ví dụ khác OCR vì prototype chạy OCR trực tiếp thay vì đọc cache).

## Ngân sách

| Việc | GPU | CU (L4 ≈ 1,54 CU/h) |
|---|---|---|
| Notebook 22: 3 cấu hình × (calib ≈ 4–6 nghìn + 4.985 + 4.985 câu), 1 forward / câu | ≈ 2–3 h L4 | ≈ 4 |
| Fix B (nếu cần) | ≈ 7 h L4 | ≈ 11 |
| Confidence + parity + đo hiệu năng | ≈ 2 h L4 | ≈ 3 |

CU được cộng thêm khi gia hạn gói ngày 02/11; giờ GPU và VRAM vẫn báo đầy đủ.

## File mới (bước 1)

| File | Nội dung | Kiểm |
|---|---|---|
| `src/eval/yesno_calib.py` | `--build` sinh calib + t2 · `--score` chấm logit token đầu theo lô (resume, chữ ký run kèm id token) · `--analyze` kiểm hợp lệ, AUC, τ*, phán quyết R-YB, báo cáo | `--selftest`: lưới τ + hoà, AUC (Mann–Whitney), loại từ nhiều token, build tất định + rời POPE-vi + loại ảnh train + số câu có = số câu không, chấm theo lô (resume, chặn lệch id, prompt dùng chung `prompt_for_sample`), phân tích 4 nhánh (ĐẠT / KHÔNG ĐẠT khi mất phân biệt / KHÔNG HỢP LỆ khi lệch greedy / bắt lỗi adapter) |
| `notebooks/22_yesbias_logit_calib_sh_qwen25vl.ipynb` | L4: kiểm OCR cache = biên bản W11, adapter = run gốc, POPE-vi = bản W12 → sinh 2 file câu hỏi → 9 lượt chấm logit → phân tích → tự trả máy | Cú pháp mọi cell; mỗi code cell có markdown mô tả + kỳ vọng |
