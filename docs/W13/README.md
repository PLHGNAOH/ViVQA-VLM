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
| 1 | Chẩn đoán thiên lệch "có": `src/eval/yesno_calib.py` + notebook 22 (logit token đầu, AUC, tập calibration) | 2,5 h (≈ 2–3 h L4) | v1 ✅ chạy 27/10 → **KHÔNG HỢP LỆ** (lỗi cách đo, xem dưới) · v2 ⏳ |
| 1b | Chẩn đoán v1 + sửa cách đo (v2: teacher forcing từ nhiều token + repetition penalty như `generate()`), không tốn GPU | 2 h | ✅ 27/10 |
| 1c–1d | Sửa ghép lô (`mm_token_type_ids`) → smoke v2 dừng ở chốt (96,7%) → v3: token đầu, max mỗi phía, forward 1 hàng | 2 h | ✅ 27/10 |
| 2 | Fix A: ngưỡng τ lúc suy luận (chọn trên calibration) — notebook 22 bản v3 | ≈ 3 h L4 (≈ 1 h nếu chỉ chạy 3 run bắt buộc) | ⏳ |
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

## R-YB v1 — kết quả (27/10/2026): **KHÔNG HỢP LỆ**

Nguồn: `experiments/W13_yesno_calib_L4_seed42/yesno_calib_summary.md` (9 run `W13_YN_*`, ≈ 3 h L4). Phán quyết theo luật: kiểm hợp lệ trượt (quyết định logit ở τ = 0 trùng greedy W12: SH **87,4%**, B1-ocr **94,4%** < 98%) → **không kết luận từ số v1**, kể cả các số ở bảng luật trông như đạt.

**Chẩn đoán** (không tốn GPU, `experiments/W13_yesno_calib_L4_seed42/v1_diagnosis.md`):
1. **Từ nhiều token bị bỏ sót.** Với tokenizer Qwen2.5-VL, "không" viết thường = `kh` + `ông`, "Có" = `C` + `ó`. v1 chỉ lấy từ 1 token, nên `s_no` thiếu đúng câu trả lời "không" phổ biến nhất của SH / B3-full. Token top-1 ngoài tập v1 gần như toàn là `kh` (SH 1.159, B3-full 2.847) hoặc `C` (B1-ocr 331).
2. **Greedy có repetition penalty, v1 thì không.** Xét các câu có top-1 (logit, không phạt) là "có": greedy W12 **luôn** đổi sang "Không"/"no" khi margin < 0,5, **không bao giờ** đổi khi margin > 1 (cả 3 cấu hình). Ngưỡng sắc nét này khớp với một khoản phạt cố định lên logit "có", không khớp với nhiễu số học. Giải thích phù hợp nhất: `model.generate()` dùng `generation_config` mặc định của Qwen2.5-VL-3B-Instruct có `repetition_penalty` > 1, phạt token đã có trong prompt (`Trả lời "có" hoặc "không"`). Giá trị thật được đọc và ghi vào `meta.json` ở v2.

**Hệ quả cho các tuần trước (ghi nhận, không sửa số):** mọi run eval W05–W12 dùng cùng `generate()` nên so sánh giữa các run vẫn công bằng. Tuy vậy, dòng `decoding: greedy (do_sample=False)` trong meta là **chưa đủ**: phần Phương pháp (R4) phải ghi giá trị repetition penalty. Giả thuyết hậu kiểm, chưa kiểm định: penalty có thể là nguyên nhân của code-switching "no" ở B3-full (17,7% câu POPE-vi, W12).

## R-YB v2 — sửa cách đo (ghi trước, commit TRƯỚC khi chạy notebook 22 bản v2)

**Chỉ đổi cách đo; mọi ngưỡng giữ nguyên v1** (trùng greedy ≥ 98%; cận dưới CI95 ΔF1 > −2,0; yes-ratio ≤ 60%; cả 3 cách chọn; mốc B1-ocr greedy W12; bootstrap theo ảnh 10.000 lần, seed 42; lưới τ −15…+15 bước 0,05, balanced accuracy, hoà chọn |τ| nhỏ nhất).

**Đo** (`teacher-forced-reppen-v2`):
- Từ ứng viên: có {có, Có, yes, Yes}, không {không, Không, no, No}. Bỏ dạng viết hoa toàn bộ (CÓ, KHÔNG) vì không run greedy W12 nào sinh ra; "KHÔNG" dài 3 token.
- `log P(từ) = Σ log p(token_i | prompt, token_<i)`, chấm bằng teacher forcing trong **một** lần forward lô M hàng (hàng j = prompt + token đầu của từ nhiều token thứ j; mọi từ nhiều token phải cùng độ dài, hiện là 2).
- Ở mỗi bước áp **đúng `RepetitionPenaltyLogitsProcessor`** của transformers với `repetition_penalty` đọc từ `model.generation_config` (tập token đã thấy = toàn bộ `input_ids` của prompt + các token đã nối). Code dừng nếu `generation_config` có thêm bộ xử lý logit khác (no_repeat_ngram, bad_words, sequence_bias…) chưa mô phỏng.
- `s_yes / s_no = logsumexp` trên các từ của từng phía; `margin = s_yes − s_no`. Ghi kèm `margin_raw` (không penalty) để mô tả.

**Kiểm hợp lệ** bổ sung so với v1: run không có `--limit` (run smoke không dùng được cho luật); mọi run cùng giá trị repetition penalty.

**Smoke test trước khi chạy đầy đủ** (không thuộc luật, chỉ để tiết kiệm GPU): 300 câu POPE-vi đầu, SH. Trùng greedy W12 < 98% → dừng, không chạy tiếp.

**Smoke lần 1 (27/10 ≈ 15:25) — dừng ở chốt chặn, chưa chấm câu nào:** processor của transformers 5.17 trả thêm khoá `mm_token_type_ids` (0 = chữ, 1 = ảnh). Qwen2.5-VL dùng khoá này để tính vị trí 3D (mRoPE) cho token ảnh; code v2 chỉ nhận các khoá đã biết nên dừng. **Sửa lỗi code (không đổi cách đo, không đổi luật):** nhân hàng khoá này như `input_ids`, token nối thêm gán 0 (chữ). Kiểm bằng Qwen2.5-VL tí hon kiến trúc thật (transformers 5.17, CPU, trong `--selftest`): logit lô 2 hàng trùng forward 1 hàng (Δ ≈ 2·10⁻⁷); bỏ khoá này thì logit lệch (Δ ≈ 1,5·10⁻³); `apply_repetition_penalty` trùng `RepetitionPenaltyLogitsProcessor` (Δ = 0). Log lần 1 xác nhận **`repetition_penalty = 1.05`** (đọc từ `generation_config` của Qwen2.5-VL-3B-Instruct) và tách token như chẩn đoán (`không` = [30664, 24015], `Có` = [34, 1794]).

**Run:** tiền tố mới `W13_YN2_*` (giữ nguyên run v1 `W13_YN_*` làm hồ sơ). Bắt buộc cho luật: SH calib, SH POPE-vi, B1-ocr POPE-vi. Chỉ mô tả: SH t2, B1-ocr calib + t2, B3-full × 3. Thiếu run mô tả thì vẫn phán quyết được, có ghi run còn thiếu.

**Câu hỏi:** dùng lại nguyên 2 file sinh ở v1 (nay commit vào `experiments/W13_yesno_questions/`): `yesno_calib_v1.json` (3.774 câu, 600 ảnh, sha256 `69d0316c…`), `pope_vi_questions_t2_v1.json` (4.985 câu, sha256 `e9a83055…`).

**Công khai về việc POPE-vi đã được xem:** POPE-vi đã chấm một lần ở v1 (không hợp lệ) trước khi chốt v2. Để giảm rủi ro "chỉnh cho đẹp":
- (i) v2 chỉ sửa hai lỗi đo, được xác định từ **phép kiểm hợp lệ** (danh sách token, so với greedy), không xuất phát từ số F1/yes-ratio;
- (ii) τ vẫn chỉ chọn trên calibration;
- (iii) mọi ngưỡng giữ nguyên;
- (iv) kết quả v1 được giữ trong repo và sẽ báo cáo cùng v2.

## R-YB v3 — sửa định nghĩa điểm (ghi trước, commit TRƯỚC khi chạy notebook 22 bản v3)

**Smoke v2 (27/10 ≈ 16:00) — dừng đúng chốt:** trùng greedy W12 **96,66%** (299 câu, 10 câu lệch, cả 10 là greedy "có" → v2 "không"). Chẩn đoán trên `logits.json` của smoke (chỉ mô tả):
- **8/10 câu:** "có" là token đơn lẻ cao nhất (ví dụ có 0,23; no 0,13; không 0,12; Không 0,02), nên greedy chọn "có". Nhưng v2 **cộng dồn** phía "không" (0,27 > 0,23) nên ra "không". Cộng dồn (logsumexp) **khác định nghĩa** với phép so của greedy (lấy token lớn nhất), nên v2 không thể đạt kiểm hợp lệ dù đo đúng.
- **2/10 câu:** gần hoà (có 0,193 so với no 0,202). Lệch do v2 chạy lô 2 hàng, khác về số học với forward 1 hàng của `generate()`.
- Nếu so token tốt nhất mỗi phía thì đã trùng 98,66% ngay trên dữ liệu lô 2 hàng của smoke; 4 câu còn lệch đều gần hoà (|Δp| ≤ 0,015).
- Sau `kh`, model viết `ông` với xác suất trung vị 0,984 (thấp nhất 0,893; 70 câu top-1 = `kh`) → token đầu đại diện được cho "không".

**Đo v3** (`first-token-max-reppen-v3`):
- Một forward **1 hàng** (cùng cách gọi model như v1 = bước prefill của `generate()`), áp repetition penalty **1,05** đúng như `generate()`, log-softmax ở bước sinh đầu.
- `s_yes = max log p` trên token đầu của {có, Có, yes, Yes} = {`có`, `C`, `yes`, `Yes`}; `s_no = max log p` trên token đầu của {không, Không, no, No} = {`kh`, `Không`, `no`, `No`}; `margin = s_yes − s_no`. τ = 0 đúng là phép so của greedy giữa các token ứng viên. Ghi kèm `margin_raw` (không penalty).
- Kiểm offline (`--selftest`): trên Qwen2.5-VL tí hon kiến trúc thật (transformers 5.17), điểm sau penalty của v3 **trùng tuyệt đối** `scores` của chính `model.generate(do_sample=False, repetition_penalty=1.05)` (6 input, Δ = 0), và token `generate()` chọn = argmax của v3.

**Mọi ngưỡng và quy trình khác giữ nguyên** (trùng greedy ≥ 98% với SH và B1-ocr; cận dưới CI95 ΔF1 > −2,0; yes-ratio ≤ 60%; cả 3 cách chọn; mốc B1-ocr greedy W12; τ chọn trên calibration bằng balanced accuracy; cùng 3 file câu hỏi; cùng run bắt buộc / chỉ mô tả; smoke 300 câu dừng nếu < 98%). Run mới dùng tiền tố `W13_YN3_*`; giữ `W13_YN_*` (v1) và smoke `W13_YN2_*` (v2) làm hồ sơ.

**Công khai:**
- (i) v2 chưa chấm calibration / POPE-vi đầy đủ (chỉ 300 câu smoke, chưa tính F1 hay yes-ratio), nên đổi sang v3 **không dựa trên kết quả luật nào**;
- (ii) lý do đổi đến từ phép kiểm hợp lệ;
- (iii) POPE-vi đầy đủ đã được chấm một lần ở v1 (không hợp lệ), như đã ghi ở mục v2.

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
| Notebook 22 v1 (đã chạy 27/10, không hợp lệ) | ≈ 3 h L4 (0,21–0,24 s/câu) | ≈ 4,6 |
| Notebook 22 v2: smoke 300 câu + 3 run bắt buộc (≈ 13.700 câu) + 6 run mô tả (≈ 27.500 câu), lô 2 hàng ≈ 0,4 s/câu (ước tính) | ≈ 1,6 h + ≈ 3 h L4 | ≈ 2,5 + ≈ 4,6 |
| Fix B (nếu cần) | ≈ 7 h L4 | ≈ 11 |
| Confidence + parity + đo hiệu năng | ≈ 2 h L4 | ≈ 3 |

CU được cộng thêm khi gia hạn gói ngày 02/11; giờ GPU và VRAM vẫn báo đầy đủ.

## File mới (bước 1)

| File | Nội dung | Kiểm |
|---|---|---|
| `src/eval/yesno_calib.py` | `--build` sinh calib + t2 · `--score` chấm logit token đầu theo lô (resume, chữ ký run kèm id token) · `--analyze` kiểm hợp lệ, AUC, τ*, phán quyết R-YB, báo cáo | `--selftest`: lưới τ + hoà, AUC (Mann–Whitney), loại từ nhiều token, build tất định + rời POPE-vi + loại ảnh train + số câu có = số câu không, chấm theo lô (resume, chặn lệch id, prompt dùng chung `prompt_for_sample`), phân tích 4 nhánh (ĐẠT / KHÔNG ĐẠT khi mất phân biệt / KHÔNG HỢP LỆ khi lệch greedy / bắt lỗi adapter) |
| `notebooks/22_yesbias_logit_calib_sh_qwen25vl.ipynb` | L4: kiểm OCR cache = biên bản W11, adapter = run gốc, POPE-vi = bản W12 → sinh 2 file câu hỏi → 9 lượt chấm logit → phân tích → tự trả máy | Cú pháp mọi cell; mỗi code cell có markdown mô tả + kỳ vọng |

## File mới / đổi (bước 1b, 27/10)

| File | Nội dung | Kiểm |
|---|---|---|
| `src/eval/yesno_calib.py` (v2) | Chấm `teacher-forced-reppen-v2`: từ nhiều token chấm chính xác, áp repetition penalty như `generate()`, ghi `margin` + `margin_raw` + log P từng từ + `generation_config`. `--agree` (smoke test). `--analyze`: run bắt buộc / chỉ mô tả, từ chối run smoke, kiểm cùng penalty | `--selftest`: penalty đúng ngữ nghĩa transformers, teacher forcing (log P("không") = log p(kh) + log p(ông\|kh) với tập token đã thấy cập nhật), penalty đảo quyết định như W12, ghép lô + cắt logit (model giả, có/không `logits_to_keep`), chặn từ nhiều token khác độ dài, analyze 6 nhánh |
| `notebooks/22_yesbias_logit_calib_sh_qwen25vl.ipynb` (v2, sửa tại chỗ; bản v1 ở commit `3147496`) | Kiểm câu hỏi repo = Drive → smoke 300 câu (dừng nếu < 98%) → 3 run bắt buộc → 6 run mô tả → phân tích → tự trả máy | Cú pháp mọi cell; mỗi code cell có markdown mô tả + kỳ vọng |
| `experiments/W13_yesno_questions/` | 2 file câu hỏi đã chốt (sinh ở v1) | sha256 ghi ở mục R-YB v2 |
| `experiments/W13_yesno_calib_L4_seed42/` | Hồ sơ v1: `yesno_calib_summary.json/.md` (KHÔNG HỢP LỆ) + `v1_diagnosis.md` | — |
| `src/eval/yesno_calib.py` (bước 1c, 27/10) | Nhân hàng `mm_token_type_ids` khi ghép lô; selftest thêm Qwen2.5-VL tí hon (kiến trúc thật) + so penalty với transformers | `--selftest`: `tiny Qwen2.5-VL OK (…)` |
| `src/eval/yesno_calib.py` (bước 1d, 27/10) | v3: forward 1 hàng, token đầu, max mỗi phía, repetition penalty như `generate()`; bỏ teacher forcing lô 2 hàng; selftest so với chính `model.generate()` trên Qwen2.5-VL tí hon | `--selftest`: `tiny Qwen2.5-VL OK (… = scores của generate …)` |
| `notebooks/22_yesbias_logit_calib_sh_qwen25vl.ipynb` (v3, sửa tại chỗ) | Run `W13_YN3_*`, kết quả `W13_yesno_calib_v3_L4_seed42/`; mục 2 bắt buộc selftest có phép so với `generate()` | Cú pháp mọi cell |
