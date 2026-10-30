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
| 1 | Chẩn đoán thiên lệch "có": `src/eval/yesno_calib.py` + notebook 22 (logit token đầu, AUC, tập calibration) | 2,5 h (≈ 2–3 h L4) | v1 → KHÔNG HỢP LỆ · v2 → dừng ở smoke · **v3 ✅ HỢP LỆ** (27/10) |
| 1b | Chẩn đoán v1 + sửa cách đo (v2: teacher forcing từ nhiều token + repetition penalty như `generate()`), không tốn GPU | 2 h | ✅ 27/10 |
| 1c–1d | Sửa ghép lô (`mm_token_type_ids`) → smoke v2 dừng ở chốt (96,7%) → v3: token đầu, max mỗi phía, forward 1 hàng | 2 h | ✅ 27/10 |
| 2 | Fix A: ngưỡng τ lúc suy luận (chọn trên calibration) — notebook 22 bản v3 | ≈ 3 h L4 | ✅ 27/10 — **R-YB v3: ĐẠT** (τ* = +4,85) |
| 3 | *(chỉ khi R-YB KHÔNG ĐẠT)* Fix B: train lại SH có câu có/không cân bằng | ≈ 10 h (≈ 7 h L4) | ⏭ bỏ (R-YB ĐẠT) |
| 4–6 | **Gộp:** backend `src/prototype/core.py` + parity (R-PAR) + độ tin cậy (R-CONF) + detector có/không (R-YB-det), notebook 23 | ≈ 3 h (≈ 1,7 h L4) | ✅ 27/10 — **R-PAR · R-CONF · R-YB-det: ĐẠT** |
| 7 | Giao diện Gradio (ảnh, câu hỏi, câu trả lời, confidence, OCR + box, cảnh báo, disclaimer) | 3 h | ✅ 29/10 — code + selftest + ảnh chụp giao diện (backend giả, CPU, 0 CU); model thật chạy ở bước 8 |
| 8 | Đo hiệu năng (latency p50/p95 tách OCR / VLM, VRAM, thời gian khởi động) + thử 15–20 ảnh thật | 2 h | ⏳ luật R-OCR / R-LIVE ghi trước (30/10) · notebook 24 |
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

## Kết quả R-YB v3 (27/10/2026): **ĐẠT** → không cần Fix B

Nguồn: `experiments/W13_yesno_calib_v3_L4_seed42/yesno_calib_summary.md` (9 run `W13_YN3_*` + smoke, L4, 16:20–19:09). Claude đã tính lại độc lập τ*, F1, yes-ratio và tỉ lệ báo sai từ `logits.json` thô: trùng summary.

**Kiểm hợp lệ:** quyết định τ = 0 trùng greedy W12: SH **100,0%** (4.977 câu đọc được), B1-ocr **100,0%**, B3-full 99,94% (3 câu lệch) → **HỢP LỆ**. Nếu bỏ penalty thì chỉ còn 99,2% / 95,8% / 97,8%, xác nhận greedy W05–W12 có `repetition_penalty = 1.05`.

**Luật** (SH với τ* = **+4,85**, chọn trên calibration: balanced accuracy 74,6% → 91,1%; so với B1-ocr greedy W12):

| Cách chọn | F1 SH-sửa | F1 B1-ocr | ΔF1 [CI95] | yes-ratio |
|---|---|---|---|---|
| random | 85,14 | 83,39 | +1,75 [+0,68; +2,89] | 41,5% |
| popular | 83,32 | 81,99 | +1,33 [+0,21; +2,50] | 43,5% |
| adversarial | 81,69 | 80,59 | +1,11 [−0,05; +2,30] | 45,4% |

So với trước khi sửa (SH greedy W12): yes-ratio 77–89% → 41–45%; F1 71–78 → 82–85.

**Chẩn đoán (chỉ mô tả):** AUC của SH 0,944 / 0,917 / 0,897, so với B1-ocr 0,955 / 0,934 / 0,910. SH vẫn xếp câu có-thật trên câu không-có gần như B1 (AUC thấp hơn 0,011–0,017), nên thiên lệch "có" của SH **chủ yếu là lệch ngưỡng**, không phải mất khả năng phân biệt. Đây là lý do dời ngưỡng sửa được mà không cần train lại.

**Phân tích sau (KHÔNG thuộc luật, phải báo cáo):**
- **Tỉ lệ báo sai** (câu "không" bị trả lời "có") của SH-sửa vẫn **cao hơn B1-ocr 2,4–4,0 điểm** (5,1 / 9,1 / 12,9% so với 2,7 / 5,7 / 8,9%). F1 đạt nhờ recall cao hơn. Đây là cùng giới hạn của F1 đã ghi ở W12 → ghi vào Discussion.
- **Mẫu câu 2** `Có {vật} trong ảnh không?` (không có chỉ dẫn trả lời):
  - SH với τ* thì **quá dè dặt** (yes-ratio 34–35%, F1 77,7–78,8, báo sai 1,6–3,8%);
  - SH ở τ = 0 cho F1 82,7–87,7 nhưng báo sai 17–30%;
  - 31% câu có token đầu **không phải** có/không (`c…`, `b…`: SH trả lời bằng tên vật).
  
  → τ* chỉ đúng với mẫu câu đã calibration. Đề xuất cho prototype (chốt và ghi trước ở bước 5): câu có/không của người dùng được **chuẩn hoá về mẫu POPE-vi** (thêm `Trả lời "có" hoặc "không".`) trước khi áp τ*.
- B1-ocr và B3-full với τ* của chính chúng (−2,05 / +0,45) chỉ đổi F1 ±1,5 điểm: hai model này gần như đã cân ngưỡng sẵn.

**Chi phí:** 9 run ≈ 41.000 câu, 0,21–0,25 s/câu, VRAM đỉnh 2,6 GB. CU: 25,70 → 21,03 (≈ 4,7, gồm cả smoke).

**Quyết định (27/10):** Fix A được chọn. Prototype dùng SH + τ* = +4,85 cho câu có/không (bộ nhận diện câu có/không + chuẩn hoá mẫu câu chốt ở bước 5). **Không** train lại SH (bỏ bước 3 / Fix B).

## Fix B (chỉ khi R-YB KHÔNG ĐẠT) — phác thảo, luật chi tiết chốt trước khi train

Train lại SH (cùng công thức S2 W11) có thêm câu có/không cân bằng 50/50 từ **ảnh ViVQA train** (annotation COCO). Mẫu câu khác POPE-vi, và không dùng ảnh POPE-vi lẫn ảnh calibration. Kiểm lại:
1. Không kém SH hiện tại trên ViVQA test và ViTextVQA test2000 (δ = 1,5 EM như W11).
2. POPE-vi ở τ = 0 theo cùng tiêu chí R-YB.

## Bước 4–6 gộp: backend prototype + R-PAR + R-CONF + R-YB-det (ghi trước, commit TRƯỚC notebook 23)

**Vì sao gộp:** cả ba việc đều cần chạy backend prototype trên tập test của SH. Gộp lại thì chỉ thuê L4 một lần (≈ 1,7 h, ≈ 2,7 CU).

**Thay đổi so với bản nháp (27/10, trước khi chạy, có lý do kiểm được):**
- (i) **Không dùng ViVQA val sạch để chọn ngưỡng độ tin cậy.** OCR cache W11 chỉ phủ **102/660** ảnh của `val_imgdisjoint.json` (cache dựng cho ảnh train + test; ảnh val không thuộc train). Chạy sẽ dừng ở `attach_ocr`. Thay bằng chia **nửa A / nửa B theo ảnh** trong chính tập test (seed 42): nửa A chọn ngưỡng, nửa B đánh giá.
- (ii) **Parity chạy trên toàn bộ tập test** (3.001 + 2.000 câu) thay vì 200 + 200, vì đằng nào cũng phải chạy đủ để có độ tin cậy.
- (iii) **OCR lấy từ cache** (giống eval), không chạy Paddle trong cùng runtime với torch (tránh lỗi W08). OCR chạy trực tiếp cho ảnh người dùng tải lên được kiểm riêng ở bước 7–8.

**Backend** (`src/prototype/core.py`, `prototype-core-v1`):
- Câu thường: `build_inputs` + `generate_from_inputs` = y hệt `generate_answer` của eval (cùng messages, processor, `model.generate(max_new_tokens=32, do_sample=False)`), chỉ thêm `output_scores` để lấy log-prob.
- Câu có/không: `yesno-detector-v1` (luật chữ, thiết kế **chỉ** trên ViVQA train + ViTextVQA dev, chốt trước khi nhìn test; bắt 1/10.799 câu ViVQA train, 3/5.155 câu ViTextVQA dev, 100% câu POPE t1/t2) → `normalize_yes_no` (thêm `Trả lời "có" hoặc "không".`) → margin token đầu (R-YB v3) so với **τ* = +4,85**.
- Điều kiện tiên quyết khi chạy: **chữ ký cấu hình** (model, 4-bit, prompt, OCR subset, dữ liệu, adapter) dựng lại phải **trùng hệt** run eval W11 của SH (`c12787d1…` ViVQA, `955fc63c…` ViTextVQA). Claude đã kiểm offline bằng dữ liệu thật: trùng cả hai.

**R-PAR (backend = eval):** trên ViVQA test (3.001) và ViTextVQA test2000 (2.000), câu trả lời của backend **trùng hệt** `predictions.json` của run W11 ở **≥ 99%** câu, **từng tập** → ĐẠT; ngược lại KHÔNG ĐẠT. Liệt kê các câu lệch. Smoke 100 câu ViVQA đầu tiên phải ≥ 99% thì notebook mới chạy tiếp.

**R-CONF (độ tin cậy câu thường):**
- `conf = exp(trung bình log-prob các token trả lời)`, bỏ token đặc biệt (`<|im_end|>`, …). Log-prob lấy từ `scores` sau bộ xử lý logit (phân phối greedy thật sự dùng, có repetition penalty 1,05). Không còn token nào thì conf = 0.
- Chia mỗi tập theo **ảnh** (sắp xếp tên ảnh, `random.Random(42).shuffle`, nửa đầu = A). Hai ngưỡng = **phân vị 1/3 và 2/3** của conf trên A (gộp hai tập): Thấp < q₁ ≤ Trung bình < q₂ ≤ Cao.
- Trên **B**:
  - **ĐẠT** nếu EM giảm ngặt Cao > Trung bình > Thấp, cả trên B gộp **và** từng tập;
  - **MỘT PHẦN** nếu chỉ đúng trên B gộp;
  - còn lại **KHÔNG ĐẠT**.
- Báo ECE (15 bin đều) + bảng reliability. **Giao diện chỉ hiện con số khi ECE (B gộp) ≤ 0,05**; còn lại chỉ hiện **mức** (Cao / Trung bình / Thấp), không bao giờ ghi "% đúng".

**R-YB-det (detector không làm hỏng câu thường — kế thừa yêu cầu đã ghi ở R-YB):** với các câu test bị detector bắt, thay câu trả lời greedy bằng "có"/"không" (margin > τ*). EM **toàn tập** giảm **≤ 1,5 điểm** ở từng tập → ĐẠT; ngược lại KHÔNG ĐẠT (prototype phải hỏi lại người dùng thay vì tự đi đường có/không).

**Chỉ mô tả, không có luật:**
- Độ tin cậy câu có/không = σ(|margin − τ*|): ngưỡng tertile trên calibration, tỉ lệ đúng theo mức trên POPE-vi (đọc từ run YN3).
- **Mẫu câu 3** `Có {vật} trong ảnh không? Trả lời "có" hoặc "không".` (= câu prototype gửi cho kiểu hỏi tự nhiên, file `pope_vi_questions_t3_v1.json`, sha256 `d03600fc…`): SH với τ* / τ = 0, so với SH τ* trên mẫu 1 và B1-ocr greedy.
- Chi phí: s/câu, VRAM.

## Kết quả bước 4–6 (27/10/2026): R-PAR · R-CONF · R-YB-det **ĐẠT**

Nguồn: `experiments/W13_prototype_check_L4_seed42/prototype_check_summary.md` (run `W13_PT_*` + `W13_YN3_…_popevi_t3_…`, L4). Chữ ký cấu hình trùng run W11 (`c12787d1…`, `955fc63c…`). Claude đã tính lại độc lập từ `records.json` và `predictions.json` W11: parity, conf, EM theo mức và ECE trùng summary (khác nội suy phân vị làm 1 câu đổi mức, không đổi phán quyết).

| Luật | Kết quả | Phán quyết |
|---|---|---|
| R-PAR | ViVQA 3.001/3.001 (100%), ViTextVQA test2000 2.000/2.000 (100%), 0 câu lệch; EM 73,51 / 49,50 = W11 | ĐẠT |
| R-CONF (nửa B, 2.505 câu; ngưỡng trên nửa A 2.496 câu: 0,839 / 0,958) | EM Cao > TB > Thấp: gộp 92,0 > 69,6 > 34,2 · ViVQA 94,9 > 76,6 > 42,5 · ViTextVQA 82,2 > 61,5 > 25,6 | ĐẠT |
| R-YB-det | Detector bắt 1/3.001 (ViVQA, EM −0,03) và 0/2.000 (ViTextVQA) | ĐẠT |

**Hệ quả cho prototype:**
- Backend = eval, nên mọi số đã báo cáo của SH đúng cho demo.
- ECE (B gộp) = **0,198** > 0,05 → giao diện **chỉ hiện mức** Cao / Trung bình / Thấp, không hiện con số. Model tự tin quá mức (bin conf ≈ 0,97 chỉ đúng 88%), nặng nhất ở ViTextVQA (ECE 0,308; ViVQA 0,125).
- Câu duy nhất detector bắt là bắt nhầm (câu dịch máy "điều này có một giấc ngủ ngắn trên một chiếc ghế dài chữ l không", gold "con chó"). Tập test gần như không có câu có/không, nên detector chủ yếu phục vụ câu hỏi của người dùng.

**Chỉ mô tả:**
- Độ tin cậy câu có/không σ(|margin − τ*|): đúng 98,2% (Cao, 952) / 95,6% (TB, 1.722) / 77,1% (Thấp, 2.311) trên POPE-vi; ECE 0,0505 > 0,05 → cũng chỉ hiện mức.
- Mẫu câu 3 (câu prototype gửi đi), SH τ*: F1 85,80 / 84,60 / 83,07 (random / popular / adversarial), yes-ratio 42–45%, báo sai 5,3 / 7,9 / 11,4%. Cùng SH ở τ = 0: báo sai 43,8 / 54,4 / 64,8%. So với B1-ocr greedy (mẫu 1): F1 +2,4 / +2,6 / +2,5 nhưng báo sai vẫn cao hơn 2,6 / 2,2 / 2,5 điểm → ghi vào Discussion.

**Chi phí:** 0,54 s/câu (ViVQA), 0,95 s/câu (ViTextVQA), VRAM đỉnh 2,6 GB, OCR lấy từ cache. CU: 21,03 → 18,58 (≈ 2,45).

## Bước 7 — giao diện Gradio (29/10/2026)

**Vấn đề trước khi dựng:**
1. **P4 — Paddle GPU xung đột torch** (W08: `paddlepaddle-gpu` ghi đè NCCL của torch). Giao diện cần OCR **và** Qwen trong cùng một phiên Colab.
2. **Giao diện không được làm lệch cái đã kiểm.** R-PAR chỉ đúng nếu ảnh, prompt, OCR và adapter đi đúng đường của `core.Backend`.
3. **Độ tin cậy:** ECE 0,198 (R-CONF) → luật chỉ cho hiện **mức**.

**Quyết định (có lý do kiểm được):**

| # | Quyết định | Lý do |
|---|---|---|
| D1 | **OCR chạy ở tiến trình riêng** (`src/prototype/ocr_worker.py`): tiến trình con giữ `PaddleDetVietOCREngine` nạp sẵn, nói chuyện bằng JSON từng dòng; tiến trình chính (Gradio + Qwen) **không import paddle** | Tách thư viện CUDA của Paddle khỏi torch. Tiến trình con = đúng môi trường đã dựng cache (Paddle + VietOCR, không transformers). Log của Paddle bị chuyển sang file, không lẫn kênh giao thức |
| D2 | Khởi động model thật **chỉ khi** (a) cả 3 luật bước 4–6 ĐẠT trong `prototype_check_summary.json`; (b) engine OCR của tiến trình con **trùng** engine đã dựng cache (`paddleocr`, `det_model`, `rec_model`, `vietocr_config`, `lang` — như luật notebook 19); (c) adapter trùng sha256 đã kiểm (tuỳ chọn `--expect_adapter_sha`) | Demo phải là đúng hệ thống đã đánh giá |
| D3 | **Ảnh dataset đi nguyên file** (Gradio `type="filepath"`, `image_mode=None` không mã hoá lại). Chỉ ảnh có cờ xoay EXIF hoặc cạnh dài > 2.048 px mới được xoay / thu nhỏ và lưu PNG | Giữ parity với eval; ảnh điện thoại chụp dọc không bị xoay ngang |
| D4 | Mức tin cậy Cao / Trung bình / Thấp, **ngưỡng đọc từ summary** (0,839 / 0,958 câu mở; 0,975 / 0,999 câu có/không), **không hiện con số** (cả hai ECE > 0,05). Có mục "Về mức tin cậy" giải thích vì sao | Luật R-CONF; tránh người dùng tin quá mức |
| D5 | Hiện **chữ OCR làm bằng chứng**: ảnh có khung đánh số theo thứ tự đọc, bảng các dòng, đánh dấu dòng khớp câu trả lời và dòng bị cắt (vượt 600 ký tự) | B3/SH phụ thuộc OCR (W09: bỏ OCR −4,25 EM) → người dùng cần thấy model "đọc" được gì |
| D6 | Cảnh báo: độ tin cậy thấp · không đọc được chữ · OCR bị cắt · câu có/không đi đường riêng · ảnh đã xoay / thu nhỏ. Disclaimer luôn hiện | Master §13 Responsible AI |
| D7 | **Không lưu ảnh / câu hỏi / câu trả lời.** Ảnh tạm xoá sau mỗi lượt. Log tuỳ chọn (`--log_jsonl`) chỉ ghi số đo (thời gian, mức, số dòng OCR, kích thước ảnh). Tắt analytics của Gradio | Riêng tư khi demo bằng ảnh thật |
| D8 | Chế độ **backend giả** (`--mock`, CPU) có băng cảnh báo vàng "không phải kết quả của model"; prompt vẫn dựng bằng `builder` thật | Duyệt giao diện không tốn GPU; không thể nhầm ảnh chụp giao diện với kết quả thật |

**Kiểm (offline, CPU):**
- `python -m src.prototype.ocr_worker --selftest`: tiến trình con thật (engine mock); log thư viện ghi ra cả stdout lẫn fd 1 cấp C mà vẫn không phá giao thức; ảnh lỗi → báo lỗi rồi chạy tiếp; 6 luồng gọi cùng lúc không lệch id; tiến trình tắt sau `close()`; backend sai → lỗi rõ, không treo; so engine với biên bản cache W11.
- `python -m src.prototype.app --selftest`:
  - ngưỡng và phán quyết đọc đúng từ summary; luật chưa ĐẠT → chặn model thật;
  - ảnh thường giữ nguyên từng byte; EXIF xoay 90°; ảnh 4.032 px → 2.048 px; file hỏng → lỗi dễ hiểu;
  - bằng chứng OCR;
  - phần hiển thị **không chứa con số độ tin cậy** (kiểm cả câu mở lẫn câu có/không);
  - chữ của model được escape HTML;
  - một lượt trọn vẹn với backend giả;
  - log không chứa câu hỏi / câu trả lời;
  - dựng được Gradio.
- Ảnh chụp giao diện thật (Chromium, backend giả, không lỗi console): `docs/W13/figures/ui_mock_desktop.png`, `ui_mock_yesno.png`, `ui_mock_mobile.png`.

**Chưa kiểm (để bước 8, cần L4):** OCR tiến trình con với Paddle GPU thật trong cùng phiên với Qwen; OCR trực tiếp có trùng cache không; latency; ảnh thật.

## Bước 8 — prototype chạy thật: R-OCR + R-LIVE (ghi trước, commit TRƯỚC notebook 24)

**Vấn đề:** R-PAR (bước 4–6) dùng OCR từ **cache**. Giao diện thật OCR **trực tiếp** từng ảnh ở tiến trình riêng, có thể trên môi trường Paddle khác lúc dựng cache (cache: Paddle GPU 3.3.1, phiên không có transformers; prototype: cùng phiên với Qwen). OCR đọc khác thì prompt khác → câu trả lời của demo có thể khác số trong khoá luận.

**Đo** (`src/eval/prototype_live.py`, `prototype-live-v1`, notebook 24, L4):
- **Mẫu:** mỗi tập (ViVQA test, ViTextVQA test2000) lấy 200 ảnh (sắp tên ảnh, `random.Random(42).shuffle`, lấy 200 đầu) và **mọi** câu hỏi của các ảnh đó. Danh sách câu tất định từ code + seed; sha256 danh sách `question_id` ghi vào `meta.json`.
- **Đường chạy = đường giao diện:** `make_real_backend` (kiểm luật bước 4–6, adapter trùng sha256 run W11, engine OCR trùng cache) → mỗi câu `prepare_image` → `Backend.ask` (OCR trực tiếp ở tiến trình riêng → `lines_to_texts` 600 ký tự → prompt → Qwen + adapter SH → greedy / detector có/không).
- OCR chạy GPU nếu Paddle GPU chạy chung được với torch (quy trình cài của notebook 19). Nếu không được thì chạy CPU, và ghi rõ device vào kết quả.

**R-OCR (OCR trực tiếp = cache):** theo **ảnh**, văn bản OCR đưa vào prompt (danh sách dòng sau `lines_to_texts`) của OCR trực tiếp **trùng hệt** văn bản dựng từ cache ở **≥ 98%** ảnh, **từng tập** → ĐẠT; ngược lại KHÔNG ĐẠT. Liệt kê các ảnh khác.

**R-LIVE (demo = số đã báo cáo):** trên các câu của mẫu, **bỏ** câu bị detector có/không bắt (đã đánh giá ở R-YB-det), **từng tập**:
- câu trả lời **trùng hệt** `predictions.json` của run eval W11 ở **≥ 97%** câu;
- **và** EM trên mẫu không thấp hơn EM của W11 trên cùng các câu quá **1,5 điểm** (δ như W11).

Đủ cả hai → ĐẠT; thiếu một → KHÔNG ĐẠT. Liệt kê câu lệch.

**Nếu KHÔNG ĐẠT:** không đổi số trong khoá luận. Giao diện vẫn dùng được, nhưng phần Prototype của báo cáo phải ghi rõ "OCR trực tiếp khác cache ở x% ảnh; số liệu chính thức dùng OCR cache", kèm phân tích nguyên nhân (device, phiên bản Paddle, thứ tự đọc).

**Chỉ mô tả, không có luật:**
- Latency p50 / p95 / max của OCR, model và tổng, theo tập; bỏ 5 câu đầu (khởi động CUDA).
- Thời gian khởi động: tiến trình OCR, tổng gồm nạp model.
- VRAM đỉnh tiến trình chính; VRAM mọi tiến trình (`nvidia-smi`, gồm tiến trình OCR).
- Số câu theo mức tin cậy.
- **Ảnh thật của nhóm** (15–20 ảnh, `ViVQA-VLM/data/real_photos_w13/` trên Drive + `cau_hoi.csv` theo mẫu `docs/W13/realphotos_cau_hoi_mau.csv`):
  - bảng câu hỏi / câu trả lời / mức / chế độ / số dòng OCR / thời gian, và EM nếu có đáp án;
  - đúng / sai do nhóm tự chấm.
  - Ảnh **không** đưa lên GitHub.
- Giao diện mở 20 phút qua link `*.gradio.live` để thử và quay video demo. Log chỉ ghi số đo.

**Kiểm offline:** `python -m src.eval.prototype_live --selftest`:
- mẫu tất định, lấy mọi câu của ảnh;
- chạy qua đường giao diện + resume;
- R-OCR, R-LIVE (bỏ câu có/không; W11 sai mà live đúng cũng tính vào Δ);
- 4 nhánh phán quyết;
- CSV từ Excel (BOM, dấu `;`, đáp án tách `|`, thiếu cột → lỗi rõ);
- báo cáo.

Thêm một lần chạy thử tích hợp `--run` → `--analyze` trên Drive giả (backend giả, cache giả có 1 ảnh khác, 1 câu W11 khác): phán quyết bắt đúng 90% / 95% → KHÔNG ĐẠT.

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

## File mới (bước 4–6, 27/10)

| File | Nội dung | Kiểm |
|---|---|---|
| `src/prototype/core.py` | Lõi prototype: sinh y hệt eval + log-prob, detector có/không v1, chuẩn hoá câu có/không, độ tin cậy, lớp `Backend` cho giao diện | qua selftest của `prototype_check` |
| `src/eval/prototype_check.py` | `--score` (backend trên test, kiểm chữ ký = W11 trước khi chạy) · `--parity` (smoke) · `--build_t3` · `--analyze` (R-PAR, R-CONF, R-YB-det + mô tả) | `--selftest`: detector (bắt 100% POPE t1/t2), độ tin cậy, ECE, Qwen2.5-VL tí hon (generate + output_scores = generate của eval), resume, phán quyết, chặn run smoke |
| `experiments/W13_yesno_questions/pope_vi_questions_t3_v1.json` | Mẫu câu 3 (4.985 câu) | sha256 `d03600fc5f7a1590f5ed44d45cd0a4429eba0bcecd328390d6af6a182db4ca8d` |
| `notebooks/23_prototype_parity_confidence_sh_qwen25vl.ipynb` | L4: kiểm → smoke parity 100 câu → ViVQA test → ViTextVQA test2000 → POPE mẫu 3 → phân tích → tự trả máy | Cú pháp mọi cell; mỗi code cell có markdown mô tả + kỳ vọng |
| `experiments/W13_prototype_check_L4_seed42/` + `meta.json` 4 run | Kết quả R-PAR / R-CONF / R-YB-det + mô tả | Claude tính lại độc lập: trùng |

## File mới (bước 7, 29/10)

| File | Nội dung | Kiểm |
|---|---|---|
| `src/prototype/ocr_worker.py` | OCR ở tiến trình riêng (`OCRWorkerClient`), so engine với cache (`check_engine_matches_cache`) | `--selftest` (mục Bước 7) |
| `src/prototype/app.py` | Giao diện Gradio: `load_ui_settings` (ngưỡng từ summary), `prepare_image`, `render_result`, `MockBackend`, `make_real_backend`, `Handler`, `build_demo`; CLI `--mock` / `--adapter … --share` | `--selftest` (mục Bước 7) |
| `docs/W13/figures/ui_mock_*.png` | Ảnh chụp giao diện (backend giả) | — |
| `requirements.txt` | + `gradio==6.31.0` | — |

## File mới (bước 8, 30/10)

| File | Nội dung | Kiểm |
|---|---|---|
| `src/eval/prototype_live.py` | `--run` (đường giao diện trên mẫu 200 ảnh / tập, OCR trực tiếp, resume) · `--analyze` (R-OCR, R-LIVE, latency) · `--realphotos` (ảnh thật + `cau_hoi.csv`) | `--selftest` + chạy thử tích hợp (mục Bước 8) |
| `notebooks/24_prototype_live_ocr_latency_sh_qwen25vl.ipynb` | L4: cài Qwen + Paddle cùng phiên (giữ torch) → selftest 4 script → kiểm luật / adapter / cache / engine OCR → chạy mẫu → phân tích → ảnh thật → giao diện 20 phút (`--share`) → tự trả máy | Cú pháp mọi cell; mỗi code cell có markdown mô tả + kỳ vọng |
| `docs/W13/realphotos_cau_hoi_mau.csv` | Mẫu file câu hỏi cho ảnh thật (cột `anh`, `cau_hoi`, `dap_an`) | đọc được bằng `read_questions_csv` |
| `src/prototype/ocr_worker.py` | + `FLAGS_allocator_strategy=auto_growth` cho tiến trình OCR (Paddle không giữ trước VRAM) | `--selftest` |
