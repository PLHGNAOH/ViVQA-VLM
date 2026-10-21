# Week 11 — Bảng đánh giá đầy đủ + adapter dùng chung hai dataset

Index tuần 11 (tuần ảo) của ViVQA-VLM. Tuần ảo W11 bắt đầu ngày 21/10/2026, thuộc **tuần thực 7** (19–25/10). **Review 2** diễn ra **thứ Năm 29/10/2026** (tuần thực 8).

Mục tiêu W11 gồm ba phần:

1. **Bảng đánh giá đầy đủ** cho mọi run chính trên cả 2 dataset. Số liệu tính lại từ file có sẵn, không dùng GPU. Bảng này là nguồn duy nhất cho phần Results (R6) và slide.
2. **Adapter dùng chung** cho ViVQA + ViTextVQA (đã hứa trên slide Review 2). Câu hỏi cần trả lời: một adapter có phục vụ được cả hai loại câu hỏi mà không kém adapter chuyên biệt hay không?
3. **Phân rã theo loại câu hỏi** cho ViTextVQA, rồi đưa kết quả W10 vào deck cộng dồn trước ngày 29/10.

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Bản tiếng Anh cho thesis sẽ chuyển từ bản này sau khi nội dung chốt.

## Kế hoạch tuần

| # | Việc | GPU | Trạng thái |
|---|------|-----|-----------|
| 1 | Luật ghi trước (README này): adapter dùng chung S2, quy ước VQA-Acc, luật bộ phân loại câu hỏi | — | ✅ 21/10 |
| 2 | `src/eval/full_eval_table.py`: bảng đầy đủ 21 dòng × 3 bảng + chi phí + ổn định theo seed + loại câu hỏi ViVQA; tự kiểm metrics.json bằng cách tính lại từ predictions.json | — | ✅ 21/10 (`experiments/W11_full_eval_table/`) |
| 3 | OCR cho ảnh ViVQA train + test (notebook `19_ocr_cache_vivqa.ipynb`) | ≈ 2 h T4 | ⏳ đang chạy (21/10) |
| 4 | Code train **trộn 2 dataset** (`train_qlora.py --mix`) + notebook `20_qlora_shared_adapter_qwen25vl.ipynb` + script kết luận `src/eval/shared_adapter.py` | — | ✅ 21/10 |
| 5 | **SH**: train adapter dùng chung + eval ViVQA test và ViTextVQA test2000 + kết luận theo luật (notebook 20) | ≈ 6,1 h + 1,3 h L4 | ⏳ |
| 6 | Bộ phân loại câu hỏi ViTextVQA (theo luật) + bảng EM/ANLS theo loại | — | ⏳ |
| 7 | Đưa kết quả W10 (và SH nếu kịp) vào deck cộng dồn Part B qua builder | — | ⏳ |
| 8 | Tập nói Review 2 | — | ⏳ |
| 9 | Đóng tuần: README, `W11_Team_Report.md`, dọn Downloads + Drive, handoff | — | ⏳ |

## Quyết định đã chốt

| Ngày | Quyết định | Lý do |
|------|-----------|-------|
| 21/10/2026 | **Adapter dùng chung theo phương án S2**: mọi câu hỏi của cả 2 dataset dùng chế độ `ocr` (VietOCR · v1 · 600 · 0). Ảnh không có chữ thì prompt tự trùng hệt zero-shot | (1) **Khớp prototype**: lúc demo, hệ thống không biết câu hỏi "thuộc dataset nào". S1 (ViVQA không OCR) buộc phải có bộ định tuyến, hoặc đưa vào dòng OCR mà adapter chưa từng thấy với câu hỏi thường. Khi đó điểm eval không phản ánh đúng demo. (2) **Tách hiệu ứng vẫn làm được mà không tốn GPU**: với câu ViVQA có ảnh không đọc ra chữ, đầu vào của SH trùng hệt C1, nên chênh lệch ở nhóm này chỉ do trộn dữ liệu. (3) **Trả lời thêm một câu hỏi nghiên cứu**: chữ không liên quan trong ảnh thường có làm hại câu hỏi chung không. Câu này nối sang phân tích hallucination ở W12. (4) Phía ViTextVQA thì S1 = S2, nên so với B3 chỉ khác đúng việc trộn dữ liệu |
| 21/10/2026 | **VQA-Acc ≡ EM** trên cả hai dataset; vẫn báo cột VQA-Acc (metric bắt buộc) kèm chú thích | `full_eval_table.py` đếm được **0** câu có hơn 1 đáp án gold ở cả ViVQA lẫn ViTextVQA. Công thức min(#khớp/3, 1) khi đó rút về đúng/sai, nên trùng EM ở **từng câu** |
| 21/10/2026 | **H4 (QLoRA không làm tăng hallucination) dời sang W12** | Đúng mốc "W12 · Analysis" trong master. W11 dành GPU cho SH. Kết quả SH (dòng OCR trên ảnh COCO) cũng là đầu vào cho phân tích "chép nhầm chữ OCR" ở W12 |

## Luật adapter dùng chung — SH (ghi TRƯỚC khi chạy, 21/10/2026)

### Run

| Mục | Giá trị |
|---|---|
| Train | `W11_SH_qlora_ocr_vietocr_v1_r16_qwen25vl_shared_vivqa_sub8k_ep1_bf16_L4_seed42` |
| Dữ liệu train | ViVQA `data/vivqa/train.json` (10.799 câu, sha256 `892243fa732f…`, **cùng file với C1**) **+** ViTextVQA `train_subset8000_byimage_seed42.json` (8.000 câu, sha256 `093ab061edbb…`, **cùng file với B3**) = **18.799 câu**. Ghép hai danh sách rồi để bộ lấy mẫu ngẫu nhiên của Trainer (seed 42) **xáo lẫn** hai dataset. **Không** tăng hay giảm tỉ lệ mẫu (giữ tỉ lệ tự nhiên 57 : 43) |
| Prompt | Chế độ `ocr` cho **mọi** câu. VietOCR (`paddledet_vietocr`) · template v1 · cắt 600 ký tự · min_conf 0. Mỗi câu tra OCR trong cache **của dataset mình**: ViVQA = cache mới W11 (biên bản `experiments/W11_ocr_cache_vivqa/`), ViTextVQA = cache W10 (sha256 `e7ce4f76d98c…`) |
| Siêu tham số | **Y hệt B3 và C1**: LoRA r 16, α 32, q/k/v/o (chỉ LLM), dropout 0,05; lr 2e-4; 1 epoch; batch 2 × tích luỹ 8 = 16; warmup 3%; weight decay 0; bf16; 4-bit NF4 + double quant; min/max_pixels 200.704 / 401.408; seed 42; L4 |
| Eval | `W11_SH_eval_…_vivqa_test_…` (3.001 câu) và `W11_SH_eval_…_vitextvqa_test2000_…` (2.000 câu). Chế độ `ocr`, greedy, max_new_tokens 32, giống eval C1 / B3 |

### Điều kiện hợp lệ (kiểm trước khi eval; sai → không dùng run, báo Claude)

- `train_log.json`: `num_samples` = 18.799, tách theo dataset **10.799 / 8.000**, `num_dropped` = 0.
- `lora_audit.by_tower` = `{'llm': 144}`, không có `vision`; tham số train = **7.372.800**.
- Dấu vân tay OCR của phần ViTextVQA (`subset_sha256`) = **`c2d7cb1f75b8…`**, trùng `train_log` của B3. Điều này đảm bảo phần ViTextVQA nhận **đúng** chuỗi OCR mà B3 đã nhận.
- Cache ViVQA trên Drive trùng sha256 trong biên bản W11. Engine OCR (paddleocr, det_model, rec_model, vietocr_config, lang) trùng cache ViTextVQA.

### So sánh chính (ghép cặp trên cùng câu hỏi)

| Dataset | Mốc (cùng điều kiện) | Vì sao chọn mốc này |
|---|---|---|
| ViVQA test (3.001) | **C1** (QLoRA ViVQA, 1 epoch, W07): EM 73,78 | Cùng file train ViVQA, cùng 1 lần duyệt dữ liệu ViVQA, cùng siêu tham số |
| ViTextVQA test2000 | **B3** (QLoRA + OCR, tập con 8k, seed 42, W08): EM 49,25 | Cùng tập con 8k, cùng OCR, cùng siêu tham số |

Thống kê: ΔEM = SH − mốc, kèm **CI95 bootstrap ghép cặp** (10.000 lần, seed 42) và **McNemar exact** trên EM. ΔANLS kèm CI95.

**Biên "không kém" δ = 1,5 điểm EM**, áp cho **từng** dataset:

| Điều kiện | Kết luận cho dataset đó |
|---|---|
| Cận dưới CI95 của ΔEM > −1,5 | **KHÔNG KÉM** |
| … và thêm ΔEM > 0, p < 0,05 | **KHÔNG KÉM — TỐT HƠN** (chuyển giao dương giữa hai loại câu hỏi) |
| Cận trên CI95 của ΔEM < −1,5 | **KÉM** |
| Còn lại | **CHƯA KẾT LUẬN** (CI95 chứa −1,5) |

**Kết luận chung:**
- Cả hai dataset **KHÔNG KÉM** → **ADAPTER DÙNG CHUNG ĐẠT**. Prototype dùng **1 adapter (SH)**. Số chính thức trong bảng Results vẫn là C3 (ViVQA) và B3-full (ViTextVQA); khoảng cách giữa SH và hai mốc này báo cáo minh bạch.
- Còn lại → **ADAPTER DÙNG CHUNG KHÔNG ĐẠT**. Prototype dùng **2 adapter chuyên biệt** (C3, B3-full) và chọn adapter theo ảnh (PEFT hỗ trợ nạp nhiều adapter). Cách chọn adapter quyết định ở tuần prototype.

**Vì sao δ = 1,5 điểm:** con số này bằng khoảng một nửa mức lợi mà OCR-aware mang lại khi đủ dữ liệu (+3,09 EM, W10) và nhỏ hơn mức lợi C1 → C3 của epoch thứ hai (+2,87). Kém hơn mức này nghĩa là xoá một phần đáng kể đóng góp của đề tài. Độ chính xác cũng đủ: nửa độ rộng CI95 của ΔEM ghép cặp là khoảng **0,95** điểm trên ViVQA (C1 → C3) và khoảng **1,0** điểm trên test2000 (ablation A1 W10). Vì vậy, nếu thật sự không có chênh lệch, cận dưới vẫn nằm trên −1,5.

### Chỉ báo cáo (không có luật kết luận)

1. **Tách hiệu ứng trên ViVQA test:** chia câu theo việc prompt S2 có dòng OCR hay không.
   - Nhóm **không có dòng OCR**: đầu vào SH trùng hệt C1, nên ΔEM ở nhóm này chỉ do **trộn dữ liệu**.
   - Nhóm **có dòng OCR**: ΔEM là tác động gộp của **trộn dữ liệu + dòng OCR**.
   - Mỗi nhóm báo ΔEM kèm CI95 và số câu.
2. **So với mô hình chính thức** C3 (ViVQA, 2 epoch) và B3-full (ViTextVQA, toàn bộ train). Điều kiện **không** giống nhau (khác số epoch, khác lượng dữ liệu), nên chỉ mô tả, không kết luận.
3. **Chi phí:** 1 adapter (≈ 29,5 MB) so với 2; giờ train SH so với tổng C1 + B3; s/câu.
4. **Chẩn đoán chép OCR trên ViVQA:** tỉ lệ dự đoán sai (EM = 0) mà là chuỗi con của văn bản OCR. Đây là số đầu vào cho phân tích hallucination ở W12.

### Không được làm

- Không chạy lại với tỉ lệ trộn, số epoch hay seed khác sau khi thấy kết quả để đổi kết luận.
- Không đổi mốc so sánh (vd so SH với C3) để "đạt".

### Giới hạn đã biết

- 1 seed (42). Độ lệch chuẩn EM của B3 qua 3 seed là 0,23 điểm (W09), nhỏ hơn nhiều so với δ.
- Lịch learning rate dài hơn (≈ 1.175 step so với 675 của C1 và 500 của B3). Điều này đi kèm việc trộn dữ liệu, không tách riêng được.
- Cache OCR ViVQA tạo ở W11 (cùng engine với ViTextVQA, đã kiểm ở notebook 19).

## Luật bộ phân loại câu hỏi ViTextVQA (ghi trước, 21/10/2026)

ViTextVQA không có nhãn loại câu hỏi (`question_type` = `unknown` ở mọi run), trong khi master yêu cầu error analysis theo loại câu hỏi.

1. Bộ phân loại là **luật từ khoá** (không dùng model). Luật được thiết kế **chỉ trên câu hỏi dev**, không nhìn câu hỏi test và không nhìn dự đoán của model.
2. Luật được **commit (đóng băng) trước** khi tính bất kỳ số nào theo loại trên test.
3. Kiểm độ chính xác: Hoàng tự gán nhãn tay **150 câu** test2000 (chọn ngẫu nhiên, seed 42) mà **không xem** dự đoán. Báo độ chính xác tổng và theo loại. Loại nào có độ chính xác < 80% thì gắn cờ "nhãn kém tin cậy" trong bảng.
4. Loại có < 50 câu trên test2000 thì gộp vào "khác".
5. Bảng theo loại chỉ để **mô tả / error analysis**, không dùng để chọn mô hình.

## Thay đổi code

| File | Thay đổi | Kiểm |
|------|----------|------|
| `src/eval/full_eval_table.py` (mới) | Danh sách 21 run chính (3 bảng: ViTextVQA full test, test2000, ViVQA test) → EM / VQA-Acc / ANLS / F1, ΔEM so với B1, chi phí lần ngược từ adapter → `train_log.json`, ổn định theo seed (B2-sub, B3), EM theo loại câu hỏi (ViVQA), kiểm quy ước VQA-Acc. **Tính lại mọi metric từ predictions.json**, dừng nếu lệch metrics.json (> 1e-9), nếu adapter không thuộc run train khai báo, hoặc nếu các run trong cùng bảng khác tập câu hỏi | `--selftest` (bắt được metrics.json bị sửa và adapter sai run). Chạy thật: 21/21 run khớp metrics.json; số khớp README W07–W10 (vd B3-full 53,35 / 73,89; B3 test2000 49,25; C3 76,64) |
| `notebooks/19_ocr_cache_vivqa.ipynb` (mới) | OCR VietOCR cho ảnh ViVQA train + test (T4). Bắt buộc engine trùng cache ViTextVQA W10. Thống kê % câu có dòng OCR, đáp án∈OCR; biên bản `experiments/W11_ocr_cache_vivqa/` | Cú pháp các cell đã kiểm; dùng lại `src/data/ocr.py` (đã hỗ trợ `--dataset vivqa`) |
| `src/adaptation/train_qlora.py` | `--mix DATASET=FILE` (lặp lại, ≥ 2 dataset) thay `--dataset/--data_file`. Mỗi phần nạp, lọc và gắn OCR **bằng đúng hàm của run đơn**, với OCR cache mặc định của dataset đó. Thiết lập OCR (engine, template, ngưỡng) **phải trùng** giữa các phần, nếu không thì dừng. Ghép rồi xáo bằng `random.Random(seed)` (Trainer xáo thêm lần nữa). Chữ ký run gồm sha256 từng file + dấu vân tay OCR từng phần. `train_log.json` có thêm `data_parts`, `ocr.stats_by_dataset`, `mix_order`. Run đơn **không đổi** chữ ký | `--selftest` thêm `mix (W11) OK`: phần ViTextVQA trong run trộn có **cùng dấu vân tay OCR** như run đơn (B3); ảnh không chữ cho prompt **trùng zero-shot**; chạy lại cùng lệnh thì cùng chữ ký (resume được); engine OCR lệch thì dừng; xáo tái lập được, trộn đều. Các selftest cũ vẫn qua |
| `src/eval/shared_adapter.py` (mới) | Luật SH: kiểm hợp lệ train_log (file = C1 + B3, siêu tham số = C1 = B3, LoRA 144 lớp LLM, 7.372.800 tham số, OCR ViTextVQA = dấu vân tay B3) và 2 run eval (cùng file test, cùng OCR với B3 trên test2000, adapter đúng run SH). Sau đó so sánh C1 → SH, B3 → SH (ΔEM + CI95, McNemar, ΔANLS) và kết luận theo δ = 1,5. Chỉ báo cáo: tách ViVQA theo dòng OCR, tỉ lệ chép chữ OCR, so với C3 / B3-full trên cùng câu, chi phí | `--selftest`. Chạy thử trên predictions thật (SH tạm thay bằng run có sẵn): B3 → A3 = +1,05 [−0,05; +2,15], p = 0,071, **khớp** README W10; C1 → C3 = +2,87, **khớp** W09 |
| `notebooks/20_qlora_shared_adapter_qwen25vl.ipynb` (mới) | Kiểm trước (sha256 2 file train = C1 / B3, 2 OCR cache = biên bản W11 / W10, phủ đủ ảnh, run mốc, selftest 5 script) → train SH → kiểm hợp lệ → eval ViVQA test → eval test2000 → kết luận. Bỏ qua phần đã xong, tự trả máy L4 | Cú pháp các cell đã kiểm |

## Ngân sách

| Việc | L4 | T4 | CU (ước tính; L4 ≈ 1,54 CU/h, T4 ≈ 1,2 CU/h) |
|---|---|---|---|
| OCR ViVQA (≈ 9–10 nghìn ảnh × ≈ 0,75 s) | — | ≈ 2 h | ≈ 2,5 |
| SH train (≈ C1 3,03 h + B3 2,98 h) | ≈ 6,1 h | — | ≈ 9,4 |
| SH eval ViVQA test + ViTextVQA test2000 (+ nạp model) | ≈ 1,3 h | — | ≈ 2 |
| **Tổng W11** | ≈ 7,4 h | ≈ 2 h | **≈ 14** (≈ 49 CU lúc 21/10 04:10 → còn ≈ 35 cho W12–W15) |

## Kết quả

### Bảng đánh giá đầy đủ (21/10/2026, không GPU)

Nguồn: `../../experiments/W11_full_eval_table/` (`full_eval_table.md` / `.csv` / `.json`). Bảng thesis lấy số **từ file này**, không chép tay từ README các tuần trước.

| Dataset · tập test | Zero-shot B1 | QLoRA không OCR | **Đề xuất** | ΔEM so với B1 |
|---|---|---|---|---|
| ViTextVQA · full test (10.028) | EM 29,12 · ANLS 47,60 | B2: 50,26 · 72,02 | **B3-full: 53,35 · 73,89** | +24,23 |
| ViVQA · test (3.001) | EM 22,09 · ANLS 28,61 | C1: 73,78 · 75,66 | **C3: 76,64 · 78,03** | +54,55 |

EM ViVQA theo loại câu hỏi (nhãn có sẵn của dataset). Đây là số mô tả, phần giải thích để dành cho error analysis W12:

| Run | object (1.247) | number (444) | color (625) | location (685) |
|---|---|---|---|---|
| B1 | 5,69 | 69,14 | 44,80 | 0,73 |
| C1 | 73,54 | 77,03 | 83,04 | 63,65 |
| C3 | 76,66 | 77,03 | 84,32 | 69,34 |

### Adapter dùng chung — SH

Run: `W11_SH_qlora_ocr_vietocr_v1_r16_qwen25vl_shared_vivqa_sub8k_ep1_bf16_L4_seed42` (train), `W11_SH_eval_qlora_ocr_vietocr_v1_r16_shared_ep1_{vivqa_test,vitextvqa_test2000}_bf16_L4_seed42` (eval), `W11_shared_adapter_L4_seed42/` (kết luận).

*(chưa chạy)*
