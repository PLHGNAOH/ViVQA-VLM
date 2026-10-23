# R6 — Results (bản nháp tiếng Việt)

> Bản nháp **tiếng Việt** (W10, 21/10/2026; cập nhật W11, 22/10/2026). Bản tiếng Anh viết từ bản này sau khi nội dung chốt; bản chính thức nằm trên Overleaf. Mục 2, 3, 4 dùng số đã chốt (W07–W11). Mục 5, 6, 7 viết ở W12 (23/10/2026) từ error analysis tự động, POPE-vi (H4) và ví dụ định tính; 5.6 (audit tay 150 lỗi) chờ SV4.
>
> Đánh số giả thuyết theo deck cộng dồn (`docs/04_slides/cumulative/`): H1 QLoRA · H2 OCR · H3 prompt định dạng tiếng Việt · H4 hallucination; cấu hình LoRA = ablation A1 (module), A2 (rank).
>
> Mọi số truy được về `experiments/<run>/metrics.json`, `predictions.json` và file tóm tắt ghi ở từng bảng. Từ W11, nguồn tổng hợp duy nhất là `experiments/W11_full_eval_table/` (`full_eval_table.py` tính lại mọi metric từ `predictions.json` và đối chiếu với `metrics.json`).

## 1. Experimental setup recap

| Hạng mục | Thiết lập chung (mọi run, trừ khi ghi khác) |
|---|---|
| Backbone | Qwen2.5-VL-3B-Instruct, lượng tử 4-bit NF4 (double quant), tính toán bf16 |
| PEFT | QLoRA: LoRA r 16, α 32, dropout 0,05, gắn q, k, v, o của LLM (7.372.800 tham số train = 0,196%) |
| Train | lr 2e-4, batch hiệu dụng 16 (2 × 8), warmup 3%, 1 epoch (ViVQA chính thức: 2 epoch, chọn trên val sạch), gradient checkpointing |
| Ảnh | min_pixels 200.704, max_pixels 401.408 (≤ 512 token ảnh) |
| Suy luận | greedy, max_new_tokens 32 |
| OCR (run "OCR") | PaddleOCR 3.7 dò chữ + VietOCR đọc chữ, template v1, cắt 600 ký tự, min_conf 0 — chọn trên dev300 (W08), đóng băng |
| Phần cứng | Colab Pro, NVIDIA L4 (train / eval), Tesla T4 (OCR) |
| Seed | 42 (độ ổn định: 42 / 43 / 44) |
| Tập đánh giá | ViVQA test 3.001 câu; ViTextVQA test chính thức **10.028 câu** (full test, số chính) và tập con **test2000** (seed 42, dùng cho ablation) |
| Metric | EM (chính), VQA-Acc (trùng EM ở mọi run), ANLS (ngưỡng 0,5), token-F1 (phụ, để so với paper) |
| Thống kê | McNemar exact (hai phía) trên EM; CI95 bootstrap ghép cặp 10.000 lần (seed 42) cho ΔANLS, ΔEM, tương tác |

## 2. Main results

### 2.1 ViTextVQA — full test 10.028 câu

| Mô hình | Train | OCR | EM | ANLS | F1 (phụ) |
|---|---|---|---|---|---|
| B1 · zero-shot | — | không | 29,12 | 47,60 | 53,89 |
| B2 · QLoRA | 35.159 câu | không | 50,26 | 72,02 | 71,16 |
| **B3-full · QLoRA nhận biết OCR (đề xuất)** | 35.159 câu | có | **53,35** | **73,89** | **73,73** |
| ViTextBLIP-2 (Nguyen et al., *ESWA* 2025, Table 6) | 100% | SwinTextSpotter | 25,48 | — | 53,95 |

Nguồn: `W07_B1_zeroshot_…_testfull_…`, `W07_B2_eval_…_testfull_…`, `W10_B3full_eval_…_testfull_…`; so sánh: `experiments/W10_B3full_vs_B2_testfull_L4_seed42/b3full_summary.json`.

- **B2 → B3-full** (cùng dữ liệu, cùng siêu tham số, chỉ khác OCR): +3,09 EM (chỉ B2 đúng 395 / chỉ B3-full đúng 705, p = 6,7 × 10⁻²¹); ΔANLS +0,0187, CI95 [+0,0138; +0,0236].
- **So với ViTextBLIP-2:** B3-full EM 53,35 [52,39; 54,33] và F1 73,73 [73,02; 74,45] — cận dưới CI95 cao hơn số công bố ở cả hai. Paper không công bố cách chuẩn hoá đáp án nên so sánh tuyệt đối mang tính tham khảo; nhóm công khai hàm chuẩn hoá (`src/eval/metrics.py`).
- B1 → B2: QLoRA tăng +21,1 EM (H1 được ủng hộ, xem Review 2).

### 2.2 ViVQA — test 3.001 câu

| Mô hình | EM | ANLS | F1 (phụ) |
|---|---|---|---|
| B1 · zero-shot | 22,09 | 28,61 | 38,05 |
| B2 · QLoRA 1 epoch (W07) | 73,78 | 75,66 | 76,20 |
| **B2 · QLoRA 2 epoch (C3, chính thức)** | **76,64** | **78,03** | **78,84** |

Nguồn: `W07_B1_zeroshot_…_vivqa_test_…`, `W07_B2_eval_…_vivqa_test_…`, `W09_C3_eval_…_vivqa_test_…`; chọn epoch: `experiments/W09_vivqa_epoch_select_L4_seed42/`.

- Số epoch chọn trên **val sạch** (688 câu có ảnh không nằm trong train), test chạm một lần: C3 hơn bản 1 epoch +2,86 EM (65 / 151, p = 4,6 × 10⁻⁹).
- Kiểm rò rỉ (sau): mức tăng như nhau trên ảnh đã thấy và chưa thấy trong train (+2,9 / +2,9 EM), trên tập sạch +3,2 (p = 1,5 × 10⁻⁴) → không phải do nhớ ảnh.
- ViVQA không dùng OCR (ảnh đời thường, câu hỏi không xoay quanh chữ trong ảnh).

## 3. Efficiency

Nguồn: `experiments/W11_full_eval_table/full_eval_table.md` (mục Cost). Giờ train và VRAM lấy từ `train_log.json`, suy luận từ `metrics.json`. Mọi run: 1 GPU NVIDIA L4 (24 GB), 4-bit NF4, bf16.

| Run | Dataset | Câu train | Epoch | Giờ train | s/step | VRAM train đỉnh | Suy luận (s/câu) | Adapter |
|---|---|---|---|---|---|---|---|---|
| B1 · zero-shot | ViTextVQA / ViVQA | — | — | — | — | — | 0,876 / 0,510 | — |
| C1 · QLoRA | ViVQA | 10.799 | 1 | 3,03 | 16,17 | 6,62 GB | 0,571 | 29,53 MB |
| **C3 · QLoRA (chính thức)** | ViVQA | 10.799 | 2 | 6,07 | 16,20 | 6,62 GB | 0,571 | 29,53 MB |
| B2 · QLoRA | ViTextVQA | 35.159 | 1 | 11,99 | 19,63 | 6,88 GB | 1,001 | 29,53 MB |
| **B3-full · QLoRA + OCR (chính thức)** | ViTextVQA | 35.159 | 1 | 12,75 | 20,89 | 8,38 GB | 0,996 (+ OCR 0,11–0,85 s/ảnh, chạy một lần) | 29,53 MB |
| SH · adapter dùng chung | cả hai | 18.799 | 1 | 6,07 | 18,58 | 8,15 GB | 0,574 / 1,002 | 29,53 MB (**1 thay vì 2**) |

- Tham số train ở mọi run chính: 7.372.800 (0,196% của 3,76 tỷ tham số). Mọi run đều vừa một GPU 24 GB với VRAM đỉnh ≤ 8,4 GB, tức chạy được trên Colab Pro (L4) — đúng ràng buộc hiệu quả của đề tài.
- OCR làm VRAM train tăng ≈ 1,5 GB (prompt dài hơn) và s/step tăng ≈ 6–8% (B2 → B3-full, B2-sub → B3); suy luận gần như không đổi. Chi phí OCR (PaddleOCR + VietOCR) chạy một lần cho mỗi ảnh và cache lại: 0,72–0,85 s/ảnh trên T4 ở W08–W10, 0,11 s/ảnh trên T4 cho ảnh ViVQA (ít chữ) ở W11.

## 4. Ablation

### 4.1 OCR lúc train × OCR lúc test (test2000, seed 42, 8.000 câu train)

| EM | Test không OCR | Test có OCR |
|---|---|---|
| Train không OCR | 45,55 | 44,90 |
| Train có OCR | 45,00 | **49,25** |

Nguồn: `experiments/W09_ocr_factorial_2x2_test2000_L4_seed42/factorial_summary.json`.

- Chỉ thêm OCR lúc test: −0,65 EM (p = 0,49). Chỉ train cùng OCR: −0,55 (p = 0,35). Cả hai: +4,35 so với chỉ OCR lúc test (p = 2,2 × 10⁻⁸).
- Tương tác +4,90 EM, CI95 [+3,20; +6,60]; ANLS +3,62 [+2,25; +5,03].
- → Lợi ích đến từ việc **dạy model dùng dòng OCR** (OCR-aware fine-tuning), không phải chỉ "thêm OCR vào prompt". Không có OCR lúc test, model đã train cùng OCR lùi về mức B2-sub (EM ngang, p = 0,35; ANLS −0,010).

### 4.2 Lượng dữ liệu × OCR (full test 10.028 câu, seed 42)

| EM | Không OCR | Có OCR | Lợi ích OCR |
|---|---|---|---|
| 23% train (8.000 câu) | 46,56 | 49,10 | +2,54 (p = 1,1 × 10⁻¹⁴) |
| 100% train (35.159 câu) | 50,26 | 53,35 | +3,09 (p = 6,7 × 10⁻²¹) |

Nguồn: `experiments/W10_B3full_vs_B2_testfull_L4_seed42/b3full_summary.json`.

- Tương tác dữ liệu × OCR: +0,55 EM, CI95 [−0,20; +1,29] → lợi ích của OCR **không giảm** khi tăng dữ liệu (cộng gộp).
- Thêm dữ liệu: +3,70 EM khi không OCR, +4,25 khi có OCR.
- Với 23% dữ liệu, OCR-aware bù ≈ 69% khoảng cách EM do cắt dữ liệu; với 100% dữ liệu, OCR-aware vượt B2.

### 4.3 Độ ổn định theo seed (test2000, 8.000 câu train)

| Seed | EM B2-sub | EM B3 | ΔEM | p |
|---|---|---|---|---|
| 42 | 45,55 | 49,25 | +3,70 | 1,7 × 10⁻⁶ |
| 43 | 44,75 | 49,00 | +4,25 | 4,0 × 10⁻⁸ |
| 44 | 46,05 | 48,80 | +2,75 | 1,4 × 10⁻⁴ |
| mean ± std | 45,45 ± 0,66 | 49,02 ± 0,23 | **+3,57 ± 0,76** | |

Nguồn: `experiments/W09_B3_seed_stability_test2000_L4/seed_stability_summary.json`. ΔANLS +0,0155 ± 0,0040 (seed 44: cận dưới CI95 sát 0).

### 4.4 Cấu hình LoRA — A1 (module) và A2 (rank) (test2000, seed 42, công thức B3)

| Ablation | Lớp gắn LoRA (chỉ LLM) | r / α | Tham số train | EM | ΔEM so với B3 (CI95) | p | s/step | Adapter |
|---|---|---|---|---|---|---|---|---|
| — (B3) | q, k, v, o | 16 / 32 | 7.372.800 | 49,25 | — | — | 21,48 | 28,2 MB |
| A1 | q, k, v, o, gate, up, down | 4 / 8 | 7.483.392 | 48,70 | −0,55 [−1,60; +0,45] | 0,34 | 25,24 | 28,6 MB |
| A2 | q, k, v, o | 64 / 128 | 29.491.200 | 50,30 | +1,05 [−0,05; +2,15] | 0,071 | 20,85 | 112,5 MB |

Nguồn: `experiments/W10_lora_ablation_test2000_L4_seed42/lora_ablation_summary.json` (run `W10_A2_*` = A1, `W10_A3_*` = A2).

- **A1:** ở cùng ngân sách tham số (lệch 1,5%), gắn LoRA vào cả MLP không tốt hơn attention-only, train chậm hơn 17,5%. QLoRA paper (Dettmers et al., NeurIPS 2023) khuyên gắn mọi lớp tuyến tính nhưng cũng nêu yếu tố quan trọng nhất là tổng số tham số LoRA; ở đây tổng đó được giữ bằng nhau.
- **A2:** r 64 không khác biệt có ý nghĩa so với r 16; CI95 còn chứa mức tăng tới +2,15 nên kết luận **yếu**. r 64 tốn adapter ×4.
- LoRA được kiểm không gắn vào vision encoder (`lora_audit` trong `train_log.json`: 252 / 144 lớp, đều thuộc LLM).

### 4.5 Adapter dùng chung cho hai dataset — SH (W11)

Một adapter train trên ViVQA (10.799 câu, file của C1) + ViTextVQA (8.000 câu, file của B3), mọi câu dùng prompt OCR (ảnh không có chữ thì prompt trùng zero-shot), siêu tham số y hệt C1 / B3. Luật ghi trước: **không kém** nếu cận dưới CI95 của ΔEM > −1,5 trên **từng** dataset.

| Dataset | Mốc → SH (EM) | ΔEM (CI95) | McNemar p | ΔANLS (CI95) | Kết luận |
|---|---|---|---|---|---|
| ViVQA test (3.001) | C1 73,78 → 73,51 | −0,27 [−1,13; +0,60] | 0,60 | −0,21 [−0,99; +0,60] | không kém |
| ViTextVQA test2000 | B3 49,25 → 49,50 | +0,25 [−0,95; +1,45] | 0,75 | +0,73 [−0,33; +1,76] | không kém |

Nguồn: `experiments/W11_shared_adapter_L4_seed42/shared_adapter_summary.json`.

- → **Adapter dùng chung đạt**: một adapter (29,5 MB) phục vụ cả hai loại câu hỏi, không mất độ chính xác so với adapter chuyên biệt cùng ngân sách. Không có bằng chứng chuyển giao dương (cả hai p ≥ 0,05).
- **Nhiễu OCR trên ảnh thường:** 44% câu ViVQA có dòng OCR (chữ tiếng Anh trên ảnh COCO, VietOCR đọc thành chuỗi vô nghĩa, đáp án gold nằm trong OCR 0,0%). Trên nhóm này ΔEM = +0,15 [−1,05; +1,36]; tỉ lệ chép chữ OCR vào đáp án sai 0,30% (C1: 0,23%) → model học được cách **bỏ qua** dòng OCR không liên quan.
- SH thấp hơn mô hình chính thức (C3: −3,13 EM; B3-full: −4,35 EM trên cùng 2.000 câu) vì chỉ 1 epoch ViVQA và 23% dữ liệu ViTextVQA (bản rẻ, đã ghi trước). Bảng kết quả chính vẫn dùng C3 và B3-full; SH dùng cho prototype.

### 4.6 Kết luận cho giả thuyết H2 (OCR)

Cộng hưởng với QLoRA: **ủng hộ mạnh** (tương tác ở 4.1; vượt B2 cùng dữ liệu ở 4.2; ổn định 3 seed ở 4.3). Ngưỡng ANLS ≥ +3: **chưa đạt** (+1,55 ± 0,40 ở 8.000 câu; +1,87 ở 100% dữ liệu), nhưng trên ngưỡng bác bỏ +1 → **H2 ủng hộ một phần**. Lợi ích chủ yếu ở việc trả lời đúng nguyên chuỗi (EM) hơn là gần đúng ký tự (ANLS).

## 5. Error analysis

Mục này trả lời câu hỏi: **lỗi còn lại của hai mô hình chính thức (B3-full, C3) đến từ đâu?** Mọi số trong mục chỉ để mô tả, không dùng để chọn mô hình. Định nghĩa được ghi trong `docs/W12/README.md` (commit `a523f35`) **trước** khi chạy `src/eval/error_analysis.py`. Nguồn số: `experiments/W12_error_analysis/error_analysis_summary.md`.

### 5.1 EM theo loại câu hỏi — ViTextVQA full test (W11)

ViTextVQA không có nhãn loại câu hỏi. Nhóm dùng bộ phân loại theo luật từ khoá `qt-v1` (9 loại), viết chỉ từ 300 câu dev và đóng băng (commit `9b441f1`) trước khi chạy trên test. Kiểm trên 150 câu test2000 gán nhãn mù: accuracy 80,7%. Ba loại `name`, `text_content`, `other` có precision < 80% nên đánh dấu ⚠ (nhãn kém tin cậy).

| Run | source | contact | number_time | person | location | name ⚠ | text_content ⚠ | product_service | other ⚠ | Tất cả |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 661 | 641 | 1.041 | 420 | 969 | 1.971 | 777 | 1.144 | 2.404 | 10.028 |
| B1 | 24,5 | 47,9 | 41,2 | 40,7 | 25,3 | 34,9 | 27,4 | 16,5 | 21,5 | 29,1 |
| B2 | 57,2 | 55,5 | 55,7 | 54,8 | 44,0 | 55,7 | 48,6 | 42,5 | 46,1 | 50,3 |
| **B3-full** | **58,1** | **64,1** | **58,8** | **61,2** | **47,0** | **57,6** | **51,1** | **44,1** | **49,6** | **53,4** |
| Δ B2 → B3-full | +0,9 | **+8,6** | +3,1 | +6,4 | +3,0 | +1,9 | +2,5 | +1,6 | +3,5 | +3,1 |

Nguồn: `experiments/W11_question_types/question_types_table.md`, `qt_label_check.json`.

- OCR giúp nhiều nhất ở loại phải **chép đúng từng ký tự** (`contact`: số điện thoại, website, +8,6 EM; `person` +6,4), và ít nhất ở `source` (+0,9: watermark nhỏ ở góc ảnh).
- Sau B3-full, loại khó nhất là câu cần **suy ra** từ chữ (`product_service` 44,1; `location` 47,0), không chỉ chép lại chữ.

### 5.2 Kiểu lỗi: QLoRA sửa lỗi gì?

Mỗi câu sai EM được xếp vào một kiểu theo thứ tự ưu tiên (`answer_format.error_bucket`, W07):
- `tone`: chỉ khác dấu thanh;
- `truncated`: câu trả lời thiếu chữ so với gold;
- `extended`: câu trả lời thừa chữ, gold nằm bên trong;
- `near_miss`: lệch ≤ 2 ký tự;
- `other`: còn lại.

| Run | Tỉ lệ sai | tone | truncated | extended | near_miss | other |
|---|---|---|---|---|---|---|
| ViTextVQA · B1 | 70,9% | 3,3 | 5,0 | **25,7** | 8,0 | 57,9 |
| ViTextVQA · B2 | 49,7% | 9,0 | 13,2 | 13,6 | 16,8 | 47,4 |
| ViTextVQA · B3-full | 46,6% | 9,6 | 15,5 | 14,5 | 16,5 | 43,9 |
| ViVQA · B1 | 77,9% | 0,0 | 3,6 | **21,3** | 6,4 | 68,6 |
| ViVQA · C1 | 26,2% | 0,5 | 2,8 | 3,2 | 15,1 | 78,4 |
| ViVQA · C3 | 23,4% | 0,0 | 2,3 | 3,9 | 16,0 | 77,9 |

(% trên số câu sai của từng run.)

QLoRA chủ yếu sửa lỗi **trả lời dài dòng**. Lỗi `extended` giảm từ 25,7% xuống 13,6% số lỗi trên ViTextVQA, và từ 21,3% xuống 3,9% trên ViVQA. Model học được định dạng đáp án của dataset, khớp kết luận audit W07. Các lỗi còn lại dịch về phía đọc sai ký tự (`near_miss`, `tone`) và sai nội dung (`other`).

### 5.3 Nguồn lỗi của B3-full: OCR hay model?

Với mỗi câu ViTextVQA, nhóm xét văn bản OCR **đúng phần đã vào prompt** (VietOCR, cắt 600 ký tự) và xếp câu vào một trong 4 trạng thái:

| Trạng thái OCR | Định nghĩa | % câu (10.028) |
|---|---|---|
| `has_answer` | đáp án gold là một cụm từ trong OCR | 33,1 |
| `wrong_diacritics` | chỉ khớp khi bỏ dấu (OCR đọc đúng chữ, sai dấu) | 11,5 |
| `answer_missing` | OCR có chữ nhưng không chứa nguyên văn đáp án | 55,1 |
| `no_text` | không đọc ra chữ | 0,3 |

Trên 4.678 câu B3-full sai:

| Nguồn lỗi | % | Kiểu lỗi chiếm nhiều nhất |
|---|---|---|
| **`ocr_upstream`**: OCR không đưa nguyên văn đáp án vào prompt | **81,0** (trong đó sai dấu 14,6, thiếu đáp án 84,8, không có chữ 0,6) | other 1.816 · near_miss 682 · truncated 537 |
| `model_wrong_span`: OCR có đáp án, model chép **sai đoạn** | 10,5 | extended 193 · truncated 190 (**cắt sai ranh giới**) |
| `model_outside_ocr`: OCR có đáp án, câu trả lời nằm ngoài OCR | 8,5 | other 143 · extended 123 |

Đọc bảng này cần thận trọng:
- **81% là cận trên của lỗi do OCR.** `answer_missing` gồm cả câu cần suy luận (đáp án không viết nguyên văn trong ảnh) và câu có gold nhiễu. Audit tay ở 5.6 sẽ tách các trường hợp này.
- Chỉ 33,1% câu có đáp án nằm nguyên văn trong OCR, trong khi EM của B3-full là 53,35%. Nghĩa là model **tự đọc ảnh đúng** ở nhiều câu mà OCR không giúp được.
- Khi OCR có đáp án mà model vẫn sai, lỗi chủ yếu là **cắt sai ranh giới**: 383/492 câu lấy thiếu hoặc thừa từ.
- **Sai dấu từ OCR là thách thức riêng của tiếng Việt.** 11,5% câu có OCR đọc đúng chữ nhưng sai dấu, và 393/450 lỗi `tone` của B3-full nằm ở nhóm `ocr_upstream`.

Theo loại câu hỏi, `ocr_upstream` cao nhất ở `contact` (90,9%) và `text_content` (87,6%), thấp nhất ở `number_time` (72,5%).

### 5.4 OCR sửa được gì và làm hỏng gì (B2 → B3-full, ghép cặp)

| Trạng thái OCR | n | sửa được | làm hỏng | cùng đúng | cùng sai |
|---|---|---|---|---|---|
| tất cả | 10.028 | 705 | 395 | 4.645 | 4.283 |
| `has_answer` | 3.321 | **326** | 116 | 2.105 | 774 |
| `wrong_diacritics` | 1.149 | 90 | 68 | 506 | 485 |
| `answer_missing` | 5.526 | 289 | 211 | 2.023 | 3.003 |
| `no_text` | 32 | 0 | 0 | 11 | 21 |

- Hiệu số 705 − 395 = +310 câu (+3,09 EM), khớp đúng mức chênh B2 → B3-full ở mục 2.1.
- Lợi ích tập trung ở câu OCR có đáp án (sửa/hỏng = 326 : 116). Ở câu OCR không có đáp án thì gần cân bằng (289 : 211).
- **OCR làm phân tâm** ở 138 câu (34,9% câu bị làm hỏng, khoảng 1,4% tập test): B2 trả lời đúng, nhưng B3-full trả lời bằng một cụm khác lấy từ OCR.

### 5.5 ViVQA (C3): nhầm gì?

- `number` (444 câu, sai 23,0%): **toàn bộ** lỗi là đếm lệch, chủ yếu ±1 (bốn → ba 11, ba → bốn 10, một → hai 10).
- `color` (625 câu, sai 15,7%): nhầm **màu gần nhau** (nâu → đen 5, đỏ → cam 5). Có cả "đỏ → tía" 4 câu, đúng lỗi dịch máy "tía" mà data audit W07 đã gặp.
- `object` (1.247 câu, sai 23,3%): nhiều cặp **đồng nghĩa hoặc khác mức chi tiết** (máy tính → laptop 4, laptop → máy vi tính 3, hải âu → con chim 3, tủ lạnh → tủ đá 6). Nhiều khả năng là đúng ý; audit 5.6 sẽ xác nhận.
- `location` (685 câu, sai 30,7%, khó nhất): lệch mức chi tiết (phòng → phòng ngủ 6, nhà vệ sinh → phòng tắm 5).

**Ảnh trùng train–test.** 49% ảnh test ViVQA có trong train/val (W07 đã đo 46% khi chỉ tính train). C3 đạt EM 77,66 trên ảnh đã thấy và 75,67 trên ảnh chưa thấy. Mức thổi phồng do trùng ảnh vì vậy nhỏ (khoảng 2 điểm). Trên tập "sạch" (ảnh chưa thấy **và** cặp câu hỏi–đáp án chưa có trong train, 1.118 câu), C3 đạt 71,74 so với 76,64 trên toàn tập. Các tập con khác nhau về tỉ lệ loại câu hỏi, nên đây chỉ là số mô tả.

### 5.6 Audit tay 150 lỗi (100 B3-full + 50 C3)

*(Đang làm: SV4 gán mù theo bộ nhãn W07. Mẫu đã khoá ở commit `e3e6168`. Sẽ có: % đúng ý, % hallucination, "EM theo người đọc", nhãn tay đối chiếu nguồn lỗi tự động ở 5.3.)*

Tham chiếu W07 (audit 200 lỗi EM của **B2**): 54% "lỗi" thực ra đúng ý. Trên ViTextVQA, nhóm lỗi đọc chữ chiếm 59% (đọc chữ sai 28%, sai vùng 23%, sai dấu thanh 8%); gold sai/nhiễu 12,5% (ViVQA 19%).

## 6. Hallucination analysis (H4)

**Thiết kế:** POPE (Li et al., EMNLP 2023) làm bằng tiếng Việt, gọi là POPE-vi.
- Ảnh: 500 ảnh ViVQA test **chưa từng có trong train/val**, nhãn vật thể lấy từ annotation COCO 2014.
- Mỗi ảnh có 3 câu "có", cùng 3 câu "không" cho mỗi cách chọn vật vắng mặt: random, popular, adversarial.
- Mẫu câu: `Trong ảnh có {vật} không? Trả lời "có" hoặc "không".`
- Bỏ trùng còn 4.985 câu duy nhất.
- Luật, từ điển 80 lớp và file câu hỏi được khoá trên git **trước** khi chạy model (commit `9209170`, `b09aff4`). Nguồn: `experiments/W12_pope_vi_L4_seed42/pope_vi_summary.md`.

| Cấu hình | F1 random | F1 popular | F1 adversarial | yes-ratio | Không đọc được |
|---|---|---|---|---|---|
| B1 (zero-shot, prompt thường) | 84,60 | 83,17 | 81,19 | 40–43% | 0,00% |
| B1-ocr (zero-shot, prompt OCR) | 83,39 | 81,99 | 80,59 | 38–41% | 0,00% |
| C3 | — | — | — | — | **91,43%** ⚠ |
| **B3-full** | **88,89** | **85,65** | **83,06** | 48–55% | 0,00% |
| SH (adapter prototype) | 77,70 | 72,83 | 71,17 | **77–89%** | 0,16% |

**Phán quyết theo luật ghi trước** (ΔF1 so với zero-shot cùng prompt, CI95 bootstrap theo ảnh, biên −2 điểm):
- B3-full: +5,50 / +3,65 / +2,47 → **KHÔNG TĂNG HALLUCINATION**.
- C3: 91,43% câu trả lời không phải "có/không" (> 5%) → không hợp lệ → **CHƯA KẾT LUẬN**.
- SH: −5,69 / −9,16 / −9,42, cận trên CI95 < −2 ở cả 3 cách → **TĂNG HALLUCINATION** (kiểm tra an toàn cho prototype, không tính vào H4).
- **H4 theo luật: MỘT PHẦN.**

**Phân tích sau (không thuộc luật, chỉ mô tả):**
1. **F1 che mất việc B3-full báo sai nhiều hơn.** Tỉ lệ báo sai (câu "không" bị trả lời "có") của B1-ocr là 2,7 / 5,7 / 8,9%, của **B3-full là 8,7 / 16,1 / 22,4%** (random / popular / adversarial). F1 tăng vì recall tăng (73,5 → 86,9%), bù cho precision giảm. Nếu đo hallucination vật thể theo nghĩa hẹp ("nói có vật khi vật không có") thì B3-full bịa nhiều hơn zero-shot, khoảng 2,5 lần ở cách adversarial. Một phần là vì zero-shot vốn quá dè dặt (yes-ratio khoảng 40%, dưới mức 50% của thiết kế POPE). **Giới hạn phương pháp:** F1, metric chọn từ W02 theo cách báo cáo của LLaVA-1.5, không đủ để kết luận về hallucination. Cần báo kèm tỉ lệ báo sai và yes-ratio.
2. **C3 quên cách trả lời có/không.** ViVQA chỉ có 4 loại câu hỏi (vật, số, màu, vị trí), không có câu có/không. Sau QLoRA 2 epoch, C3 trả lời câu có/không bằng danh từ ("bức ảnh" 525 lần, "cái ghế" 434 lần); chỉ 372/4.985 câu là "không". Đây là một dạng quên kỹ năng tuân theo chỉ dẫn khi fine-tune trên tập đáp án hẹp.
3. **SH thiên lệch "có" mạnh:** trả lời "có" 3.773/4.985 câu; tỉ lệ báo sai 55–79%. Đếm trong dữ liệu train cho thấy cả ViTextVQA train (35.159 câu) lẫn ViVQA train (10.799 câu) **không có** đáp án "có", nên giả thuyết "học lệch từ dữ liệu có/không" **bị bác**. Nguyên nhân chưa rõ; sẽ chẩn đoán và xử lý ở W13 trước khi dùng SH cho prototype.
4. **Code-switching:** B3-full trả lời bằng tiếng Anh "no" ở 881/4.985 câu (17,7%), và gần như luôn đúng (802 câu ở câu "không").
5. Vật bị báo sai nhiều nhất ở mọi cấu hình là "cốc" (cup), rồi ghế, bát, chai. Có thể một phần do annotation COCO thiếu nhãn (giới hạn đã ghi trước).
6. Dòng OCR làm F1 zero-shot giảm nhẹ (−0,6 đến −1,2 điểm), chủ yếu qua recall.

Tham chiếu W07: trong 200 lỗi EM của B2, hallucination chiếm khoảng 7,0% [4,2; 11,4] (gán tay).

## 7. Qualitative examples

Chọn từ các ứng viên mà `error_analysis.py` rút tất định (seed 42). Ảnh sẽ chèn khi viết bản tiếng Anh; nhận định về **nội dung ảnh** cần xem ảnh xác nhận (đánh dấu †).

| # | Nhóm | question_id | Câu hỏi | Gold | B2 (không OCR) | B3-full (OCR) | Nhận xét |
|---|---|---|---|---|---|---|---|
| 1 | OCR sửa đúng (tên riêng) | test_42304 | trạm xăng này có tên là gì ? | mipec | mipeo | **mipec** | Dòng OCR giúp chép đúng từng ký tự |
| 2 | OCR sửa đúng (số) | test_13350 | bức hình được chụp vào năm nào ? | 2016 | 2017 | **2016** | Số dễ đọc nhầm khi chỉ nhìn ảnh |
| 3 | OCR sửa đúng (tên trường) | test_30959 | trường thpt này mang tên gọi là gì ? | đào duy từ | thành phố đào duy trì | **đào duy từ** | Sửa cả sai chữ lẫn thừa chữ |
| 4 | OCR làm phân tâm (sai dấu từ OCR) | test_26011 | … liên hệ với người nào ? | cô tuyết | **cô tuyết** | có tuyết | OCR đọc sai dấu ("có"), model chép theo OCR |
| 5 | Cắt sai ranh giới | test_43847 | món cơm gà hải nam làm theo nước nào ? | kiểu singapore | **kiểu singapore** | singapore | Đúng ý nhưng thiếu chữ → EM sai |
| 6 | Sai dấu thanh | test_46777 | di tích lịch sử này tên gì ? | nhà tù hòa bình | nhà từ hòa bình | nhà tu hòa bình | Cả hai model sai dấu "tù" |
| 7 | Định dạng (đúng ý) | test_26064 | máy mài góc … là sản phẩm thuộc combo nào ? | 3 | combo 3 | combo 3 | Gold chỉ ghi "3"; câu trả lời đúng ý |
| 8 | Chuỗi dài, đọc sai ký tự | test_34842 | địa chỉ email của công ty là gì ? | tnhhanphatvn@gmail . com | tnhanhphatvn@gmail . com | tnhphanhatvn@gmail . com | OCR không đọc đúng chuỗi dài; model vẫn lệch vài ký tự |
| 9 | ViVQA · câu hỏi nhiễu do dịch máy | test_1885 | đồng hồ và tranh cũ dòng những gì | tường | (B1) đồng hồ và tranh cũ dòng những bức tranh | (C3) con chim | Câu hỏi tiếng Việt khó hiểu (dịch từ COCO-QA) † |
| 10 | ViVQA · nhầm màu / gold "tía" | test_384 | màu của nho là gì | màu đỏ | (B1) Đen | (C3) màu tía | Ranh giới đỏ / tía / đen của nho mơ hồ † |
| 11 | POPE-vi · lệch định dạng | — | Trong ảnh có {vật} không? Trả lời "có" hoặc "không". | có / không | — | C3: "bức ảnh", "cái ghế" | C3 trả lời bằng danh từ (mục 6, ý 2) |
| 12 | POPE-vi · thiên lệch "có" | — | (câu "không", adversarial) | không | — | SH: "có" (78,7% câu "không") | Rủi ro cho prototype (mục 6, ý 3) |
