# R6 — Results (bản nháp tiếng Việt)

> Bản nháp **tiếng Việt** (W10, 21/10/2026). Bản tiếng Anh viết từ bản này sau khi nội dung chốt; bản chính thức nằm trên Overleaf. Mục 2 và 4 dùng số đã chốt (W07–W10). Mục 3, 5, 6, 7 chờ W11–W12.
>
> Đánh số giả thuyết theo deck cộng dồn Review 2 (`docs/04_slides/cumulative/`): H1 QLoRA · H2 OCR · H3 prompt định dạng tiếng Việt · H4 hallucination; cấu hình LoRA = ablation A1 (module), A2 (rank).
>
> Mọi số truy được về `experiments/<run>/metrics.json`, `predictions.json` và file tóm tắt ghi ở từng bảng.

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

*(W11: bảng đầy đủ thời gian train / suy luận / VRAM mọi run.)* Số đã có cho phương pháp đề xuất:

| Run | Giờ train (L4) | s/step | VRAM đỉnh | Suy luận (s/câu) | Adapter |
|---|---|---|---|---|---|
| B2 · ViTextVQA | 11,99 | 19,63 | 6,88 GB | 1,001 | cùng cỡ (r16 · attention) |
| B3-full · ViTextVQA | 12,75 | 20,89 | 8,38 GB | 0,996 (+ OCR 0,72–0,85 s/ảnh, chạy một lần) | 29,53 MB |

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

### 4.5 Kết luận cho giả thuyết H2 (OCR)

Cộng hưởng với QLoRA: **ủng hộ mạnh** (tương tác ở 4.1; vượt B2 cùng dữ liệu ở 4.2; ổn định 3 seed ở 4.3). Ngưỡng ANLS ≥ +3: **chưa đạt** (+1,55 ± 0,40 ở 8.000 câu; +1,87 ở 100% dữ liệu), nhưng trên ngưỡng bác bỏ +1 → **H2 ủng hộ một phần**. Lợi ích chủ yếu ở việc trả lời đúng nguyên chuỗi (EM) hơn là gần đúng ký tự (ANLS).

## 5. Error analysis

*(W12 — phân tích lỗi B3-full theo loại câu hỏi và kiểu lỗi.)* Đã có: audit 200 lỗi EM của B2 (W07, `docs/W07/W07_Team_Report.md`) — 54% "lỗi" thực ra đúng ý; trên ViTextVQA, nhóm lỗi đọc chữ chiếm 59% (đọc chữ sai 28%, sai vùng 23%, sai dấu thanh 8%) → căn cứ chọn OCR; gold sai / nhiễu 12,5% (ViVQA 19%).

## 6. Hallucination analysis

*(W12 — đo kiểu POPE trên B1 / B2 / B3-full, kiểm H4.)* Đã có: hallucination ≈ 7,0% [4,2; 11,4] trong 200 lỗi EM của B2 (W07).

## 7. Qualitative examples

*(W12 — ví dụ OCR sửa đúng / OCR làm sai / model chép lỗi OCR.)*
