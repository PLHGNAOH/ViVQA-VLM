# W10 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W10 — Ablation cấu hình LoRA + B3 trên toàn bộ dữ liệu train |
| Thời gian thực | 19–21/10/2026 (tuần thực 7) |
| Mốc kế tiếp | **Review 2 — thứ Năm 29/10/2026** (tuần thực 8; bộ môn dời lịch, báo ngày 20/10) |
| Phần cứng | Colab Pro: NVIDIA L4 (train/eval) · Tesla T4 (OCR) |
| Commit cuối tuần | commit đóng tuần sau `0cf9829` (xem mục "Dấu vết commit") |

## Mục tiêu

Sau W09, phương pháp đề xuất — QLoRA nhận biết OCR (B3) — còn hai câu hỏi hội đồng chắc chắn hỏi:

1. **Cấu hình LoRA (r = 16, chỉ 4 lớp attention) chọn từ W04 có hợp lý không?** QLoRA paper khuyên gắn mọi lớp tuyến tính; chưa kiểm rank. → ablation A1 (module) và A2 (rank).
2. **Đủ dữ liệu thì OCR-aware có vượt B2 không?** W09: B3 (23% dữ liệu) thấp hơn B2 (100%) 1,16 EM trên full test → đóng góp mới chỉ là "hiệu quả dữ liệu". → B3-full.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | `src/adaptation/lora_targets.py`: preset `attn` / `all_linear_llm` (regex chỉ LLM), audit LoRA theo khối + đếm tham số; `train_qlora.py` thêm `--lora_r / --lora_alpha / --lora_targets` + `lora_audit` trong `train_log.json`; `lora_ablation.py`; notebook 17; luật ghi trước | ✅ `8be932e` — selftest bắt được bẫy: list hậu tố MLP sẽ gắn LoRA vào 96 lớp của **vision encoder** |
| 2 | Ablation A1 — all-linear r4 (cùng ngân sách tham số với B3) | ✅ **không tốt hơn** attention-only (−0,55 EM, p = 0,34), **chậm hơn 17,5%** |
| 3 | Ablation A2 — attention r64 (×4 tham số) | ✅ **không khác biệt có ý nghĩa** (+1,05 EM, p = 0,071; CI95 [−0,05; +2,15] → kết luận yếu), adapter ×4 |
| 4 | Notebook 10 Phần D: OCR thêm 9.063 ảnh train | ✅ 0 lỗi, 0,72 s/ảnh; 6.289 ảnh cũ giữ nguyên dấu vân tay |
| 5 | `src/eval/b3full.py` + notebook 18: B3-full (công thức B3 trên 35.159 câu) + full test | ✅ **OCR-AWARE VƯỢT B2 KHI ĐỦ DỮ LIỆU**: +3,09 EM, p = 6,7 × 10⁻²¹ |
| 6 | R6 Results bản nháp tiếng Việt (mục 2–4) | ✅ `docs/03_report/R6_Results.md` |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| Dataset / OCR | OCR toàn bộ train + bảo vệ cache cũ (4) |
| PEFT / Training | Preset + audit LoRA (1); ablation A1, A2 (2, 3); B3-full (5) |
| Evaluation | Luật A1/A2 + CI95 ΔEM (1–3); luật B3-full + bảng 2×2 dữ liệu × OCR (5); R6 (6) |
| VLM / Baseline | Mô hình ViTextVQA chính thức đổi sang B3-full (5) |

### Definition of Done

- [x] Ablation module gắn LoRA ở **cùng ngân sách tham số**, LoRA không lọt vision (audit trong log)
- [x] Ablation rank, α/r giữ nguyên
- [x] OCR cache phủ toàn bộ train, phần cũ không đổi
- [x] B3-full so với B2 cùng dữ liệu, cùng siêu tham số — luật ghi trước, ghép cặp trên full test
- [x] Mọi số liệu truy được về file trong `experiments/`; mọi luật kết luận commit **trước** khi có số
- [x] R6 bản nháp tiếng Việt cho phần OCR-aware
- [x] Dọn Downloads + Drive

## Kết quả / số liệu

### 1. Ablation cấu hình LoRA (test2000, seed 42, công thức B3)

| Ablation | Run | Cấu hình | Tham số | EM | ΔEM (CI95) | p | s/step | Adapter |
|---|---|---|---|---|---|---|---|---|
| — | B3 (mốc) | attention · r16 | 7,37 tr | 0,4925 | — | — | 21,48 | 28,2 MB |
| **A1** module | `W10_A2_*` | all-linear · r4 | 7,48 tr | 0,4870 | −0,55 [−1,60; +0,45] | 0,34 | 25,24 | 28,6 MB |
| **A2** rank | `W10_A3_*` | attention · r64 | 29,49 tr | 0,5030 | +1,05 [−0,05; +2,15] | 0,071 | 20,85 | 112,5 MB |

→ Giữ **attention-only r16**: rẻ nhất, không kém có ý nghĩa. r64 có xu hướng tăng nhưng chưa đủ bằng chứng (1 seed, test2000) → hướng cải tiến tiếp.

### 2. B3-full — dữ liệu × OCR (full test 10.028 câu, seed 42)

| EM | Không OCR | Có OCR (OCR-aware) | Lợi ích OCR |
|---|---|---|---|
| 23% train | 46,56 (B2-sub) | 49,10 (B3) | +2,54 (p = 1,1 × 10⁻¹⁴) |
| 100% train | 50,26 (B2) | **53,35 (B3-full)** | **+3,09 (p = 6,7 × 10⁻²¹)** |

- ANLS B2 → B3-full: 72,02 → 73,89 (+0,0187, CI95 [+0,0138; +0,0236]); F1 71,16 → 73,73.
- Tương tác dữ liệu × OCR: +0,55 EM, CI95 [−0,20; +1,29] → lợi ích OCR **không giảm** khi thêm dữ liệu.
- So với ViTextBLIP-2 (EM 25,48 · F1 53,95): B3-full **cao hơn** cả hai (cận dưới CI95: 52,39 / 73,02).
- Chi phí: train 12,75 h so với 11,99 h (+6%), VRAM 8,38 so với 6,88 GB, suy luận ≈ 1 s/câu như nhau (chưa tính OCR).

### 3. Đối chiếu giả thuyết (đánh số chính thức từ deck cộng dồn Review 2)

| Mục | Trạng thái sau W10 | Lý do |
|---|---|---|
| H2 · OCR tăng ANLS ≥ +3, cộng hưởng với QLoRA | **Ủng hộ một phần** (không đổi) | Cộng hưởng mạnh và nay vượt cả B2 cùng dữ liệu (+3,09 EM); nhưng ΔANLS +1,87 < +3 |
| Ablation A1 · module gắn LoRA | Attention-only giữ nguyên | All-linear không tốt hơn ở cùng ngân sách, chậm hơn 17,5% |
| Ablation A2 · rank | r16 đủ (kết luận yếu) | r64 +1,05 EM, p = 0,071; CI95 chạm +2,15 |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|---|---|---|
| B1 | Kết quả W10 (B3-full, A1/A2) chưa có trong deck cộng dồn Review 2 | Cao (review 29/10) | Tuần thực 8, trước 29/10: thêm vào Phần B của `docs/04_slides/cumulative/build_cumulative_deck.js` (không sửa pptx tay) |
| B2 | `docs/W02/Hypotheses.md` còn đánh số cũ H1–H8 | Thấp | Cập nhật theo đánh số mới (H1–H4, A1/A2) khi viết R4 Methodology |
| B3 | B3-full mới 1 seed | Thấp | W09: std 0,23 EM ở 8k, 3/3 seed ổn định; ghi giới hạn. Chạy thêm seed chỉ khi reviewer yêu cầu (≈ 16 h L4 / seed) |
| B4 | A2 (rank) kết luận yếu | Thấp | Ghi hướng cải tiến tiếp; không chạy thêm để "câu" kết luận |
| B5 | H4 (hallucination) còn chờ | Trung bình | W12: đo kiểu POPE trên B1 / B2 / B3-full |
| B6 | Ngân sách Colab | Trung bình | Kiểm số CU thật khi đóng tuần; adapter dùng chung (W11) ≈ 15–19 CU |

## Kế hoạch tuần sau (W11)

1. **Bảng đánh giá đầy đủ** (EM, VQA-Acc, ANLS, F1 phụ; thời gian train/suy luận; VRAM) cho mọi run chính của cả hai dataset — từ file đã có, không tốn GPU.
2. **Adapter dùng chung 2 dataset** (bản rẻ): thiết kế + luật ghi trước, OCR ảnh ViVQA nếu cần, train + eval.
3. Bắt đầu chuẩn bị nội dung W10 cho deck cộng dồn Review 2 (thực hiện trước 29/10).

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `8be932e` | Bước 0: preset + audit LoRA, `--lora_*`, `lora_ablation.py`, notebook 17, luật A1/A2 (số cũ H3/H4) + luật B3-full ghi trước |
| `530e957` | Kết quả ablation A1/A2; notebook 10 Phần D, notebook 18, `b3full.py` |
| `6359085` | Deck cộng dồn Review 2 (phiên riêng): Phần A = Review 1 đã sửa, Phần B = Review 2; đánh số giả thuyết mới |
| `0cf9829` | Dọn file ghi chép một lần trong `docs/` |
| *(commit đóng tuần)* | Biên bản OCR toàn bộ train, kết quả B3-full, README + Team Report W10, R6 bản nháp |

## Link experiment log

- Ablation LoRA: `experiments/W10_lora_ablation_test2000_L4_seed42/` + run `W10_A2_*`, `W10_A3_*`
- OCR toàn bộ train: `experiments/W10_ocr_cache_fulltrain/`
- B3-full: `experiments/W10_B3full_vs_B2_testfull_L4_seed42/` + run `W10_B3full_*`
- Adapter (không lên git, trên Drive): `adapters/W10_*`
