# W07 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W07 — Review 2 (baseline đã tái lập + cải tiến đầu tiên) |
| Kết thúc | 2026-10-12 |
| Mốc kế tiếp | **Review 2** — tuần thực 7 (19–25/10/2026); ngày cụ thể bộ môn báo qua email trong tuần đó |
| Phần cứng dùng trong tuần | NVIDIA L4 22 GB (Colab Pro), bf16 native |
| Commit cuối tuần | `2f19ca4` (xem mục "Dấu vết commit") |

## Mục tiêu

Theo yêu cầu của GVHD Phạm Minh Trí (họp 07/10/2026): **chưa bàn cải tiến khi chưa có baseline rõ ràng**. Cụ thể tuần này phải:

1. Có sơ đồ kiến trúc baseline và hiểu từng thành phần; giải thích dataset, nhiễu và cách xử lý.
2. Chốt cấu hình trên L4 rồi chạy lại baseline **một lần** với giao thức đánh giá nêu rõ (B1).
3. Chứng minh pipeline đánh giá đáng tin bằng cách tái lập một con số đã công bố.
4. Chạy cải tiến đầu tiên (QLoRA, B2), so với B1 dưới điều kiện y hệt, có kiểm định thống kê.
5. Error analysis + hallucination analysis trên lỗi của B2.
6. Slide Review 2 (tiếng Anh, trình bày tiếng Việt).

Baseline **Qwen2.5-VL-3B-Instruct** được GVHD xác nhận ngày 07/10/2026.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | Chốt cấu hình trên L4: dtype probe fp16 / fp32 / bf16 (73 câu) + so 512 vs 1.280 token khi train | ✅ **bf16** (0 câu hỏng) · **512 token** (1.280 train chậm 1,91× > ngưỡng 1,5×) |
| 2 | Checkpoint / resume cho huấn luyện + chữ ký run (sha256 dữ liệu + cấu hình) | ✅ resume thử trên L4; chạy tiếp với cấu hình khác bị chặn |
| 3 | **B1** — baseline zero-shot chính thức trên L4 | ✅ 0 câu hỏng trên cả hai dataset |
| 4 | **R0** — tái lập TextVQA val của Qwen2.5-VL-3B (kiểm chứng pipeline) | ✅ 79,60 so với 79,3 công bố (ngưỡng ±2,0) |
| 5 | **B2** — QLoRA 1 epoch trên ViVQA và ViTextVQA | ✅ EM tăng mạnh trên cả hai, McNemar p < 0,05 |
| 6 | Kiểm tra rò rỉ train/dev → test + tách kết quả ViVQA theo mức trùng | ✅ mức tăng của B2 không do rò rỉ |
| 7 | Full test ViTextVQA (10.028 câu) cho B1 và B2, so với SOTA của paper | ✅ B2 vượt ViTextBLIP-2 ở cả EM và F1 (theo CI95) |
| 8 | **R1** — Qwen2-VL-7B zero-shot (đối chiếu Table 7 paper ViTextVQA) | ✅ 7B kém 3B; nguyên nhân chính là trả lời dài |
| 9 | **FS5** — đối chứng few-shot 5 ví dụ | ✅ few-shot không sửa được định dạng; B2 vượt FS5 → QLoRA học hơn định dạng |
| 10 | Error audit 200 lỗi EM của B2 (4 người gán nhãn + nhãn gợi ý AI, có phân xử) | ✅ xem mục "Error audit" |
| 11 | 3 sơ đồ theo yêu cầu GVHD (quy trình + nhiễu ViVQA, kiến trúc baseline, giao thức đánh giá) | ✅ `figures/` (bản EN + TV) |
| 12 | Slide Review 2 EN + TV (16 slide) + kịch bản thuyết trình | ✅ |
| 13 | Đưa báo cáo Review 1 (`docs/W03/index_TV/TA`) về đúng nội dung mốc Review 1 | ✅ tag `review1-report`; phần W4–W6 chuyển sang Review 2 |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| VLM / Baseline | Chốt cấu hình (1), B1 (3), R0 (4), R1 (8), sơ đồ kiến trúc |
| Dataset / OCR | Kiểm tra rò rỉ (6), sơ đồ quy trình + nhiễu ViVQA, số mẫu ViTextVQA chính thức |
| PEFT / Training | Checkpoint/resume (2), B2 (5), đếm tham số train |
| Evaluation | Full test (7), FS5 (9), error audit (10), `paper_consistency.py`, `answer_format.py`, kiểm định McNemar + bootstrap |

### Definition of Done

- [x] Sơ đồ kiến trúc baseline + sơ đồ dataset/nhiễu + sơ đồ giao thức đánh giá (yêu cầu GVHD 07/10)
- [x] Cấu hình chốt bằng luật đặt trước, ghi vào `configs/qwen_lora.yaml`
- [x] Baseline B1 chạy một lần trên cấu hình chốt, 0 câu hỏng
- [x] Pipeline tái lập được số công bố (R0)
- [x] Cải tiến B2 so với B1 cùng điều kiện, có McNemar + CI95 bootstrap
- [x] Error + hallucination analysis có độ tin cậy giữa người gán nhãn (κ)
- [x] Slide Review 2 EN + kịch bản TV
- [ ] `data_audit` 200 nhãn train (đo nhiễu nhãn để làm ablation làm sạch dữ liệu) → **dời W8**
- [ ] Gán `question_type` cho ViTextVQA (để tách kết quả theo loại câu hỏi) → **dời W8**
- [ ] QLoRA chính thức nhiều epoch, chọn epoch trên dev → **dời W8**

## Kết quả / số liệu

### Cấu hình chốt (áp dụng cho mọi run từ B1)

Qwen2.5-VL-3B-Instruct · bf16 · 4-bit NF4 + double quant · `max_pixels` 401.408 (≤ 512 token ảnh) · greedy, `max_new_tokens` 32 · seed 42 · L4.
QLoRA: r = 16, α = 32, dropout 0,05, target q/k/v/o · **7.372.800 tham số train = 0,196%** · lr 2e-4 · batch hiệu dụng 16 (2 × 8) · warmup 3% · checkpoint mỗi 200 step.

| Quyết định | Luật đặt trước | Số đo | Kết luận |
|---|---|---|---|
| dtype | bf16 nếu 0 câu hỏng và thoái lui đối chứng ≤ 2/50 | fp16: 23/23 câu hỏng (tái hiện đúng lỗi T4) · bf16: 0 câu hỏng, 1/50 thoái lui so với fp32 | **bf16** |
| Độ phân giải | 1.280 token nếu train chậm ≤ 1,5× so với 512 | ViTextVQA: 34,77 vs 18,18 s/step = **1,91×** | **512 token** (1.280 để làm ablation) |

### B1 — baseline chính thức (zero-shot)

| Tập | N | EM | ANLS | token-F1 (phụ) | s/câu |
|---|---|---|---|---|---|
| ViVQA test | 3.001 | 0,2209 | 0,2861 | 0,3805 | 0,51 |
| ViTextVQA test2000 | 2.000 | 0,2880 | 0,4753 | 0,5377 | 0,86 |
| ViTextVQA full test | 10.028 | 0,2912 | 0,4760 | 0,5389 | 0,88 |

VQA-Acc = EM vì mỗi câu chỉ có 1 đáp án gold. B0 (W05, fp16 trên T4) chỉ dùng **tham khảo**: fp16 gây 23 câu hỏng trên ViTextVQA nên không so trực tiếp với B1.

### R0 — kiểm chứng pipeline bằng số đã công bố

| Run | N | VQA-Acc (%) | Công bố | VRAM đỉnh |
|---|---|---|---|---|
| R0 (cấu hình paper: bf16, không lượng tử, độ phân giải mặc định) | 5.000 | **79,60** | 79,3 (Bai et al., 2025, Table 5) | 10,9 GB |
| R0b (cấu hình B1: 4-bit, 512 token) | 5.000 | 75,21 | — | 2,54 GB |

R0 nằm trong ngưỡng ±2,0 điểm đặt trước → **pipeline tái lập được**. Cấu hình tiết kiệm của nhóm mất 4,39 điểm (CI95 [−5,35; −3,45]) nhưng giảm VRAM 4,3×.

### B2 — QLoRA 1 epoch so với B1

| Tập | N | EM B1 → B2 | ANLS B1 → B2 | McNemar (chỉ B1 đúng / chỉ B2 đúng) | ΔANLS CI95 |
|---|---|---|---|---|---|
| ViVQA test | 3.001 | 0,2209 → **0,7378** | 0,2861 → 0,7566 | 35 / 1.586, p ≈ 0 | [+0,453; +0,489] |
| ViTextVQA test2000 | 2.000 | 0,2880 → **0,4970** | 0,4753 → 0,7191 | 86 / 504, p ≈ 7 × 10⁻⁷³ | — |
| ViTextVQA full | 10.028 | 0,2912 → **0,5026** | 0,4760 → 0,7202 | 423 / 2.543, p ≈ 0 | [+0,235; +0,253] |

| Chi phí train | ViVQA | ViTextVQA |
|---|---|---|
| Số mẫu train | 10.799 | 35.159 |
| Step / thời gian 1 epoch | 675 / **3,03 h** | 2.198 / **11,99 h** |
| VRAM đỉnh | 6,62 GB | 6,88 GB |
| Loss cuối | 0,476 | 0,618 |

Độ dài câu trả lời (số từ trung bình) trên ViVQA: gold 1,57 · B1 2,88 · B2 1,55.

**So với SOTA của paper ViTextVQA** (full test; luật: "cao hơn" nếu cận dưới CI95 bootstrap > số công bố):

| | EM (CI95) | F1 (CI95) | Kết luận |
|---|---|---|---|
| ViTextBLIP-2 (Nguyen et al., 2025, Table 6) | 25,48 | 53,95 | — |
| B1 zero-shot | 29,12 [28,23; 29,99] | 53,89 [53,11; 54,66] | EM cao hơn · F1 không phân biệt được |
| B2 QLoRA | **50,26** [49,27; 51,23] | **71,16** [70,42; 71,89] | **Cao hơn ở cả EM và F1** |

Kiểm tra phụ: dự đoán trên 2.000 câu chung trùng 100% giữa run full và run test2000 (tất định); EM của test2000 nằm trong CI95 của full test → test2000 đại diện tốt.

### Rò rỉ train → test

| | ViVQA | ViTextVQA |
|---|---|---|
| Ảnh train trùng ảnh test | 1.274 | **0** |
| Câu test có ảnh đã gặp trong train | 1.356 / 3.001 (45,2%) | 0 |
| Cặp hỏi–đáp test đã gặp trong train | 541 | 407 (khác ảnh) |

Trên tập **clean** ViVQA (1.219 câu không trùng ảnh, không trùng cặp hỏi–đáp): B1 → B2 = 0,250 → 0,691, ΔEM CI95 [+0,41; +0,47]. Mức tăng của B2 **không do rò rỉ**. ViTextVQA không có ảnh trùng nên là phép thử tổng quát hoá sạch hơn.

### R1 — backbone lớn hơn (Qwen2-VL-7B, zero-shot)

| Run | EM | F1 (phụ) | ANLS | s/câu | VRAM đỉnh |
|---|---|---|---|---|---|
| B1 · Qwen2.5-VL-3B · 4-bit · 512 token | **28,8** | 53,77 | 47,53 | 0,86 | — |
| R1 · Qwen2-VL-7B · cấu hình paper | 16,7 | 43,50 | 29,94 | 2,85 | 20,27 GB |
| R1b · Qwen2-VL-7B · cấu hình B1 | 16,2 | 41,31 | 29,04 | 1,03 | 6,2 GB |
| Paper Table 7 "QwenVL-7b" (100 câu) | 28,0 | 53,8 | — | — | — |

ViTextVQA test2000. R1 lệch Table 7 (|z| > 1,96); B1 lại nhất quán với dòng đó. Backbone lớn hơn **không** tự động tốt hơn: 7B kém 3B khoảng 12 điểm EM, chủ yếu vì trả lời dài.

### FS5 — đối chứng few-shot (tách lợi ích định dạng khỏi QLoRA)

| Tập | EM B1 | EM FS5 | EM B2 | Few-shot sửa định dạng? | B2 > FS5? |
|---|---|---|---|---|---|
| ViVQA | 22,09 | 20,03 | 73,78 | **Không** — EM giảm (p < 0,05) | Có (p < 0,05) |
| ViTextVQA test2000 | 28,80 | 27,60 | 49,70 | Chưa kết luận (p ≥ 0,05) | Có (p < 0,05) |

5 ví dụ chỉ có chữ, lấy từ train (seed 42). Kết luận: mức tăng của B2 không thể đạt được chỉ bằng prompt; QLoRA học nhiều hơn định dạng câu trả lời.

### Error audit — 200 câu B2 sai EM (100 mỗi dataset)

| Chỉ số | Toàn bộ | ViVQA | ViTextVQA |
|---|---|---|---|
| B2 **đúng ý** (người đọc chấp nhận) | **54,0%** [47,1; 60,8] | 62,0% | 46,0% |
| Hallucination | **7,0%** [4,2; 11,4] | 5,0% | 9,0% |

Phân loại nguyên nhân chính (%):

| Loại lỗi | Toàn bộ | ViVQA | ViTextVQA |
|---|---|---|---|
| Định dạng / độ dài (ý đúng) | 21,0 | 9,0 | **33,0** |
| Đồng nghĩa / khác từ (ý đúng) | 17,5 | **34,0** | 1,0 |
| Sai vùng / không có trong ảnh | 15,5 | 8,0 | 23,0 |
| Đọc chữ sai (OCR) | 14,0 | — | **28,0** |
| Khác | 12,5 | 25,0 | — |
| Gold sai / nhiễu | 12,5 | 19,0 | 6,0 |
| Sai dấu thanh | 4,0 | — | 8,0 |
| Đếm sai | 2,5 | 5,0 | — |
| Code-switching | 0,5 | — | 1,0 |

Khoảng tin cậy 95% theo Wilson. Độ tin cậy gán nhãn: κ Cohen giữa hai người trên 20 dòng chung = 0,62 (loại lỗi) / 0,57 (đúng ý); κ người–AI ở hallucination = 0,30 (thấp → nhãn hallucination dựa vào người + phân xử). Bỏ 42 dòng có hỗ trợ AI: đúng ý 53,8%, hallucination 7,0% → **kết luận không đổi**.

**Ý nghĩa:**
- Hơn một nửa "lỗi" EM của B2 thực ra đúng ý → EM đánh giá thấp mô hình; cần báo cáo kèm ANLS/F1 và phân tích định tính.
- Trên ViTextVQA, nhóm lỗi đọc chữ (OCR 28 + sai vùng 23 + dấu thanh 8 = 59%) là điểm yếu chính → căn cứ chọn **OCR-enhanced prompting** làm cải tiến ở W8.
- Gold nhiễu 19% ở ViVQA (dataset dịch máy) → cần `data_audit` và ablation làm sạch nhãn.

Giới hạn: người phân xử đã thấy nhãn AI (có thể bị neo theo AI — anchoring); so sánh hallucination B1 vs B2 (H8) cần audit thêm lỗi của B1 (dự kiến W11).

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|--------|-----|-------------|
| B1 | 1 epoch ViTextVQA mất 12 h L4 → không đủ ngân sách Colab để train nhiều epoch hoặc nhiều biến thể trên full train | **Cao** | Nhiều epoch chỉ trên ViVQA; B3 (QLoRA + OCR) và B2-sub train trên **tập con cố định ≈ 8.000 mẫu, seed 42** (chốt 12/10, xem `../W08/README.md`) |
| B2 | OCR-enhanced prompting (ablation bắt buộc theo đề cương) chưa nối vào `run_baseline.py` / `train_qlora.py` | **Cao** | W8: nối `mode="ocr"` với OCR cache + selftest |
| B3 | EM đánh giá thấp mô hình (54% lỗi EM là đúng ý) | Trung bình | Báo cáo song song ANLS + token-F1 + error audit; không đổi metric chính |
| B4 | Nhiễu nhãn ViVQA chưa đo trực tiếp trên train | Trung bình | W8: `data_audit` 200 nhãn train |
| B5 | Few-shot / prompt không cải thiện → hướng "prompt optimization" trong kế hoạch W06 bị thay bằng đối chứng FS5 | Thấp | Giữ FS5 làm bằng chứng; không đầu tư thêm vào prompt thuần |

## Kế hoạch tuần sau (W08 — Improvement: OCR-enhanced prompting)

1. Hồ sơ Review 2: Team Report + README W07 (file này); sau buổi Review 2 điền checklist AI_Capstone_Review123 (sheet Review2) theo góp ý reviewer.
2. Nối OCR vào `run_baseline.py` và `train_qlora.py` (`--mode ocr` đọc OCR cache) + selftest.
3. Tạo OCR cache ViTextVQA (dev → test → train), đo tốc độ trên 100 ảnh trước.
4. Chọn backend OCR (`paddleocr` vs `paddledet_vietocr`) và mẫu prompt trên dev300.
5. Ablation OCR chỉ ở inference trên test2000: B1 + OCR, B2 + OCR.
6. QLoRA ViVQA 2 epoch, đánh giá val sau mỗi epoch để chọn epoch.

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `72e3a79`, `22877ec`, `7729301` | Notebook 05 + chốt cấu hình L4 (bf16, 512 token, token-F1 phụ, checkpoint/resume, chữ ký run) |
| `280a543`, `e0984e8` | Sửa `final_loss` sau resume; bằng chứng resume trên L4 |
| `e9acd3c`, `b5a58dd` | 3 sơ đồ theo góp ý GVHD; đưa báo cáo Review 1 về đúng mốc (tag `review1-report`) |
| `c96e3f3`, `c8cf419` | Notebook 02 + kết quả B1 |
| `3fcb1f4`, `fefa9ed` | `--adapter`, `--epochs`; R0 notebook 06 |
| `5e4154f` | B2 ViVQA |
| `026e3b1`, `53a89ed` | Kiểm tra rò rỉ; tách kết quả theo mức trùng |
| `22b44a3` | Kết quả R0 / R0b |
| `997f9c3` | B2 ViTextVQA |
| `e8d8fa2`, `884d964` | `--variant` Qwen2-VL-7B + `paper_consistency.py`; kết quả R1 / R1b |
| `b7dc647`, `8fdf8d2` | `--few_shot_k`, `answer_format.py`, `make_audit_sheet.py` |
| `ce54e31` | Full test ViTextVQA 10.028 câu (B1, B2) |
| `60ec444` | Kết quả FS5 |
| `f562e07` | Slide Review 2 EN + TV + kịch bản + sơ đồ EN |
| `4fcc084`, `665b966`, `2f19ca4` | Error audit: thiết lập, kết quả, nguồn gốc nhãn AI |
| `3b41b14` | Script thu hồi chia sẻ Drive (bản dùng thật `thuHoiNhanh` v2 commit ở `5d161c4`, W8) |

## Link experiment log

- Chốt cấu hình: `experiments/W07_config_freeze_L4_seed42/`, `experiments/W07_probe_*_L4_seed42/`, `experiments/W07_smoke_qlora_*_L4_seed42/`, `experiments/W07_resume_test_qlora_qwen25vl_vivqa_bf16_L4_seed42/`
- B1: `experiments/W07_B1_summary_L4_seed42/` + 3 run `W07_B1_zeroshot_*`
- B2: `experiments/W07_B2_summary_L4_seed42/` + 2 run train `W07_B2_qlora_*` + 3 run eval `W07_B2_eval_*`
- Full test: `experiments/W07_fulltest_summary_L4_seed42/`
- R0: `experiments/W07_R0_summary_L4/`, `experiments/W07_R0_textvqa_val_qwen25vl3b_paper_bf16_L4/`, `experiments/W07_R0b_textvqa_val_qwen25vl3b_b1cfg_L4/`
- R1: `experiments/W07_R1_summary_L4_seed42/` + 2 run `W07_R1*`
- FS5: `experiments/W07_FS_summary_L4_seed42/` + 2 run `W07_FS5_*`
- Rò rỉ: `experiments/W07_data_checks/`
- Error audit: `docs/W07/audit/W07_error_audit_final.json` (tái tạo bằng `docs/W07/audit/compute_error_audit.py`)
