# Week 9 — Củng cố B3 + Review 2

Index tuần 9 (tuần ảo) của ViVQA-VLM. Tuần ảo W09 chạy trong **tuần thực 6–7** (15/10 → Review 2 ở tuần thực 7, 19–25/10).

Mục tiêu W9: **củng cố kết quả chính W08** (B3 = QLoRA + OCR thắng B2-sub) trước khi đưa lên Review 2 — thêm seed, kiểm phụ thuộc OCR, chạy full test — và **cập nhật slide Review 2** với phương pháp đề xuất + kết quả sơ bộ.

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Vì sao phải củng cố B3

Kết quả W08 (`../W08/README.md`): B3 vs B2-sub trên test2000 — EM 0,4555 → 0,4925, McNemar p = 1,7 × 10⁻⁶, ΔANLS +0,019 [+0,007; +0,031]. Còn 4 giới hạn đã ghi:

1. **Mới 1 seed (42)** → lợi ích có thể do khởi tạo LoRA / thứ tự batch "may mắn".
2. Mới test2000, chưa full test 10.028 câu.
3. Chưa biết B3 có **phụ thuộc OCR** không (train có OCR; lúc dùng thật OCR có thể hỏng).
4. ANLS B3 vẫn thấp hơn B2 full.

Giới hạn 1 và 3 quyết định độ tin cậy của đóng góp chính → làm trước.

## Kế hoạch tuần

| # | Việc | GPU | Trạng thái |
|---|------|-----|-----------|
| 0 | Code: `--seed` cho `train_qlora.py` (selftest), `src/eval/seed_stability.py`, notebook `14_b3_seed_stability_qwen25vl.ipynb`, luật ghi trước (README này) | — | ✅ 15/10 |
| 1 | **Phần A:** B3 seed 42 eval **bỏ OCR** lúc test (test2000) | ≈ 40 phút L4 | ✅ 15/10 — B3 PHỤ THUỘC OCR; bỏ OCR thì NGANG B2-sub (EM) |
| 2 | **Phần B:** seed 43 — train + eval B2-sub và B3 | ≈ 7 h L4 | ✅ 15/10 (B3 chạy tiếp từ checkpoint-100 sau khi runtime bị xoá) |
| 3 | **Phần B':** seed 44 — như trên | ≈ 7 h L4 | ✅ 16/10 — **CẢI THIỆN ỔN ĐỊNH** (3/3 seed, ΔEM +3,57 ± 0,76 điểm) |
| 4a | **Ô 2×2:** B2-sub seed 42 eval **có OCR** lúc test (test2000) → đủ bảng OCR lúc train × OCR lúc test — notebook `15_ocr_factorial_fulltest_qwen25vl.ipynb` Phần A | ≈ 40 phút L4 | ⏳ |
| 4b | OCR phần ảnh full test còn thiếu (≈ 1.867 ảnh) — notebook 10 **Phần C** | ≈ 0,5–1 h T4 | ⏳ |
| 4c | **Full test 10.028 câu:** B3 và B2-sub seed 42 — notebook 15 Phần B | ≈ 5,7 h L4 | ⏳ |
| 5 | **Review 2:** thêm Part 06 (phương pháp OCR + kết quả B3 + độ ổn định seed) vào slide `docs/W07/ViVQA-VLM_Review2_*.pptx`, sửa slide "Plan", cập nhật script thuyết trình | — | ⏳ trước 19/10 |
| 6 | QLoRA ViVQA 2 epoch, chọn epoch trên val (dời từ W8) | ≈ 7,5 h L4 | ⏳ (sau Review 2 được) |
| 7 | Sau buổi Review 2: checklist AI_Capstone_Review123 (sheet Review2), gửi mail, upload Drive | — | ⏳ |
| 8 | Đóng tuần: `W09_Team_Report.md` | — | ⏳ |

## Quyết định đã chốt

| Ngày | Quyết định | Lý do |
|------|-----------|-------|
| 15/10/2026 | **3 seed huấn luyện (42 / 43 / 44)** cho B2-sub và B3 | 2 seed không ước lượng được độ lệch chuẩn có nghĩa; "mean ± std trên 3 seed" là mức tối thiểu hội đồng chấp nhận. Chi phí thêm ≈ 14 h L4 ≈ 22 CU (còn 133 CU ngày 14/10) |
| 15/10/2026 | **Giữ cố định** tập con train 8.000 câu (`train_subset8000_byimage_seed42`); chỉ đổi **seed huấn luyện** (`--seed`) | Tách riêng một nguồn dao động: khởi tạo LoRA + thứ tự batch + dropout. Đổi cả tập con thì không biết chênh lệch đến từ đâu |
| 15/10/2026 | Seed trong tên run eval (`…_seed43`) = **seed huấn luyện** của adapter | Suy luận greedy (không lấy mẫu) nên seed không ảnh hưởng eval |
| 15/10/2026 | **Đưa kết quả B3 vào slide Review 2** (Part 06) | Master §9: Review 2 hỏi "proposed method là gì, có preliminary results không". Kết quả có trước ngày review nên thuộc mốc Review 2 |
| 15/10/2026 | `--seed` không truyền → seed của config (42) → **chữ ký các run W07/W08 không đổi** | Selftest `seed override OK` kiểm điều này; run cũ resume/tái lập được như trước |
| 15/10/2026 | **Thêm ô 2×2** (B2-sub + OCR lúc test) | Phần A cho thấy B3 phụ thuộc OCR, nhưng chưa tách được "train cùng OCR" khỏi "có OCR lúc test" trên cùng tập con. Chỉ ≈ 1 CU |
| 16/10/2026 | Notebook 15 thêm cell **tự trả máy** (`runtime.unassign()`) ở cuối | Notebook 14 xong lúc 1:37 nhưng máy L4 bị giữ tới khi Colab tự ngắt → tốn thêm ≈ 2–3 CU |
| 15/10/2026 | **Full test chạy cả B3 lẫn B2-sub** (không chỉ B3) | So sánh chính B2-sub → B3 được lặp lại trên 10.028 câu (độ nhạy cao hơn test2000 ≈ 5 lần về số câu), cùng tập paper dùng để so ViTextBLIP-2. Thêm ≈ 2,8 h L4 ≈ 4 CU |

## Luật Phần A — B3 bỏ OCR lúc test (ghi TRƯỚC khi chạy, 15/10/2026)

Adapter B3 seed 42 (W08), eval test2000 với prompt **zero-shot** (không dòng OCR). Hai so sánh ghép cặp (McNemar exact trên EM, ANLS kèm CI95 bootstrap 10.000 lần, seed 42):

| So sánh | Câu hỏi | Kết luận |
|---|---|---|
| B3 (có OCR) → B3 bỏ OCR | B3 có dựa vào dòng OCR không? | EM giảm **và** p < 0,05 → **B3 PHỤ THUỘC OCR**; còn lại → không phụ thuộc có ý nghĩa |
| B2-sub → B3 bỏ OCR | Train cùng OCR có làm model yếu đi khi thiếu OCR? | EM giảm & p < 0,05 → **KÉM HƠN B2-sub**; EM tăng & p < 0,05 → **VẪN TỐT HƠN B2-sub**; còn lại → **NGANG B2-sub** |

Đây là **chẩn đoán**, không phải kết quả cải tiến. Hệ quả cho prototype: nếu "PHỤ THUỘC OCR" + "KÉM HƠN B2-sub" thì demo phải chạy OCR trước và báo rõ khi OCR lỗi.

## Luật độ ổn định theo seed (ghi TRƯỚC khi chạy seed 43/44, 15/10/2026)

Mỗi seed s ∈ {42, 43, 44}: B2-sub(s) và B3(s) cùng tập con, cùng siêu tham số, cùng cấu hình OCR đóng băng (VietOCR · v1 · 600 · 0), eval test2000, so sánh ghép cặp B2-sub(s) → B3(s). Áp bằng `src/eval/seed_stability.py`:

| Kết luận | Điều kiện |
|---|---|
| **CẢI THIỆN ỔN ĐỊNH** | Cả 3 seed: EM(B3) > EM(B2-sub) **và** McNemar exact p < 0,05 |
| **HƯỚNG NHẤT QUÁN, CHƯA ỔN ĐỊNH** | Cả 3 seed EM(B3) > EM(B2-sub), nhưng có seed p ≥ 0,05 |
| **KHÔNG ỔN ĐỊNH** | Có seed EM(B3) ≤ EM(B2-sub) |

- Báo cáo EM, ANLS, ΔEM, ΔANLS dạng **mean ± std** (độ lệch chuẩn mẫu, ddof = 1) trên 3 seed, kèm bảng từng seed (chỉ A đúng / chỉ B đúng, p, ΔANLS CI95).
- **Không** chạy thêm seed sau khi thấy kết quả để "câu" kết luận. Kết quả nào cũng báo cáo đầy đủ.
- Seed 42 tính lại từ predictions W08 phải khớp `../../experiments/W08_B3_vs_B2sub_test2000_L4_seed42/paired_B2sub_vs_B3.json` (notebook 14 tự kiểm).

## Luật ô 2×2 — OCR lúc train × OCR lúc test (ghi TRƯỚC khi chạy ô c01, 15/10/2026)

Cùng test2000, cùng tập con train seed 42, cùng cấu hình OCR đóng băng (VietOCR · v1 · 600 · 0):

| | Test **không** OCR | Test **có** OCR |
|---|---|---|
| Train **không** OCR | c00 = B2-sub (W08) | **c01 = B2-sub + OCR (mới)** |
| Train **có** OCR | c10 = B3 bỏ OCR (W09 Phần A) | c11 = B3 (W08) |

Áp bằng `src/eval/factorial_2x2.py`. So sánh quyết định **K = c01 → c11** (khi lúc test ĐÃ có OCR, train cùng OCR có thêm lợi ích không) và **tương tác** d = (c11 − c10) − (c01 − c00) tính theo từng câu, CI95 bootstrap ghép cặp (10.000, seed 42):

| Kết luận | Điều kiện |
|---|---|
| **TƯƠNG TÁC: CẦN OCR Ở CẢ TRAIN LẪN TEST** | K: EM tăng **và** p < 0,05, **và** cận dưới CI95 tương tác EM > 0 |
| **CỘNG GỘP: TRAIN CÙNG OCR CÓ LỢI RIÊNG** | K: EM tăng và p < 0,05, nhưng CI95 tương tác EM chứa 0 |
| **OCR LÚC TEST LÀ ĐỦ** | K: p ≥ 0,05 → train cùng OCR không thêm lợi ích có ý nghĩa (phải ghi thật vào báo cáo: đóng góp chính khi đó là OCR lúc test, không phải QLoRA + OCR) |
| **TRAIN CÙNG OCR KÉM HƠN** | K: EM giảm và p < 0,05 |

Báo cáo đủ 4 hiệu ứng đơn (c00→c01, c10→c11, c00→c10, c01→c11) kèm McNemar p và ΔANLS CI95, cùng tương tác ANLS.

## Luật full test 10.028 câu (ghi TRƯỚC khi chạy, 15/10/2026)

Run: B3 seed 42 (OCR train + test) và B2-sub seed 42 (không OCR) trên `data/vitextvqa_official/test.json` — cùng adapter, cùng cấu hình như test2000. OCR cache phải phủ đủ 3.353 ảnh test.

1. **Tái lập (điều kiện dừng):** trên 2.000 câu chung, dự đoán của run full phải **trùng ≥ 99%** với run test2000 tương ứng (W08). Dưới 99% → không dùng số liệu, báo Claude.
2. **So sánh chính lặp lại:** B2-sub → B3 trên full test: "TÁI LẬP TRÊN FULL TEST" nếu EM tăng **và** McNemar exact p < 0,05; còn lại báo cáo đúng như kết quả.
3. **So với ViTextBLIP-2** (Table 6, Nguyen et al., *ESWA* 2025: EM 25,48 · F1 53,95): chỉ viết "cao hơn" khi **cận dưới CI95 bootstrap** (10.000, seed 42) của B3 lớn hơn số công bố, xét riêng EM và F1 — như luật W07 (`notebooks/08_…`). Giới hạn: paper không công bố cách chuẩn hoá khi chấm.
4. **Chỉ báo cáo:** B2 full train (W07, full test) → B3; B3 dùng 23% dữ liệu train.
5. **Cache OCR mở rộng không được đổi phần cũ:** dấu vân tay (`cache_fingerprint`) của 4.422 ảnh đã có phải giữ nguyên sau khi OCR thêm (notebook 10 Phần C tự kiểm). Phiên bản `paddlepaddle` của đợt OCR mới được ghi vào biên bản `experiments/W09_ocr_cache_fulltest/`.

## Thay đổi code

| File | Thay đổi | Kiểm thử |
|---|---|---|
| `src/adaptation/train_qlora.py` | `--seed` ghi đè `run.seed`; seed nằm trong chữ ký run (seed khác = run khác, không resume nhầm); dòng log đầu in `seed=` | selftest `seed override OK`: seed 42 mặc định trùng chữ ký cũ, seed 43 khác chữ ký |
| `src/eval/seed_stability.py` | Gộp `paired_*.json` qua các seed, mean ± std, áp luật trên | `--selftest` (4 tình huống luật + chặn số câu khác nhau) |
| `notebooks/14_b3_seed_stability_qwen25vl.ipynb` | Phần A (bỏ OCR) → Phần B seed 43 → B' seed 44 → Phần C gộp; bỏ qua phần đã xong; train lỗi thì dừng, không eval | Mô phỏng offline mục 4 và mục 8 bằng predictions W08 |
| `src/eval/factorial_2x2.py` | Bảng 2×2 OCR lúc train × OCR lúc test: 4 hiệu ứng đơn + tương tác (difference-in-differences theo câu) + luật trên | `--selftest` (4 tình huống luật + chặn lệch question_id) |
| `notebooks/10_ocr_cache_vitextvqa.ipynb` | Thêm **Phần C** (mục 11–12): OCR phần ảnh full test còn thiếu, kiểm dấu vân tay phần cache cũ, ghi biên bản W09 | Chạy trên T4 |
| `notebooks/15_ocr_factorial_fulltest_qwen25vl.ipynb` | Phần A: ô c01 + bảng 2×2. Phần B: B3 + B2-sub full test, kiểm tái lập ≥ 99%, so ViTextBLIP-2, so sánh chính lặp lại | Mô phỏng offline mục kết quả |

## Ngân sách

| Hạng mục | GPU | CU (1,54 CU/h L4) |
|---|---|---|
| Phần A | ≈ 0,7 h | ≈ 1 |
| Seed 43 + 44 | ≈ 14 h | ≈ 22 |
| Ô 2×2 | ≈ 0,7 h | ≈ 1 |
| Full test B3 + B2-sub (+ OCR T4) | ≈ 6,5 h | ≈ 10 |
| ViVQA 2 epoch | ≈ 7,5 h | ≈ 12 |
| **Tổng W9** | **≈ 29 h** | **≈ 46** (còn ≈ 87 CU cho W10–W15) |

## Kết quả

### Phần A — B3 bỏ OCR lúc test (15/10/2026, L4, bf16, 4-bit, seed 42)

Nguồn: `../../experiments/W09_B3_ocr_dependency_test2000_L4_seed42/` + eval `W09_B3noocr_eval_qlora_ocr_vietocr_v1_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed42`.

| Run | OCR lúc train | OCR lúc test | EM | ANLS | token-F1 (phụ) | s/câu |
|---|---|---|---|---|---|---|
| B2-sub | không | không | 0,4555 | 0,6848 | 0,6692 | 1,021 |
| B3 | có | có | 0,4925 | 0,7038 | 0,6981 | 1,001 |
| B3 bỏ OCR | có | **không** | 0,4500 | 0,6746 | 0,6617 | 0,979 |

| So sánh | Chỉ A đúng / chỉ B đúng | McNemar p | ΔANLS (CI95) | Kết luận theo luật |
|---|---|---|---|---|
| B3 → B3 bỏ OCR | 147 / 62 | 3,8 × 10⁻⁹ | −0,0292 [−0,0401; −0,0185] | **B3 PHỤ THUỘC OCR** |
| B2-sub → B3 bỏ OCR | 63 / 52 | 0,351 | −0,0101 [−0,0194; −0,0007] | **NGANG B2-sub** (theo EM) |

**Diễn giải:**
- Bỏ OCR lúc test làm B3 mất 4,25 điểm EM — nhiều hơn toàn bộ phần lợi +3,70 điểm so với B2-sub. → Lợi ích của B3 đến từ việc model **thật sự dùng dòng OCR**, không phải do train "khéo hơn" nói chung: bằng chứng cơ chế cho kết quả chính W08.
- Thiếu OCR, B3 lùi về mức B2-sub (EM ngang, p = 0,35): suy giảm êm, không sụp.
- **Giới hạn nhỏ:** ANLS của B3 bỏ OCR thấp hơn B2-sub −0,010 (CI95 không chứa 0) → train cùng OCR có thể làm model hơi giảm khả năng tự đọc chữ chính xác từng ký tự. Luật kết luận theo EM nên ghi "NGANG", nhưng phải nêu trong báo cáo.
- Cùng với W08 bước 6 (chèn OCR chỉ lúc suy luận không giúp B1/B2), kết quả **gợi ý** OCR cần có ở **cả train lẫn test**. Chưa khẳng định được vì thiếu ô "B2-sub train không OCR + test có OCR" trên cùng tập con.
- Hệ quả prototype: luôn chạy OCR trước khi hỏi model, hiển thị văn bản OCR, cảnh báo khi OCR rỗng/lỗi (chất lượng khi đó ≈ B2-sub).

### Độ ổn định theo seed (16/10/2026, L4, bf16, 4-bit, test2000)

Nguồn: `../../experiments/W09_B3_seed_stability_test2000_L4/` (`seed_stability_summary.json` + 3 file `paired_B2sub_vs_B3_seed{42,43,44}.json`); train `W09_B2sub_qlora_…_seed{43,44}`, `W09_B3_qlora_ocr_…_seed{43,44}`; eval `W09_B2sub_eval_…_seed{43,44}`, `W09_B3_eval_…_seed{43,44}`. Seed 42 = run W08, tính lại từ predictions khớp 81 / 155.

| Seed | EM B2-sub | EM B3 | ΔEM (điểm) | Chỉ B2-sub đúng / chỉ B3 đúng | McNemar p | ΔANLS (CI95) |
|---|---|---|---|---|---|---|
| 42 | 0,4555 | 0,4925 | +3,70 | 81 / 155 | 1,7 × 10⁻⁶ | +0,0191 [+0,0073; +0,0309] |
| 43 | 0,4475 | 0,4900 | +4,25 | 77 / 162 | 4,0 × 10⁻⁸ | +0,0161 [+0,0040; +0,0280] |
| 44 | 0,4605 | 0,4880 | +2,75 | 74 / 129 | 1,4 × 10⁻⁴ | +0,0112 [+0,0000; +0,0224] |
| **mean ± std** | 45,45 ± 0,66 | 49,02 ± 0,23 | **+3,57 ± 0,76** | | | +0,0155 ± 0,0040 |

ANLS mean ± std: B2-sub 0,6857 ± 0,0031 · B3 0,7011 ± 0,0024.

**Kết luận theo luật ghi trước: CẢI THIỆN ỔN ĐỊNH** — cả 3 seed EM(B3) > EM(B2-sub) và p ≤ 1,4 × 10⁻⁴.

**Diễn giải:**
- Lợi ích trung bình +3,57 điểm EM, lớn gấp ≈ 5 lần độ lệch chuẩn của B2-sub giữa các seed (0,66) → không phải do khởi tạo / thứ tự batch "may mắn".
- ANLS cùng chiều ở cả 3 seed, nhưng seed 44 có cận dưới CI95 = +0,0000 (sát 0) → lợi ích ANLS **yếu hơn** lợi ích EM. Luật kết luận dựa trên EM; ANLS phải báo cáo đúng như vậy.
- Độ lệch chuẩn EM của B3 (0,23) nhỏ hơn B2-sub (0,66). Chỉ 3 seed nên **không** kết luận B3 ổn định hơn — chỉ ghi nhận.

| Chi phí train | Seed | Giờ train | s/step | VRAM đỉnh (GB) | Loss cuối | Chạy tiếp từ |
|---|---|---|---|---|---|---|
| B2-sub | 42 | 2,76 | 19,89 | 6,85 | 0,7161 | — |
| B3 | 42 | 2,98 | 21,48 | 8,25 | 0,6605 | — |
| B2-sub | 43 | 2,69 | 19,40 | 6,84 | 0,7105 | — |
| B3 | 43 | 2,40\* | 21,56 | 8,24 | 0,6044\* | checkpoint-100 |
| B2-sub | 44 | 2,79 | 20,06 | 6,81 | 0,7162 | — |
| B3 | 44 | 3,01 | 21,66 | 8,25 | 0,6603 | — |

\* B3 seed 43: runtime bị xoá giữa chừng (15/10 ≈ 15:30), chạy tiếp từ checkpoint-100 (trạng thái optimizer + lịch learning rate + RNG khôi phục theo checkpoint). Giờ train và loss cuối chỉ tính **phiên cuối** (step 101–500) → không so trực tiếp với các run khác. Chi phí một run trọn: B2-sub ≈ 2,7–2,8 h, B3 ≈ 3,0 h (+8% s/step, +1,4 GB VRAM).

## Link experiment log

- Phần A: `experiments/W09_B3_ocr_dependency_test2000_L4_seed42/` + eval `W09_B3noocr_eval_…_seed42`
- Seed: `experiments/W09_B3_seed_stability_test2000_L4/` + train `W09_B2sub_qlora_…_seed{43,44}`, `W09_B3_qlora_ocr_…_seed{43,44}` + eval `W09_B2sub_eval_…`, `W09_B3_eval_…`
- Adapter (không lên git, trên Drive): `adapters/W09_*_seed{43,44}/epoch_1`
