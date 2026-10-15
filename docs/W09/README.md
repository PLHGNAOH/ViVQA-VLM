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
| 2 | **Phần B:** seed 43 — train + eval B2-sub và B3 | ≈ 7 h L4 | ⏳ |
| 3 | **Phần B':** seed 44 — như trên | ≈ 7 h L4 | ⏳ |
| 4 | OCR phần ảnh full test còn thiếu (T4) → B3 trên **full test 10.028** | ≈ 0,5 h T4 + ≈ 2,8 h L4 | ⏳ |
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

## Thay đổi code

| File | Thay đổi | Kiểm thử |
|---|---|---|
| `src/adaptation/train_qlora.py` | `--seed` ghi đè `run.seed`; seed nằm trong chữ ký run (seed khác = run khác, không resume nhầm); dòng log đầu in `seed=` | selftest `seed override OK`: seed 42 mặc định trùng chữ ký cũ, seed 43 khác chữ ký |
| `src/eval/seed_stability.py` | Gộp `paired_*.json` qua các seed, mean ± std, áp luật trên | `--selftest` (4 tình huống luật + chặn số câu khác nhau) |
| `notebooks/14_b3_seed_stability_qwen25vl.ipynb` | Phần A (bỏ OCR) → Phần B seed 43 → B' seed 44 → Phần C gộp; bỏ qua phần đã xong; train lỗi thì dừng, không eval | Mô phỏng offline mục 4 và mục 8 bằng predictions W08 |

## Ngân sách

| Hạng mục | GPU | CU (1,54 CU/h L4) |
|---|---|---|
| Phần A | ≈ 0,7 h | ≈ 1 |
| Seed 43 + 44 | ≈ 14 h | ≈ 22 |
| Full test B3 (+ OCR T4) | ≈ 3,3 h | ≈ 5 |
| ViVQA 2 epoch | ≈ 7,5 h | ≈ 12 |
| **Tổng W9** | **≈ 25 h** | **≈ 40** (còn ≈ 93 CU cho W10–W15) |

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

## Link experiment log

- Phần A: `experiments/W09_B3_ocr_dependency_test2000_L4_seed42/` + eval `W09_B3noocr_eval_…_seed42`
- Seed: `experiments/W09_B3_seed_stability_test2000_L4/` + train `W09_B2sub_qlora_…_seed{43,44}`, `W09_B3_qlora_ocr_…_seed{43,44}` + eval `W09_B2sub_eval_…`, `W09_B3_eval_…`
- Adapter (không lên git, trên Drive): `adapters/W09_*_seed{43,44}/epoch_1`
