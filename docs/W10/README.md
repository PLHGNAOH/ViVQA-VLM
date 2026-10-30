# Week 10 — Ablation cấu hình LoRA + B3 trên toàn bộ dữ liệu train

Index tuần 10 (tuần ảo) của ViVQA-VLM. Tuần ảo W10 chạy trong **tuần thực 7** (19–25/10/2026). Phần Review 2 của tuần thực 7 theo dõi riêng (checklist AI_Capstone_Review123), không nằm trong README này.

Mục tiêu W10: trả lời **hai câu hỏi còn mở** về công thức B3 (OCR-aware QLoRA) trước khi viết phần Results:

1. **Cấu hình LoRA đã hợp lý chưa?** — B3 dùng r = 16, chỉ 4 lớp attention; chưa từng kiểm (H3, H4).
2. **Đủ dữ liệu thì OCR-aware có vượt B2 không?** — W09: B3 (23% dữ liệu) thấp hơn B2 dữ liệu đầy đủ 1,16 EM trên full test.

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

> **Đánh số giả thuyết.** Luật ghi trước của tuần này (19/10) dùng đánh số **cũ** trong `docs/W02/Hypotheses.md`: H3 = vị trí gắn LoRA, H4 = rank. Ngày 20/10, deck cộng dồn Review 2 (`docs/04_slides/cumulative/`, commit `6359085`) chốt đánh số **chính thức mới** H1–H4 và chuyển cấu hình LoRA thành **ablation A1 (module gắn LoRA) và A2 (rank)**. Luật đã commit trước khi có số nên **giữ nguyên chữ**; bảng ánh xạ dùng cho báo cáo:
>
> | Luật tuần này (số cũ) | Tên chính thức (số mới) | Run (tên thư mục, không đổi) |
> |---|---|---|
> | H3 · all-linear so với attention-only, cùng ngân sách | **Ablation A1** · module gắn LoRA | `W10_A2_*` (all-linear r4) |
> | H4 · rank 64 so với 16 | **Ablation A2** · rank | `W10_A3_*` (attention r64) |
> | — | **H2** · OCR (≥ +3 ANLS, cộng hưởng với QLoRA) | `W10_B3full_*` |

## Kế hoạch tuần

| # | Việc | GPU | Trạng thái |
|---|------|-----|-----------|
| 0 | Code: `src/adaptation/lora_targets.py` (preset + chặn LoRA lọt vision), `--lora_r / --lora_alpha / --lora_targets` cho `train_qlora.py`, LoRA audit trong `train_log.json`, `src/eval/lora_ablation.py`, notebook `17_lora_ablation_rank_modules_qwen25vl.ipynb`, luật ghi trước (README này) | — | ✅ 19/10 (`8be932e`) |
| 1 | **A2** — all-linear r = 4 (cùng ngân sách tham số với B3), train + eval test2000 | ≈ 3,9 h L4 | ✅ 20/10 — **H3 KHÔNG ĐƯỢC ỦNG HỘ** (−0,55 EM, p = 0,34), chậm hơn 17,5% |
| 2 | **A3** — attention r = 64, train + eval test2000 | ≈ 3,7 h L4 | ✅ 20/10 — **ỦNG HỘ H4 (yếu)** (+1,05 EM, p = 0,071, CI95 [−0,05; +2,15]), adapter ×4 |
| 3 | OCR cho phần ảnh train còn thiếu (≈ 9.063 ảnh) — notebook 10 **Phần D** (mục 13–14) | ≈ 2–2,5 h T4 | ✅ 20/10 — +9.063 ảnh, 0 lỗi, 0,72 s/ảnh (≈ 1,9 h T4); 6.289 ảnh cũ giữ nguyên; cache 15.352 ảnh |
| 4 | **B3-full** — QLoRA + OCR trên toàn bộ train (35.159 câu), eval full test 10.028 câu — notebook `18_qlora_ocr_b3full_vs_b2_qwen25vl.ipynb`, kết luận bằng `src/eval/b3full.py` | ≈ 13 h + 2,8 h L4 | ✅ 21/10 — **OCR-AWARE VƯỢT B2 KHI ĐỦ DỮ LIỆU** (+3,09 EM, p = 6,7 × 10⁻²¹) |
| 5 | Results (R6) bản tiếng Việt — phần QLoRA nhận biết OCR (số liệu W08–W10) | — | ✅ 21/10 — `docs/03_report/R6_Results.md` mục 2–4 (bản nháp) |
| 6 | Đóng tuần: `W10_Team_Report.md`, dọn Downloads + Drive, handoff | — | ✅ 21/10 |

## Quyết định đã chốt

| Ngày | Quyết định | Lý do |
|------|-----------|-------|
| 19/10/2026 | Ablation LoRA chạy trên **công thức B3** (tập con 8k + OCR, seed 42), so với **B3 seed 42 (W08)** đã có | B3 là phương pháp đề xuất — câu hỏi "cấu hình LoRA có hợp lý không" phải trả lời trên chính nó. Dùng lại B3 làm mốc → chỉ tốn 2 run mới |
| 19/10/2026 | **A2 = all-linear r = 4** (7.483.392 tham số), không phải all-linear r = 16 | H3 hỏi "ở **cùng ngân sách tham số**". All-linear r = 16 có 29,9 triệu tham số (×4) → nếu tốt hơn cũng không biết do vị trí gắn hay do nhiều tham số. r = 4 lệch B3 chỉ +1,5% |
| 19/10/2026 | **A3 = attention r = 64** (29.491.200 tham số) | Đúng phát biểu H4 (r = 64 so với r = 16). Cùng vị trí gắn với B3 → chỉ khác rank |
| 19/10/2026 | **α = 2r** ở mọi run (r4/α8, r16/α32, r64/α128) | Giữ hệ số scale α/r = 2 như cấu hình gốc → so sánh rank không bị lẫn với thay đổi độ lớn bước cập nhật. `--lora_r` không kèm `--lora_alpha` tự đặt α = 2r |
| 19/10/2026 | Preset `all_linear_llm` dùng **regex** chỉ khớp `layers.<i>.(self_attn\|mlp)…` và loại tên chứa `visual` | Vision tower của Qwen2.5-VL cũng có `gate_proj/up_proj/down_proj` (`modeling_qwen2_5_vl.py`, transformers 5.17.0). List hậu tố sẽ gắn LoRA vào 32 × 3 lớp vision. `train_qlora` dừng nếu audit thấy lớp vision |
| 19/10/2026 | **1 seed (42)** cho ablation | Std EM của B3 qua 3 seed = 0,23 điểm (W09), nhỏ hơn nhiều mức test2000 phát hiện được (≈ 1,7 điểm). Thêm seed tốn ≈ 7 h L4 / seed / biến thể. Ghi là giới hạn |
| 19/10/2026 | **B3-full dùng nguyên công thức B3** (attention r = 16) bất kể kết quả ablation | B3-full so với B2 W07 (cũng attention r = 16). Đổi cấu hình LoRA theo kết quả ablation là chọn cấu hình bằng test → rò rỉ. Nếu A2/A3 thắng, ghi là hướng cải tiến tiếp |
| 20/10/2026 | **Review 2 dời sang thứ Năm 29/10/2026** (tuần thực 8), theo thông báo của bộ môn | Kết quả W10 có trước ngày review → theo quy ước W09 thuộc mốc Review 2; việc đưa vào deck cộng dồn làm trong tuần thực 8, trước buổi review |
| 21/10/2026 | **Mô hình ViTextVQA chính thức = B3-full** (thay B2 W07) cho bảng kết quả chính và prototype | Thắng B2 W07 theo luật ghi trước (cùng dữ liệu, cùng siêu tham số, chỉ khác OCR). Số B2, B2-sub, B3 vẫn báo cáo đủ trong bảng ablation. ViVQA giữ C3 (W09) |
| 21/10/2026 | **Adapter dùng chung 2 dataset: làm bản rẻ ở W11** (ViVQA đầy đủ + ViTextVQA tập con, 1 epoch); thiết kế + luật ghi trước chốt đầu W11 | Đã hứa trên slide Review 2 (kế hoạch tới Review 3); trả lời câu hỏi chuyển giao giữa hai loại câu hỏi; demo dùng 1 adapter nếu không kém. Nếu kém → demo chuyển adapter theo ảnh (PEFT hỗ trợ sẵn) |

## Luật ablation LoRA — H3 (vị trí gắn) và H4 (rank) (ghi TRƯỚC khi chạy, 19/10/2026)

Mọi thứ khác **y hệt B3 seed 42 (W08)**: `train_subset8000_byimage_seed42` (sha256 trong manifest W08), OCR VietOCR · v1 · 600 · 0, lr 2e-4, 1 epoch (500 step), batch hiệu dụng 16, warmup 3%, dropout 0,05, bf16, 4-bit NF4 + double quant, max_pixels 401.408, L4, eval test2000 greedy (max_new_tokens 32). `src/eval/lora_ablation.py` **tự kiểm** A2/A3 chỉ khác B3 ở mục LoRA.

| Run | Lớp gắn LoRA (chỉ LLM, 36 layer) | r | α | Tham số train |
|---|---|---|---|---|
| B3 (mốc, W08) | q, k, v, o | 16 | 32 | 7.372.800 |
| A2 | q, k, v, o, gate, up, down | 4 | 8 | 7.483.392 |
| A3 | q, k, v, o | 64 | 128 | 29.491.200 |

**Điều kiện hợp lệ (kiểm trước khi eval):** `train_log.json` có `lora_audit.by_tower` = `{'llm': 252}` (A2) / `{'llm': 144}` (A3), không có `vision`; số tham số train (PEFT đếm) = số tính tay ở bảng trên. Sai → không dùng run, báo Claude.

So sánh ghép cặp trên cùng 2.000 câu (`compare_runs.py`: McNemar exact trên EM, ΔANLS kèm CI95 bootstrap 10.000 lần, seed 42); thêm CI95 của ΔEM (bootstrap ghép cặp từ bảng chéo, seed 42):

| Giả thuyết | So sánh | Kết luận |
|---|---|---|
| **H3** — all-linear tốt hơn attention-only ở cùng ngân sách | B3 → A2 | EM(A2) > EM(B3) **và** p < 0,05 → **ỦNG HỘ H3** · EM(A2) < EM(B3) và p < 0,05 → **BÁC BỎ H3** (attention-only tốt hơn) · p ≥ 0,05 → **H3 KHÔNG ĐƯỢC ỦNG HỘ** (không khác biệt có ý nghĩa) |
| **H4** — bão hoà ở r ≤ 16; r = 64 không hơn quá 1 điểm | B3 → A3 | ΔEM > +1 điểm **và** p < 0,05 → **BÁC BỎ H4** · p ≥ 0,05 → **ỦNG HỘ H4** · A3 kém hơn và p < 0,05 → **ỦNG HỘ H4 (r = 64 kém hơn)** · p < 0,05 và 0 < ΔEM ≤ 1 → **ỦNG HỘ H4 (tăng ≤ 1 điểm)** |

- **Độ mạnh của "ủng hộ H4":** cận trên CI95 ΔEM < +1 điểm → *mạnh*; còn chứa giá trị > +1 → *yếu* (test2000 chưa đủ nhạy để loại trừ mức tăng 1 điểm).
- "Không khác biệt có ý nghĩa" nghĩa là **nhỏ hơn mức đo được** (≈ 1,7 điểm EM trên test2000), không phải "bằng nhau" — báo cáo kèm CI95.
- Báo cáo đủ: EM, ANLS, token-F1 (phụ), chi phí (giờ train, s/step, VRAM đỉnh, dung lượng adapter) của cả 3 run.
- **Không** chạy thêm seed hay cấu hình sau khi thấy kết quả để "câu" kết luận.

## Luật B3-full — OCR-aware QLoRA trên toàn bộ train (ghi TRƯỚC khi chạy, 19/10/2026)

Run **B3-full**: công thức B3 (attention r = 16, α = 32, OCR VietOCR · v1 · 600 · 0, lr 2e-4, 1 epoch, seed 42, L4) trên **toàn bộ** `data/vitextvqa_official/train.json` (35.159 câu, 11.733 ảnh, sha256 `dec0acc1ca99…` — cùng file với B2 W07). Eval full test 10.028 câu, có OCR.

0. **Điều kiện trước khi train:** OCR cache phủ đủ 11.733 ảnh train; dấu vân tay của các ảnh đã có trong cache **giữ nguyên** sau khi OCR thêm (như luật full test W09).
1. **So sánh chính:** B2 W07 (full train, không OCR, full test EM 50,26) → B3-full, ghép cặp trên 10.028 câu:
   - EM tăng **và** McNemar exact p < 0,05 → **OCR-AWARE VƯỢT B2 KHI ĐỦ DỮ LIỆU**
   - EM giảm **và** p < 0,05 → **OCR-AWARE KÉM B2 KHI ĐỦ DỮ LIỆU**
   - còn lại → **NGANG B2 KHI ĐỦ DỮ LIỆU** (đóng góp của OCR-aware khi đó chỉ là hiệu quả dữ liệu — ghi đúng như vậy vào thesis)
   - báo kèm ΔANLS + CI95 bootstrap.
2. **So với ViTextBLIP-2** (Table 6, Nguyen et al., *ESWA* 2025: EM 25,48 · F1 53,95): chỉ viết "cao hơn" khi cận dưới CI95 bootstrap của B3-full lớn hơn số công bố (như luật W09).
3. **Chỉ báo cáo** (không kết luận riêng): bảng 2 × 2 *lượng dữ liệu (23% / 100%) × OCR (không / có)* gồm B2-sub, B3, B2 W07, B3-full trên full test — lợi ích của OCR thay đổi thế nào khi tăng dữ liệu.
4. **Giới hạn đã biết:** B2 W07 train bằng code commit W07; B3-full train bằng code W10. Prompt zero-shot và cách train không đổi giữa hai commit (W08 chỉ thêm dòng OCR), nhưng vẫn ghi là giới hạn.

## Thay đổi code

| File | Thay đổi | Kiểm |
|------|----------|------|
| `src/adaptation/lora_targets.py` (mới) | Preset `attn` (list cũ, giữ nguyên) và `all_linear_llm` (regex chỉ LLM); `audit_lora_modules` đếm lớp LoRA theo khối (LLM / vision) và hậu tố, đếm tham số LoRA từ `lora_A/lora_B`; `expected_lora_params` tính tay theo kích thước Qwen2.5-VL-3B | `--selftest`: 7.372.800 khớp log W07–W09; regex đúng 252 lớp LLM, 0 lớp vision trên bố cục tên transformers 5.x **và** bố cục cũ; list hậu tố MLP ngây thơ bị chặn (lọt 96 lớp vision) |
| `src/adaptation/train_qlora.py` | `--lora_r`, `--lora_alpha`, `--lora_targets`; gọi `audit_lora_modules` ngay sau khi nạp model (dừng nếu sai); ghi `lora_audit` vào `train_log.json` | `--selftest` thêm `lora override OK`: không truyền → chữ ký W07–W09 giữ nguyên; `attn` r16 trùng chữ ký cũ; r64 → α128; all-linear r4 → regex + α8 |
| `src/eval/lora_ablation.py` (mới) | Luật H3/H4, CI95 ΔEM, kiểm "chỉ khác LoRA" | `--selftest` |
| `notebooks/17_lora_ablation_rank_modules_qwen25vl.ipynb` (mới) | A2 → A3 → kết luận; bỏ qua phần đã xong; tự trả máy L4 | — |
| `notebooks/10_ocr_cache_vitextvqa.ipynb` | Thêm **Phần D** (mục 13–14): OCR phần ảnh còn thiếu của `train.json`; cache phải trùng sha256 biên bản W09 trước khi mở rộng, dấu vân tay 6.289 ảnh cũ phải giữ nguyên; biên bản `experiments/W10_ocr_cache_fulltrain/` | — |
| `src/eval/b3full.py` (mới) | Luật B3-full (B2 → B3-full) + bảng 2×2 dữ liệu × OCR, 4 hiệu ứng đơn, tương tác theo câu (chỉ báo cáo); ghép cặp theo `question_id` | `--selftest`; chạy thử trên predictions thật W07/W09 (B3-full tạm thay bằng B2): B2-sub → B3 = 417 / 672, p = 1,06 × 10⁻¹⁴, ΔANLS CI95 [+0,0106; +0,0208] — **khớp** `W09_fulltest_B3_vs_B2sub_L4_seed42`; B3 → B2 = +1,16 EM, p = 0,0025 — **khớp** đính chính W09; ≈ 3 phút CPU |
| `notebooks/18_qlora_ocr_b3full_vs_b2_qwen25vl.ipynb` (mới) | Kiểm trước: `train.json` trùng sha256 file B2 W07, OCR cache trùng biên bản Phần D và phủ đủ train + test; train → kiểm `train_log` (35.159 câu, r16/α32/attention, LoRA chỉ LLM, OCR khớp, siêu tham số = B2 W07) → eval full test → kết luận; tự trả máy L4 | — |

Run cũ không bị ảnh hưởng: audit chỉ chạy khi train mới; với cấu hình attention r16, audit kỳ vọng đúng 144 lớp LLM và 7.372.800 tham số (như mọi run W07–W09).

## Ngân sách

| Việc | L4 | T4 | CU (ước tính, L4 = 1,54 CU/h) |
|---|---|---|---|
| A2 + A3 | ≈ 7,6 h | — | ≈ 12 |
| OCR train còn thiếu | — | ≈ 2,5 h | ≈ 3 |
| B3-full train + eval full test | ≈ 16 h | — | ≈ 25 |
| **Tổng W10 (ước tính trước)** | ≈ 23,6 h | ≈ 2,5 h | **≈ 40** (còn ≈ 84 trước W10 → ≈ 44 cho W11–W15) |
| **Thực tế** | ≈ 23,8 h (A2 3,51 + A3 2,90 train, 2 eval ≈ 1,2 h; B3-full 12,75 train + 2,78 eval; nạp/kiểm ≈ 0,7 h) | ≈ 2,1 h | số CU thật: đọc ở Colab → Resources khi đóng tuần (trước W10: 86,52 CU lúc 19/10 23:49) |

## Kết quả

### Ablation LoRA — H3 và H4 (20/10/2026, L4, bf16, 4-bit, seed 42, test2000)

Nguồn: `../../experiments/W10_lora_ablation_test2000_L4_seed42/` (`lora_ablation_summary.json`, `paired_B3_vs_A2.json`, `paired_B3_vs_A3.json`); train `W10_A2_qlora_ocr_vietocr_v1_alllinear_r4_…`, `W10_A3_qlora_ocr_vietocr_v1_r64_…`; eval `W10_A2_eval_…`, `W10_A3_eval_…`. Mốc B3 = run W08 seed 42. `lora_ablation.py` đã kiểm A2, A3 chỉ khác B3 ở mục LoRA.

| Run | Lớp gắn LoRA | r / α | Tham số train | EM | ΔEM so với B3 (CI95) | Chỉ B3 đúng / chỉ biến thể đúng | McNemar p | ΔANLS (CI95) |
|---|---|---|---|---|---|---|---|---|
| B3 (mốc) | q, k, v, o | 16 / 32 | 7.372.800 (0,196%) | 0,4925 | — | — | — | — |
| A2 | q, k, v, o, gate, up, down (252 lớp, chỉ LLM) | 4 / 8 | 7.483.392 (0,199%) | 0,4870 | −0,55 [−1,60; +0,45] | 60 / 49 | 0,34 | −0,0025 [−0,0116; +0,0068] |
| A3 | q, k, v, o (144 lớp, chỉ LLM) | 64 / 128 | 29.491.200 (0,779%) | 0,5030 | +1,05 [−0,05; +2,15] | 51 / 72 | 0,071 | +0,0049 [−0,0040; +0,0140] |

**Chi phí** (cùng L4, cùng 500 step):

| Run | Giờ train | s/step | VRAM đỉnh | Loss cuối | Adapter |
|---|---|---|---|---|---|
| B3 | 2,98 | 21,48 | 8,25 GB | 0,6605 | 28,2 MB |
| A2 | 3,51 | 25,24 (+17,5%) | 8,26 GB | 0,6495 | 28,6 MB |
| A3 | 2,90 | 20,85 | 8,58 GB | 0,6380 | 112,5 MB (×4) |

**Kết luận theo luật ghi trước:**
- **H3 — KHÔNG ĐƯỢC ỦNG HỘ** (không khác biệt có ý nghĩa). Ở cùng ngân sách tham số, gắn LoRA vào cả MLP không tốt hơn attention-only (ΔEM −0,55, CI95 [−1,60; +0,45]) nhưng train chậm hơn 17,5%. → Lựa chọn attention-only của B3 được giữ: **rẻ hơn, chất lượng không kém**. QLoRA paper (Dettmers et al., NeurIPS 2023) khuyên gắn LoRA vào mọi lớp tuyến tính, nhưng chính paper cũng nêu yếu tố quan trọng nhất là **tổng số tham số LoRA** — ở đây tổng đó được giữ bằng nhau (lệch 1,5%), và paper đo trên LLM thuần văn bản, không phải VLM tiếng Việt.
- **H4 — ỦNG HỘ (yếu)**: r = 64 không khác biệt có ý nghĩa so với r = 16 (p = 0,071). *Yếu* vì cận trên CI95 ΔEM (+2,15) còn vượt +1 điểm: test2000 chưa đủ nhạy để loại trừ mức tăng 1–2 điểm. r = 64 tốn adapter ×4 và VRAM +0,33 GB.

**Giới hạn và ghi chú trung thực:**
- 1 seed huấn luyện. EM của A3 (0,5030) cao hơn cả 3 seed B3 (0,4880–0,4925, std 0,23 điểm — W09), nhưng đây là quan sát **sau khi thấy số**, không nằm trong luật; theo luật không chạy thêm seed để đổi kết luận. Ghi r = 64 là **hướng cải tiến tiếp** (cần ≥ 3 seed hoặc full test để kết luận).
- B3-full vẫn dùng attention r = 16 (quyết định 19/10, chốt trước khi có kết quả ablation).

### OCR toàn bộ train — notebook 10 Phần D (20/10/2026, T4, GPU)

Nguồn: `../../experiments/W10_ocr_cache_fulltrain/ocr_cache_manifest.json`. Cache trên Drive trùng sha256 biên bản W09 trước khi mở rộng (6.289 ảnh) → OCR thêm **9.063 ảnh, 0 lỗi**, 99,6% ảnh mới có chữ, 0,72 s/ảnh → **15.352 ảnh** (sha256 `e7ce4f76d98c…`). Dấu vân tay 6.289 ảnh cũ **giữ nguyên**; engine trùng đợt W08/W09 (PaddleOCR 3.7.0 · paddlepaddle 3.3.1 · PP-OCRv6_medium_det + VietOCR vgg_transformer). Phủ đủ train 11.733/11.733 và full test 3.353/3.353 ảnh.

### B3-full — QLoRA nhận biết OCR trên toàn bộ train (21/10/2026, L4, bf16, 4-bit, seed 42, full test 10.028 câu)

Nguồn: `../../experiments/W10_B3full_vs_B2_testfull_L4_seed42/b3full_summary.json`; train `W10_B3full_qlora_ocr_vietocr_v1_r16_qwen25vl_vitextvqa_full_ep1_bf16_L4_seed42`; eval `W10_B3full_eval_qlora_ocr_vietocr_v1_r16_full_ep1_vitextvqa_testfull_bf16_L4_seed42`. Các run so sánh: B2 W07, B2-sub và B3 W09 (full test).

**Kiểm hợp lệ (notebook 18 tự kiểm, đều đạt):** `train.json` trùng sha256 file B2 W07; 35.159 câu, loại 0; r 16 / α 32 / attention, LoRA audit 144 lớp chỉ LLM, 7.372.800 tham số; OCR VietOCR · v1 · 600 · 0 (99,77% mẫu train có chữ); lr, epoch, batch, warmup trùng B2 W07; train chạy liền một phiên (không resume).

| Run | Dữ liệu train | OCR | EM | ANLS | F1 (phụ) | s/câu |
|---|---|---|---|---|---|---|
| B2-sub (W09) | 8.000 câu (23%) | không | 46,56 | 68,90 | 67,83 | 0,983 |
| B3 (W09) | 8.000 câu (23%) | có | 49,10 | 70,47 | 70,03 | 0,983 |
| B2 (W07) | 35.159 câu (100%) | không | 50,26 | 72,02 | 71,16 | 1,001 |
| **B3-full** | 35.159 câu (100%) | có | **53,35** | **73,89** | **73,73** | 0,996 |
| ViTextBLIP-2 (paper, Table 6) | 100% | SwinTextSpotter | 25,48 | — | 53,95 | — |

| Hiệu ứng (ghép cặp 10.028 câu) | So sánh | Chỉ A đúng / chỉ B đúng | McNemar p | ΔEM (điểm) | ΔANLS (CI95) |
|---|---|---|---|---|---|
| OCR-aware, 23% dữ liệu (W09) | B2-sub → B3 | 417 / 672 | 1,1 × 10⁻¹⁴ | +2,54 | +0,0157 [+0,0106; +0,0208] |
| **OCR-aware, 100% dữ liệu — SO SÁNH CHÍNH** | **B2 → B3-full** | **395 / 705** | **6,7 × 10⁻²¹** | **+3,09** | **+0,0187 [+0,0138; +0,0236]** |
| Thêm dữ liệu, không OCR | B2-sub → B2 | 355 / 726 | 6,1 × 10⁻³⁰ | +3,70 | +0,0312 [+0,0260; +0,0365] |
| Thêm dữ liệu, có OCR | B3 → B3-full | 341 / 767 | 3,0 × 10⁻³⁸ | +4,25 | +0,0343 [+0,0289; +0,0397] |

Tương tác dữ liệu × OCR (theo câu, chỉ báo cáo): EM +0,0055 [−0,0020; +0,0129] · ANLS +0,0031 [−0,0030; +0,0092].

So với ViTextBLIP-2 (cận dưới CI95 bootstrap > số công bố): EM 53,35 [52,39; 54,33] > 25,48 · F1 73,73 [73,02; 74,45] > 53,95 → **CAO HƠN** cả EM lẫn F1.

| Chi phí | Giờ train | s/step | VRAM đỉnh | Loss cuối | Adapter |
|---|---|---|---|---|---|
| B2 (W07) | 11,99 | 19,63 | 6,88 GB | 0,6179 | cùng cỡ (cùng r16 · attention) |
| B3-full | 12,75 (+6,3%) | 20,89 | 8,38 GB (+1,5) | 0,5598 | 29,53 MB |

Suy luận full test: 9.991 s (0,996 s/câu, chưa tính OCR; OCR chạy trước một lần và cache, 0,72–0,85 s/ảnh trên GPU).

**Kết luận theo luật ghi trước: OCR-AWARE VƯỢT B2 KHI ĐỦ DỮ LIỆU.**

**Diễn giải:**
- **Đính chính W09.** W09 chỉ nói được "hiệu quả dữ liệu" (với 23% dữ liệu, OCR bù ≈ 69% khoảng cách). Nay: **cùng dữ liệu, cùng siêu tham số, chỉ khác OCR**, QLoRA nhận biết OCR hơn QLoRA thường +3,09 EM (p = 6,7 × 10⁻²¹) — cải tiến thật so với baseline đã reproduce.
- **Lợi ích của OCR không mất đi khi thêm dữ liệu** (+2,54 ở 23% → +3,09 ở 100%; tương tác +0,55, CI95 chứa 0 → cộng gộp). Lo ngại "đủ dữ liệu thì model tự đọc chữ tốt, OCR thành thừa" không xảy ra trong phạm vi đo.
- Hai yếu tố cộng dồn: B2-sub → B3-full = +6,79 EM (46,56 → 53,35).
- **Đối chiếu H2** (số mới: OCR tăng ANLS ≥ +3 điểm và cộng hưởng với QLoRA): EM tăng mạnh, nhưng ΔANLS = +1,87 điểm, **vẫn dưới ngưỡng +3** (trên ngưỡng bác bỏ +1) → **H2 vẫn ủng hộ một phần**, như kết luận Review 2. Lợi ích chủ yếu ở việc trả lời **đúng nguyên chuỗi** (EM) hơn là gần đúng ký tự (ANLS).
- **Chi phí của cải tiến nhỏ:** +6% thời gian train, +1,5 GB VRAM (prompt dài hơn ≈ 140 ký tự OCR), tốc độ suy luận như cũ; cộng bước OCR chạy trước.

**Giới hạn:**
- B3-full 1 seed (42). Ở tập con 8k, độ lệch chuẩn EM của B3 qua 3 seed là 0,23 điểm và lợi ích của OCR ổn định 3/3 seed (W09) — nhưng chưa đo lại ở 100% dữ liệu.
- B2 W07 train bằng code commit `22b44a3`, B3-full bằng code W10. Giữa hai commit: `configs/qwen_lora.yaml` không đổi; `src/prompting/builder.py` chỉ thêm nhánh OCR; `train_qlora.py` thêm seed / OCR / resume / LoRA audit (không đổi phép tính khi không bật). B2-sub (code W08) và B2 W07 cùng prompt zero-shot.
- ViTextBLIP-2: paper không công bố cách chuẩn hoá đáp án → so sánh tuyệt đối mang tính tham khảo; khoảng cách (+27,9 EM) lớn hơn nhiều mọi khác biệt chuẩn hoá hợp lý.

## Link experiment log

- Ablation LoRA (A1, A2): `../../experiments/W10_lora_ablation_test2000_L4_seed42/`, `../../experiments/W10_A2_*`, `../../experiments/W10_A3_*`
- OCR cache toàn bộ train: `../../experiments/W10_ocr_cache_fulltrain/ocr_cache_manifest.json`
- B3-full: `../../experiments/W10_B3full_vs_B2_testfull_L4_seed42/` + train `W10_B3full_qlora_…` + eval `W10_B3full_eval_…`
- Adapter (không lên git, trên Drive): `adapters/W10_A2_*`, `adapters/W10_A3_*`, `adapters/W10_B3full_*` (`epoch_1`)

## Deliverables tuần này

| File | Vai trò |
|------|---------|
| [W10_Team_Report.md](W10_Team_Report.md) | Báo cáo tuần: việc đã làm, số liệu chính, blocker, kế hoạch W11 |
| README này | Quyết định, luật ghi trước, kết quả từng bước |
| [../03_report/R6_Results.md](../03_report/R6_Results.md) | R6 bản nháp tiếng Việt: kết quả chính + ablation QLoRA nhận biết OCR |
| `src/adaptation/lora_targets.py`, `src/eval/lora_ablation.py`, `src/eval/b3full.py` | Code mới, có selftest |
| Notebook 17, 18; notebook 10 Phần D | Chạy lại được toàn bộ W10 |

## Trạng thái

✅ **Đóng tuần 21/10/2026.** Kết quả chính: **QLoRA nhận biết OCR vượt QLoRA thường khi đủ dữ liệu** (B3-full 53,35 so với B2 50,26 EM, +3,09, p = 6,7 × 10⁻²¹; ANLS +1,87) — từ "hiệu quả dữ liệu" (W09) thành cải tiến thật; lợi ích OCR không giảm khi thêm dữ liệu. Ablation cấu hình LoRA: A1 (module) — all-linear không tốt hơn attention-only ở cùng ngân sách nhưng chậm hơn 17,5%; A2 (rank) — r = 64 không khác biệt có ý nghĩa (yếu: CI95 chạm +2,15). H2 vẫn ủng hộ một phần (ANLS +1,87 < +3).
