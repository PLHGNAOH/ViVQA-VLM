# Week 10 — Ablation cấu hình LoRA + B3 trên toàn bộ dữ liệu train

Index tuần 10 (tuần ảo) của ViVQA-VLM. Tuần ảo W10 chạy trong **tuần thực 7** (19–25/10/2026). Phần Review 2 của tuần thực 7 theo dõi riêng (checklist AI_Capstone_Review123), không nằm trong README này.

Mục tiêu W10: trả lời **hai câu hỏi còn mở** về công thức B3 (OCR-aware QLoRA) trước khi viết phần Results:

1. **Cấu hình LoRA đã hợp lý chưa?** — B3 dùng r = 16, chỉ 4 lớp attention; chưa từng kiểm (H3, H4).
2. **Đủ dữ liệu thì OCR-aware có vượt B2 không?** — W09: B3 (23% dữ liệu) thấp hơn B2 dữ liệu đầy đủ 1,16 EM trên full test.

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Kế hoạch tuần

| # | Việc | GPU | Trạng thái |
|---|------|-----|-----------|
| 0 | Code: `src/adaptation/lora_targets.py` (preset + chặn LoRA lọt vision), `--lora_r / --lora_alpha / --lora_targets` cho `train_qlora.py`, LoRA audit trong `train_log.json`, `src/eval/lora_ablation.py`, notebook `17_lora_ablation_rank_modules_qwen25vl.ipynb`, luật ghi trước (README này) | — | ⏳ |
| 1 | **A2** — all-linear r = 4 (cùng ngân sách tham số với B3), train + eval test2000 | ≈ 3,9 h L4 | ⏳ |
| 2 | **A3** — attention r = 64, train + eval test2000 | ≈ 3,7 h L4 | ⏳ |
| 3 | OCR cho phần ảnh train còn thiếu (9.063 ảnh) — notebook 10 phần mới | ≈ 2,5 h T4 | ⏳ |
| 4 | **B3-full** — QLoRA + OCR trên toàn bộ train (35.159 câu), eval full test 10.028 câu | ≈ 13 h + 2,8 h L4 | ⏳ |
| 5 | Results (R6) bản tiếng Việt — phần OCR-aware QLoRA (số liệu W08–W09) | — | ⏳ |
| 6 | Đóng tuần: `W10_Team_Report.md`, dọn Downloads + Drive, handoff | — | ⏳ |

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

Run cũ không bị ảnh hưởng: audit chỉ chạy khi train mới; với cấu hình attention r16, audit kỳ vọng đúng 144 lớp LLM và 7.372.800 tham số (như mọi run W07–W09).

## Ngân sách

| Việc | L4 | T4 | CU (ước tính, L4 = 1,54 CU/h) |
|---|---|---|---|
| A2 + A3 | ≈ 7,6 h | — | ≈ 12 |
| OCR train còn thiếu | — | ≈ 2,5 h | ≈ 3 |
| B3-full train + eval full test | ≈ 16 h | — | ≈ 25 |
| **Tổng W10** | ≈ 23,6 h | ≈ 2,5 h | **≈ 40** (còn ≈ 84 trước W10 → ≈ 44 cho W11–W15) |

## Kết quả

*(chưa chạy)*

## Link experiment log

- Ablation LoRA: `../../experiments/W10_lora_ablation_test2000_L4_seed42/` (sau khi chạy)
