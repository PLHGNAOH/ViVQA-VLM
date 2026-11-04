# R4 — Phương pháp (Methodology)

> **Trạng thái:** bản tiếng Việt, viết cuối W6 (2026-09-29). Sẽ chuyển sang tiếng Anh khi nội dung đã chốt và không còn lỗi.
> **Ký hiệu:** ⏳ W7 = quyết định còn mở, chờ đo trên GPU L4 ở tuần 7. Được ghi tường minh kèm lý do, không viết như thể đã chốt.
> **Nguồn số liệu:** mọi con số trong tài liệu này truy được về một file trong `experiments/` hoặc `src/` của repo, ghi ở cuối mỗi mục.

---

## 1. Tổng quan

Đề tài kiểm chứng một giả thuyết kỹ thuật: **có thể thích nghi một VLM mã nguồn mở cho hỏi đáp trực quan tiếng Việt bằng tinh chỉnh hiệu quả tham số (PEFT), dưới ngân sách phần cứng của Colab**, và đo được cải thiện so với baseline zero-shot dưới điều kiện y hệt.

Quy trình gồm năm bước, mỗi bước chỉ bắt đầu khi bước trước có số liệu tái lập được:

```text
[1] Chuẩn bị dữ liệu        ViVQA + ViTextVQA (nguồn chính thức), split cố định, seed 42
        ↓
[2] Reproduce baseline      Qwen2.5-VL-3B-Instruct, zero-shot, 4-bit NF4
        ↓
[3] Cải tiến (PEFT-only)    QLoRA trên decoder ngôn ngữ  (+ OCR-prompting ở W8)
        ↓
[4] Đánh giá                EM · VQA-Acc · ANLS · kiểm định bắt cặp · ablation · error/hallucination
        ↓
[5] Prototype web           upload ảnh + hỏi tiếng Việt + hiển thị confidence / OCR
```

**Ràng buộc cứng (không thương lượng):**

- **Chỉ PEFT.** Không full fine-tuning. Trọng số gốc bị đóng băng và lượng tử hoá 4-bit; chỉ LoRA adapter được học.
- **Đúng một baseline**, reproduce trước mọi cải tiến.
- **So sánh dưới điều kiện y hệt:** cùng split, seed, prompt, độ phân giải, dtype, phần cứng.
- **Chạy được trên Colab.** Hiệu quả là một ràng buộc thiết kế tường minh, không phải hệ quả phụ.

---

## 2. Dữ liệu và phân chia

### 2.1. Hai bộ dữ liệu

| | ViVQA (Tran et al., PACLIC 2021) | ViTextVQA (Nguyen et al., 2024) |
|---|---|---|
| Loại câu hỏi | Tổng quát | Scene-text (đọc chữ trong ảnh) |
| Nguồn dùng | CSV gốc trên GitHub `kh4nh12/ViVQA` + ảnh COCO | Kho HuggingFace **chính thức, có kiểm soát truy cập** `minhquan6203/ViTextVQA` |
| Số đáp án mỗi câu | 1 | Có thể nhiều |

**Hai quyết định về nguồn dữ liệu, và lý do:**

1. **Loại bỏ ViVQA-X.** Dữ liệu cục bộ ban đầu là ViVQA-X — một bộ khác với bộ đề cương yêu cầu. Phát hiện ở W5, thay bằng ViVQA gốc.
2. **Bỏ bản mirror ViTextVQA không chính thức**, chuyển sang kho chính thức. Kho này có một cái bẫy: `ViTextVQA_test.json` là **mẫu nộp bài Kaggle** (đáp án đều là chuỗi giữ chỗ), không phải đáp án thật. Đáp án thật nằm ở `ViTextVQA_test_gt.json`. Dùng nhầm file mẫu sẽ cho ra điểm vô nghĩa mà không báo lỗi.

### 2.2. Phân chia (split)

| Tập | ViVQA | ViTextVQA | Dùng để |
|---|---|---|---|
| Train | **10.799** | 35.159 câu / 11.733 ảnh | Huấn luyện QLoRA |
| Val / Dev | 1.200 | 5.155 câu / 1.676 ảnh | Chọn cấu hình, tối ưu prompt |
| Test đầy đủ | 3.001 | 10.028 câu / 3.353 ảnh | — |
| **Test dùng để báo cáo** | **3.001 (đầy đủ)** | **2.000 (tập con, seed 42)** | Con số cuối cùng |
| Dev dùng cho ablation | — | 300 (tập con, seed 42) | Ablation cấu hình W6 |

- **ViVQA không có tập val gốc.** Val được cắt 10% từ train (xáo trộn seed 42): 15.000 − 3.001 = 11.999 câu train gốc → 1.200 val + 10.799 train. Đây là quyết định của đề tài, không phải của tác giả dataset — ghi nhận như một giới hạn.
- **ViTextVQA báo cáo trên tập con 2.000 câu** thay vì 10.028 câu vì chi phí suy luận (~1,35 s/câu trên T4 ở cấu hình baseline → tập đầy đủ ~3,8 giờ). Danh sách ID của tập con được lưu (`test_subset2000_seed42_ids.json`) nên tập con là cố định, không lấy lại ngẫu nhiên giữa các lần chạy.
- **Nguyên tắc tách vai trò:** mọi lựa chọn cấu hình (dtype, độ phân giải, prompt) được thực hiện trên **dev**, không bao giờ trên test. Test chỉ dùng để báo cáo. Chọn cấu hình trên test sẽ làm con số báo cáo bị thổi phồng.

### 2.3. Chuẩn hoá văn bản tiếng Việt

- **Unicode NFC** cho cả câu hỏi, dự đoán và đáp án. Tiếng Việt có hai cách mã hoá cùng một ký tự có dấu (dựng sẵn và tổ hợp); không chuẩn hoá thì hai chuỗi trông giống hệt nhau vẫn bị chấm khác nhau.
- Chữ thường, gọn khoảng trắng.
- **Chính sách chọn đáp án khi huấn luyện:** lấy đáp án **đầu tiên** trong danh sách. Ghi vào log mỗi lần chạy (`answer_policy`).

*Nguồn:* `src/data/prepare_vivqa.py`, `src/data/prepare_vitextvqa.py`, `src/eval/metrics.py` (`normalize_vi`), `docs/W05/`.

---

## 3. Baseline

### 3.1. Mô hình và cấu hình

| Thành phần | Giá trị |
|---|---|
| Mô hình | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Lượng tử hoá | 4-bit NF4, double quantization |
| Độ phân giải ảnh | `min_pixels` 200.704 (256 token) · `max_pixels` 401.408 (512 token) |
| Giải mã | Greedy (`do_sample=False`), `max_new_tokens` 32 |
| Prompt | Zero-shot; system prompt tiếng Việt yêu cầu trả lời ngắn gọn, chỉ đưa đáp án |
| Thư viện | `transformers` 5.17.0 (cố định phiên bản) |
| Phần cứng | Tesla T4 16 GB, khả dụng ~14,6 GiB (Colab) |

**Lý do chọn Qwen2.5-VL-3B:** nằm trong danh sách baseline được phép, công cụ hỗ trợ trưởng thành, đủ nhẹ để chạy 4-bit trên T4, yêu cầu `transformers` thấp hơn các bản mới → tái lập chắc chắn hơn. Qwen3-VL-4B được giữ làm **mô hình tham chiếu** (chỉ zero-shot, không huấn luyện) để tránh mở rộng phạm vi thành hai baseline song song.

### 3.2. "Reproduce" nghĩa là gì trong đề tài này

Paper gốc của Qwen2.5-VL **không báo cáo** trên ViVQA hay ViTextVQA. Nên "reproduce baseline" ở đây không có nghĩa là tái lập một con số đã công bố, mà là: **chạy mô hình pretrained ở chế độ zero-shot dưới một giao thức cố định, log đầy đủ, để tạo ra con số tham chiếu** mà mọi cải tiến sau này đo hơn/kém so với nó.

### 3.3. Kết quả baseline (W5)

| Dataset | n | EM | ANLS |
|---|---|---|---|
| ViVQA test | 3.001 | **0,225** | **0,288** |
| ViTextVQA test (tập con) | 2.000 | **0,285** | **0,468** |

*Tốc độ:* 1,351 s/câu trên ViTextVQA (T4, 4-bit, fp16).

**Không so sánh trực tiếp EM giữa hai dataset.** ViTextVQA có EM cao hơn dù khó hơn, vì đáp án thường chính là dòng chữ trong ảnh — model đọc được thì dễ khớp tuyệt đối; ViVQA đòi đúng từ vựng của đáp án gốc.

### 3.4. Phân tích lỗi baseline (định hướng cho cải tiến)

| Hiện tượng | Bằng chứng | Gap liên quan |
|---|---|---|
| **Chữ tiếng Việt có dấu khó hơn chữ Latin** | Trên ViTextVQA: EM 0,394 (đáp án Latin) so với 0,224 (đáp án tiếng Việt có dấu) | G1, G2 |
| Chuyển sang tiếng Anh (code-switching) | ~90 dự đoán trả lời bằng tiếng Anh | G1 |
| Lệch phong cách | Đáp án chuẩn thường bỏ loại từ (`con`, `chiếc`); model thì giữ | — |
| Thiên lệch màu | Dự đoán `đỏ` 117 lần, trong khi đáp án đúng chỉ có 75 lần | G4 |
| **Output suy biến** | 23 dự đoán toàn ký tự `!` — xem mục 5.1 | — |

⏳ W7: **baseline sẽ được chạy lại** dưới cấu hình cuối cùng (xem mục 5.4), vì cấu hình W5 dùng fp16 — đã chứng minh gây lỗi ở mục 5.1.

*Nguồn:* `experiments/W05_zeroshot_qwen25vl_vivqa_test_seed42/`, `experiments/W05_zeroshot_qwen25vl_vitextvqa_test2000_seed42/`, `docs/W05/W05_Baseline_Report.md`.

---

## 4. Phương pháp đề xuất: QLoRA

### 4.1. Cấu hình adapter

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Rank `r` | 16 | |
| `lora_alpha` | 32 | = 2r, theo thông lệ |
| `lora_dropout` | 0,05 | |
| Module đích | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention của decoder ngôn ngữ |
| Trọng số gốc | Đóng băng, 4-bit NF4 | |

### 4.2. Bằng chứng PEFT-only — đo được, không chỉ tuyên bố

```
trainable params: 7,372,800 || all params: 3,761,995,776 || trainable%: 0.1960
```

**0,196%** tham số được huấn luyện. Con số này còn **kiểm chứng được bằng phép tính**, cho biết adapter nằm ở đâu:

| Module | Kích thước (vào → ra) | Tham số LoRA = r × (vào + ra) |
|---|---|---|
| `q_proj` | 2048 → 2048 | 16 × 4096 = 65.536 |
| `k_proj` | 2048 → 256 | 16 × 2304 = 36.864 |
| `v_proj` | 2048 → 256 | 16 × 2304 = 36.864 |
| `o_proj` | 2048 → 2048 | 16 × 4096 = 65.536 |
| **Mỗi lớp decoder** | | **204.800** |
| **× 36 lớp** | | **7.372.800** ✔ |

Khớp chính xác với số PEFT báo cáo. Tức **toàn bộ adapter nằm trong decoder ngôn ngữ**; vision encoder và lớp nối ảnh–chữ không bị chạm tới. (Các lớp attention của vision encoder dùng tên module khác nên không khớp danh sách đích.)

> **Lưu ý về cách đếm:** trọng số 4-bit được đóng gói hai giá trị trong một byte, nên đếm bằng `p.numel()` cho ra **thiếu một nửa** (2,04 tỉ thay vì 3,76 tỉ → tỉ lệ sai 0,361%). Script dùng hàm đếm của PEFT, có xử lý đóng gói 4-bit, và ghi phương pháp đếm vào log (`param_count_method`). Lỗi này được phát hiện và sửa ở W6; run smoke đầu tiên còn mang số đếm sai và được chú thích rõ trong `smoke_summary.json`.

### 4.3. Che nhãn: chỉ tính loss trên đáp án

Mỗi mẫu huấn luyện là một chuỗi:

```text
[system] [ảnh → ~512 token] Câu hỏi: ... | <đáp án><|im_end|>
└──────────── nhãn = −100 (bỏ qua) ─────┘ └──── tính loss ───┘
```

**Vì sao bắt buộc:** mô hình sinh văn bản học bằng cách đoán token kế tiếp trên toàn chuỗi. Không che thì phần lớn tín hiệu học dồn vào việc sinh lại câu hỏi và token ảnh — hơn 500 token so với vài token đáp án. Loss vẫn giảm đẹp nhưng chất lượng trả lời không lên: một lỗi không tự báo.

**Ba chi tiết triển khai:**

1. **Độ dài prompt phụ thuộc ảnh.** Mỗi ảnh nở ra số token khác nhau theo độ phân giải, nên không thể đoán vị trí che bằng độ dài chữ. Độ dài prompt được đo riêng cho từng mẫu bằng cách cho processor xử lý phần prompt **cùng đúng ảnh đó**.
2. **Đệm bên phải** (`padding_side = "right"`), đặt tường minh — để prompt luôn nằm đầu chuỗi và vị trí che đúng.
3. **Đáp án kèm token kết thúc lượt `<|im_end|>`.** Không có nó, mô hình không học được khi nào dừng — nguyên nhân của lỗi trả lời lan man ở baseline.

**Prompt lúc huấn luyện trùng khít prompt lúc đánh giá.** Lệch prompt giữa hai giai đoạn là lệch phân phối, làm kết quả mất ý nghĩa.

*Nguồn:* `src/adaptation/train_qlora.py` (`build_labels`, `make_collate_fn`), `src/models/vlm_loader.py`, `configs/qwen_lora.yaml`.

---

## 5. Thiết lập huấn luyện và ba phát hiện thực nghiệm (W6)

Trước khi huấn luyện thật, W6 chạy ba thí nghiệm cấu hình. Mỗi thí nghiệm **cố định tiêu chí kết luận trước khi xem số**, để tránh diễn giải kết quả theo hướng có lợi.

### 5.1. Phát hiện 1 — Tràn số fp16 gây output suy biến

**Hiện tượng:** 23/2.000 dự đoán baseline ViTextVQA là chuỗi `!` dài đúng 32 ký tự.

**Chẩn đoán:** ba dấu hiệu cùng chỉ về tràn số:

- Độ dài bằng đúng `max_new_tokens = 32` → mô hình sinh cùng một token đến hết hạn mức, không bao giờ kết thúc.
- `!` là **token id 0** trong tokenizer Qwen2.
- Khi logits thành `NaN`/`Inf`, phép chọn giá trị lớn nhất rơi về chỉ số 0 ở mọi bước.

fp16 chỉ biểu diễn được tới 65.504; giá trị trung gian vượt ngưỡng thành vô cực.

**Thí nghiệm A/B:** chạy lại đúng 23 câu hỏng + **50 câu đối chứng** (chọn ngẫu nhiên seed 42 từ phần còn lại), chỉ đổi kiểu số khi tính toán:

| | fp16 | fp32 |
|---|---|---|
| Câu suy biến | **23** | **0** |
| Nhóm 23 câu hỏng — EM / ANLS | 0,000 / 0,000 | 0,348 / 0,517 |
| Nhóm 50 đối chứng — EM | 0,280 | 0,280 |
| Câu thoái lui (đúng → sai) | — | **0** |
| Thời gian | 1,965 s/câu | 2,739 s/câu (**1,39×**) |

**Kết luận:** tràn số fp16 là nguyên nhân, đã xác nhận nhân quả bằng thí nghiệm một biến. fp32 phục hồi 8/23 câu về đúng hoàn toàn, **không làm hỏng câu nào vốn đúng** (EM đối chứng trùng khít).

**Hai lưu ý diễn giải:**

- **Không trình bày đây là cải tiến điểm số.** Tập probe có 31,5% câu hỏng; trên test thật chỉ 1,15%. Tác động ước tính lên test đầy đủ: EM +0,004, ANLS +0,006. Giá trị của phát hiện nằm ở chỗ khác: gỡ một biến nhiễu khỏi mọi thí nghiệm sau, và **cảnh báo rủi ro khi huấn luyện** — gradient có dải giá trị rộng hơn activation, nên fp16 dễ gây NaN loss, hỏng toàn bộ adapter chứ không chỉ vài câu trả lời.
- **Tràn số là hiện tượng rời rạc, không phải nhiễu liên tục.** EM nhóm đối chứng trùng khít giữa hai dtype cho thấy fp16 hoặc hỏng hoàn toàn, hoặc không ảnh hưởng gì.

**Ràng buộc phần cứng:** giải pháp chuẩn là bf16 (cùng 16 bit nhưng dải rộng như fp32). **T4 không hỗ trợ bf16** (kiến trúc Turing, sm_75; bf16 cần từ Ampere sm_80). Trên T4 chỉ còn fp32.

*Nguồn:* `experiments/W06_dtype_probe_seed42/` (`failed_ids.json`, `probe_summary.json`, `probe_breakdown.json`), `src/eval/find_degraded.py`, `src/eval/probe_breakdown.py`, `notebooks/03_ablation_config_qwen25vl.ipynb`.

### 5.2. Phát hiện 2 — Độ phân giải: đánh đổi, không phải cải thiện đơn thuần

**Câu hỏi:** tăng số token ảnh từ 512 lên 1.280 có giúp đọc chữ tốt hơn không?

**Thiết lập:** 300 câu dev ViTextVQA (seed 42), fp32, 4-bit, chỉ đổi `max_pixels`.

| | A1 — 512 token | A2 — 1.280 token | Chênh |
|---|---|---|---|
| EM | 0,3333 | 0,3600 | +0,0267 |
| ANLS | 0,4654 | 0,5041 | +0,0387 |
| Thời gian | 2,605 s/câu | 3,807 s/câu | **1,46×** |

**Kiểm định bắt cặp** (cùng 300 câu nên so sánh từng câu được):

| | A2 đúng | A2 sai |
|---|---|---|
| **A1 đúng** | 93 | 7 |
| **A1 sai** | 15 | 185 |

- **EM:** McNemar chính xác, b = 7, c = 15 → **p = 0,134**. Không có ý nghĩa thống kê.
- **ANLS:** chênh +0,0387, khoảng tin cậy 95% bootstrap bắt cặp (10.000 lần) **[+0,012; +0,066]**. Không chứa 0 → có ý nghĩa.

**Diễn giải — hai metric không mâu thuẫn:** EM là thang nhị phân, chỉ thấy 22 câu đổi trạng thái đúng/sai — quá ít để kết luận. ANLS là thang liên tục, thấy cải thiện một phần trên cả 300 câu. Kết luận hợp lệ: *độ phân giải cao giúp nhận dạng ký tự chính xác hơn trên toàn tập, nhưng chưa đủ để chuyển thành nhiều câu đúng hoàn toàn hơn.* Nhất quán với phát hiện baseline: rào cản chính là **dấu thanh tiếng Việt**, không phải độ nét.

**Độ phân giải cao cũng có cái giá về chất lượng:** 7 câu A1 đúng mà A2 sai. Giả thuyết: ảnh nét hơn đưa thêm chữ vào tầm nhìn, mô hình chọn nhầm dòng chữ. Chỉ lộ ra nhờ kiểm định bắt cặp; so sánh bằng số tổng sẽ che mất.

**Ràng buộc:** độ phân giải dùng khi huấn luyện phải trùng độ phân giải khi đánh giá. Huấn luyện ở 512 rồi đánh giá ở 1.280 là lệch phân phối.

*Nguồn:* `experiments/W06_ablation_config_dev300_seed42/` (`ablation_summary.json`, `paired_A1_vs_A2.json`), `src/eval/compare_runs.py`.

### 5.3. Phát hiện 3 — Chi phí huấn luyện trên T4

**Thiết lập smoke run:** 200 mẫu ViVQA, 512 token, fp32, 4-bit.

| Cấu hình | Batch hiệu dụng | Gradient checkpointing | s/mẫu | VRAM đỉnh | Kết quả |
|---|---|---|---|---|---|
| Batch 2 × tích luỹ 8 | 16 | Bật | **4,93** | 5,93 GB | ✅ 30 step |
| Batch 6 × tích luỹ 3 | 18 | **Tắt** | — | 14,46 / 14,56 GB | ❌ Hết VRAM |
| Batch 6 × tích luỹ 3 | 18 | Bật | **5,10** | 11,6 GB | ✅ 10 step |

**Đường ray huấn luyện hoạt động đúng:** loss giảm đều 1,61 → 1,03 → 0,88 qua 30 step. Nếu che nhãn sai hoặc LoRA không gắn, loss sẽ đứng im.

**T4 bão hoà tính toán ở fp32:** gấp ba batch, gấp đôi VRAM, tốc độ không đổi (4,93 so với 5,10 s/mẫu; chênh 3% nằm trong nhiễu, và run 10 step chịu chi phí khởi động nặng hơn nên nếu có thì sai lệch làm batch 6 trông chậm hơn thực tế). **Không tắt được gradient checkpointing** trên T4 ở cấu hình này. Tối ưu phần mềm không cứu được; cần đổi GPU.

**Ngân sách thực tế:** 10.799 mẫu × 4,93 s/mẫu ≈ **14,77 giờ / epoch ViVQA** trên T4. Cấu hình mặc định 3 epoch ≈ 44 giờ. Không khả thi trên Colab miễn phí cho 8 ablation. Đây là căn cứ định lượng cho việc chuyển sang GPU L4 (Colab Pro) từ W7.

**Giữ batch 2 làm mặc định:** cùng tốc độ, dùng ít hơn một nửa VRAM, biên an toàn rộng.

*Nguồn:* `experiments/W06_smoke_qlora_seed42/smoke_summary.json`, `experiments/W06_smoke_qlora_qwen25vl_vivqa_mp512*_seed42/train_log.json`, `notebooks/04_qlora_qwen25vl.ipynb`.

### 5.4. Siêu tham số huấn luyện

| Tham số | Giá trị | Trạng thái |
|---|---|---|
| Learning rate | 2 × 10⁻⁴ | Chốt |
| Warmup | 3% tổng số step (quy đổi sang số step) | Chốt |
| Weight decay | 0 | Chốt |
| Batch × tích luỹ gradient | 2 × 8 = 16 | Chốt (mục 5.3) |
| Gradient checkpointing | Bật | Chốt (mục 5.3) |
| Số epoch | 3 | ⏳ W7 — tuỳ ngân sách L4 |
| **Kiểu số khi tính toán** | fp32 trên T4 | ⏳ **W7** — probe ba nhánh fp16 / fp32 / bf16 trên L4 |
| **`max_pixels`** | 512 token (thiên hướng) | ⏳ **W7** — cần chi phí huấn luyện trên L4 |
| Lượng tử hoá 4-bit vs không | 4-bit | ⏳ W7 — trên T4 bản không lượng tử hoá ở fp32 cần ~12,4 GB chỉ riêng trọng số |

**Vì sao chưa chốt dtype và độ phân giải ở W6:** phần cứng sẽ đổi ở W7 (T4 → L4). Chốt trên T4 rồi đổi GPU nghĩa là chạy lại baseline hai lần. Baseline chỉ có ý nghĩa khi đo cùng điều kiện với mô hình cải tiến, nên nó sẽ được chạy đúng một lần — trên phần cứng và cấu hình cuối cùng.

---

## 6. Giao thức đánh giá

### 6.1. Metric chính

| Metric | Định nghĩa | Ghi chú |
|---|---|---|
| **Exact Match (EM)** | Khớp tuyệt đối sau chuẩn hoá | Chuẩn hoá **chặt**: không gộp bí danh màu, chuẩn hoá số chỉ khi toàn chuỗi là số |
| **VQA Accuracy** | Theo chuẩn VQA | Với ViVQA (1 đáp án/câu), VQA-Acc ≈ EM sau chuẩn hoá — ghi rõ để tránh hiểu nhầm |
| **ANLS** | Độ giống chuỗi chuẩn hoá, ngưỡng 0,5 | Metric chính cho ViTextVQA; nhạy với câu đúng một phần |

BLEU/CIDEr **không** dùng làm metric chính, chỉ là chỉ số phụ nếu cần cho câu trả lời mở.

### 6.2. So sánh công bằng

- Mọi biến thể chạy qua **cùng một mã đánh giá** (`src/eval/run_baseline.py`), chỉ khác file cấu hình.
- **Mỗi thí nghiệm đổi đúng một biến.** Bài học từ W6: đổi hai biến cùng lúc (batch + checkpointing) dẫn tới hết VRAM mà không xác định được nguyên nhân.
- **Tiêu chí kết luận cố định trước khi xem số.**
- **Kiểm định thống kê**, không chỉ so số tổng:
  - EM: **McNemar chính xác** (nhị thức hai phía) trên các cặp bất đồng.
  - ANLS: **bootstrap bắt cặp** 10.000 lần, seed 42, khoảng tin cậy 95% theo bách phân vị.
- **Tập đối chứng** khi sửa một lỗi cụ thể, để phát hiện thoái lui.

### 6.3. Kế hoạch ablation (W8–W12)

| # | So sánh | Trả lời câu hỏi |
|---|---|---|
| 1 | Zero-shot vs prompt tối ưu vs QLoRA | Cải thiện đến từ đâu: prompt hay huấn luyện? |
| 2 | Có vs không OCR-enhanced prompting | OCR tiếng Việt có bù được điểm yếu dấu thanh? |
| 3 | Độ phân giải 512 vs 1.280 (huấn luyện + đánh giá cùng mức) | Đánh đổi chất lượng–chi phí khi đã có adapter |
| 4 | Qwen2.5-VL-3B vs Qwen3-VL-4B (tham chiếu, zero-shot) | Backbone mới hơn có thu hẹp khoảng cách không? |

Giả thuyết chi tiết: `docs/W02/Hypotheses.md`.

### 6.4. Phân tích lỗi và hallucination

- **Phân rã theo loại câu hỏi:** yes/no, đếm, đọc chữ, suy luận, nhận dạng đối tượng. ViVQA có sẵn nhãn loại; ViTextVQA cần gán (⏳ W7).
- **Case study tiếng Việt:** dấu thanh, code-switching, lỗi OCR.
- **Hallucination:** theo phương pháp POPE (Li et al., EMNLP 2023) — hỏi yes/no về sự tồn tại của đối tượng, đo tỉ lệ khẳng định sai.
- **Phát hiện output suy biến tự động** (`src/eval/find_degraded.py`) cho mọi run.

### 6.5. Tái lập

Mỗi lần chạy ghi: dataset split, seed (42), mã băm SHA-256 của file dữ liệu, commit git, loại GPU và khả năng bf16, phiên bản thư viện, toàn bộ cấu hình, thời gian suy luận/huấn luyện, VRAM đỉnh. Kết quả được lưu theo lô, chạy lại được khi mất kết nối.

Quy ước đặt tên: `experiments/W0X_<chế độ>_<model>_<dataset>_seed42/`.

---

## 7. Đạo đức và AI có trách nhiệm (CLO7)

- **Quyền riêng tư:** ảnh người dùng tải lên prototype có thể chứa khuôn mặt, giấy tờ, dữ liệu cá nhân. Thiết kế dự kiến (⏳ W14): không lưu ảnh sau khi xử lý.
- **Giấy phép dữ liệu:** ViTextVQA dùng qua kho có kiểm soát truy cập, tuân theo điều khoản của tác giả.
- **Thiên lệch:** mô hình pretrain chủ yếu trên dữ liệu tiếng Anh. Bằng chứng định lượng từ baseline: chênh 17 điểm EM giữa đáp án Latin và đáp án tiếng Việt có dấu; ~90 câu trả lời bằng tiếng Anh cho câu hỏi tiếng Việt.
- **Độ tin cậy:** câu trả lời trôi chảy nhưng sai là rủi ro chính. Prototype hiển thị confidence và văn bản OCR trích được để người dùng tự kiểm tra, không trình bày câu trả lời như sự thật tuyệt đối.

---

## Phụ lục — Các quyết định còn mở (⏳ W7)

| Quyết định | Căn cứ đã có | Còn thiếu |
|---|---|---|
| Kiểu số khi tính toán | fp16 gây tràn số (5.1); T4 không có bf16 | Probe fp16 / fp32 / bf16 trên L4 |
| `max_pixels` | +0,039 ANLS có ý nghĩa, EM không; chi phí suy luận 1,46× (5.2) | Chi phí huấn luyện trên L4 |
| Lượng tử hoá 4-bit vs không | — | Đo trên L4 (22,5 GB VRAM) |
| Số epoch | 14,77 giờ/epoch trên T4 (5.3) | Tốc độ thực trên L4 |
| Chạy lại baseline | Cấu hình W5 dùng fp16 | Chạy **một lần** sau khi chốt ba quyết định trên |
