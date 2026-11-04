# W5 — Báo cáo Baseline chính thức (ViVQA-VLM)

> **Mục tiêu tài liệu:** tổng kết baseline zero-shot chính thức của W5 để (1) làm mốc so sánh cho mọi cải tiến sau này, (2) làm nguyên liệu cho Review 2 (W7), (3) làm nguồn để viết R4/R6 bản tiếng Anh.
> **Phạm vi:** Qwen2.5-VL-3B-Instruct, **zero-shot** (không train), trên ViVQA và ViTextVQA chính thức.
> **Trạng thái bản:** tiếng Việt, nháp v1 — chờ nhóm duyệt. Thuật ngữ tra ở `docs/Glossary.md`.

---

## 1. Kết quả chính

| Dataset | Tập đánh giá | N | EM | VQA-Acc | ANLS | Thời gian suy luận |
|---------|--------------|--:|:--:|:-------:|:----:|-------------------|
| ViVQA | test đầy đủ | 3.001 | **0.2246** | 0.2246 | **0.2876** | 2.532 s (0,84 s/mẫu) |
| ViTextVQA (chính thức) | tập con test cố định, seed 42 | 2.000 | **0.2850** | 0.2850 | **0.4676** | 2.702 s (1,35 s/mẫu) |

**Đọc số:**
- VQA-Acc = EM vì mỗi câu chỉ có **một** đáp án chuẩn (công thức đa người gán nhãn của VQA v2 thu về EM).
- **ANLS cao hơn EM rõ rệt**, nhất là ở ViTextVQA (0.47 vs 0.29): model thường trả lời **gần đúng** nhưng không khớp tuyệt đối. Phần lớn chênh lệch đến từ phong cách trả lời (§6, §7).
- **Không so sánh EM trực tiếp giữa hai dataset:** ViTextVQA thường có đáp án là chính dòng chữ trong ảnh; ViVQA đòi đúng từ vựng của đáp án dịch máy.

---

## 2. Protocol đánh giá (bắt buộc để tái lập)

| Hạng mục | Giá trị |
|----------|---------|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Lượng tử hoá | 4-bit NF4, double quant, compute dtype float16 (**cùng base mà QLoRA sẽ dùng**) |
| torch_dtype | float16 (T4 không hỗ trợ bf16 native) |
| Độ phân giải ảnh | `min_pixels = 256·28·28` (200.704), `max_pixels = 512·28·28` (401.408) → tối đa 512 visual token/ảnh |
| Prompt | chế độ `zero_shot` (`src/prompting/builder.py`), system prompt tiếng Việt yêu cầu trả lời ngắn |
| Sinh câu trả lời | greedy (`do_sample=False`), `max_new_tokens = 32` |
| Chuẩn hoá khi chấm (áp **đối xứng** cho dự đoán và đáp án) | NFC · chữ thường · dấu câu → khoảng trắng · gộp khoảng trắng · bỏ tiền tố `màu ` ở đầu chuỗi · đổi số chữ ↔ chữ số (0–20, chục tròn tới 100) **chỉ khi cả chuỗi là một số** · **giữ dấu thanh tiếng Việt** · không dùng bảng đồng nghĩa |
| EM | khớp tuyệt đối sau chuẩn hoá (EM "chặt") |
| ANLS | 1 − khoảng cách Levenshtein chuẩn hoá, ngưỡng τ = 0.5 |
| Seed | 42 |
| Phần cứng | Google Colab, Tesla T4 (15 GB) |
| Phiên bản | transformers 5.17.0 · torch 2.11.0+cu128 |
| Config | `configs/qwen_lora.yaml` (commit `37da80a`) — notebook/script đọc thẳng từ config, không gõ tay |
| Script chạy | `src/eval/run_baseline.py` (chạy theo lô 500 câu, resume được, ghi log đầy đủ) |

**Nguyên tắc đã giữ:** luật chuẩn hoá và config được **chốt trước khi chạy tập test** và không sửa sau khi thấy kết quả. Mọi con số "sau khi nới luật" trong tài liệu này đều ghi rõ là *phân tích phụ*.

---

## 3. Dữ liệu

### 3.1 ViVQA

| Mục | Giá trị |
|-----|---------|
| Nguồn | repo chính thức `github.com/kh4nh12/ViVQA` (`train.csv` + `test.csv`); ảnh tải từ MS COCO 2014 theo `img_id` |
| Quy mô | 15.000 cặp hỏi–đáp, 10.328 ảnh (tải đủ 10.328/10.328, 0 lỗi) |
| Split | train 10.799 · **val 1.200** (cắt 10% từ train, seed 42 — bản gốc không có val) · test 3.001 |
| Loại câu hỏi (cột `type`, quy ước COCO-QA) | 0 = object · 1 = number · 2 = color · 3 = location |
| Phân bố test | object 1.247 · location 685 · color 625 · number 444 |
| Script | `src/data/prepare_vivqa.py` → `data/vivqa/` trên Drive (`prep_report.json`) |

**Loại trừ ViVQA-X:** file `ViVQA-X_*.json` có trên máy là **dataset khác** (VQA kèm giải thích, VQA-NLE), không phải ViVQA. Không dùng cho baseline.

### 3.2 ViTextVQA (chính thức)

| Mục | Giá trị |
|-----|---------|
| Nguồn | kho chính thức của tác giả `huggingface.co/datasets/minhquan6203/ViTextVQA` (gated), revision `02964f3489631f5754c0847e8167c0a1d5a3c2ec` |
| Giấy phép | CC BY-NC 3.0 (nghiên cứu, phi thương mại, phải trích dẫn) |
| Split | train 35.159 câu / 11.733 ảnh · dev 5.155 / 1.676 · test 10.028 / 3.353 — tổng 50.342 câu / 16.762 ảnh, khớp paper |
| Đáp án test | lấy từ `ViTextVQA_test_gt.json`. `ViTextVQA_test.json` chỉ là **mẫu nộp Kaggle** (đáp án `your answer`) — script chặn tự động |
| Tập đánh giá W5 | tập con cố định 2.000 câu từ test, seed 42, SHA-256 `69a8b35c…f68e` |
| Nhiễu dataset | cặp (ảnh, câu hỏi) trùng nhưng khác đáp án: 53 (train) · 15 (dev) · 18 (test đầy đủ) |
| Loại câu hỏi | dataset không gán nhãn → tất cả là `unknown` (cần gán ở W6 để làm error analysis theo loại) |
| Script | `src/data/prepare_vitextvqa.py` → `data/vitextvqa_official/` trên Drive |

**Về bản mirror dùng ở W4** (`nhonhoccode/ViTextVQA`): toàn bộ cặp (ảnh, câu hỏi) của nó nằm trong test chính thức, tức là bản sao của test. Kết quả W4 không sai về dữ liệu, nhưng từ W5 chỉ dùng bản chính thức (nguồn gốc rõ, có train/dev, đúng giấy phép).

---

## 4. Kết quả chi tiết — ViVQA (test, 3.001 câu)

| Loại câu hỏi | N | EM | ANLS |
|--------------|--:|:--:|:----:|
| number | 444 | **0.696** | 0.697 |
| color | 625 | **0.450** | 0.554 |
| object | 1.247 | 0.061 | 0.128 |
| location | 685 | 0.012 | 0.070 |
| **Tổng** | **3.001** | **0.225** | **0.288** |

Model làm tốt câu **đếm** và **màu**, rất yếu ở **object** và **location** — hai loại chiếm 65% tập test.

---

## 5. Kết quả chi tiết — ViTextVQA (tập con 2.000 câu)

| Metric | Giá trị | Ghi chú |
|--------|:-------:|---------|
| EM | 0.285 | metric chính |
| VQA-Acc | 0.285 | metric chính |
| ANLS | 0.468 | metric chính |
| Token F1 | 0.530 | *phân tích phụ*, chỉ để đối chiếu với paper ViTextVQA (paper dùng EM + F1) |

---

## 6. Error analysis — ViVQA

| Kiểu lỗi | Bằng chứng | Liên hệ Research Gap |
|----------|------------|----------------------|
| **Lệch phong cách trả lời** | ~19% câu object/location/color: thêm giới từ/loại từ — `trên giường` vs `giường`, `con cừu` vs `cừu`, `xanh lá cây` vs `xanh lá` | Model chưa quen văn phong đáp án ViVQA |
| **Trả lời bằng tiếng Anh** | ít nhất 90 câu chỉ với 6 từ: `giraffe` 32, `kitchen` 24, `bus` 19, `bench` 11, `truck` 2, `laptop` 2 | **G1** (pretrain thiên tiếng Anh), code-switching (**G2**) |
| **Đồng nghĩa / đáp án dịch máy** | `xe hơi` vs `xe ô tô`, `nhà bếp` vs `phòng bếp` | Nhãn ViVQA dịch tự động từ COCO-QA |
| **Thiên lệch màu** | đoán `đỏ` 117 lần trong khi đáp án `đỏ` chỉ có 75 lần (trên 625 câu màu) | Thiên lệch của model |
| **Hallucination** | `Điện thoại Nokia 6310i`, `bò cạp` (ảnh là con chim) | **G4** |

**Phân tích phụ (không phải metric chính):** nếu bỏ giới từ/loại từ ở đầu chuỗi cho cả hai vế, EM tăng từ 0.225 lên **0.301** → khoảng **7,6 điểm** mất do phong cách trả lời, phần còn lại là lỗi thật, tiếng Anh và đồng nghĩa.

---

## 7. Error analysis — ViTextVQA

### 7.1 Phân loại kết quả (2.000 câu)

| Kiểu | Tỉ lệ | Ví dụ (dự đoán → đáp án) |
|------|:-----:|--------------------------|
| Đúng (EM) | 28,5% | |
| **Thừa từ** (đáp án nằm trong dự đoán) | 19,6% | `Nhà thuốc Long Châu` → `long châu`; `Nhà xuất bản: Nhà Xuất Bản Trẻ` → `nhà xuất bản trẻ` |
| Trùng một phần | 21,9% | `www.alphabeta.vn` → `www.alphabeta.com.vn` |
| **Sai hẳn** | 24,9% | nhiều câu **lặp lại câu hỏi** thay vì trả lời: `Chợ Bình Tây quyết tâm thực hiện t…` |
| Thiếu từ | 4,0% | `10 km` → `dài 10 km` |
| Output hỏng `!!!!…` | 1,1% | xem §7.4 |

Độ dài trung bình: dự đoán 5,4 từ, đáp án 3,7 từ.

### 7.2 Chữ Latin vs chữ tiếng Việt có dấu (bằng chứng G1/G2)

| Nhóm câu (theo đáp án) | Tỉ lệ tập | EM |
|------------------------|:---------:|:--:|
| Đáp án thuần chữ Latin (tên riêng, tiếng Anh, số, website) | 36% | **0.394** |
| Đáp án tiếng Việt có dấu | 64% | **0.224** |

Chênh ~17 điểm: model đọc chữ Latin tốt hơn hẳn chữ tiếng Việt — dấu vết của pretrain thiên tiếng Anh.

### 7.3 Lỗi chỉ do dấu thanh (G2)

45 câu (2,3%) đúng chữ nhưng sai dấu: `Khác Dấu` → `khắc dấu`, `BUT DO` → `bút đỏ`, `Bé Nguyễn` → `bé nguyên`. Có vài câu **đáp án gốc tự thiếu dấu** (`nguyen thi binh an`) — tính là nhiễu của dataset.

### 7.4 Vấn đề kỹ thuật: output hỏng `!!!!…`

- 23/2.000 câu ViTextVQA (1,15%) sinh toàn dấu chấm than; ViVQA 0/3.001.
- **Giả thuyết (chưa kiểm chứng):** tràn số khi tính ở float16 (bắt buộc trên T4), dễ xảy ra hơn với ảnh nhiều chữ.
- **Xử lý hiện tại:** giữ nguyên con số baseline (23 câu tính là sai), báo cáo minh bạch tỉ lệ; ảnh hưởng tối đa ~1 điểm EM.
- **Vì sao quan trọng:** nếu đúng do fp16, việc train QLoRA ở W8 trên T4 cũng có nguy cơ loss thành NaN → kiểm chứng ở W6 (chạy lại 23 câu với compute fp32).

---

## 8. Vì sao số W5 khác số W4?

W4 báo cáo ViTextVQA **EM 38.3% / ANLS 55.9%**; W5 là **28.5% / 46.8%**. Hai lần chạy **không so sánh được** vì khác cấu hình ở 4 điểm:

| Hạng mục | W4 | W5 |
|----------|----|----|
| Lượng tử hoá | không (float16 đầy đủ) | **4-bit NF4** |
| `max_pixels` | 1024·28·28 (≤1.024 token/ảnh) | **512·28·28** (≤512 token/ảnh) |
| Dữ liệu | mirror, 300 câu | chính thức, tập con 2.000 câu |
| Luật chấm | bản cũ | bản chặt đã chốt |

**Giả thuyết:** phần lớn chênh lệch đến từ **độ phân giải giảm một nửa** (chữ nhỏ khó đọc hơn) và **lượng tử hoá 4-bit**. Chưa thể tách riêng tác động nếu không có ablation.

**Hệ quả cho thiết kế:** `max_pixels = 512` được chọn để thống nhất với QLoRA và vừa VRAM T4 — hợp lý cho ViVQA, nhưng **có thể là cái giá đáng kể với ViTextVQA**. Kế hoạch W6: ablation trên **dev** (không phải test) — (a) 512 vs 1024 token, (b) 4-bit vs float16 — để quyết định có cần độ phân giải riêng cho ViTextVQA hay không. Nếu đổi config, **baseline phải chạy lại** với config mới để mọi so sánh vẫn công bằng.

Số W4 được giữ làm lịch sử (`experiments/W04_*`), **không** dùng trong thesis.

---

## 9. Nhật ký quyết định W5

| Quyết định | Lý do |
|------------|-------|
| Dùng ViVQA gốc (kh4nh12), loại ViVQA-X | Đúng dataset trong đề cương; ViVQA-X là bài toán khác |
| Dùng ViTextVQA chính thức thay mirror | Nguồn gốc rõ, có train/dev, đúng giấy phép |
| EM chặt: không bảng đồng nghĩa màu; đổi số chỉ khi cả chuỗi là một số | Tránh nới luật tuỳ tiện; phần "gần đúng" để ANLS đo |
| Chốt metric + config trước khi chạy test, không sửa sau | Tránh chỉnh thước đo cho vừa tập test |
| `max_pixels = 512·28·28`, float16 | Vừa VRAM T4, thống nhất với QLoRA; T4 không có bf16 |
| Tập con ViTextVQA 2.000 câu cố định, seed 42 | Giới hạn compute; mọi thí nghiệm sau dùng đúng tập này |
| Tinh chỉnh prompt trên VAL/dev, báo cáo trên test | Không để test ảnh hưởng thiết kế |

---

## 10. Hướng cải tiến có bằng chứng (→ W6 trở đi)

| Hướng | Nhắm vào lỗi nào | Bằng chứng từ baseline |
|-------|------------------|------------------------|
| **Prompt optimization** (tinh chỉnh trên VAL/dev) | Thừa từ, lặp câu hỏi, trả lời tiếng Anh | ~20% câu thừa từ ở cả hai dataset; ≥90 câu tiếng Anh ở ViVQA |
| **OCR-enhanced prompting** (PaddleOCR / VietOCR) | Đọc sai chữ tiếng Việt có dấu, chữ nhỏ bị thu nhỏ | EM 0.224 (tiếng Việt) vs 0.394 (Latin); 45 lỗi dấu; nghi vấn độ phân giải §8 |
| **QLoRA** | Phong cách đáp án, từ vựng tiếng Việt | object/location ViVQA gần 0; dự đoán dài hơn đáp án ~1,5 lần |
| **Đo hallucination** | Bịa chi tiết | ví dụ §6, §7 |

Giả thuyết kiểm chứng được: sau QLoRA, **object và location (ViVQA)** và **nhóm đáp án tiếng Việt có dấu (ViTextVQA)** tăng nhiều nhất. (Đối chiếu với `docs/W02/Hypotheses.md` khi viết bản thesis.)

---

## 11. Tái lập

```bash
# Chuẩn bị dữ liệu (Colab, CPU đủ)
python -m src.data.prepare_vivqa --out_dir "<Drive>/ViVQA-VLM/data/vivqa" --val_ratio 0.1 --seed 42
python -m src.data.prepare_vitextvqa --out_dir "<Drive>/ViVQA-VLM/data/vitextvqa_official"   # cần HF_TOKEN

# Chạy baseline (Colab, T4 GPU)
python -u -m src.eval.run_baseline --dataset vivqa --run_name W05_zeroshot_qwen25vl_vivqa_test_seed42
python -u -m src.eval.run_baseline --dataset vitextvqa_official --run_name W05_zeroshot_qwen25vl_vitextvqa_test2000_seed42
```

| Commit | Nội dung |
|--------|----------|
| `37da80a` | Config chuẩn W05 (`max_pixels` 512·28·28, fp16) |
| `6e391ad` | Kết quả ViVQA + notebook baseline |
| `3a9594a` | `prepare_vitextvqa.py` (bản chính thức) |
| `ae1cf6d` | `run_baseline.py` (script chạy chung) |
| `fb32c3e` | Kết quả ViTextVQA + notebook runner |

Ghi chú: kết quả ViVQA được sinh bởi notebook `01` (logic giống hệt `run_baseline.py`, cùng config và metric); ViTextVQA sinh bởi `run_baseline.py`.
