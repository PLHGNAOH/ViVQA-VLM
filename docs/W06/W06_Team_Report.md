# W06 — Team Weekly Report

> Bản tiếng Việt (bản làm việc).

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W06 — Baseline → Method (Report 4) |
| Kết thúc | 2026-09-29 |
| Mốc kế tiếp | **W07 — Review 2** (baseline + phương pháp đề xuất + kết quả sơ bộ) |
| Phần cứng dùng trong tuần | Tesla T4 (Colab miễn phí) |
| Commit cuối tuần | xem mục "Dấu vết commit" |

## Mục tiêu

1. Chẩn đoán 23 output suy biến của baseline W5 trước khi huấn luyện.
2. Ablation cấu hình suy luận (độ phân giải, kiểu số) trên **dev**, không đụng test.
3. Dựng và kiểm chứng đường ray huấn luyện QLoRA; đo chi phí thật.
4. Viết Methodology (R4) bản tiếng Việt.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | Dtype probe (23 câu hỏng + 50 đối chứng) | ✅ fp16 là nguyên nhân; fp32 sửa được, không thoái lui |
| 2 | Ablation độ phân giải trên dev 300 câu + kiểm định bắt cặp | ✅ ANLS cải thiện có ý nghĩa, EM không; chi phí 1,46× |
| 3 | `train_qlora.py`: che nhãn chỉ đáp án, đo `prompt_len` theo từng ảnh, log chi phí | ✅ selftest offline; đã huấn luyện thật 30 step trên T4 |
| 4 | Smoke run: batch 2 vs 6, bật/tắt gradient checkpointing | ✅ T4 bão hoà tính toán; tắt checkpointing → hết VRAM |
| 5 | Sửa lỗi đếm tham số với trọng số 4-bit (đếm thiếu một nửa) | ✅ dùng hàm đếm của PEFT: **0,196%** |
| 6 | Tương thích `transformers` 5.17.0 (bỏ `warmup_ratio`) | ✅ quy đổi sang `warmup_steps` + lọc tham số theo API thật |
| 7 | R4 Methodology bản tiếng Việt | ✅ 23/23 con số đối chiếu khớp dữ liệu repo |
| 8 | Xác nhận GPU của trường | ✅ Trường **không có GPU** → chuyển Colab Pro (L4) từ W7 |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| VLM / Baseline | Dtype probe (1), ablation độ phân giải (2) |
| Dataset / OCR | Đếm mẫu train thật (10.799), chính sách chọn đáp án |
| PEFT / Training | `train_qlora.py` (3), smoke run (4), đếm tham số (5), tương thích thư viện (6) |
| Evaluation | `find_degraded.py`, `probe_breakdown.py`, `compare_runs.py` (McNemar + bootstrap) |

### Definition of Done

- [x] Nguyên nhân output suy biến được xác nhận bằng thí nghiệm một biến
- [x] Mọi so sánh cấu hình thực hiện trên dev, có kiểm định thống kê
- [x] Đường ray huấn luyện chạy hết, loss giảm, tỉ lệ tham số < 1%
- [x] Chi phí huấn luyện đo được (s/mẫu, VRAM đỉnh, giờ/epoch)
- [x] R4 bản tiếng Việt, mọi con số truy được về file trong repo
- [ ] Đóng băng cấu hình → **dời W7** (có lý do, xem Blocker)
- [ ] Prompt optimization, gán `question_type` ViTextVQA → **dời W7**

## Kết quả / số liệu

**Dtype probe** (73 câu ViTextVQA test: 23 hỏng + 50 đối chứng):

| | fp16 | fp32 |
|---|---|---|
| Câu suy biến | 23 | 0 |
| Nhóm hỏng — EM / ANLS | 0,000 / 0,000 | 0,348 / 0,517 |
| Nhóm đối chứng — EM | 0,280 | 0,280 |
| Thoái lui | — | 0 |
| s/câu | 1,965 | 2,739 |

**Ablation độ phân giải** (dev 300 câu, fp32, 4-bit):

| | 512 token | 1.280 token |
|---|---|---|
| EM | 0,3333 | 0,3600 |
| ANLS | 0,4654 | 0,5041 |
| s/câu | 2,605 | 3,807 |

Bảng chéo EM: 93 cùng đúng · 7 chỉ 512 đúng · 15 chỉ 1.280 đúng · 185 cùng sai.

**Smoke run QLoRA** (200 mẫu ViVQA, 512 token, fp32):

| Batch × tích luỹ | Checkpointing | s/mẫu | VRAM đỉnh | Kết quả |
|---|---|---|---|---|
| 2 × 8 | Bật | 4,93 | 5,93 GB | ✅ loss 1,61 → 0,88 |
| 6 × 3 | Tắt | — | 14,46 / 14,56 GB | ❌ hết VRAM |
| 6 × 3 | Bật | 5,10 | 11,6 GB | ✅ |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|--------|-----|-------------|
| B1 | Baseline Qwen2.5-VL-3B chưa được GVHD xác nhận (treo từ W4) | Trung bình | Trao đổi trực tiếp với hai GVHD tại buổi họp offline |
| B2 | T4 không có bf16, fp32 quá chậm: 14,77 giờ/epoch | **Cao** | Colab Pro, GPU L4 từ W7 |
| B3 | Cấu hình baseline W5 dùng fp16 (đã chứng minh gây lỗi) | Cao | Chạy lại baseline **một lần** sau khi đóng băng cấu hình ở W7 |
| B4 | `run_baseline.py` kiểm tra cứng `torch_dtype = float16` (`CANONICAL`) | Trung bình | Cập nhật cùng lúc với việc đóng băng cấu hình ở W7 |
| B5 | Review 2 cần kết quả sơ bộ; chưa có số EM/ANLS của mô hình đã huấn luyện | Cao | Run QLoRA ngắn trên L4 nếu kịp; không dùng adapter smoke run |

**Vì sao dời đóng băng cấu hình sang W7:** phần cứng đổi từ T4 sang L4. Chốt trên T4 rồi đổi GPU nghĩa là chạy lại baseline hai lần.

## Kế hoạch tuần sau (W07 — Review 2)

1. Nâng Colab Pro; xác nhận được cấp L4 và `bf16_supported = True`.
2. Probe ba nhánh fp16 / fp32 / bf16 trên L4 (tái dùng 73 câu probe).
3. Smoke run huấn luyện 512 vs 1.280 token trên L4 → chốt `max_pixels`; đo thêm trục 4-bit vs không lượng tử hoá.
4. Đóng băng cấu hình: cập nhật `configs/qwen_lora.yaml` + `CANONICAL` trong `run_baseline.py`.
5. Chạy lại baseline trên cả hai dataset, một lần.
6. Run QLoRA sơ bộ + đánh giá trên dev → kết quả sơ bộ cho Review 2 (nếu kịp).
7. Prompt optimization trên dev; gán `question_type` cho ViTextVQA.
8. Slide Review 2 (tiếng Anh, trình bày tiếng Việt); sau review điền checklist AI_Capstone_Review123.

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `26af132` | `find_degraded.py` + danh sách 23 câu hỏng |
| `23e0421` | Notebook 03 (dtype probe + ablation cấu hình) |
| `43aa272` | Kết quả dtype probe + phân tích theo nhóm |
| `6cb7c18` | Ablation độ phân giải + kiểm định McNemar |
| `c9ab5f7` | `train_qlora.py` |
| `0c93f78` | Notebook 04 (smoke run) |
| `e7f9a40` | Tương thích `transformers` 5.17 |
| `d22595c` | Đếm tham số 4-bit đúng + tham số batch/checkpointing qua CLI |
| `5e7e2e6` | Kết quả smoke run |
| `ebe5d91` | Số mẫu train thật + chú thích 4-bit và 10 step |
| `357e166` | R4 Methodology bản tiếng Việt |

## Link experiment log

- `experiments/W06_dtype_probe_seed42/`
- `experiments/W06_probe_fp16_qwen25vl_vitextvqa73_seed42/`, `experiments/W06_probe_fp32_qwen25vl_vitextvqa73_seed42/`
- `experiments/W06_ablation_config_dev300_seed42/`
- `experiments/W06_ablation_A1_mp512_4bit_qwen25vl_vitextvqa_dev300_seed42/`, `experiments/W06_ablation_A2_mp1280_4bit_qwen25vl_vitextvqa_dev300_seed42/`
- `experiments/W06_smoke_qlora_seed42/`
- `experiments/W06_smoke_qlora_qwen25vl_vivqa_mp512_seed42/`, `experiments/W06_smoke_qlora_qwen25vl_vivqa_mp512_bs6_seed42/`
