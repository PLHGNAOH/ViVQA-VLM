# W13 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W13 — Prototype web (mốc master "W14 · Prototype + thesis" kéo lên sớm) |
| Thời gian thực | 26–31/10/2026 (tuần thực 8) |
| Mốc kế tiếp | W14 · Thesis (R6/R7 bản tiếng Anh, phần Prototype) · Hội đồng cấp Khoa tuần thực 13 (30/11–06/12) |
| Phần cứng | Colab Pro: NVIDIA L4 (notebook 22, 23, 24) · CPU / máy local (selftest, giao diện backend giả) |
| CU Colab | 25,95 (đầu tuần) → 17,38 (cuối tuần) |
| Commit cuối tuần | commit đóng tuần sau `7c2dd64` (xem mục "Dấu vết commit") |

## Mục tiêu

Dựng prototype web hỏi đáp ảnh tiếng Việt (master §11): upload ảnh, câu hỏi tiếng Việt, câu trả lời, confidence, OCR làm bằng chứng. Prototype phải là **đúng hệ thống đã đánh giá** chứ không phải một bản demo riêng. Bốn vấn đề phải giải trước (README W13):
- P1: thiên lệch "có" của SH;
- P2: confidence chưa được kiểm;
- P3: lệch pipeline giữa demo và eval;
- P4: Paddle GPU xung đột torch.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | Chẩn đoán thiên lệch "có" của SH bằng logit token đầu (`src/eval/yesno_calib.py`, notebook 22) | v1 **không hợp lệ** (bỏ sót token "không" nhiều mảnh; greedy có repetition penalty 1,05) → v2 dừng ở chốt smoke → **v3 hợp lệ** (trùng greedy W12 100%) |
| 2 | Fix A — ngưỡng τ lúc suy luận, chọn trên tập calibration rời POPE-vi | **R-YB v3: ĐẠT**, τ\* = +4,85. AUC cho thấy thiên lệch chủ yếu là lệch ngưỡng → không cần train lại (bỏ Fix B) |
| 3 | Backend prototype `src/prototype/core.py` + `src/eval/prototype_check.py`, notebook 23 | **R-PAR** 100% / 100% · **R-CONF** ĐẠT (ECE 0,198 → chỉ hiện mức) · **R-YB-det** ĐẠT |
| 4 | Giao diện Gradio `src/prototype/app.py` + OCR tiến trình riêng `src/prototype/ocr_worker.py` | Quyết định D1–D8; selftest offline; ảnh chụp giao diện (backend giả) |
| 5 | Prototype chạy thật (`src/eval/prototype_live.py`, notebook 24) | **R-LIVE** ĐẠT · **R-OCR** KHÔNG ĐẠT (ViTextVQA 87,5%) · latency p50/p95 0,98 / 3,30 s |
| 6 | 20 ảnh thật (Trường chụp, Drive `data/real_photos_w13/`) | EM 14/20, câu có/không 5/5 |
| 7 | Tài liệu: `docs/W13/Prototype_Guide.md`, README `src/prototype`, Team Report | ✅ |
| 8 | Video demo dự phòng | ⏳ mang sang (cần một phiên L4 mở giao diện) |
| 9 | Đóng tuần: README, dọn Downloads + Drive | ✅ |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| VLM / Baseline | Repetition penalty 1,05 của `generate()` (phát hiện ở chẩn đoán R-YB v1); `mm_token_type_ids` của transformers 5.x cho mRoPE |
| Dataset / OCR | OCR tiến trình riêng, kiểm engine = cache; R-OCR và phân tích nguyên nhân; 20 ảnh thật (SV chụp: Trường) |
| PEFT / Training | Quyết định không train lại SH (Fix A đủ, AUC SH ≈ B1-ocr) |
| Evaluation / Deployment | R-YB, R-PAR, R-CONF, R-YB-det, R-LIVE; giao diện Gradio; latency / VRAM; hướng dẫn chạy |

### Definition of Done

- [x] Mọi luật (R-YB, R-PAR, R-CONF, R-YB-det, R-OCR, R-LIVE) commit **trước** khi chạy notebook tương ứng
- [x] Lần đo không hợp lệ (R-YB v1) được giữ làm hồ sơ, có chẩn đoán; bản sửa chỉ đổi cách đo, không đổi ngưỡng
- [x] Prototype dùng chung lõi với kiểm chứng; chữ ký cấu hình trùng run eval W11 trước khi chạy
- [x] Confidence hiển thị đúng luật (ECE > 0,05 → chỉ mức, không có con số)
- [x] Luật KHÔNG ĐẠT (R-OCR) được báo đúng như luật; phân tích nguyên nhân ghi nhãn "phân tích sau"
- [x] Ảnh thật không lên GitHub; giao diện không lưu ảnh / câu hỏi
- [x] Mọi số truy được về `experiments/W13_*`; Claude tính lại độc lập từ dữ liệu thô
- [ ] Video demo dự phòng (mang sang)

## Kết quả / số liệu

### 1. Thiên lệch "có" của SH (R-YB v3, nguồn `experiments/W13_yesno_calib_v3_L4_seed42/`)

| Cách chọn | F1 SH-sửa | F1 B1-ocr | ΔF1 [CI95] | yes-ratio |
|---|---|---|---|---|
| random | 85,14 | 83,39 | +1,75 [+0,68; +2,89] | 41,5% |
| popular | 83,32 | 81,99 | +1,33 [+0,21; +2,50] | 43,5% |
| adversarial | 81,69 | 80,59 | +1,11 [−0,05; +2,30] | 45,4% |

So với SH greedy (W12):
- yes-ratio 77–89% → 41–45%;
- tỉ lệ báo sai 55–79% → 5,1–12,9%.

Tỉ lệ báo sai vẫn cao hơn B1-ocr 2,4–4,0 điểm, cần ghi vào Discussion.

### 2. Backend = eval (nguồn `experiments/W13_prototype_check_L4_seed42/`)

| Luật | Kết quả |
|---|---|
| R-PAR | ViVQA 3.001/3.001, ViTextVQA test2000 2.000/2.000 trùng hệt W11 |
| R-CONF | EM theo mức trên nửa B: Cao 92,0 · TB 69,6 · Thấp 34,2 (giảm dần ở cả hai tập); ECE 0,198 (ViTextVQA 0,308) |
| R-YB-det | 1/3.001 câu bị bắt (EM −0,03) |

Mẫu câu prototype (t3), SH với τ\*: F1 85,8 / 84,6 / 83,1; báo sai 5,3 / 7,9 / 11,4% (SH ở τ = 0: 43,8 / 54,4 / 64,8%).

### 3. Prototype chạy thật (nguồn `experiments/W13_prototype_live_L4_seed42/`)

| Luật | ViVQA (200 ảnh / 211 câu) | ViTextVQA (200 ảnh / 261 câu) | Phán quyết |
|---|---|---|---|
| R-LIVE | 100%, EM 75,83 = 75,83 | 98,5%, EM 50,19 vs 49,81 | **ĐẠT** |
| R-OCR | 100% | 87,5% | **KHÔNG ĐẠT** |

Nguyên nhân R-OCR (phân tích sau):
- 18 ảnh lớn hơn 2.048 px bị giao diện thu nhỏ: 17 ảnh lệch OCR, và cả 4 câu trả lời lệch đều nằm ở đây;
- 8 ảnh nguyên file lệch nhẹ, do khác GPU;
- 2 ảnh có cache đáng ngờ (giả thuyết EXIF / độ phân giải) → W14.

Tốc độ (L4):
- tổng p50 0,98 s, p95 3,30 s;
- OCR p95 1,92 s; model p95 1,58 s;
- khởi động 60 s; VRAM 2,6 + 1,9 GB.

### 4. Ảnh thật (nguồn `experiments/W13_REAL_sh_realphotos_bf16_L4/`)

- **EM 14/20**. Câu có/không 5/5 đúng, gồm 3 câu hỏi về vật không có trong ảnh.
- Lỗi:
  - trả lời thiếu chữ (2);
  - OCR đọc sai (1);
  - gần đúng (1);
  - sai thật (2: đếm, nhận đồ vật).
- VietOCR sinh chữ tiếng Anh vô nghĩa trên chữ nghiêng hoặc vùng không có chữ.

### 5. Đối chiếu giả thuyết

| Mục | Trạng thái sau W13 | Lý do |
|---|---|---|
| H1–H3 | Không đổi | W13 không chạy thí nghiệm mới cho H1–H3 |
| H4 · QLoRA không tăng hallucination | Một phần (không đổi theo luật W12) | Thông tin thêm: SH với τ\* đưa báo sai từ 55–79% xuống 5–13%, nhưng vẫn cao hơn B1-ocr 2–4 điểm |
| Prototype đúng hệ thống đã đánh giá | Đạt (R-PAR, R-LIVE) | OCR trực tiếp lệch cache ở 12,5% ảnh ViTextVQA (R-OCR), nhưng câu trả lời không đổi trên ảnh nguyên file |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|---|---|---|
| B1 | Cache OCR ViTextVQA có thể sai trên ảnh có cờ EXIF xoay (1849.jpg, 14277.jpg) | **Trung bình–cao** (nếu đúng thì ảnh hưởng dữ liệu train/eval của B3/SH) | W14: đếm ảnh có EXIF Orientation ≠ 1, so OCR trên ảnh đã xoay đúng (Colab CPU). Kết luận trước khi viết R6 bản cuối |
| B2 | Model tự tin quá mức (ECE 0,198) | Thấp | Đã xử lý ở giao diện (chỉ hiện mức). Ghi vào Discussion; hiệu chỉnh nhiệt độ là hướng tương lai |
| B3 | SH báo sai cao hơn B1-ocr 2–4 điểm sau khi sửa | Thấp | Ghi giới hạn; F1 đạt nhờ recall |
| B4 | VietOCR sinh chữ vô nghĩa trên vùng không có chữ / chữ nghiêng | Thấp | Ghi vào error analysis; ngưỡng `min_conf` là hướng thử (cần luật mới, không đổi kết quả đã báo cáo) |
| B5 | Video demo dự phòng chưa quay | Thấp | Chạy notebook 24 với `UI_MINUTES = 20` (khoảng 1 CU) |

## Kế hoạch tuần sau (W14 · Thesis)

1. Kiểm giả thuyết EXIF của cache OCR ViTextVQA (B1), không tốn GPU.
2. Quay video demo dự phòng (B5).
3. R6 / R7 bản tiếng Anh: thêm phần Prototype (R-PAR, R-CONF, R-LIVE, R-OCR, latency, ảnh thật) và các giới hạn (ECE, báo sai, OCR).
4. Cập nhật deck cộng dồn cho hội đồng cấp Khoa: slide prototype dùng ảnh chụp giao diện thật.

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `3147496` | Bước 0–2: luật R-YB, `yesno_calib.py` (v1), notebook 22 |
| `0d1e86e` | R-YB v1 không hợp lệ (chẩn đoán), luật v2 (chỉ đổi cách đo) |
| `f9175f6` | Nhân hàng `mm_token_type_ids` khi ghép lô |
| `5fd5675` | R-YB v3: token đầu, max mỗi phía, forward 1 hàng; selftest so với `generate()` |
| `f8b273b` | Kết quả R-YB v3: ĐẠT |
| `b00cf41` | Backend prototype + luật R-PAR / R-CONF / R-YB-det, notebook 23, mẫu câu 3 |
| `b4d98dc` | Kết quả bước 4–6: ĐẠT ×3 |
| `8e7e501` | Bước 7: giao diện Gradio + OCR tiến trình riêng |
| `5c162d2` | Bước 8: luật R-OCR / R-LIVE, `prototype_live.py`, notebook 24 |
| `7c2dd64` | Kết quả bước 8 + sửa lỗi tạo thư mục log |
| *(commit đóng tuần)* | Hướng dẫn chạy prototype, Team Report, README W13 + gốc |

Commit deck Review 2 trong tuần (`6b84e3d`, `0170bda`, `cbc4e0d`, `86f5de8`) thuộc phiên Review 2, không thuộc phạm vi W13.

## Link experiment log

- R-YB: `experiments/W13_yesno_questions/`, `experiments/W13_YN3_*`, `experiments/W13_yesno_calib_v3_L4_seed42/` (v1: `experiments/W13_yesno_calib_L4_seed42/`)
- Backend: `experiments/W13_PT_*`, `experiments/W13_prototype_check_L4_seed42/`
- Chạy thật: `experiments/W13_LIVE_sh_ocrlive_sample200img_bf16_L4_seed42/`, `experiments/W13_prototype_live_L4_seed42/`
- Ảnh thật: `experiments/W13_REAL_sh_realphotos_bf16_L4/` (ảnh trên Drive `data/real_photos_w13/`, không lên GitHub)
- Notebook: `notebooks/22_yesbias_logit_calib_sh_qwen25vl.ipynb`, `23_prototype_parity_confidence_sh_qwen25vl.ipynb`, `24_prototype_live_ocr_latency_sh_qwen25vl.ipynb`
- Hướng dẫn: `docs/W13/Prototype_Guide.md`
