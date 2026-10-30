# Hướng dẫn chạy prototype ViVQA-VLM (W13)

> Người đọc: thành viên nhóm, GVHD, hội đồng. Mục tiêu: chạy được giao diện hỏi đáp ảnh tiếng Việt **đúng hệ thống đã đánh giá** (R-PAR, R-LIVE) và biết nó làm được gì, không làm được gì.

## 1. Prototype là gì

Người dùng tải một ảnh lên rồi gõ câu hỏi tiếng Việt. Giao diện trả về:
- câu trả lời ngắn bằng tiếng Việt;
- **mức** tin cậy (Cao / Trung bình / Thấp);
- chữ OCR đọc được trong ảnh, có khung đánh số và đánh dấu dòng khớp câu trả lời;
- cảnh báo và disclaimer.

| Thành phần | Chi tiết |
|---|---|
| Model | Qwen2.5-VL-3B-Instruct, 4-bit NF4, bf16 + **adapter QLoRA dùng chung SH** (W11, `c9791e96…`) |
| OCR | PaddleOCR (dò hộp, PP-OCRv6_medium_det) + VietOCR (đọc, vgg_transformer), template v1, ≤ 600 ký tự. Chạy ở **tiến trình riêng** (`src/prototype/ocr_worker.py`) |
| Câu có/không | `yesno-detector-v1` → thêm `Trả lời "có" hoặc "không".` → quyết định bằng margin token đầu so với **τ\* = +4,85** (R-YB v3) |
| Độ tin cậy | Chỉ hiện mức. Ngưỡng đọc từ `experiments/W13_prototype_check_L4_seed42/prototype_check_summary.json`. ECE 0,198 > 0,05 nên **không hiện con số** |
| Code | `src/prototype/core.py` (lõi, dùng chung với kiểm chứng) · `app.py` (Gradio) · `ocr_worker.py` (OCR) |

## 2. Cách chạy

### 2.1 Trên Colab L4 (khuyến nghị, cách đã kiểm)

1. Mở `notebooks/24_prototype_live_ocr_latency_sh_qwen25vl.ipynb` từ GitHub trên Colab, chọn **Runtime → Change runtime type → L4**.
2. Ở cell mục 1, đặt `UI_MINUTES = 20` (số phút mở giao diện).
3. Bấm **Runtime → Run all**. Mục 2 cài thư viện (6–8 phút). Mục 4–6 tự bỏ qua hoặc chạy nhanh nếu kết quả đã có trên Drive.
4. Ở mục 7, notebook in ra `Running on public URL: https://….gradio.live`. Mở link đó trên điện thoại hoặc máy tính.
5. Hết `UI_MINUTES` phút, giao diện tự tắt và máy tự trả.

Chi phí khoảng 1 CU cho 20 phút giao diện, gồm cả cài đặt.

**Chạy tay** (khi đã cài thư viện như mục 2 của notebook 24):
```bash
python -m src.prototype.app \
  --adapter /content/drive/MyDrive/ViVQA-VLM/adapters/W11_SH_qlora_ocr_vietocr_v1_r16_qwen25vl_shared_vivqa_sub8k_ep1_bf16_L4_seed42/epoch_1 \
  --expect_adapter_sha c9791e961aaa58fdcd7d9baafede3bd7bd8c902acc9cc43917306f1e0fecc354 \
  --ocr_device gpu --share
```
Khi khởi động, giao diện tự kiểm ba điều kiện, sai một điều thì không chạy:
- (a) 3 luật bước 4–6 đều ĐẠT;
- (b) adapter trùng sha256;
- (c) engine OCR trùng engine đã dựng cache.

### 2.2 Chỉ xem giao diện, không cần GPU

```bash
pip install gradio==6.31.0 pillow
python -m src.prototype.app --mock        # mở http://localhost:7860
```
Chế độ này dùng backend giả: câu trả lời và OCR là dữ liệu mẫu, giao diện có băng vàng cảnh báo. Ảnh chụp giao diện: `docs/W13/figures/ui_mock_*.png`.

### 2.3 Kiểm thử offline

```bash
python -m src.prototype.app --selftest
python -m src.prototype.ocr_worker --selftest
python -m src.eval.prototype_live --selftest
python -m src.eval.prototype_check --selftest
```
Trên Windows, đặt `PYTHONIOENCODING=utf-8` trước khi chạy.

## 3. Prototype đã được kiểm những gì

| Luật (ghi trước) | Kết quả | Nguồn |
|---|---|---|
| R-PAR: backend (OCR cache) = eval W11 | 100% / 100% (5.001 câu) | `experiments/W13_prototype_check_L4_seed42/` |
| R-CONF: EM giảm dần theo mức | Cao 92,0 > TB 69,6 > Thấp 34,2 (nửa B); ECE 0,198 → chỉ hiện mức | như trên |
| R-YB-det: detector có/không không làm hỏng câu thường | 1/3.001 câu bị bắt, EM −0,03 | như trên |
| R-YB v3: ngưỡng τ\* sửa thiên lệch "có" của SH | ΔF1 so với B1-ocr +1,1 đến +1,8; yes-ratio 41–45% | `experiments/W13_yesno_calib_v3_L4_seed42/` |
| R-LIVE: trả lời qua đường giao diện (OCR trực tiếp) = eval W11 | ViVQA 100%, ViTextVQA 98,5% (ΔEM +0,38) — **ĐẠT** | `experiments/W13_prototype_live_L4_seed42/` |
| R-OCR: OCR trực tiếp = cache | ViVQA 100%, ViTextVQA **87,5% — KHÔNG ĐẠT** (giao diện thu nhỏ ảnh > 2.048 px; khác GPU) | như trên |

Tốc độ trên L4: tổng **p50 0,98 s, p95 3,30 s** mỗi câu (ViTextVQA p95 3,65 s). Khởi động 60 s. VRAM 2,6 GB (Qwen) + 1,9 GB (OCR).

## 4. Giới hạn cần nói rõ khi demo

- **Model tự tin quá mức.** Mức "Cao" vẫn sai khoảng 8% trên test; ảnh thật lệch phân phối nên mức thường thấp hơn.
- **OCR là điểm yếu chính** ở ảnh có chữ:
  - chữ nghiêng hoặc cong dễ bị đọc sai;
  - VietOCR có thể sinh chữ tiếng Anh vô nghĩa trên vùng không phải chữ.
- **Câu trả lời thường ngắn hơn đáp án** (vd "orion" thay vì "nha khoa orion"): EM chấm sai, nhưng người đọc thấy đúng ý.
- **Đếm số lượng và nhận đồ vật hiếm** còn sai (ảnh thật: đếm 6 thay vì 5 con mèo; "dây" thay vì "móc áo").
- **Ảnh lớn hơn 2.048 px bị thu nhỏ** trước khi xử lý. Ảnh dataset đều đi nguyên file, trừ khoảng 9% ảnh ViTextVQA lớn hơn ngưỡng này.
- Ảnh HEIC (iPhone) chưa hỗ trợ: chụp ở chế độ "Tương thích nhất" (JPG).
- Link `*.gradio.live` là **công khai** trong lúc giao diện mở: không chia sẻ link. Giao diện không lưu ảnh hay câu hỏi; log (nếu bật) chỉ ghi số đo.

## 5. Sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Mục 2 notebook 24 báo `OCR_DEVICE = cpu` | Paddle GPU không chạy chung được với torch → OCR chạy CPU (chậm hơn). Kết quả R-OCR khi đó có thể khác; ghi lại device |
| `❌ engine OCR khác engine đã dựng cache` | Sai phiên bản `paddleocr` / model. Cài đúng `paddleocr==3.7.0` như mục 2 |
| `❌ adapter … ≠ adapter đã kiểm` | Adapter trên Drive bị đổi. Dùng đúng thư mục `W11_SH_…/epoch_1` |
| Giao diện báo "Không đọc được ảnh" | Ảnh HEIC hoặc file hỏng → đổi sang JPG / PNG |
| Mất link `gradio.live` | Chạy lại cell mục 7 (model nạp lại khoảng 1 phút) |
