# src/prototype — web app

Upload ảnh tiếng Việt · câu hỏi tiếng Việt · câu trả lời · mức tin cậy · chữ OCR làm bằng chứng.

Runtime: Colab Pro/Premium (L4) hoặc GPU server trường. Giao diện chạy được trên CPU với backend giả.

| File | Nội dung |
|---|---|
| `core.py` (W13) | Lõi dùng chung cho demo và kiểm chứng: sinh câu trả lời **y hệt eval** (`build_inputs` + `generate_from_inputs`, thêm log-prob), `yesno-detector-v1` + `normalize_yes_no` + quyết định có/không bằng margin token đầu so với τ* = 4,85 (R-YB v3), độ tin cậy (`open_confidence`, `yes_no_confidence`, `conf_level`), lớp `Backend` cho giao diện |
| `ocr_worker.py` (W13 bước 7) | OCR chạy ở **tiến trình riêng** (tránh xung đột Paddle GPU ↔ torch), cùng engine đã dựng cache; kiểm engine trùng cache |
| `app.py` (W13 bước 7) | Giao diện Gradio: mức tin cậy Cao / Trung bình / Thấp (ngưỡng đọc từ `experiments/W13_prototype_check_L4_seed42/prototype_check_summary.json`, không hiện con số), OCR + khung + bằng chứng, cảnh báo, disclaimer; không lưu ảnh / câu hỏi |

Chạy:
```bash
python -m src.prototype.app --mock                     # backend giả, CPU: duyệt giao diện
python -m src.prototype.app --adapter <…/W11_SH_…/epoch_1> --share     # model thật (L4, notebook 24)
```

Kiểm chứng:
- Offline: `python -m src.prototype.app --selftest` · `python -m src.prototype.ocr_worker --selftest` · `python -m src.eval.prototype_check --selftest`
- GPU: notebook 23 (L4: parity với eval W11, độ tin cậy, detector).

Hướng dẫn chạy đầy đủ (Colab L4, backend giả, giới hạn, sự cố thường gặp): `docs/W13/Prototype_Guide.md` · notebook `notebooks/24_prototype_live_ocr_latency_sh_qwen25vl.ipynb` (đặt `UI_MINUTES = 20` để mở giao diện).

Luật và quyết định: `docs/W13/README.md`.

## Chạy trên laptop Windows (RTX 4 GB)

Môi trường đo 2026-11-13: RTX 3050 Ti Laptop 4 GB · CPU i5-12500H · venv `D:\envs\vivqa-demo` · `HF_HOME=D:\hf_cache` · adapter `D:\vivqa_models\SH\epoch_1`.

### Cài 1 lần (tóm tắt)

1. Python 3.11 + venv ngoài repo: `py -3.11 -m venv D:\envs\vivqa-demo`
2. `setx HF_HOME D:\hf_cache` (và set trong phiên). Cache Hugging Face / pip để trên ổ D:.
3. Cài `torch==2.11.0` từ `https://download.pytorch.org/whl/cu130`, rồi đúng pin `requirements-colab.txt` (`transformers==5.17.0`, `peft==0.21.2`, `accelerate==1.15.0`, `bitsandbytes==0.50.2`, `gradio==6.31.0`, `qwen-vl-utils==0.0.14`, `Pillow==11.3.0`) + `torchvision` khớp cu130; OCR CPU: `paddlepaddle==3.3.1` (không -gpu), `paddleocr==3.7.0`, `pip install --no-deps vietocr==0.3.13` + `gdown einops`.
4. Chép adapter SH từ Drive (`adapters/W11_SH_…/epoch_1`) → `D:\vivqa_models\SH\epoch_1`; sha256 `adapter_model.safetensors` phải = `c9791e961aaa58fdcd7d9baafede3bd7bd8c902acc9cc43917306f1e0fecc354`.
5. Selftest: `python -m src.prototype.app --selftest` · `python -m src.prototype.ocr_worker --selftest`.

### Lệnh chạy

Cách nhanh: `.\demo` (offline, tự mở trình duyệt **nền sáng** `http://127.0.0.1:7860/?__theme=light`) · `.\demo share` (link gradio.live) · hoặc **Tasks: Run Task** → `ViVQA demo (offline)` / `ViVQA demo (share)`.

Khi trình chiếu / máy chiếu: mở fullscreen **F11**, phóng to trang **Ctrl + '+'** (Ctrl + '-' thu nhỏ). Muốn nền tối: chạy PS1 với `-Dark` (không gắn `?__theme=light`).

```powershell
# Từ gốc repo (khuyến nghị):
.\demo
.\demo share

# Tương đương đầy đủ:
powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Share
powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Offline
# (tuỳ chọn) số luồng OCR CPU — mặc định 12; chuỗi OCR giống hệt 1 luồng:
powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Offline -OcrThreads 12
# không tự mở trình duyệt:
powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Offline -NoBrowser
# nền tối (không thêm ?__theme=light):
powershell -ExecutionPolicy Bypass -File src\prototype\run_local_windows.ps1 -Offline -Dark
```

Khởi động ~48 s với `OMP_NUM_THREADS=12` (trước đó ~78 s khi Paddle tự gán 1 luồng). Máy phải bật suốt lúc demo. Không đổi prompt / tham số sinh / min-max pixels / model OCR.

### Latency (2026-11-15, RTX 3050 Ti 4 GB, OCR CPU T\*=12)

Đo bằng `python -m src.prototype.bench_latency` trên `examples_local.json` (5 câu × 3 lần, median). Số liệu: `docs/04_eval/latency/`.

| Giai đoạn (median-of-medians, s) | Baseline (mọi lần chạy OCR) | (a) Câu đầu, ảnh mới, bấm ngay | (b) Câu đầu, OCR nền đã xong | (c) Câu 2 cùng ảnh (cache) |
|---|---:|---:|---:|---:|
| OCR det | 13,77 | 5,81 | 0 | 0 |
| OCR rec | 0,91 | 0,06 | 0 | 0 |
| OCR tổng | 14,76 | 6,42 | 0 | 0 |
| Mô hình | 2,11 | 1,16 | 0,92 | 0,95 |
| **Tổng end-to-end** | **20,13** | **7,61** | **0,93** | **0,96** |

**A1 — Prefetch OCR khi tải ảnh:** Gradio `image.change` chạy OCR nền; bấm **Trả lời** chỉ chờ đúng future đó (không OCR lần 2).  
**A2 — Cache OCR theo sha1(bytes ảnh đã chuẩn hoá), tối đa 32 ảnh:** hỏi câu mới trên cùng ảnh → `OCR: dùng lại từ cache`.  
**A3 — Chi tiết kỹ thuật** hiện OCR / Mô hình / Tổng (kèm nguồn cache/prefetch).

Parity output vs `reference_outputs.json`: **5/5** (plain) và **15/15** (a+b+c) — answer, confidence, ocr_lines, matched_line.

**B1 (VietOCR GPU) / B2 (batch):** không bật thêm. B2 đã là đường mặc định trong `PaddleDetVietOCREngine.read` (`predict_batch`). B1 không giữ trên máy 4 GB (dễ đụng VRAM với Qwen 4-bit; rủi ro lệch float CPU↔GPU → fail parity).

### Vì sao OCR chạy CPU

VRAM 4 GB chỉ đủ cho Qwen2.5-VL-3B 4-bit + adapter (~2,3–2,4 GB peak). OCR (PaddleOCR + VietOCR) chạy tiến trình riêng trên CPU (`--ocr_device cpu`) để tránh OOM và tránh xung đột Paddle GPU ↔ torch (cùng lý do `ocr_worker` trên Colab).

Paddle **tự gán `OMP_NUM_THREADS=1`** khi import nếu biến chưa có → OCR chậm. Script `run_local_windows.ps1` đặt `OMP_NUM_THREADS` / `MKL_NUM_THREADS` trước khi chạy (mặc định **12**). Không bật mkldnn (vẫn `enable_mkldnn=False` như eval CPU).

### Quét số luồng OCR CPU (30 ảnh B5b, giống hệt T=1)

| T (OMP/MKL) | p50 (s/ảnh) | p95 (s/ảnh) | Khớp cache | Giống hệt T=1 |
|---|---|---|---|---|
| 1 | 21,7 | 38,3 | 29/30 | 30/30 |
| 4 | 18,3 | 27,7 | 29/30 | 30/30 |
| 8 | 7,4 | 21,5 | 29/30 | 30/30 |
| **12 (T\*)** | **7,3** | **10,5** | 29/30 | **30/30** |

### Số đo B5 với T\*=12 (3 câu demo, 2026-11-13)

Tải Qwen lần đầu: **1346,6 s** (riêng, lần cài). Khởi động (nạp OCR+model+adapter): **47,9 s**. Peak VRAM ~**2,40 GB**; RAM ~11–12 GB. Câu trả lời **trùng** bản đo 1 luồng.

| Câu (slot) | Lần 1 tổng / OCR / Qwen (s) | Lần 2 (nóng) | OCR chiếm (lần 1) | Trả lời | Mức |
|---|---|---|---|---|---|
| ViTextVQA chữ (198.jpg) | 9,2 / 5,5 / 3,6 | 7,5 / 5,4 / 2,1 | ~60% | trẻ em phải đội mũ bảo hiểm khi đi xe máy | Cao |
| ViVQA đếm | 3,9 / 2,9 / 1,0 | 4,0 / 3,0 / 1,0 | ~75% | hai | Cao |
| Có/không (ảnh thật) | 8,3 / 7,1 / 1,2 | 8,1 / 7,1 / 1,1 | ~86% | không | Trung bình |

Trước (OMP=1): tổng ~9–25 s/câu; OCR chiếm ~80–94% thời gian mỗi câu. Sau (T\*=12): tổng ~4–9 s/câu; OCR vẫn chiếm phần lớn nhưng tuyệt đối giảm ~3×.

### OCR CPU so với cache (B5b)

30 ảnh ViTextVQA đầu có trong `paddledet_vietocr.json` (eval): **29/30** khớp hoàn toàn chuỗi prompt (`" | "`, cắt 600 ký tự). Với T=1: p50 **21,7 s**/ảnh. Lệch 1 ảnh (`11081.jpg`) — khác GPU so với cache T4/L4, cùng hiện tượng đã ghi ở W14 R-OCR. T\*=12 giữ nguyên 29/30 khớp cache và 30/30 giống chuỗi T=1.

### Cách chia sẻ

- `-Share` → link công khai `*.gradio.live` (Gradio).
- Cùng Wi-Fi: `http://<IP-LAN>:7860` (script bind `0.0.0.0`).
- Bảo vệ: `-Offline` + `http://127.0.0.1:7860`.
