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
