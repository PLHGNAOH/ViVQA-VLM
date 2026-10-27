# src/prototype — web app

Upload ảnh tiếng Việt · câu hỏi tiếng Việt · câu trả lời · confidence · OCR/evidence.

Runtime: Colab Pro/Premium hoặc GPU server trường.

| File | Nội dung |
|---|---|
| `core.py` (W13) | Lõi dùng chung cho demo và kiểm chứng: sinh câu trả lời **y hệt eval** (`build_inputs` + `generate_from_inputs`, thêm log-prob), `yesno-detector-v1` + `normalize_yes_no` + quyết định có/không bằng margin token đầu so với τ* = 4,85 (R-YB v3), độ tin cậy (`open_confidence`, `yes_no_confidence`, `conf_level`), lớp `Backend` cho giao diện |

Kiểm chứng: `python -m src.eval.prototype_check --selftest` (offline) · notebook 23 (L4: parity với eval W11, độ tin cậy, detector). Luật: `docs/W13/README.md`.
