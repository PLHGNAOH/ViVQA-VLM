# Reproducibility Package Checklist

> Bản nộp cuối. Chi tiết hạng mục cũng có trong `docs/templates/` (review) và Master Instruction §10.

## 1. Source code

- [ ] Code inference + (nếu có) PEFT train trong `src/` (`data`, `baseline`, `adaptation`, `eval`, `prototype`)
- [ ] Không phụ thuộc path máy cá nhân không ghi chú
- [ ] Phiên bản thư viện ghi trong README

## 2. LoRA / QLoRA adapter weights

- [ ] Adapter ở `adapters/` (hoặc link drive/HF **có version**)
- [ ] **Không** nộp full FT weights (không được tồn tại trong protocol)
- [ ] Ghi model base id + revision tương thích

## 3. Configs

- [ ] File cấu hình từng run chính trong `configs/`
- [ ] Split + seed ghi trong config hoặc `data/splits/`

## 4. Inference script

- [ ] Một lệnh/entry point chạy demo hoặc batch eval
- [ ] Ghi input/output format

## 5. README setup / run

- [ ] Root `README.md` + hướng dẫn GPU (Colab Pro/Premium hoặc server trường)
- [ ] Mục PEFT-only / cấm full FT

## 6. Experiment log

- [ ] `experiments/Wxx_<ten>/` hoặc bảng từ `Experiment_Log_Template.md`
- [ ] Mỗi run: split, seed, hardware, hyperparameters, EM, VQA-Acc, ANLS, train time, infer time

## 7. Extra

- [ ] Thesis PDF/docx (thường ở `03_Report/`, không xóa file cũ)
- [ ] Slides defense
- [ ] Licence dataset tôn trọng
