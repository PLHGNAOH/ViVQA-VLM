# Hướng dẫn tái lập ViVQA-VLM (gói tái lập — master §10)

> Mục tiêu: một người **không có trong nhóm** dựng lại được mọi số chính thức của khoá luận và chạy được prototype, chỉ từ repo này + thư mục Drive `ViVQA-VLM/` (adapter, ảnh, OCR cache).
> Bảng số chính thức: [`docs/W14/official_models.md`](W14/official_models.md), sinh tự động từ [`configs/official_models.json`](../configs/official_models.json) bằng `python -m src.eval.repro_manifest --build` (không gõ tay).
> Mọi luật kết luận (ngưỡng, kiểm định) được **ghi trước** trong `docs/Wxx/README.md` của tuần tương ứng, commit trước khi chạy.

## 0. Gói tái lập gồm những gì

| Thành phần (master §10) | Ở đâu |
|---|---|
| Source code | `src/` (data · adaptation · eval · prompting · prototype) |
| Adapter LoRA / QLoRA | Drive `ViVQA-VLM/adapters/<run>/epoch_N/` (`adapter_model.safetensors` + `adapter_config.json`, ~30 MB/adapter). **Không** lên git. sha256 ghi trong `configs/official_models.json` |
| File cấu hình | `configs/qwen_lora.yaml` (model, lượng tử hoá, LoRA, huấn luyện, eval), `configs/pope_vi_objects.json`, `experiments/W08_ocr_select_dev300_L4_seed42/ocr_selection.json` (cấu hình OCR đóng băng) |
| Script suy luận | `python -m src.prototype.infer` (dòng lệnh) · `python -m src.prototype.app` (giao diện Gradio) |
| README chạy lại | file này |
| Log thí nghiệm | `experiments/<run>/` — `metrics.json`, `predictions.json`, `run_signature.json`, `train_log.json` (split, seed, phần cứng, siêu tham số, thời gian, phiên bản thư viện, commit git) |

## 1. Môi trường

- **Máy:** Google Colab Pro. **L4** (24 GB) cho train / eval / prototype; **T4** cho dựng OCR cache. Không dùng full fine-tuning (chỉ QLoRA 4-bit NF4).
- **Thư viện:** ghim ở [`requirements-colab.txt`](../requirements-colab.txt).
  - `torch` dùng bản Colab cài sẵn: 2.11.0 + CUDA 13.0, ghi trong mọi `train_log.json` / `metrics.json`.
  - `transformers==5.17.0` ghim cứng; mọi run chính đều dùng bản này.
- **PaddleOCR GPU cài chung với torch** phải theo đúng quy trình ở notebook 19 / 24 / 28, mục 2. Lý do: `paddlepaddle-gpu` (CUDA 12.6) ghi đè NCCL của torch làm `import torch` hỏng. Quy trình gồm 4 bước:
  1. chụp danh sách `nvidia-*` của torch;
  2. cài Paddle;
  3. cài lại NCCL của torch;
  4. kiểm cả hai ở tiến trình riêng.
- **VietOCR:** `pip install --no-deps vietocr==0.3.13` rồi `pip install gdown einops`. Bản PyPI khoá `pillow==10.2.0` và sẽ hạ cấp Pillow của Colab.
- **Drive:** mount ở `/content/drive/MyDrive/ViVQA-VLM`. Mọi notebook đặt `DRIVE_ROOT` = đường này.
- **Mỗi notebook:** mở từ GitHub (File → Open notebook → GitHub → `PLHGNAOH/ViVQA-VLM`) → chọn GPU ghi ở đầu notebook → **Run all**. Notebook tự kiểm sha256 dữ liệu / cache / adapter trước khi tốn GPU và tự trả máy khi xong.

## 2. Dữ liệu

| Bước | Lệnh / notebook | Kiểm |
|---|---|---|
| ViVQA | `python -m src.data.prepare_vivqa` → `data/vivqa/{train,test}.json` + ảnh COCO | train 10.799 câu · test 3.001 câu |
| ViTextVQA (bản chính thức, HF gated `minhquan6203/ViTextVQA`, cần `HF_TOKEN`) | notebook `00_data_vitextvqa_official.ipynb` (`src/data/prepare_vitextvqa.py`) | train 35.159 · dev 5.155 · test 10.028 câu (đáp án từ `ViTextVQA_test_gt.json`); tập con `test_subset2000_seed42.json`, `train_subset8000_byimage_seed42.json` (`src/data/make_train_subset.py`) |
| Kiểm rò rỉ split | `python -m src.data.check_split_overlap` | ViTextVQA 0 ảnh trùng; ViVQA trùng ảnh train–test → báo cáo phân tầng (W07, W09) |

`data_file_sha256` của từng run nằm trong `configs/official_models.json`. File dữ liệu tạo lại phải trùng sha256 này.

## 3. OCR cache (PaddleOCR dò chữ + VietOCR đọc chữ, đóng băng ở W08)

| Cache | Notebook (T4) | Biên bản (sha256) |
|---|---|---|
| ViTextVQA (W08–W10, dùng cho mọi run chính thức) | `10_ocr_cache_vitextvqa.ipynb` (Phần A–D) | `experiments/W10_ocr_cache_fulltrain/ocr_cache_manifest.json` (`e7ce4f76d98c…`) |
| ViVQA (W11, cho SH) | `19_ocr_cache_vivqa.ipynb` | `experiments/W11_ocr_cache_vivqa/ocr_cache_manifest.json` |
| ViTextVQA đã sửa EXIF (W14, chỉ cho dòng độ nhạy) | `25_exif_ocr_audit_vitextvqa.ipynb` → `26_ocr_cache_exiffix_vitextvqa.ipynb` | `experiments/W14_exif_fix/ocr_cache_exiffix_manifest.json` (`41abe2ad85c5…`) |

**Lưu ý EXIF (W14).** Code trước W14 dò chữ và cắt chữ trên hai ảnh khác chiều với 184 ảnh ViTextVQA có cờ xoay. `src/data/ocr.py` hiện tại đã sửa lỗi này.
- Muốn tái lập **đúng cache W10**: checkout commit trước W14 (vd `b9f7fd4`) khi dựng cache. Biên bản sha256 ở trên dùng để đối chiếu.
- Cache dựng bằng code hiện tại sẽ trùng cache **đã sửa** (`_exiffix`).

Chi tiết và luật: `docs/W14/README.md`.

## 4. Huấn luyện + đánh giá: số chính thức ↔ notebook ↔ run

Danh sách đầy đủ (EM / ANLS / F1, adapter, sha256): [`docs/W14/official_models.md`](W14/official_models.md).

| Số trong khoá luận | Notebook (L4) | Run train → run eval |
|---|---|---|
| R0 — tái lập Qwen2.5-VL-3B trên TextVQA val (79,60 so với 79,3 công bố) | `06_r0_textvqa_repro_qwen25vl.ipynb` | — → `W07_R0_textvqa_val_qwen25vl3b_paper_bf16_L4` |
| B1 zero-shot (ViVQA 22,09 · ViTextVQA full 29,12 EM) | `02_run_baseline_qwen25vl.ipynb`, `08_fulltest_vitextvqa_b1_b2.ipynb` | — → `W07_B1_zeroshot_…` |
| B2 QLoRA ViTextVQA (50,26 EM) | `04_qlora_qwen25vl.ipynb`, `08_…` | `W07_B2_qlora_r16_…_vitextvqa_ep1…` → `W07_B2_eval_…_testfull…` |
| **C3** QLoRA ViVQA, chính thức (76,64 EM) | `16_qlora_vivqa_2epoch_select_qwen25vl.ipynb` | `W09_B2_qlora_r16_…_vivqa_ep2…` (epoch 2) → `W09_C3_eval_…_vivqa_test…` |
| **B3-full** QLoRA nhận biết OCR, chính thức, đóng góp chính (53,35 EM; +3,09 so với B2) | `18_qlora_ocr_b3full_vs_b2_qwen25vl.ipynb` | `W10_B3full_qlora_…` → `W10_B3full_eval_…_testfull…` |
| B3-full độ nhạy OCR sửa EXIF (53,63 EM) | `27_b3full_exiffix_testfull_qwen25vl.ipynb` | (adapter W10) → `W14_B3full_eval_exiffix_…` |
| SH — adapter dùng chung, dùng cho prototype (ViVQA 73,51 · ViTextVQA test2000 49,50) | `20_qlora_shared_adapter_qwen25vl.ipynb` | `W11_SH_qlora_…` → `W11_SH_eval_…` |

**Lệnh nền** (mọi notebook gọi hai lệnh này với tham số ghi sẵn):
```bash
python -m src.adaptation.train_qlora --dataset vitextvqa_official --run_name <run> --epochs 1 --save_steps 100 \
    --mode ocr --ocr_backend paddledet_vietocr --ocr_template v1 --ocr_max_chars 600 --ocr_min_conf 0.0
python -m src.eval.run_baseline --dataset vitextvqa_official --data_file <…/test.json> --run_name <run_eval> \
    --chunk 500 --adapter <Drive>/adapters/<run>/epoch_1 --mode ocr --ocr_backend paddledet_vietocr \
    --ocr_template v1 --ocr_max_chars 600 --ocr_min_conf 0.0
```
- Bỏ các tham số `--mode ocr …` thì thành B2 / C3.
- Bỏ `--adapter` thì thành B1.
- Cấu hình chung (Qwen2.5-VL-3B-Instruct, bf16, 4-bit NF4 + double quant, `max_pixels` 401.408, greedy, `max_new_tokens` 32, seed 42, LoRA r16 / α32 trên q/k/v/o của LLM, lr 2e-4, batch hiệu dụng 16, warmup 3%) nằm ở `configs/qwen_lora.yaml`.
- `run_signature.json` chặn việc chạy tiếp một run bằng cấu hình khác.

**Độ tất định:** eval greedy, chạy từng câu.
- Chạy lại trên cùng loại GPU (L4) cho dự đoán **trùng 100%**: W13 R-PAR; W14 V5 với 9.902 / 9.902 câu.
- Khác GPU (T4 ↔ L4) thì OCR có thể lệch nhẹ ở vài ảnh (W13 / W14), nên phải dùng đúng loại GPU ghi trong notebook.

## 5. Phân tích (chỉ CPU, đọc `experiments/`)

| Phân tích | Lệnh / notebook | Kết quả |
|---|---|---|
| Bảng đánh giá đầy đủ (tính lại metric từ `predictions.json`) | `python -m src.eval.full_eval_table` | `experiments/W11_full_eval_table/` |
| So ghép cặp (McNemar, bootstrap) | `python -m src.eval.compare_runs --run_a … --run_b … --out …` | các file `paired_*.json` |
| Ổn định 3 seed (42 / 43 / 44) | notebook `14_b3_seed_stability_qwen25vl.ipynb`, `src/eval/seed_stability.py` | `experiments/W09_B3_seed_stability_test2000_L4/` |
| Bảng 2×2 OCR lúc train × lúc test | notebook `15_ocr_factorial_fulltest_qwen25vl.ipynb`, `src/eval/factorial_2x2.py` | `experiments/W09_ocr_factorial_2x2_test2000_L4_seed42/` |
| Ablation rank / module LoRA | notebook `17_lora_ablation_rank_modules_qwen25vl.ipynb`, `src/eval/lora_ablation.py` | `experiments/W10_lora_ablation_test2000_L4_seed42/` |
| Hallucination POPE-vi | notebook `21_pope_vi_hallucination_qwen25vl.ipynb`, `src/eval/pope_vi.py` | `experiments/W12_pope_vi_L4_seed42/` |
| Error analysis tự động + gán nhãn tay | `src/eval/error_analysis.py`, `src/eval/audit_w12.py` | `experiments/W12_error_analysis/`, `W12_error_audit/` |
| Lỗi EXIF trong OCR cache | notebook 25 → 26 → 27, `src/data/exif_audit.py` | `experiments/W14_exif_audit_vitextvqa/`, `W14_exif_fix/` |

## 6. Prototype

- **Giao diện:** notebook `28_prototype_v2_live_demo_sh_qwen25vl.ipynb` (L4). Notebook kiểm R-OCR v2 / R-LIVE v2 rồi mở link Gradio. Hướng dẫn: `docs/W13/Prototype_Guide.md`, kịch bản video: `docs/W14/Demo_Script.md`.
- **Dòng lệnh** (sau khi cài thư viện như notebook 28 mục 2):
  ```bash
  python -m src.prototype.infer --official sh --drive_root /content/drive/MyDrive/ViVQA-VLM \
      --image anh.jpg --question "cửa hàng này tên là gì?" --question "trong ảnh có xe máy không?" --out ketqua.jsonl
  ```
  `--official sh` tự lấy adapter SH và kiểm sha256 theo `configs/official_models.json`. Thử không cần GPU: thêm `--mock`.
- **Prototype là đúng hệ thống đã đánh giá:** W13 R-PAR 100%; W14 R-OCR v2 / R-LIVE v2 100% / 100% (`docs/W14/README.md`).

## 7. Kiểm nhanh trước khi nộp

```bash
python -m src.eval.repro_manifest --build                                   # dựng lại biên bản từ experiments/
python -m src.eval.repro_manifest --check_adapters "G:\My Drive\ViVQA-VLM"   # adapter trên Drive trùng sha256
for m in src.data.ocr src.data.exif_audit src.eval.run_baseline src.adaptation.train_qlora src.eval.full_eval_table \
         src.eval.prototype_live src.eval.prototype_check src.prototype.app src.prototype.infer src.prototype.demo_examples \
         src.prototype.ocr_worker src.eval.repro_manifest; do python -m $m --selftest; done
```

## 8. Giới hạn đã biết (ghi vào khoá luận)

1. Phiên bản thư viện: `torch`, `transformers`, `peft`, `paddleocr`, `paddlepaddle` được ghi trong log của từng run. `bitsandbytes`, `accelerate`, `qwen-vl-utils` không được log theo run: chỉ có ở W04 (`experiments/W04_zeroshot_qwen25vl_seed42/env.json`) và trong ảnh chụp môi trường W14 (`experiments/W14_env_freeze_L4.txt`). Hai lần ghi này cho cùng `bitsandbytes` 0.50.2 và `accelerate` 1.15.0.
2. Torch là bản Colab cài sẵn. Nếu Colab đổi bản mặc định, cần cài đúng `torch==2.11.0` (CUDA 13.0) như ghi trong log.
3. OCR cache W10 có lỗi EXIF ở 1,256% câu test / 1,28% câu train ViTextVQA. Số chính thức giữ nguyên theo luật L3; dòng độ nhạy ở mục 4.
4. Dataset ViTextVQA là gated: người tái lập cần tự xin quyền truy cập trên Hugging Face.
