# Config compare: Colab/luận văn vs local demo

| Tham số | Colab / luận văn | Local demo | File:dòng |
|---|---|---|---|
| min_pixels | 200704 (qwen_lora.yaml / W11) | 200704 | `configs/qwen_lora.yaml:26` |
| max_pixels | 401408 (512 token, W07/W11) | 401408 | `configs/qwen_lora.yaml:27` |
| torch_dtype | bfloat16 | bfloat16 | `configs/qwen_lora.yaml + run_baseline.load_model` |
| load_in_4bit | true NF4 | true NF4 | `configs/qwen_lora.yaml quantization` |
| bnb_4bit_compute_dtype | bfloat16 | bfloat16 | `src/eval/run_baseline.py CANONICAL_QUANT` |
| max_new_tokens | 32 | 32 | `configs/qwen_lora.yaml:66` |
| decoding | greedy do_sample=False | greedy do_sample=False | `src/prototype/core.py generate_from_inputs` |
| prompt OCR template | v1 (ocr_selection) | v1 | `experiments/W08_ocr_select.../ocr_selection.json` |
| chat template | processor.apply_chat_template | cùng qua to_chat_messages | `src/prompting/builder.py + core.build_inputs` |
| OCR engine | paddledet_vietocr | paddledet_vietocr CPU (demo) | `app make_real_backend --ocr_device cpu` |
| EXIF / RGB | oriented_image / convert RGB | prepare_image + Image.open.convert(RGB) | `src/data/ocr.py oriented_image; core.build_inputs` |
| τ* yes/no | 4.85 | 4.85 | `src/prototype/core.py TAU_STAR_SH` |
| adapter | SH epoch_1 | D:\vivqa_models\SH\epoch_1 | `run_local_windows.ps1` |

Ghi chú: demo Windows chạy OCR trên **CPU** (tránh OOM 4 GB); Colab eval thường GPU. Engine info vẫn khớp cache (check_engine_matches_cache).
