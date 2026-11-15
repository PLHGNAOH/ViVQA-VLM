"""
diagnose_pipeline.py — chẩn đoán một (ảnh, câu hỏi) qua Backend.ask (không UI cache).

Lưu docs/04_eval/diag/<YYYY-MM-DD>/ :
  meta.json, prompt.txt, image_to_processor.png, configs_compare.md (nếu --write_compare),
  answers_4cfg.json

Chạy:
  python -m src.prototype.diagnose_pipeline \\
    --image D:\\vivqa_models\\diag\\cat.jpg \\
    --question \"Con vật trong ảnh là con gì\"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure_threads() -> Dict[str, str]:
    """Đặt OMP/MKL trước mọi import Paddle (giống run_local_windows.ps1)."""
    for k, v in (("OMP_NUM_THREADS", "12"), ("MKL_NUM_THREADS", "12")):
        if not os.environ.get(k):
            os.environ[k] = v
    return {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")}


def _image_meta(path: str) -> Dict[str, Any]:
    from PIL import Image
    with Image.open(path) as im:
        exif = im.getexif()
        orient = exif.get(274, 1) if exif else 1
        return {"path": path, "size": list(im.size), "mode": im.mode, "format": im.format,
                "exif_orientation": orient}


def _build_inputs_debug(processor, image_path: str, prompt_text: str, device, dump_png: str):
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from src.prompting.builder import to_chat_messages
    image = Image.open(image_path).convert("RGB")
    image.save(dump_png)
    messages = to_chat_messages(image, prompt_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    if device is not None:
        inputs = inputs.to(device)
    pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_pad = int((inputs["input_ids"] == pad_id).sum().item()) if pad_id is not None else -1
    pv = inputs.get("pixel_values")
    grid = inputs.get("image_grid_thw")
    return {
        "chat_text_head": text[:500],
        "pixel_values.shape": list(pv.shape) if pv is not None else None,
        "image_grid_thw": grid.tolist() if grid is not None else None,
        "n_image_pad": n_pad,
        "input_ids_len": int(inputs["input_ids"].shape[1]),
    }, inputs


def _adapter_info(model) -> Dict[str, Any]:
    info: Dict[str, Any] = {"type": type(model).__name__}
    try:
        info["active_adapters"] = list(model.active_adapters) if hasattr(model, "active_adapters") else None
    except Exception as e:
        info["active_adapters_error"] = str(e)
    try:
        if hasattr(model, "peft_config"):
            cfg = model.peft_config
            # peft_config may be dict of configs
            if isinstance(cfg, dict):
                def _ser(x):
                    if isinstance(x, set):
                        return sorted(x)
                    return x
                info["peft_config"] = {k: {kk: _ser(getattr(v, kk, None)) for kk in
                                           ("peft_type", "r", "lora_alpha", "target_modules")}
                                       for k, v in cfg.items()}
            else:
                info["peft_config"] = str(cfg)[:500]
    except Exception as e:
        info["peft_config_error"] = str(e)
    lora_b_nonzero = None
    try:
        for n, p in model.named_parameters():
            if "lora_B" in n and p is not None:
                lora_b_nonzero = bool((p.detach().float().abs() > 0).any().item())
                info["lora_B_sample"] = n
                info["lora_B_shape"] = list(p.shape)
                break
    except Exception as e:
        info["lora_B_error"] = str(e)
    info["lora_B_nonzero"] = lora_b_nonzero
    return info


def _ask_cfg(backend, image_path: str, question: str, use_ocr: bool, use_adapter: bool) -> Dict[str, Any]:
    from src.data.ocr import lines_to_texts
    from src.prototype.core import answer_one, conf_level

    lines: List[Dict[str, Any]] = []
    if use_ocr:
        read_fn = getattr(backend, "ocr_read", None)
        if callable(read_fn):
            raw = read_fn(image_path)
            lines = list(raw.get("lines") or [])
        else:
            lines = backend.ocr_fn(image_path)
    texts, _ = lines_to_texts({"lines": lines}, backend.ocr_max_chars, backend.ocr_min_conf)

    ctx = None
    if not use_adapter and hasattr(backend.model, "disable_adapter"):
        ctx = backend.model.disable_adapter()
        ctx.__enter__()
    try:
        r = answer_one(backend.model, backend.processor, image_path, question, texts, backend.cfg["prompting"],
                       backend.yn_tokens, backend.penalty, backend.tau, backend.special_ids,
                       backend.cfg["prompting"].get("max_new_tokens", 32))
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    cuts = backend.cuts_yn if r["mode"] == "yesno" else backend.cuts_open
    return {
        "use_ocr": use_ocr, "use_adapter": use_adapter,
        "answer": r["answer"], "confidence": float(r["confidence"]),
        "level": conf_level(r["confidence"], cuts), "mode": r["mode"],
        "prompt": r.get("prompt"),
        "n_ocr_lines": len(lines),
        "ocr_texts": texts,
    }


def write_compare_table(out_path: str) -> None:
    rows = [
        ("min_pixels", "200704 (qwen_lora.yaml / W11)", "200704", "configs/qwen_lora.yaml:26"),
        ("max_pixels", "401408 (512 token, W07/W11)", "401408", "configs/qwen_lora.yaml:27"),
        ("torch_dtype", "bfloat16", "bfloat16", "configs/qwen_lora.yaml + run_baseline.load_model"),
        ("load_in_4bit", "true NF4", "true NF4", "configs/qwen_lora.yaml quantization"),
        ("bnb_4bit_compute_dtype", "bfloat16", "bfloat16", "src/eval/run_baseline.py CANONICAL_QUANT"),
        ("max_new_tokens", "32", "32", "configs/qwen_lora.yaml:66"),
        ("decoding", "greedy do_sample=False", "greedy do_sample=False", "src/prototype/core.py generate_from_inputs"),
        ("prompt OCR template", "v1 (ocr_selection)", "v1", "experiments/W08_ocr_select.../ocr_selection.json"),
        ("chat template", "processor.apply_chat_template", "cùng qua to_chat_messages", "src/prompting/builder.py + core.build_inputs"),
        ("OCR engine", "paddledet_vietocr", "paddledet_vietocr CPU (demo)", "app make_real_backend --ocr_device cpu"),
        ("EXIF / RGB", "oriented_image / convert RGB", "prepare_image + Image.open.convert(RGB)",
         "src/data/ocr.py oriented_image; core.build_inputs"),
        ("τ* yes/no", "4.85", "4.85", "src/prototype/core.py TAU_STAR_SH"),
        ("adapter", "SH epoch_1", "D:\\vivqa_models\\SH\\epoch_1", "run_local_windows.ps1"),
    ]
    lines = ["# Config compare: Colab/luận văn vs local demo", "",
             "| Tham số | Colab / luận văn | Local demo | File:dòng |",
             "|---|---|---|---|"]
    for p, colab, local, loc in rows:
        lines.append(f"| {p} | {colab} | {local} | `{loc}` |")
    lines.append("")
    lines.append("Ghi chú: demo Windows chạy OCR trên **CPU** (tránh OOM 4 GB); Colab eval thường GPU. "
                 "Engine info vẫn khớp cache (check_engine_matches_cache).")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    threads = _ensure_threads()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--adapter", default=r"D:\vivqa_models\SH\epoch_1")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "qwen_lora.yaml"))
    ap.add_argument("--ocr_device", default="cpu")
    ap.add_argument("--expect_adapter_sha",
                    default="c9791e961aaa58fdcd7d9baafede3bd7bd8c902acc9cc43917306f1e0fecc354")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--write_compare", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.exists(args.image):
        print(f"STOP: không thấy ảnh {args.image}", file=sys.stderr)
        print("Hãy đặt file vào đúng đường dẫn rồi chạy lại (B1 yêu cầu D:\\vivqa_models\\diag\\cat.jpg).",
              file=sys.stderr)
        return 3

    day = date.today().isoformat()
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "docs", "04_eval", "diag", day)
    os.makedirs(out_dir, exist_ok=True)

    from src.prototype.app import load_ui_settings, load_ocr_selection, make_real_backend, prepare_image
    import tempfile

    st, ocr_cfg = load_ui_settings(), load_ocr_selection()
    backend, client, info = make_real_backend(
        args.adapter, args.config, st, ocr_cfg, ocr_device=args.ocr_device,
        expect_adapter_sha=args.expect_adapter_sha,
    )
    tmp = tempfile.mkdtemp(prefix="vivqa_diag_")
    try:
        used, prep = prepare_image(args.image, tmp)
        meta = {
            "threads": threads,
            "image_orig": _image_meta(args.image),
            "prepare_image": prep,
            "image_used": _image_meta(used),
            "start_info": {k: info[k] for k in info if k != "ocr_engine"},
            "ocr_engine": info.get("ocr_engine"),
            "model_cfg": {
                "min_pixels": backend.cfg["model"].get("min_pixels"),
                "max_pixels": backend.cfg["model"].get("max_pixels"),
                "max_new_tokens": backend.cfg["prompting"].get("max_new_tokens"),
                "load_in_4bit": (backend.cfg.get("quantization") or {}).get("load_in_4bit"),
            },
            "adapter": _adapter_info(backend.model),
            "question": args.question,
        }

        # 4 configs
        answers = {}
        for key, use_ad, use_ocr in (
            ("i_adapter_ocr", True, True),
            ("ii_adapter_no_ocr", True, False),
            ("iii_no_adapter_ocr", False, True),
            ("iv_no_adapter_no_ocr", False, False),
        ):
            print(f"Running {key}…")
            answers[key] = _ask_cfg(backend, used, args.question, use_ocr=use_ocr, use_adapter=use_ad)

        # debug tensors from config i prompt
        dump_png = os.path.join(out_dir, "image_to_processor.png")
        dbg, _ = _build_inputs_debug(
            backend.processor, used, answers["i_adapter_ocr"]["prompt"], backend.model.device, dump_png,
        )
        meta["processor_debug"] = dbg

        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(answers["i_adapter_ocr"].get("prompt") or "")
        with open(os.path.join(out_dir, "answers_4cfg.json"), "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)
        if args.write_compare:
            write_compare_table(os.path.join(out_dir, "configs_compare.md"))

        print(json.dumps({"out_dir": out_dir, "answers": {k: {"answer": v["answer"], "level": v["level"]}
                                                          for k, v in answers.items()}},
                         ensure_ascii=False, indent=2))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
