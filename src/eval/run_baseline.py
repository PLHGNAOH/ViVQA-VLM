"""
run_baseline.py — Inference script dùng chung cho mọi dataset của ViVQA-VLM.

Chạy 1 model (theo configs/qwen_lora.yaml) trên 1 file data đã chuẩn hoá, theo LÔ:
  - mỗi lô ghi ngay lên Drive: <out_root>/<run_name>/chunks/chunk_XX.json (ghi .tmp rồi đổi tên)
  - Colab rớt -> chạy lại đúng lệnh, lô đã xong tự bỏ qua (có kiểm tra id khớp)
  - gộp -> tính EM / VQA-Acc / ANLS trên toàn bộ -> metrics.json + predictions.json

Dataset hỗ trợ:
  vivqa               data/vivqa/test.json                                ảnh: chép Drive -> /content
  vitextvqa_official  data/vitextvqa_official/test_subset2000_seed42.json ảnh: giải nén zip -> /content

Cách chạy (Colab, GPU):
  !python -u -m src.eval.run_baseline --dataset vitextvqa_official \
        --run_name W05_zeroshot_qwen25vl_vitextvqa_test2000_seed42
  (-u để log tiến độ hiện ngay, không bị dồn cuối)
Biến thể backbone (W07 R1, chỉ để đối chiếu paper / ablation backbone — KHÔNG thay baseline):
  --variant qwen2vl7b_paper   Qwen2-VL-7B-Instruct, bf16, không lượng tử, độ phân giải mặc định
  --variant qwen2vl7b_b1cfg   Qwen2-VL-7B-Instruct, đúng cấu hình B1 (4-bit NF4, 512 token)
  Config gốc vẫn phải đúng bản chốt W07; biến thể ghi đè TƯỜNG MINH và nằm trong run_signature.
Kiểm thử offline (không GPU, không mạng):
  python -m src.eval.run_baseline --selftest
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.data.vivqa_dataset import load_vivqa
from src.eval.metrics import evaluate_predictions
from src.eval.run_eval import run_evaluation, generate_answer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/ViVQA-VLM"
DATASETS = {
    "vivqa": {"data_rel": "data/vivqa/test.json"},
    "vitextvqa_official": {"data_rel": "data/vitextvqa_official/test_subset2000_seed42.json"},
}
# Cấu hình chuẩn — chốt W07 trên GPU L4 (notebook 05, experiments/W07_config_freeze_L4_seed42/):
#   bf16: không tràn số như fp16 (23/23 câu hỏng tái hiện trên cả T4 lẫn L4), nhanh hơn fp32 1,71×,
#         thoái lui 1/50 câu đối chứng so với fp32 (luật định trước: <= 2).
#   512 token: train ViTextVQA ở 1.280 token chậm 1,91× (luật định trước: chọn 1.280 nếu <= 1,5×).
# Lịch sử: W05 dùng float16 (commit 37da80a) -> đã chứng minh gây lỗi, KHÔNG dùng để so sánh.
CANONICAL = {"max_pixels": 401408, "torch_dtype": "bfloat16"}
CANONICAL_QUANT = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16"}

# W07 R1 — đối chiếu dòng "QwenVL-7b" ở Table 7 paper ViTextVQA (Nguyen et al., 2025; 100 câu test, zero-shot,
# không công bố prompt/checkpoint; phần thân bài trích dẫn Qwen2-VL -> chọn Qwen2-VL-7B-Instruct).
# Baseline của đề tài VẪN là Qwen2.5-VL-3B; các biến thể này chỉ dùng cho R1 và ablation "backbone khác nhau".
QWEN2VL7B = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
    "qwen2vl7b_paper": {
        "model": {"backbone": "qwen2_vl", "model_id": QWEN2VL7B, "torch_dtype": "bfloat16",
                  "min_pixels": None, "max_pixels": None,          # None = mặc định của processor
                  "oom_fallback_max_pixels": 1003520},             # 1.280 token; chỉ dùng khi 1 ảnh tràn VRAM (có ghi log)
        "quantization": {"enabled": False},
        "why": "gần paper nhất có thể: bf16, không lượng tử, độ phân giải mặc định (paper không công bố cấu hình)",
    },
    "qwen2vl7b_b1cfg": {
        "model": {"backbone": "qwen2_vl", "model_id": QWEN2VL7B, "torch_dtype": "bfloat16",
                  "min_pixels": 200704, "max_pixels": 401408},
        "quantization": None,                                      # None = giữ nguyên 4-bit NF4 của B1
        "why": "đúng cấu hình B1 (4-bit NF4, 512 token, cùng prompt) -> ablation backbone 3B vs 7B",
    },
}


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: str, obj: Any, indent: Optional[int] = None) -> None:
    """Ghi .tmp rồi đổi tên -> không bao giờ để lại file dở dang."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# Config + model
# --------------------------------------------------------------------------- #
def load_config(path: str, mode: Optional[str] = None, check: bool = True) -> Dict[str, Any]:
    """Đọc config; kiểm tra đúng bản chuẩn W05; tuỳ chọn ghi đè prompting.mode."""
    import yaml
    with open(path, encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    if check:
        m = cfg["model"]
        for k, v in CANONICAL.items():
            if m.get(k) != v:
                raise ValueError(f"Config {k}={m.get(k)} khác bản chuẩn W07 ({v}) -> git pull?")
        q = cfg.get("quantization", {})
        for k, v in CANONICAL_QUANT.items():
            if q.get(k) != v:
                raise ValueError(f"Config quantization.{k}={q.get(k)} khác bản chuẩn W07 ({v}) -> git pull?")
    if mode:
        cfg["prompting"]["mode"] = mode
    return cfg


def apply_variant(cfg: Dict[str, Any], variant: Optional[str]) -> Dict[str, Any]:
    """Ghi đè backbone theo MODEL_VARIANTS. variant=None -> trả cfg NGUYÊN VẸN (chữ ký B1/B2 không đổi)."""
    if variant is None:
        return cfg
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"--variant phải là một trong {list(MODEL_VARIANTS)}")
    v = MODEL_VARIANTS[variant]
    cfg = copy.deepcopy(cfg)
    cfg["model"].update(copy.deepcopy(v["model"]))
    cfg["model"]["variant"] = variant
    if v["quantization"] is not None:
        cfg["quantization"] = copy.deepcopy(v["quantization"])
    return cfg


def model_load_plan(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Phần thuần (không cần torch) của load_model: chọn lớp model, có lượng tử hay không, tham số processor."""
    m, q = cfg["model"], cfg.get("quantization", {}) or {}
    classes = {"qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration", "qwen2_vl": "Qwen2VLForConditionalGeneration"}
    backbone = m.get("backbone", "qwen2_5_vl")
    if backbone not in classes:
        raise ValueError(f"backbone '{backbone}' chưa hỗ trợ (có: {list(classes)})")
    quant = None
    if q.get("load_in_4bit"):
        quant = {k: q[k] for k in ("load_in_4bit", "bnb_4bit_quant_type",
                                   "bnb_4bit_use_double_quant", "bnb_4bit_compute_dtype")}
    return {"model_class": classes[backbone], "model_id": m["model_id"], "torch_dtype": m["torch_dtype"],
            "quant": quant,
            "processor_kwargs": {k: m[k] for k in ("min_pixels", "max_pixels") if m.get(k) is not None}}


def load_model(cfg: Dict[str, Any]):
    """Nạp model (4-bit nếu config bật) + processor đúng theo config/biến thể."""
    import torch
    import transformers
    from transformers import AutoProcessor, BitsAndBytesConfig
    plan = model_load_plan(cfg)
    kw = {"device_map": "auto", "torch_dtype": getattr(torch, plan["torch_dtype"])}
    if plan["quant"]:
        q = plan["quant"]
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=q["load_in_4bit"],
            bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        )
    model = getattr(transformers, plan["model_class"]).from_pretrained(plan["model_id"], **kw)
    processor = AutoProcessor.from_pretrained(plan["model_id"], **plan["processor_kwargs"])
    return model, processor


def downscale_image(image_path: str, max_pixels: int, out_dir: str) -> Dict[str, Any]:
    """Thu nhỏ ảnh (giữ tỉ lệ) để w*h <= max_pixels; lưu PNG (không nén mất mát thêm). Trả đường dẫn + kích thước."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, (max_pixels / float(w * h)) ** 0.5)
    nw, nh = max(28, int(w * scale)), max(28, int(h * scale))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0] + f"_max{max_pixels}.png")
    img.resize((nw, nh), Image.BICUBIC).save(out)
    return {"path": out, "orig_size": [w, h], "new_size": [nw, nh]}


def _oom_types():
    try:
        import torch
        return (torch.cuda.OutOfMemoryError,)
    except Exception:
        return ()


def with_oom_fallback(generate_fn: Callable, max_pixels: int, log_path: str,
                      oom_types: Optional[tuple] = None, tmp_dir: str = "/tmp/oom_fallback") -> Callable:
    """
    Bọc generate_fn: nếu MỘT ảnh làm tràn VRAM (độ phân giải mặc định của Qwen2-VL rất cao),
    thu ảnh đó về <= max_pixels rồi sinh lại, và GHI LẠI sự kiện vào log_path (để báo cáo, không giấu).
    Ảnh không tràn VRAM -> giữ nguyên độ phân giải mặc định.
    """
    types = oom_types if oom_types is not None else _oom_types()

    def wrapped(model, processor, image_path, prompt_text, max_new_tokens=32):
        try:
            return generate_fn(model, processor, image_path, prompt_text, max_new_tokens=max_new_tokens)
        except types:
            pass
        # Ra khỏi khối except rồi mới dọn: traceback của lượt hỏng còn giữ tensor lớn, phải nhả trước khi sinh lại
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        small = downscale_image(image_path, max_pixels, tmp_dir)
        ans = generate_fn(model, processor, small["path"], prompt_text, max_new_tokens=max_new_tokens)
        events = []
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as fp:
                events = json.load(fp)
        events.append({"image": os.path.basename(image_path), "orig_size": small["orig_size"],
                       "new_size": small["new_size"], "max_pixels": max_pixels})
        _write_json(log_path, events, indent=2)
        _log(f"  ⚠ tràn VRAM ở {os.path.basename(image_path)} {small['orig_size']} -> chạy lại ở {small['new_size']}")
        return ans
    return wrapped


# --------------------------------------------------------------------------- #
# Ảnh
# --------------------------------------------------------------------------- #
def sync_images(src: str, dst: str, label: str = "") -> str:
    """
    Chép ảnh src -> dst, CHỈ chép file còn thiếu; mỗi file ghi tạm rồi đổi tên (không để lại file dở).
    W07: bản cũ bỏ qua cả bước chép nếu dst đã tồn tại -> thư mục chép dở (Colab bị ngắt) làm
    mất ảnh âm thầm. Giờ luôn đối chiếu tên file với nguồn, nên chạy lại là tự bù đủ.
    """
    names = sorted(n for n in os.listdir(src) if not n.startswith("."))
    os.makedirs(dst, exist_ok=True)
    have = {n for n in os.listdir(dst) if not n.startswith(".")}
    missing = [n for n in names if n not in have]
    if missing:
        t0 = time.time()
        for n in missing:
            tmp = os.path.join(dst, f".{n}.part")
            shutil.copyfile(os.path.join(src, n), tmp)
            os.replace(tmp, os.path.join(dst, n))
        _log(f"  chép {len(missing)} ảnh {label} về {dst} (đã có sẵn {len(names) - len(missing)}) "
             f"trong {time.time() - t0:.0f}s")
    else:
        _log(f"  [có sẵn] đủ {len(names)} ảnh {label} ở {dst}")
    return dst


def prepare_images(dataset: str, drive_root: str) -> str:
    """Đưa ảnh về đĩa Colab (/content) cho nhanh; trả thư mục chứa ảnh."""
    if dataset == "vivqa":
        return sync_images(os.path.join(drive_root, "data/vivqa/images"), "/content/vivqa_images", "ViVQA")
    if dataset == "vitextvqa_official":
        from src.data.prepare_vitextvqa import ensure_images
        info = ensure_images(zip_dir=os.path.join(drive_root, "data/vitextvqa_official"),
                             images_dir="/content/vitextvqa_images")
        return info["image_root"]
    raise ValueError(f"dataset không hỗ trợ: {dataset}")


# --------------------------------------------------------------------------- #
# Chữ ký run: chặn việc "resume" lô cũ của cấu hình khác (vd W05 fp16) vào run mới
# --------------------------------------------------------------------------- #
SIGNATURE_FILE = "run_signature.json"


def adapter_fingerprint(adapter_dir: str) -> Dict[str, Any]:
    """Định danh adapter LoRA bằng nội dung (sha256 trọng số + config), không bằng đường dẫn."""
    w = os.path.join(adapter_dir, "adapter_model.safetensors")
    c = os.path.join(adapter_dir, "adapter_config.json")
    if not (os.path.isfile(w) and os.path.isfile(c)):
        raise FileNotFoundError(f"{adapter_dir} thiếu adapter_model.safetensors / adapter_config.json")
    return {"weights_sha256": _sha256(w), "config_sha256": _sha256(c),
            "weights_mb": round(os.path.getsize(w) / 1e6, 2)}


def run_signature(cfg: Dict[str, Any], data_sha256: str, limit: Optional[int], chunk: int,
                  adapter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dấu vân tay của mọi thứ ảnh hưởng tới dự đoán. Khác chữ ký = run khác.

    adapter=None (zero-shot) -> payload GIỮ NGUYÊN như W07 B1, nên chữ ký B1 không đổi.
    """
    c = copy.deepcopy(cfg)
    c.get("run", {}).pop("name", None)
    c.get("data", {}).pop("test_path", None)        # đường dẫn tuyệt đối đổi theo máy; nội dung đã có sha256
    payload = {"model": c.get("model"), "quantization": c.get("quantization"),
               "prompting": c.get("prompting"), "eval": c.get("eval"),
               "data_sha256": data_sha256, "limit": limit, "chunk": chunk}
    if adapter is not None:
        payload["adapter"] = {k: adapter[k] for k in ("weights_sha256", "config_sha256")}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"sha256": digest, **payload}


def check_or_write_signature(out_dir: str, sig: Dict[str, Any]) -> None:
    """
    Lần đầu: ghi chữ ký. Lần sau: phải trùng.
    Thư mục run CŨ đã có lô nhưng không có chữ ký (run trước W07) -> chặn, vì không chứng minh
    được các lô đó chạy cùng cấu hình (vd lô W05 chạy fp16, config hiện tại bf16).
    """
    path = os.path.join(out_dir, SIGNATURE_FILE)
    chunk_dir = os.path.join(out_dir, "chunks")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fp:
            if json.load(fp).get("sha256") != sig["sha256"]:
                raise ValueError(f"{path} khác cấu hình/dữ liệu hiện tại -> dùng run_name mới")
        return
    if os.path.isdir(chunk_dir) and os.listdir(chunk_dir):
        raise ValueError(f"{out_dir} đã có lô nhưng không có {SIGNATURE_FILE} (run trước W07) "
                         f"-> KHÔNG chạy tiếp được; dùng run_name mới")
    _write_json(path, sig, indent=2)


# --------------------------------------------------------------------------- #
# Chạy theo lô + gộp
# --------------------------------------------------------------------------- #
def run_chunks(samples: List[Dict[str, Any]], cfg: Dict[str, Any], out_dir: str, chunk: int,
               model, processor, generate_fn: Callable = generate_answer) -> int:
    """Chạy từng lô, bỏ qua lô đã có (và kiểm tra id lô cũ khớp dữ liệu hiện tại)."""
    chunk_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    n_chunks = (len(samples) + chunk - 1) // chunk
    for k in range(n_chunks):
        part = samples[k * chunk:(k + 1) * chunk]
        ids = [s["question_id"] for s in part]
        path = os.path.join(chunk_dir, f"chunk_{k:02d}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fp:
                old_ids = [r["question_id"] for r in json.load(fp)["predictions"]]
            if old_ids != ids:
                raise ValueError(f"{path} không khớp dữ liệu hiện tại (data/chunk đổi?) "
                                 f"-> dùng run_name mới hoặc xoá thư mục chunks")
            _log(f"[lô {k + 1}/{n_chunks}] đã có -> bỏ qua")
            continue
        res = run_evaluation(model, processor, part, cfg, generate_fn=generate_fn, verbose_every=100)
        _write_json(path, {"predictions": res["predictions"],
                           "inference_time_sec": res["meta"]["inference_time_sec"],
                           "gpu": res["meta"]["hardware"]["gpu"]})
        _log(f"[lô {k + 1}/{n_chunks}] xong {len(part)} mẫu | "
             f"{res['meta']['inference_time_sec']:.0f}s | EM lô = {res['metrics']['exact_match']:.3f}")
    return n_chunks


def merge_and_report(samples: List[Dict[str, Any]], cfg: Dict[str, Any], out_dir: str,
                     n_chunks: int, extra_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Gộp các lô, kiểm tra đủ + không trùng, tính metric toàn tập, ghi metrics/predictions."""
    records, total_time, gpus = [], 0.0, set()
    for k in range(n_chunks):
        with open(os.path.join(out_dir, "chunks", f"chunk_{k:02d}.json"), encoding="utf-8") as fp:
            d = json.load(fp)
        records += d["predictions"]
        total_time += d["inference_time_sec"]
        gpus.add(d["gpu"])
    if [r["question_id"] for r in records] != [s["question_id"] for s in samples]:
        raise ValueError("Dự đoán gộp không khớp danh sách mẫu (thiếu/thừa/sai thứ tự)")

    metrics = evaluate_predictions(
        [r["prediction"] for r in records], [r["answers"] for r in records],
        metric_names=list(cfg["eval"]["primary_metrics"]) + list(cfg["eval"].get("secondary_metrics", [])),
        anls_threshold=cfg["eval"]["anls_threshold"],
        question_types=[r["question_type"] for r in records])
    m, q = cfg["model"], cfg["quantization"]
    meta = {
        "run_name": cfg["run"]["name"],
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_samples": len(records),
        "question_type_counts": dict(Counter(r["question_type"] for r in records)),
        "prompting_mode": cfg["prompting"]["mode"],
        "decoding": "greedy (do_sample=False)",
        "max_new_tokens": cfg["prompting"].get("max_new_tokens", 32),
        "model_id": m["model_id"],
        "backbone": m.get("backbone", "qwen2_5_vl"),
        "variant": m.get("variant"),
        "quantization": {"4bit": q.get("load_in_4bit"), "type": q.get("bnb_4bit_quant_type"),
                         "double_quant": q.get("bnb_4bit_use_double_quant"),
                         "compute_dtype": q.get("bnb_4bit_compute_dtype")},
        "torch_dtype": m.get("torch_dtype"),
        "min_pixels": m.get("min_pixels"), "max_pixels": m.get("max_pixels"),
        "seed": cfg["run"]["seed"],
        "git_commit": git_commit(),
        "config_file": "configs/qwen_lora.yaml",
        "hardware": {"gpus_seen": sorted(gpus), "platform": platform.platform()},
        "inference_time_sec": round(total_time, 1),
        "sec_per_sample": round(total_time / max(len(records), 1), 3),
        **extra_meta,
    }
    try:
        import torch, transformers
        meta["versions"] = {"transformers": transformers.__version__, "torch": torch.__version__}
    except Exception:
        meta["versions"] = "unavailable"
    _write_json(os.path.join(out_dir, "metrics.json"), {"metrics": metrics, "meta": meta}, indent=2)
    _write_json(os.path.join(out_dir, "predictions.json"), records, indent=2)
    return {"metrics": metrics, "meta": meta}


def run(dataset: str, run_name: str, drive_root: str = DEFAULT_DRIVE_ROOT,
        data_file: Optional[str] = None, out_root: Optional[str] = None,
        config: str = os.path.join(REPO_ROOT, "configs/qwen_lora.yaml"), mode: Optional[str] = None,
        chunk: int = 500, image_dir: Optional[str] = None, limit: Optional[int] = None,
        adapter: Optional[str] = None, variant: Optional[str] = None,
        model=None, processor=None, generate_fn: Callable = generate_answer,
        check_config: bool = True) -> Dict[str, Any]:
    """Toàn bộ quy trình. model/processor/generate_fn truyền vào được để kiểm thử offline."""
    if dataset not in DATASETS:
        raise ValueError(f"--dataset phải là một trong {list(DATASETS)}")
    if variant and adapter:
        raise ValueError("--adapter (train trên Qwen2.5-VL-3B) không dùng được với --variant (backbone khác)")
    data_file = data_file or os.path.join(drive_root, DATASETS[dataset]["data_rel"])
    out_dir = os.path.join(out_root or os.path.join(drive_root, "experiments"), run_name)
    cfg = copy.deepcopy(load_config(config, mode=mode, check=check_config))   # config gốc phải đúng bản chốt
    cfg = apply_variant(cfg, variant)                                            # rồi mới ghi đè tường minh
    cfg["run"]["name"] = run_name
    cfg.setdefault("data", {})                     # run_evaluation ghi meta từ mục này
    cfg["data"]["dataset_name"] = dataset
    cfg["data"]["test_path"] = data_file

    _log(f"== {run_name} | dataset={dataset} | mode={cfg['prompting']['mode']} | model={cfg['model']['model_id']}"
         + (f" | variant={variant}" if variant else ""))
    _log(f"   data: {data_file}")
    _log(f"   out : {out_dir}")
    image_dir = image_dir or prepare_images(dataset, drive_root)
    samples = load_vivqa(data_file, image_dir=image_dir, max_samples=limit)
    missing = [s["image_path"] for s in samples if not os.path.exists(s["image_path"])]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ảnh thiếu, vd {missing[0]}")
    _log(f"   {len(samples)} mẫu, ảnh đủ")
    data_sha = _sha256(data_file)
    ad_info = None
    if adapter:
        ad_info = {"path": adapter, **adapter_fingerprint(adapter)}
        _log(f"   adapter: {adapter} ({ad_info['weights_mb']} MB, sha256 {ad_info['weights_sha256'][:12]}…)")
    check_or_write_signature(out_dir, run_signature(cfg, data_sha, limit, chunk, adapter=ad_info))

    real_model = model is None
    if model is None:
        _log("   nạp model...")
        model, processor = load_model(cfg)
        if adapter:
            # Cùng model gốc 4-bit như B1, chỉ GẮN THÊM LoRA đã train -> khác B1 đúng một biến
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            model.eval()
            _log("   đã gắn adapter LoRA (PEFT, chỉ suy luận)")
    oom_log = os.path.join(out_dir, "oom_fallbacks.json")
    if cfg["model"].get("oom_fallback_max_pixels"):
        generate_fn = with_oom_fallback(generate_fn, cfg["model"]["oom_fallback_max_pixels"], oom_log,
                                        oom_types=None if real_model else (MemoryError,))
    if real_model:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    n_chunks = run_chunks(samples, cfg, out_dir, chunk, model, processor, generate_fn)
    extra: Dict[str, Any] = {}
    if real_model:
        import torch
        if torch.cuda.is_available():
            extra["peak_vram_gb_this_session"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if variant:
        events = json.load(open(oom_log, encoding="utf-8")) if os.path.exists(oom_log) else []
        extra["variant_why"] = MODEL_VARIANTS[variant]["why"]
        extra["oom_fallback_images"] = sorted({e["image"] for e in events})
    result = merge_and_report(samples, cfg, out_dir, n_chunks, extra_meta={**extra,
        "dataset": dataset,
        "data_file": os.path.relpath(data_file, drive_root) if data_file.startswith(drive_root) else data_file,
        "data_file_sha256": data_sha,
        "limit": limit, "chunk_size": chunk,
        "adapter": ad_info,
    })
    mt = result["metrics"]
    f1_txt = f" | token-F1 (phụ)={mt['token_f1']:.4f}" if "token_f1" in mt else ""
    _log(f"== KẾT QUẢ: EM={mt['exact_match']:.4f} | VQA-Acc={mt['vqa_accuracy']:.4f} | "
         f"ANLS={mt['anls']:.4f}{f1_txt} | {result['meta']['inference_time_sec']}s")
    _log(f"   đã lưu -> {out_dir}")
    return result


# --------------------------------------------------------------------------- #
# Kiểm thử offline
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        img = os.path.join(td, "img")
        os.makedirs(img)
        recs = []
        for i in range(7):
            open(os.path.join(img, f"{i}.jpg"), "wb").close()
            recs.append({"question_id": f"test_{i}", "question": f"câu {i}?", "answers": ["đỏ" if i % 2 else "hai"],
                         "image": f"{i}.jpg", "question_type": "unknown"})
        data = os.path.join(td, "data.json")
        _write_json(data, recs)
        cfg_path = os.path.join(td, "cfg.yaml")
        with open(cfg_path, "w", encoding="utf-8") as fp:
            fp.write("run: {name: x, seed: 42}\n"
                     "model: {model_id: fake, torch_dtype: bfloat16, min_pixels: 200704, max_pixels: 401408}\n"
                     "quantization: {enabled: true, load_in_4bit: true, bnb_4bit_quant_type: nf4, "
                     "bnb_4bit_use_double_quant: true, bnb_4bit_compute_dtype: bfloat16}\n"
                     "prompting: {mode: zero_shot, max_new_tokens: 32}\n"
                     "eval: {primary_metrics: [exact_match, vqa_accuracy, anls], secondary_metrics: [token_f1], anls_threshold: 0.5}\n")

        calls = []
        def fake_gen(model, processor, image_path, prompt_text, max_new_tokens=32):
            calls.append(image_path)
            i = int(os.path.basename(image_path).split(".")[0])
            return "Màu đỏ" if i % 2 else "2"          # đúng hết sau chuẩn hoá

        common = dict(dataset="vivqa", run_name="t", drive_root=td, data_file=data,
                      out_root=os.path.join(td, "exp"), config=cfg_path, chunk=3, image_dir=img,
                      model="fake", processor="fake", generate_fn=fake_gen)
        r1 = run(**common)
        assert r1["meta"]["num_samples"] == 7 and abs(r1["metrics"]["exact_match"] - 1.0) < 1e-9
        assert abs(r1["metrics"]["token_f1"] - 1.0) < 1e-9, "token_f1 (metric phụ) phải được tính"
        assert len(calls) == 7 and r1["meta"]["data_file_sha256"] == _sha256(data)

        # resume: chạy lại -> không gọi model thêm lần nào
        calls.clear()
        run(**common)
        assert calls == [], "resume phải bỏ qua các lô đã có"

        # mất 1 lô -> chỉ chạy lại đúng lô đó
        os.remove(os.path.join(td, "exp", "t", "chunks", "chunk_01.json"))
        calls.clear()
        run(**common)
        assert len(calls) == 3

        # đổi dữ liệu mà giữ run_name -> phải báo lỗi, không trộn kết quả
        recs[0]["question_id"] = "test_X"
        _write_json(data, recs)
        try:
            run(**common)
            raise AssertionError("không phát hiện lô cũ lệch dữ liệu")
        except ValueError as exc:   # chữ ký (sha256 dữ liệu) chặn trước; lớp kiểm tra id lô là lớp thứ hai
            assert "khác cấu hình/dữ liệu" in str(exc) or "không khớp" in str(exc), exc

        # đổi CẤU HÌNH (vd prompt) mà giữ run_name -> chữ ký lệch -> chặn
        recs[0]["question_id"] = "test_0"
        _write_json(data, recs)
        try:
            run(**{**common, "mode": "ocr"})
            raise AssertionError("không phát hiện chạy tiếp với cấu hình khác")
        except ValueError as exc:
            assert "chữ ký" in str(exc) or "khác cấu hình" in str(exc), exc

        # thư mục run cũ (trước W07): có lô nhưng không có chữ ký -> chặn
        legacy = os.path.join(td, "exp", "legacy", "chunks")
        os.makedirs(legacy)
        _write_json(os.path.join(legacy, "chunk_00.json"), {"predictions": []})
        try:
            run(**{**common, "run_name": "legacy"})
            raise AssertionError("không chặn thư mục run cũ không có chữ ký")
        except ValueError as exc:
            assert "không có" in str(exc), exc

        # config sai bản chuẩn -> chặn
        bad = cfg_path.replace("cfg.yaml", "bad.yaml")
        with open(cfg_path, encoding="utf-8") as fp:
            open(bad, "w", encoding="utf-8").write(fp.read().replace("401408", "1003520"))
        try:
            load_config(bad)
            raise AssertionError("không chặn config sai")
        except ValueError:
            pass

        # dtype cũ của W05 (float16) -> phải bị chặn, kể cả khi chỉ sai ở phần lượng tử hoá
        for old, new in [("torch_dtype: bfloat16", "torch_dtype: float16"),
                         ("bnb_4bit_compute_dtype: bfloat16", "bnb_4bit_compute_dtype: float16")]:
            bad2 = cfg_path.replace("cfg.yaml", "bad2.yaml")
            with open(cfg_path, encoding="utf-8") as fp:
                open(bad2, "w", encoding="utf-8").write(fp.read().replace(old, new))
            try:
                load_config(bad2)
                raise AssertionError(f"không chặn config sai ({new})")
            except ValueError:
                pass
    # adapter: chữ ký zero-shot giữ nguyên như W07 B1; có adapter -> chữ ký khác; thiếu file -> báo lỗi
    import tempfile as _tf
    base_cfg = {"run": {"name": "x"}, "model": {"m": 1}, "quantization": {"q": 1},
                "prompting": {"mode": "zero_shot"}, "eval": {"e": 1}}
    s0 = run_signature(base_cfg, "sha", None, 500)
    assert "adapter" not in s0 and s0 == run_signature(base_cfg, "sha", None, 500, adapter=None)
    with _tf.TemporaryDirectory() as ad:
        try:
            adapter_fingerprint(ad)
            raise AssertionError("không báo thiếu adapter")
        except FileNotFoundError:
            pass
        open(os.path.join(ad, "adapter_model.safetensors"), "wb").write(b"w1")
        open(os.path.join(ad, "adapter_config.json"), "w").write("{}")
        f1 = adapter_fingerprint(ad)
        s1 = run_signature(base_cfg, "sha", None, 500, adapter={"path": "/x", **f1})
        open(os.path.join(ad, "adapter_model.safetensors"), "wb").write(b"w2")
        s2 = run_signature(base_cfg, "sha", None, 500, adapter={"path": "/y", **adapter_fingerprint(ad)})
        assert s1["sha256"] != s0["sha256"] and s1["sha256"] != s2["sha256"], "adapter phải đổi chữ ký"
    _log("adapter signature OK")
    # sync_images: thư mục đích chép dở (thiếu file + file .part rác) -> chạy lại phải bù đủ, đúng nội dung
    with _tf.TemporaryDirectory() as root:
        src, dst = os.path.join(root, "src"), os.path.join(root, "dst")
        os.makedirs(src); os.makedirs(dst)
        for i in range(5):
            open(os.path.join(src, f"{i}.jpg"), "w").write(f"img{i}")
        open(os.path.join(dst, "0.jpg"), "w").write("img0")
        open(os.path.join(dst, ".3.jpg.part"), "w").write("rác")
        sync_images(src, dst, "test")
        got = sorted(n for n in os.listdir(dst) if not n.startswith("."))
        assert got == [f"{i}.jpg" for i in range(5)], got
        assert open(os.path.join(dst, "4.jpg")).read() == "img4"
        sync_images(src, dst, "test")      # lần 2: không chép gì
    _log("sync_images OK")
    _selftest_variants()
    _log("SELFTEST OK")


def _selftest_variants() -> None:
    """W07 R1: biến thể backbone + dự phòng tràn VRAM + chữ ký B1/B2 đã lưu không đổi."""
    import tempfile
    import yaml
    base = yaml.safe_load("run: {name: x, seed: 42}\n"
                          "model: {backbone: qwen2_5_vl, model_id: Qwen/Qwen2.5-VL-3B-Instruct, torch_dtype: bfloat16, "
                          "min_pixels: 200704, max_pixels: 401408}\n"
                          "quantization: {enabled: true, load_in_4bit: true, bnb_4bit_quant_type: nf4, "
                          "bnb_4bit_use_double_quant: true, bnb_4bit_compute_dtype: bfloat16}\n"
                          "prompting: {mode: zero_shot, max_new_tokens: 32}\n"
                          "eval: {primary_metrics: [exact_match], anls_threshold: 0.5}\n")
    # variant=None -> không đụng cfg; kế hoạch nạp = đúng B1 (Qwen2.5-VL, 4-bit, 512 token)
    assert apply_variant(base, None) is base
    p0 = model_load_plan(base)
    assert p0["model_class"] == "Qwen2_5_VLForConditionalGeneration" and p0["quant"]["bnb_4bit_quant_type"] == "nf4"
    assert p0["processor_kwargs"] == {"min_pixels": 200704, "max_pixels": 401408}
    # paper: Qwen2-VL, KHÔNG lượng tử, processor mặc định, có ngưỡng dự phòng
    cp = apply_variant(base, "qwen2vl7b_paper")
    pp = model_load_plan(cp)
    assert pp["model_class"] == "Qwen2VLForConditionalGeneration" and pp["model_id"] == QWEN2VL7B
    assert pp["quant"] is None and pp["processor_kwargs"] == {}, pp
    assert cp["model"]["oom_fallback_max_pixels"] == 1003520 and base["model"]["model_id"].endswith("3B-Instruct")
    # b1cfg: Qwen2-VL nhưng giữ 4-bit NF4 + 512 token của B1
    cb = apply_variant(base, "qwen2vl7b_b1cfg")
    pb = model_load_plan(cb)
    assert pb["model_class"] == "Qwen2VLForConditionalGeneration" and pb["quant"] == p0["quant"]
    assert pb["processor_kwargs"] == p0["processor_kwargs"] and "oom_fallback_max_pixels" not in cb["model"]
    sigs = {run_signature(c, "sha", None, 500)["sha256"] for c in (base, cp, cb)}
    assert len(sigs) == 3, "mỗi biến thể phải có chữ ký riêng"
    try:
        apply_variant(base, "khong_co")
        raise AssertionError("không chặn biến thể lạ")
    except ValueError:
        pass
    # chữ ký các run W07 đã lưu trong repo phải tái tạo y hệt từ config hiện tại (không làm hỏng resume B1/B2)
    cfg_path = os.path.join(REPO_ROOT, "configs/qwen_lora.yaml")
    checked = 0
    for run_name in ("W07_B1_zeroshot_qwen25vl_vitextvqa_test2000_bf16_L4_seed42",
                     "W07_B2_eval_qlora_r16_ep1_vitextvqa_test2000_bf16_L4_seed42"):
        sp = os.path.join(REPO_ROOT, "experiments", run_name, SIGNATURE_FILE)
        if os.path.exists(sp):
            with open(sp, encoding="utf-8") as fp:
                saved = json.load(fp)
            cfg = load_config(cfg_path)
            assert run_signature(cfg, saved["data_sha256"], saved["limit"], saved["chunk"],
                                 adapter=saved.get("adapter"))["sha256"] == saved["sha256"], run_name
            checked += 1
    _log(f"variant OK (chữ ký đã lưu khớp: {checked}/2)")

    # dự phòng tràn VRAM: ảnh lớn "tràn" -> thu nhỏ, sinh lại, ghi log; ảnh nhỏ giữ nguyên
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        big, small = os.path.join(td, "big.jpg"), os.path.join(td, "small.jpg")
        Image.new("RGB", (4000, 3000)).save(big)
        Image.new("RGB", (640, 480)).save(small)
        seen = []

        def fake_gen(model, processor, image_path, prompt_text, max_new_tokens=32):
            w, h = Image.open(image_path).size
            seen.append((os.path.basename(image_path), w * h))
            if w * h > 2_000_000:
                raise MemoryError("giả lập CUDA OOM")
            return "ok"

        log = os.path.join(td, "oom_fallbacks.json")
        g = with_oom_fallback(fake_gen, 1003520, log, oom_types=(MemoryError,), tmp_dir=os.path.join(td, "tmp"))
        assert g(None, None, small, "q") == "ok" and not os.path.exists(log)
        assert g(None, None, big, "q") == "ok"
        ev = json.load(open(log, encoding="utf-8"))
        assert len(ev) == 1 and ev[0]["image"] == "big.jpg" and ev[0]["new_size"][0] * ev[0]["new_size"][1] <= 1003520
        assert seen[-1][1] <= 1003520 and seen[0] == ("small.jpg", 640 * 480)

        # chạy trọn run() với biến thể + model giả: meta ghi biến thể, log OOM; adapter + biến thể bị chặn
        img = os.path.join(td, "img")
        os.makedirs(img)
        recs = []
        for i in range(4):
            Image.new("RGB", (4000, 3000) if i == 2 else (64, 48)).save(os.path.join(img, f"{i}.jpg"))
            recs.append({"question_id": f"t{i}", "question": f"câu {i}?", "answers": ["ok"],
                         "image": f"{i}.jpg", "question_type": "unknown"})
        data = os.path.join(td, "data.json")
        _write_json(data, recs)
        cfgp = os.path.join(td, "cfg.yaml")
        with open(cfgp, "w", encoding="utf-8") as fp:
            yaml.safe_dump({**base, "eval": {"primary_metrics": ["exact_match", "vqa_accuracy", "anls"],
                                             "secondary_metrics": ["token_f1"], "anls_threshold": 0.5}}, fp,
                           allow_unicode=True)
        common = dict(dataset="vitextvqa_official", run_name="r1", drive_root=td, data_file=data,
                      out_root=os.path.join(td, "exp"), config=cfgp, chunk=3, image_dir=img,
                      model="fake", processor="fake", generate_fn=fake_gen)
        r = run(**common, variant="qwen2vl7b_paper")
        assert r["meta"]["variant"] == "qwen2vl7b_paper" and r["meta"]["model_id"] == QWEN2VL7B
        assert r["meta"]["backbone"] == "qwen2_vl" and r["meta"]["quantization"]["4bit"] is None
        assert r["meta"]["oom_fallback_images"] == ["2.jpg"] and r["metrics"]["exact_match"] == 1.0
        try:
            run(**{**common, "run_name": "r1"})            # cùng thư mục, bỏ biến thể -> chữ ký lệch -> chặn
            raise AssertionError("không chặn trộn biến thể")
        except ValueError:
            pass
        try:
            run(**{**common, "run_name": "r1x"}, variant="qwen2vl7b_b1cfg", adapter="/x")
            raise AssertionError("không chặn adapter + biến thể")
        except ValueError:
            pass
    _log("oom fallback OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inference/eval theo lô, resume được (ViVQA-VLM).")
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--run_name", help="Tên thư mục kết quả, vd W05_zeroshot_qwen25vl_vitextvqa_test2000_seed42")
    ap.add_argument("--drive_root", default=DEFAULT_DRIVE_ROOT)
    ap.add_argument("--data_file", default=None, help="Ghi đè file data mặc định của dataset")
    ap.add_argument("--out_root", default=None, help="Mặc định <drive_root>/experiments")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs/qwen_lora.yaml"))
    ap.add_argument("--mode", default=None, help="Ghi đè prompting.mode (zero_shot, ocr, ...)")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="Chỉ lấy N mẫu đầu (thử nhanh)")
    ap.add_argument("--adapter", default=None,
                    help="Thư mục adapter LoRA (vd <drive>/adapters/<run>/epoch_1) -> đánh giá B2; bỏ trống = zero-shot")
    ap.add_argument("--variant", default=None, choices=list(MODEL_VARIANTS),
                    help="Backbone khác cho R1/ablation (vd qwen2vl7b_paper); bỏ trống = baseline Qwen2.5-VL-3B")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.dataset or not a.run_name:
        ap.error("cần --dataset và --run_name (trừ khi --selftest)")
    run(a.dataset, a.run_name, drive_root=a.drive_root, data_file=a.data_file, out_root=a.out_root,
        config=a.config, mode=a.mode, chunk=a.chunk, limit=a.limit, adapter=a.adapter, variant=a.variant)


if __name__ == "__main__":
    main()
