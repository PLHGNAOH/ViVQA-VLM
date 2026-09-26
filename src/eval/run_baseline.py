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
CANONICAL = {"max_pixels": 401408, "torch_dtype": "float16"}   # chốt W05 (commit 37da80a)


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
                raise ValueError(f"Config {k}={m.get(k)} khác bản chuẩn W05 ({v}) -> git pull?")
    if mode:
        cfg["prompting"]["mode"] = mode
    return cfg


def load_model(cfg: Dict[str, Any]):
    """Nạp model 4-bit + processor đúng theo config (giống notebook 01, phần 3)."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    m, q = cfg["model"], cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        m["model_id"], quantization_config=bnb, device_map="auto",
        torch_dtype=getattr(torch, m["torch_dtype"]))
    processor = AutoProcessor.from_pretrained(
        m["model_id"], min_pixels=m["min_pixels"], max_pixels=m["max_pixels"])
    return model, processor


# --------------------------------------------------------------------------- #
# Ảnh
# --------------------------------------------------------------------------- #
def prepare_images(dataset: str, drive_root: str) -> str:
    """Đưa ảnh về đĩa Colab (/content) cho nhanh; trả thư mục chứa ảnh."""
    if dataset == "vivqa":
        src, dst = os.path.join(drive_root, "data/vivqa/images"), "/content/vivqa_images"
        if not os.path.exists(dst):
            t0 = time.time()
            shutil.copytree(src, dst)
            _log(f"  chép {len(os.listdir(dst))} ảnh ViVQA về {dst} trong {time.time() - t0:.0f}s")
        return dst
    if dataset == "vitextvqa_official":
        from src.data.prepare_vitextvqa import ensure_images
        info = ensure_images(zip_dir=os.path.join(drive_root, "data/vitextvqa_official"),
                             images_dir="/content/vitextvqa_images")
        return info["image_root"]
    raise ValueError(f"dataset không hỗ trợ: {dataset}")


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
        metric_names=cfg["eval"]["primary_metrics"], anls_threshold=cfg["eval"]["anls_threshold"],
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
        model=None, processor=None, generate_fn: Callable = generate_answer,
        check_config: bool = True) -> Dict[str, Any]:
    """Toàn bộ quy trình. model/processor/generate_fn truyền vào được để kiểm thử offline."""
    if dataset not in DATASETS:
        raise ValueError(f"--dataset phải là một trong {list(DATASETS)}")
    data_file = data_file or os.path.join(drive_root, DATASETS[dataset]["data_rel"])
    out_dir = os.path.join(out_root or os.path.join(drive_root, "experiments"), run_name)
    cfg = copy.deepcopy(load_config(config, mode=mode, check=check_config))
    cfg["run"]["name"] = run_name
    cfg.setdefault("data", {})                     # run_evaluation ghi meta từ mục này
    cfg["data"]["dataset_name"] = dataset
    cfg["data"]["test_path"] = data_file

    _log(f"== {run_name} | dataset={dataset} | mode={cfg['prompting']['mode']}")
    _log(f"   data: {data_file}")
    _log(f"   out : {out_dir}")
    image_dir = image_dir or prepare_images(dataset, drive_root)
    samples = load_vivqa(data_file, image_dir=image_dir, max_samples=limit)
    missing = [s["image_path"] for s in samples if not os.path.exists(s["image_path"])]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ảnh thiếu, vd {missing[0]}")
    _log(f"   {len(samples)} mẫu, ảnh đủ")

    if model is None:
        _log("   nạp model...")
        model, processor = load_model(cfg)
    n_chunks = run_chunks(samples, cfg, out_dir, chunk, model, processor, generate_fn)
    result = merge_and_report(samples, cfg, out_dir, n_chunks, extra_meta={
        "dataset": dataset,
        "data_file": os.path.relpath(data_file, drive_root) if data_file.startswith(drive_root) else data_file,
        "data_file_sha256": _sha256(data_file),
        "limit": limit, "chunk_size": chunk,
    })
    mt = result["metrics"]
    _log(f"== KẾT QUẢ: EM={mt['exact_match']:.4f} | VQA-Acc={mt['vqa_accuracy']:.4f} | "
         f"ANLS={mt['anls']:.4f} | {result['meta']['inference_time_sec']}s")
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
                     "model: {model_id: fake, torch_dtype: float16, min_pixels: 200704, max_pixels: 401408}\n"
                     "quantization: {enabled: true, load_in_4bit: true, bnb_4bit_quant_type: nf4, "
                     "bnb_4bit_use_double_quant: true, bnb_4bit_compute_dtype: float16}\n"
                     "prompting: {mode: zero_shot, max_new_tokens: 32}\n"
                     "eval: {primary_metrics: [exact_match, vqa_accuracy, anls], anls_threshold: 0.5}\n")

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
        except ValueError as exc:
            assert "không khớp" in str(exc)

        # config sai bản chuẩn -> chặn
        bad = cfg_path.replace("cfg.yaml", "bad.yaml")
        with open(cfg_path, encoding="utf-8") as fp:
            open(bad, "w", encoding="utf-8").write(fp.read().replace("401408", "1003520"))
        try:
            load_config(bad)
            raise AssertionError("không chặn config sai")
        except ValueError:
            pass
    _log("SELFTEST OK")


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
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.dataset or not a.run_name:
        ap.error("cần --dataset và --run_name (trừ khi --selftest)")
    run(a.dataset, a.run_name, drive_root=a.drive_root, data_file=a.data_file, out_root=a.out_root,
        config=a.config, mode=a.mode, chunk=a.chunk, limit=a.limit)


if __name__ == "__main__":
    main()
