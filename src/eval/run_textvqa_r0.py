"""
run_textvqa_r0.py — R0: kiểm chứng pipeline bằng cách tái lập một con số CHÍNH THỨC của baseline.

Vấn đề: chưa paper nào công bố Qwen2.5-VL-3B trên ViVQA/ViTextVQA, nên B1 không có số để đối chiếu.
Cách kiểm chứng: chạy đúng model đó trên một benchmark mà technical report CÓ công bố, rồi so.
    Qwen2.5-VL Technical Report (Bai et al., 2025, arXiv:2502.13923), Bảng 5:
    Qwen2.5-VL-3B — TextVQA (val) = 79.3

Hai biến thể (cùng 5.000 câu TextVQA val, cùng prompt, greedy):
    paper : bf16, KHÔNG lượng tử hoá, độ phân giải mặc định của processor  -> so với 79.3
    b1cfg : đúng cấu hình B1 (4-bit NF4, bf16, max_pixels 401.408)          -> đo cái giá của cấu hình tiết kiệm

Luật chốt TRƯỚC khi chạy: biến thể 'paper' trên toàn bộ val (5.000 câu) đạt nếu |VQA-Acc − 79,3| ≤ 2,0 điểm.
(SE của 5.000 câu ≈ 0,6 điểm; nới thêm cho khác biệt prompt/độ phân giải mà report không công bố.)

Prompt theo quy ước lmms-eval cho TextVQA (report không công bố prompt):
    "<question>\\nAnswer the question using a single word or phrase."
Không dùng system prompt tiếng Việt của B1 -> chat template mặc định của Qwen.
Metric: VQA accuracy chính thức (EvalAI): chuẩn hoá đáp án + trung bình leave-one-out trên 10 đáp án.

Dữ liệu: HF lmms-lab/textvqa, split validation (5.000 câu, 10 đáp án/câu); ghi revision vào log.

Chạy (Colab, GPU):
    !python -u -m src.eval.run_textvqa_r0 --variant paper --run_name W07_R0_textvqa_val_qwen25vl3b_paper_bf16_L4
    !python -u -m src.eval.run_textvqa_r0 --variant b1cfg --run_name W07_R0b_textvqa_val_qwen25vl3b_b1cfg_L4
Kiểm thử offline:
    python -m src.eval.run_textvqa_r0 --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/ViVQA-VLM"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
HF_REPO = "lmms-lab/textvqa"
SPLIT = "validation"
PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."
MAX_NEW_TOKENS = 32
REPORTED = {"value": 79.3, "benchmark": "TextVQA val", "model": "Qwen2.5-VL-3B",
            "source": "Qwen2.5-VL Technical Report (Bai et al., 2025, arXiv:2502.13923), Table 5"}
TOLERANCE = 2.0          # điểm phần trăm, chốt trước khi chạy
VARIANTS = {
    "paper": {"quant": None, "torch_dtype": "bfloat16", "min_pixels": None, "max_pixels": None},
    "b1cfg": {"quant": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "bfloat16"},
              "torch_dtype": "bfloat16", "min_pixels": 200704, "max_pixels": 401408},
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: str, obj: Any, indent: Optional[int] = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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


# --------------------------------------------------------------------------- #
# 1) VQA accuracy chính thức (bộ chuẩn hoá EvalAI, giống VQA/TextVQA eval + lmms-eval)
# --------------------------------------------------------------------------- #
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
    "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
    "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've",
}
_NUMBER_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
               "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
          ">", "<", "@", "`", ",", "?", "!"]


def _process_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or (re.search(_COMMA_STRIP, text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return _PERIOD_STRIP.sub("", out, re.UNICODE)


def _process_digit_article(text: str) -> str:
    words = []
    for w in text.lower().split():
        w = _NUMBER_MAP.setdefault(w, w)
        if w not in _ARTICLES:
            words.append(w)
    return " ".join(_CONTRACTIONS.get(w, w) for w in words)


def normalize_en(answer: str) -> str:
    """Chuẩn hoá đáp án kiểu EvalAI (dùng cho cả dự đoán lẫn đáp án chuẩn)."""
    a = (answer or "").replace("\n", " ").replace("\t", " ").strip().lower()
    return _process_digit_article(_process_punctuation(a)).strip()


def vqa_accuracy_official(pred: str, answers: List[str]) -> float:
    """Trung bình leave-one-out trên 10 đáp án: min(1, số người khác trả lời trùng / 3)."""
    p = normalize_en(pred)
    gts = [normalize_en(a) for a in answers]
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        accs.append(min(1.0, sum(g == p for g in others) / 3.0))
    return sum(accs) / len(accs) if accs else 0.0


# --------------------------------------------------------------------------- #
# 2) Dữ liệu: tải split validation từ HF, ghi ảnh ra /content, ghi bản chuẩn hoá lên Drive
# --------------------------------------------------------------------------- #
def prepare_textvqa(drive_root: str, local_dir: str = "/content/textvqa_val") -> Dict[str, Any]:
    """Trả {'records': [...], 'data_file': path, 'revision': sha, 'image_dir': local_dir}."""
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    out_dir = os.path.join(drive_root, "data", "textvqa_val")
    data_file = os.path.join(out_dir, "val.json")
    marker = os.path.join(local_dir, ".extracted_ok")
    api = HfApi()
    revision = api.dataset_info(HF_REPO).sha
    if os.path.exists(data_file) and os.path.exists(marker):
        recs = json.load(open(data_file, encoding="utf-8"))
        _log(f"  [có sẵn] {len(recs)} câu TextVQA val, ảnh ở {local_dir}")
        rep = json.load(open(os.path.join(out_dir, "prep_report.json"), encoding="utf-8"))
        return {"records": recs, "data_file": data_file, "revision": rep["revision"], "image_dir": local_dir}

    files = sorted(f for f in api.list_repo_files(HF_REPO, repo_type="dataset", revision=revision)
                   if f.endswith(".parquet") and SPLIT in f)
    if not files:
        raise FileNotFoundError(f"Không thấy file parquet split '{SPLIT}' trong {HF_REPO}")
    t0 = time.time()
    os.makedirs(local_dir, exist_ok=True)
    recs, n_img = [], 0
    for f in files:
        path = hf_hub_download(HF_REPO, f, repo_type="dataset", revision=revision)
        df = pd.read_parquet(path)
        for row in df.itertuples(index=False):
            img_name = f"{row.image_id}.jpg"
            img_path = os.path.join(local_dir, img_name)
            if not os.path.exists(img_path):
                img = row.image
                data = img["bytes"] if isinstance(img, dict) else img
                # Ghi NGUYÊN byte gốc (không nén lại) để ảnh giống hệt bản phát hành; PIL nhận dạng theo nội dung
                with open(img_path + ".part", "wb") as fp:
                    fp.write(data)
                os.replace(img_path + ".part", img_path)
                n_img += 1
            recs.append({"question_id": int(row.question_id), "question": str(row.question),
                         "answers": [str(a) for a in row.answers], "image": img_name,
                         "image_id": str(row.image_id)})
    recs.sort(key=lambda r: r["question_id"])
    _write_json(data_file, recs)
    _write_json(os.path.join(out_dir, "prep_report.json"),
                {"hf_repo": HF_REPO, "split": SPLIT, "revision": revision, "files": files,
                 "num_questions": len(recs), "num_images": len({r["image_id"] for r in recs}),
                 "data_sha256": _sha256(data_file), "prepared_utc": datetime.now(timezone.utc).isoformat()})
    with open(marker, "w") as fp:
        fp.write(revision)
    _log(f"  tải {HF_REPO}@{revision[:10]}: {len(recs)} câu, ghi {n_img} ảnh trong {time.time() - t0:.0f}s")
    return {"records": recs, "data_file": data_file, "revision": revision, "image_dir": local_dir}


# --------------------------------------------------------------------------- #
# 3) Model + sinh câu trả lời
# --------------------------------------------------------------------------- #
def load_model(variant: str):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    v = VARIANTS[variant]
    kw = {"torch_dtype": getattr(torch, v["torch_dtype"]), "device_map": "auto"}
    if v["quant"]:
        q = v["quant"]
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **kw)
    pkw = {k: v[k] for k in ("min_pixels", "max_pixels") if v[k] is not None}
    processor = AutoProcessor.from_pretrained(MODEL_ID, **pkw)
    return model, processor


def generate(model, processor, image_path: str, question: str) -> str:
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(image_path).convert("RGB")},
        {"type": "text", "text": question + PROMPT_SUFFIX}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True,
                       return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()


# --------------------------------------------------------------------------- #
# 4) Chạy theo lô (resume được) + chấm
# --------------------------------------------------------------------------- #
def run_signature(variant: str, data_sha256: str, revision: str, limit: Optional[int]) -> Dict[str, Any]:
    payload = {"model_id": MODEL_ID, "variant": VARIANTS[variant], "prompt_suffix": PROMPT_SUFFIX,
               "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy", "hf_repo": HF_REPO,
               "revision": revision, "data_sha256": data_sha256, "limit": limit}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {"sha256": digest, **payload}


def check_or_write_signature(out_dir: str, sig: Dict[str, Any]) -> None:
    path = os.path.join(out_dir, "run_signature.json")
    if os.path.exists(path):
        if json.load(open(path, encoding="utf-8")).get("sha256") != sig["sha256"]:
            raise ValueError(f"{path} khác cấu hình/dữ liệu hiện tại -> dùng run_name mới")
        return
    _write_json(path, sig)


def verdict(acc_pct: float, variant: str, n: int, n_full: int) -> str:
    if variant != "paper":
        return "không áp luật (biến thể đo chi phí cấu hình B1)"
    if n != n_full:
        return f"không áp luật (chỉ {n}/{n_full} câu; luật cần toàn bộ val)"
    gap = acc_pct - REPORTED["value"]
    ok = abs(gap) <= TOLERANCE
    return (f"{'TÁI LẬP ĐƯỢC' if ok else 'LỆCH'}: {acc_pct:.2f} so với {REPORTED['value']} "
            f"(chênh {gap:+.2f}, ngưỡng ±{TOLERANCE})")


def run(variant: str, run_name: str, drive_root: str = DEFAULT_DRIVE_ROOT, out_root: Optional[str] = None,
        limit: Optional[int] = None, chunk: int = 500, data: Optional[Dict[str, Any]] = None,
        model=None, processor=None, generate_fn: Callable = generate) -> Dict[str, Any]:
    import torch
    if variant not in VARIANTS:
        raise ValueError(f"--variant phải là một trong {list(VARIANTS)}")
    out_dir = os.path.join(out_root or os.path.join(drive_root, "experiments"), run_name)
    data = data or prepare_textvqa(drive_root)
    recs_all = data["records"]
    recs = recs_all[:limit] if limit else recs_all
    missing = [r["image"] for r in recs if not os.path.exists(os.path.join(data["image_dir"], r["image"]))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ảnh thiếu, vd {missing[0]}")
    data_sha = _sha256(data["data_file"])
    check_or_write_signature(out_dir, run_signature(variant, data_sha, data["revision"], limit))
    _log(f"== {run_name} | variant={variant} | {len(recs)} câu | {VARIANTS[variant]}")

    if model is None:
        _log("   nạp model...")
        model, processor = load_model(variant)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    chunk_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    n_chunks = (len(recs) + chunk - 1) // chunk
    t_infer = 0.0
    for k in range(n_chunks):
        part = recs[k * chunk:(k + 1) * chunk]
        path = os.path.join(chunk_dir, f"chunk_{k:02d}.json")
        if os.path.exists(path):
            done = json.load(open(path, encoding="utf-8"))
            if [d["question_id"] for d in done["preds"]] == [r["question_id"] for r in part]:
                _log(f"   [lô {k + 1}/{n_chunks}] đã có -> bỏ qua")
                continue
            raise ValueError(f"{path} không khớp danh sách câu -> dùng run_name mới")
        t0 = time.time()
        preds = [{"question_id": r["question_id"], "question": r["question"], "answers": r["answers"],
                  "prediction": generate_fn(model, processor, os.path.join(data["image_dir"], r["image"]),
                                            r["question"])} for r in part]
        dt = time.time() - t0
        t_infer += dt
        _write_json(path, {"preds": preds, "sec": round(dt, 1)}, indent=None)
        _log(f"   [lô {k + 1}/{n_chunks}] xong {len(part)} câu trong {dt:.0f}s ({dt / max(len(part), 1):.2f} s/câu)")

    preds, secs = [], 0.0
    for k in range(n_chunks):
        c = json.load(open(os.path.join(chunk_dir, f"chunk_{k:02d}.json"), encoding="utf-8"))
        preds += c["preds"]
        secs += c.get("sec", 0.0)
    for p in preds:
        p["vqa_acc"] = vqa_accuracy_official(p["prediction"], p["answers"])
    acc = 100 * sum(p["vqa_acc"] for p in preds) / len(preds)
    peak = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    try:
        import subprocess
        commit = subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = "unknown"
    meta = {"run_name": run_name, "variant": variant, "variant_config": VARIANTS[variant], "model_id": MODEL_ID,
            "hf_repo": HF_REPO, "split": SPLIT, "revision": data["revision"], "data_sha256": data_sha,
            "num_samples": len(preds), "num_full_split": len(recs_all), "limit": limit,
            "prompt_suffix": PROMPT_SUFFIX, "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy",
            "metric": "VQA accuracy (EvalAI normalization, leave-one-out over 10 answers)",
            "inference_time_sec": round(secs, 1), "sec_per_sample": round(secs / max(len(preds), 1), 3),
            "peak_vram_gb_this_session": round(peak, 2), "git_commit": commit,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "platform": platform.platform(), "finished_utc": datetime.now(timezone.utc).isoformat(),
            "reported": REPORTED, "tolerance_points": TOLERANCE}
    v = verdict(acc, variant, len(preds), len(recs_all))
    result = {"metrics": {"vqa_accuracy": acc / 100}, "verdict": v, "meta": meta}
    _write_json(os.path.join(out_dir, "metrics.json"), result)
    _write_json(os.path.join(out_dir, "predictions.json"), preds)
    _log(f"== KẾT QUẢ: VQA-Acc = {acc:.2f} | {v}")
    return result


# --------------------------------------------------------------------------- #
# 5) Kiểm thử offline
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import tempfile
    # metric: các trường hợp kinh điển của VQA eval
    assert vqa_accuracy_official("two", ["2"] * 10) == 1.0
    assert vqa_accuracy_official("The cat.", ["cat"] * 10) == 1.0
    assert abs(vqa_accuracy_official("cat", ["cat"] * 3 + ["dog"] * 7) - 0.9) < 1e-9
    assert abs(vqa_accuracy_official("cat", ["cat"] * 1 + ["dog"] * 9) - 0.3) < 1e-9
    assert vqa_accuracy_official("dog", ["cat"] * 10) == 0.0
    assert normalize_en("Don't!") == normalize_en("dont")
    assert normalize_en("1,000") == "1000" and normalize_en("Coca-Cola") == "coca cola"
    _log("metric OK")
    # luật
    assert verdict(79.0, "paper", 5000, 5000).startswith("TÁI LẬP")
    assert verdict(76.0, "paper", 5000, 5000).startswith("LỆCH")
    assert verdict(79.0, "paper", 100, 5000).startswith("không áp luật")
    assert verdict(70.0, "b1cfg", 5000, 5000).startswith("không áp luật")
    _log("verdict OK")
    # chạy theo lô + resume + chữ ký, với model giả
    with tempfile.TemporaryDirectory() as td:
        img = os.path.join(td, "img")
        os.makedirs(img)
        recs = []
        for i in range(7):
            open(os.path.join(img, f"{i}.jpg"), "wb").close()
            recs.append({"question_id": i, "question": f"q{i}", "answers": ["yes"] * 10, "image": f"{i}.jpg"})
        df = os.path.join(td, "val.json")
        _write_json(df, recs)
        data = {"records": recs, "data_file": df, "revision": "rev", "image_dir": img}
        calls = []
        fake = lambda m, p, path, q: (calls.append(q), "Yes." if int(q[1:]) % 2 == 0 else "no")[1]
        r1 = run("paper", "r", out_root=td, chunk=3, data=data, model=object(), processor=None, generate_fn=fake)
        assert abs(r1["metrics"]["vqa_accuracy"] - 4 / 7) < 1e-9, r1["metrics"]
        n_calls = len(calls)
        run("paper", "r", out_root=td, chunk=3, data=data, model=object(), processor=None, generate_fn=fake)
        assert len(calls) == n_calls, "lô đã xong phải được bỏ qua"
        try:
            run("b1cfg", "r", out_root=td, chunk=3, data=data, model=object(), processor=None, generate_fn=fake)
            raise AssertionError("không chặn đổi biến thể cùng run_name")
        except ValueError:
            pass
    _log("run/resume/signature OK")
    _log("SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="R0: tái lập TextVQA val của Qwen2.5-VL-3B (kiểm chứng pipeline).")
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--run_name")
    ap.add_argument("--drive_root", default=DEFAULT_DRIVE_ROOT)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Chỉ N câu đầu (thử nhanh; luật chỉ áp cho toàn bộ val)")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.variant or not a.run_name:
        ap.error("cần --variant và --run_name (trừ khi --selftest)")
    run(a.variant, a.run_name, drive_root=a.drive_root, out_root=a.out_root, limit=a.limit, chunk=a.chunk)


if __name__ == "__main__":
    main()
