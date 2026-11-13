"""
token_budget.py — W15 (góp ý Review 2, G1): mỗi câu hỏi tốn bao nhiêu token, so với sức chứa của model.

VẤN ĐỀ (hội đồng Review 2, 29/10/2026): slide kiến trúc chưa nói rõ ảnh, câu hỏi và dòng OCR
mỗi phần chiếm bao nhiêu token, và tổng đó so với context window của Qwen2.5-VL-3B ra sao.

CÁCH ĐO (chính xác, không ước lượng bằng số ký tự):
  - Token ẢNH: đi đúng đường của run_eval.generate_answer — qwen_vl_utils.fetch_image (smart_resize lần 1,
    mặc định của qwen_vl_utils) rồi image processor của Qwen (smart_resize lần 2 với min/max_pixels của
    config: 200.704 / 401.408 px = 256 / 512 token). Số token = (H/14)·(W/14)/4 (patch 14, gộp 2×2).
    Chỉ cần kích thước ảnh (đọc header), không giải mã ảnh -> nhanh.
  - Token CHỮ: tokenizer Qwen trên chuỗi chat template thật (system + câu hỏi [+ dòng OCR]),
    chuỗi này có đúng 1 token <|image_pad|>; processor nở token đó thành N token ảnh. Vậy:
        tổng_input = len(ids(chat_template)) − 1 + N_ảnh
  - Token ĐÁP ÁN (lúc train): build_target_text(pick_answer(s)) như train_qlora.
  - KIỂM CHỨNG: với một mẫu ngẫu nhiên mỗi tập (seed 42), chạy NGUYÊN đường processor(text, images)
    như lúc suy luận và so từng câu: tổng input_ids và số <|image_pad|> phải TRÙNG HỆT. Lệch dù 1 câu
    -> tập đó tự chuyển sang đo bằng đường đầy đủ cho mọi câu (chậm hơn nhưng vẫn đúng).

Kết quả: experiments/W15_token_budget/{summary.json, summary.md, per_sample_<split>.csv, validation.json}

Chạy (Colab, CPU là đủ — không cần GPU):
  python -m src.eval.token_budget --run --drive_root /content/drive/MyDrive/ViVQA-VLM \
         --out /content/drive/MyDrive/ViVQA-VLM/experiments/W15_token_budget
Kiểm thử offline (không mạng, không transformers):
  python -m src.eval.token_budget --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

TOKEN_BUDGET_VERSION = "token-budget-v1"
OUT_NAME = "W15_token_budget"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"      # = experiments/W07_config_freeze_L4_seed42
CANON_MIN_PIXELS = 200704       # 256 token — configs/qwen_lora.yaml
CANON_MAX_PIXELS = 401408       # 512 token — chốt W07
ALT_MAX_PIXELS = 1003520        # 1.280 token — mức đã thử ở W06/W07
MAX_NEW_TOKENS = 32             # configs/qwen_lora.yaml prompting.max_new_tokens
OCR_BACKEND = "paddledet_vietocr"
OCR_MAX_CHARS = 600
VALIDATE_N = 150                # số câu kiểm chứng bằng đường processor đầy đủ, mỗi tập
SEED = 42

# Tập đo: model chính thức dùng tập nào, ở mode prompt nào (configs/official_models.json).
#   C3 (ViVQA) = zero_shot · B1/B2 = zero_shot · B3-full (ViTextVQA) = ocr · SH (cả hai) = ocr
SPLITS: Dict[str, Dict[str, Any]] = {
    "vivqa_test":      {"dataset": "vivqa", "data_rel": "data/vivqa/test.json", "train": False},
    "vivqa_train":     {"dataset": "vivqa", "data_rel": "data/vivqa/train.json", "train": True},
    "vitextvqa_test":  {"dataset": "vitextvqa_official", "data_rel": "data/vitextvqa_official/test.json", "train": False},
    "vitextvqa_train": {"dataset": "vitextvqa_official", "data_rel": "data/vitextvqa_official/train.json", "train": True},
}
MODES = ("zero_shot", "ocr")
PCTS = (50, 95, 99)


# --------------------------------------------------------------------------- #
# 1) Hàm thuần (kiểm thử offline được)
# --------------------------------------------------------------------------- #
def percentile(xs: List[float], q: float) -> float:
    """Percentile kiểu 'nearest-rank' (không nội suy) -> luôn là một giá trị có thật trong dữ liệu."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(1, math.ceil(q / 100.0 * len(s)))
    return float(s[k - 1])


def describe(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0}
    d = {"n": len(xs), "min": float(min(xs)), "mean": round(sum(xs) / len(xs), 1), "max": float(max(xs))}
    for q in PCTS:
        d[f"p{q}"] = percentile(xs, q)
    return d


def visual_tokens(h: int, w: int, resize1: Callable[[int, int], Tuple[int, int]],
                  resize2: Callable[[int, int], Tuple[int, int]], patch: int = 14, merge: int = 2) -> Tuple[int, int, int]:
    """(H, W, N_token) sau 2 lần smart_resize. N = (H/patch)·(W/patch)/merge²."""
    h1, w1 = resize1(h, w)
    h2, w2 = resize2(h1, w1)
    return h2, w2, (h2 // patch) * (w2 // patch) // (merge * merge)


def kv_cache_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int, bytes_per_value: int = 2) -> int:
    """KV cache của 1 token: K và V, mỗi lớp, mỗi KV-head, head_dim giá trị, bf16 = 2 byte."""
    return 2 * num_layers * num_kv_heads * head_dim * bytes_per_value


def assemble_prompt_parts(sample: Dict[str, Any], mode: str) -> str:
    """Phần chữ của prompt — đúng hàm prompt_for_sample mà train/eval dùng chung."""
    from src.prompting.builder import prompt_for_sample
    prompting = {"mode": mode, "ocr": {"template": "v1"}} if mode == "ocr" else {"mode": mode}
    return prompt_for_sample(sample, prompting)


def summarize(rows: List[Dict[str, Any]], context_window: int, kv_per_token: int,
              max_new_tokens: int = MAX_NEW_TOKENS, is_train: bool = False) -> Dict[str, Any]:
    """Thống kê 1 tập × 1 mode. rows: mỗi câu một dict số nguyên (xem measure_split)."""
    img = [r["img_tok"] for r in rows]
    out: Dict[str, Any] = {
        "n_questions": len(rows),
        "n_images": len({r["image"] for r in rows}),
        "image_tokens": describe(img),
        "pct_images_capped_512": round(100 * sum(r["native_tok"] > r["img_tok"] for r in rows) / max(len(rows), 1), 2),
        "pct_images_raised_to_256": round(100 * sum(r["native_tok"] < r["img_tok"] for r in rows) / max(len(rows), 1), 2),
        "native_image_tokens": describe([r["native_tok"] for r in rows]),
        "image_tokens_if_1280_cap": describe([r["alt_tok"] for r in rows]),
        "question_tokens": describe([r["q_tok"] for r in rows]),
        "text_tokens": describe([r["text_tok"] for r in rows]),
        "input_tokens": describe([r["input_tok"] for r in rows]),
        "max_sequence_infer": describe([r["input_tok"] + max_new_tokens for r in rows]),
    }
    ocr = [r["ocr_tok"] for r in rows]
    if any(ocr):
        out["ocr_line_tokens"] = describe(ocr)
        out["pct_questions_with_ocr_line"] = round(100 * sum(1 for x in ocr if x > 0) / len(rows), 2)
        out["ocr_line_tokens_when_present"] = describe([x for x in ocr if x > 0])
    if is_train:
        out["answer_tokens"] = describe([r["ans_tok"] for r in rows])
        out["sequence_train"] = describe([r["input_tok"] + r["ans_tok"] for r in rows])
    longest = max(r["input_tok"] + max(max_new_tokens, r.get("ans_tok", 0)) for r in rows)
    out["context_window"] = context_window
    out["longest_sequence"] = longest
    out["pct_of_context_longest"] = round(100 * longest / context_window, 3)
    out["kv_cache_mb_longest"] = round(longest * kv_per_token / 2 ** 20, 2)
    out["image_share_of_input_p50"] = round(100 * percentile([r["img_tok"] / r["input_tok"] for r in rows], 50), 1)
    return out


def render_md(summary: Dict[str, Any]) -> str:
    """Bảng markdown gọn cho thesis/slide."""
    c = summary["model"]
    lines = [f"# Token budget — {c['model_id']} ({TOKEN_BUDGET_VERSION})", "",
             f"Context window (max_position_embeddings): **{c['context_window']:,}** token · "
             f"KV cache {c['kv_cache_bytes_per_token']:,} byte/token (bf16) · "
             f"token ảnh: min {CANON_MIN_PIXELS:,} px = 256, max {CANON_MAX_PIXELS:,} px = 512 · "
             f"max_new_tokens {MAX_NEW_TOKENS}", "",
             "| Tập | Mode | Câu | Token ảnh p50 / max | % ảnh bị chặn 512 | Câu hỏi p50 / max | Dòng OCR p50 / max (khi có) "
             "| Input p50 / p95 / max | Chuỗi dài nhất | % context | KV cache dài nhất |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, s in summary["results"].items():
        split, mode = key.rsplit("|", 1)
        o = s.get("ocr_line_tokens_when_present")
        ocr = f"{o['p50']:.0f} / {o['max']:.0f}" if o else "—"
        it, q, inp = s["image_tokens"], s["question_tokens"], s["input_tokens"]
        lines.append(f"| {split} | {mode} | {s['n_questions']:,} | {it['p50']:.0f} / {it['max']:.0f} | "
                     f"{s['pct_images_capped_512']} | {q['p50']:.0f} / {q['max']:.0f} | {ocr} | "
                     f"{inp['p50']:.0f} / {inp['p95']:.0f} / {inp['max']:.0f} | {s['longest_sequence']:,} | "
                     f"{s['pct_of_context_longest']} | {s['kv_cache_mb_longest']} MB |")
    lines += ["", "Nếu không chặn ở 512 token (độ phân giải gốc qua smart_resize của qwen_vl_utils):", "",
              "| Tập | Token ảnh gốc p50 / p95 / max | Nếu chặn 1.280: p50 / max |", "|---|---|---|"]
    for key, s in summary["results"].items():
        if key.endswith("|zero_shot"):
            n, a = s["native_image_tokens"], s["image_tokens_if_1280_cap"]
            lines.append(f"| {key.split('|')[0]} | {n['p50']:.0f} / {n['p95']:.0f} / {n['max']:.0f} | "
                         f"{a['p50']:.0f} / {a['max']:.0f} |")
    v = summary.get("validation", {})
    lines += ["", f"Kiểm chứng bằng đường processor đầy đủ: {v.get('n_checked', 0)} câu, "
                  f"lệch {v.get('n_mismatch', 0)} câu ({'ĐẠT' if v.get('ok') else 'KHÔNG ĐẠT'})."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 2) Phần cần transformers / qwen_vl_utils (import trễ)
# --------------------------------------------------------------------------- #
def load_qwen_bits() -> Dict[str, Any]:
    """Processor + config + 2 hàm smart_resize đúng như đường suy luận."""
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as hf_smart_resize
    from qwen_vl_utils import vision_process as vp

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                              min_pixels=CANON_MIN_PIXELS, max_pixels=CANON_MAX_PIXELS)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    ip = processor.image_processor
    p_min, p_max = int(ip.size["shortest_edge"]), int(ip.size["longest_edge"])   # transformers 5.x: size dict
    if (p_min, p_max) != (CANON_MIN_PIXELS, CANON_MAX_PIXELS):
        raise ValueError(f"processor min/max_pixels = {p_min}/{p_max}, khác config {CANON_MIN_PIXELS}/{CANON_MAX_PIXELS}")
    patch, merge = int(ip.patch_size), int(ip.merge_size)
    factor = patch * merge
    # qwen_vl_utils 0.0.14: fetch_image(ele, image_patch_size=14) -> factor 14·SPATIAL_MERGE_SIZE,
    # min/max = IMAGE_MIN/MAX_TOKEN_NUM·factor² (bản cũ: IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS — cùng giá trị)
    vu_factor = 14 * getattr(vp, "SPATIAL_MERGE_SIZE", 2)
    vu_min = getattr(vp, "IMAGE_MIN_TOKEN_NUM", None)
    vu_min = vu_min * vu_factor ** 2 if vu_min else getattr(vp, "MIN_PIXELS", 4 * 28 * 28)
    vu_max = getattr(vp, "IMAGE_MAX_TOKEN_NUM", None)
    vu_max = vu_max * vu_factor ** 2 if vu_max else getattr(vp, "MAX_PIXELS", 16384 * 28 * 28)

    def resize1(h, w):                     # qwen_vl_utils.fetch_image (mặc định, không truyền min/max)
        return vp.smart_resize(h, w, factor=vu_factor, min_pixels=vu_min, max_pixels=vu_max)

    def resize2(h, w, max_pixels=CANON_MAX_PIXELS):    # image processor của Qwen
        return hf_smart_resize(h, w, factor=factor, min_pixels=CANON_MIN_PIXELS, max_pixels=max_pixels)

    tc = getattr(cfg, "text_config", None) or cfg
    n_layers, n_heads = int(tc.num_hidden_layers), int(tc.num_attention_heads)
    n_kv = int(getattr(tc, "num_key_value_heads", n_heads))
    head_dim = int(getattr(tc, "head_dim", None) or tc.hidden_size // n_heads)
    ctx = int(getattr(tc, "max_position_embeddings", None) or cfg.max_position_embeddings)
    image_pad = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return {"processor": processor, "resize1": resize1, "resize2": resize2, "patch": patch, "merge": merge,
            "image_pad_id": image_pad, "vu_defaults": {"factor": vu_factor, "min_pixels": vu_min, "max_pixels": vu_max},
            "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "context_window": ctx,
                      "num_hidden_layers": n_layers, "num_attention_heads": n_heads, "num_key_value_heads": n_kv,
                      "head_dim": head_dim, "kv_cache_bytes_per_token": kv_cache_bytes_per_token(n_layers, n_kv, head_dim)}}


def image_size(path: str) -> Tuple[int, int]:
    """(H, W) như generate_answer thấy: Image.open(...).convert('RGB') KHÔNG xoay EXIF -> kích thước thô."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return h, w


_TINY = None


def chat_text(processor, prompt_text: str) -> str:
    """Chuỗi chat template (tokenize=False) — giống generate_answer; ảnh chỉ là chỗ giữ, chưa nở token."""
    global _TINY
    from PIL import Image
    from src.prompting.builder import to_chat_messages
    if _TINY is None:
        _TINY = Image.new("RGB", (28, 28))
    return processor.apply_chat_template(to_chat_messages(_TINY, prompt_text), tokenize=False,
                                         add_generation_prompt=True)


def full_path_count(qb: Dict[str, Any], image_path: str, prompt_text: str) -> Tuple[int, int]:
    """Đường đầy đủ như run_eval.generate_answer: trả (len(input_ids), số <|image_pad|>)."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from src.prompting.builder import to_chat_messages
    processor = qb["processor"]
    image = Image.open(image_path).convert("RGB")
    messages = to_chat_messages(image, prompt_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    enc = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    return len(ids), sum(1 for t in ids if t == qb["image_pad_id"])


def measure_split(name: str, samples: List[Dict[str, Any]], qb: Dict[str, Any], mode: str, is_train: bool,
                  size_cache: Dict[str, Tuple[int, int]]) -> List[Dict[str, Any]]:
    """Đo từng câu bằng đường nhanh (kích thước ảnh + tokenizer)."""
    from src.adaptation.train_qlora import build_target_text, pick_answer
    tok = qb["processor"].tokenizer
    rows, t0 = [], time.time()
    q_cache: Dict[str, int] = {}
    for i, s in enumerate(samples):
        p = s["image_path"]
        if p not in size_cache:
            size_cache[p] = image_size(p)
        h, w = size_cache[p]
        _, _, n_img = visual_tokens(h, w, qb["resize1"], qb["resize2"], qb["patch"], qb["merge"])
        _, _, n_alt = visual_tokens(h, w, qb["resize1"], lambda a, b: qb["resize2"](a, b, ALT_MAX_PIXELS),
                                    qb["patch"], qb["merge"])
        h1, w1 = qb["resize1"](h, w)
        n_native = (h1 // qb["patch"]) * (w1 // qb["patch"]) // (qb["merge"] ** 2)
        prompt = assemble_prompt_parts(s, mode)
        n_chat = len(tok(chat_text(qb["processor"], prompt), add_special_tokens=False)["input_ids"])
        q = s["question"]
        if q not in q_cache:
            q_cache[q] = len(tok(q, add_special_tokens=False)["input_ids"])
        ocr_tok = 0
        if mode == "ocr" and s.get("ocr_texts"):
            ocr_line = prompt.split("\n")[0]                     # đúng dòng "Văn bản đọc được trong ảnh: …"
            ocr_tok = len(tok(ocr_line + "\n", add_special_tokens=False)["input_ids"])
        row = {"question_id": s["question_id"], "image": os.path.basename(p), "h": h, "w": w,
               "img_tok": n_img, "native_tok": n_native, "alt_tok": n_alt, "q_tok": q_cache[q],
               "ocr_tok": ocr_tok, "text_tok": n_chat - 1, "input_tok": n_chat - 1 + n_img, "ans_tok": 0}
        if is_train:
            row["ans_tok"] = len(tok(build_target_text(pick_answer(s)), add_special_tokens=False)["input_ids"])
        rows.append(row)
        if (i + 1) % 5000 == 0:
            print(f"  [{name}|{mode}] {i + 1}/{len(samples)} câu ({time.time() - t0:.0f}s)", flush=True)
    return rows


def validate(name: str, samples: List[Dict[str, Any]], rows: List[Dict[str, Any]], qb: Dict[str, Any],
             mode: str, n: int = VALIDATE_N) -> Dict[str, Any]:
    """So đường nhanh với đường processor đầy đủ trên n câu ngẫu nhiên (seed 42)."""
    idx = sorted(random.Random(SEED).sample(range(len(samples)), min(n, len(samples))))
    bad = []
    for i in idx:
        total, n_pad = full_path_count(qb, samples[i]["image_path"], assemble_prompt_parts(samples[i], mode))
        r = rows[i]
        if total != r["input_tok"] or n_pad != r["img_tok"]:
            bad.append({"question_id": r["question_id"], "fast": [r["input_tok"], r["img_tok"]], "full": [total, n_pad]})
    return {"split": name, "mode": mode, "n_checked": len(idx), "n_mismatch": len(bad), "examples": bad[:5]}


def fix_image_tokens(samples: List[Dict[str, Any]], rows: List[Dict[str, Any]], qb: Dict[str, Any]) -> None:
    """Dự phòng mức 1: token ảnh lấy từ processor thật, MỘT lần mỗi ảnh (giữ phần chữ của đường nhanh)."""
    per_img: Dict[str, int] = {}
    for s, r in zip(samples, rows):
        p = s["image_path"]
        if p not in per_img:
            per_img[p] = full_path_count(qb, p, "Câu hỏi: ?")[1]
        r["img_tok"] = per_img[p]
        r["input_tok"] = r["text_tok"] + r["img_tok"]


def full_measure(samples: List[Dict[str, Any]], rows: List[Dict[str, Any]], qb: Dict[str, Any], mode: str) -> None:
    """Dự phòng mức 2 (chậm): đo MỌI câu bằng đường đầy đủ (ghi đè input_tok/img_tok/text_tok)."""
    for s, r in zip(samples, rows):
        total, n_pad = full_path_count(qb, s["image_path"], assemble_prompt_parts(s, mode))
        r["input_tok"], r["img_tok"], r["text_tok"] = total, n_pad, total - n_pad


def load_split(drive_root: str, key: str, image_dirs: Dict[str, str]) -> List[Dict[str, Any]]:
    from src.data.ocr import attach_ocr, default_cache_path, load_ocr_cache
    from src.data.vivqa_dataset import load_vivqa
    from src.adaptation.train_qlora import filter_trainable
    spec = SPLITS[key]
    samples = load_vivqa(os.path.join(drive_root, spec["data_rel"]), image_dir=image_dirs[spec["dataset"]])
    if spec["train"]:
        samples, dropped = filter_trainable(samples)     # đúng như lúc train
        print(f"  {key}: bỏ {len(dropped)} câu không train được (như train_qlora)")
    missing = [s["image_path"] for s in samples if not os.path.exists(s["image_path"])]
    if missing:
        raise FileNotFoundError(f"{key}: thiếu {len(missing)} ảnh, vd {missing[0]}")
    cache = load_ocr_cache(default_cache_path(drive_root, spec["dataset"], OCR_BACKEND))
    attach_ocr(samples, cache, max_chars=OCR_MAX_CHARS, min_conf=0.0)
    return samples


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    cols = ["question_id", "image", "h", "w", "img_tok", "native_tok", "alt_tok", "q_tok", "ocr_tok",
            "text_tok", "input_tok", "ans_tok"]
    with open(path, "w", encoding="utf-8", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=cols, lineterminator="\n")
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r[c] for c in cols})


def run(drive_root: str, out: str, splits: Optional[List[str]] = None) -> Dict[str, Any]:
    from src.eval.run_baseline import prepare_images
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    qb = load_qwen_bits()
    print("model:", json.dumps(qb["model"]), "| qwen_vl_utils mặc định:", qb["vu_defaults"], flush=True)
    image_dirs = {ds: prepare_images(ds, drive_root) for ds in ("vivqa", "vitextvqa_official")}
    results, validation, size_cache = {}, [], {}
    for key in (splits or list(SPLITS)):
        samples = load_split(drive_root, key, image_dirs)
        print(f"== {key}: {len(samples)} câu", flush=True)
        for mode in MODES:
            rows = measure_split(key, samples, qb, mode, SPLITS[key]["train"], size_cache)
            v = validate(key, samples, rows, qb, mode)
            v["method"] = "fast"
            if v["n_mismatch"]:
                print(f"  ⚠ {key}|{mode}: đường nhanh lệch {v['n_mismatch']}/{v['n_checked']} câu "
                      f"(vd {v['examples'][:2]}) -> lấy token ảnh từ processor, mỗi ảnh 1 lần", flush=True)
                fix_image_tokens(samples, rows, qb)
                v2 = validate(key, samples, rows, qb, mode)
                v["method"], v["after_fix"] = "image_tokens_from_processor", v2
                if v2["n_mismatch"]:
                    print(f"  ⚠ vẫn lệch {v2['n_mismatch']} câu -> đo đầy đủ mọi câu (chậm)", flush=True)
                    full_measure(samples, rows, qb, mode)
                    v["method"] = "full_processor_fallback"
            validation.append(v)
            results[f"{key}|{mode}"] = summarize(rows, qb["model"]["context_window"],
                                                 qb["model"]["kv_cache_bytes_per_token"], is_train=SPLITS[key]["train"])
            write_csv(os.path.join(out, f"per_sample_{key}_{mode}.csv"), rows)
            s = results[f"{key}|{mode}"]
            print(f"  {key}|{mode}: input p50/p95/max = {s['input_tokens']['p50']:.0f}/{s['input_tokens']['p95']:.0f}/"
                  f"{s['input_tokens']['max']:.0f} | ảnh p50 {s['image_tokens']['p50']:.0f} | "
                  f"chặn 512: {s['pct_images_capped_512']}% | kiểm chứng lệch {v['n_mismatch']}/{v['n_checked']}", flush=True)
    n_chk = sum(v["n_checked"] for v in validation)
    n_bad = sum(v["n_mismatch"] for v in validation if v["method"] == "fast")
    try:
        import transformers, qwen_vl_utils  # noqa: F401
        versions = {"transformers": transformers.__version__,
                    "qwen_vl_utils": getattr(qwen_vl_utils, "__version__", "unknown")}
    except Exception:
        versions = {}
    summary = {"version": TOKEN_BUDGET_VERSION, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "model": qb["model"], "qwen_vl_utils_defaults": qb["vu_defaults"],
               "config": {"min_pixels": CANON_MIN_PIXELS, "max_pixels": CANON_MAX_PIXELS, "alt_max_pixels": ALT_MAX_PIXELS,
                          "max_new_tokens": MAX_NEW_TOKENS, "ocr_backend": OCR_BACKEND, "ocr_template": "v1",
                          "ocr_max_chars": OCR_MAX_CHARS, "seed": SEED},
               "validation": {"n_checked": n_chk, "n_mismatch": n_bad, "ok": True, "per_split": validation,
                              "note": "n_mismatch tính trên đường nhanh; tập lệch đã đo lại bằng processor (xem method)"},
               "results": results, "elapsed_sec": round(time.time() - t_start, 1),
               "platform": platform.platform(), "versions": versions}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as fp:
        fp.write(render_md(summary))
    with open(os.path.join(out, "validation.json"), "w", encoding="utf-8") as fp:
        json.dump(validation, fp, ensure_ascii=False, indent=2)
    print(f"✅ xong trong {summary['elapsed_sec']:.0f}s -> {out}", flush=True)
    return summary


# --------------------------------------------------------------------------- #
# 3) Selftest offline
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2 and percentile([5], 95) == 5 and percentile(list(range(1, 101)), 95) == 95
    d = describe([10, 20, 30])
    assert d["min"] == 10 and d["max"] == 30 and d["mean"] == 20 and d["p50"] == 20

    # smart_resize giả kiểu Qwen: làm tròn bội 28 + kẹp số điểm ảnh
    def fake_resize(h, w, mn=CANON_MIN_PIXELS, mx=CANON_MAX_PIXELS, f=28):
        hb, wb = max(f, round(h / f) * f), max(f, round(w / f) * f)
        if hb * wb > mx:
            b = math.sqrt(h * w / mx)
            hb, wb = math.floor(h / b / f) * f, math.floor(w / b / f) * f
        elif hb * wb < mn:
            b = math.sqrt(mn / (h * w))
            hb, wb = math.ceil(h * b / f) * f, math.ceil(w * b / f) * f
        return hb, wb

    ident = lambda h, w: (h, w)  # noqa: E731
    h, w, n = visual_tokens(364, 644, ident, fake_resize)
    assert (h, w, n) == (364, 644, 299), (h, w, n)                 # mẫu thật của slide 18 (640×360 -> 299 token)
    assert visual_tokens(4000, 3000, ident, fake_resize)[2] <= 512  # ảnh lớn bị chặn 512
    assert visual_tokens(100, 100, ident, fake_resize)[2] >= 256    # ảnh nhỏ nâng lên ≥ 256
    assert kv_cache_bytes_per_token(36, 2, 128) == 36864            # Qwen2.5-VL-3B: 36 lớp, 2 KV-head, head_dim 128

    s = {"question_id": "t1", "question": "biển ghi gì?", "ocr_texts": ["CẤM ĐỖ XE"], "answers": ["cấm đỗ xe"]}
    assert assemble_prompt_parts(s, "zero_shot") == "Câu hỏi: biển ghi gì?"
    assert assemble_prompt_parts(s, "ocr") == "Văn bản đọc được trong ảnh: CẤM ĐỖ XE\nCâu hỏi: biển ghi gì?"
    assert assemble_prompt_parts({**s, "ocr_texts": []}, "ocr") == "Câu hỏi: biển ghi gì?"

    rows = [{"question_id": f"q{i}", "image": f"i{i % 2}", "h": 1, "w": 1, "img_tok": t, "native_tok": nt,
             "alt_tok": nt, "q_tok": 8, "ocr_tok": o, "text_tok": 40 + o, "input_tok": 40 + o + t, "ans_tok": 4}
            for i, (t, nt, o) in enumerate([(512, 900, 30), (300, 300, 0), (256, 100, 12)])]
    sm = summarize(rows, 128000, 36864, is_train=True)
    assert sm["n_questions"] == 3 and sm["n_images"] == 2
    assert sm["pct_images_capped_512"] == 33.33 and sm["pct_images_raised_to_256"] == 33.33
    assert sm["longest_sequence"] == 40 + 30 + 512 + 32 and sm["pct_of_context_longest"] == round(100 * 614 / 128000, 3)
    assert sm["pct_questions_with_ocr_line"] == 66.67 and sm["sequence_train"]["max"] == 40 + 30 + 512 + 4
    md = render_md({"model": {"model_id": MODEL_ID, "context_window": 128000, "kv_cache_bytes_per_token": 36864},
                    "results": {"x_test|zero_shot": sm, "x_test|ocr": sm},
                    "validation": {"n_checked": 3, "n_mismatch": 0, "ok": True}})
    assert "| x_test | ocr | 3 |" in md and "ĐẠT" in md
    print("token_budget SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--drive_root", default="/content/drive/MyDrive/ViVQA-VLM")
    ap.add_argument("--out", default=None, help="mặc định <drive_root>/experiments/W15_token_budget")
    ap.add_argument("--splits", nargs="*", choices=list(SPLITS), default=None)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    if a.run:
        run(a.drive_root, a.out or os.path.join(a.drive_root, "experiments", OUT_NAME), a.splits)
    if not (a.selftest or a.run):
        ap.print_help()


if __name__ == "__main__":
    main()
