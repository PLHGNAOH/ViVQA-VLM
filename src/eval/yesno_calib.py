"""
yesno_calib.py — W13 bước 1–2: chẩn đoán + sửa thiên lệch "có" của adapter dùng chung SH bằng NGƯỠNG lúc suy luận.

Vấn đề (W12, POPE-vi): SH (adapter của prototype) trả lời "có" ở 77–89% câu, ΔF1 so với B1-ocr −5,7 đến −9,4
-> hỏi "Trong ảnh có X không?" thì gần như luôn "có". Train lại tốn ≈ 7 h L4; trước đó cần biết: thiên lệch là
"lệch ngưỡng" (điểm "có" vẫn xếp đúng thứ tự, chỉ bị đẩy lên -> dời ngưỡng là sửa được) hay "mất khả năng phân biệt"
(điểm "có" không còn tách câu có-thật / không-có -> phải train lại).

Cách đo v3 (`first-token-max-reppen-v3`; v1 KHÔNG HỢP LỆ, v2 dừng ở smoke test — xem docs/W13/README.md):
    MỘT forward 1 hàng (= bước prefill của model.generate), áp ĐÚNG repetition penalty mà generate() áp
    (đọc từ model.generation_config, Qwen2.5-VL-3B-Instruct: 1.05), log-softmax -> log p của TOKEN ĐẦU:
        s_yes = max log p(t) | t ∈ token đầu của {có, Có, yes, Yes} = {có, C, yes, Yes}
        s_no  = max log p(t) | t ∈ token đầu của {không, Không, no, No} = {kh, Không, no, No}
    margin m = s_yes − s_no;  quyết định "có" ⇔ m > τ.  τ = 0 ⇔ đúng phép so của greedy ở bước đầu
    (lấy MAX như greedy, không cộng dồn; kiểm hợp lệ: trùng greedy W12 >= 98%). Ghi kèm margin_raw (không penalty).
Luật ghi trước: docs/W13/README.md (R-YB). τ chọn CHỈ trên tập calibration (ảnh khác POPE-vi), áp lên POPE-vi.

Năm việc:
  --build     Sinh tập calibration + POPE-vi mẫu câu 2 (giữ nguyên từ v1; file đã sinh thì KHÔNG sinh lại).
  --score     (GPU) chấm một cấu hình × một file câu hỏi; theo lô, resume, chữ ký run.
  --agree     Kiểm nhanh: quyết định τ = 0 của một run trùng greedy W12 bao nhiêu % (dùng cho smoke test).
  --analyze   Kiểm hợp lệ, AUC, chọn τ trên calibration, áp lên POPE-vi, phán quyết R-YB -> yesno_calib_summary.json + .md
  --selftest  Kiểm thử offline (không GPU, không mạng; phần toán dùng torch CPU nếu có).

Chạy:
    python -m src.eval.yesno_calib --score --data_file <file> --run_name <run> [--adapter <dir>] --mode ocr \
        --ocr_backend paddledet_vietocr --ocr_template v1 --ocr_max_chars 600 --ocr_min_conf 0 [--limit 300]
    python -m src.eval.yesno_calib --agree --run_dir <exp>/<run> --greedy_dir <exp>/<run greedy W12>
    python -m src.eval.yesno_calib --analyze --questions_dir <exp>/W13_yesno_questions \
        --pope experiments/W12_pope_vi_questions/pope_vi_questions_v1.json --exp_root <exp> --out <exp>/W13_yesno_calib_v3_L4_seed42
    python -m src.eval.yesno_calib --selftest
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.eval.pope_vi import (OBJECTS_FILE, QUESTION_TEMPLATE, SETTINGS, MARGIN_PTS, N_BOOT, SEED,  # noqa: E402
                              blocked_classes, bootstrap_delta_f1, coco_split_of, load_coco_classes,
                              load_objects, metrics_of, parse_yes_no, per_image_counts)

# --------------------------------------------------------------------------- #
# Hằng số (ghi trước — đổi bất kỳ giá trị nào = luật khác, phải ghi lại README W13)
# --------------------------------------------------------------------------- #
BUILD_VERSION = "yesno-calib-build-v1"            # bộ câu hỏi giữ nguyên từ v1
SCORING_VERSION = "first-token-max-reppen-v3"
CALIB_MAX_IMAGES = 600
CALIB_MIN_CLASSES = 1
N_POS = 3
N_NEG = 3
TEMPLATE2 = "Có {obj} trong ảnh không?"
# từ v2: bỏ dạng VIẾT HOA TOÀN BỘ (CÓ, KHÔNG) — không run greedy W12 nào sinh ra
YES_WORDS = ("có", "Có", "yes", "Yes")
NO_WORDS = ("không", "Không", "no", "No")
TAU_MIN, TAU_MAX, TAU_STEP = -15.0, 15.0, 0.05
AGREE_MIN = 0.98              # kiểm hợp lệ: quyết định ở τ = 0 trùng greedy W12 >= 98% (SH, B1-ocr) — giữ nguyên v1
MAX_YES_RATIO = 0.60          # R-YB: yes-ratio sau khi sửa <= 60% ở cả 3 cách chọn — giữ nguyên v1
CONFIGS = ("SH", "B1_ocr", "B3full")
QSETS = ("calib", "popevi", "popevi_t2")
QFILES = {"calib": "yesno_calib_v1.json", "popevi_t2": "pope_vi_questions_t2_v1.json"}   # popevi = file W12
RUN_TAG = {"SH": "sh_qlora_ocr_vietocr_v1_r16_shared", "B1_ocr": "b1_ocr_vietocr_v1_qwen25vl",
           "B3full": "b3full_qlora_ocr_vietocr_v1_r16"}
# Thứ tự chạy: 3 run BẮT BUỘC cho luật trước (dừng sau đó vẫn phán quyết được), rồi các run chỉ mô tả
REQUIRED_RUNS = (("SH", "calib"), ("SH", "popevi"), ("B1_ocr", "popevi"))
OPTIONAL_RUNS = (("SH", "popevi_t2"), ("B1_ocr", "calib"), ("B1_ocr", "popevi_t2"),
                 ("B3full", "calib"), ("B3full", "popevi"), ("B3full", "popevi_t2"))
RUN_ORDER = REQUIRED_RUNS + OPTIONAL_RUNS
# Run greedy W12 trên POPE-vi (cùng file câu hỏi) -> mốc so sánh + kiểm hợp lệ
GREEDY_W12 = {
    "B1_ocr": "W12_POPE_b1_ocr_vietocr_v1_qwen25vl_popevi_v1_bf16_L4_seed42",
    "B3full": "W12_POPE_b3full_qlora_ocr_vietocr_v1_r16_popevi_v1_bf16_L4_seed42",
    "SH": "W12_POPE_sh_qlora_ocr_vietocr_v1_r16_shared_popevi_v1_bf16_L4_seed42",
}
VERDICTS = ("ĐẠT", "KHÔNG ĐẠT", "KHÔNG HỢP LỆ")


def logit_run_name(label: str, qset: str) -> str:
    # v3 dùng tiền tố YN3 -> không đụng thư mục run v1 (W13_YN_*) / smoke v2 (W13_YN2_*), giữ làm hồ sơ
    return f"W13_YN3_{RUN_TAG[label]}_{qset}_bf16_L4_seed42"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for b in iter(lambda: fp.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _write_json(path: str, obj: Any, indent: Optional[int] = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=indent)
        fp.write("\n")
    os.replace(tmp, path)


# =============================================================================
# 1) Sinh câu hỏi: calibration + POPE-vi mẫu câu 2
# =============================================================================
def build_calib(vivqa_dir: str, coco_ann_dir: str, pope_file: str, objects_path: str = OBJECTS_FILE,
                max_images: int = CALIB_MAX_IMAGES, seed: int = SEED) -> Dict[str, Any]:
    """Tập calibration: cùng khuôn POPE-vi, nhưng trên ảnh ViVQA test chưa thấy lúc train và KHÔNG thuộc POPE-vi.
    Ảnh có ít hơn 3 lớp vẫn dùng: số câu "có" = số câu "không" mỗi cách chọn = min(3, số lớp)."""
    objs = load_objects(objects_path)
    vi, group_of = objs["objects"], objs["group_of"]
    used = [s for s in ("train", "val", "test") if os.path.exists(os.path.join(vivqa_dir, f"{s}.json"))]
    if "train" not in used or "test" not in used:
        raise FileNotFoundError(f"{vivqa_dir} thiếu train.json / test.json")
    recs = {s: _load_json(os.path.join(vivqa_dir, f"{s}.json")) for s in used}
    seen_ids = {r["img_id"] for s in used if s != "test" for r in recs[s]}
    test_imgs: Dict[int, str] = {}
    for r in recs["test"]:
        test_imgs.setdefault(int(r["img_id"]), r["image"])
    pope_imgs = {int(q["img_id"]) for q in _load_json(pope_file)["data"]}
    unseen = {i: f for i, f in test_imgs.items() if i not in seen_ids}
    cand = {i: f for i, f in unseen.items() if i not in pope_imgs}
    needed: Dict[str, set] = defaultdict(set)
    for i, f in cand.items():
        needed[coco_split_of(f)].add(i)
    classes = load_coco_classes(coco_ann_dir, needed)
    unknown = {c for v in classes.values() for c in v} - set(vi)
    if unknown:
        raise ValueError(f"lớp COCO không có trong từ điển: {sorted(unknown)}")
    pool = {i: sorted(classes[(coco_split_of(f), i)]) for i, f in cand.items()
            if len(classes[(coco_split_of(f), i)]) >= CALIB_MIN_CLASSES}
    if not pool:
        raise ValueError("pool calibration rỗng")

    freq = Counter(c for cs in pool.values() for c in cs)
    cooc: Dict[str, Counter] = defaultdict(Counter)
    for cs in pool.values():
        for a in cs:
            for b in cs:
                if a != b:
                    cooc[a][b] += 1
    popular_rank = sorted(vi, key=lambda c: (-freq[c], c))
    chosen = sorted(random.Random(seed).sample(sorted(pool), min(max_images, len(pool))))

    rows: Dict[Tuple[int, str], Dict[str, Any]] = {}

    def add(img_id: int, cls: str, label: str, setting: str) -> None:
        key = (img_id, cls)
        if key not in rows:
            rows[key] = {"question_id": f"calib_{img_id}_{cls.replace(' ', '_')}",
                         "question": QUESTION_TEMPLATE.format(obj=vi[cls]),
                         "answers": ["có" if label == "yes" else "không"],
                         "image": cand[img_id], "img_id": img_id,
                         "question_type": "pope_yes" if label == "yes" else "pope_no",
                         "coco_class": cls, "label": label, "settings": []}
        if rows[key]["label"] != label:
            raise AssertionError(f"{key}: một cặp vừa là câu có vừa là câu không")
        rows[key]["settings"].append(setting)

    for img_id in chosen:
        present = pool[img_id]
        k = min(N_POS, len(present))
        rng = random.Random(seed * 1_000_003 + img_id)
        pos = sorted(rng.sample(present, k))
        allowed = [c for c in sorted(vi) if c not in blocked_classes(present, group_of)]
        adv = {c: sum(cooc[p][c] for p in present) for c in allowed}
        kn = min(N_NEG, k)
        negs = {"random": sorted(rng.sample(allowed, kn)),
                "popular": [c for c in popular_rank if c in set(allowed)][:kn],
                "adversarial": sorted(allowed, key=lambda c: (-adv[c], -freq[c], c))[:kn]}
        for c in pos:
            for s in SETTINGS:
                add(img_id, c, "yes", s)
        for s in SETTINGS:
            for c in negs[s]:
                add(img_id, c, "no", s)

    questions = sorted(rows.values(), key=lambda r: (r["img_id"], r["label"] != "yes", r["coco_class"]))
    stats = {
        "build_version": BUILD_VERSION, "objects_version": objs["version"], "objects_sha256": objs["sha256"],
        "seed": seed, "template": QUESTION_TEMPLATE, "purpose": "calibration (chọn τ) — KHÔNG dùng để đánh giá",
        "test_images": len(test_imgs), "test_images_unseen": len(unseen), "excluded_pope_images": len(pope_imgs),
        "candidate_images": len(cand), "pool_images": len(pool), "chosen_images": len(chosen),
        "unique_questions": len(questions),
        "yes_questions": sum(r["label"] == "yes" for r in questions),
        "no_questions": sum(r["label"] == "no" for r in questions),
        "items_per_setting": {s: sum(s in r["settings"] for r in questions) for s in SETTINGS},
        "pope_file_sha256": _sha256(pope_file),
        "vivqa_sha256": {s: _sha256(os.path.join(vivqa_dir, f"{s}.json")) for s in used},
        "coco_ann_sha256": {s: _sha256(os.path.join(coco_ann_dir, f"instances_{s}.json")) for s in sorted(needed)},
    }
    return {"data": questions, "stats": stats}


def build_template2(pope_file: str, objects_path: str = OBJECTS_FILE) -> Dict[str, Any]:
    """POPE-vi y hệt (cùng ảnh, cùng lớp, cùng question_id), chỉ đổi câu hỏi sang mẫu 2 (không chỉ dẫn trả lời)."""
    vi = load_objects(objects_path)["objects"]
    src = _load_json(pope_file)
    data = [{**q, "question": TEMPLATE2.format(obj=vi[q["coco_class"]]), "template": "t2"} for q in src["data"]]
    return {"data": data, "stats": {"build_version": BUILD_VERSION, "template": TEMPLATE2,
                                    "source": os.path.basename(pope_file), "source_sha256": _sha256(pope_file),
                                    "unique_questions": len(data), "purpose": "chỉ mô tả (độ bền với cách hỏi)"}}


# =============================================================================
# 2) Chấm v3: MỘT forward 1 hàng (= prefill của generate) + repetition penalty như generate() + so token đầu
# =============================================================================
def candidate_tokens(tokenizer) -> Dict[str, Any]:
    """Token ĐẦU của các từ ứng viên — đúng thứ greedy so ở bước sinh đầu tiên.
    "không" = kh + ông, "Có" = C + ó: dùng kh / C (smoke v2: P(ông | kh) trung vị 0,984, thấp nhất 0,893).
    Hai phía không được chung token đầu (nếu chung thì không tách được quyết định ở bước đầu)."""
    cands: Dict[str, Dict[str, List[int]]] = {"yes": {}, "no": {}}
    for side, words in (("yes", YES_WORDS), ("no", NO_WORDS)):
        for w in words:
            ids = [int(t) for t in tokenizer.encode(w, add_special_tokens=False)]
            if not ids:
                raise ValueError(f"từ '{w}' mã hoá rỗng")
            cands[side][w] = ids
    first = {s: {w: ids[0] for w, ids in cands[s].items()} for s in cands}
    if set(first["yes"].values()) & set(first["no"].values()):
        raise ValueError(f"hai phía chung token đầu: {first}")
    return {"cands": cands, "first": first,
            "first_ids": sorted(set(first["yes"].values()) | set(first["no"].values()))}


def apply_repetition_penalty(logits, seen_ids: Sequence[int], penalty: float):
    """Đúng RepetitionPenaltyLogitsProcessor của transformers: token đã có trong input -> logit > 0 chia penalty,
    logit < 0 nhân penalty. Trả tensor mới (không sửa tại chỗ)."""
    import torch
    out = logits.clone()
    if penalty is None or penalty == 1.0 or not len(seen_ids):
        return out
    idx = torch.as_tensor(sorted(set(int(i) for i in seen_ids)), dtype=torch.long, device=out.device)
    sc = out.index_select(-1, idx)
    out[..., idx] = torch.where(sc < 0, sc * penalty, sc / penalty)
    return out


def score_from_logits(first_logits, prompt_ids: Sequence[int], tokens: Dict[str, Any],
                      penalty: float) -> Dict[str, Any]:
    """Phần toán thuần (kiểm offline được). first_logits: [V] logit bước sinh đầu (sau prompt).
    margin = max log p(token đầu phía có) − max log p(token đầu phía không), sau penalty như generate()
    -> margin > 0 ⇔ trong các token ứng viên, greedy xếp một token phía "có" cao nhất."""
    import torch
    res: Dict[str, Any] = {}
    for tag, pen in (("", penalty), ("_raw", 1.0)):
        lp1 = torch.log_softmax(apply_repetition_penalty(first_logits.float(), list(prompt_ids), pen), dim=-1)
        lp = {w: round(float(lp1[t]), 5) for s in ("yes", "no") for w, t in tokens["first"][s].items()}
        sy = max(float(lp1[t]) for t in tokens["first"]["yes"].values())
        sn = max(float(lp1[t]) for t in tokens["first"]["no"].values())
        res[f"lp{tag}"], res[f"s_yes{tag}"], res[f"s_no{tag}"], res[f"margin{tag}"] = lp, sy, sn, sy - sn
        if tag == "":
            top = int(torch.argmax(lp1))
            res["top1_id"], res["top1_logp"] = top, float(lp1[top])
    return res


def forward_first(model, inputs) -> Any:
    """Logit bước sinh đầu: đúng cách gọi của v1 (đã chạy trơn 9 run trên L4) — 1 hàng, mọi khoá processor."""
    try:
        lg = model(**inputs, logits_to_keep=1).logits
    except TypeError:                                         # bản transformers không có logits_to_keep
        lg = model(**inputs).logits
    return lg[0, -1]


def score_sample(model, processor, image_path: str, prompt_text: str, tokens: Dict[str, Any],
                 penalty: float) -> Dict[str, Any]:
    """Cùng input như generate_answer (src/eval/run_eval.py)."""
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from src.prompting.builder import to_chat_messages
    image = Image.open(image_path).convert("RGB")
    messages = to_chat_messages(image, prompt_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        first = forward_first(model, inputs)
    return score_from_logits(first, inputs["input_ids"][0].tolist(), tokens, penalty)


def generation_penalty(model) -> Tuple[float, Dict[str, Any]]:
    """repetition_penalty mà model.generate(do_sample=False) sẽ áp + các khoá generation_config liên quan (ghi meta)."""
    gc = getattr(model, "generation_config", None)
    keys = ("repetition_penalty", "do_sample", "temperature", "top_k", "top_p", "no_repeat_ngram_size",
            "encoder_repetition_penalty", "bad_words_ids", "sequence_bias")
    info = {k: getattr(gc, k, None) for k in keys} if gc is not None else {}
    for k in ("no_repeat_ngram_size", "encoder_repetition_penalty", "bad_words_ids", "sequence_bias"):
        if info.get(k) not in (None, 0, 1.0):
            raise ValueError(f"generation_config.{k}={info[k]} -> generate() có thêm bộ xử lý logit chưa mô phỏng")
    p = info.get("repetition_penalty") or 1.0
    return float(p), info


def logit_signature(cfg: Dict[str, Any], data_sha256: str, limit: Optional[int], chunk: int,
                    adapter: Optional[Dict[str, Any]], tokens: Dict[str, Any], penalty: float) -> Dict[str, Any]:
    from src.eval.run_baseline import run_signature
    base = run_signature(cfg, data_sha256, limit, chunk, adapter=adapter)
    base.pop("sha256")
    payload = {**base, "scoring": {"version": SCORING_VERSION, "cands": tokens["cands"], "repetition_penalty": penalty}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"sha256": digest, **payload}


def score_chunks(samples: List[Dict[str, Any]], prompting: Dict[str, Any], out_dir: str, chunk: int,
                 model, processor, tokens: Dict[str, Any], penalty: float, score_fn: Callable = score_sample,
                 gpu_name: str = "unknown") -> int:
    """Chấm theo lô, ghi lô ngay khi xong; lô đã có thì kiểm id rồi bỏ qua."""
    from src.prompting.builder import prompt_for_sample
    cdir = os.path.join(out_dir, "chunks")
    os.makedirs(cdir, exist_ok=True)
    n = (len(samples) + chunk - 1) // chunk
    for k in range(n):
        part = samples[k * chunk:(k + 1) * chunk]
        ids = [s["question_id"] for s in part]
        path = os.path.join(cdir, f"chunk_{k:02d}.json")
        if os.path.exists(path):
            if [r["question_id"] for r in _load_json(path)["records"]] != ids:
                raise ValueError(f"{path} không khớp dữ liệu hiện tại -> dùng run_name mới")
            _log(f"[lô {k + 1}/{n}] đã có -> bỏ qua")
            continue
        t0, recs = time.time(), []
        for j, s in enumerate(part):
            r = score_fn(model, processor, s["image_path"], prompt_for_sample(s, prompting), tokens, penalty)
            recs.append({"question_id": s["question_id"], **r})
            if (j + 1) % 100 == 0:
                _log(f"   {j + 1}/{len(part)} | {(time.time() - t0) / (j + 1):.3f} s/câu")
        dt = time.time() - t0
        _write_json(path, {"records": recs, "time_sec": round(dt, 2), "gpu": gpu_name}, indent=None)
        _log(f"[lô {k + 1}/{n}] xong {len(part)} câu | {dt:.0f}s")
    return n


def score_run(data_file: str, run_name: str, drive_root: str, out_root: Optional[str] = None,
              config: str = os.path.join(REPO_ROOT, "configs/qwen_lora.yaml"), mode: Optional[str] = None,
              chunk: int = 500, image_dir: Optional[str] = None, limit: Optional[int] = None,
              adapter: Optional[str] = None, ocr_backend: Optional[str] = None, ocr_cache: Optional[str] = None,
              ocr_template: str = "v1", ocr_max_chars: Optional[int] = 600, ocr_min_conf: float = 0.0,
              model=None, processor=None, score_fn: Callable = score_sample, penalty: Optional[float] = None,
              check_config: bool = True) -> Dict[str, Any]:
    from src.data.ocr import default_cache_path, setup_ocr_prompting
    from src.data.vivqa_dataset import load_vivqa
    from src.eval.run_baseline import adapter_fingerprint, check_or_write_signature, load_config, load_model, prepare_images
    from src.prompting.builder import OCR_MODES
    out_dir = os.path.join(out_root or os.path.join(drive_root, "experiments"), run_name)
    cfg = copy.deepcopy(load_config(config, mode=mode, check=check_config))
    cfg["run"]["name"] = run_name
    use_ocr = cfg["prompting"]["mode"] in OCR_MODES
    if use_ocr and not (ocr_backend or ocr_cache):
        raise ValueError("--mode ocr cần --ocr_backend hoặc --ocr_cache")
    image_dir = image_dir or prepare_images("vivqa", drive_root)
    samples = load_vivqa(data_file, image_dir=image_dir, max_samples=limit)
    missing = [s["image_path"] for s in samples if not os.path.exists(s["image_path"])]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ảnh thiếu, vd {missing[0]}")
    ocr_stats = None
    if use_ocr:
        ocr_path = ocr_cache or default_cache_path(drive_root, "vivqa", ocr_backend)
        ocr_stats = setup_ocr_prompting(cfg, samples, ocr_path, template=ocr_template,
                                        max_chars=ocr_max_chars, min_conf=ocr_min_conf)
    ad_info = {"path": adapter, **adapter_fingerprint(adapter)} if adapter else None
    _log(f"== {run_name} | {len(samples)} câu | mode={cfg['prompting']['mode']} | adapter={'có' if adapter else 'không'}")
    real = model is None
    gen_info: Dict[str, Any] = {}
    if real:
        model, processor = load_model(cfg)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        penalty, gen_info = generation_penalty(model)
    if penalty is None:
        raise ValueError("cần penalty (model thật tự đọc từ generation_config)")
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokens = candidate_tokens(tokenizer)
    _log(f"   ứng viên: {tokens['cands']} | token đầu: {tokens['first']} | repetition_penalty (như generate) = {penalty}")
    data_sha = _sha256(data_file)
    check_or_write_signature(out_dir, logit_signature(cfg, data_sha, limit, chunk, ad_info, tokens, penalty))
    gpu = "cpu"
    if real:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats()
    n = score_chunks(samples, cfg["prompting"], out_dir, chunk, model, processor, tokens, penalty, score_fn, gpu)
    recs, tsec, gpus = [], 0.0, set()
    for k in range(n):
        d = _load_json(os.path.join(out_dir, "chunks", f"chunk_{k:02d}.json"))
        recs += d["records"]
        tsec += d["time_sec"]
        gpus.add(d["gpu"])
    if [r["question_id"] for r in recs] != [s["question_id"] for s in samples]:
        raise ValueError("bản gộp không khớp danh sách câu")
    names: Dict[int, str] = {}
    for r in recs:
        if r["top1_id"] not in names:
            names[r["top1_id"]] = tokenizer.convert_ids_to_tokens(r["top1_id"]) if hasattr(
                tokenizer, "convert_ids_to_tokens") else str(r["top1_id"])
        r["top1_token"] = names[r["top1_id"]]
    meta = {"run_name": run_name, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scoring": SCORING_VERSION, "num_samples": len(recs), "limit": limit,
            "data_file": os.path.basename(data_file), "data_file_sha256": data_sha,
            "prompting_mode": cfg["prompting"]["mode"], "tokens": tokens, "repetition_penalty": penalty,
            "generation_config": gen_info, "adapter": ad_info,
            "ocr": ({**cfg["prompting"]["ocr"], "stats": ocr_stats} if use_ocr else None),
            "model_id": cfg["model"]["model_id"], "torch_dtype": cfg["model"]["torch_dtype"],
            "max_pixels": cfg["model"].get("max_pixels"), "seed": cfg["run"].get("seed"),
            "hardware": {"gpus_seen": sorted(gpus)}, "time_sec": round(tsec, 1),
            "sec_per_sample": round(tsec / max(len(recs), 1), 4)}
    if real:
        import torch, transformers
        meta["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
        if torch.cuda.is_available():
            meta["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    _write_json(os.path.join(out_dir, "logits.json"), recs, indent=None)
    _write_json(os.path.join(out_dir, "meta.json"), meta)
    _log(f"== xong {len(recs)} câu | {meta['sec_per_sample']} s/câu | {tsec / 60:.1f} phút -> {out_dir}")
    return {"meta": meta, "records": recs}


# =============================================================================
# 3) Phân tích + phán quyết R-YB
# =============================================================================
def tau_grid() -> List[float]:
    n = int(round((TAU_MAX - TAU_MIN) / TAU_STEP))
    return [round(TAU_MIN + i * TAU_STEP, 2) for i in range(n + 1)]


def decide(margin: float, tau: float) -> str:
    return "yes" if margin > tau else "no"


def balanced_accuracy(labels: Sequence[str], margins: Sequence[float], tau: float) -> float:
    tp = sum(1 for l, m in zip(labels, margins) if l == "yes" and m > tau)
    tn = sum(1 for l, m in zip(labels, margins) if l == "no" and m <= tau)
    npos = sum(l == "yes" for l in labels)
    nneg = len(labels) - npos
    return 0.5 * (tp / max(npos, 1) + tn / max(nneg, 1))


def select_tau(labels: Sequence[str], margins: Sequence[float]) -> Dict[str, Any]:
    """τ* = argmax balanced accuracy trên calibration (mỗi câu duy nhất tính 1 lần); hoà -> |τ| nhỏ nhất, rồi τ nhỏ hơn."""
    best = None
    for t in tau_grid():
        ba = balanced_accuracy(labels, margins, t)
        key = (round(ba, 12), -abs(t), -t)
        if best is None or key > best[0]:
            best = (key, t, ba)
    return {"tau": best[1], "balanced_accuracy": best[2], "balanced_accuracy_at_0": balanced_accuracy(labels, margins, 0.0)}


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """ROC-AUC = P(margin câu có-thật > margin câu không-có), hoà tính 1/2 (Mann–Whitney)."""
    import numpy as np
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_by_setting(questions: List[Dict[str, Any]], margin: Dict[str, float]) -> Dict[str, float]:
    pos = [margin[q["question_id"]] for q in questions if q["label"] == "yes"]
    return {s: auc(pos, [margin[q["question_id"]] for q in questions if q["label"] == "no" and s in q["settings"]])
            for s in SETTINGS}


def setting_metrics(questions: List[Dict[str, Any]], parsed: Dict[str, Optional[str]],
                    images: Sequence[int]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, List[int]]]]:
    counts = per_image_counts(questions, parsed, images)
    out = {}
    for s in SETTINGS:
        m = metrics_of(counts[s])
        tot = {k: sum(v) for k, v in counts[s].items()}
        m["false_alarm"] = tot["FP"] / max(tot["FP"] + tot["TN"], 1)
        out[s] = m
    return out, counts


def check_logit_run(label: str, qset: str, meta: Dict[str, Any], recs: List[Dict[str, Any]],
                    questions: List[Dict[str, Any]], q_sha: str, greedy_meta: Dict[str, Any],
                    sh_ocr: Dict[str, Any]) -> List[str]:
    errs, tag = [], f"{label}/{qset}"
    if [r["question_id"] for r in recs] != [q["question_id"] for q in questions]:
        errs.append(f"{tag}: logits không khớp danh sách câu ({len(recs)} vs {len(questions)})")
    if meta.get("data_file_sha256") != q_sha:
        errs.append(f"{tag}: chấm trên file câu hỏi khác")
    if meta.get("limit") is not None:
        errs.append(f"{tag}: run có --limit {meta.get('limit')} (smoke), không dùng cho luật")
    if meta.get("prompting_mode") != "ocr":
        errs.append(f"{tag}: mode {meta.get('prompting_mode')} ≠ ocr")
    want = (greedy_meta.get("adapter") or {}).get("weights_sha256")
    got = (meta.get("adapter") or {}).get("weights_sha256")
    if want != got:
        errs.append(f"{tag}: adapter khác run greedy W12 ({str(got)[:12]} vs {str(want)[:12]})")
    keys = ("backend", "template", "max_chars", "min_conf")
    o = meta.get("ocr") or {}
    if any(o.get(k) != sh_ocr.get(k) for k in keys):
        errs.append(f"{tag}: thiết lập OCR khác SH W12")
    if meta.get("scoring") != SCORING_VERSION:
        errs.append(f"{tag}: scoring {meta.get('scoring')} ≠ {SCORING_VERSION}")
    return errs


def agreement(questions: List[Dict[str, Any]], margin: Dict[str, float],
              greedy: Dict[str, Optional[str]]) -> Dict[str, Any]:
    ids = [q["question_id"] for q in questions if q["question_id"] in margin and greedy.get(q["question_id"]) in ("yes", "no")]
    same = sum(decide(margin[i], 0.0) == greedy[i] for i in ids)
    by = Counter((greedy[i], decide(margin[i], 0.0)) for i in ids if decide(margin[i], 0.0) != greedy[i])
    return {"n_parsed": len(ids), "n_total": len(questions), "agree": same / max(len(ids), 1),
            "mismatch_greedy_vs_logit": {f"{a}->{b}": c for (a, b), c in sorted(by.items())}}


def agree_run(run_dir: str, greedy_dir: str) -> Dict[str, Any]:
    """Smoke test: quyết định τ = 0 của run (có thể --limit) so với greedy W12 trên các câu chung."""
    recs = _load_json(os.path.join(run_dir, "logits.json"))
    g = {p["question_id"]: parse_yes_no(p["prediction"]) for p in _load_json(os.path.join(greedy_dir, "predictions.json"))}
    qs = [{"question_id": r["question_id"]} for r in recs]
    return agreement(qs, {r["question_id"]: r["margin"] for r in recs}, g)


def verdict_ryb(valid: bool, ci: Dict[str, Tuple[float, float]], yes_ratio: Dict[str, float]) -> str:
    if not valid:
        return "KHÔNG HỢP LỆ"
    if all(lo > -MARGIN_PTS for lo, _ in ci.values()) and all(v <= MAX_YES_RATIO for v in yes_ratio.values()):
        return "ĐẠT"
    return "KHÔNG ĐẠT"


def analyze(pope_file: str, questions_dir: str, exp_root: str, n_boot: int = N_BOOT) -> Dict[str, Any]:
    qfiles = {"popevi": pope_file, **{k: os.path.join(questions_dir, v) for k, v in QFILES.items()}}
    qdocs = {k: _load_json(p)["data"] for k, p in qfiles.items()}
    qsha = {k: _sha256(p) for k, p in qfiles.items()}
    out: Dict[str, Any] = {"rules": {"agree_min": AGREE_MIN, "margin_pts": MARGIN_PTS, "max_yes_ratio": MAX_YES_RATIO,
                                     "tau_grid": [TAU_MIN, TAU_MAX, TAU_STEP], "n_boot": n_boot, "seed": SEED,
                                     "scoring": SCORING_VERSION, "required_runs": REQUIRED_RUNS},
                           "questions_sha256": qsha, "n_questions": {k: len(v) for k, v in qdocs.items()},
                           "errors": [], "missing_optional": []}
    greedy_meta, greedy_parsed = {}, {}
    for label, run in GREEDY_W12.items():
        d = os.path.join(exp_root, run)
        gm = _load_json(os.path.join(d, "metrics.json"))["meta"]
        gp = _load_json(os.path.join(d, "predictions.json"))
        if gm.get("data_file_sha256") != qsha["popevi"]:
            out["errors"].append(f"greedy W12 {label}: file câu hỏi khác POPE-vi v1")
        greedy_meta[label] = gm
        greedy_parsed[label] = {p["question_id"]: parse_yes_no(p["prediction"]) for p in gp}
    sh_ocr = greedy_meta["SH"].get("ocr") or {}
    margins: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    margins_raw: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    run_info: Dict[str, Any] = {}
    penalties = set()
    for label, qset in RUN_ORDER:
        run = logit_run_name(label, qset)
        d = os.path.join(exp_root, run)
        if not os.path.exists(os.path.join(d, "logits.json")):
            if (label, qset) in REQUIRED_RUNS:
                out["errors"].append(f"thiếu run BẮT BUỘC {label}/{qset} ({run})")
            else:
                out["missing_optional"].append(f"{label}/{qset}")
            continue
        meta = _load_json(os.path.join(d, "meta.json"))
        recs = _load_json(os.path.join(d, "logits.json"))
        out["errors"] += check_logit_run(label, qset, meta, recs, qdocs[qset], qsha[qset], greedy_meta[label], sh_ocr)
        penalties.add(meta.get("repetition_penalty"))
        margins[label][qset] = {r["question_id"]: r["margin"] for r in recs}
        margins_raw[label][qset] = {r["question_id"]: r["margin_raw"] for r in recs}
        first = set((meta.get("tokens") or {}).get("first_ids", []))
        run_info[f"{label}/{qset}"] = {"run": run, "sec_per_sample": meta.get("sec_per_sample"),
                                       "gpus": (meta.get("hardware") or {}).get("gpus_seen"),
                                       "peak_vram_gb": meta.get("peak_vram_gb"),
                                       "repetition_penalty": meta.get("repetition_penalty"),
                                       "top1_is_candidate": sum(r["top1_id"] in first for r in recs) / max(len(recs), 1),
                                       "top1_other_examples": Counter(r.get("top1_token") for r in recs
                                                                      if r["top1_id"] not in first).most_common(5)}
    if len(penalties) > 1:
        out["errors"].append(f"các run dùng repetition_penalty khác nhau: {sorted(map(str, penalties))}")
    out["repetition_penalty"] = sorted(map(str, penalties))
    out["runs"] = run_info
    if out["errors"]:
        out["verdict"] = "KHÔNG HỢP LỆ"
        return out

    pope = qdocs["popevi"]
    images = sorted({q["img_id"] for q in pope})
    have = lambda l, q: q in margins.get(l, {})
    # (1) kiểm hợp lệ: quyết định ở τ = 0 trùng greedy W12
    agree = {l: agreement(pope, margins[l]["popevi"], greedy_parsed[l]) for l in CONFIGS if have(l, "popevi")}
    valid = all(agree[l]["agree"] >= AGREE_MIN for l in ("SH", "B1_ocr"))
    out["agreement_tau0_vs_greedy"] = agree
    out["agreement_raw_tau0_vs_greedy"] = {l: agreement(pope, margins_raw[l]["popevi"], greedy_parsed[l])["agree"]
                                           for l in agree}
    out["valid"] = valid
    # (2) chẩn đoán (chỉ mô tả): AUC
    out["auc"] = {}
    for l in CONFIGS:
        a: Dict[str, Any] = {}
        for qset in ("popevi", "popevi_t2"):
            if have(l, qset):
                a[qset] = auc_by_setting(qdocs[qset], margins[l][qset])
        if have(l, "popevi"):
            a["popevi_raw"] = auc_by_setting(pope, margins_raw[l]["popevi"])
        if have(l, "calib"):
            c = qdocs["calib"]
            a["calib"] = auc([margins[l]["calib"][q["question_id"]] for q in c if q["label"] == "yes"],
                             [margins[l]["calib"][q["question_id"]] for q in c if q["label"] == "no"])
        out["auc"][l] = a
    # (3) τ trên calibration
    lab_c = [q["label"] for q in qdocs["calib"]]
    out["tau"] = {l: select_tau(lab_c, [margins[l]["calib"][q["question_id"]] for q in qdocs["calib"]])
                  for l in CONFIGS if have(l, "calib")}
    tau_sh = out["tau"]["SH"]["tau"]
    # (4) mốc: B1-ocr greedy W12 (đúng mốc của luật H4)
    base_m, base_c = setting_metrics(pope, greedy_parsed["B1_ocr"], images)

    def apply(label: str, qset: str, tau: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        qs = qdocs[qset]
        parsed = {q["question_id"]: decide(margins[label][qset][q["question_id"]], tau) for q in qs}
        return setting_metrics(qs, parsed, sorted({q["img_id"] for q in qs}))

    fix_m, fix_c = apply("SH", "popevi", tau_sh)
    ci = {s: bootstrap_delta_f1(fix_c[s], base_c[s], n_boot) for s in SETTINGS}
    delta = {s: 100 * (fix_m[s]["f1"] - base_m[s]["f1"]) for s in SETTINGS}
    yr = {s: fix_m[s]["yes_ratio"] for s in SETTINGS}
    out["rule"] = {"tau_sh": tau_sh, "sh_fixed": fix_m, "b1_ocr_greedy": base_m, "delta_f1_pts": delta, "ci95_pts": ci,
                   "yes_ratio": yr}
    out["verdict"] = verdict_ryb(valid, ci, yr)
    # (5) chỉ mô tả
    desc: Dict[str, Any] = {"SH_tau0_popevi": apply("SH", "popevi", 0.0)[0],
                            "SH_greedy_popevi": setting_metrics(pope, greedy_parsed["SH"], images)[0]}
    for l in ("B1_ocr", "B3full"):
        if have(l, "popevi") and l in out["tau"]:
            desc[f"{l}_own_tau_popevi"] = apply(l, "popevi", out["tau"][l]["tau"])[0]
    if have("SH", "popevi_t2"):
        desc["SH_fixed_t2"] = apply("SH", "popevi_t2", tau_sh)[0]
        desc["SH_tau0_t2"] = apply("SH", "popevi_t2", 0.0)[0]
    for l in ("B1_ocr", "B3full"):
        if have(l, "popevi_t2"):
            desc[f"{l}_tau0_t2"] = apply(l, "popevi_t2", 0.0)[0]
    out["descriptive"] = desc
    return out


def report(s: Dict[str, Any]) -> str:
    L = ["# W13 · Thiên lệch \"có\" của SH — R-YB v3 (token đầu, max mỗi phía, repetition penalty như generate)", ""]
    if s.get("errors"):
        return "\n".join(L + ["## ❌ KHÔNG HỢP LỆ — không kết luận", ""] + [f"- {e}" for e in s["errors"]]) + "\n"
    pct = lambda x: f"{100 * x:.1f}%"
    L += [f"repetition_penalty (đọc từ generation_config, như generate): **{', '.join(s['repetition_penalty'])}**", ""]
    if s["missing_optional"]:
        L += [f"Run chỉ mô tả chưa chạy: {', '.join(s['missing_optional'])}", ""]
    L += ["## 1. Kiểm hợp lệ: quyết định τ = 0 trùng greedy W12", "",
          "| Cấu hình | Trùng (v3) | Câu đọc được | Lệch greedy→logit | Trùng nếu bỏ penalty (mô tả) |", "|---|---|---|---|---|"]
    for l, a in s["agreement_tau0_vs_greedy"].items():
        mm = ", ".join(f"{k} {v}" for k, v in a["mismatch_greedy_vs_logit"].items()) or "—"
        L.append(f"| {l} | {pct(a['agree'])} | {a['n_parsed']}/{a['n_total']} | {mm} | {pct(s['agreement_raw_tau0_vs_greedy'][l])} |")
    L += ["", f"Ngưỡng hợp lệ (SH, B1-ocr): ≥ {pct(AGREE_MIN)} → **{'HỢP LỆ' if s['valid'] else 'KHÔNG HỢP LỆ'}**", "",
          "## 2. Chẩn đoán (chỉ mô tả): AUC của margin = P(câu có-thật xếp trên câu không-có)", "",
          "| Cấu hình | Tập | random | popular | adversarial |", "|---|---|---|---|---|"]
    for l, a in s["auc"].items():
        if "calib" in a:
            L.append(f"| {l} | calib (gộp) | {a['calib']:.3f} | | |")
        for qset in ("popevi", "popevi_raw", "popevi_t2"):
            if qset in a:
                L.append(f"| {l} | {qset} | " + " | ".join(f"{a[qset][x]:.3f}" for x in SETTINGS) + " |")
    L += ["", "## 3. Ngưỡng τ chọn trên calibration (balanced accuracy)", "",
          "| Cấu hình | τ* | BA(τ*) | BA(τ = 0) |", "|---|---|---|---|"]
    for l, t in s["tau"].items():
        L.append(f"| {l} | {t['tau']:+.2f} | {pct(t['balanced_accuracy'])} | {pct(t['balanced_accuracy_at_0'])} |")
    r = s["rule"]
    L += ["", f"## 4. Luật R-YB — SH với τ* = {r['tau_sh']:+.2f} so với B1-ocr greedy (POPE-vi)", "",
          "| Cách chọn | F1 SH-sửa | F1 B1-ocr | ΔF1 [CI95] | yes-ratio | báo sai SH-sửa | báo sai B1-ocr |",
          "|---|---|---|---|---|---|---|"]
    for st in SETTINGS:
        a, b = r["sh_fixed"][st], r["b1_ocr_greedy"][st]
        lo, hi = r["ci95_pts"][st]
        L.append(f"| {st} | {100 * a['f1']:.2f} | {100 * b['f1']:.2f} | {r['delta_f1_pts'][st]:+.2f} [{lo:+.2f}; {hi:+.2f}] | "
                 f"{pct(a['yes_ratio'])} | {pct(a['false_alarm'])} | {pct(b['false_alarm'])} |")
    L += ["", f"## ➜ R-YB v3: **{s['verdict']}**", "",
          "## 5. Chỉ mô tả (không thuộc luật)", "",
          "| Cấu hình | Cách chọn | F1 | yes-ratio | báo sai |", "|---|---|---|---|---|"]
    for name, m in s["descriptive"].items():
        for st in SETTINGS:
            L.append(f"| {name} | {st} | {100 * m[st]['f1']:.2f} | {pct(m[st]['yes_ratio'])} | {pct(m[st]['false_alarm'])} |")
    L += ["", "## 6. Chi phí", "", "| Run | s/câu | GPU | VRAM đỉnh (GB) | top-1 là token đầu của từ ứng viên | top-1 khác |",
          "|---|---|---|---|---|---|"]
    for k, v in s["runs"].items():
        oth = ", ".join(f"{t} {c}" for t, c in v["top1_other_examples"]) or "—"
        L.append(f"| {k} | {v['sec_per_sample']} | {', '.join(v['gpus'] or [])} | {v['peak_vram_gb']} | "
                 f"{pct(v['top1_is_candidate'])} | {oth} |")
    return "\n".join(L) + "\n"


# =============================================================================
# 4) Kiểm thử offline
# =============================================================================
class _FakeTok:
    """Tokenizer giả mô phỏng đúng kiểu tách của Qwen (meta v1/v2): 'không' = kh + ông, 'Có' = C + ó."""
    vocab = {"có": [10], "Có": [40, 41], "yes": [12], "Yes": [13],
             "không": [30, 31], "Không": [21], "no": [22], "No": [23]}

    def encode(self, w, add_special_tokens=False):
        return list(self.vocab.get(w, [1, 2, 3]))

    def convert_ids_to_tokens(self, i):
        return f"<{i}>"


def _selftest_math() -> bool:
    try:
        import torch
    except ImportError:
        print("⚠ không có torch -> bỏ qua kiểm phần toán (Colab có torch, notebook chạy selftest trên Colab)")
        return False
    tk = candidate_tokens(_FakeTok())
    V = 60
    # (1) repetition penalty đúng ngữ nghĩa transformers
    x = torch.tensor([2.0, -2.0, 3.0, 0.5])
    y = apply_repetition_penalty(x, [0, 1, 1], 2.0)
    assert torch.allclose(y, torch.tensor([1.0, -4.0, 3.0, 0.5])) and torch.equal(x, torch.tensor([2.0, -2.0, 3.0, 0.5]))
    assert torch.equal(apply_repetition_penalty(x, [0], 1.0), x)
    # (2) margin = so token TỐT NHẤT mỗi phía (như greedy), KHÔNG cộng dồn — đúng ca lệch của smoke v2:
    #     có 0,23 > từng token phía không (kh 0,13; no 0,13; Không 0,02) nhưng tổng phía không 0,28 > 0,23
    p = torch.full((V,), 1e-4)
    p[10], p[30], p[22], p[21] = 0.23, 0.13, 0.13, 0.02
    r = score_from_logits(torch.log(p / p.sum()), [5], tk, 1.0)
    assert r["margin"] > 0 and r["top1_id"] == 10, r                       # greedy chọn "có" -> margin > 0
    # (3) penalty đảo quyết định như W12: "có" (10) có trong prompt bị phạt, "Không" (21) thì không
    first = torch.zeros(V)
    first[10], first[21], first[30] = 12.0, 11.6, 9.0
    r = score_from_logits(first, [5, 10, 30, 31], tk, 1.05)
    assert r["margin_raw"] > 0 > r["margin"] and r["top1_id"] == 21, r
    assert abs(r["lp"]["có"] - float(torch.log_softmax(apply_repetition_penalty(first, [5, 10, 30, 31], 1.05), -1)[10])) < 1e-5
    print("toán OK (penalty như transformers, margin = token tốt nhất mỗi phía như greedy, penalty đảo quyết định)")
    return True


def _selftest_tiny_qwen() -> bool:
    """Kiến trúc Qwen2.5-VL THẬT (transformers đang cài), trọng số ngẫu nhiên, rất nhỏ, chạy CPU vài giây.
    So với CHÍNH model.generate(do_sample=False, repetition_penalty=1.05): điểm sau penalty của bước sinh đầu
    (output_scores) phải trùng điểm v3 tính; token generate chọn phải là argmax của v3. Kiểm thêm: bỏ
    mm_token_type_ids thì logit lệch (đường forward phải giữ mọi khoá processor)."""
    try:
        import torch
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    except Exception as e:                                    # máy không có transformers/torch -> Colab sẽ kiểm
        print(f"⚠ bỏ qua kiểm Qwen2.5-VL tí hon ({type(e).__name__})")
        return False
    torch.manual_seed(0)
    img = 151655
    cfg = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2,
                         rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}, max_position_embeddings=2048),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=14, spatial_merge_size=2, temporal_patch_size=2, in_channels=3, window_size=56,
                           fullatt_block_indexes=[1]),
        image_token_id=img, video_token_id=151656, vision_start_token_id=151652)
    m = Qwen2_5_VLForConditionalGeneration(cfg).eval()
    tok = {"first": {"yes": {"có": 129133, "Có": 34, "yes": 9693, "Yes": 9454},
                     "no": {"không": 30664, "Không": 142899, "no": 2152, "No": 2753}}}
    worst, n_flip = 0.0, 0
    for k in range(6):
        g = torch.Generator().manual_seed(100 + k)
        txt = torch.randint(200, 150000, (1, 9), generator=g)
        ids = torch.cat([txt[:, :2], torch.tensor([[151652] + [img] * 4 + [151653]]), txt[:, 2:],
                         torch.tensor([[129133, 30664, 24015]])], 1)    # prompt chứa "có" và "không" như POPE-vi
        tt = (ids == img).long()
        inp = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "image_grid_thw": torch.tensor([[1, 4, 4]]),
               "pixel_values": torch.randn(16, 3 * 2 * 14 * 14, generator=g), "mm_token_type_ids": tt}
        with torch.inference_mode():
            first = forward_first(m, inp)
            out = m.generate(**inp, max_new_tokens=1, do_sample=False, repetition_penalty=1.05,
                             output_scores=True, return_dict_in_generate=True)
            no_tt = m(**{a: b for a, b in inp.items() if a != "mm_token_type_ids"}).logits[0, -1]
        gen_scores = out.scores[0][0].float()
        mine = apply_repetition_penalty(first.float(), ids[0].tolist(), 1.05)
        worst = max(worst, float((gen_scores - mine).abs().max()))
        assert int(out.sequences[0, -1]) == int(torch.argmax(mine)), "token generate ≠ argmax v3"
        r = score_from_logits(first, ids[0].tolist(), tok, 1.05)
        assert r["top1_id"] == int(out.sequences[0, -1])
        n_flip += int(torch.argmax(first)) != r["top1_id"]
        assert float((no_tt - first).abs().max()) > 1e-4, "bỏ mm_token_type_ids mà logit không đổi"
    assert worst < 1e-4, f"điểm v3 ≠ scores của generate (max |Δ| = {worst})"
    print(f"tiny Qwen2.5-VL OK (6 input: điểm v3 = scores của generate, max |Δ| {worst:.1e}; token generate = argmax v3; "
          f"bỏ mm_token_type_ids thì lệch)")
    return True


def _selftest() -> None:
    import tempfile
    from src.eval.pope_vi import _fake_world, build_questions
    import numpy as np
    # (a) τ, AUC
    assert len(tau_grid()) == 601 and tau_grid()[0] == -15.0 and tau_grid()[-1] == 15.0 and 0.0 in tau_grid()
    assert abs(auc([2, 3], [0, 1]) - 1.0) < 1e-12 and abs(auc([1], [1]) - 0.5) < 1e-12 and abs(auc([0], [1])) < 1e-12
    labs = ["yes"] * 50 + ["no"] * 50
    mg = [4.0 + 0.01 * i for i in range(50)] + [1.0 + 0.01 * i for i in range(50)]
    t = select_tau(labs, mg)
    assert 1.49 <= t["tau"] < 4.0 and t["balanced_accuracy"] == 1.0 and t["balanced_accuracy_at_0"] == 0.5, t
    assert select_tau(labs, [1.0] * 50 + [-1.0] * 50)["tau"] == 0.0
    print("tau + auc OK")
    # (b) token ứng viên
    tk = candidate_tokens(_FakeTok())
    assert tk["first"]["no"]["không"] == 30 and tk["first"]["yes"]["Có"] == 40 and tk["first_ids"] == [10, 12, 13, 21, 22, 23, 30, 40]

    class _BadTok(_FakeTok):
        vocab = {**_FakeTok.vocab, "Không": [10, 50]}
    try:
        candidate_tokens(_BadTok())
        raise AssertionError("không chặn hai phía chung token đầu")
    except ValueError:
        pass
    print("token ứng viên OK (token đầu của từ nhiều token, chặn hai phía chung token đầu)")
    if _selftest_math():
        _selftest_tiny_qwen()

    with tempfile.TemporaryDirectory() as td:
        vv, coco = _fake_world(td, n_test_img=120)
        pope = build_questions(vv, coco, n_images=60)
        pf = os.path.join(td, "pope.json")
        _write_json(pf, pope)
        c1, c2 = build_calib(vv, coco, pf, max_images=600), build_calib(vv, coco, pf, max_images=600)
        assert json.dumps(c1, sort_keys=True) == json.dumps(c2, sort_keys=True), "build calib phải tất định"
        pimg = {q["img_id"] for q in pope["data"]}
        cimg = {q["img_id"] for q in c1["data"]}
        assert not cimg & pimg and not cimg & {1, 2, 3, 4, 5}, "calib phải loại ảnh POPE-vi + ảnh train"
        assert 6 in cimg and len(cimg) == 55, (sorted(cimg), c1["stats"])
        for img in cimg:
            qs = [q for q in c1["data"] if q["img_id"] == img]
            ny = sum(q["label"] == "yes" for q in qs)
            for s in SETTINGS:
                assert sum(q["label"] == "no" and s in q["settings"] for q in qs) == ny
        t2 = build_template2(pf)
        assert [q["question_id"] for q in t2["data"]] == [q["question_id"] for q in pope["data"]]
        assert all(q["question"].startswith("Có ") and q["question"].endswith(" trong ảnh không?") for q in t2["data"])
        print("build calib + t2 OK")

        # (c) score_chunks: resume + kiểm id
        from src.data.vivqa_dataset import load_vivqa
        qfile = os.path.join(td, "q.json")
        _write_json(qfile, pope)
        samples = load_vivqa(qfile, image_dir=td)
        calls = []

        def fake_score(model, processor, image_path, prompt, tokens, penalty):
            calls.append((prompt, penalty))
            return {"margin": 1.0, "margin_raw": 1.5, "top1_id": 10, "top1_logp": -0.1}
        od = os.path.join(td, "run")
        prm = {"mode": "zero_shot"}
        n = score_chunks(samples, prm, od, 50, None, None, tk, 1.05, fake_score)
        k1 = len(calls)
        assert k1 == len(samples) and n == (len(samples) + 49) // 50 and calls[0][1] == 1.05
        os.remove(os.path.join(od, "chunks", "chunk_01.json"))
        score_chunks(samples, prm, od, 50, None, None, tk, 1.05, fake_score)
        assert len(calls) - k1 == len(samples[50:100]), "resume phải chỉ chạy lại lô thiếu"
        try:
            score_chunks(samples[::-1], prm, od, 50, None, None, tk, 1.05, fake_score)
            raise AssertionError("không chặn lô lệch id")
        except ValueError:
            pass
        assert calls[0][0].startswith("Câu hỏi: Trong ảnh có ")
        print("score_chunks OK (resume, chặn lệch id, prompt dùng chung prompt_for_sample)")

        # (d) analyze trên dữ liệu giả
        qdir = os.path.join(td, "qdir")
        _write_json(os.path.join(qdir, QFILES["calib"]), c1)
        _write_json(os.path.join(qdir, QFILES["popevi_t2"]), t2)
        exp = os.path.join(td, "exp")
        ocr = {"backend": "b", "template": "v1", "max_chars": 600, "min_conf": 0.0}
        qd = {"calib": c1["data"], "popevi": pope["data"], "popevi_t2": t2["data"]}
        qs_sha = {"calib": _sha256(os.path.join(qdir, QFILES["calib"])), "popevi": _sha256(pf),
                  "popevi_t2": _sha256(os.path.join(qdir, QFILES["popevi_t2"]))}

        def world(bias_sh: float, sep_sh: float, greedy_noise: bool = False, skip=()) -> None:
            import shutil
            shutil.rmtree(exp, ignore_errors=True)
            rng = np.random.default_rng(0)
            shift = {"SH": bias_sh, "B1_ocr": -0.5, "B3full": 0.3}
            sep = {"SH": sep_sh, "B1_ocr": 3.0, "B3full": 3.0}
            for label in CONFIGS:
                ad = None if label == "B1_ocr" else {"weights_sha256": f"w{label}"}
                for qset in QSETS:
                    recs = []
                    for q in qd[qset]:
                        m = shift[label] + (sep[label] if q["label"] == "yes" else -sep[label]) + rng.normal(0, 0.5)
                        recs.append({"question_id": q["question_id"], "margin": m, "margin_raw": m + 0.7,
                                     "top1_id": 10 if m > 0 else 30, "top1_token": "có" if m > 0 else "kh"})
                    if qset == "popevi":
                        gd = os.path.join(exp, GREEDY_W12[label])
                        preds = [{"question_id": r["question_id"],
                                  "prediction": ("không" if (greedy_noise and i % 3 == 0) == (r["margin"] > 0)
                                                 else "có")} for i, r in enumerate(recs)]
                        _write_json(os.path.join(gd, "predictions.json"), preds)
                        _write_json(os.path.join(gd, "metrics.json"), {"meta": {
                            "data_file_sha256": qs_sha["popevi"], "adapter": ad, "ocr": ocr}})
                    if (label, qset) in skip:
                        continue
                    d = os.path.join(exp, logit_run_name(label, qset))
                    _write_json(os.path.join(d, "logits.json"), recs)
                    _write_json(os.path.join(d, "meta.json"), {
                        "data_file_sha256": qs_sha[qset], "prompting_mode": "ocr", "adapter": ad, "ocr": ocr,
                        "scoring": SCORING_VERSION, "limit": None, "repetition_penalty": 1.05,
                        "tokens": {"first_ids": [10, 30]}, "sec_per_sample": 0.4, "hardware": {"gpus_seen": ["fake"]}})

        world(bias_sh=4.0, sep_sh=3.0)
        s = analyze(pf, qdir, exp, n_boot=300)
        assert not s["errors"] and s["valid"], s.get("errors")
        assert s["tau"]["SH"]["tau"] > 2.0 and s["verdict"] == "ĐẠT", (s["tau"]["SH"], s["rule"]["ci95_pts"], s["rule"]["yes_ratio"])
        assert s["descriptive"]["SH_tau0_popevi"]["random"]["yes_ratio"] > 0.8
        assert "R-YB v3: **ĐẠT**" in report(s)
        assert abs(agree_run(os.path.join(exp, logit_run_name("SH", "popevi")), os.path.join(exp, GREEDY_W12["SH"]))["agree"] - 1.0) < 1e-12
        world(bias_sh=4.0, sep_sh=0.0)
        assert analyze(pf, qdir, exp, n_boot=300)["verdict"] == "KHÔNG ĐẠT"
        world(bias_sh=4.0, sep_sh=3.0, greedy_noise=True)
        s3 = analyze(pf, qdir, exp, n_boot=300)
        assert not s3["valid"] and s3["verdict"] == "KHÔNG HỢP LỆ"
        # thiếu run chỉ mô tả -> vẫn phán quyết; thiếu run bắt buộc -> lỗi
        world(bias_sh=4.0, sep_sh=3.0, skip=OPTIONAL_RUNS)
        s5 = analyze(pf, qdir, exp, n_boot=300)
        assert s5["verdict"] == "ĐẠT" and len(s5["missing_optional"]) == len(OPTIONAL_RUNS), s5.get("errors")
        report(s5)
        world(bias_sh=4.0, sep_sh=3.0, skip=(("B1_ocr", "popevi"),))
        assert analyze(pf, qdir, exp, n_boot=300)["verdict"] == "KHÔNG HỢP LỆ"
        # adapter khác run greedy / run smoke (--limit) -> lỗi hợp lệ
        world(bias_sh=4.0, sep_sh=3.0)
        mp = os.path.join(exp, logit_run_name("SH", "calib"), "meta.json")
        m = _load_json(mp)
        m["adapter"] = {"weights_sha256": "khac"}
        _write_json(mp, m)
        s4 = analyze(pf, qdir, exp, n_boot=300)
        assert s4["errors"] and s4["verdict"] == "KHÔNG HỢP LỆ" and "KHÔNG HỢP LỆ" in report(s4)
        world(bias_sh=4.0, sep_sh=3.0)
        mp = os.path.join(exp, logit_run_name("SH", "popevi"), "meta.json")
        m = _load_json(mp)
        m["limit"] = 300
        _write_json(mp, m)
        assert any("--limit" in e for e in analyze(pf, qdir, exp, n_boot=300)["errors"])
        print("analyze OK (ĐẠT / KHÔNG ĐẠT / KHÔNG HỢP LỆ, run tuỳ chọn thiếu, run bắt buộc thiếu, adapter, smoke)")
    print("yesno_calib selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(description="W13: thiên lệch 'có' của SH — R-YB v3 (token đầu + repetition penalty như generate)")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--agree", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--coco_ann_dir")
    ap.add_argument("--vivqa_dir")
    ap.add_argument("--pope", default=os.path.join(REPO_ROOT, "experiments/W12_pope_vi_questions/pope_vi_questions_v1.json"))
    ap.add_argument("--out_dir", help="--build: thư mục ghi 2 file câu hỏi")
    ap.add_argument("--questions_dir", help="--analyze: thư mục có 2 file câu hỏi W13")
    ap.add_argument("--exp_root")
    ap.add_argument("--out", help="--analyze: thư mục ghi summary")
    ap.add_argument("--run_dir", help="--agree: thư mục run logit")
    ap.add_argument("--greedy_dir", help="--agree: thư mục run greedy W12")
    ap.add_argument("--data_file")
    ap.add_argument("--run_name")
    ap.add_argument("--drive_root", default="/content/drive/MyDrive/ViVQA-VLM")
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ocr_backend", default=None)
    ap.add_argument("--ocr_cache", default=None)
    ap.add_argument("--ocr_template", default="v1")
    ap.add_argument("--ocr_max_chars", type=int, default=600)
    ap.add_argument("--ocr_min_conf", type=float, default=0.0)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return 0
    if a.build:
        if not (a.coco_ann_dir and a.vivqa_dir and a.out_dir):
            ap.error("--build cần --coco_ann_dir --vivqa_dir --out_dir")
        for name, doc in ((QFILES["calib"], build_calib(a.vivqa_dir, a.coco_ann_dir, a.pope)),
                          (QFILES["popevi_t2"], build_template2(a.pope))):
            p = os.path.join(a.out_dir, name)
            if os.path.exists(p):
                _log(f"⏭ {name} đã có -> KHÔNG sinh lại (bộ câu hỏi đã chốt) | sha256 {_sha256(p)}")
                continue
            _write_json(p, doc)
            _log(f"✔ {name}: {doc['stats']['unique_questions']} câu | sha256 {_sha256(p)}")
        return 0
    if a.score:
        if not (a.data_file and a.run_name):
            ap.error("--score cần --data_file --run_name")
        score_run(a.data_file, a.run_name, a.drive_root, out_root=a.out_root, mode=a.mode, chunk=a.chunk,
                  image_dir=a.image_dir, limit=a.limit, adapter=a.adapter, ocr_backend=a.ocr_backend,
                  ocr_cache=a.ocr_cache, ocr_template=a.ocr_template, ocr_max_chars=a.ocr_max_chars,
                  ocr_min_conf=a.ocr_min_conf)
        return 0
    if a.agree:
        if not (a.run_dir and a.greedy_dir):
            ap.error("--agree cần --run_dir --greedy_dir")
        r = agree_run(a.run_dir, a.greedy_dir)
        print(f"trùng greedy W12: {100 * r['agree']:.2f}% trên {r['n_parsed']} câu | lệch {r['mismatch_greedy_vs_logit']}")
        if r["agree"] < AGREE_MIN:
            print(f"❌ < {100 * AGREE_MIN:.0f}% -> DỪNG, gửi Claude output này (không chạy run đầy đủ)")
            return 1
        print(f"✅ ≥ {100 * AGREE_MIN:.0f}% -> cách đo v3 tái hiện greedy, chạy tiếp")
        return 0
    if a.analyze:
        if not (a.questions_dir and a.exp_root and a.out):
            ap.error("--analyze cần --questions_dir --exp_root --out")
        s = analyze(a.pope, a.questions_dir, a.exp_root)
        _write_json(os.path.join(a.out, "yesno_calib_summary.json"), s)
        rep = report(s)
        with open(os.path.join(a.out, "yesno_calib_summary.md"), "w", encoding="utf-8", newline="\n") as fp:
            fp.write(rep)
        print(rep)
        return 0 if not s.get("errors") else 1
    ap.error("chọn một: --build | --score | --agree | --analyze | --selftest")
    return 2


if __name__ == "__main__":
    sys.exit(main())
