"""
prototype_check.py — W13 bước 4–6: backend prototype có cho ĐÚNG câu trả lời như eval không (R-PAR), độ tin cậy
hiển thị có ý nghĩa không (R-CONF), và bộ nhận diện câu có/không có làm hỏng câu thường không (R-YB-det).
Luật ghi trước: docs/W13/README.md.

Năm việc:
  --score    (GPU) chạy src.prototype.core trên ViVQA test / ViTextVQA test2000 với adapter SH: greedy y hệt eval
             + log-prob từng token; câu bị detector bắt thì chấm thêm margin có/không. Trước khi chạy: chữ ký cấu
             hình phải TRÙNG run eval W11 của SH (cùng model, 4-bit, prompt, OCR, adapter, dữ liệu).
  --parity   Smoke test: % câu trả lời trùng hệt run W11 (dừng nếu < 99%).
  --build_t3 Sinh POPE-vi mẫu câu 3 = mẫu 2 + đuôi chỉ dẫn (đúng câu prototype gửi cho câu hỏi kiểu
             "Có X trong ảnh không?").
  --analyze  R-PAR, R-CONF, R-YB-det + mô tả (độ tin cậy câu có/không, τ* trên mẫu câu 3) -> prototype_check_summary.json/.md
  --selftest Kiểm thử offline (phần model: Qwen2.5-VL tí hon kiến trúc thật nếu có torch + transformers).
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
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.prototype.core import (CORE_VERSION, DETECTOR_VERSION, TAU_STAR_SH, YESNO_INSTRUCTION,  # noqa: E402
                                is_yes_no_question, normalize_yes_no, open_confidence, yes_no_confidence)

CHECK_VERSION = "prototype-check-v1"
DATASETS = ("vivqa", "vitextvqa_official")
DATA_REL = {"vivqa": "data/vivqa/test.json", "vitextvqa_official": "data/vitextvqa_official/test_subset2000_seed42.json"}
REF_RUN = {"vivqa": "W11_SH_eval_qlora_ocr_vietocr_v1_r16_shared_ep1_vivqa_test_bf16_L4_seed42",
           "vitextvqa_official": "W11_SH_eval_qlora_ocr_vietocr_v1_r16_shared_ep1_vitextvqa_test2000_bf16_L4_seed42"}
PT_RUN = {"vivqa": "W13_PT_sh_vivqa_test_bf16_L4_seed42",
          "vitextvqa_official": "W13_PT_sh_vitextvqa_test2000_bf16_L4_seed42"}
SMOKE_RUN = "W13_PT_smoke100_sh_vivqa_test_bf16_L4_seed42"
REF_CHUNK = 500                         # chunk của run W11 (nằm trong chữ ký) -> dùng đúng để so chữ ký
T2_FILE = "experiments/W13_yesno_questions/pope_vi_questions_t2_v1.json"
T3_NAME = "pope_vi_questions_t3_v1.json"
T3_RUN = "W13_YN3_sh_qlora_ocr_vietocr_v1_r16_shared_popevi_t3_bf16_L4_seed42"
YN3_SH = {"calib": "W13_YN3_sh_qlora_ocr_vietocr_v1_r16_shared_calib_bf16_L4_seed42",
          "popevi": "W13_YN3_sh_qlora_ocr_vietocr_v1_r16_shared_popevi_bf16_L4_seed42"}
B1_GREEDY_POPE = "W12_POPE_b1_ocr_vietocr_v1_qwen25vl_popevi_v1_bf16_L4_seed42"
POPE_FILE = "experiments/W12_pope_vi_questions/pope_vi_questions_v1.json"

# Luật ghi trước (docs/W13/README.md)
PARITY_MIN = 0.99          # R-PAR: câu trả lời trùng hệt W11 >= 99% ở từng tập
SPLIT_SEED = 42            # R-CONF: chia test theo ẢNH thành nửa A (chọn ngưỡng) / nửa B (đánh giá)
N_BINS = 15                # ECE: 15 bin đều
ECE_SHOW_NUMBER = 0.05     # giao diện chỉ hiện con số khi ECE (B gộp) <= 0.05; còn lại chỉ hiện mức
EM_DROP_MAX = 1.5          # R-YB-det: EM toàn tập giảm <= 1,5 điểm khi câu bị detector bắt đi đường có/không


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
# 1) Mẫu câu 3
# =============================================================================
def build_t3(t2_file: str) -> Dict[str, Any]:
    src = _load_json(t2_file)
    data = [{**q, "question": normalize_yes_no(q["question"]), "template": "t3"} for q in src["data"]]
    return {"data": data, "stats": {"template": "t2 + đuôi chỉ dẫn = normalize_yes_no (src.prototype.core)",
                                    "source": os.path.basename(t2_file), "source_sha256": _sha256(t2_file),
                                    "unique_questions": len(data), "purpose": "chỉ mô tả (τ* với câu prototype gửi)"}}


# =============================================================================
# 2) Chạy backend (GPU)
# =============================================================================
def score_samples(samples: List[Dict[str, Any]], prompting: Dict[str, Any], out_dir: str, chunk: int,
                  gen_fn: Callable, yn_fn: Callable, gpu_name: str = "unknown") -> int:
    """gen_fn(sample, prompt) -> (pred, token_ids, logprobs); yn_fn(sample, prompt) -> margin. Theo lô, resume."""
    from src.prompting.builder import prompt_for_sample
    cdir = os.path.join(out_dir, "chunks")
    os.makedirs(cdir, exist_ok=True)
    n = (len(samples) + chunk - 1) // chunk
    for k in range(n):
        part = samples[k * chunk:(k + 1) * chunk]
        path = os.path.join(cdir, f"chunk_{k:02d}.json")
        if os.path.exists(path):
            if [r["question_id"] for r in _load_json(path)["records"]] != [s["question_id"] for s in part]:
                raise ValueError(f"{path} không khớp dữ liệu hiện tại -> dùng run_name mới")
            _log(f"[lô {k + 1}/{n}] đã có -> bỏ qua")
            continue
        t0, recs = time.time(), []
        for j, s in enumerate(part):
            pred, ids, lps = gen_fn(s, prompt_for_sample(s, prompting))
            rec = {"question_id": s["question_id"], "image": os.path.basename(s["image_path"]),
                   "question": s["question"], "answers": s["answers"], "question_type": s["question_type"],
                   "prediction": pred, "token_ids": ids, "logprobs": [round(x, 6) for x in lps]}
            if is_yes_no_question(s["question"]):
                rec["yn_margin"] = yn_fn(s, prompt_for_sample({**s, "question": normalize_yes_no(s["question"])},
                                                              prompting))
            recs.append(rec)
            if (j + 1) % 100 == 0:
                _log(f"   {j + 1}/{len(part)} | {(time.time() - t0) / (j + 1):.3f} s/câu")
        dt = time.time() - t0
        _write_json(path, {"records": recs, "time_sec": round(dt, 2), "gpu": gpu_name}, indent=None)
        _log(f"[lô {k + 1}/{n}] xong {len(part)} câu | {dt:.0f}s")
    return n


def score_run(dataset: str, run_name: str, drive_root: str, exp_root: Optional[str] = None, limit: Optional[int] = None,
              chunk: int = 500, ocr_backend: str = "paddledet_vietocr", ocr_template: str = "v1",
              ocr_max_chars: int = 600, ocr_min_conf: float = 0.0,
              config: str = os.path.join(REPO_ROOT, "configs/qwen_lora.yaml")) -> Dict[str, Any]:
    import torch
    from peft import PeftModel
    from src.data.ocr import default_cache_path, setup_ocr_prompting
    from src.data.vivqa_dataset import load_vivqa
    from src.eval.metrics import exact_match
    from src.eval.run_baseline import (adapter_fingerprint, check_or_write_signature, load_config, load_model,
                                       prepare_images, run_signature)
    from src.eval.yesno_calib import candidate_tokens, generation_penalty, score_sample
    from src.prototype.core import build_inputs, generate_from_inputs
    exp_root = exp_root or os.path.join(drive_root, "experiments")
    ref_dir = os.path.join(exp_root, REF_RUN[dataset])
    ref_meta = _load_json(os.path.join(ref_dir, "metrics.json"))["meta"]
    ref_sig = _load_json(os.path.join(ref_dir, "run_signature.json"))["sha256"]
    data_file = os.path.join(drive_root, DATA_REL[dataset])
    cfg = copy.deepcopy(load_config(config, mode="ocr"))
    cfg["run"]["name"] = run_name
    samples = load_vivqa(data_file, image_dir=prepare_images(dataset, drive_root))
    missing = [s["image_path"] for s in samples if not os.path.exists(s["image_path"])]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ảnh thiếu, vd {missing[0]}")
    ocr_stats = setup_ocr_prompting(cfg, samples, default_cache_path(drive_root, dataset, ocr_backend),
                                    template=ocr_template, max_chars=ocr_max_chars, min_conf=ocr_min_conf)
    adapter = ref_meta["adapter"]["path"]
    ad = {"path": adapter, **adapter_fingerprint(adapter)}
    if ad["weights_sha256"] != ref_meta["adapter"]["weights_sha256"]:
        raise ValueError("adapter trên Drive KHÁC adapter của run W11")
    # Điều kiện tiên quyết: cấu hình (model, 4-bit, prompt, OCR, dữ liệu, adapter) TRÙNG HỆT run eval W11
    sig_now = run_signature(cfg, _sha256(data_file), None, REF_CHUNK, adapter=ad)["sha256"]
    if sig_now != ref_sig:
        raise ValueError(f"chữ ký cấu hình {sig_now[:12]} ≠ run W11 {ref_sig[:12]} -> backend không cùng cấu hình eval")
    _log(f"== {run_name} | {dataset} | chữ ký cấu hình = W11 ({ref_sig[:12]}) ✅ | OCR {ocr_stats['pct_samples_with_text']}% câu có chữ")
    samples = samples[:limit] if limit else samples
    model, processor = load_model(cfg)
    model = PeftModel.from_pretrained(model, adapter).eval()
    penalty, gen_info = generation_penalty(model)
    yn_tokens = candidate_tokens(processor.tokenizer)
    special = list(processor.tokenizer.all_special_ids)
    mnt = cfg["prompting"].get("max_new_tokens", 32)
    out_dir = os.path.join(exp_root, run_name)
    my_sig = {"ref_signature": ref_sig, "core": CORE_VERSION, "check": CHECK_VERSION, "detector": DETECTOR_VERSION,
              "instruction": YESNO_INSTRUCTION, "limit": limit, "chunk": chunk, "repetition_penalty": penalty,
              "yn_first_tokens": yn_tokens["first"]}
    my_sig["sha256"] = hashlib.sha256(json.dumps(my_sig, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    check_or_write_signature(out_dir, my_sig)

    def gen_fn(s, prompt):
        inputs = build_inputs(processor, s["image_path"], prompt, device=model.device)
        ids, lps = generate_from_inputs(model, inputs, mnt)
        pred = processor.batch_decode([ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return pred, ids, lps

    def yn_fn(s, prompt):
        return score_sample(model, processor, s["image_path"], prompt, yn_tokens, penalty)["margin"]

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    n = score_samples(samples, cfg["prompting"], out_dir, chunk, gen_fn, yn_fn, gpu)
    recs, tsec, gpus = [], 0.0, set()
    for k in range(n):
        d = _load_json(os.path.join(out_dir, "chunks", f"chunk_{k:02d}.json"))
        recs += d["records"]
        tsec += d["time_sec"]
        gpus.add(d["gpu"])
    if [r["question_id"] for r in recs] != [s["question_id"] for s in samples]:
        raise ValueError("bản gộp không khớp danh sách câu")
    for r in recs:
        c = open_confidence(r["token_ids"], r["logprobs"], special)
        r["conf"], r["n_tokens"] = c["conf"], c["n_tokens"]
        r["em"] = exact_match(r["prediction"], r["answers"])
    import transformers
    meta = {"run_name": run_name, "dataset": dataset, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "check": CHECK_VERSION, "core": CORE_VERSION, "detector": DETECTOR_VERSION, "limit": limit,
            "num_samples": len(recs), "data_file": DATA_REL[dataset], "data_file_sha256": _sha256(data_file),
            "ref_run": REF_RUN[dataset], "ref_signature": ref_sig, "config_signature_matches_ref": True,
            "adapter": ad, "ocr": {**cfg["prompting"]["ocr"], "stats": ocr_stats}, "repetition_penalty": penalty,
            "generation_config": gen_info, "special_ids": special, "em": sum(r["em"] for r in recs) / max(len(recs), 1),
            "n_yesno_detected": sum("yn_margin" in r for r in recs),
            "hardware": {"gpus_seen": sorted(gpus)}, "time_sec": round(tsec, 1),
            "sec_per_sample": round(tsec / max(len(recs), 1), 4),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
    _write_json(os.path.join(out_dir, "records.json"), recs, indent=None)
    _write_json(os.path.join(out_dir, "meta.json"), meta)
    _log(f"== xong {len(recs)} câu | EM {100 * meta['em']:.2f} | {meta['sec_per_sample']} s/câu | {tsec / 60:.1f} phút")
    return {"meta": meta, "records": recs}


# =============================================================================
# 3) Phân tích
# =============================================================================
def parity(recs: List[Dict[str, Any]], ref_preds: Dict[str, str]) -> Dict[str, Any]:
    ids = [r["question_id"] for r in recs if r["question_id"] in ref_preds]
    if len(ids) != len(recs):
        raise ValueError(f"{len(recs) - len(ids)} câu không có trong run W11")
    bad = [{"question_id": r["question_id"], "backend": r["prediction"], "w11": ref_preds[r["question_id"]]}
           for r in recs if r["prediction"].strip() != str(ref_preds[r["question_id"]]).strip()]
    return {"n": len(recs), "identical": 1 - len(bad) / max(len(recs), 1), "n_mismatch": len(bad), "mismatches": bad[:20]}


def split_by_image(recs: List[Dict[str, Any]], seed: int = SPLIT_SEED) -> Tuple[set, set]:
    imgs = sorted({r["image"] for r in recs})
    random.Random(seed).shuffle(imgs)
    a = set(imgs[: len(imgs) // 2])
    return a, set(imgs) - a


def tertile_cuts(values: Sequence[float]) -> Tuple[float, float]:
    v = sorted(values)
    if not v:
        raise ValueError("không có giá trị để chia ngưỡng")
    return v[len(v) // 3], v[(2 * len(v)) // 3]


def bucket_of(conf: float, cuts: Tuple[float, float]) -> str:
    return "Cao" if conf >= cuts[1] else ("Trung bình" if conf >= cuts[0] else "Thấp")


def bucket_table(recs: List[Dict[str, Any]], cuts: Tuple[float, float], key: str = "conf", ok: str = "em") -> Dict[str, Any]:
    out = {}
    for b in ("Cao", "Trung bình", "Thấp"):
        sel = [r[ok] for r in recs if bucket_of(r[key], cuts) == b]
        out[b] = {"n": len(sel), "acc": (sum(sel) / len(sel)) if sel else None}
    return out


def monotone(t: Dict[str, Any]) -> bool:
    a = [t[b]["acc"] for b in ("Cao", "Trung bình", "Thấp")]
    return all(x is not None for x in a) and a[0] > a[1] > a[2]


def ece(confs: Sequence[float], oks: Sequence[float], n_bins: int = N_BINS) -> Dict[str, Any]:
    bins = [[] for _ in range(n_bins)]
    for c, o in zip(confs, oks):
        bins[min(int(c * n_bins), n_bins - 1)].append((c, o))
    tot, rows, e = len(confs), [], 0.0
    for i, b in enumerate(bins):
        if not b:
            continue
        mc, mo = sum(x for x, _ in b) / len(b), sum(y for _, y in b) / len(b)
        e += len(b) / tot * abs(mc - mo)
        rows.append({"bin": [round(i / n_bins, 3), round((i + 1) / n_bins, 3)], "n": len(b), "mean_conf": mc, "acc": mo})
    return {"ece": e, "bins": rows}


def verdict_conf(pooled_ok: bool, per_ds_ok: Dict[str, bool]) -> str:
    if pooled_ok and all(per_ds_ok.values()):
        return "ĐẠT"
    return "MỘT PHẦN" if pooled_ok else "KHÔNG ĐẠT"


def analyze(exp_root: str, repo_root: str = REPO_ROOT) -> Dict[str, Any]:
    from src.eval.pope_vi import SETTINGS, parse_yes_no
    from src.eval.yesno_calib import decide, setting_metrics
    s: Dict[str, Any] = {"check": CHECK_VERSION, "rules": {"parity_min": PARITY_MIN, "split_seed": SPLIT_SEED,
                         "n_bins": N_BINS, "ece_show_number": ECE_SHOW_NUMBER, "em_drop_max": EM_DROP_MAX,
                         "tau_star_sh": TAU_STAR_SH, "detector": DETECTOR_VERSION}, "errors": []}
    recs, metas = {}, {}
    for ds in DATASETS:
        d = os.path.join(exp_root, PT_RUN[ds])
        if not os.path.exists(os.path.join(d, "records.json")):
            s["errors"].append(f"thiếu run {PT_RUN[ds]}")
            continue
        metas[ds] = _load_json(os.path.join(d, "meta.json"))
        recs[ds] = _load_json(os.path.join(d, "records.json"))
        m = metas[ds]
        if m.get("limit") is not None or m.get("check") != CHECK_VERSION or not m.get("config_signature_matches_ref"):
            s["errors"].append(f"{ds}: run không hợp lệ cho luật (limit={m.get('limit')}, check={m.get('check')})")
    if s["errors"]:
        s["verdicts"] = None
        return s
    # (1) R-PAR
    s["parity"] = {}
    for ds in DATASETS:
        ref = {p["question_id"]: p["prediction"] for p in _load_json(os.path.join(exp_root, REF_RUN[ds], "predictions.json"))}
        p = parity(recs[ds], ref)
        p["em_backend"] = 100 * metas[ds]["em"]
        p["em_w11"] = 100 * _load_json(os.path.join(exp_root, REF_RUN[ds], "metrics.json"))["metrics"]["exact_match"]
        s["parity"][ds] = p
    v_par = "ĐẠT" if all(s["parity"][ds]["identical"] >= PARITY_MIN for ds in DATASETS) else "KHÔNG ĐẠT"
    # (2) R-CONF
    A, B = [], []
    per_ds_B: Dict[str, List[Dict[str, Any]]] = {}
    for ds in DATASETS:
        a_img, _ = split_by_image(recs[ds])
        A += [r for r in recs[ds] if r["image"] in a_img]
        per_ds_B[ds] = [r for r in recs[ds] if r["image"] not in a_img]
        B += per_ds_B[ds]
    cuts = tertile_cuts([r["conf"] for r in A])
    tb_pooled = bucket_table(B, cuts)
    tb_ds = {ds: bucket_table(per_ds_B[ds], cuts) for ds in DATASETS}
    e_pooled = ece([r["conf"] for r in B], [r["em"] for r in B])
    e_ds = {ds: ece([r["conf"] for r in per_ds_B[ds]], [r["em"] for r in per_ds_B[ds]])["ece"] for ds in DATASETS}
    v_conf = verdict_conf(monotone(tb_pooled), {ds: monotone(tb_ds[ds]) for ds in DATASETS})
    s["confidence"] = {"cuts_open": cuts, "n_A": len(A), "n_B": len(B), "B_pooled": tb_pooled, "B_by_dataset": tb_ds,
                       "ece_B_pooled": e_pooled["ece"], "ece_B_by_dataset": e_ds, "reliability_B_pooled": e_pooled["bins"],
                       "ui_shows_number": e_pooled["ece"] <= ECE_SHOW_NUMBER}
    # (3) R-YB-det: câu bị detector bắt đi đường có/không -> EM toàn tập giảm bao nhiêu
    s["detector"] = {}
    v_det = "ĐẠT"
    for ds in DATASETS:
        fired = [r for r in recs[ds] if "yn_margin" in r]
        from src.eval.metrics import exact_match
        hyb = []
        for r in recs[ds]:
            if "yn_margin" in r:
                hyb.append(exact_match("có" if r["yn_margin"] > TAU_STAR_SH else "không", r["answers"]))
            else:
                hyb.append(r["em"])
        em_g = 100 * sum(r["em"] for r in recs[ds]) / len(recs[ds])
        em_h = 100 * sum(hyb) / len(hyb)
        s["detector"][ds] = {"n": len(recs[ds]), "fired": len(fired), "em_greedy": em_g, "em_hybrid": em_h,
                             "drop": em_g - em_h, "examples": [{"question": r["question"], "answers": r["answers"][:3],
                                                                "greedy": r["prediction"],
                                                                "yesno": "có" if r["yn_margin"] > TAU_STAR_SH else "không"}
                                                               for r in fired[:10]]}
        if em_g - em_h > EM_DROP_MAX:
            v_det = "KHÔNG ĐẠT"
    s["verdicts"] = {"R-PAR": v_par, "R-CONF": v_conf, "R-YB-det": v_det}
    # (4) Chỉ mô tả: độ tin cậy câu có/không (YN3 SH calib -> ngưỡng; POPE-vi -> đúng/sai theo mức)
    pope = _load_json(os.path.join(repo_root, POPE_FILE))["data"]
    lab = {q["question_id"]: q["label"] for q in pope}
    try:
        cal_q = _load_json(os.path.join(repo_root, "experiments/W13_yesno_questions/yesno_calib_v1.json"))["data"]
        cal_lab = {q["question_id"]: q["label"] for q in cal_q}
        mc = {r["question_id"]: r["margin"] for r in _load_json(os.path.join(exp_root, YN3_SH["calib"], "logits.json"))}
        mp = {r["question_id"]: r["margin"] for r in _load_json(os.path.join(exp_root, YN3_SH["popevi"], "logits.json"))}
        cuts_yn = tertile_cuts([yes_no_confidence(m, TAU_STAR_SH) for m in mc.values()])
        rows = [{"conf": yes_no_confidence(mp[i], TAU_STAR_SH),
                 "em": float(decide(mp[i], TAU_STAR_SH) == lab[i])} for i in mp]
        cal_acc = sum(decide(mc[i], TAU_STAR_SH) == cal_lab[i] for i in mc) / len(mc)
        s["yesno_confidence"] = {"cuts_yn": cuts_yn, "popevi_by_level": bucket_table(rows, cuts_yn),
                                 "calib_acc": cal_acc, "ece_popevi": ece([r["conf"] for r in rows], [r["em"] for r in rows])["ece"]}
    except FileNotFoundError as e:
        s["yesno_confidence"] = {"skipped": str(e)}
    # (5) Chỉ mô tả: τ* với mẫu câu 3 (câu prototype gửi cho kiểu hỏi "Có X trong ảnh không?")
    t3_dir = os.path.join(exp_root, T3_RUN)
    if os.path.exists(os.path.join(t3_dir, "logits.json")):
        m3 = {r["question_id"]: r["margin"] for r in _load_json(os.path.join(t3_dir, "logits.json"))}
        images = sorted({q["img_id"] for q in pope})
        t3 = {}
        for name, tau in (("SH_tau_star_t3", TAU_STAR_SH), ("SH_tau0_t3", 0.0)):
            t3[name] = setting_metrics(pope, {i: decide(m3[i], tau) for i in m3}, images)[0]
        if os.path.exists(os.path.join(exp_root, YN3_SH["popevi"], "logits.json")):
            mp = {r["question_id"]: r["margin"] for r in _load_json(os.path.join(exp_root, YN3_SH["popevi"], "logits.json"))}
            t3["SH_tau_star_t1"] = setting_metrics(pope, {i: decide(mp[i], TAU_STAR_SH) for i in mp}, images)[0]
        gp = {p["question_id"]: parse_yes_no(p["prediction"])
              for p in _load_json(os.path.join(exp_root, B1_GREEDY_POPE, "predictions.json"))}
        t3["B1_ocr_greedy_t1"] = setting_metrics(pope, gp, images)[0]
        s["t3"] = {k: {st: {"f1": v[st]["f1"], "yes_ratio": v[st]["yes_ratio"], "false_alarm": v[st]["false_alarm"]}
                       for st in SETTINGS} for k, v in t3.items()}
    else:
        s["t3"] = {"skipped": "chưa chạy run mẫu câu 3"}
    s["runs"] = {ds: {k: metas[ds].get(k) for k in ("sec_per_sample", "peak_vram_gb", "n_yesno_detected", "hardware", "versions")}
                 for ds in DATASETS}
    return s


def report(s: Dict[str, Any]) -> str:
    L = ["# W13 · Prototype backend — parity (R-PAR), độ tin cậy (R-CONF), detector có/không (R-YB-det)", ""]
    if s.get("errors"):
        return "\n".join(L + ["## ❌ KHÔNG HỢP LỆ — không kết luận", ""] + [f"- {e}" for e in s["errors"]]) + "\n"
    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    v = s["verdicts"]
    L += ["## 1. R-PAR — backend trả lời trùng hệt run eval W11", "",
          "| Tập | Câu | Trùng hệt | Lệch | EM backend | EM W11 |", "|---|---|---|---|---|---|"]
    for ds, p in s["parity"].items():
        L.append(f"| {ds} | {p['n']} | {pct(p['identical'])} | {p['n_mismatch']} | {p['em_backend']:.2f} | {p['em_w11']:.2f} |")
    L += ["", f"➜ **R-PAR: {v['R-PAR']}** (ngưỡng ≥ {pct(PARITY_MIN)} từng tập)", ""]
    for ds, p in s["parity"].items():
        for m in p["mismatches"][:5]:
            L.append(f"- {ds} `{m['question_id']}`: backend «{m['backend']}» / W11 «{m['w11']}»")
    c = s["confidence"]
    L += ["", f"## 2. R-CONF — độ tin cậy câu thường (ngưỡng chọn trên nửa A theo ảnh: Thấp < {c['cuts_open'][0]:.3f} ≤ Trung bình < {c['cuts_open'][1]:.3f} ≤ Cao)", "",
          f"Nửa A {c['n_A']} câu (chọn ngưỡng) · nửa B {c['n_B']} câu (đánh giá).", "",
          "| Nửa B | Cao: EM (n) | Trung bình: EM (n) | Thấp: EM (n) | ECE |", "|---|---|---|---|---|"]
    def row(name, t, e):
        return f"| {name} | " + " | ".join(f"{pct(t[b]['acc'])} ({t[b]['n']})" for b in ("Cao", "Trung bình", "Thấp")) + f" | {e:.3f} |"
    L.append(row("gộp", c["B_pooled"], c["ece_B_pooled"]))
    for ds in c["B_by_dataset"]:
        L.append(row(ds, c["B_by_dataset"][ds], c["ece_B_by_dataset"][ds]))
    L += ["", f"➜ **R-CONF: {v['R-CONF']}** (EM giảm dần Cao > Trung bình > Thấp: gộp và từng tập) · giao diện "
          f"{'hiện con số' if c['ui_shows_number'] else 'chỉ hiện MỨC (ECE > 0,05), không hiện con số'}", "",
          "## 3. R-YB-det — câu bị detector bắt đi đường có/không", "",
          "| Tập | Câu bị bắt | EM greedy | EM nếu đi đường có/không | Giảm |", "|---|---|---|---|---|"]
    for ds, d in s["detector"].items():
        L.append(f"| {ds} | {d['fired']}/{d['n']} | {d['em_greedy']:.2f} | {d['em_hybrid']:.2f} | {d['drop']:+.2f} |")
    L += ["", f"➜ **R-YB-det: {v['R-YB-det']}** (giảm ≤ {EM_DROP_MAX} điểm từng tập)", ""]
    for ds, d in s["detector"].items():
        for e in d["examples"][:5]:
            L.append(f"- {ds}: «{e['question']}» gold {e['answers']} · greedy «{e['greedy']}» · có/không «{e['yesno']}»")
    y = s.get("yesno_confidence", {})
    L += ["", "## 4. Chỉ mô tả — độ tin cậy câu có/không (σ(|margin − τ*|), ngưỡng chọn trên calibration)", ""]
    if "skipped" in y:
        L.append(f"(bỏ qua: {y['skipped']})")
    else:
        t = y["popevi_by_level"]
        L += [f"Ngưỡng: {y['cuts_yn'][0]:.3f} / {y['cuts_yn'][1]:.3f} · ECE POPE-vi {y['ece_popevi']:.3f}", "",
              "| POPE-vi | Cao | Trung bình | Thấp |", "|---|---|---|---|",
              "| đúng (n) | " + " | ".join(f"{pct(t[b]['acc'])} ({t[b]['n']})" for b in ("Cao", "Trung bình", "Thấp")) + " |"]
    L += ["", "## 5. Chỉ mô tả — τ* với mẫu câu 3 (`Có X trong ảnh không? Trả lời \"có\" hoặc \"không\".`)", ""]
    if "skipped" in s["t3"]:
        L.append(f"(bỏ qua: {s['t3']['skipped']})")
    else:
        L += ["| Cấu hình | Cách chọn | F1 | yes-ratio | báo sai |", "|---|---|---|---|---|"]
        for k, vv in s["t3"].items():
            for st, m in vv.items():
                L.append(f"| {k} | {st} | {100 * m['f1']:.2f} | {pct(m['yes_ratio'])} | {pct(m['false_alarm'])} |")
    L += ["", "## 6. Chi phí", "", "| Tập | s/câu | VRAM đỉnh (GB) | câu bị detector bắt |", "|---|---|---|---|"]
    for ds, r in s["runs"].items():
        L.append(f"| {ds} | {r['sec_per_sample']} | {r['peak_vram_gb']} | {r['n_yesno_detected']} |")
    return "\n".join(L) + "\n"


# =============================================================================
# 4) Kiểm thử offline
# =============================================================================
def _selftest_tiny() -> bool:
    """Qwen2.5-VL tí hon kiến trúc thật: generate_from_inputs cho đúng token như model.generate thường
    (cách gọi của generate_answer) và log-prob = log_softmax(scores)."""
    try:
        import torch
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    except Exception as e:
        print(f"⚠ bỏ qua kiểm Qwen2.5-VL tí hon ({type(e).__name__})")
        return False
    from src.prototype.core import generate_from_inputs
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
    m.generation_config.repetition_penalty = 1.05             # như generation_config của Qwen2.5-VL-3B-Instruct
    for k in range(4):
        g = torch.Generator().manual_seed(200 + k)
        txt = torch.randint(200, 150000, (1, 7), generator=g)
        ids = torch.cat([txt[:, :2], torch.tensor([[151652] + [img] * 4 + [151653]]), txt[:, 2:]], 1)
        inp = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "image_grid_thw": torch.tensor([[1, 4, 4]]),
               "pixel_values": torch.randn(16, 3 * 2 * 14 * 14, generator=g), "mm_token_type_ids": (ids == img).long()}
        new_ids, lps = generate_from_inputs(m, inp, max_new_tokens=6)
        with torch.inference_mode():
            plain = m.generate(**inp, max_new_tokens=6, do_sample=False)[0][ids.shape[1]:].tolist()
        assert new_ids == plain, (new_ids, plain)
        assert len(lps) == len(new_ids) and all(lp <= 0 for lp in lps)
    print("tiny Qwen2.5-VL OK (generate + output_scores cho đúng token như generate thường của eval)")
    return True


def _selftest() -> None:
    import tempfile
    # (a) detector + chuẩn hoá
    yes = ["Trong ảnh có con chó không?", "Đây có phải là quán cà phê không?", "Cửa hàng đã mở cửa chưa?",
           "Người này đang cười phải không?", "Có xe máy trong ảnh hay không?",
           'Trong ảnh có chai không? Trả lời "có" hoặc "không".']
    no = ["Biển hiệu ghi gì không?", "Có bao nhiêu người không?", "những gì đậu trong một cảng hàng không",
          "cái gì đang bay trên không", "còn số nào không ?", "Màu của chiếc xe là gì?", ""]
    assert all(is_yes_no_question(q) for q in yes), [q for q in yes if not is_yes_no_question(q)]
    assert not any(is_yes_no_question(q) for q in no), [q for q in no if is_yes_no_question(q)]
    t1 = 'Trong ảnh có chai không? Trả lời "có" hoặc "không".'
    assert normalize_yes_no(t1) == t1 and normalize_yes_no("Có chai trong ảnh không") == \
        'Có chai trong ảnh không? Trả lời "có" hoặc "không".'
    for f in (POPE_FILE, T2_FILE):
        p = os.path.join(REPO_ROOT, f)
        if os.path.exists(p):
            qs = _load_json(p)["data"]
            assert all(is_yes_no_question(q["question"]) for q in qs), f
    print("detector OK (bắt 100% câu POPE t1/t2, bỏ qua câu hỏi có từ để hỏi / danh từ 'hàng không')")
    # (b) độ tin cậy
    c = open_confidence([5, 6, 151645], [-0.1, -0.3, -0.01], [151645])
    assert abs(c["conf"] - math.exp(-0.2)) < 1e-12 and c["n_tokens"] == 2
    assert open_confidence([151645], [-0.2], [151645])["conf"] == 0.0
    assert abs(yes_no_confidence(4.85, 4.85) - 0.5) < 1e-12 and yes_no_confidence(10, 4.85) == yes_no_confidence(-0.3, 4.85)
    assert tertile_cuts(list(range(9))) == (3, 6) and bucket_of(6, (3, 6)) == "Cao" and bucket_of(2.9, (3, 6)) == "Thấp"
    e = ece([0.95] * 10 + [0.05] * 10, [1] * 10 + [0] * 10)
    assert e["ece"] < 0.051 and len(e["bins"]) == 2
    print("độ tin cậy OK (bỏ token đặc biệt, tertile, ECE)")
    _selftest_tiny()
    # (c) score_samples + analyze trên thế giới giả
    with tempfile.TemporaryDirectory() as td:
        rng = random.Random(1)
        exp = os.path.join(td, "exp")

        def mk_samples(ds, n):
            return [{"question_id": f"{ds}_{i}", "question": ("Trong ảnh có chó không?" if i % 50 == 0 else f"Cái gì {i}?"),
                     "answers": ["mèo"], "question_type": "x", "image_path": f"/img/{ds}_{i // 2}.jpg", "ocr_texts": []}
                    for i in range(n)]
        calls = []
        for ds in DATASETS:
            smp = mk_samples(ds, 400)

            def gen_fn(s, prompt, _ds=ds):
                calls.append(prompt)
                lp = -rng.uniform(0.0, 2.0)                       # tin cậy liên tục; câu tin cậy cao đúng nhiều hơn
                good = rng.random() < math.exp(lp)
                return ("mèo" if good else "chó"), [11, 12, 151645], [lp, lp, -0.01]

            out = os.path.join(exp, PT_RUN[ds])
            n = score_samples(smp, {"mode": "zero_shot"}, out, 150, gen_fn, lambda s, p: 6.0)
            recs = []
            for k in range(n):
                recs += _load_json(os.path.join(out, "chunks", f"chunk_{k:02d}.json"))["records"]
            from src.eval.metrics import exact_match
            for r in recs:
                cc = open_confidence(r["token_ids"], r["logprobs"], [151645])
                r["conf"], r["em"] = cc["conf"], exact_match(r["prediction"], r["answers"])
            _write_json(os.path.join(out, "records.json"), recs)
            _write_json(os.path.join(out, "meta.json"), {"limit": None, "check": CHECK_VERSION,
                                                          "config_signature_matches_ref": True,
                                                          "em": sum(r["em"] for r in recs) / len(recs)})
            ref = os.path.join(exp, REF_RUN[ds])
            _write_json(os.path.join(ref, "predictions.json"),
                        [{"question_id": r["question_id"], "prediction": r["prediction"] if i else "khác"}
                         for i, r in enumerate(recs)])
            _write_json(os.path.join(ref, "metrics.json"), {"metrics": {"exact_match": 0.6}})
        assert sum("Trả lời" in p for p in calls) == 0 and len(calls) == 800
        s = analyze(exp)
        assert not s["errors"], s["errors"]
        assert s["verdicts"]["R-PAR"] == "ĐẠT" and s["parity"]["vivqa"]["n_mismatch"] == 1
        assert s["verdicts"]["R-CONF"] == "ĐẠT", s["confidence"]["B_pooled"]
        assert s["detector"]["vivqa"]["fired"] == 8 and s["verdicts"]["R-YB-det"] == "ĐẠT"
        assert "skipped" in s["yesno_confidence"] and "skipped" in s["t3"]
        rep = report(s)
        assert "R-PAR: ĐẠT" in rep and "R-CONF: ĐẠT" in rep
        # parity < 99% -> KHÔNG ĐẠT; run smoke (limit) -> không hợp lệ
        ref = os.path.join(exp, REF_RUN["vivqa"], "predictions.json")
        _write_json(ref, [{**p, "prediction": "x"} if i % 10 == 0 else p for i, p in enumerate(_load_json(ref))])
        assert analyze(exp)["verdicts"]["R-PAR"] == "KHÔNG ĐẠT"
        mp = os.path.join(exp, PT_RUN["vivqa"], "meta.json")
        _write_json(mp, {**_load_json(mp), "limit": 100})
        assert analyze(exp)["errors"] and "KHÔNG HỢP LỆ" in report(analyze(exp))
        # resume: xoá 1 lô -> chỉ chạy lại lô đó
        out = os.path.join(exp, PT_RUN["vivqa"])
        os.remove(os.path.join(out, "chunks", "chunk_01.json"))
        k0 = len(calls)
        score_samples(mk_samples("vivqa", 400), {"mode": "zero_shot"}, out, 150,
                      lambda s, p: (calls.append(p) or ("mèo", [11], [-0.1])), lambda s, p: 6.0)
        assert len(calls) - k0 == 150
    print("score_samples + analyze OK (resume, parity, R-CONF, detector, chặn run smoke)")
    print("prototype_check selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(description="W13: kiểm backend prototype (R-PAR, R-CONF, R-YB-det)")
    for f in ("score", "parity", "build_t3", "analyze", "selftest"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--run_name")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--drive_root", default="/content/drive/MyDrive/ViVQA-VLM")
    ap.add_argument("--exp_root", default=None)
    ap.add_argument("--run_dir")
    ap.add_argument("--ref_dir")
    ap.add_argument("--out")
    ap.add_argument("--ocr_backend", default="paddledet_vietocr")
    ap.add_argument("--ocr_template", default="v1")
    ap.add_argument("--ocr_max_chars", type=int, default=600)
    ap.add_argument("--ocr_min_conf", type=float, default=0.0)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return 0
    if a.build_t3:
        p = a.out or os.path.join(REPO_ROOT, "experiments/W13_yesno_questions", T3_NAME)
        if os.path.exists(p):
            _log(f"⏭ {p} đã có -> KHÔNG sinh lại | sha256 {_sha256(p)}")
            return 0
        _write_json(p, build_t3(os.path.join(REPO_ROOT, T2_FILE)))
        _log(f"✔ {p} | sha256 {_sha256(p)}")
        return 0
    if a.score:
        if not (a.dataset and a.run_name):
            ap.error("--score cần --dataset --run_name")
        score_run(a.dataset, a.run_name, a.drive_root, exp_root=a.exp_root, limit=a.limit, chunk=a.chunk,
                  ocr_backend=a.ocr_backend, ocr_template=a.ocr_template, ocr_max_chars=a.ocr_max_chars,
                  ocr_min_conf=a.ocr_min_conf)
        return 0
    if a.parity:
        recs = _load_json(os.path.join(a.run_dir, "records.json"))
        ref = {p["question_id"]: p["prediction"] for p in _load_json(os.path.join(a.ref_dir, "predictions.json"))}
        p = parity(recs, ref)
        print(f"trùng hệt run W11: {100 * p['identical']:.2f}% trên {p['n']} câu | lệch {p['n_mismatch']}")
        for m in p["mismatches"][:10]:
            print(f"   {m['question_id']}: backend «{m['backend']}» / W11 «{m['w11']}»")
        if p["identical"] < PARITY_MIN:
            print(f"❌ < {100 * PARITY_MIN:.0f}% -> DỪNG, gửi Claude output này (không chạy run đầy đủ)")
            return 1
        print(f"✅ ≥ {100 * PARITY_MIN:.0f}% -> backend tái hiện eval, chạy tiếp")
        return 0
    if a.analyze:
        exp = a.exp_root or os.path.join(a.drive_root, "experiments")
        s = analyze(exp)
        out = a.out or os.path.join(exp, "W13_prototype_check_L4_seed42")
        _write_json(os.path.join(out, "prototype_check_summary.json"), s)
        rep = report(s)
        with open(os.path.join(out, "prototype_check_summary.md"), "w", encoding="utf-8", newline="\n") as fp:
            fp.write(rep)
        print(rep)
        return 0 if not s.get("errors") else 1
    ap.error("chọn một: --score | --parity | --build_t3 | --analyze | --selftest")
    return 2


if __name__ == "__main__":
    sys.exit(main())
