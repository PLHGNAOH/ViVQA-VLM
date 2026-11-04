"""
pope_vi.py — H4 (W12): QLoRA có làm tăng hallucination vật thể không? Đo kiểu POPE bằng tiếng Việt (POPE-vi).

Vấn đề: EM/ANLS không cho biết model có "bịa" vật thể không có trong ảnh hay không. POPE (Li et al., EMNLP 2023)
hỏi "Trong ảnh có X không?" với X vừa là vật CÓ trong ảnh, vừa là vật KHÔNG có (chọn theo 3 cách: random / popular /
adversarial), rồi tính Accuracy / Precision / Recall / F1 / yes-ratio. Luật ghi trước: docs/W12/README.md (mục H4).

Ba việc:
  --build   Sinh file câu hỏi (định dạng ViVQA, chạy thẳng bằng src.eval.run_baseline --data_file):
            ảnh ViVQA test có annotation COCO 2014, >= 3 lớp, KHÔNG nằm trong ViVQA train/val; 500 ảnh, seed 42.
            Mỗi ảnh: 3 câu "có" (dùng chung cho 3 cách chọn) + 3 câu "không" cho mỗi cách chọn.
            Một cặp (ảnh, lớp) trùng giữa các cách chọn chỉ hỏi MỘT lần (greedy -> cùng prompt = cùng đáp án),
            trường "settings" ghi cặp đó thuộc cách chọn nào.
  --score   Đọc predictions.json của 5 cấu hình (B1-plain, B1-ocr, C3, B3-full, SH), kiểm hợp lệ, tính metric,
            bootstrap theo ảnh, phán quyết theo luật -> pope_vi_summary.json + .md
  --selftest  Kiểm thử offline (không GPU, không mạng).

Chạy:
    python -m src.eval.pope_vi --build --coco_ann_dir <dir có instances_train2014.json, instances_val2014.json> \
        --vivqa_dir <drive>/data/vivqa --out <exp>/W12_pope_vi_questions/pope_vi_questions_v1.json
    python -m src.eval.pope_vi --score --questions <file> --exp_root <drive>/experiments \
        --run B1_plain=<run> --run B1_ocr=<run> --run C3=<run> --run B3full=<run> --run SH=<run> \
        --out <exp>/W12_pope_vi_L4_seed42
    python -m src.eval.pope_vi --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OBJECTS_FILE = os.path.join(REPO_ROOT, "configs", "pope_vi_objects.json")
SEED = 42
N_IMAGES = 500
N_POS = 3
N_NEG = 3
MIN_CLASSES = 3
SETTINGS = ("random", "popular", "adversarial")
QUESTION_TEMPLATE = 'Trong ảnh có {obj} không? Trả lời "có" hoặc "không".'
BUILD_VERSION = "pope-vi-build-v1"

MARGIN_PTS = 2.0              # kế thừa W02 (H8 cũ): F1 giảm > 2 điểm -> bác bỏ
MAX_UNPARSED = 0.05           # > 5% câu không đọc được -> cấu hình không hợp lệ cho H4
N_BOOT = 10_000
CONFIGS = ("B1_plain", "B1_ocr", "C3", "B3full", "SH")
# (adapter, mốc): C3 train không OCR -> so B1 cùng prompt thường; B3-full, SH train có OCR -> so B1 cùng prompt ocr
COMPARISONS = (("C3", "B1_plain"), ("B3full", "B1_ocr"), ("SH", "B1_ocr"))
H4_ADAPTERS = ("C3", "B3full")       # SH: kiểm tra an toàn cho prototype, không tính vào H4
EXPECT_MODE = {"B1_plain": "zero_shot", "B1_ocr": "ocr", "C3": "zero_shot", "B3full": "ocr", "SH": "ocr"}
# Run eval gốc (W09–W11) -> adapter phải trùng dấu vân tay trọng số của run này
REF_EVAL = {
    "C3": "W09_C3_eval_qlora_r16_vivqa_test_bf16_L4_seed42",
    "B3full": "W10_B3full_eval_qlora_ocr_vietocr_v1_r16_full_ep1_vitextvqa_testfull_bf16_L4_seed42",
    "SH": "W11_SH_eval_qlora_ocr_vietocr_v1_r16_shared_ep1_vivqa_test_bf16_L4_seed42",
}
VERDICTS = ("KHÔNG TĂNG HALLUCINATION", "TĂNG HALLUCINATION", "CHƯA KẾT LUẬN")
H4_VERDICTS = ("ỦNG HỘ", "BÁC BỎ", "MỘT PHẦN")


# =============================================================================
# Tiện ích
# =============================================================================
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=1)
        fp.write("\n")
    os.replace(tmp, path)


def load_objects(path: str = OBJECTS_FILE) -> Dict[str, Any]:
    d = _load_json(path)
    objs = d["objects"]
    if len(objs) != 80 or len(set(objs.values())) != 80:
        raise ValueError(f"{path}: cần đúng 80 lớp với 80 tên tiếng Việt khác nhau")
    group_of: Dict[str, set] = defaultdict(set)
    for name, members in d["confusable_groups"].items():
        for m in members:
            if m not in objs:
                raise ValueError(f"nhóm {name}: lớp '{m}' không có trong objects")
            group_of[m] |= set(members)
    return {"version": d["version"], "objects": objs, "group_of": dict(group_of), "sha256": _sha256(path)}


def blocked_classes(present: Iterable[str], group_of: Dict[str, set]) -> set:
    """Lớp không được hỏi làm câu 'không': lớp có trong ảnh + mọi lớp cùng nhóm dễ nhầm với chúng."""
    out = set(present)
    for p in present:
        out |= group_of.get(p, set())
    return out


# =============================================================================
# 1) Sinh câu hỏi
# =============================================================================
def coco_split_of(filename: str) -> str:
    m = re.match(r"COCO_(train2014|val2014)_\d+\.jpg$", os.path.basename(filename))
    if not m:
        raise ValueError(f"Tên ảnh không theo khuôn COCO 2014: {filename}")
    return m.group(1)


def load_coco_classes(coco_ann_dir: str, needed: Dict[str, set]) -> Dict[Tuple[str, int], set]:
    """{(split, img_id): {tên lớp COCO}} cho các ảnh cần. Đọc instances_<split>.json."""
    out: Dict[Tuple[str, int], set] = {}
    for split, ids in sorted(needed.items()):
        path = os.path.join(coco_ann_dir, f"instances_{split}.json")
        d = _load_json(path)
        cat = {c["id"]: c["name"] for c in d["categories"]}
        for img_id in ids:
            out[(split, img_id)] = set()
        for a in d["annotations"]:
            key = (split, a["image_id"])
            if key in out:
                out[key].add(cat[a["category_id"]])
    return out


def build_questions(vivqa_dir: str, coco_ann_dir: str, objects_path: str = OBJECTS_FILE,
                    n_images: int = N_IMAGES, seed: int = SEED) -> Dict[str, Any]:
    objs = load_objects(objects_path)
    vi, group_of = objs["objects"], objs["group_of"]
    # val.json (nếu có) cũng loại: tách từ train, có thể đã dùng để chọn epoch
    used = [s for s in ("train", "val", "test") if os.path.exists(os.path.join(vivqa_dir, f"{s}.json"))]
    if "train" not in used or "test" not in used:
        raise FileNotFoundError(f"{vivqa_dir} thiếu train.json / test.json")
    split_recs = {s: _load_json(os.path.join(vivqa_dir, f"{s}.json")) for s in used}
    seen_ids = {r["img_id"] for s in used if s != "test" for r in split_recs[s]}
    test_imgs: Dict[int, str] = {}
    for r in split_recs["test"]:
        test_imgs.setdefault(int(r["img_id"]), r["image"])
    unseen = {i: f for i, f in test_imgs.items() if i not in seen_ids}
    needed: Dict[str, set] = defaultdict(set)
    for i, f in unseen.items():
        needed[coco_split_of(f)].add(i)
    classes = load_coco_classes(coco_ann_dir, needed)
    unknown = {c for v in classes.values() for c in v} - set(vi)
    if unknown:
        raise ValueError(f"lớp COCO không có trong từ điển: {sorted(unknown)}")
    pool = {i: sorted(classes[(coco_split_of(f), i)]) for i, f in unseen.items()
            if len(classes[(coco_split_of(f), i)]) >= MIN_CLASSES}
    if len(pool) < n_images:
        raise ValueError(f"pool chỉ có {len(pool)} ảnh đủ điều kiện, cần {n_images}")

    # Thống kê trên TOÀN pool (không chỉ 500 ảnh chọn) -> popular / adversarial
    freq = Counter(c for cs in pool.values() for c in cs)
    cooc: Dict[str, Counter] = defaultdict(Counter)
    for cs in pool.values():
        for a in cs:
            for b in cs:
                if a != b:
                    cooc[a][b] += 1
    popular_rank = sorted(vi, key=lambda c: (-freq[c], c))

    chosen = sorted(random.Random(seed).sample(sorted(pool), n_images))
    rows: Dict[Tuple[int, str], Dict[str, Any]] = {}
    per_image: List[Dict[str, Any]] = []

    def add(img_id: int, cls: str, label: str, setting: str) -> None:
        key = (img_id, cls)
        if key not in rows:
            rows[key] = {"question_id": f"pope_{img_id}_{cls.replace(' ', '_')}",
                         "question": QUESTION_TEMPLATE.format(obj=vi[cls]),
                         "answers": ["có" if label == "yes" else "không"],
                         "image": unseen[img_id], "img_id": img_id,
                         "question_type": "pope_yes" if label == "yes" else "pope_no",
                         "coco_class": cls, "label": label, "settings": []}
        if rows[key]["label"] != label:
            raise AssertionError(f"{key}: một cặp vừa là câu có vừa là câu không")
        rows[key]["settings"].append(setting)

    for img_id in chosen:
        present = pool[img_id]
        rng = random.Random(seed * 1_000_003 + img_id)
        pos = sorted(rng.sample(present, N_POS))
        allowed = [c for c in sorted(vi) if c not in blocked_classes(present, group_of)]
        adv_score = {c: sum(cooc[p][c] for p in present) for c in allowed}
        negs = {
            "random": sorted(rng.sample(allowed, N_NEG)),
            "popular": [c for c in popular_rank if c in set(allowed)][:N_NEG],
            "adversarial": sorted(allowed, key=lambda c: (-adv_score[c], -freq[c], c))[:N_NEG],
        }
        for c in pos:
            for s in SETTINGS:
                add(img_id, c, "yes", s)
        for s in SETTINGS:
            for c in negs[s]:
                add(img_id, c, "no", s)
        per_image.append({"img_id": img_id, "present": present, "pos": pos, **{f"neg_{s}": negs[s] for s in SETTINGS}})

    questions = sorted(rows.values(), key=lambda r: (r["img_id"], r["label"] != "yes", r["coco_class"]))
    stats = {
        "build_version": BUILD_VERSION, "objects_version": objs["version"], "objects_sha256": objs["sha256"],
        "seed": seed, "template": QUESTION_TEMPLATE,
        "test_images": len(test_imgs), "test_images_unseen": len(unseen), "pool_images": len(pool),
        "chosen_images": len(chosen), "unique_questions": len(questions),
        "items_per_setting": {s: sum(s in r["settings"] for r in questions) for s in SETTINGS},
        "yes_questions": sum(r["label"] == "yes" for r in questions),
        "dedup_saved": len(chosen) * (N_POS + N_NEG * len(SETTINGS)) - len(questions),
        "vivqa_sha256": {s: _sha256(os.path.join(vivqa_dir, f"{s}.json")) for s in used},
        "coco_ann_sha256": {s: _sha256(os.path.join(coco_ann_dir, f"instances_{s}.json")) for s in sorted(needed)},
        "popular_top10": popular_rank[:10],
    }
    # khoá "data": src.data.vivqa_dataset.load_vivqa đọc đúng khoá này -> chạy thẳng bằng run_baseline --data_file
    return {"data": questions, "per_image": per_image, "stats": stats}


# =============================================================================
# 2) Chấm
# =============================================================================
def parse_yes_no(pred: Any) -> Optional[str]:
    """Luật đọc (ghi trước): từ ĐẦU TIÊN 'có'/'yes' -> yes, 'không'/'no' -> no;
    không phải hai loại đó mà chứa 'không có' / 'không thấy' -> no; còn lại -> None (không đọc được)."""
    t = unicodedata.normalize("NFC", str(pred or "")).lower()
    t = " ".join(re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE).split())
    first = t.split(" ", 1)[0] if t else ""
    if first in ("có", "yes"):
        return "yes"
    if first in ("không", "no"):
        return "no"
    if "không có" in t or "không thấy" in t:
        return "no"
    return None


def outcome(label: str, parsed: Optional[str]) -> str:
    """TP/FN/FP/TN. Không đọc được = SAI theo cách nghiêm (câu 'có' -> FN, câu 'không' -> FP)."""
    if label == "yes":
        return "TP" if parsed == "yes" else "FN"
    return "TN" if parsed == "no" else "FP"


def prf(tp: float, fp: float, fn: float, tn: float) -> Dict[str, float]:
    n = tp + fp + fn + tn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"accuracy": (tp + tn) / n if n else 0.0, "precision": p, "recall": r, "f1": f1}


def per_image_counts(questions: List[Dict[str, Any]], parsed: Dict[str, Optional[str]],
                     images: Sequence[int]) -> Dict[str, Dict[str, List[int]]]:
    """{setting: {TP/FP/FN/TN/YES: list theo thứ tự `images`}} -> dùng cho bootstrap theo ảnh."""
    idx = {i: k for k, i in enumerate(images)}
    out = {s: {k: [0] * len(images) for k in ("TP", "FP", "FN", "TN", "YES")} for s in SETTINGS}
    for q in questions:
        o = outcome(q["label"], parsed[q["question_id"]])
        for s in q["settings"]:
            out[s][o][idx[q["img_id"]]] += 1
            out[s]["YES"][idx[q["img_id"]]] += parsed[q["question_id"]] == "yes"
    return out


def metrics_of(counts: Dict[str, List[int]]) -> Dict[str, float]:
    tot = {k: sum(v) for k, v in counts.items()}
    m = prf(tot["TP"], tot["FP"], tot["FN"], tot["TN"])
    m["yes_ratio"] = tot["YES"] / max(tot["TP"] + tot["FP"] + tot["FN"] + tot["TN"], 1)
    m["n"] = tot["TP"] + tot["FP"] + tot["FN"] + tot["TN"]
    return m


def bootstrap_delta_f1(a: Dict[str, List[int]], b: Dict[str, List[int]], n_boot: int = N_BOOT,
                       seed: int = SEED) -> Tuple[float, float]:
    """CI95 (điểm %) của F1(a) − F1(b), lấy mẫu lại THEO ẢNH (ghép cặp: cùng ảnh cho cả hai cấu hình)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(a["TP"])
    idx = rng.integers(0, n, size=(n_boot, n))

    def f1(c: Dict[str, List[int]]) -> "np.ndarray":
        tp, fp, fn = (np.asarray(c[k])[idx].sum(axis=1).astype(float) for k in ("TP", "FP", "FN"))
        p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        return np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)

    d = 100 * (f1(a) - f1(b))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def verdict_adapter(ci_by_setting: Dict[str, Tuple[float, float]], valid: bool) -> str:
    if not valid:
        return "CHƯA KẾT LUẬN"
    if any(hi < -MARGIN_PTS for _, hi in ci_by_setting.values()):
        return "TĂNG HALLUCINATION"
    if all(lo > -MARGIN_PTS for lo, _ in ci_by_setting.values()):
        return "KHÔNG TĂNG HALLUCINATION"
    return "CHƯA KẾT LUẬN"


def verdict_h4(per_adapter: Dict[str, str]) -> str:
    v = [per_adapter[a] for a in H4_ADAPTERS]
    if all(x == "KHÔNG TĂNG HALLUCINATION" for x in v):
        return "ỦNG HỘ"
    if any(x == "TĂNG HALLUCINATION" for x in v):
        return "BÁC BỎ"
    return "MỘT PHẦN"


def check_run(label: str, meta: Dict[str, Any], preds: List[Dict[str, Any]], questions: List[Dict[str, Any]],
              q_sha: str, ref_meta: Optional[Dict[str, Any]], ocr_ref: Optional[Dict[str, Any]] = None) -> List[str]:
    """Kiểm hợp lệ 1 cấu hình (trước khi đọc kết quả). Trả list lỗi (rỗng = hợp lệ)."""
    errs = []
    if [p["question_id"] for p in preds] != [q["question_id"] for q in questions]:
        errs.append(f"{label}: predictions không khớp danh sách câu hỏi ({len(preds)} vs {len(questions)})")
    if meta.get("data_file_sha256") != q_sha:
        errs.append(f"{label}: chạy trên file câu hỏi khác (sha256 {str(meta.get('data_file_sha256'))[:12]}…)")
    if meta.get("prompting_mode") != EXPECT_MODE[label]:
        errs.append(f"{label}: mode {meta.get('prompting_mode')} ≠ {EXPECT_MODE[label]}")
    ad = meta.get("adapter")
    if ref_meta is None:
        if ad:
            errs.append(f"{label}: zero-shot nhưng lại có adapter")
    else:
        want = (ref_meta.get("adapter") or {}).get("weights_sha256")
        if not ad or ad.get("weights_sha256") != want:
            errs.append(f"{label}: adapter khác run gốc {REF_EVAL[label]}")
    if EXPECT_MODE[label] == "ocr":
        # mọi cấu hình ocr (kể cả B1_ocr) phải cùng backend/template/ngưỡng với SH -> cùng prompt
        keys = ("backend", "template", "max_chars", "min_conf")
        got, ref = meta.get("ocr") or {}, ocr_ref or {}
        if not got or any(got.get(k) != ref.get(k) for k in keys):
            errs.append(f"{label}: thiết lập OCR khác SH ({ {k: got.get(k) for k in keys} })")
    return errs


def score(questions_file: str, runs: Dict[str, Dict[str, Any]],
          ref_metas: Dict[str, Dict[str, Any]], n_boot: int = N_BOOT) -> Dict[str, Any]:
    """runs = {nhãn: {"meta": metrics.json['meta'], "predictions": predictions.json}}."""
    qdoc = _load_json(questions_file)
    questions, q_sha = qdoc["data"], _sha256(questions_file)
    images = sorted({q["img_id"] for q in questions})
    out: Dict[str, Any] = {"questions_file": os.path.basename(questions_file), "questions_sha256": q_sha,
                           "n_images": len(images), "n_unique_questions": len(questions),
                           "rules": {"margin_pts": MARGIN_PTS, "max_unparsed": MAX_UNPARSED, "n_boot": n_boot,
                                     "seed": SEED, "comparisons": COMPARISONS, "h4_adapters": H4_ADAPTERS},
                           "configs": {}, "errors": []}
    counts: Dict[str, Any] = {}
    valid: Dict[str, bool] = {}
    for label in CONFIGS:
        r = runs[label]
        errs = check_run(label, r["meta"], r["predictions"], questions, q_sha, ref_metas.get(label),
                         ocr_ref=(ref_metas.get("SH") or {}).get("ocr"))
        out["errors"] += errs
        parsed = {p["question_id"]: parse_yes_no(p["prediction"]) for p in r["predictions"]}
        unparsed = sum(v is None for v in parsed.values()) / max(len(parsed), 1)
        valid[label] = not errs and unparsed <= MAX_UNPARSED
        if errs:
            out["configs"][label] = {"valid": False}
            continue
        counts[label] = per_image_counts(questions, parsed, images)
        fp_cls = Counter(q["coco_class"] for q in questions
                         if q["label"] == "no" and parsed[q["question_id"]] == "yes")
        examples = [{"question_id": p["question_id"], "prediction": p["prediction"]}
                    for p in r["predictions"] if parsed[p["question_id"]] is None][:10]
        out["configs"][label] = {
            "valid": valid[label], "unparsed_rate": unparsed, "unparsed_examples": examples,
            "by_setting": {s: metrics_of(counts[label][s]) for s in SETTINGS},
            "top_fp_classes": fp_cls.most_common(10),
            "sec_per_sample": r["meta"].get("sec_per_sample"), "gpus": (r["meta"].get("hardware") or {}).get("gpus_seen"),
        }
    if out["errors"]:
        out["verdicts"], out["h4"] = None, None
        return out
    out["comparisons"], per_adapter = {}, {}
    for a, b in COMPARISONS:
        ci = {s: bootstrap_delta_f1(counts[a][s], counts[b][s], n_boot) for s in SETTINGS}
        delta = {s: 100 * (out["configs"][a]["by_setting"][s]["f1"] - out["configs"][b]["by_setting"][s]["f1"])
                 for s in SETTINGS}
        per_adapter[a] = verdict_adapter(ci, valid[a] and valid[b])
        out["comparisons"][f"{a}_vs_{b}"] = {"delta_f1_pts": delta, "ci95_pts": ci, "verdict": per_adapter[a],
                                             "valid": valid[a] and valid[b]}
    out["descriptive"] = {"B1_ocr_vs_B1_plain": {
        s: {"delta_f1_pts": 100 * (out["configs"]["B1_ocr"]["by_setting"][s]["f1"]
                                   - out["configs"]["B1_plain"]["by_setting"][s]["f1"]),
            "ci95_pts": bootstrap_delta_f1(counts["B1_ocr"][s], counts["B1_plain"][s], n_boot)} for s in SETTINGS}}
    out["verdicts"] = per_adapter
    out["h4"] = verdict_h4(per_adapter)
    return out


def report(s: Dict[str, Any]) -> str:
    L = [f"# POPE-vi (H4) — {s['n_images']} ảnh · {s['n_unique_questions']} câu duy nhất · sha256 {s['questions_sha256'][:12]}…", ""]
    if s["errors"]:
        L += ["## ❌ KHÔNG HỢP LỆ — không kết luận", ""] + [f"- {e}" for e in s["errors"]]
        return "\n".join(L) + "\n"
    L += ["| Cấu hình | Cách chọn | Acc | Prec | Rec | **F1** | yes-ratio | không đọc được |", "|---|---|---|---|---|---|---|---|"]
    for c in CONFIGS:
        cf = s["configs"][c]
        for st in SETTINGS:
            m = cf["by_setting"][st]
            L.append(f"| {c}{'' if cf['valid'] else ' ⚠'} | {st} | {100 * m['accuracy']:.2f} | {100 * m['precision']:.2f} | "
                     f"{100 * m['recall']:.2f} | **{100 * m['f1']:.2f}** | {100 * m['yes_ratio']:.1f}% | {100 * cf['unparsed_rate']:.2f}% |")
    L += ["", "## So sánh theo luật (ΔF1 điểm, CI95 bootstrap theo ảnh)", "",
          "| So sánh | random | popular | adversarial | Phán quyết |", "|---|---|---|---|---|"]
    for k, v in s["comparisons"].items():
        cells = [f"{v['delta_f1_pts'][st]:+.2f} [{v['ci95_pts'][st][0]:+.2f}, {v['ci95_pts'][st][1]:+.2f}]" for st in SETTINGS]
        L.append(f"| {k} | " + " | ".join(cells) + f" | **{v['verdict']}** |")
    d = s["descriptive"]["B1_ocr_vs_B1_plain"]
    L += ["", "Chỉ mô tả — B1_ocr vs B1_plain: " + " · ".join(
        f"{st} {d[st]['delta_f1_pts']:+.2f} [{d[st]['ci95_pts'][0]:+.2f}, {d[st]['ci95_pts'][1]:+.2f}]" for st in SETTINGS),
          "", f"## ➜ H4 THEO LUẬT: **{s['h4']}** (C3: {s['verdicts']['C3']} · B3-full: {s['verdicts']['B3full']}) · "
              f"SH (prototype): {s['verdicts']['SH']}"]
    return "\n".join(L) + "\n"


# =============================================================================
# 3) Kiểm thử offline
# =============================================================================
def _fake_world(td: str, n_test_img: int = 40) -> Tuple[str, str]:
    """ViVQA giả + annotation COCO giả. Ảnh 1..n_test_img thuộc test; ảnh 1..5 cũng có trong train (phải bị loại)."""
    objs = load_objects()["objects"]
    names = sorted(objs)
    rng = random.Random(7)
    vv, coco = os.path.join(td, "vivqa"), os.path.join(td, "coco")
    os.makedirs(vv)
    os.makedirs(coco)
    test = [{"question_id": f"t{i}", "question": "?", "answers": ["x"], "img_id": i,
             "image": f"COCO_{'val2014' if i % 2 else 'train2014'}_{i:012d}.jpg", "question_type": "object"}
            for i in range(1, n_test_img + 1)]
    train = [{**test[i - 1], "question_id": f"tr{i}"} for i in range(1, 6)]
    for name, recs in (("train", train), ("val", []), ("test", test)):
        _write_json(os.path.join(vv, f"{name}.json"), recs)
    for split in ("train2014", "val2014"):
        cats = [{"id": k + 1, "name": n} for k, n in enumerate(names)]
        anns = []
        for i in range(1, n_test_img + 1):
            if (split == "val2014") != bool(i % 2):
                continue
            k = 2 if i == 6 else 3 + i % 4                         # ảnh 6 chỉ có 2 lớp -> bị loại
            present = ["person", "couch"] + rng.sample([n for n in names if n not in ("person", "couch")], k)
            present = present[:k]
            anns += [{"image_id": i, "category_id": names.index(c) + 1} for c in present]
        _write_json(os.path.join(coco, f"instances_{split}.json"), {"categories": cats, "annotations": anns})
    return vv, coco


def _selftest() -> None:
    import tempfile
    # (a) đọc câu trả lời
    cases = {"Có": "yes", "có.": "yes", "Có, trong ảnh có một con chó": "yes", "Yes": "yes",
             "Không": "no", "không có": "no", "No.": "no", "Trong ảnh không có chó": "no",
             "Có, nhưng không thấy rõ": "yes", "con chó": None, "": None, "Trong ảnh có một con chó": None,
             "KHÔNG": "no", unicodedata.normalize("NFD", "Có"): "yes"}
    for k, v in cases.items():
        assert parse_yes_no(k) == v, (k, parse_yes_no(k), v)
    assert outcome("yes", None) == "FN" and outcome("no", None) == "FP"
    assert outcome("no", "no") == "TN" and outcome("yes", "yes") == "TP"
    # (b) từ điển + nhóm dễ nhầm
    o = load_objects()
    assert "chair" in blocked_classes(["couch"], o["group_of"]) and "truck" in blocked_classes(["car"], o["group_of"])
    print("parse + objects OK")

    with tempfile.TemporaryDirectory() as td:
        vv, coco = _fake_world(td)
        b1 = build_questions(vv, coco, n_images=20)
        b2 = build_questions(vv, coco, n_images=20)
        assert json.dumps(b1, sort_keys=True) == json.dumps(b2, sort_keys=True), "build phải tất định"
        st = b1["stats"]
        assert st["test_images_unseen"] == 35 and st["pool_images"] == 34, st
        qs = b1["data"]
        imgs = {q["img_id"] for q in qs}
        assert len(imgs) == 20 and not imgs & {1, 2, 3, 4, 5, 6}, "phải loại ảnh train và ảnh < 3 lớp"
        assert all(v == 20 * 6 for v in st["items_per_setting"].values()), st["items_per_setting"]
        assert st["unique_questions"] + st["dedup_saved"] == 20 * 12
        pi = {p["img_id"]: p for p in b1["per_image"]}
        for q in qs:
            p = pi[q["img_id"]]
            if q["label"] == "no":
                assert q["coco_class"] not in blocked_classes(p["present"], o["group_of"]), q
            else:
                assert q["coco_class"] in p["present"] and set(q["settings"]) == set(SETTINGS)
        assert len({q["question_id"] for q in qs}) == len(qs)
        assert all(q["question"].startswith("Trong ảnh có ") for q in qs)
        print("build OK (tất định, loại ảnh train, loại lớp dễ nhầm, dedup)")

        # (c) chấm + phán quyết trên dự đoán giả
        qf = os.path.join(td, "q.json")
        _write_json(qf, b1)
        qsha = _sha256(qf)
        ref = {k: {"adapter": {"weights_sha256": f"w{k}"}, "ocr": {"backend": "b", "template": "v1",
                                                                    "max_chars": 600, "min_conf": 0.0}} for k in REF_EVAL}

        def mk(label: str, answer_fn) -> Dict[str, Any]:
            meta = {"data_file_sha256": qsha, "prompting_mode": EXPECT_MODE[label],
                    "adapter": ref[label]["adapter"] if label in ref else None,
                    "ocr": ref["SH"]["ocr"] if EXPECT_MODE[label] == "ocr" else None}
            return {"meta": meta, "predictions": [{"question_id": q["question_id"], "prediction": answer_fn(q)} for q in qs]}

        truth = lambda q: "có" if q["label"] == "yes" else "không"
        yes_all = lambda q: "Có"
        runs = {"B1_plain": mk("B1_plain", truth), "B1_ocr": mk("B1_ocr", truth), "C3": mk("C3", truth),
                "B3full": mk("B3full", truth), "SH": mk("SH", yes_all)}
        s = score(qf, runs, ref, n_boot=500)
        assert not s["errors"], s["errors"]
        assert s["verdicts"] == {"C3": "KHÔNG TĂNG HALLUCINATION", "B3full": "KHÔNG TĂNG HALLUCINATION",
                                 "SH": "TĂNG HALLUCINATION"}, s["verdicts"]
        assert s["h4"] == "ỦNG HỘ"
        assert abs(s["configs"]["SH"]["by_setting"]["random"]["yes_ratio"] - 1.0) < 1e-9
        # B3-full nói "có" hết -> BÁC BỎ
        s2 = score(qf, {**runs, "B3full": mk("B3full", yes_all)}, ref, n_boot=500)
        assert s2["h4"] == "BÁC BỎ"
        # C3 trả lời lan man > 5% -> không hợp lệ -> CHƯA KẾT LUẬN -> H4 MỘT PHẦN
        s3 = score(qf, {**runs, "C3": mk("C3", lambda q: "con chó")}, ref, n_boot=500)
        assert s3["verdicts"]["C3"] == "CHƯA KẾT LUẬN" and s3["h4"] == "MỘT PHẦN"
        assert s3["configs"]["C3"]["unparsed_rate"] == 1.0
        # adapter sai / file câu hỏi sai / mode sai -> lỗi, không kết luận
        bad = mk("C3", truth)
        bad["meta"]["adapter"] = {"weights_sha256": "khác"}
        s4 = score(qf, {**runs, "C3": bad}, ref, n_boot=200)
        assert s4["h4"] is None and any("adapter khác" in e for e in s4["errors"])
        bad = mk("B1_ocr", truth)
        bad["meta"]["data_file_sha256"] = "x"
        bad["meta"]["prompting_mode"] = "zero_shot"
        s5 = score(qf, {**runs, "B1_ocr": bad}, ref, n_boot=200)
        assert len(s5["errors"]) >= 2 and s5["h4"] is None
        bad = mk("B1_ocr", truth)
        bad["meta"]["ocr"] = {**bad["meta"]["ocr"], "template": "v2"}
        s6 = score(qf, {**runs, "B1_ocr": bad}, ref, n_boot=200)
        assert s6["h4"] is None and any("OCR khác SH" in e for e in s6["errors"])
        # file câu hỏi đọc được bằng loader thật của run_baseline
        try:
            from src.data.vivqa_dataset import load_vivqa
            ld = load_vivqa(qf, image_dir="/img")
            assert [x["question_id"] for x in ld] == [q["question_id"] for q in qs] and ld[0]["image_path"] == os.path.join("/img", qs[0]["image"])
            print("load_vivqa đọc file câu hỏi OK")
        except ImportError:
            print("(bỏ qua kiểm load_vivqa: chạy ngoài repo)")
        # CI theo ảnh: hai cấu hình giống hệt -> CI = [0, 0]
        assert s["comparisons"]["C3_vs_B1_plain"]["ci95_pts"]["random"] == (0.0, 0.0)
        assert "H4 THEO LUẬT" in report(s) and "KHÔNG HỢP LỆ" in report(s4)
        print("score + verdict OK")
    # (d) luật phán quyết ở biên
    assert verdict_adapter({"a": (-1.9, 1.0), "b": (-0.5, 0.5), "c": (-1.0, 0.0)}, True) == "KHÔNG TĂNG HALLUCINATION"
    assert verdict_adapter({"a": (-1.9, 1.0), "b": (-2.1, 0.5), "c": (-1.0, 0.0)}, True) == "CHƯA KẾT LUẬN"
    assert verdict_adapter({"a": (-5.0, -2.1), "b": (-0.5, 0.5), "c": (-1.0, 0.0)}, True) == "TĂNG HALLUCINATION"
    assert verdict_adapter({"a": (1.0, 2.0)}, False) == "CHƯA KẾT LUẬN"
    print("pope_vi selftest OK")


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="POPE-vi (H4, W12): sinh câu hỏi + chấm + phán quyết theo luật ghi trước")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--coco_ann_dir")
    ap.add_argument("--vivqa_dir")
    ap.add_argument("--objects", default=OBJECTS_FILE)
    ap.add_argument("--questions")
    ap.add_argument("--exp_root")
    ap.add_argument("--run", action="append", default=[], help="NHÃN=tên_run, nhãn ∈ " + ", ".join(CONFIGS))
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return 0
    if a.build:
        if not (a.coco_ann_dir and a.vivqa_dir and a.out):
            ap.error("--build cần --coco_ann_dir --vivqa_dir --out")
        if os.path.exists(a.out):
            print(f"⏭ {a.out} đã có -> KHÔNG sinh lại (file câu hỏi đã chốt). sha256 {_sha256(a.out)}")
            return 0
        b = build_questions(a.vivqa_dir, a.coco_ann_dir, a.objects)
        _write_json(a.out, b)
        st = b["stats"]
        print(f"✔ {st['unique_questions']} câu duy nhất | {st['chosen_images']} ảnh (pool {st['pool_images']} / "
              f"{st['test_images_unseen']} ảnh test chưa thấy lúc train / {st['test_images']} ảnh test) | "
              f"mỗi cách chọn {st['items_per_setting']} | tiết kiệm {st['dedup_saved']} câu nhờ dedup")
        print("   popular top10:", ", ".join(st["popular_top10"]))
        for q in b["data"][:8]:
            print(f"   {q['image']} | {q['question']} -> {q['answers'][0]} {q['settings']}")
        print(f"   sha256 {_sha256(a.out)}")
        return 0
    if a.score:
        if not (a.questions and a.exp_root and a.out and a.run):
            ap.error("--score cần --questions --exp_root --run (x5) --out")
        names = dict(r.split("=", 1) for r in a.run)
        if set(names) != set(CONFIGS):
            ap.error(f"--run phải có đủ {CONFIGS}")
        runs = {}
        for label, run in names.items():
            d = os.path.join(a.exp_root, run)
            runs[label] = {"meta": _load_json(os.path.join(d, "metrics.json"))["meta"],
                           "predictions": _load_json(os.path.join(d, "predictions.json"))}
        refs = {k: _load_json(os.path.join(a.exp_root, v, "metrics.json"))["meta"] for k, v in REF_EVAL.items()}
        s = score(a.questions, runs, refs)
        s["runs"] = names
        os.makedirs(a.out, exist_ok=True)
        _write_json(os.path.join(a.out, "pope_vi_summary.json"), s)
        md = report(s)
        with open(os.path.join(a.out, "pope_vi_summary.md"), "w", encoding="utf-8", newline="\n") as fp:
            fp.write(md)
        print(md)
        return 1 if s["errors"] else 0
    ap.error("chọn --build, --score hoặc --selftest")
    return 2


if __name__ == "__main__":
    sys.exit(main())
