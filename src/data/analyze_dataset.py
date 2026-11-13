"""
analyze_dataset.py — Week-2 Dataset Deep-Dive for ViVQA-VLM (produces REAL numbers).

WHAT THIS SCRIPT DOES
    Computes reproducible descriptive statistics for the two project datasets
    required by the master plan (§9): **ViVQA** (Tran et al., PACLIC 2021) and
    **ViTextVQA** (Nguyen et al., 2024). It also (optionally) profiles the local
    **ViVQA-X** JSONs that already live in this repo, clearly labelled as a
    RELATED but DIFFERENT dataset (translated VQA-X w/ explanations) — never as a
    substitute for ViVQA.

HARD RULES HONOURED (Week-2 brief)
    * NEVER fabricate output. If a dataset cannot be loaded we write a stats JSON
      with ``"status": "PENDING_RUN"`` plus the exact error/blocker and exit that
      dataset cleanly — no made-up sizes/metrics.
    * All Vietnamese text is normalised with **Unicode NFC**.
    * Fixed random seed = 42 (logged into every stats JSON).
    * Question-type classifier is a transparent, editable rule set (see
      ``QUESTION_TYPE_RULES``) and is flagged as *heuristic / exploratory only*.

LOADING STRATEGY (per dataset)
    1. HuggingFace ``datasets`` — try a list of candidate hub IDs.
    2. ``--data_root`` local fallback — read JSON/CSV from disk.
    If both fail → PENDING_RUN JSON with the collected errors.

OUTPUTS (under experiments/W02_dataset_stats/)
    {vivqa,vitextvqa,vivqa_x}_stats.json         # full descriptive stats
    samples/<dataset>__<qtype>.json              # 5 real samples per question-type

USAGE
    python src/data/analyze_dataset.py
    python src/data/analyze_dataset.py --data_root data --top_k 30
    python src/data/analyze_dataset.py --only vivqa_x
    python src/data/analyze_dataset.py --vivqa_hf_id <hub/id> --vitextvqa_hf_id <hub/id>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import traceback
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

SEED = 42
SCRIPT_VERSION = "w02-analyze-1.0"

# Repo-root-relative default output directory (allowed path per Week-2 brief).
DEFAULT_OUT_DIR = os.path.join("experiments", "W02_dataset_stats")


# =============================================================================
# 0) TEXT NORMALISATION — Unicode NFC (mandatory for Vietnamese)
# =============================================================================
def nfc(text: Any) -> str:
    """Return the NFC-normalised string form of ``text`` (safe on non-str)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFC", text).strip()


def token_len(text: str) -> int:
    """Whitespace token count of an already-normalised string."""
    return len(text.split()) if text else 0


# =============================================================================
# 1) QUESTION-TYPE CLASSIFIER  (HEURISTIC, EXPLORATORY ONLY)
# -----------------------------------------------------------------------------
# Editable rule set. Each entry: type -> list of lowercase Vietnamese cue
# substrings. Precedence is defined by CLASSIFY_ORDER (checked top to bottom).
# This is a rough, rule-based first pass for EDA / error-analysis bootstrapping,
# NOT a validated annotation. Human review is required before any thesis claim.
# =============================================================================
QUESTION_TYPE_RULES: Dict[str, List[str]] = {
    # Scene-text / OCR — reading text printed in the image.
    "Text-reading": [
        "chữ gì", "dòng chữ", "hàng chữ", "biển", "bảng hiệu", "ghi gì",
        "ghi là gì", "viết gì", "viết là gì", "nội dung", "dòng chữ nào",
        "chữ trên", "chữ ở", "tên cửa hàng", "tên biển", "logo", "nhãn hiệu",
        "khẩu hiệu", "số điện thoại", "địa chỉ", "giá bao nhiêu", "thương hiệu",
    ],
    # Counting — quantities.
    "Counting": [
        "bao nhiêu", "mấy", "số lượng", "đếm", "có mấy",
    ],
    # Yes/No — polar questions.
    "Yes/No": [
        "có phải", "phải không", "đúng không", "có đúng", "có... không",
        "có không", "có hay không", "phải hay không",
    ],
    # Reasoning — why / how / cause-effect / action inference.
    "Reasoning": [
        "tại sao", "vì sao", "lý do", "như thế nào", "thế nào",
        "bằng cách nào", "để làm gì", "đang làm gì", "làm gì",
    ],
    # Object-recognition — what/which/where/color/who identification.
    "Object-recognition": [
        "là gì", "cái gì", "con gì", "vật gì", "con vật", "màu gì",
        "màu sắc", "ở đâu", "chỗ nào", "ai", "người nào", "loại gì",
        "hình gì", "đồ vật", "cây gì", "hoa gì",
    ],
}

# Precedence: more specific / higher-signal buckets first.
CLASSIFY_ORDER = [
    "Text-reading",
    "Counting",
    "Yes/No",
    "Reasoning",
    "Object-recognition",
]

OTHER_TYPE = "Other"
ALL_TYPES = CLASSIFY_ORDER + [OTHER_TYPE]

DISCLAIMER = (
    "[DISCLAIMER] Question-type labels are produced by a RULE-BASED Vietnamese "
    "cue-word heuristic. They are EXPLORATORY ONLY (EDA / error-analysis "
    "bootstrapping) and are NOT a validated annotation. Human review required "
    "before any thesis-bound claim."
)


def classify_question(question_nfc: str) -> str:
    """Rule-based Vietnamese question-type classifier (heuristic)."""
    q = question_nfc.lower()

    # Special-case the discontinuous 'có ... không ?' polar pattern.
    if q.startswith("có ") and ("không" in q or q.endswith("không")):
        return "Yes/No"
    if q.rstrip(" ?.!").endswith("không"):
        return "Yes/No"

    for qtype in CLASSIFY_ORDER:
        for cue in QUESTION_TYPE_RULES[qtype]:
            if "..." in cue:
                # discontinuous cue like "có... không"
                left, right = [c.strip() for c in cue.split("...", 1)]
                if left in q and right in q:
                    return qtype
            elif cue in q:
                return qtype
    return OTHER_TYPE


# =============================================================================
# 2) UNIFIED SAMPLE SCHEMA
# -----------------------------------------------------------------------------
# Every loader returns a list of records shaped like:
#   {"question_id", "question", "answers"(List[str]), "answer"(str),
#    "image_id", "image_path", "split"}
# =============================================================================
def make_record(
    question: Any,
    answers: List[str],
    image_id: Any = "",
    image_path: Any = "",
    question_id: Any = "",
    split: str = "unknown",
) -> Dict[str, Any]:
    ans = [nfc(a) for a in answers if nfc(a)]
    return {
        "question_id": nfc(question_id),
        "question": nfc(question),
        "answers": ans,
        "answer": ans[0] if ans else "",
        "image_id": nfc(image_id),
        "image_path": nfc(image_path),
        "split": split,
    }


# =============================================================================
# 3) LOADERS
# =============================================================================
def _coerce_answers(item: Dict[str, Any]) -> List[str]:
    """Extract an answer list from many possible field shapes."""
    for key in ("answers", "answer", "label", "labels", "answer_text"):
        if key not in item:
            continue
        val = item[key]
        if isinstance(val, list):
            out = []
            for a in val:
                if isinstance(a, dict):
                    a = a.get("answer") or a.get("text") or ""
                if isinstance(a, (str, int, float)) and str(a).strip():
                    out.append(str(a))
            if out:
                return out
        elif isinstance(val, (str, int, float)) and str(val).strip():
            return [str(val)]
    return []


def _first_present(item: Dict[str, Any], keys: Tuple[str, ...], default: str = "") -> Any:
    for k in keys:
        if k in item and item[k] not in (None, ""):
            return item[k]
    return default


def _records_from_dicts(records: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            continue
        question = _first_present(item, ("question", "question_text", "text", "query"))
        answers = _coerce_answers(item)
        image_id = _first_present(item, ("image_id", "imageId", "img_id", "image"))
        image_path = _first_present(
            item, ("image_path", "image_name", "img_path", "filename", "file_name", "image")
        )
        qid = _first_present(item, ("question_id", "qid", "id"), default=idx)
        out.append(make_record(question, answers, image_id, image_path, qid, split))
    return out


def _adapt_generic(rows: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    """One HF row == one QA record (e.g. minhnguyent546/ViVQA)."""
    return _records_from_dicts(rows, split_name)


def _adapt_nested_img_ann(rows: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    """COCO-style nested container: each row holds ``images`` + ``annotations``
    lists (e.g. nhonhoccode/ViTextVQA). Expand annotations into QA records and
    resolve image_id -> filename via the images list.
    """
    id2file: Dict[Any, str] = {}
    annotations: List[Dict[str, Any]] = []
    for row in rows:
        for img in row.get("images", []) or []:
            if isinstance(img, dict) and "id" in img:
                id2file[img["id"]] = img.get("filename") or img.get("file_name") or str(img["id"])
        anns = row.get("annotations", []) or []
        if isinstance(anns, list):
            annotations.extend(a for a in anns if isinstance(a, dict))

    out: List[Dict[str, Any]] = []
    for a in annotations:
        img_id = a.get("image_id")
        filename = id2file.get(img_id, str(img_id))
        out.append(make_record(
            question=a.get("question", ""),
            answers=_coerce_answers(a),
            image_id=f"{split_name}:{img_id}",   # split-prefixed to keep global uniqueness
            image_path=filename,
            question_id=a.get("id", ""),
            split=split_name,
        ))
    return out


HF_ADAPTERS: Dict[str, Callable[[List[Dict[str, Any]], str], List[Dict[str, Any]]]] = {
    "generic": _adapt_generic,
    "nested_img_ann": _adapt_nested_img_ann,
}


def load_hf(
    hf_ids: List[str],
    seed: int,
    adapter: str = "generic",
) -> Tuple[Optional[Dict[str, List[Dict[str, Any]]]], List[str]]:
    """Try to load a dataset from HuggingFace hub by candidate IDs.

    Returns (splits_dict, errors). ``splits_dict`` maps split-name -> records.
    ``adapter`` selects how raw rows are turned into unified QA records.
    """
    errors: List[str] = []
    adapt_fn = HF_ADAPTERS.get(adapter, _adapt_generic)
    try:
        from datasets import load_dataset  # noqa: WPS433 (import inside fn is intentional)
    except Exception as exc:  # datasets not installed / import failed
        errors.append(f"import datasets failed: {type(exc).__name__}: {exc}")
        return None, errors

    for hf_id in hf_ids:
        if not hf_id:
            continue

        # Attempt A: standard (non-streaming) full load.
        splits: Dict[str, List[Dict[str, Any]]] = {}
        try:
            ds = load_dataset(hf_id)
            for split_name in ds.keys():
                rows = [dict(r) for r in ds[split_name]]
                splits[split_name] = adapt_fn(rows, split_name)
            if any(splits.values()):
                errors.append(f"OK: loaded '{hf_id}' [{adapter}] with splits {list(splits.keys())}")
                return splits, errors
        except Exception as exc:
            errors.append(f"load_dataset('{hf_id}') failed: {type(exc).__name__}: {exc}")

        # Attempt B: streaming fallback, per-split tolerant (some mirrors ship an
        # empty split file that breaks the eager builder but not streaming).
        try:
            ids = load_dataset(hf_id, streaming=True)
        except Exception as exc:
            errors.append(f"stream load_dataset('{hf_id}') failed: {type(exc).__name__}: {exc}")
            continue
        splits = {}
        for split_name in ids.keys():
            try:
                rows = [dict(r) for r in ids[split_name]]
                if rows:
                    splits[split_name] = adapt_fn(rows, split_name)
            except Exception as exc:
                errors.append(f"stream split '{hf_id}:{split_name}' skipped: "
                              f"{type(exc).__name__}: {str(exc)[:120]}")
        if any(splits.values()):
            errors.append(f"OK: streamed '{hf_id}' [{adapter}] with splits {list(splits.keys())}")
            return splits, errors
    return None, errors


def _load_json_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        if isinstance(raw.get("data"), list):
            return raw["data"]
        if isinstance(raw.get("annotations"), list):
            return raw["annotations"]
        # dict-of-records
        vals = list(raw.values())
        if vals and isinstance(vals[0], dict):
            return vals
        return []
    if isinstance(raw, list):
        return raw
    return []


def load_local(spec: Dict[str, Any], data_root: str) -> Tuple[Optional[Dict[str, List[Dict[str, Any]]]], List[str]]:
    """Load a dataset from local files described by ``spec['local']``.

    spec['local'] is a dict: {split_name: [candidate relative paths...]}.
    """
    errors: List[str] = []
    local_map: Dict[str, List[str]] = spec.get("local", {})
    splits: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, candidates in local_map.items():
        found = None
        for rel in candidates:
            for base in (data_root, "."):
                path = rel if os.path.isabs(rel) else os.path.join(base, rel)
                if os.path.exists(path):
                    found = path
                    break
            if found:
                break
        if not found:
            errors.append(f"local split '{split_name}': none of {candidates} found under '{data_root}'")
            continue
        try:
            rows = _load_json_file(found)
            splits[split_name] = _records_from_dicts(rows, split_name)
            errors.append(f"OK: local '{split_name}' <- {found} ({len(splits[split_name])} rows)")
        except Exception as exc:
            errors.append(f"reading '{found}' failed: {type(exc).__name__}: {exc}")
    if any(splits.values()):
        return splits, errors
    return None, errors


# =============================================================================
# 4) STATISTICS
# =============================================================================
def _dist(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    values_sorted = sorted(values)
    return {
        "count": len(values),
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "mean": round(statistics.mean(values), 3),
        "median": statistics.median(values),
        "p25": values_sorted[int(0.25 * (len(values) - 1))],
        "p75": values_sorted[int(0.75 * (len(values) - 1))],
        "p90": values_sorted[int(0.90 * (len(values) - 1))],
    }


def compute_stats(
    dataset_key: str,
    display_name: str,
    source_kind: str,
    source_detail: str,
    splits: Dict[str, List[Dict[str, Any]]],
    top_k: int,
    load_log: List[str],
) -> Dict[str, Any]:
    all_records: List[Dict[str, Any]] = []
    split_sizes: Dict[str, int] = {}
    for split_name, recs in splits.items():
        split_sizes[split_name] = len(recs)
        all_records.extend(recs)

    q_lens = [token_len(r["question"]) for r in all_records]
    a_lens = [token_len(r["answer"]) for r in all_records]

    unique_images = {r["image_id"] for r in all_records if r["image_id"]}
    if not unique_images:
        unique_images = {r["image_path"] for r in all_records if r["image_path"]}

    answer_counter: Counter = Counter(r["answer"] for r in all_records if r["answer"])
    top_answers = [{"answer": a, "count": c} for a, c in answer_counter.most_common(top_k)]

    # Question-type distribution (heuristic).
    qtype_counter: Counter = Counter()
    for r in all_records:
        qt = classify_question(r["question"])
        r["_qtype"] = qt
        qtype_counter[qt] += 1
    total = len(all_records) or 1
    qtype_dist = {
        qt: {
            "count": qtype_counter.get(qt, 0),
            "pct": round(100.0 * qtype_counter.get(qt, 0) / total, 2),
        }
        for qt in ALL_TYPES
    }

    one_word = sum(1 for r in all_records if token_len(r["answer"]) == 1)
    le_three = sum(1 for r in all_records if 1 <= token_len(r["answer"]) <= 3)

    return {
        "dataset_key": dataset_key,
        "display_name": display_name,
        "status": "OK",
        "seed": SEED,
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_kind": source_kind,          # "huggingface" | "local"
        "source_detail": source_detail,
        "num_images": len(unique_images),
        "num_qa": len(all_records),
        "split_sizes": split_sizes,
        "question_length_tokens": _dist(q_lens),
        "answer_length_tokens": _dist(a_lens),
        "answer_stats": {
            "unique_answers": len(answer_counter),
            "pct_answer_one_word": round(100.0 * one_word / total, 2),
            "pct_answer_le_3_words": round(100.0 * le_three / total, 2),
        },
        "top_answers": top_answers,
        "question_type_distribution": qtype_dist,
        "question_type_method": "rule_based_vi_cue_words (heuristic, exploratory only)",
        "load_log": load_log,
    }


def pending_stats(
    dataset_key: str,
    display_name: str,
    errors: List[str],
) -> Dict[str, Any]:
    return {
        "dataset_key": dataset_key,
        "display_name": display_name,
        "status": "PENDING_RUN",
        "seed": SEED,
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "blocker": "Dataset could not be loaded from HuggingFace or local --data_root.",
        "resolution_hint": (
            "Install `datasets` + provide network access, OR place the raw files "
            "under --data_root and re-run. See 'errors' for the exact failures."
        ),
        "errors": errors,
        "num_images": None,
        "num_qa": None,
    }


def dump_samples(
    out_dir: str,
    dataset_key: str,
    splits: Dict[str, List[Dict[str, Any]]],
    n_per_type: int,
    rng: random.Random,
) -> Dict[str, int]:
    """Write up to ``n_per_type`` real samples per question-type."""
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    by_type: Dict[str, List[Dict[str, Any]]] = {t: [] for t in ALL_TYPES}
    for recs in splits.values():
        for r in recs:
            qt = r.get("_qtype") or classify_question(r["question"])
            by_type.setdefault(qt, []).append(r)

    written: Dict[str, int] = {}
    for qt, recs in by_type.items():
        if not recs:
            continue
        rng.shuffle(recs)
        chosen = recs[:n_per_type]
        payload = [
            {
                "image_path": r["image_path"],
                "image_id": r["image_id"],
                "question": r["question"],
                "answer": r["answer"],
                "answers": r["answers"],
                "predicted_type": qt,
                "split": r["split"],
            }
            for r in chosen
        ]
        safe_qt = qt.replace("/", "-").replace(" ", "_")
        path = os.path.join(samples_dir, f"{dataset_key}__{safe_qt}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {"dataset": dataset_key, "predicted_type": qt,
                 "disclaimer": DISCLAIMER, "samples": payload},
                fh, ensure_ascii=False, indent=2,
            )
        written[qt] = len(payload)
    return written


# =============================================================================
# 5) DATASET REGISTRY
# -----------------------------------------------------------------------------
# HF candidate IDs are ATTEMPTED at runtime; if they do not resolve the script
# records the exact error and falls back to local, then PENDING_RUN. No numbers
# are ever invented.
# =============================================================================
def build_registry(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    return {
        "vivqa": {
            "display_name": "ViVQA (Tran et al., PACLIC 2021) — general Vietnamese VQA",
            "hf_adapter": "generic",
            # minhnguyent546/ViVQA mirrors the original ViVQA (Tran et al., 2021)
            # with clean flat columns (question/answer/img_id/image_path).
            "hf_ids": [x for x in [args.vivqa_hf_id,
                                    "minhnguyent546/ViVQA",
                                    "ThucPD/ViVQA"] if x],
            "local": {
                "train": ["vivqa/train.json", "raw/vivqa/train.json", "vivqa/ViVQA_train.json"],
                "val":   ["vivqa/val.json", "raw/vivqa/val.json", "vivqa/ViVQA_val.json"],
                "test":  ["vivqa/test.json", "raw/vivqa/test.json", "vivqa/ViVQA_test.json"],
            },
        },
        "vitextvqa": {
            "display_name": "ViTextVQA (Nguyen et al., 2024) — Vietnamese scene-text VQA",
            "hf_adapter": "nested_img_ann",
            # nhonhoccode/ViTextVQA & xuandin/ViTextVQA: ungated mirrors in the
            # COCO-style {images, annotations} container format.
            "hf_ids": [x for x in [args.vitextvqa_hf_id,
                                    "nhonhoccode/ViTextVQA",
                                    "xuandin/ViTextVQA"] if x],
            "local": {
                "train": ["vitextvqa/train.json", "raw/vitextvqa/train.json"],
                "val":   ["vitextvqa/val.json", "raw/vitextvqa/val.json"],
                "test":  ["vitextvqa/test.json", "raw/vitextvqa/test.json"],
            },
        },
        # RELATED (not a substitute): the only Vietnamese VQA data already in-repo.
        "vivqa_x": {
            "display_name": ("ViVQA-X (translated VQA-X w/ explanations, COCO images) "
                             "— RELATED local dataset, NOT ViVQA"),
            "hf_ids": [],  # local-only
            "local": {
                "train": ["ViVQA-X/data/final/ViVQA-X_train.json"],
                "val":   ["ViVQA-X/data/final/ViVQA-X_val.json"],
                "test":  ["ViVQA-X/data/final/ViVQA-X_test.json"],
            },
        },
    }


# =============================================================================
# 6) MAIN
# =============================================================================
def process_dataset(
    key: str,
    spec: Dict[str, Any],
    args: argparse.Namespace,
    rng: random.Random,
) -> Dict[str, Any]:
    display = spec["display_name"]
    load_log: List[str] = []
    splits = None
    source_kind = None
    source_detail = ""

    # 1) HuggingFace attempt (skip for local-only datasets).
    if spec.get("hf_ids"):
        splits, hf_errors = load_hf(spec["hf_ids"], SEED,
                                    adapter=spec.get("hf_adapter", "generic"))
        load_log.extend(hf_errors)
        if splits:
            source_kind = "huggingface"
            source_detail = next((e.split(": ", 1)[1] for e in hf_errors
                                  if e.startswith("OK: loaded ") or e.startswith("OK: streamed ")),
                                 "huggingface")

    # 2) Local fallback.
    if not splits:
        splits, local_errors = load_local(spec, args.data_root)
        load_log.extend(local_errors)
        if splits:
            source_kind = "local"
            source_detail = f"data_root={args.data_root}"

    # 3) PENDING_RUN if still nothing.
    if not splits:
        print(f"  [PENDING_RUN] {key}: could not load (see stats JSON errors).")
        return pending_stats(key, display, load_log)

    stats = compute_stats(key, display, source_kind, source_detail,
                          splits, args.top_k, load_log)
    written = dump_samples(args.out_dir, key, splits, args.samples_per_type, rng)
    stats["samples_written_per_type"] = written
    print(f"  [OK] {key}: {stats['num_qa']} QA, {stats['num_images']} images, "
          f"source={source_kind}; samples/type={written}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default="data",
                        help="Local fallback root for dataset files (default: data).")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help="Output directory for stats/samples.")
    parser.add_argument("--top_k", type=int, default=30, help="Top-K answers to record.")
    parser.add_argument("--samples_per_type", type=int, default=5,
                        help="Real samples to dump per question-type.")
    parser.add_argument("--only", nargs="*", default=None,
                        choices=["vivqa", "vitextvqa", "vivqa_x"],
                        help="Restrict to these dataset keys.")
    parser.add_argument("--vivqa_hf_id", default=None,
                        help="Override/prepend a HuggingFace ID for ViVQA.")
    parser.add_argument("--vitextvqa_hf_id", default=None,
                        help="Override/prepend a HuggingFace ID for ViTextVQA.")
    args = parser.parse_args()

    # Fixed seed for reproducibility (logged into each stats JSON).
    random.seed(SEED)
    rng = random.Random(SEED)

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 78)
    print(f"ViVQA-VLM Week-2 dataset analysis | seed={SEED} | version={SCRIPT_VERSION}")
    print(f"python={sys.version.split()[0]} | out_dir={args.out_dir} | data_root={args.data_root}")
    print(DISCLAIMER)
    print("=" * 78)

    registry = build_registry(args)
    keys = args.only if args.only else list(registry.keys())

    summary: Dict[str, str] = {}
    for key in keys:
        print(f"[dataset] {key}")
        spec = registry[key]
        try:
            stats = process_dataset(key, spec, args, rng)
        except Exception as exc:  # never crash the whole run on one dataset
            stats = pending_stats(key, spec["display_name"],
                                  [f"UNEXPECTED: {type(exc).__name__}: {exc}",
                                   traceback.format_exc()])
            print(f"  [ERROR] {key}: {exc}")
        out_path = os.path.join(args.out_dir, f"{key}_stats.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2)
        summary[key] = stats["status"]
        print(f"  -> wrote {out_path}")

    print("=" * 78)
    print("SUMMARY:", json.dumps(summary, ensure_ascii=False))
    print(DISCLAIMER)
    # Exit 0 always: PENDING_RUN is a valid, honest outcome (not a crash).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
