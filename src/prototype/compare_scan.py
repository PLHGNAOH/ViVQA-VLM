"""
compare_scan.py — quét so sánh mô hình nhóm (SH) với mô hình gốc trên laptop.

Luật chọn ví dụ nằm ở docs/W15/demo_compare/README.md (chốt trước khi chạy). File này chỉ áp luật đó.

OCR chạy một lần ở lượt SH đầu. Hai lượt sau dùng lại đúng các dòng OCR đó, nên ba câu SH
chỉ khác nhau ở cờ compare (rỗng / b1+b1_ocr / rỗng) — đúng phép thử "bật so sánh không đổi câu SH".

  python -m src.prototype.compare_scan --selftest
  python -m src.prototype.compare_scan
"""
from __future__ import annotations

import os

# Trùng run_local_windows.ps1 -Offline. Đặt trước mọi import Paddle / transformers.
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_CACHE"] = r"D:\hf_cache\hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import argparse
import csv
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs" / "W15" / "demo_compare"
SCAN_JSON = OUT_DIR / "scan_results.json"
SCAN_MD = OUT_DIR / "scan_table.md"
IMG_DIR = REPO / "demo_images" / "compare"
EX_JSON = Path(r"D:\vivqa_models\demo_compare\examples_compare.json")

ADAPTER = r"D:\vivqa_models\SH\epoch_1"
ADAPTER_SHA = "c9791e961aaa58fdcd7d9baafede3bd7bd8c902acc9cc43917306f1e0fecc354"
CONFIG = REPO / "configs" / "qwen_lora.yaml"

DRIVE = Path(r"G:\My Drive\ViVQA-VLM\data")
VITEXT_JSON = DRIVE / "vitextvqa_official" / "test.json"
VITEXT_ZIP = DRIVE / "vitextvqa_official" / "ViTextVQA_images.zip"
VIVQA_JSON = DRIVE / "vivqa" / "test.json"
VIVQA_TRAIN = DRIVE / "vivqa" / "train.json"
VIVQA_IMG = DRIVE / "vivqa" / "images"
REAL_DIR = Path(r"D:\vivqa_models\real_photos\w13")
REAL_CSV = REAL_DIR / "cau_hoi.csv"

VITEXT_QIDS = (
    "test_85", "test_6734", "test_9245", "test_11314", "test_10580", "test_15656",
    "test_6945", "test_10736", "test_5081", "test_9472", "test_3037", "test_11785",
)
VIVQA_QIDS = ("test_1315", "test_631", "test_1314", "test_1", "test_17", "test_2600")

SO_LIEU = ("vitextvqa_test_6734", "vitextvqa_test_9245", "vitextvqa_test_11314")
DAU_THANH = ("vitextvqa_test_85",)
GOC_EN = ("vivqa_test_1315", "vitextvqa_test_5081", "vivqa_test_631", "vitextvqa_test_9472")
GOC_BIA = ("vitextvqa_test_10580",)
VITEXT_ORDER = tuple(f"vitextvqa_{q}" for q in VITEXT_QIDS)
SLOT_ORDER = (
    "anh_that_chu", "so_lieu", "dau_thanh", "goc_tra_loi_tieng_anh",
    "goc_bia", "day_du_2x2", "co_khong",
)

REF_RUNS = {
    ("vitextvqa", "b1"): "experiments/W07_B1_zeroshot_qwen25vl_vitextvqa_test2000_bf16_L4_seed42",
    ("vivqa", "b1"): "experiments/W07_B1_zeroshot_qwen25vl_vivqa_test_bf16_L4_seed42",
    ("vitextvqa", "b1_ocr"): "experiments/W08_B1ocr_vietocr_v1_qwen25vl_vitextvqa_test2000_bf16_L4_seed42",
    ("vitextvqa", "sh"): "experiments/W11_SH_eval_qlora_ocr_vietocr_v1_r16_shared_ep1_vitextvqa_test2000_bf16_L4_seed42",
    ("vivqa", "sh"): "experiments/W11_SH_eval_qlora_ocr_vietocr_v1_r16_shared_ep1_vivqa_test_bf16_L4_seed42",
}


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def pred_map(path: Path) -> Dict[str, str]:
    """question_id → prediction (chuỗi gốc, chưa strip)."""
    out: Dict[str, str] = {}
    for row in _load_json(path):
        qid = str(row.get("question_id") or "")
        if qid:
            out[qid] = str(row.get("prediction") or "")
    return out


def ref_equal(pred: Optional[str], ref: Optional[str]) -> Optional[bool]:
    """Khớp tham chiếu: so sau strip. Không có tham chiếu → None. Không lower-case."""
    if ref is None or pred is None:
        return None
    return pred.strip() == ref.strip()


def _index_questions(path: Path, want: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    need = set(want)
    found: Dict[str, Dict[str, Any]] = {}
    for row in _load_json(path):
        qid = str(row.get("question_id") or "")
        if qid in need:
            found[qid] = row
    return found


def _train_keys(path: Path) -> Tuple[set, set]:
    names, ids = set(), set()
    for row in _load_json(path):
        if row.get("image"):
            names.add(str(row["image"]))
        if row.get("img_id") is not None:
            ids.add(str(row["img_id"]))
    return names, ids


def _copy_member(zip_path: Path, member: str, dest: Path) -> bool:
    with zipfile.ZipFile(zip_path) as zf:
        try:
            src = zf.open(member)
        except KeyError:
            return False
        with src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)
    return True


def _split_answers(cell: str) -> List[str]:
    return [a.strip() for a in str(cell or "").split("|") if a.strip()]


def build_manifest(img_dir: Path) -> List[Dict[str, Any]]:
    """Danh sách VIỆC 3: ảnh thật trước, rồi ViTextVQA, rồi ViVQA. Thiếu ảnh thì ghi missing, không dừng."""
    img_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    if REAL_CSV.is_file():
        with REAL_CSV.open(encoding="utf-8-sig", newline="") as f:
            for i, row in enumerate(csv.DictReader(f), 1):
                name = (row.get("anh") or "").strip()
                q = (row.get("cau_hoi") or "").strip()
                answers = _split_answers(row.get("dap_an") or "")
                src = REAL_DIR / name
                qid = Path(name).stem or f"row{i:03d}"
                item_id = f"real_{qid}"
                dest = img_dir / f"{item_id}{src.suffix or '.jpg'}"
                missing = not (name and q and src.is_file())
                if not missing:
                    shutil.copyfile(src, dest)
                items.append({
                    "id": item_id, "source": "real", "question_id": None, "file": name,
                    "question": q, "answers": answers, "image": str(dest) if not missing else "",
                    "missing": missing, "trung_train": None,
                })
    else:
        items.append({
            "id": "real_csv", "source": "real", "question_id": None, "file": "",
            "question": "", "answers": [], "image": "", "missing": True, "trung_train": None,
            "missing_note": f"không thấy {REAL_CSV}",
        })

    vitext = _index_questions(VITEXT_JSON, VITEXT_QIDS) if VITEXT_JSON.is_file() else {}
    extracted: Dict[str, Path] = {}
    if vitext and VITEXT_ZIP.is_file():
        with zipfile.ZipFile(VITEXT_ZIP) as zf:
            names = set(zf.namelist())
            for qid, row in vitext.items():
                member = f"st_images/{row.get('image')}"
                dest = img_dir / f"vitextvqa_{qid}.jpg"
                if member not in names:
                    continue
                with zf.open(member) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
                extracted[qid] = dest
    for qid in VITEXT_QIDS:
        row = vitext.get(qid)
        item_id = f"vitextvqa_{qid}"
        dest = extracted.get(qid)
        missing = dest is None
        items.append({
            "id": item_id, "source": "vitextvqa", "question_id": qid,
            "file": (row or {}).get("image") or "",
            "question": str((row or {}).get("question") or ""),
            "answers": list((row or {}).get("answers") or []),
            "image": str(dest) if not missing else "",
            "missing": missing, "trung_train": None,
        })

    vivqa = _index_questions(VIVQA_JSON, VIVQA_QIDS) if VIVQA_JSON.is_file() else {}
    train_names, train_ids = _train_keys(VIVQA_TRAIN) if VIVQA_TRAIN.is_file() else (set(), set())
    for qid in VIVQA_QIDS:
        row = vivqa.get(qid)
        item_id = f"vivqa_{qid}"
        image_name = str((row or {}).get("image") or "")
        ext = Path(image_name).suffix or ".jpg"
        dest = img_dir / f"{item_id}{ext}"
        src = VIVQA_IMG / image_name if image_name else None
        missing = row is None or src is None or not src.is_file()
        if not missing and src is not None:
            shutil.copyfile(src, dest)
        img_id = (row or {}).get("img_id")
        trung = None
        if row is not None:
            trung = image_name in train_names or str(img_id) in train_ids
        items.append({
            "id": item_id, "source": "vivqa", "question_id": qid, "file": image_name,
            "question": str((row or {}).get("question") or ""),
            "answers": list((row or {}).get("answers") or []),
            "image": str(dest) if not missing else "",
            "missing": missing, "trung_train": trung,
        })
    return items


def _cmp_item(rows: Sequence[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        if row.get("key") == key:
            return row
    return None


def _em(pred: Optional[str], answers: Sequence[str]) -> Optional[float]:
    if pred is None or not answers:
        return None
    from src.eval.metrics import exact_match
    return float(exact_match(pred, list(answers)))


def measure_item(backend, item: Dict[str, Any], tmp_dir: str, refs: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Ba lượt ask. OCR lấy ở lượt đầu và dùng lại, để chỉ cờ compare đổi giữa các lượt."""
    from src.prototype.app import prepare_image
    from src.prototype.core import is_yes_no_question

    base = {
        "id": item["id"], "source": item["source"], "question_id": item.get("question_id"),
        "file": item.get("file") or "", "question": item.get("question") or "",
        "gold": list(item.get("answers") or []), "trung_train": item.get("trung_train"),
        "image": item.get("image") or "",
        "missing": False, "yesno": is_yes_no_question(item.get("question") or ""),
    }
    if item.get("missing") or not item.get("image"):
        base.update({"missing": True, "error_sh": None, "sh_stable": False, "n_ocr_lines": 0})
        return base

    used = None
    try:
        used, _info = prepare_image(item["image"], tmp_dir)
        r_a = backend.ask(used, item["question"], compare=())
        lines = list(r_a.get("ocr_lines") or [])
        timing = {
            "sec_det": float(r_a.get("sec_ocr_det") or 0.0),
            "sec_rec": float(r_a.get("sec_ocr_rec") or 0.0),
            "sec_ocr": float(r_a.get("sec_ocr") or 0.0),
            "ocr_source": "cache",
        }
        r_b = backend.ask(used, item["question"], ocr_lines=lines, ocr_timing=timing,
                          compare=("b1", "b1_ocr"))
        r_c = backend.ask(used, item["question"], ocr_lines=lines, ocr_timing=timing, compare=())
    except Exception as e:
        base.update({"error_sh": f"{type(e).__name__}: {e}"[:300], "sh_stable": False, "n_ocr_lines": 0})
        return base
    finally:
        if used and used != item["image"] and os.path.exists(used):
            try:
                os.remove(used)
            except OSError:
                pass

    sh = [str(r_a.get("answer") or ""), str(r_b.get("answer") or ""), str(r_c.get("answer") or "")]
    b1 = _cmp_item(r_b.get("compare") or [], "b1") or {}
    b1o = _cmp_item(r_b.get("compare") or [], "b1_ocr") or {}
    pred_b1 = None if b1.get("error") else (None if "answer" not in b1 else str(b1.get("answer") or ""))
    pred_b1o = None if b1o.get("error") else (None if "answer" not in b1o else str(b1o.get("answer") or ""))
    sec_b1 = None if b1.get("error") or "sec_model" not in b1 else float(b1["sec_model"])
    sec_b1o = None if b1o.get("error") or "sec_model" not in b1o else float(b1o["sec_model"])
    sec_sh = float(r_a.get("sec_model") or r_a.get("sec") or 0.0)
    sec_ocr = float(r_a.get("sec_ocr") or 0.0)
    sec_tong = float(r_a.get("sec_total") or 0.0) + (sec_b1 or 0.0) + (sec_b1o or 0.0)
    if sec_b1 is None:
        sec_tong = None
    qid = item.get("question_id")
    src = item["source"]

    def _ref(kind: str, pred: Optional[str]) -> Optional[bool]:
        table = refs.get(f"{src}:{kind}")
        if not qid or table is None or qid not in table:
            return None
        return ref_equal(pred, table[qid])

    base.update({
        "error_sh": None,
        "sh": sh[0], "sh_answers": sh, "sh_stable": sh[0] == sh[1] == sh[2],
        "b1": pred_b1, "b1_ocr": pred_b1o,
        "b1_error": b1.get("error"), "b1_ocr_error": b1o.get("error"),
        "level": r_a.get("level"), "mode": r_a.get("mode"),
        "n_ocr_lines": int(r_a.get("n_ocr_lines") or len(lines)),
        "em_sh": _em(sh[0], base["gold"]), "em_b1": _em(pred_b1, base["gold"]),
        "em_b1_ocr": _em(pred_b1o, base["gold"]),
        "match_b1": _ref("b1", pred_b1), "match_b1_ocr": _ref("b1_ocr", pred_b1o),
        "match_sh": _ref("sh", sh[0]),
        "sec_ocr": sec_ocr, "sec_sh": sec_sh, "sec_b1": sec_b1, "sec_b1_ocr": sec_b1o,
        "sec_tong": sec_tong,
    })
    return base


def is_showcase(rec: Dict[str, Any]) -> bool:
    """Ứng viên chào hàng — đúng docs/W15/demo_compare/README.md, không thêm điều kiện."""
    if rec.get("missing") or rec.get("error_sh") or not rec.get("sh_stable"):
        return False
    if rec.get("em_sh") != 1.0 or rec.get("em_b1") != 0.0:
        return False
    if rec.get("source") == "vitextvqa" and rec.get("em_b1_ocr") != 0.0:
        return False
    if rec.get("level") == "Thấp" or not rec.get("level"):
        return False
    sec = rec.get("sec_tong")
    if not isinstance(sec, (int, float)) or sec > 8.0:
        return False
    if rec.get("source") == "vivqa" and rec.get("trung_train"):
        return False
    return True


def _first(records: Dict[str, Dict[str, Any]], ids: Sequence[str], pred, used: set) -> Optional[Dict[str, Any]]:
    for i in ids:
        rec = records.get(i)
        if rec is None or i in used:
            continue
        if pred(rec):
            return rec
    return None


def _why_empty(cands: List[Dict[str, Any]]) -> str:
    if cands and all(c.get("missing") for c in cands):
        return "thiếu ảnh"
    return "không có item đạt luật"


def select_slots(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bảy ô, item đầu tiên đạt luật, bỏ item đã chọn ở ô trước."""
    by_id = {r["id"]: r for r in rows}
    real_ids = [r["id"] for r in rows if r.get("source") == "real"]
    used: set = set()
    out = []

    def take(slot: str, ids: Sequence[str], pred) -> None:
        cands = [by_id[i] for i in ids if i in by_id]
        hit = _first(by_id, ids, pred, used)
        if hit:
            used.add(hit["id"])
            out.append({"slot": slot, "id": hit["id"], "reason": ""})
        else:
            out.append({"slot": slot, "id": None, "reason": _why_empty(cands)})

    take("anh_that_chu", real_ids, lambda r: is_showcase(r) and r.get("n_ocr_lines", 0) >= 1)
    take("so_lieu", SO_LIEU, is_showcase)
    take("dau_thanh", DAU_THANH, is_showcase)
    take("goc_tra_loi_tieng_anh", GOC_EN, is_showcase)
    take("goc_bia", GOC_BIA, is_showcase)
    take("day_du_2x2", VITEXT_ORDER, lambda r: is_showcase(r) and r.get("source") == "vitextvqa")
    take("co_khong", real_ids, lambda r: is_showcase(r) and r.get("yesno"))
    return out


def _fmt_sec(v: Any) -> str:
    return f"{float(v):.2f}".replace(".", ",") if isinstance(v, (int, float)) else "—"


def _fmt_em(v: Any) -> str:
    if v == 1.0:
        return "1"
    if v == 0.0:
        return "0"
    return "—"


def _fmt_match(v: Any) -> str:
    if v is True:
        return "khớp"
    if v is False:
        return "lệch"
    return "—"


def _cell(text: Any) -> str:
    s = str(text if text is not None else "")
    return s.replace("|", "/").replace("\n", " ").strip()


def _ans_cell(rec: Dict[str, Any], key: str, err_key: str) -> str:
    if rec.get("missing"):
        return "thiếu"
    if rec.get(err_key):
        return "⚠ lỗi: " + _cell(str(rec[err_key])[:80])
    if rec.get(key) is None:
        return "—"
    return _cell(rec.get(key))


def render_table(rows: Sequence[Dict[str, Any]]) -> str:
    head = ("id", "nguồn", "câu hỏi", "gold", "gốc", "gốc+OCR", "nhóm", "mức",
            "EM gốc", "EM gốc+OCR", "EM nhóm", "sh_stable", "khớp B1", "khớp B1-OCR",
            "khớp SH", "trùng_train", "tổng giây")
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for r in rows:
        if r.get("missing"):
            sh_s, level, gold = "—", "—", "—"
        else:
            sh_s = "có" if r.get("sh_stable") else "không"
            level = _cell(r.get("level") or "—")
            gold = _cell(" / ".join(r.get("gold") or []))
        trung = r.get("trung_train")
        trung_s = "—" if trung is None else ("có" if trung else "không")
        cells = [
            _cell(r.get("id")), _cell(r.get("source")), _cell(r.get("question")), gold,
            _ans_cell(r, "b1", "b1_error"), _ans_cell(r, "b1_ocr", "b1_ocr_error"),
            "thiếu" if r.get("missing") else _cell(r.get("sh")),
            level, _fmt_em(r.get("em_b1")), _fmt_em(r.get("em_b1_ocr")), _fmt_em(r.get("em_sh")),
            sh_s, _fmt_match(r.get("match_b1")), _fmt_match(r.get("match_b1_ocr")),
            _fmt_match(r.get("match_sh")), trung_s, _fmt_sec(r.get("sec_tong")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def summarize(rows: Sequence[Dict[str, Any]], slots: Sequence[Dict[str, Any]], vram_gb: Optional[float]) -> Dict[str, Any]:
    ran = [r for r in rows if not r.get("missing")]
    stable = [r for r in ran if r.get("sh_stable")]
    sh_err = [r for r in ran if r.get("error_sh")]

    def _count(key: str) -> Dict[str, int]:
        judged = [r for r in ran if r.get(key) is not None]
        return {"khop": sum(1 for r in judged if r[key] is True), "tong": len(judged)}

    def _secs(key: str) -> List[float]:
        return [float(r[key]) for r in ran if isinstance(r.get(key), (int, float))]

    gate = bool(ran) and not sh_err and len(stable) == len(ran)
    return {
        "gate_pass": gate,
        "n_items": len(rows),
        "n_ran": len(ran),
        "n_thieu": sum(1 for r in rows if r.get("missing")),
        "n_sh_stable": len(stable),
        "n_sh_error": len(sh_err),
        "sh_stable_rate": (len(stable) / len(ran) if ran else 0.0),
        "sh_errors": [{"id": r["id"], "error": r.get("error_sh")} for r in sh_err],
        "unstable": [r["id"] for r in ran if not r.get("sh_stable") and not r.get("error_sh")],
        "ref_b1": _count("match_b1"),
        "ref_b1_ocr": _count("match_b1_ocr"),
        "ref_sh": _count("match_sh"),
        "sec_mean": {
            "ocr": _mean(_secs("sec_ocr")),
            "sh": _mean(_secs("sec_sh")),
            "b1": _mean(_secs("sec_b1")),
            "b1_ocr": _mean(_secs("sec_b1_ocr")),
            "tong": _mean([float(r["sec_tong"]) for r in ran if isinstance(r.get("sec_tong"), (int, float))]),
        },
        "vram_peak_allocated_gb": vram_gb,
        "slots": list(slots),
    }


def _public_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Bỏ đường dẫn ảnh. Giữ câu hỏi và câu trả lời (chữ, không phải file ảnh)."""
    skip = {"image"}
    return {k: v for k, v in rec.items() if k not in skip}


def _example_image(rec: Dict[str, Any]) -> str:
    raw = str(rec.get("image") or "")
    if raw and Path(raw).is_file():
        return raw
    hits = sorted(IMG_DIR.glob(str(rec.get("id") or "") + ".*"))
    return str(hits[0]) if hits else ""


def write_outputs(rows: Sequence[Dict[str, Any]], slots: Sequence[Dict[str, Any]],
                  summary: Dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "items": [_public_row(r) for r in rows]}
    SCAN_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SCAN_MD.write_text(render_table(rows), encoding="utf-8")
    by_id = {r["id"]: r for r in rows}
    examples = []
    for slot in slots:
        rec = by_id.get(slot.get("id") or "")
        image = _example_image(rec) if rec else ""
        if rec and image and rec.get("question"):
            examples.append({"image": image, "question": rec["question"]})
    EX_JSON.parent.mkdir(parents=True, exist_ok=True)
    EX_JSON.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")


def _fake_ok(item_id: str, source: str, **kw) -> Dict[str, Any]:
    rec = {
        "id": item_id, "source": source, "question": kw.get("question", "câu hỏi mẫu"),
        "gold": ["đáp"], "missing": False, "error_sh": None, "sh_stable": True,
        "em_sh": 1.0, "em_b1": 0.0, "em_b1_ocr": 0.0, "level": "Cao",
        "sec_tong": 5.0, "n_ocr_lines": 2, "yesno": False, "trung_train": False,
        "sh": "đáp", "b1": "sai", "b1_ocr": "sai",
    }
    rec.update(kw)
    return rec


def _selftest() -> None:
    assert ref_equal("  có ", "có") is True
    assert ref_equal("Có", "có") is False
    assert ref_equal("có", None) is None
    vivqa_b1 = pred_map(REPO / REF_RUNS[("vivqa", "b1")] / "predictions.json")
    assert vivqa_b1["test_0"].strip() == "Xanh nhạt"

    # Luật: đủ điều kiện thì đạt; từng điều kiện gãy thì rớt. Không nhìn kết quả GPU.
    good = _fake_ok("vitextvqa_test_15656", "vitextvqa")
    assert is_showcase(good)
    assert not is_showcase({**good, "sh_stable": False})
    assert not is_showcase({**good, "em_sh": 0.0})
    assert not is_showcase({**good, "em_b1": 1.0})
    assert not is_showcase({**good, "em_b1_ocr": 1.0})
    assert not is_showcase({**good, "em_b1": None})
    assert is_showcase({**good, "source": "vivqa", "em_b1_ocr": 1.0, "trung_train": False})
    assert not is_showcase({**good, "source": "vivqa", "trung_train": True})
    assert not is_showcase({**good, "level": "Thấp"})
    assert not is_showcase({**good, "sec_tong": 8.01})
    assert is_showcase({**good, "sec_tong": 8.0})
    assert not is_showcase({**good, "error_sh": "boom"})

    rows = [
        _fake_ok("real_a", "real", n_ocr_lines=3, yesno=False, question="biển ghi gì"),
        _fake_ok("real_b", "real", n_ocr_lines=1, yesno=True, question="có mèo không", sec_tong=9.0),
        _fake_ok("real_c", "real", n_ocr_lines=0, yesno=True, question="có xe không"),
        _fake_ok("real_d", "real", n_ocr_lines=2, yesno=True, question="có chữ không"),
        _fake_ok("vitextvqa_test_6734", "vitextvqa", sec_tong=9.0),
        _fake_ok("vitextvqa_test_9245", "vitextvqa"),
        _fake_ok("vitextvqa_test_11314", "vitextvqa", sec_tong=9.0),
        _fake_ok("vitextvqa_test_85", "vitextvqa"),
        _fake_ok("vivqa_test_1315", "vivqa", em_b1_ocr=1.0, trung_train=False),
        _fake_ok("vitextvqa_test_5081", "vitextvqa", level="Thấp"),
        _fake_ok("vivqa_test_631", "vivqa", trung_train=True, em_b1_ocr=1.0),
        _fake_ok("vitextvqa_test_9472", "vitextvqa"),
        _fake_ok("vitextvqa_test_10580", "vitextvqa", em_sh=0.0),
        _fake_ok("vitextvqa_test_15656", "vitextvqa"),
        _fake_ok("vitextvqa_test_6945", "vitextvqa"),
    ]
    slots = select_slots(rows)
    chosen = {s["slot"]: s["id"] for s in slots}
    assert chosen["anh_that_chu"] == "real_a"
    assert chosen["so_lieu"] == "vitextvqa_test_9245"          # 6734 quá 8 s
    assert chosen["dau_thanh"] == "vitextvqa_test_85"
    assert chosen["goc_tra_loi_tieng_anh"] == "vivqa_test_1315"
    assert chosen["goc_bia"] is None and slots[4]["reason"] == "không có item đạt luật"
    assert chosen["day_du_2x2"] == "vitextvqa_test_15656"      # 85, 9245 đã dùng; 6734/11314 quá 8 s; 10580 SH sai
    assert chosen["co_khong"] == "real_c"                      # câu có/không; real_b quá 8 s; không cần có chữ
    assert len({s["id"] for s in slots if s["id"]}) == 6

    # Backend giả: ba lượt, OCR của lượt sau là dòng của lượt đầu, SH không đổi.
    from PIL import Image
    from src.prototype.app import MockBackend, load_ocr_selection, load_ui_settings

    class _Rec:
        def __init__(self):
            self.calls = []

        def ask(self, image, question, ocr_lines=None, ocr_timing=None, compare=()):
            self.calls.append({"compare": tuple(compare or ()), "n_ocr": None if ocr_lines is None else len(ocr_lines)})
            out = {
                "mode": "open", "answer": "nhóm", "level": "Cao", "confidence": 0.9,
                "sec_total": 1.2, "sec_ocr": 0.7, "sec_model": 0.4, "sec": 0.4,
                "ocr_lines": [{"text": "DÒNG", "box": [0, 0, 1, 1]}], "n_ocr_lines": 1,
            }
            if compare:
                out["compare"] = [
                    {"key": k, "answer": "gốc " + k, "sec_model": 0.3} for k in compare
                ]
            return out

    tmp = tempfile.mkdtemp(prefix="cmp_scan_")
    try:
        img = os.path.join(tmp, "a.jpg")
        Image.new("RGB", (32, 32), (10, 20, 30)).save(img, quality=90)
        backend = _Rec()
        item = {"id": "real_a", "source": "real", "question_id": None, "file": "a.jpg",
                "question": "biển ghi gì", "answers": ["nhóm"], "image": img, "missing": False,
                "trung_train": None}
        rec = measure_item(backend, item, tmp, {})
        assert [c["compare"] for c in backend.calls] == [(), ("b1", "b1_ocr"), ()]
        assert backend.calls[0]["n_ocr"] is None and backend.calls[1]["n_ocr"] == 1
        assert rec["sh_stable"] and rec["sh_answers"] == ["nhóm", "nhóm", "nhóm"]
        assert rec["em_sh"] == 1.0 and rec["em_b1"] == 0.0 and rec["error_sh"] is None
        assert "image" not in _public_row(rec)
        mock = MockBackend(load_ui_settings(), load_ocr_selection())
        mrec = measure_item(mock, {**item, "question": "tên cửa hàng là gì", "answers": ["cửa hàng mẫu"]}, tmp, {})
        assert mrec["sh_stable"] and mrec["sh"] == "cửa hàng mẫu"
        assert mrec["b1"] and "dữ liệu mẫu" in mrec["b1"]
        md = render_table([rec, {**item, "missing": True, "question": "không có file", "gold": []}])
        assert "sh_stable" in md and "tổng giây" in md and img not in md
        miss = measure_item(backend, {**item, "missing": True, "image": ""}, tmp, {})
        assert miss["missing"] and miss["sh_stable"] is False
        summary = summarize([rec, miss], select_slots([rec, miss]), 1.5)
        assert summary["n_ran"] == 1 and summary["n_thieu"] == 1 and summary["gate_pass"] is True
        bad = summarize([{**rec, "sh_stable": False, "error_sh": None}], [], None)
        assert bad["gate_pass"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("compare_scan selftest OK")


def _vram_gb() -> Optional[float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        return None


def _load_refs() -> Dict[str, Dict[str, str]]:
    out = {}
    for (src, kind), rel in REF_RUNS.items():
        path = REPO / rel / "predictions.json"
        out[f"{src}:{kind}"] = pred_map(path)
    return out


def run_gpu() -> int:
    items = build_manifest(IMG_DIR)
    n_miss = sum(1 for it in items if it.get("missing"))
    print(f"manifest {len(items)} item, thiếu {n_miss}", flush=True)
    for it in items:
        if it.get("missing"):
            print(f"  thiếu {it['id']} {it.get('file') or it.get('missing_note') or ''}", flush=True)
    runnable = [it for it in items if not it.get("missing")]
    if not runnable:
        print("Không có ảnh nào để chạy", flush=True)
        slots = select_slots([{"id": it["id"], "source": it["source"], "missing": True,
                                "question": it.get("question"), "gold": it.get("answers") or []} for it in items])
        rows = [{"id": it["id"], "source": it["source"], "question": it.get("question"),
                 "gold": it.get("answers") or [], "missing": True, "sh_stable": False} for it in items]
        summary = summarize(rows, slots, None)
        write_outputs(rows, slots, summary)
        return 2

    from src.prototype.app import load_ocr_selection, load_ui_settings, make_real_backend

    settings = load_ui_settings()
    ocr_cfg = load_ocr_selection()
    backend, client, info = make_real_backend(
        ADAPTER, str(CONFIG), settings, ocr_cfg,
        ocr_device="cpu", expect_adapter_sha=ADAPTER_SHA,
    )
    print("model ready " + json.dumps({k: info.get(k) for k in ("sec_start_total", "sec_model_load")},
                                      ensure_ascii=False), flush=True)
    refs = _load_refs()
    from src.prototype.app import prepare_image
    tmp = tempfile.mkdtemp(prefix="cmp_scan_run_")
    rows: List[Dict[str, Any]] = []
    vram = None
    try:
        warm = runnable[0]
        try:
            used, _ = prepare_image(warm["image"], tmp)
            try:
                backend.ask(used, warm["question"], compare=())
            finally:
                if used != warm["image"] and os.path.exists(used):
                    os.remove(used)
            print(f"warmup xong ({warm['id']}), không tính vào bảng", flush=True)
        except Exception as e:
            print(f"warmup lỗi (bỏ qua, vẫn đo chính thức): {type(e).__name__}: {e}", flush=True)
        for i, it in enumerate(items, 1):
            rec = measure_item(backend, it, tmp, refs)
            rows.append(rec)
            flag = "thiếu" if rec.get("missing") else ("LỖI SH" if rec.get("error_sh") else
                                                       ("ổn" if rec.get("sh_stable") else "lệch SH"))
            print(f"[{i}/{len(items)}] {it['id']} {flag} sh={rec.get('sh')!r} tong={rec.get('sec_tong')}",
                  flush=True)
        vram = _vram_gb()
    finally:
        if vram is None:
            vram = _vram_gb()
        if client is not None:
            client.close()
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            import gc
            import torch
            del backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    slots = select_slots(rows)
    summary = summarize(rows, slots, vram)
    write_outputs(rows, slots, summary)
    print(json.dumps({k: summary[k] for k in (
        "gate_pass", "sh_stable_rate", "n_ran", "n_thieu", "n_sh_error", "ref_b1", "ref_b1_ocr", "ref_sh",
        "sec_mean", "vram_peak_allocated_gb")}, ensure_ascii=False), flush=True)
    for s in slots:
        print(f"  ô {s['slot']}: {s['id'] or 'TRỐNG — ' + s['reason']}", flush=True)
    return 0 if summary["gate_pass"] else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.selftest:
        _selftest()
        return 0
    return run_gpu()


if __name__ == "__main__":
    raise SystemExit(main())
