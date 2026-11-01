"""
exif_audit.py — Kiểm cờ xoay EXIF có làm hỏng OCR cache ViTextVQA không (W14, bước 1, luật R-EXIF).

VẤN ĐỀ (W13, bước 8): ở 1849.jpg và 14277.jpg (4000×3000), cache OCR chứa chữ không liên quan, còn OCR trên
ảnh đã xoay/thu nhỏ thì đọc đúng biển hiệu. Giả thuyết cơ chế:
    - `PaddleDetVietOCREngine.read` đưa ĐƯỜNG DẪN ảnh cho Paddle dò hộp chữ. Paddle đọc ảnh bằng OpenCV,
      mà `cv2.imread` mặc định XOAY ảnh theo cờ EXIF Orientation.
    - VietOCR lại cắt hộp từ ảnh PIL (`Image.open`), mà PIL KHÔNG xoay theo EXIF.
    -> ảnh có Orientation ≠ 1: hộp ở toạ độ ảnh đã xoay, cắt trên ảnh chưa xoay -> cắt nhầm vùng -> chữ rác.

KIỂM 3 BƯỚC (luật ghi trước ở docs/W14/README.md, mục R-EXIF):
    --scan     chỉ đọc header ảnh (CPU): đếm ảnh Orientation ≠ 1 và số câu hỏi bị ảnh hưởng theo split,
               cho ViTextVQA (train / dev / test full) và ViVQA (train / test).
    --reocr    (cần PaddleOCR + VietOCR, GPU T4) trên mẫu cố định: ≤ 300 ảnh EXIF ≠ 1 + 200 ảnh đối chứng
               EXIF = 1 (seed 42, luôn gồm 1849.jpg / 14277.jpg). Mỗi ảnh OCR 3 kiểu:
                 asis  = đúng đường dẫn như lúc dựng cache (tái hiện cách cũ);
                 fixed = ảnh đã xoay theo EXIF, lưu PNG không còn cờ (OpenCV và PIL nhìn cùng một ảnh);
                 small = như fixed rồi thu cạnh dài về 2.048 px (đúng đường ảnh của giao diện W13) — chỉ ảnh > 2.048 px.
               So từng kiểu với cache bằng độ giống chuỗi; tính "đáp án gold nằm trong OCR" trên câu test.
    --analyze  áp luật L1–L3, ghi exif_audit_summary.json.

KHÔNG đổi cache, không đổi code OCR, không đổi số đã báo cáo. Sửa (nếu luật bảo sửa) là bước 2, commit riêng.

Chạy (Colab):
    python -m src.data.exif_audit --scan --vitextvqa_images <dir> --vivqa_images <dir> --out <dir>
    python -m src.data.exif_audit --reocr --scan_json <dir>/exif_scan.json --vitextvqa_images <dir> --device gpu --out <dir>
    python -m src.data.exif_audit --analyze --scan_json <dir>/exif_scan.json --reocr_json <dir>/exif_reocr.json --out <dir>
Kiểm thử offline:
    python -m src.data.exif_audit --selftest
"""
from __future__ import annotations

import argparse
import datetime
import difflib
import hashlib
import json
import os
import random
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

ORIENTATION_TAG = 274                  # EXIF Orientation
SWAP_WH = {5, 6, 7, 8}                 # các hướng làm đổi chỗ rộng/cao
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/ViVQA-VLM"
SPLITS = {"vitextvqa_official": {"train": "train.json", "dev": "dev.json", "test": "test.json"},
          "vivqa": {"train": "train.json", "test": "test.json"}}
EXPECTED_Q = {("vitextvqa_official", "train"): 35159, ("vitextvqa_official", "test"): 10028,
              ("vivqa", "train"): 10799, ("vivqa", "test"): 3001}
MUST_INCLUDE = ("1849.jpg", "14277.jpg")   # 2 ảnh nghi vấn ở W13
UI_MAX_SIDE = 2048                     # = src/prototype/app.py MAX_SIDE (W13, quyết định D3)

# ---- luật R-EXIF (ghi trước, docs/W14/README.md). Đổi số ở đây = đổi luật -> phải commit README trước ----
RULE = {
    "sim_mismatch": 0.5,       # độ giống cache vs OCR ảnh đã xoay < 0.5 -> "lệch"
    "l1_exif_min": 0.80,       # L1: >= 80% ảnh EXIF≠1 trong mẫu bị lệch ...
    "l1_control_max": 0.05,    # ... và < 5% ảnh đối chứng EXIF=1 bị lệch -> cơ chế XÁC NHẬN
    "l2_test_frac": 0.01,      # L2: > 1% câu full test ViTextVQA có ảnh EXIF≠1 -> sửa + suy luận lại B3-full
    "l3_train_frac": 0.05,     # L3: > 5% câu train ViTextVQA có ảnh EXIF≠1 -> train lại B3-full với cache đã sửa
    "n_affected": 300, "n_control": 200, "seed": 42,
}


# =============================================================================
# 1) HÀM THUẦN
# =============================================================================
def exif_orientation(path: str) -> int:
    """Giá trị EXIF Orientation (1 = bình thường; không có cờ / không đọc được EXIF -> 1)."""
    from PIL import Image
    with Image.open(path) as im:
        try:
            o = im.getexif().get(ORIENTATION_TAG, 1)
        except Exception:
            return 1
        try:
            o = int(o)
        except (TypeError, ValueError):
            return 1
        return o if 1 <= o <= 8 else 1


def scan_images(images: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """{tên: đường_dẫn} -> {tên: {"o": orientation, "w", "h" (kích thước PIL, CHƯA xoay)}}. Chỉ đọc header."""
    from PIL import Image
    out: Dict[str, Dict[str, Any]] = {}
    for name, p in sorted(images.items()):
        with Image.open(p) as im:
            w, h = im.size
        out[name] = {"o": exif_orientation(p), "w": w, "h": h}
    return out


def split_stats(records: List[Dict[str, Any]], scan: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Đếm câu / ảnh và phần có EXIF ≠ 1 của một split."""
    imgs = sorted({str(r["image"]) for r in records})
    missing = [k for k in imgs if k not in scan]
    if missing:
        raise KeyError(f"{len(missing)} ảnh chưa được quét, vd {missing[:3]}")
    bad_imgs = [k for k in imgs if scan[k]["o"] != 1]
    bad_set = set(bad_imgs)
    n_q_bad = sum(1 for r in records if str(r["image"]) in bad_set)
    by_o: Dict[str, int] = {}
    for k in bad_imgs:
        by_o[str(scan[k]["o"])] = by_o.get(str(scan[k]["o"]), 0) + 1
    big = sum(1 for k in imgs if max(scan[k]["w"], scan[k]["h"]) > UI_MAX_SIDE)
    return {"n_questions": len(records), "n_images": len(imgs),
            "n_images_exif": len(bad_imgs), "n_questions_exif": n_q_bad,
            "pct_images_exif": round(100.0 * len(bad_imgs) / max(len(imgs), 1), 3),
            "pct_questions_exif": round(100.0 * n_q_bad / max(len(records), 1), 3),
            "exif_by_orientation": dict(sorted(by_o.items())),
            "n_images_gt_ui_max_side": big}


def norm_text(entry: Optional[Dict[str, Any]]) -> str:
    """Các dòng OCR của 1 ảnh -> 1 chuỗi chuẩn hoá (NFC, thường, bỏ dấu câu), giữ thứ tự đọc."""
    from src.eval.ocr_coverage import norm
    if not entry:
        return ""
    return norm(" ".join(ln.get("text", "") for ln in entry.get("lines", [])))


def text_sim(a: str, b: str) -> float:
    """Độ giống 2 chuỗi OCR (difflib ratio, 0..1). Cả hai rỗng = 1; một bên rỗng = 0."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 4)


def answer_in(text: str, answers: List[str]) -> bool:
    from src.eval.ocr_coverage import contains_phrase, norm
    return any(contains_phrase(text, norm(a)) for a in answers if str(a).strip())


def choose_sample(scan: Dict[str, Dict[str, Any]], test_imgs: List[str], train_imgs: List[str],
                  n_affected: int = RULE["n_affected"], n_control: int = RULE["n_control"],
                  seed: int = RULE["seed"], must: Tuple[str, ...] = MUST_INCLUDE) -> Dict[str, List[str]]:
    """
    Mẫu cố định (chỉ phụ thuộc kết quả quét + seed):
      affected: ảnh EXIF≠1, ưu tiên ảnh test (đo được đáp án∈OCR), thêm ảnh train nếu thiếu; tối đa n_affected.
      control : n_control ảnh test EXIF=1 ngẫu nhiên.
      must    : 1849.jpg / 14277.jpg luôn có mặt (vào nhóm theo cờ EXIF thật của chúng).
    """
    rng = random.Random(seed)
    test_set = sorted(set(test_imgs))
    train_only = sorted(set(train_imgs) - set(test_set))
    aff_test = [k for k in test_set if scan[k]["o"] != 1]
    aff_train = [k for k in train_only if scan[k]["o"] != 1]
    rng.shuffle(aff_test)
    rng.shuffle(aff_train)
    affected = (aff_test + aff_train)[:n_affected]
    ctrl_pool = [k for k in test_set if scan[k]["o"] == 1]
    rng.shuffle(ctrl_pool)
    control = ctrl_pool[:n_control]
    for m in must:
        if m in scan:
            grp = affected if scan[m]["o"] != 1 else control
            if m not in grp:
                grp.append(m)
    return {"affected": sorted(affected), "control": sorted(control)}


def fixed_copy(path: str, tmpdir: str, max_side: Optional[int] = None) -> Tuple[str, Tuple[int, int]]:
    """Ảnh đã xoay theo EXIF (và thu nhỏ nếu max_side), lưu PNG KHÔNG còn cờ EXIF. Trả (đường_dẫn, kích thước)."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im2 = ImageOps.exif_transpose(im).convert("RGB")
    if max_side and max(im2.size) > max_side:
        im2.thumbnail((max_side, max_side), Image.LANCZOS)      # như src/prototype/app.py (W13)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(tmpdir, f"{stem}__{'small' if max_side else 'fixed'}.png")
    im2.save(out)
    return out, im2.size


def reocr_one(engine: Any, name: str, path: str, cache_entry: Dict[str, Any], tmpdir: str,
              meta: Dict[str, Any]) -> Dict[str, Any]:
    """OCR 1 ảnh theo 3 kiểu, so với cache. Trả bản ghi (không chứa văn bản dài, chỉ 160 ký tự đầu để xem tận mắt)."""
    rec: Dict[str, Any] = {"image": name, "o": meta["o"], "w": meta["w"], "h": meta["h"]}
    c_txt = norm_text(cache_entry)
    rec["cache_text"] = c_txt[:160]
    variants = [("asis", path)]
    p_fixed, sz = fixed_copy(path, tmpdir)
    variants.append(("fixed", p_fixed))
    rec["fixed_size"] = list(sz)
    if max(meta["w"], meta["h"]) > UI_MAX_SIDE:
        p_small, _ = fixed_copy(path, tmpdir, max_side=UI_MAX_SIDE)
        variants.append(("small", p_small))
    for tag, p in variants:
        t0 = time.time()
        try:
            e = engine.read(p)
            txt = norm_text(e)
            rec[f"sim_{tag}"] = text_sim(c_txt, txt)
            rec[f"n_lines_{tag}"] = len(e.get("lines", []))
            rec[f"text_{tag}"] = txt[:160]
            rec[f"_full_{tag}"] = txt                    # bỏ khỏi file trước khi ghi (dùng cho đáp án∈OCR)
        except Exception as ex:
            rec[f"err_{tag}"] = f"{type(ex).__name__}: {ex}"[:200]
        rec[f"sec_{tag}"] = round(time.time() - t0, 3)
    rec["_full_cache"] = c_txt
    for tag, p in variants[1:]:
        try:
            os.remove(p)
        except OSError:
            pass
    return rec


def _wilson(k: int, n: int, z: float = 1.96) -> List[Optional[float]]:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, 100 * (c - h)), 2), round(min(100.0, 100 * (c + h)), 2)]


def verdict(scan_summary: Dict[str, Any], records: List[Dict[str, Any]], groups: Dict[str, List[str]],
            rule: Dict[str, Any] = RULE) -> Dict[str, Any]:
    """Áp luật L1–L3 (docs/W14/README.md, R-EXIF). Không có ảnh EXIF≠1 -> 'không có ảnh để kiểm', L2/L3 = 0."""
    by = {r["image"]: r for r in records}
    thr = rule["sim_mismatch"]

    def mismatch_rate(names: List[str], key: str) -> Dict[str, Any]:
        vals = [by[n][key] for n in names if n in by and key in by[n]]
        k = sum(1 for v in vals if v < thr)
        return {"n": len(vals), "n_mismatch": k, "pct": round(100.0 * k / len(vals), 2) if vals else None,
                "ci95": _wilson(k, len(vals))}

    aff, ctl = groups["affected"], groups["control"]
    m_aff, m_ctl = mismatch_rate(aff, "sim_fixed"), mismatch_rate(ctl, "sim_fixed")
    tv = scan_summary["vitextvqa_official"]
    test_frac = tv["test"]["n_questions_exif"] / max(tv["test"]["n_questions"], 1)
    train_frac = tv["train"]["n_questions_exif"] / max(tv["train"]["n_questions"], 1)
    if not aff:
        l1 = "KHÔNG CÓ ẢNH EXIF≠1 ĐỂ KIỂM"
        confirmed = False
    else:
        confirmed = (m_aff["pct"] is not None and m_aff["pct"] >= 100 * rule["l1_exif_min"]
                     and m_ctl["pct"] is not None and m_ctl["pct"] < 100 * rule["l1_control_max"])
        l1 = "CƠ CHẾ XÁC NHẬN" if confirmed else "CƠ CHẾ KHÔNG XÁC NHẬN"
    if not confirmed:
        l2 = "KHÔNG ÁP DỤNG (L1 không xác nhận) -> giữ số, ghi giới hạn"
        l3 = "KHÔNG ÁP DỤNG (L1 không xác nhận)"
    else:
        l2 = ("> 1% -> SỬA ocr.py + OCR lại ảnh EXIF≠1 + suy luận lại B3-full full test, báo cả hai số"
              if test_frac > rule["l2_test_frac"] else
              "≤ 1% -> giữ số chính thức, ghi giới hạn + báo EM khi bỏ các câu này")
        l3 = ("> 5% -> TRAIN LẠI B3-full với cache đã sửa; số chính thức ViTextVQA = bản train lại"
              if train_frac > rule["l3_train_frac"] else "≤ 5% -> không train lại")
    return {"rule": rule, "L1": l1, "L1_mechanism_confirmed": confirmed,
            "mismatch_affected_fixed": m_aff, "mismatch_control_fixed": m_ctl,
            "mismatch_affected_asis": mismatch_rate(aff, "sim_asis"),
            "mismatch_control_asis": mismatch_rate(ctl, "sim_asis"),
            "test_frac_questions_exif": round(100 * test_frac, 3), "L2": l2,
            "train_frac_questions_exif": round(100 * train_frac, 3), "L3": l3}


def answer_stats(records: List[Dict[str, Any]], names: List[str], test_recs: List[Dict[str, Any]],
                 full_texts: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Chỉ mô tả: % câu test (ảnh trong `names`) có đáp án gold nằm trong OCR — cache vs từng kiểu OCR lại."""
    names_set = set(names)
    qs = [r for r in test_recs if str(r["image"]) in names_set and str(r["image"]) in full_texts]
    out: Dict[str, Any] = {"n_questions": len(qs)}
    for tag in ("cache", "asis", "fixed", "small"):
        hit = [answer_in(full_texts[str(r["image"])][tag], r.get("answers") or [])
               for r in qs if tag in full_texts[str(r["image"])]]
        out[tag] = {"n": len(hit), "pct_answer_in_ocr": round(100.0 * sum(hit) / len(hit), 2) if hit else None}
    return out


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# =============================================================================
# 2) BA BƯỚC
# =============================================================================
def _records(drive_root: str, ds: str, split: str) -> List[Dict[str, Any]]:
    p = os.path.join(drive_root, "data", ds, SPLITS[ds][split])
    recs = json.load(open(p, encoding="utf-8"))
    exp = EXPECTED_Q.get((ds, split))
    if exp is not None and len(recs) != exp:
        raise ValueError(f"{ds}/{split}: {len(recs)} câu, khác số đã dùng cho mọi run ({exp}) -> DỪNG")
    return recs


def run_scan(drive_root: str, image_dirs: Dict[str, str], out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    summary: Dict[str, Any] = {}
    exif_images: Dict[str, Dict[str, Any]] = {}
    for ds, img_dir in image_dirs.items():
        if not img_dir:
            continue
        recs = {sp: _records(drive_root, ds, sp) for sp in SPLITS[ds]
                if os.path.exists(os.path.join(drive_root, "data", ds, SPLITS[ds][sp]))}
        names = sorted({str(r["image"]) for rs in recs.values() for r in rs})
        paths = {n: os.path.join(img_dir, n) for n in names}
        miss = [n for n, p in paths.items() if not os.path.exists(p)]
        if miss:
            raise FileNotFoundError(f"{ds}: {len(miss)} ảnh không có trên đĩa, vd {miss[:3]}")
        t0 = time.time()
        sc = scan_images(paths)
        summary[ds] = {sp: split_stats(rs, sc) for sp, rs in recs.items()}
        summary[ds]["all_images"] = {"n": len(sc), "n_exif": sum(1 for v in sc.values() if v["o"] != 1),
                                     "scan_sec": round(time.time() - t0, 1)}
        exif_images[ds] = {k: v for k, v in sc.items() if v["o"] != 1}
        if ds == "vitextvqa_official":
            exif_images["_vitextvqa_all_sizes"] = {k: [v["w"], v["h"], v["o"]] for k, v in sc.items()}
            exif_images["_must_include"] = {m: sc.get(m) for m in MUST_INCLUDE}
    from PIL import __version__ as pil_v
    out = {"what": "R-EXIF bước A — quét cờ EXIF Orientation (chỉ đọc header, không OCR)",
           "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
           "pillow": pil_v, "summary": summary, "images": exif_images}
    path = os.path.join(out_dir, "exif_scan.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def run_reocr(scan: Dict[str, Any], drive_root: str, img_dir: str, cache_path: str, engine: Any,
              out_dir: str, expect_cache_sha: Optional[str] = None) -> Dict[str, Any]:
    from src.data.ocr import load_ocr_cache
    sha = sha256_file(cache_path)
    if expect_cache_sha and sha != expect_cache_sha:
        raise ValueError(f"OCR cache sha256 {sha[:12]}… khác biên bản ({expect_cache_sha[:12]}…) -> DỪNG")
    cache = load_ocr_cache(cache_path)
    sizes = scan["images"]["_vitextvqa_all_sizes"]
    meta = {k: {"w": v[0], "h": v[1], "o": v[2]} for k, v in sizes.items()}
    test_imgs = sorted({str(r["image"]) for r in _records(drive_root, "vitextvqa_official", "test")})
    train_imgs = sorted({str(r["image"]) for r in _records(drive_root, "vitextvqa_official", "train")})
    groups = choose_sample(meta, test_imgs, train_imgs)
    todo = groups["affected"] + groups["control"]
    miss = [k for k in todo if k not in cache["images"]]
    if miss:
        raise KeyError(f"{len(miss)} ảnh mẫu không có trong cache, vd {miss[:3]}")
    print(f"[R-EXIF] mẫu: {len(groups['affected'])} ảnh EXIF≠1 + {len(groups['control'])} ảnh đối chứng", flush=True)
    recs: List[Dict[str, Any]] = []
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        for i, k in enumerate(todo, 1):
            r = reocr_one(engine, k, os.path.join(img_dir, k), cache["images"][k], td, meta[k])
            r["group"] = "affected" if k in groups["affected"] else "control"
            recs.append(r)
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[R-EXIF] {i}/{len(todo)} | {el / i:.2f} s/ảnh | còn ~{el / i * (len(todo) - i) / 60:.0f} phút",
                      flush=True)
    full = {r["image"]: {t: r.pop(f"_full_{t}") for t in ("cache", "asis", "fixed", "small") if f"_full_{t}" in r}
            for r in recs}
    test_recs = _records(drive_root, "vitextvqa_official", "test")
    out = {"what": "R-EXIF bước B — OCR lại mẫu cố định theo 3 kiểu (asis / fixed / small), so với cache",
           "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
           "cache_file": cache_path.replace(drive_root.rstrip("/") + "/", ""), "cache_sha256": sha,
           "cache_engine_info": cache.get("engine_info"), "engine_info_now": getattr(engine, "info", {}),
           "groups": groups, "records": recs,
           "answer_in_ocr_test": {g: answer_stats(recs, groups[g], test_recs, full) for g in ("affected", "control")}}
    with open(os.path.join(out_dir, "exif_reocr.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def run_analyze(scan: Dict[str, Any], reocr: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    v = verdict(scan["summary"], reocr["records"], reocr["groups"])
    recs = reocr["records"]
    small = [r for r in recs if "sim_small" in r]
    big_ctl = [r for r in small if r["group"] == "control"]
    out = {"what": "R-EXIF — phán quyết theo luật ghi trước (docs/W14/README.md)",
           "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
           "scan_summary": scan["summary"], "verdict": v,
           "answer_in_ocr_test": reocr.get("answer_in_ocr_test"),
           "descriptive": {
               "must_include": {m: next(({k: r.get(k) for k in ("group", "o", "w", "h", "sim_asis", "sim_fixed",
                                                                  "sim_small", "cache_text", "text_fixed", "text_small")}
                                         for r in recs if r["image"] == m), None) for m in MUST_INCLUDE},
               "n_large_images_in_sample": len(small),
               "control_large_mismatch_small": sum(1 for r in big_ctl if r.get("sim_small", 1) < RULE["sim_mismatch"]),
               "control_large_n": len(big_ctl),
               "n_ocr_errors": sum(1 for r in recs for t in ("asis", "fixed", "small") if f"err_{t}" in r)}}
    with open(os.path.join(out_dir, "exif_audit_summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


# =============================================================================
# 3) KIỂM THỬ OFFLINE
# =============================================================================
def _selftest() -> None:
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        # ảnh: 0 bình thường, 1 xoay 90° (o=6), 2 lật 180° (o=3), 3 ảnh lớn bình thường, 4 PNG không EXIF
        def mk(name, size, o=None, fmt="JPEG"):
            p = os.path.join(td, name)
            im = Image.new("RGB", size, "white")
            im.paste((0, 0, 0), (0, 0, size[0] // 4, size[1] // 4))
            if o is None:
                im.save(p, fmt)
            else:
                ex = Image.Exif()
                ex[ORIENTATION_TAG] = o
                im.save(p, fmt, exif=ex)
            return p
        paths = {"0.jpg": mk("0.jpg", (400, 300)), "1.jpg": mk("1.jpg", (400, 300), 6),
                 "2.jpg": mk("2.jpg", (400, 300), 3), "3.jpg": mk("3.jpg", (3000, 1000), 1),
                 "4.png": mk("4.png", (50, 40), None, "PNG")}
        sc = scan_images(paths)
        assert {k: v["o"] for k, v in sc.items()} == {"0.jpg": 1, "1.jpg": 6, "2.jpg": 3, "3.jpg": 1, "4.png": 1}
        assert sc["1.jpg"]["w"] == 400 and sc["1.jpg"]["h"] == 300            # PIL: kích thước CHƯA xoay
        recs = [{"image": "0.jpg", "answers": ["a"]}, {"image": "1.jpg", "answers": ["b"]},
                {"image": "1.jpg", "answers": ["c"]}, {"image": "2.jpg", "answers": ["d"]},
                {"image": "3.jpg", "answers": ["e"]}]
        st = split_stats(recs, sc)
        assert (st["n_images"], st["n_images_exif"], st["n_questions_exif"]) == (4, 2, 3), st
        assert st["exif_by_orientation"] == {"3": 1, "6": 1} and st["n_images_gt_ui_max_side"] == 1
        try:
            split_stats([{"image": "9.jpg"}], sc)
            raise AssertionError("không chặn ảnh chưa quét")
        except KeyError:
            pass
        # bản sửa: đã xoay đúng, không còn cờ; bản nhỏ: cạnh dài 2048
        p_fix, sz = fixed_copy(paths["1.jpg"], td)
        assert sz == (300, 400) and exif_orientation(p_fix) == 1 and Image.open(p_fix).size == (300, 400)
        p_s, sz2 = fixed_copy(paths["3.jpg"], td, max_side=UI_MAX_SIDE)
        assert max(sz2) == UI_MAX_SIDE
        # chuỗi
        assert text_sim("", "") == 1.0 and text_sim("abc", "") == 0.0 and text_sim("nhà thuốc", "nhà thuốc") == 1.0
        assert text_sim("whirs transly", "nhà thuốc an khang") < 0.5
        assert norm_text({"lines": [{"text": "NHÀ THUỐC,"}, {"text": "An Khang"}]}) == "nhà thuốc an khang"
        assert answer_in("nhà thuốc an khang", ["An Khang"]) and not answer_in("bán hàng", ["an"])
        # mẫu cố định, gồm ảnh bắt buộc, chạy lại ra cùng kết quả
        big = {f"{i}.jpg": {"o": (6 if i % 3 == 0 else 1), "w": 10, "h": 10} for i in range(60)}
        big["1849.jpg"] = {"o": 1, "w": 4000, "h": 3000}
        big["14277.jpg"] = {"o": 6, "w": 4000, "h": 3000}
        te = [f"{i}.jpg" for i in range(30)] + ["1849.jpg"]
        tr = [f"{i}.jpg" for i in range(30, 60)] + ["14277.jpg"]
        g1 = choose_sample(big, te, tr, n_affected=12, n_control=5)
        g2 = choose_sample(big, te, tr, n_affected=12, n_control=5)
        assert g1 == g2 and "1849.jpg" in g1["control"] and "14277.jpg" in g1["affected"]
        assert all(big[k]["o"] != 1 for k in g1["affected"]) and all(big[k]["o"] == 1 for k in g1["control"])
        assert sum(1 for k in g1["affected"] if k in te) == 10        # ưu tiên toàn bộ 10 ảnh EXIF≠1 của test
        # vòng OCR lại với engine giả MÔ PHỎNG lỗi: chữ phụ thuộc kích thước ảnh nhìn thấy
        class _Eng:
            info = {"engine": "mock-size"}

            def read(self, p):
                with Image.open(p) as im:
                    w, h = im.size
                return {"w": w, "h": h, "lines": [{"text": f"CHU {w} {h}", "conf": 0.9, "box": [0, 0, 1, 1]}]}
        meta1 = {"o": 6, "w": 400, "h": 300}
        cache_bad = {"lines": [{"text": "whirs transly contracts"}]}           # cache rác
        r = reocr_one(_Eng(), "1.jpg", paths["1.jpg"], cache_bad, td, meta1)
        assert r["sim_fixed"] < 0.5 and r["text_fixed"] == "chu 300 400" and "sim_small" not in r
        cache_ok = {"lines": [{"text": "CHU 3000 1000"}]}
        r3 = reocr_one(_Eng(), "3.jpg", paths["3.jpg"], cache_ok, td, {"o": 1, "w": 3000, "h": 1000})
        assert r3["sim_asis"] == 1.0 and r3["sim_fixed"] == 1.0 and r3["sim_small"] < 1.0
        assert not [f for f in os.listdir(td) if f.endswith(".png") and "__" in f and f != os.path.basename(p_fix)
                    and f != os.path.basename(p_s)], "file tạm chưa xoá"
        # luật
        ss = {"vitextvqa_official": {"test": {"n_questions": 1000, "n_questions_exif": 20},
                                     "train": {"n_questions": 1000, "n_questions_exif": 80}}}
        recs_v = ([{"image": f"a{i}", "sim_fixed": 0.1, "sim_asis": 0.1} for i in range(9)]
                  + [{"image": "a9", "sim_fixed": 0.9, "sim_asis": 0.9}]
                  + [{"image": f"c{i}", "sim_fixed": 0.95, "sim_asis": 0.9} for i in range(20)])
        grp = {"affected": [f"a{i}" for i in range(10)], "control": [f"c{i}" for i in range(20)]}
        v = verdict(ss, recs_v, grp)
        assert v["L1_mechanism_confirmed"] and v["mismatch_affected_fixed"]["pct"] == 90.0
        assert v["L2"].startswith("> 1%") and v["L3"].startswith("> 5%")
        ss2 = {"vitextvqa_official": {"test": {"n_questions": 1000, "n_questions_exif": 5},
                                      "train": {"n_questions": 1000, "n_questions_exif": 10}}}
        v2 = verdict(ss2, recs_v, grp)
        assert v2["L2"].startswith("≤ 1%") and v2["L3"].startswith("≤ 5%")
        recs_c = recs_v[:10] + [{"image": f"c{i}", "sim_fixed": (0.1 if i < 2 else 0.95)} for i in range(20)]
        v3 = verdict(ss, recs_c, grp)                                # đối chứng lệch 10% -> không xác nhận
        assert not v3["L1_mechanism_confirmed"] and v3["L2"].startswith("KHÔNG ÁP DỤNG")
        v4 = verdict(ss, [], {"affected": [], "control": []})
        assert v4["L1"] == "KHÔNG CÓ ẢNH EXIF≠1 ĐỂ KIỂM" and not v4["L1_mechanism_confirmed"]
        # thống kê đáp án∈OCR
        full = {"1.jpg": {"cache": "whirs", "fixed": "nhà thuốc an khang", "asis": "whirs"}}
        a = answer_stats([], ["1.jpg"], [{"image": "1.jpg", "answers": ["an khang"]}], full)
        assert a["cache"]["pct_answer_in_ocr"] == 0.0 and a["fixed"]["pct_answer_in_ocr"] == 100.0
        assert a["small"]["n"] == 0
    print("exif_audit SELFTEST OK", flush=True)


# =============================================================================
# 4) CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="R-EXIF: cờ xoay EXIF có làm hỏng OCR cache không (W14)")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--reocr", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--drive_root", default=DEFAULT_DRIVE_ROOT)
    ap.add_argument("--vitextvqa_images", default=None)
    ap.add_argument("--vivqa_images", default=None)
    ap.add_argument("--scan_json", default=None)
    ap.add_argument("--reocr_json", default=None)
    ap.add_argument("--cache", default=None, help="mặc định data/vitextvqa_official/ocr/paddledet_vietocr.json")
    ap.add_argument("--expect_cache_sha", default=None)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.out:
        ap.error("cần --out")
    os.makedirs(a.out, exist_ok=True)
    if a.scan:
        s = run_scan(a.drive_root, {"vitextvqa_official": a.vitextvqa_images, "vivqa": a.vivqa_images}, a.out)
        print(json.dumps(s["summary"], ensure_ascii=False, indent=1))
    if a.reocr:
        from src.data.ocr import default_cache_path, get_ocr_engine
        scan = json.load(open(a.scan_json or os.path.join(a.out, "exif_scan.json"), encoding="utf-8"))
        eng = get_ocr_engine("paddledet_vietocr", device=a.device)
        print(f"[R-EXIF] engine: {json.dumps(eng.info, ensure_ascii=False)}", flush=True)
        cp = a.cache or default_cache_path(a.drive_root, "vitextvqa_official", "paddledet_vietocr")
        run_reocr(scan, a.drive_root, a.vitextvqa_images, cp, eng, a.out, a.expect_cache_sha)
    if a.analyze:
        scan = json.load(open(a.scan_json or os.path.join(a.out, "exif_scan.json"), encoding="utf-8"))
        re_ = json.load(open(a.reocr_json or os.path.join(a.out, "exif_reocr.json"), encoding="utf-8"))
        s = run_analyze(scan, re_, a.out)
        print(json.dumps(s["verdict"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
