"""
prepare_vitextvqa.py — Chuẩn bị dataset ViTextVQA CHÍNH THỨC cho ViVQA-VLM.

Nguồn chính thức (gated, cần đồng ý điều kiện + HF_TOKEN):
    https://huggingface.co/datasets/minhquan6203/ViTextVQA   (licence CC BY-NC 3.0)
    ViTextVQA_train.json    35.159 câu  | 11.733 ảnh
    ViTextVQA_dev.json       5.155 câu  |  1.676 ảnh
    ViTextVQA_test_gt.json  10.028 câu  |  3.353 ảnh   <- đáp án THẬT của test
    ViTextVQA_test.json     = mẫu nộp Kaggle (đáp án 'your answer') -> KHÔNG dùng để chấm
    ViTextVQA_images.zip    ~5,6 GB, 16.762 ảnh

Định dạng gốc (COCO-style):
    images:      [{id, filename}, ...]
    annotations: [{id, image_id, question, answers}, ...]

Module này:
  1. Giữ nguyên các hàm W4 (find_image_root, flatten_coco_style, flatten_coco_file)
     vì notebook W4 còn import.
  2. Thêm CLI chuẩn bị bản chính thức:
       - tải 3 file chú thích (train / dev / test_gt) vào <out_dir>/_raw/
       - làm phẳng -> <out_dir>/{train,dev,test}.json, question_id dạng "<split>_<id>"
       - kiểm tra: đúng số câu, không còn 'your answer', đáp án không rỗng, id không trùng
       - chọn tập con test cố định (mặc định 2.000 câu, seed 42) -> lưu danh sách id
       - (tùy chọn) tải zip ảnh lên <zip_dir> MỘT lần, giải nén vào <images_dir> (/content)
       - ghi prep_report.json (số liệu + sha256 file gốc + commit của kho HF)

Cách chạy (Colab):
    import os; from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    !python -m src.data.prepare_vitextvqa --out_dir "/content/drive/MyDrive/ViVQA-VLM/data/vitextvqa_official"
Kiểm thử offline (không cần mạng):
    python -m src.data.prepare_vitextvqa --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Hằng số của bản chính thức
# --------------------------------------------------------------------------- #
REPO_ID = "minhquan6203/ViTextVQA"
SPLIT_FILES = {
    "train": "ViTextVQA_train.json",
    "dev": "ViTextVQA_dev.json",
    "test": "ViTextVQA_test_gt.json",   # đáp án thật; KHÔNG dùng ViTextVQA_test.json
}
ZIP_NAME = "ViTextVQA_images.zip"
EXPECTED_COUNTS = {"train": 35159, "dev": 5155, "test": 10028}
EXPECTED_IMAGES_TOTAL = 16762
PLACEHOLDER_ANSWER = "your answer"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------- #
# Các hàm W4 — GIỮ NGUYÊN chữ ký (notebook W4 đang import)
# --------------------------------------------------------------------------- #
def find_image_root(root: str) -> str:
    """Thư mục thật chứa ảnh (.jpg) — zip đôi khi lồng thêm 1 lớp folder."""
    for dirpath, _, filenames in os.walk(root):
        if any(name.lower().endswith(".jpg") for name in filenames):
            return dirpath
    return root


def flatten_coco_style(coco: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Join images <-> annotations -> list phẳng (1 phần tử / câu hỏi):
        question_id, question, answers, image (tên file), question_type="unknown".
    Bỏ annotation nếu image_id không có trong images.
    """
    images = coco.get("images") or []
    annotations = coco.get("annotations") or []
    id2file = {im["id"]: im["filename"] for im in images}

    flat: List[Dict[str, Any]] = []
    for ann in annotations:
        filename = id2file.get(ann["image_id"])
        if filename is None:
            continue
        flat.append({
            "question_id": ann["id"],
            "question": ann["question"],
            "answers": ann["answers"],
            "image": filename,
            "question_type": "unknown",
        })
    return flat


def flatten_coco_file(test_json_path: str, out_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Đọc file COCO-style, flatten, tùy chọn ghi JSON (một mảng)."""
    with open(test_json_path, encoding="utf-8") as fp:
        coco = json.load(fp)
    if not isinstance(coco, dict):
        raise ValueError(f"Kỳ vọng dict COCO-style, nhận {type(coco).__name__}: {test_json_path}")
    flat = flatten_coco_style(coco)
    if out_path:
        _write_json(out_path, flat)
    return flat


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def _write_json(path: str, obj: Any, indent: Optional[int] = None) -> None:
    """Ghi JSON an toàn: ghi .tmp rồi đổi tên (không để file dở dang)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    """Mã băm SHA-256 của file — bằng chứng nguồn gốc dữ liệu cho báo cáo."""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _hf_token() -> Optional[str]:
    """Đọc token từ biến môi trường HF_TOKEN (Colab: nạp từ Secrets)."""
    return os.environ.get("HF_TOKEN") or None


# --------------------------------------------------------------------------- #
# Bước 1 — tải chú thích
# --------------------------------------------------------------------------- #
def download_annotations(raw_dir: str, repo_id: str = REPO_ID) -> Dict[str, str]:
    """Tải 3 file chú thích (train/dev/test_gt) vào raw_dir. Trả {split: đường dẫn}."""
    from huggingface_hub import hf_hub_download  # import muộn: selftest không cần mạng

    os.makedirs(raw_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for split, fname in SPLIT_FILES.items():
        local = os.path.join(raw_dir, fname)
        if not os.path.exists(local):
            print(f"  [tải] {fname}")
            hf_hub_download(repo_id, fname, repo_type="dataset", token=_hf_token(), local_dir=raw_dir)
        else:
            print(f"  [có sẵn] {fname}")
        paths[split] = local
    return paths


def repo_revision(repo_id: str = REPO_ID) -> Optional[str]:
    """Commit hiện tại của kho HF (ghi vào report để tái lập). None nếu không lấy được."""
    try:
        from huggingface_hub import HfApi
        return HfApi(token=_hf_token()).dataset_info(repo_id).sha
    except Exception as exc:  # không chặn cả pipeline chỉ vì thiếu thông tin phụ
        print(f"  [cảnh báo] không lấy được revision: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Bước 2 — làm phẳng + kiểm tra
# --------------------------------------------------------------------------- #
def build_split(raw_path: str, split: str) -> List[Dict[str, Any]]:
    """Làm phẳng 1 split; question_id đổi thành '<split>_<id>' để không trùng giữa các split."""
    with open(raw_path, encoding="utf-8") as fp:
        coco = json.load(fp)
    n_ann = len(coco.get("annotations") or [])
    flat = flatten_coco_style(coco)
    if len(flat) != n_ann:
        raise ValueError(f"[{split}] {n_ann - len(flat)} annotation trỏ tới image_id không tồn tại")
    for rec in flat:
        rec["question_id"] = f"{split}_{rec['question_id']}"
        rec["answers"] = [a.strip() for a in rec["answers"] if isinstance(a, str) and a.strip()]
    return flat


def validate_split(flat: List[Dict[str, Any]], split: str,
                   expected: Optional[int] = None) -> Dict[str, Any]:
    """Kiểm tra 1 split; lỗi nghiêm trọng -> raise. Trả thống kê."""
    if expected is not None and len(flat) != expected:
        raise ValueError(f"[{split}] có {len(flat)} câu, kỳ vọng {expected}")
    ids = [r["question_id"] for r in flat]
    if len(set(ids)) != len(ids):
        raise ValueError(f"[{split}] question_id bị trùng")
    empty = [r["question_id"] for r in flat if not r["answers"]]
    if empty:
        raise ValueError(f"[{split}] {len(empty)} câu không có đáp án, vd {empty[:3]}")
    placeholder = [r["question_id"] for r in flat
                   if any(a.lower() == PLACEHOLDER_ANSWER for a in r["answers"])]
    if placeholder:
        raise ValueError(f"[{split}] {len(placeholder)} câu có đáp án mẫu '{PLACEHOLDER_ANSWER}' "
                         f"-> đang dùng nhầm file mẫu nộp Kaggle?")
    dup_pairs = len(flat) - len({(r["image"], r["question"].strip()) for r in flat})
    return {
        "num_questions": len(flat),
        "num_images": len({r["image"] for r in flat}),
        "duplicate_image_question_pairs": dup_pairs,
        "answers_per_question_max": max(len(r["answers"]) for r in flat),
    }


def make_subset(flat: List[Dict[str, Any]], n: int, seed: int) -> List[str]:
    """Chọn n question_id ngẫu nhiên, cố định theo seed; trả danh sách đã sắp xếp."""
    ids = sorted(r["question_id"] for r in flat)
    if n >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, n))


# --------------------------------------------------------------------------- #
# Bước 3 — ảnh (tùy chọn)
# --------------------------------------------------------------------------- #
def ensure_images(zip_dir: str, images_dir: str, repo_id: str = REPO_ID) -> Dict[str, Any]:
    """
    Zip ảnh lưu ở zip_dir (Drive) — chỉ tải lần đầu.
    Giải nén vào images_dir (đĩa Colab /content) — mỗi phiên làm 1 lần, có marker.
    Trả {image_root, num_images_in_zip, ...}.
    """
    zip_path = os.path.join(zip_dir, ZIP_NAME)
    if not os.path.exists(zip_path):
        from huggingface_hub import hf_hub_download
        print(f"  [tải] {ZIP_NAME} (~5,6 GB) -> {zip_dir}")
        os.makedirs(zip_dir, exist_ok=True)
        t0 = time.time()
        hf_hub_download(repo_id, ZIP_NAME, repo_type="dataset", token=_hf_token(), local_dir=zip_dir)
        print(f"  [xong] tải zip trong {time.time() - t0:.0f}s")
    else:
        print(f"  [có sẵn] {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path} không phải zip hợp lệ (tải dở?) -> xoá file rồi chạy lại")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(IMAGE_EXTS)]

    marker = os.path.join(images_dir, ".extracted_ok")
    if not os.path.exists(marker):
        print(f"  [giải nén] {len(members)} ảnh -> {images_dir}")
        t0 = time.time()
        os.makedirs(images_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(images_dir)
        with open(marker, "w") as fp:
            fp.write(datetime.now(timezone.utc).isoformat())
        print(f"  [xong] giải nén trong {time.time() - t0:.0f}s")
    else:
        print(f"  [có sẵn] ảnh đã giải nén ở {images_dir}")

    return {"zip_path": zip_path, "num_images_in_zip": len(members),
            "image_root": find_image_root(images_dir)}


def check_images(splits: Dict[str, List[Dict[str, Any]]], image_root: str) -> Dict[str, int]:
    """Đếm số ảnh được tham chiếu nhưng không tồn tại trong image_root."""
    present = set(os.listdir(image_root))
    return {s: sum(1 for r in recs if r["image"] not in present) for s, recs in splits.items()}


# --------------------------------------------------------------------------- #
# Pipeline chính
# --------------------------------------------------------------------------- #
def prepare(out_dir: str, zip_dir: Optional[str] = None, images_dir: str = "/content/vitextvqa_images",
            skip_images: bool = False, subset_n: int = 2000, seed: int = 42,
            repo_id: str = REPO_ID, expected_counts: Optional[Dict[str, int]] = EXPECTED_COUNTS,
            raw_paths: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Chạy toàn bộ pipeline, ghi file vào out_dir, trả report (dict)."""
    t_start = time.time()
    zip_dir = zip_dir or out_dir
    print(f"[1] Chú thích -> {out_dir}/_raw")
    downloaded = raw_paths is None          # False khi selftest truyền file giả
    if downloaded:
        raw_paths = download_annotations(os.path.join(out_dir, "_raw"), repo_id)

    print("[2] Làm phẳng + kiểm tra")
    splits: Dict[str, List[Dict[str, Any]]] = {}
    stats: Dict[str, Any] = {}
    for split, path in raw_paths.items():
        flat = build_split(path, split)
        stats[split] = validate_split(flat, split, (expected_counts or {}).get(split))
        _write_json(os.path.join(out_dir, f"{split}.json"), flat)
        splits[split] = flat
        print(f"  {split:5s}: {stats[split]}")

    print(f"[3] Tập con test: n={subset_n}, seed={seed}")
    subset_ids = make_subset(splits["test"], subset_n, seed)
    subset_name = f"test_subset{len(subset_ids)}_seed{seed}"
    keep = set(subset_ids)
    _write_json(os.path.join(out_dir, f"{subset_name}_ids.json"), subset_ids, indent=0)
    _write_json(os.path.join(out_dir, f"{subset_name}.json"),
                [r for r in splits["test"] if r["question_id"] in keep])
    print(f"  -> {subset_name}_ids.json + {subset_name}.json")

    images_info: Dict[str, Any] = {"skipped": True}
    if not skip_images:
        print("[4] Ảnh")
        images_info = ensure_images(zip_dir, images_dir, repo_id)
        missing = check_images(splits, images_info["image_root"])
        images_info["missing_per_split"] = missing
        print(f"  ảnh thiếu theo split: {missing}")
        if any(missing.values()):
            raise ValueError(f"Có ảnh được tham chiếu nhưng không tồn tại: {missing}")

    report = {
        "dataset": "vitextvqa_official",
        "source_repo": repo_id,
        "source_revision": repo_revision(repo_id) if downloaded else None,
        "licence": "CC BY-NC 3.0",
        "raw_files": {s: {"file": os.path.basename(p), "sha256": _sha256(p)} for s, p in raw_paths.items()},
        "split_stats": stats,
        "subset": {"name": subset_name, "n": len(subset_ids), "seed": seed, "from_split": "test"},
        "images": {k: v for k, v in images_info.items() if k != "zip_path"},
        "elapsed_seconds": round(time.time() - t_start, 1),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json(os.path.join(out_dir, "prep_report.json"), report, indent=2)
    print(f"[xong] prep_report.json -> {out_dir}")
    return report


# --------------------------------------------------------------------------- #
# Kiểm thử offline
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Kiểm thử logic bằng dữ liệu giả, không cần mạng."""
    import tempfile

    # (a) hàm W4 vẫn chạy như cũ
    demo = {"images": [{"id": 7, "filename": "foo.jpg"}],
            "annotations": [{"id": 29, "image_id": 7, "question": "chữ gì?", "answers": ["cool air"]},
                            {"id": 99, "image_id": 999, "question": "mồ côi", "answers": ["x"]}]}
    assert len(flatten_coco_style(demo)) == 1

    with tempfile.TemporaryDirectory() as td:
        def coco(n_img, n_q, answer="đáp án"):
            return {"images": [{"id": i, "filename": f"{i}.jpg"} for i in range(n_img)],
                    "annotations": [{"id": q, "image_id": q % n_img, "question": f"câu {q} ?",
                                     "answers": [answer]} for q in range(n_q)]}
        raw = {}
        for split, (ni, nq) in {"train": (6, 20), "dev": (3, 8), "test": (4, 12)}.items():
            p = os.path.join(td, f"{split}.json")
            _write_json(p, coco(ni, nq))
            raw[split] = p

        # (b) chạy pipeline đầy đủ (bỏ ảnh), subset 5
        out = os.path.join(td, "out")
        rep = prepare(out, skip_images=True, subset_n=5, seed=42,
                      expected_counts={"train": 20, "dev": 8, "test": 12}, raw_paths=raw)
        assert rep["split_stats"]["test"]["num_questions"] == 12
        ids = json.load(open(os.path.join(out, "test_subset5_seed42_ids.json"), encoding="utf-8"))
        assert len(ids) == 5 and all(i.startswith("test_") for i in ids)
        assert ids == make_subset(json.load(open(os.path.join(out, "test.json"), encoding="utf-8")), 5, 42)

        # (c) phải CHẶN file mẫu Kaggle ('your answer')
        bad = os.path.join(td, "bad.json")
        _write_json(bad, coco(4, 12, answer="your answer"))
        try:
            validate_split(build_split(bad, "test"), "test")
            raise AssertionError("không chặn được 'your answer'")
        except ValueError as exc:
            assert "your answer" in str(exc)

        # (d) phải chặn sai số câu
        try:
            validate_split(build_split(raw["test"], "test"), "test", expected=999)
            raise AssertionError("không chặn được sai số câu")
        except ValueError:
            pass

        # (e) giải nén + kiểm tra ảnh từ zip giả (có lớp thư mục lồng)
        zdir = os.path.join(td, "zip")
        os.makedirs(zdir)
        with zipfile.ZipFile(os.path.join(zdir, ZIP_NAME), "w") as zf:
            for i in range(6):
                zf.writestr(f"images/{i}.jpg", b"x")
        info = ensure_images(zdir, os.path.join(td, "img"))
        assert info["num_images_in_zip"] == 6 and info["image_root"].endswith("images")
        splits = {s: build_split(p, s) for s, p in raw.items()}
        assert check_images(splits, info["image_root"]) == {"train": 0, "dev": 0, "test": 0}
        info2 = ensure_images(zdir, os.path.join(td, "img"))   # lần 2: không giải nén lại
        assert info2["image_root"] == info["image_root"]
    print("SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chuẩn bị ViTextVQA chính thức (minhquan6203/ViTextVQA).")
    ap.add_argument("--out_dir", help="Thư mục kết quả (Drive), vd .../data/vitextvqa_official")
    ap.add_argument("--zip_dir", default=None, help="Nơi lưu zip ảnh (mặc định = out_dir)")
    ap.add_argument("--images_dir", default="/content/vitextvqa_images", help="Nơi giải nén ảnh (đĩa Colab)")
    ap.add_argument("--skip_images", action="store_true", help="Chỉ làm chú thích, không tải/giải nén ảnh")
    ap.add_argument("--subset_n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo_id", default=REPO_ID)
    ap.add_argument("--selftest", action="store_true", help="Kiểm thử offline rồi thoát")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.out_dir:
        ap.error("--out_dir là bắt buộc (trừ khi --selftest)")
    prepare(args.out_dir, zip_dir=args.zip_dir, images_dir=args.images_dir,
            skip_images=args.skip_images, subset_n=args.subset_n, seed=args.seed, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
