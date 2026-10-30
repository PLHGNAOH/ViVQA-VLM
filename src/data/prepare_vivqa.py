"""
prepare_vivqa.py — Chuẩn bị dataset ViVQA (Tran et al., PACLIC 2021).

ViVQA THẬT (KHÁC ViVQA-X): nguồn chính thức github.com/kh4nh12/ViVQA ship 2 file
CSV (train.csv + test.csv) với các cột: [Unnamed: 0, question, answer, img_id, type].
Ảnh KHÔNG kèm trong CSV — ``img_id`` là mã ảnh MS COCO 2014, phải tải riêng từ
host images.cocodataset.org.

Script này (chỉ dùng thư viện phổ thông + requests/pandas/tqdm):
    1) Tải train.csv/test.csv (nếu chưa có trong out_dir/_csv/).
    2) Đọc bằng pandas, bỏ cột "Unnamed: 0", giải mã cột ``type`` (quy ước COCO-QA).
    3) NFC-normalise tiếng Việt cho question/answer.
    4) Sinh record theo schema thống nhất (answers LUÔN là list).
    5) Cắt VAL từ train.csv theo val_ratio (shuffle có seed); test.csv giữ làm TEST.
    6) Tải ảnh COCO cần dùng (đa luồng, resumable, có smoke-test host trước).
    7) Điền record["image"], bỏ record ảnh hỏng, ghi {train,val,test}.json.
    8) Ghi prep_report.json (thống kê + thời gian chạy).

CHẠY:
    python -m src.data.prepare_vivqa --out_dir data/vivqa
    python -m src.data.prepare_vivqa --out_dir data/vivqa_smoke --limit 20   # smoke-test

LƯU Ý: script CÓ tải dữ liệu qua mạng khi chạy. Việc chạy do người dùng thực hiện.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


# =============================================================================
# Hằng số cấu hình nguồn
# =============================================================================
CSV_URLS: Dict[str, str] = {
    "train": "https://raw.githubusercontent.com/kh4nh12/ViVQA/main/train.csv",
    "test": "https://raw.githubusercontent.com/kh4nh12/ViVQA/main/test.csv",
}

# Quy ước nhãn loại câu hỏi của COCO-QA (ViVQA kế thừa).
TYPE_MAP: Dict[int, str] = {
    0: "object",
    1: "number",
    2: "color",
    3: "location",
}

# Hai split ảnh khả dĩ trên host COCO 2014.
COCO_HOST = "http://images.cocodataset.org"
COCO_SPLITS = ("train2014", "val2014")

# Timeout (giây) cho mỗi request ảnh/CSV.
REQ_TIMEOUT = 30


# =============================================================================
# 1) TEXT NORMALISATION
# =============================================================================
def nfc(text: Any) -> str:
    """Chuẩn hoá chuỗi về dạng Unicode NFC (bắt buộc cho tiếng Việt)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFC", text).strip()


def decode_type(raw: Any) -> str:
    """Giải mã cột ``type`` (int COCO-QA) → nhãn chữ; mã lạ → 'unknown'."""
    try:
        return TYPE_MAP.get(int(raw), "unknown")
    except (TypeError, ValueError):
        return "unknown"


# =============================================================================
# 2) TẢI CSV
# =============================================================================
def download_csv(url: str, dest_path: str) -> None:
    """Tải 1 file CSV về ``dest_path`` nếu chưa tồn tại (resumable đơn giản)."""
    if os.path.exists(dest_path):
        print(f"  [cached] {os.path.basename(dest_path)} đã có, bỏ qua tải.")
        return
    print(f"  [get] {url}")
    resp = requests.get(url, timeout=REQ_TIMEOUT)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as fh:
        fh.write(resp.content)
    print(f"  [ok] lưu {dest_path} ({len(resp.content)} bytes)")


def load_csv(csv_path: str) -> pd.DataFrame:
    """Đọc CSV bằng pandas, bỏ cột 'Unnamed: 0' nếu có."""
    df = pd.read_csv(csv_path)
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


# =============================================================================
# 3) SINH RECORD TỪ DATAFRAME
# =============================================================================
def build_records(df: pd.DataFrame, origin: str) -> List[Dict[str, Any]]:
    """
    Chuyển từng dòng CSV → record trung gian (chưa có question_id/image).

    ``origin`` ∈ {"train", "test"} — dùng để quyết định thứ tự thử split ảnh COCO.
    answers LUÔN là list; img_id ép về int.
    """
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            img_id = int(row["img_id"])
        except (TypeError, ValueError, KeyError):
            # Bỏ dòng không có img_id hợp lệ (không thể ánh xạ tới ảnh COCO).
            continue
        records.append({
            "question": nfc(row.get("question")),
            "answers": [nfc(row.get("answer"))],
            "img_id": img_id,
            "question_type": decode_type(row.get("type")),
            "image": None,          # điền sau khi tải ảnh
            "_origin": origin,      # nội bộ: nguồn CSV để chọn thứ tự split ảnh
        })
    return records


def assign_question_ids(records: List[Dict[str, Any]], split: str) -> None:
    """Gán question_id duy nhất theo split cuối cùng: f'{split}_{i}' (in-place)."""
    for i, rec in enumerate(records):
        rec["question_id"] = f"{split}_{i}"


# =============================================================================
# 4) CẮT VAL
# =============================================================================
def split_train_val(
    train_records: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Xáo trộn train_records (seed cố định) và tách val_ratio làm VAL.

    Trả về (train_còn_lại, val). Phần VAL lấy từ ĐẦU danh sách đã shuffle.
    """
    rng = random.Random(seed)
    shuffled = train_records[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_ratio))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


# =============================================================================
# 5) TẢI ẢNH COCO
# =============================================================================
def coco_filename(split: str, img_id: int) -> str:
    """Tên file COCO chuẩn: COCO_<split>_<img_id 12 chữ số>.jpg."""
    return f"COCO_{split}_{img_id:012d}.jpg"


def coco_url(split: str, img_id: int) -> str:
    """URL ảnh COCO tương ứng với split + img_id."""
    return f"{COCO_HOST}/{split}/{coco_filename(split, img_id)}"


def trial_order(origin: str) -> Tuple[str, ...]:
    """
    Thứ tự thử split ảnh theo nguồn CSV.

    Ảnh từ train.csv: thử 'train2014' trước; ảnh từ test.csv: thử 'val2014' trước.
    """
    if origin == "test":
        return ("val2014", "train2014")
    return ("train2014", "val2014")


def download_image(
    img_id: int,
    order: Tuple[str, ...],
    images_dir: str,
    timeout: int = REQ_TIMEOUT,
) -> Dict[str, Any]:
    """
    Tải 1 ảnh COCO theo thứ tự split ``order``. Resumable: có sẵn thì bỏ qua.

    Trả về dict: {img_id, image (tên file|None), split (|None), status}.
    status ∈ {cached, downloaded, not_found, network_error}.
      - not_found: host phản hồi (vd 404) nhưng không split nào có ảnh.
      - network_error: KHÔNG kết nối được host ở mọi lần thử.
    """
    # (a) đã tải trước đó ở bất kỳ split khả dĩ nào?
    for split in order:
        fname = coco_filename(split, img_id)
        dest = os.path.join(images_dir, fname)
        if os.path.exists(dest):
            return {"img_id": img_id, "image": fname, "split": split, "status": "cached"}

    # (b) thử tải theo thứ tự.
    got_response = False
    for split in order:
        fname = coco_filename(split, img_id)
        dest = os.path.join(images_dir, fname)
        url = coco_url(split, img_id)
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            continue  # lỗi mạng ở split này → thử split kế
        got_response = True
        if resp.status_code == 200:
            tmp = dest + ".part"
            os.makedirs(os.path.dirname(tmp), exist_ok=True)  # chắc chắn có thư mục ảnh
            try:
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
                os.replace(tmp, dest)  # atomic: chỉ hiện file khi tải xong
            finally:
                resp.close()
                if os.path.exists(tmp):
                    os.remove(tmp)
            return {"img_id": img_id, "image": fname, "split": split, "status": "downloaded"}
        resp.close()

    status = "not_found" if got_response else "network_error"
    return {"img_id": img_id, "image": None, "split": None, "status": status}


def smoke_test_host(sample_img_id: int, order: Tuple[str, ...], images_dir: str) -> None:
    """
    Thử tải đúng 1 ảnh để chắc host COCO tới được, TRƯỚC khi tải hàng loạt.

    Nếu network_error → in lỗi rõ ràng và thoát (tránh lặp vô ích 10k lần).
    (not_found vẫn coi là host OK — chỉ là ảnh đó tình cờ không có.)
    """
    print(f"[smoke] thử tải 1 ảnh (img_id={sample_img_id}) để kiểm tra host COCO...")
    result = download_image(sample_img_id, order, images_dir)
    if result["status"] == "network_error":
        print(
            "[LỖI] Không kết nối được host ảnh COCO "
            f"({COCO_HOST}). Kiểm tra mạng/proxy rồi chạy lại. Đã dừng.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[smoke] OK — host phản hồi (status={result['status']}).")


def collect_unique_images(all_records: List[Dict[str, Any]]) -> Dict[int, Tuple[str, ...]]:
    """
    Gom img_id DUY NHẤT của cả 3 split → thứ tự thử split ảnh.

    Nếu 1 img_id xuất hiện ở cả train lẫn test, ưu tiên thứ tự của train.
    """
    order_by_id: Dict[int, Tuple[str, ...]] = {}
    for rec in all_records:
        img_id = rec["img_id"]
        origin = rec["_origin"]
        if img_id not in order_by_id:
            order_by_id[img_id] = trial_order(origin)
        elif origin == "train":
            # train được ưu tiên hơn (ghi đè thứ tự của test nếu có).
            order_by_id[img_id] = trial_order("train")
    return order_by_id


def download_all_images(
    order_by_id: Dict[int, Tuple[str, ...]],
    images_dir: str,
    workers: int,
) -> Tuple[Dict[int, str], List[int]]:
    """
    Tải đa luồng toàn bộ ảnh cần dùng, có tqdm.

    Trả về (id2file: img_id→tên file tải được, failed: list img_id thất bại).
    """
    os.makedirs(images_dir, exist_ok=True)
    id2file: Dict[int, str] = {}
    failed: List[int] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_image, img_id, order, images_dir): img_id
            for img_id, order in order_by_id.items()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="images", unit="img"):
            res = fut.result()
            if res["image"] is not None:
                id2file[res["img_id"]] = res["image"]
            else:
                failed.append(res["img_id"])
    return id2file, failed


# =============================================================================
# 6) ĐIỀN ẢNH + GHI FILE
# =============================================================================
def attach_images(
    records: List[Dict[str, Any]],
    id2file: Dict[int, str],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Điền record['image'] từ id2file; bỏ record nào ảnh tải hỏng.

    Trả về (records_giữ_lại, số_bị_bỏ). Xoá luôn khoá nội bộ '_origin'.
    """
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for rec in records:
        fname = id2file.get(rec["img_id"])
        if fname is None:
            dropped += 1
            continue
        rec["image"] = fname
        rec.pop("_origin", None)
        kept.append(rec)
    return kept, dropped


def write_json(records: List[Dict[str, Any]], path: str) -> None:
    """Ghi list record ra JSON (ensure_ascii=False để giữ tiếng Việt)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)


# =============================================================================
# 7) MAIN
# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out_dir", default="data/vivqa",
                        help="Thư mục output (mặc định: data/vivqa).")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Tỉ lệ tách VAL từ train.csv (mặc định: 0.1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed shuffle khi cắt VAL (mặc định: 42).")
    parser.add_argument("--workers", type=int, default=16,
                        help="Số luồng tải ảnh (mặc định: 16).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số dòng đọc mỗi CSV để smoke-test (mặc định: None = tất cả).")
    args = parser.parse_args()

    t0 = time.time()
    out_dir = args.out_dir
    csv_dir = os.path.join(out_dir, "_csv")
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    # Tạo sẵn thư mục ảnh TRƯỚC cả smoke test lẫn vòng tải hàng loạt,
    # tránh FileNotFoundError khi ghi file .part.
    os.makedirs(images_dir, exist_ok=True)

    print("=" * 78)
    print(f"ViVQA prepare | out_dir={out_dir} | seed={args.seed} | "
          f"val_ratio={args.val_ratio} | workers={args.workers} | limit={args.limit}")
    print("=" * 78)

    # --- (1) tải CSV -------------------------------------------------------
    print("[1] Tải CSV nguồn...")
    csv_paths: Dict[str, str] = {}
    for name, url in CSV_URLS.items():
        dest = os.path.join(csv_dir, f"{name}.csv")
        download_csv(url, dest)
        csv_paths[name] = dest

    # --- (2) đọc + giải mã type -------------------------------------------
    print("[2] Đọc CSV bằng pandas + giải mã cột 'type'...")
    df_train = load_csv(csv_paths["train"])
    df_test = load_csv(csv_paths["test"])
    if args.limit is not None:
        # Lấy MẪU NGẪU NHIÊN theo seed (thay vì N dòng đầu) để phân bố
        # question_type trải đủ các loại khi smoke-test.
        df_train = df_train.sample(
            n=min(args.limit, len(df_train)), random_state=args.seed
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(args.limit, len(df_test)), random_state=args.seed
        ).reset_index(drop=True)
    print(f"  train.csv: {len(df_train)} dòng, cột={list(df_train.columns)}")
    print(f"  test.csv : {len(df_test)} dòng, cột={list(df_test.columns)}")

    # In phân bố question_type cho người dùng kiểm tra.
    for name, df in (("train", df_train), ("test", df_test)):
        if "type" in df.columns:
            decoded = df["type"].map(decode_type)
            print(f"  [{name}] phân bố question_type:")
            print(decoded.value_counts().to_string())
        else:
            print(f"  [{name}] CẢNH BÁO: không thấy cột 'type'.")

    # --- (3)+(4) sinh record (đã NFC) --------------------------------------
    print("[3/4] Sinh record (NFC + schema)...")
    train_all = build_records(df_train, origin="train")
    test_records = build_records(df_test, origin="test")

    # --- (5) cắt VAL từ train ----------------------------------------------
    print("[5] Cắt VAL từ train...")
    train_records, val_records = split_train_val(train_all, args.val_ratio, args.seed)
    assign_question_ids(train_records, "train")
    assign_question_ids(val_records, "val")
    assign_question_ids(test_records, "test")
    print(f"  train={len(train_records)} | val={len(val_records)} | test={len(test_records)}")

    # --- (6) tải ảnh COCO --------------------------------------------------
    print("[6] Chuẩn bị tải ảnh COCO...")
    all_records = train_records + val_records + test_records
    order_by_id = collect_unique_images(all_records)
    n_unique = len(order_by_id)
    print(f"  Ảnh DUY NHẤT cần tải: {n_unique}")

    if n_unique == 0:
        print("[LỖI] Không có img_id hợp lệ nào — dừng.", file=sys.stderr)
        return 1

    # smoke-test host trước khi tải hàng loạt.
    first_id = next(iter(order_by_id))
    smoke_test_host(first_id, order_by_id[first_id], images_dir)

    id2file, failed = download_all_images(order_by_id, images_dir, args.workers)
    print(f"  Tải được: {len(id2file)} | thất bại: {len(failed)}")

    # ghi log img_id thất bại.
    failed_path = os.path.join(out_dir, "failed_images.txt")
    with open(failed_path, "w", encoding="utf-8") as fh:
        for img_id in failed:
            fh.write(f"{img_id}\n")
    if failed:
        print(f"  Log ảnh lỗi -> {failed_path}")

    # --- (7) điền ảnh + ghi JSON -------------------------------------------
    print("[7] Điền record['image'] + ghi JSON...")
    split_counts: Dict[str, int] = {}
    dropped_total = 0
    for split_name, recs in (("train", train_records),
                             ("val", val_records),
                             ("test", test_records)):
        kept, dropped = attach_images(recs, id2file)
        dropped_total += dropped
        write_json(kept, os.path.join(out_dir, f"{split_name}.json"))
        split_counts[split_name] = len(kept)
        print(f"  {split_name}: giữ {len(kept)} | bỏ (ảnh hỏng) {dropped}")

    # --- (8) prep_report.json ----------------------------------------------
    elapsed = round(time.time() - t0, 2)
    report = {
        "dataset": "vivqa",
        "source": "github.com/kh4nh12/ViVQA (train.csv + test.csv)",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "limit": args.limit,
        "split_record_counts": split_counts,
        "records_dropped_missing_image": dropped_total,
        "images_needed_unique": n_unique,
        "images_downloaded": len(id2file),
        "images_failed": len(failed),
        "elapsed_seconds": elapsed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = os.path.join(out_dir, "prep_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("=" * 78)
    print("REPORT:", json.dumps(report, ensure_ascii=False, indent=2))
    print(f"-> {report_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
