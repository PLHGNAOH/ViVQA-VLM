"""
realphotos_local.py — chạy đúng đường demo local (prepare_image → Backend.ask) trên ảnh thật.

Không đổi pipeline / max_pixels / prompt / decoding / OCR / τ*. Chỉ đo EM + latency và
(tuỳ chọn) chọn bộ demo lên Desktop.

Ví dụ:
  # selftest (không GPU):
  python -m src.eval.realphotos_local --selftest

  # chạy W13 (+ W15 nếu có), ghi summary, dựng demo:
  set OMP_NUM_THREADS=12
  set MKL_NUM_THREADS=12
  python -m src.eval.realphotos_local ^
    --photos_root D:\\vivqa_models\\real_photos ^
    --adapter D:\\vivqa_models\\SH\\epoch_1 ^
    --ocr_device cpu --build-demo
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Paddle gán OMP=1 lúc import nếu chưa có — đặt TRƯỚC mọi import paddle/OCR.
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHOTOS = Path(r"D:\vivqa_models\real_photos")
DEFAULT_ADAPTER = Path(r"D:\vivqa_models\SH\epoch_1")
DEFAULT_EXPECT_SHA = "c9791e961aaa58fdcd7d9baafede3bd7bd8c902acc9cc43917306f1e0fecc354"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "qwen_lora.yaml"
DEFAULT_OUT = REPO_ROOT / "experiments" / "W15_REAL_local_sh"
COLAB_W14 = REPO_ROOT / "experiments" / "W14_REAL_sh_realphotos_v2_bf16_L4" / "realphotos_summary.json"

# Nhãn loại câu (dùng chọn main + ghi chú lỗi) — theo CSV W13 đã biết.
TAG_BY_IMAGE: Dict[str, str] = {
    "xe_buyt_nha_khoa.jpg": "doc_chu_thuong_hieu",
    "bang_huong_dan_y_te.jpg": "doc_chu_nhieu_dau",
    "tui_cua_hang.jpg": "doc_chu_thuong_hieu",
    "cua_hang_dac_san.jpg": "doc_chu_dia_danh",
    "bang_noi_quy_thang_may.jpg": "so_gia",
    "binh_chua_chay.jpg": "doc_chu_thuong_hieu",
    "may_anh_canon.jpg": "yesno_co_vat",
    "chai_xit_con_trung.jpg": "doc_chu_nhieu_dau",
    "bia_sach.jpg": "doc_chu_nhieu_dau",
    "bang_go_trang_tri.jpg": "doc_chu_nhieu_dau",
    "bang_thong_tin_duoi_uoi.jpg": "doc_chu_ten",
    "ga_metro.jpg": "yesno_chu",
    "goi_khan_uot.jpg": "doc_chu_thuong_hieu",
    "hop_dien_thoai.jpg": "yesno_khong_co_vat",
    "anh_meo_trong_khung.jpg": "yesno_khong_co_vat",
    "moc_ao.jpg": "vat_the",
    "bo_ho_nuoc_ban_dem.jpg": "vat_the_mau_sac",
    "dan_meo_con.jpg": "dem_so_nho",
    "duong_pho_chieu_ta.jpg": "yesno_khong_co_vat",
    "canh_xe_dem.jpg": "dem_so_nho",
}


def _split_answers(cell: str) -> List[str]:
    return [a.strip() for a in str(cell).split("|") if a.strip()]


def load_csv(csv_path: Path, image_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            img = (row.get("anh") or "").strip()
            q = (row.get("cau_hoi") or "").strip()
            ans = _split_answers(row.get("dap_an") or "")
            if not img or not q or not ans:
                raise ValueError(f"{csv_path}: dòng {i} thiếu anh/cau_hoi/dap_an")
            path = image_dir / img
            if not path.is_file():
                raise FileNotFoundError(f"không thấy ảnh {path}")
            rows.append({
                "qid": f"{image_dir.name}_{i:03d}",
                "image": img,
                "image_path": str(path),
                "question": q,
                "answers": ans,
                "tag": TAG_BY_IMAGE.get(img, "khac"),
            })
    return rows


def percentile(xs: Sequence[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(float(x) for x in xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def run_rows(backend, rows: List[Dict[str, Any]], tmp_dir: str, set_name: str) -> List[Dict[str, Any]]:
    from src.eval.metrics import exact_match
    from src.prototype.app import prepare_image

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        used, info = prepare_image(row["image_path"], tmp_dir)
        try:
            r = backend.ask(used, row["question"])
        finally:
            if used != row["image_path"] and os.path.exists(used):
                try:
                    os.remove(used)
                except OSError:
                    pass
        pred = str(r.get("answer") or "")
        em = float(exact_match(pred, row["answers"]))
        rec = {
            "set": set_name,
            "qid": row["qid"],
            "image": row["image"],
            "question": row["question"],
            "answers": row["answers"],
            "prediction": pred,
            "em": em,
            "mode": r.get("mode"),
            "level": r.get("level"),
            "confidence": float(r.get("confidence") or 0.0),
            "n_ocr_lines": int(r.get("n_ocr_lines") or len(r.get("ocr_lines") or [])),
            "ocr_texts": list(r.get("ocr_texts") or []),
            "img_size": list(info.get("used_size") or info.get("orig_size") or []),
            "orig_size": list(info.get("orig_size") or []),
            "rotated": bool(info.get("rotated")),
            "resized": bool(info.get("resized")),
            "t_ocr": float(r.get("sec_ocr") or 0.0),
            "t_model": float(r.get("sec_model") or r.get("sec") or 0.0),
            "t_total": float(r.get("sec_total") or 0.0),
            "tag": row["tag"],
        }
        out.append(rec)
        print(f"[{set_name} {i}/{len(rows)}] em={em:.0f} {row['image']}: {pred!r} ({r.get('level')})",
              flush=True)
    return out


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def subset(pred) -> List[Dict[str, Any]]:
        return [r for r in records if pred(r)]

    def em_rate(rs: List[Dict[str, Any]]) -> Optional[float]:
        if not rs:
            return None
        return sum(float(r["em"]) for r in rs) / len(rs)

    by_set = {}
    for s in sorted({r["set"] for r in records}):
        rs = subset(lambda r, s=s: r["set"] == s)
        by_set[s] = {"n": len(rs), "em": em_rate(rs),
                     "em_open": em_rate(subset(lambda r, s=s: r["set"] == s and r["mode"] == "open")),
                     "em_yesno": em_rate(subset(lambda r, s=s: r["set"] == s and r["mode"] == "yesno")),
                     "n_open": len(subset(lambda r, s=s: r["set"] == s and r["mode"] == "open")),
                     "n_yesno": len(subset(lambda r, s=s: r["set"] == s and r["mode"] == "yesno"))}

    levels = Counter(str(r.get("level") or "?") for r in records)
    tots = [float(r["t_total"]) for r in records]
    return {
        "n_total": len(records),
        "em_overall": em_rate(records),
        "by_set": by_set,
        "by_mode": {
            "open": {"n": len(subset(lambda r: r["mode"] == "open")),
                     "em": em_rate(subset(lambda r: r["mode"] == "open"))},
            "yesno": {"n": len(subset(lambda r: r["mode"] == "yesno")),
                      "em": em_rate(subset(lambda r: r["mode"] == "yesno"))},
        },
        "confidence_levels": dict(levels),
        "latency": {
            "t_total_p50": percentile(tots, 50),
            "t_total_p95": percentile(tots, 95),
            "t_ocr_p50": percentile([float(r["t_ocr"]) for r in records], 50),
            "t_model_p50": percentile([float(r["t_model"]) for r in records], 50),
        },
    }


def compare_colab(local_w13: List[Dict[str, Any]], colab_path: Path) -> Dict[str, Any]:
    if not colab_path.is_file():
        return {"available": False, "path": str(colab_path)}
    colab = json.loads(colab_path.read_text(encoding="utf-8"))
    by_img = {r["image"]: r for r in colab.get("records", [])}
    rows = []
    n_match = 0
    for r in local_w13:
        c = by_img.get(r["image"])
        if c is None:
            rows.append({"image": r["image"], "match": False, "reason": "missing_in_colab"})
            continue
        same = str(r["prediction"]).strip() == str(c.get("prediction") or "").strip()
        n_match += int(same)
        rows.append({
            "image": r["image"],
            "question": r["question"],
            "answers": r["answers"],
            "local_pred": r["prediction"],
            "colab_pred": c.get("prediction"),
            "local_em": r["em"],
            "colab_em": c.get("em"),
            "local_level": r["level"],
            "colab_level": c.get("level"),
            "answer_match": same,
        })
    return {
        "available": True,
        "path": str(colab_path),
        "n": len(local_w13),
        "n_answer_match": n_match,
        "rows": rows,
        "mismatches": [x for x in rows if not x.get("answer_match")],
    }


def classify_error(rec: Dict[str, Any]) -> str:
    """Loại lỗi ngắn cho gioi_han (chỉ khi EM=0)."""
    if float(rec.get("em") or 0) >= 1.0:
        return ""
    tag = rec.get("tag") or ""
    pred = str(rec.get("prediction") or "").lower()
    golds = [str(g).lower() for g in rec.get("answers") or []]
    ocr = " ".join(str(t) for t in rec.get("ocr_texts") or []).lower()
    if tag.startswith("dem_"):
        return "đếm"
    if tag.startswith("yesno"):
        return "lệch đáp án"
    # OCR distractor heuristic: prediction appears in OCR but not gold-ish
    if any(p in ocr for p in pred.split() if len(p) >= 3) and not any(g in pred for g in golds):
        if tag.startswith("doc_chu") or tag.startswith("so_"):
            return "OCR đọc sai" if not any(g.replace(" ", "") in ocr.replace(" ", "") for g in golds) else "chữ gây nhiễu"
    if tag.startswith("doc_chu") or tag.startswith("so_"):
        # gold fragment missing from OCR → thiếu chữ / OCR sai
        joined = ocr.replace(" ", "")
        if not any(g.replace(" ", "")[:4] in joined for g in golds if len(g) >= 3):
            return "thiếu chữ"
        return "OCR đọc sai"
    return "lệch đáp án"


def _pick_main(correct: List[Dict[str, Any]], k_min: int = 6, k_max: int = 8) -> List[Dict[str, Any]]:
    """Chọn 6–8 ảnh đúng, phủ đủ loại; ưu tiên Cao/Trung bình và câu không-đọc-chữ ít chữ lạ."""
    def score(r: Dict[str, Any]) -> Tuple[int, int, float]:
        lvl = {"Cao": 2, "Trung bình": 1, "Thấp": 0}.get(str(r.get("level")), 0)
        # ưu tiên ảnh nhóm tự chụp: hầu hết W13; phạt nhẹ ảnh mèo khung nếu có chữ nhiễu nhiều
        n_ocr = int(r.get("n_ocr_lines") or 0)
        no_read = 1 if r.get("tag") in ("vat_the", "vat_the_mau_sac", "dem_so_nho", "yesno_khong_co_vat") else 0
        clean = 1 if (no_read and n_ocr <= 2) else 0
        return (lvl, clean, -float(r.get("t_total") or 0))

    buckets = {
        "doc_chu": [],
        "doc_chu_dau": [],
        "so_gia": [],
        "yesno_khong": [],
        "vat": [],
        "dem": [],
        "other": [],
    }
    for r in correct:
        tag = r.get("tag") or ""
        if tag == "doc_chu_nhieu_dau":
            buckets["doc_chu_dau"].append(r)
        elif tag.startswith("doc_chu"):
            buckets["doc_chu"].append(r)
        elif tag == "so_gia":
            buckets["so_gia"].append(r)
        elif tag == "yesno_khong_co_vat":
            buckets["yesno_khong"].append(r)
        elif tag in ("vat_the", "vat_the_mau_sac"):
            buckets["vat"].append(r)
        elif tag == "dem_so_nho":
            buckets["dem"].append(r)
        else:
            buckets["other"].append(r)

    for b in buckets.values():
        b.sort(key=score, reverse=True)

    chosen: List[Dict[str, Any]] = []
    used = set()

    def take(pool: List[Dict[str, Any]], n: int = 1) -> int:
        got = 0
        for r in pool:
            if got >= n:
                break
            if r["image"] in used:
                continue
            chosen.append(r)
            used.add(r["image"])
            got += 1
        return got

    def n_doc() -> int:
        return sum(1 for r in chosen if str(r.get("tag", "")).startswith("doc_chu"))

    # Bắt buộc phủ loại
    take(buckets["doc_chu_dau"], 1)
    take(buckets["doc_chu"], 2)
    if n_doc() < 3:
        take(buckets["doc_chu_dau"] + buckets["doc_chu"] + buckets["other"], 3 - n_doc())
    take(buckets["so_gia"], 1)
    take(buckets["yesno_khong"], 1)
    take(buckets["vat"], 1)
    take(buckets["dem"], 1)

    rest = sorted([r for r in correct if r["image"] not in used], key=score, reverse=True)
    for r in rest:
        if len(chosen) >= k_max:
            break
        if len(chosen) >= k_min and score(r)[0] < 1:
            continue
        chosen.append(r)
        used.add(r["image"])
    for r in rest:
        if len(chosen) >= k_min:
            break
        if r["image"] not in used:
            chosen.append(r)
            used.add(r["image"])
    return chosen[:k_max]


def build_demo(records: List[Dict[str, Any]], photos_root: Path, desktop_demo: Path) -> Dict[str, Any]:
    correct = [r for r in records if float(r["em"]) >= 1.0]
    wrong = [r for r in records if float(r["em"]) < 1.0]
    main = _pick_main(correct)
    main_imgs = {r["image"] for r in main}
    du_phong = [r for r in correct if r["image"] not in main_imgs]
    gioi_han = wrong

    for sub in ("main", "du_phong", "gioi_han"):
        p = desktop_demo / sub
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    def copy_named(rs: List[Dict[str, Any]], sub: str, numbered: bool) -> List[Dict[str, Any]]:
        out = []
        for i, r in enumerate(rs, 1):
            src = photos_root / r["set"] / r["image"]
            if not src.is_file():
                # set name may be w13/w15
                alt = list(photos_root.glob(f"*/{r['image']}"))
                src = alt[0] if alt else src
            stem = Path(r["image"]).stem
            name = f"{i:02d}_{stem}.jpg" if numbered else r["image"]
            dst = desktop_demo / sub / name
            shutil.copy2(src, dst)
            err = classify_error(r) if sub == "gioi_han" else ""
            out.append({**r, "demo_file": name, "demo_subdir": sub, "error_type": err})
            # also write sidecar txt with question for convenience
            (desktop_demo / sub / (Path(name).stem + ".txt")).write_text(
                f"Câu hỏi: {r['question']}\nĐáp án: {' | '.join(r['answers'])}\n"
                f"Local: {r['prediction']}\nMức: {r['level']}\nEM: {r['em']}\n",
                encoding="utf-8")
        return out

    main_rows = copy_named(main, "main", True)
    du_rows = copy_named(du_phong, "du_phong", False)
    gh_rows = copy_named(gioi_han, "gioi_han", False)

    criteria = [
        "Chỉ chọn từ kết quả local (đúng đường demo Backend.ask).",
        "main: 6–8 ảnh EM=1; ưu tiên mức Cao/Trung bình; ưu tiên ảnh nhóm tự chụp.",
        "Phủ loại: ≥3 đọc chữ (tên/thương hiệu, ≥1 nhiều dấu thanh), ≥1 số/giá, "
        "≥1 có/không với vật không có trong ảnh, ≥1 vật thể/màu sắc, ≥1 đếm số nhỏ.",
        "Ưu tiên câu không-đọc-chữ khi ảnh ít chữ lạ (OCR lines ≤ 2).",
        "du_phong: ảnh đúng còn lại; gioi_han: ảnh sai (giải thích nếu hội đồng hỏi).",
        "Không xoá câu sai khỏi báo cáo experiments — chỉ tách thư mục demo.",
    ]

    sheet_rows = main_rows + du_rows + gh_rows
    md = _cheatsheet_md(sheet_rows, criteria, summarize(records))
    html = _cheatsheet_html(sheet_rows, criteria, summarize(records))
    (desktop_demo / "demo_cheatsheet.md").write_text(md, encoding="utf-8")
    (desktop_demo / "demo_cheatsheet.html").write_text(html, encoding="utf-8")
    return {
        "desktop": str(desktop_demo),
        "main": [{"file": r["demo_file"], "image": r["image"], "question": r["question"],
                  "prediction": r["prediction"], "level": r["level"], "tag": r["tag"]} for r in main_rows],
        "du_phong": [r["demo_file"] for r in du_rows],
        "gioi_han": [{"file": r["demo_file"], "prediction": r["prediction"],
                      "answers": r["answers"], "error_type": r["error_type"],
                      "question": r["question"]} for r in gh_rows],
        "criteria": criteria,
    }


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}%"


def _cheatsheet_md(rows: List[Dict[str, Any]], criteria: List[str], summ: Dict[str, Any]) -> str:
    lines = [
        "# Demo cheat sheet — ViVQA-VLM (local SH)",
        "",
        f"Tạo: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        "",
        "## Tiêu chí chọn",
        *[f"- {c}" for c in criteria],
        "",
        f"EM tổng (local): {_fmt_pct(summ.get('em_overall'))} · "
        f"latency t_total p50={summ['latency']['t_total_p50']:.2f}s "
        f"p95={summ['latency']['t_total_p95']:.2f}s" if summ.get("latency", {}).get("t_total_p50") is not None else "",
        "",
        "| # | Thư mục | File | Câu hỏi | Đáp án mong đợi | Local | Mức | t_total (s) | Ghi chú |",
        "|---:|---|---|---|---|---|---|---:|---|",
    ]
    for i, r in enumerate(rows, 1):
        note = r.get("error_type") or r.get("tag") or ""
        gold = " \\| ".join(r["answers"])
        lines.append(
            f"| {i} | {r['demo_subdir']} | `{r['demo_file']}` | {r['question']} | "
            f"{gold} | {r['prediction']} | {r['level']} | "
            f"{float(r['t_total']):.2f} | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


def _cheatsheet_html(rows: List[Dict[str, Any]], criteria: List[str], summ: Dict[str, Any]) -> str:
    import html as html_lib
    crit = "".join(f"<li>{html_lib.escape(c)}</li>" for c in criteria)
    body_rows = []
    for i, r in enumerate(rows, 1):
        note = r.get("error_type") or r.get("tag") or ""
        cls = "ok" if r["demo_subdir"] != "gioi_han" else "bad"
        body_rows.append(
            "<tr class='{cls}'><td>{i}</td><td>{sub}</td><td>{fn}</td><td>{q}</td>"
            "<td>{g}</td><td>{p}</td><td>{lv}</td><td>{t:.2f}</td><td>{n}</td></tr>".format(
                cls=cls, i=i, sub=html_lib.escape(r["demo_subdir"]),
                fn=html_lib.escape(r["demo_file"]), q=html_lib.escape(r["question"]),
                g=html_lib.escape(" | ".join(r["answers"])),
                p=html_lib.escape(str(r["prediction"])),
                lv=html_lib.escape(str(r["level"])),
                t=float(r["t_total"]), n=html_lib.escape(str(note)),
            )
        )
    lat = summ.get("latency") or {}
    return f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"/>
<title>Demo cheat sheet — ViVQA-VLM</title>
<style>
@page {{ size: A4; margin: 12mm; }}
body {{ font-family: Segoe UI, Arial, sans-serif; font-size: 11px; color: #111; }}
h1 {{ font-size: 16px; margin: 0 0 6px; }}
ul {{ margin: 4px 0 10px 18px; padding: 0; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 3px 5px; vertical-align: top; }}
th {{ background: #f3f3f3; }}
tr.bad {{ background: #fff1f0; }}
tr.ok {{ background: #f6ffed; }}
.meta {{ color: #444; margin-bottom: 8px; }}
</style></head><body>
<h1>Demo cheat sheet — ViVQA-VLM (local SH)</h1>
<div class="meta">EM tổng: {_fmt_pct(summ.get('em_overall'))} ·
t_total p50={lat.get('t_total_p50') and f"{lat['t_total_p50']:.2f}s" or "—"} ·
p95={lat.get('t_total_p95') and f"{lat['t_total_p95']:.2f}s" or "—"}</div>
<strong>Tiêu chí chọn</strong>
<ul>{crit}</ul>
<table>
<thead><tr><th>#</th><th>Thư mục</th><th>File</th><th>Câu hỏi</th><th>Đáp án</th>
<th>Local</th><th>Mức</th><th>t_total</th><th>Ghi chú</th></tr></thead>
<tbody>
{''.join(body_rows)}
</tbody></table>
</body></html>
"""


def write_summary(out_dir: Path, records: List[Dict[str, Any]], summ: Dict[str, Any],
                  cmp_w13: Dict[str, Any], demo_info: Optional[Dict[str, Any]],
                  skipped: Dict[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "skipped": skipped,
        "summary": summ,
        "w13_vs_colab": cmp_w13,
        "demo": {k: v for k, v in (demo_info or {}).items() if k != "criteria"} if demo_info else None,
        "records": [{k: v for k, v in r.items() if k != "image_path"} for r in records],
    }
    (out_dir / "realphotos_local_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Real photos — local SH (demo path)",
        "",
        f"- Created: {payload['created_utc']}",
        f"- OMP/MKL threads: {payload['omp_num_threads']}/{payload['mkl_num_threads']}",
        f"- N total: {summ['n_total']}",
        f"- EM overall: {_fmt_pct(summ['em_overall'])}",
        "",
        "## EM by set",
        "",
        "| Set | N | EM | EM open | EM yesno |",
        "|---|---:|---:|---:|---:|",
    ]
    for s, v in summ["by_set"].items():
        lines.append(
            f"| {s} | {v['n']} | {_fmt_pct(v['em'])} | {_fmt_pct(v['em_open'])} | {_fmt_pct(v['em_yesno'])} |"
        )
    lines += [
        "",
        "## EM by mode (all sets)",
        "",
        f"- open: n={summ['by_mode']['open']['n']} EM={_fmt_pct(summ['by_mode']['open']['em'])}",
        f"- yesno: n={summ['by_mode']['yesno']['n']} EM={_fmt_pct(summ['by_mode']['yesno']['em'])}",
        "",
        "## Confidence levels",
        "",
        ", ".join(f"{k}: {v}" for k, v in summ["confidence_levels"].items()),
        "",
        "## Latency",
        "",
        f"- t_total p50={summ['latency']['t_total_p50']:.3f}s  p95={summ['latency']['t_total_p95']:.3f}s",
        f"- t_ocr p50={summ['latency']['t_ocr_p50']:.3f}s  t_model p50={summ['latency']['t_model_p50']:.3f}s",
        "",
    ]
    if skipped:
        lines += ["## Skipped", ""] + [f"- {k}: {v}" for k, v in skipped.items()] + [""]
    if cmp_w13.get("available"):
        lines += [
            "## W13 local vs Colab W14 (answer string match)",
            "",
            f"- Match: {cmp_w13['n_answer_match']}/{cmp_w13['n']}",
            "",
            "| Image | Local | Colab | Match | Local EM | Colab EM |",
            "|---|---|---|---|---:|---:|",
        ]
        for row in cmp_w13["rows"]:
            lines.append(
                f"| {row['image']} | {row.get('local_pred')} | {row.get('colab_pred')} | "
                f"{'yes' if row.get('answer_match') else 'NO'} | {row.get('local_em')} | {row.get('colab_em')} |"
            )
        if cmp_w13.get("mismatches"):
            lines += ["", "### Mismatches", ""]
            for m in cmp_w13["mismatches"]:
                lines.append(f"- {m['image']}: local={m.get('local_pred')!r} colab={m.get('colab_pred')!r}")
    lines += ["", "## Per-question", "",
              "| Set | Image | Q | Gold | Pred | EM | Mode | Level | t_total |",
              "|---|---|---|---|---|---:|---|---|---:|"]
    for r in records:
        gold = " \\| ".join(r["answers"])
        lines.append(
            f"| {r['set']} | {r['image']} | {r['question']} | {gold} | "
            f"{r['prediction']} | {r['em']:.0f} | {r['mode']} | {r['level']} | {r['t_total']:.2f} |"
        )
    (out_dir / "realphotos_local_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selftest() -> None:
    from PIL import Image
    from src.eval.metrics import exact_match
    from src.prototype.app import MockBackend, load_ocr_selection, load_ui_settings, prepare_image

    assert exact_match("7-eleven", ["7-eleven", "7 eleven"]) == 1.0
    assert exact_match("eleven", ["7-eleven", "7 eleven"]) == 0.0
    assert exact_match("1150 kg", ["1150kg", "1150 kg"]) == 1.0

    st, ocr_cfg = load_ui_settings(), load_ocr_selection()
    backend = MockBackend(st, ocr_cfg)
    with tempfile.TemporaryDirectory() as td:
        img = Path(td) / "a.jpg"
        Image.new("RGB", (640, 480), "white").save(img)
        csv_path = Path(td) / "cau_hoi.csv"
        csv_path.write_text(
            "anh,cau_hoi,dap_an\n"
            "a.jpg,Cửa hàng tên gì?,cửa hàng mẫu\n"
            "a.jpg,Trong ảnh có con chó không?,không|no\n",
            encoding="utf-8",
        )
        rows = load_csv(csv_path, Path(td))
        assert len(rows) == 2 and rows[0]["answers"] == ["cửa hàng mẫu"]
        # warm-up + run
        used, _ = prepare_image(str(img), td)
        _ = backend.ask(used, "warmup?")
        recs = run_rows(backend, rows, td, "selftest")
        assert len(recs) == 2
        assert recs[0]["mode"] == "open" and recs[1]["mode"] == "yesno"
        assert all(k in recs[0] for k in ("em", "level", "t_ocr", "t_model", "t_total", "n_ocr_lines", "img_size"))
        # EM uses metrics.exact_match
        assert recs[0]["em"] == exact_match(recs[0]["prediction"], rows[0]["answers"])
        summ = summarize(recs)
        assert summ["n_total"] == 2
    print("SELFTEST PASS", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--photos_root", type=Path, default=DEFAULT_PHOTOS)
    ap.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--expect_adapter_sha", default=DEFAULT_EXPECT_SHA)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--ocr_device", default="cpu")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--colab_summary", type=Path, default=COLAB_W14)
    ap.add_argument("--build-demo", action="store_true")
    ap.add_argument("--demo_dir", type=Path, default=None,
                    help="mặc định: Desktop\\demo_images")
    ap.add_argument("--sets", nargs="*", default=["w13", "w15"])
    ap.add_argument("--mock", action="store_true", help="backend giả (không GPU)")
    a = ap.parse_args(list(argv) if argv is not None else None)

    if a.selftest:
        selftest()
        return 0

    skipped: Dict[str, str] = {}
    all_rows: List[Tuple[str, List[Dict[str, Any]]]] = []
    for s in a.sets:
        d = a.photos_root / s
        csv_path = d / "cau_hoi.csv"
        if not d.is_dir() or not csv_path.is_file():
            skipped[s] = f"thiếu thư mục hoặc cau_hoi.csv ({d})"
            print(f"SKIP {s}: {skipped[s]}", flush=True)
            continue
        rows = load_csv(csv_path, d)
        all_rows.append((s, rows))
        print(f"LOAD {s}: {len(rows)} câu", flush=True)

    if not all_rows:
        raise SystemExit("Không có tập nào để chạy")

    from src.prototype.app import MockBackend, load_ocr_selection, load_ui_settings, make_real_backend, prepare_image

    st, ocr_cfg = load_ui_settings(), load_ocr_selection()
    client = None
    if a.mock:
        backend = MockBackend(st, ocr_cfg)
        start = {"mock": True}
    else:
        backend, client, start = make_real_backend(
            str(a.adapter), str(a.config), st, ocr_cfg,
            ocr_device=a.ocr_device, expect_adapter_sha=a.expect_adapter_sha or None,
        )
    print(json.dumps({"startup": {k: start.get(k) for k in ("sec_start_total", "mock", "adapter") if k in start}},
                     ensure_ascii=False), flush=True)

    tmp = tempfile.mkdtemp(prefix="vivqa_realphotos_")
    records: List[Dict[str, Any]] = []
    try:
        # warm-up 1 câu
        warm_img = all_rows[0][1][0]["image_path"]
        used, _ = prepare_image(warm_img, tmp)
        _ = backend.ask(used, all_rows[0][1][0]["question"])
        if used != warm_img and os.path.exists(used):
            try:
                os.remove(used)
            except OSError:
                pass
        print("WARMUP done", flush=True)

        for set_name, rows in all_rows:
            records.extend(run_rows(backend, rows, tmp, set_name))
    finally:
        if client is not None:
            client.close()
        shutil.rmtree(tmp, ignore_errors=True)

    summ = summarize(records)
    local_w13 = [r for r in records if r["set"] == "w13"]
    cmp_w13 = compare_colab(local_w13, a.colab_summary) if local_w13 else {"available": False}

    demo_info = None
    if a.build_demo:
        demo_dir = a.demo_dir
        if demo_dir is None:
            # Desktop thật của Windows
            try:
                import subprocess
                desk = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command",
                     "[Environment]::GetFolderPath('Desktop')"],
                    text=True,
                ).strip()
            except Exception:
                desk = str(Path.home() / "Desktop")
            demo_dir = Path(desk) / "demo_images"
        demo_info = build_demo(records, a.photos_root, demo_dir)
        print(f"DEMO -> {demo_dir}", flush=True)

    write_summary(a.out_dir, records, summ, cmp_w13, demo_info, skipped)
    print(f"SUMMARY -> {a.out_dir}", flush=True)
    print(json.dumps({
        "n": summ["n_total"],
        "em": summ["em_overall"],
        "by_set": summ["by_set"],
        "w13_match_colab": f"{cmp_w13.get('n_answer_match')}/{cmp_w13.get('n')}" if cmp_w13.get("available") else None,
        "skipped": skipped,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
