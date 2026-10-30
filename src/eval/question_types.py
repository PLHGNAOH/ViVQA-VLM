"""
question_types.py — Bộ phân loại câu hỏi ViTextVQA theo LUẬT TỪ KHOÁ (W11, bước 6) + bảng EM/ANLS theo loại.

Vấn đề: master yêu cầu error analysis theo loại câu hỏi, nhưng ViTextVQA không có nhãn loại (`question_type` =
"unknown" ở mọi run). Luật ghi trước (docs/W11/README.md, "Luật bộ phân loại câu hỏi ViTextVQA"):
    1. luật từ khoá, thiết kế CHỈ trên câu hỏi dev (dev300, không nhìn test, không nhìn dự đoán);
    2. ĐÓNG BĂNG (commit) trước khi tính bất kỳ số nào theo loại trên test — phiên bản: QT_VERSION;
    3. kiểm bằng 150 câu test2000 (seed 42) Hoàng gán nhãn tay, KHÔNG xem dự đoán và KHÔNG xem nhãn máy;
    4. loại < 50 câu trên test2000 -> gộp vào "other";
    5. bảng theo loại chỉ để mô tả / error analysis, không dùng chọn mô hình.

Loại (thứ tự = ưu tiên: câu khớp nhiều loại -> lấy loại đứng TRƯỚC):
    source          nguồn ảnh: báo, kênh, trang web (watermark)        "bức ảnh này lấy nguồn từ đâu ?"
    contact         số điện thoại, hotline, fax, zalo, website, email   "số điện thoại của quán này là gì ?"
    number_time     giá, số lượng, số nhà, biển số, ngày/năm/giờ, lần thứ  "dưa hấu có giá bao nhiêu ?"
    person          tên người: tác giả, người dịch, "ai"                 "tác giả của cuốn sách là ai ?"
    location        địa chỉ, ở đâu, tỉnh/quận/... nào, hướng đi           "địa chỉ quán chè hẻm là gì ?"
    name            tên riêng của cửa hàng / sản phẩm / tổ chức / sách / thương hiệu  "cửa hàng này có tên là gì ?"
    text_content    đọc một dòng chữ / nội dung / khẩu hiệu / thông báo   "dòng chữ trên bảng đen là gì ?"
    product_service bán gì, dịch vụ gì, kinh doanh gì, loại cửa hàng      "cửa hàng này bán gì ?"
    other           còn lại (sự kiện, mô tả, suy luận, "như thế nào")     "cuốn sách viết nhằm mục đích gì ?"

Chạy:
    python -m src.eval.question_types --make_sheet --test2000_run <exp>/W07_B1_..._test2000_... --out docs/W11/qt_label_sheet_150.csv
    python -m src.eval.question_types --score <labels.csv> --test2000_run <...> --out <dir>/qt_label_check.json
    python -m src.eval.question_types --table --exp_root experiments [--labels <labels.csv>] --out experiments/W11_question_types
Kiểm thử offline:
    python -m src.eval.question_types --selftest
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

QT_VERSION = "qt-v1 (W11, 21/10/2026, thiết kế trên dev300)"
SHEET_N, SHEET_SEED = 150, 42
MIN_N_TYPE = 50              # luật 4: loại < 50 câu trên test2000 -> gộp vào "other"
MIN_PRECISION = 0.80         # luật 3: độ chính xác < 80% -> gắn cờ "nhãn kém tin cậy"
MIN_N_FLAG = 5               # (ghi trước khi có nhãn) loại có < 5 câu trong mẫu 150 -> ghi "n quá nhỏ", không kết luận cờ
OTHER = "other"

# (loại, [regex]) — áp trên câu hỏi đã chuẩn hoá (NFC, chữ thường, gộp khoảng trắng). ĐÓNG BĂNG từ qt-v1.
RULES: List[Tuple[str, List[str]]] = [
    ("source", [r"\bnguồn\b", r"lấy từ", r"trích từ", r"\bbáo nào\b", r"tờ báo", r"\bkênh\b",
                r"\b(trang )?web nào\b", r"đăng tải"]),
    ("contact", [r"số điện thoại", r"\bhotline\b", r"\bfax\b", r"\bzalo\b", r"\bwebsite\b", r"\bemail\b",
                 r"\bsđt\b", r"liên (hệ|lạc) .*\bsố\b"]),
    ("number_time", [r"bao nhiêu", r"\bmấy\b", r"(?<!giảm )(?<!đánh )\bgiá\b", r"lần thứ", r"phần trăm",
                     r"\bnăm nào\b", r"\bngày nào\b", r"\bsố nhà\b", r"biển số", r"mã số", r"giấy phép",
                     r"dung tích", r"\bsố áo\b", r"\bsố cuối\b", r"\bkhi nào\b", r"\bmấy giờ\b"]),
    ("person", [r"\bai\b", r"tác giả", r"người dịch", r"chủ biên"]),
    ("location", [r"ở đâu", r"tại đâu", r"\bđịa chỉ\b", r"\bđỉa chỉ\b",
                  r"\b(tỉnh|quận|huyện|thành phố|tp|xã|phường|ấp|nước|quốc gia|đường|ngõ|hẻm|nơi) nào\b",
                  r"nằm ở", r"(đến|tới) đâu", r"trước đâu", r"\bhướng\b"]),
    ("name", [r"\btên\b", r"tựa đề", r"\bhãng nào\b", r"thương hiệu", r"nhà xuất bản",
              r"\b(công ty|siêu thị|cửa hàng|quán|chợ|trường|tạp hóa|cơ quan|sự kiện|chương trình|đại lý|"
              r"ngân hàng|bệnh viện|nhà thuốc) nào\b"]),
    ("text_content", [r"dòng chữ", r"\bchữ\b", r"nội dung", r"nói (gì|đến|về|tới)", r"\bghi\b", r"slogan",
                      r"khẩu hiệu", r"thông báo gì", r"yêu cầu", r"\bcấm\b", r"hỏi (gì|về)"]),
    ("product_service", [r"\bbán\b", r"kinh doanh", r"kinh doang", r"dịch vụ", r"cho thuê", r"sản xuất",
                         r"\bchuyên\b", r"\bmón\b",
                         r"\b(cửa hàng|quán|tiệm|nhà hàng|shop|siêu thị|công ty|cửa tiệm|quầy|gian hàng) gì\b",
                         r"làm (gì|công việc)"]),
]
TYPES = [t for t, _ in RULES] + [OTHER]
_COMPILED = [(t, [re.compile(p) for p in pats]) for t, pats in RULES]


def rules_sha256() -> str:
    """Dấu vân tay của bộ luật (ghi vào mọi bảng kết quả -> biết bảng tính bằng luật nào)."""
    return hashlib.sha256(json.dumps(RULES, ensure_ascii=False).encode("utf-8")).hexdigest()


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(q)).lower()).strip()


def classify(question: str) -> str:
    n = norm_q(question)
    for t, pats in _COMPILED:
        if any(p.search(n) for p in pats):
            return t
    return OTHER


def _load(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# =============================================================================
# Phiếu gán nhãn tay (150 câu test2000, seed 42) — CHỈ có câu hỏi, không dự đoán, không nhãn máy
# =============================================================================
def sample_ids(qids: List[str], n: int = SHEET_N, seed: int = SHEET_SEED) -> List[str]:
    return random.Random(seed).sample(sorted(qids), n)


def make_sheet(records: List[Dict[str, Any]], out_csv: str) -> List[Dict[str, str]]:
    by_id = {r["question_id"]: r["question"] for r in records}
    rows = [{"stt": str(i), "question_id": q, "question": by_id[q], "label": "", "note": ""}
            for i, q in enumerate(sample_ids(list(by_id)), 1)]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stt", "question_id", "question", "label", "note"])
        w.writeheader()
        w.writerows(rows)
    return rows


def read_labels(path: str) -> Dict[str, str]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        lab = (r.get("label") or "").strip()
        if lab not in TYPES:
            raise ValueError(f"Nhãn không hợp lệ ở {r.get('question_id')}: {lab!r} (hợp lệ: {TYPES})")
        out[r["question_id"]] = lab
    if len(out) != SHEET_N:
        raise ValueError(f"Cần {SHEET_N} nhãn, có {len(out)}")
    return out


def score(labels: Dict[str, str], questions: Dict[str, str]) -> Dict[str, Any]:
    """So nhãn máy với nhãn tay. Độ chính xác theo loại = precision (trong các câu MÁY gán loại T, bao nhiêu % người
    cũng gán T) — đúng thứ ảnh hưởng tới bảng theo loại (bảng nhóm câu theo nhãn máy). Kèm recall để tham khảo."""
    expected = set(sample_ids(list(questions)))
    if set(labels) != expected:
        raise ValueError("Phiếu nhãn không đúng 150 câu đã chọn (seed 42) — sai file hoặc bị sửa question_id")
    auto = {q: classify(questions[q]) for q in labels}
    n_ok = sum(auto[q] == labels[q] for q in labels)
    per = {}
    for t in TYPES:
        a = [q for q in labels if auto[q] == t]
        h = [q for q in labels if labels[q] == t]
        prec = sum(labels[q] == t for q in a) / len(a) if a else None
        rec = sum(auto[q] == t for q in h) / len(h) if h else None
        flag = ("n quá nhỏ" if len(a) < MIN_N_FLAG else
                "NHÃN KÉM TIN CẬY" if prec is not None and prec < MIN_PRECISION else "ok")
        per[t] = {"n_auto": len(a), "n_human": len(h), "precision": prec, "recall": rec, "flag": flag}
    confusion = collections.Counter(f"{labels[q]} -> {auto[q]}" for q in labels if auto[q] != labels[q])
    return {"qt_version": QT_VERSION, "rules_sha256": rules_sha256(), "n": len(labels),
            "accuracy": n_ok / len(labels), "per_type": per,
            "confusion_human_to_auto": dict(confusion.most_common())}


# =============================================================================
# Bảng EM / ANLS theo loại (chỉ mô tả)
# =============================================================================
def merged_types(counts: Dict[str, int], min_n: int = MIN_N_TYPE) -> Dict[str, str]:
    """loại -> loại báo cáo (loại < min_n câu trên test2000 gộp vào other)."""
    return {t: (t if counts.get(t, 0) >= min_n or t == OTHER else OTHER) for t in TYPES}


def per_type_scores(records: List[Dict[str, Any]], tmap: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    from src.eval.compare_runs import per_sample_scores
    sc = per_sample_scores(records)
    acc: Dict[str, List[Tuple[float, float]]] = collections.defaultdict(list)
    for r, s in zip(records, sc):
        t = tmap[classify(r["question"])]
        acc[t].append((s["exact_match"], s["anls"]))
        acc["ALL"].append((s["exact_match"], s["anls"]))
    return {t: {"n": len(v), "em": sum(x[0] for x in v) / len(v), "anls": sum(x[1] for x in v) / len(v)}
            for t, v in acc.items()}


def build_table(exp_root: str, runs: List[Dict[str, Any]], flags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """runs: [{id, section ('vitextvqa_test2000'|'vitextvqa_testfull'), eval}]. Loại gộp tính trên test2000."""
    recs = {(r["section"], r["id"]): _load(os.path.join(exp_root, r["eval"], "predictions.json")) for r in runs}
    t2000 = next(v for (sec, _), v in recs.items() if sec == "vitextvqa_test2000")
    counts = collections.Counter(classify(r["question"]) for r in t2000)
    tmap = merged_types(counts)
    out = {"qt_version": QT_VERSION, "rules_sha256": rules_sha256(), "counts_test2000_raw": dict(counts),
           "type_map": tmap, "label_flags": flags or {}, "sections": {}}
    for (sec, rid), rr in recs.items():
        out["sections"].setdefault(sec, {})[rid] = per_type_scores(rr, tmap)
    return out


def to_markdown(res: Dict[str, Any]) -> str:
    L = ["# Per-question-type results — ViTextVQA (W11, descriptive only)", "",
         f"Classifier `{res['qt_version']}`, rules sha256 `{res['rules_sha256'][:12]}`. Types with < {MIN_N_TYPE} "
         f"questions on test2000 merged into `other`. EM % (ANLS % in brackets).", ""]
    titles = {"vitextvqa_testfull": "Full test (10,028 q)", "vitextvqa_test2000": "test2000 (2,000 q)"}
    for sec in ("vitextvqa_testfull", "vitextvqa_test2000"):
        runs = res["sections"].get(sec)
        if not runs:
            continue
        first = next(iter(runs.values()))
        types = [t for t in TYPES if t in first] + ["ALL"]
        L += [f"## {titles[sec]}", "",
              "| Run | " + " | ".join(f"{t} (n={first[t]['n']}){' ⚠' if res['label_flags'].get(t) == 'NHÃN KÉM TIN CẬY' else ''}"
                                     for t in types) + " |",
              "|---|" + "---|" * len(types)]
        for rid, v in runs.items():
            L.append(f"| {rid} | " + " | ".join(f"{100 * v[t]['em']:.1f} ({100 * v[t]['anls']:.1f})" for t in types) + " |")
        L.append("")
    if res["label_flags"]:
        L += ["⚠ = type flagged by the 150-question manual check (precision < 80%).", ""]
    return "\n".join(L)


def default_runs() -> List[Dict[str, Any]]:
    from src.eval.full_eval_table import RUNS
    return [{"id": r["id"], "section": r["section"], "eval": r["eval"]} for r in RUNS
            if r["section"] in ("vitextvqa_test2000", "vitextvqa_testfull")]


# =============================================================================
# Selftest
# =============================================================================
def _selftest() -> None:
    cases = {
        "bức ảnh này lấy nguồn từ đâu ?": "source",
        "số điện thoại của quán cắt tóc này là gì ?": "contact",
        "tên website của dien may xanh ?": "contact",
        "dưa hấu ở siêu thị này có giá bao nhiêu ?": "number_time",
        "chương trình giảm giá này diễn ra ở đâu ?": "location",      # "giảm giá" không phải hỏi giá
        "sản phẩm này được đánh giá như thế nào ?": OTHER,
        "tác giả của cuốn sách này là ai ?": "person",
        "tên tác giả cuốn sách là gì ?": "person",
        "địa chỉ quán chè hẻm là gì ?": "location",
        "cửa hàng này có tên là gì ?": "name",
        "kệ sữa trong hình là của hãng nào ?": "name",
        "cửa tiệm bảng hiệu đen chữ trắng trong ảnh tên là gì ?": "name",   # có "chữ" nhưng hỏi TÊN
        "dòng chữ ghi trên bảng đen là gì ?": "text_content",
        "cửa hàng này bán cái gì ?": "product_service",
        "cuốn sách này được viết nhằm mục đích gì ?": OTHER,
        "Cửa  Hàng   NÀY có TÊN là gì ?": "name",                       # chuẩn hoá hoa/thường + khoảng trắng
    }
    for q, want in cases.items():
        assert classify(q) == want, (q, classify(q), want)
    assert TYPES[-1] == OTHER and len(set(TYPES)) == len(TYPES)
    # gộp loại ít câu
    m = merged_types({"name": 400, "person": 30, OTHER: 10})
    assert m["name"] == "name" and m["person"] == OTHER and m[OTHER] == OTHER
    # phiếu + chấm: tái lập, đúng 150 câu, nhãn lạ -> lỗi, câu lạ -> lỗi
    import tempfile
    recs = [{"question_id": f"t{i}", "question": q} for i, q in enumerate(list(cases) * 12)]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.csv")
        rows = make_sheet(recs, p)
        assert len(rows) == SHEET_N and [r["question_id"] for r in rows] == sample_ids([r["question_id"] for r in recs])
        assert all(r["label"] == "" for r in rows) and "label" in open(p, encoding="utf-8-sig").readline()
        qs = {r["question_id"]: r["question"] for r in recs}
        with open(p, encoding="utf-8-sig", newline="") as fh:
            filled = list(csv.DictReader(fh))
        for r in filled:
            r["label"] = classify(r["question"])
        filled[0]["label"] = "person" if filled[0]["label"] != "person" else "name"     # 1 câu bất đồng
        with open(p, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filled[0]))
            w.writeheader()
            w.writerows(filled)
        s = score(read_labels(p), qs)
        assert abs(s["accuracy"] - 149 / 150) < 1e-12 and sum(s["confusion_human_to_auto"].values()) == 1
        filled[1]["label"] = "abc"
        with open(p, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filled[0]))
            w.writeheader()
            w.writerows(filled)
        try:
            read_labels(p)
            raise AssertionError("không chặn nhãn lạ")
        except ValueError:
            pass
        try:
            score({"x": "name"}, qs)
            raise AssertionError("không chặn phiếu sai câu")
        except ValueError:
            pass
        # bảng theo loại trên dữ liệu giả
        ev = os.path.join(td, "ev")
        os.makedirs(ev)
        preds = [{"question_id": f"t{i}", "question": q, "prediction": "a" if i % 2 else "b", "answers": ["a"]}
                 for i, q in enumerate(list(cases) * 10)]
        json.dump(preds, open(os.path.join(ev, "predictions.json"), "w", encoding="utf-8"), ensure_ascii=False)
        res = build_table(td, [{"id": "X", "section": "vitextvqa_test2000", "eval": "ev"}])
        allv = res["sections"]["vitextvqa_test2000"]["X"]["ALL"]
        assert allv["n"] == len(preds) and abs(allv["em"] - 0.5) < 1e-12
        assert sum(v["n"] for k, v in res["sections"]["vitextvqa_test2000"]["X"].items() if k != "ALL") == len(preds)
        assert "test2000" in to_markdown(res)
    print("question types rules OK")
    print("SELFTEST OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make_sheet", action="store_true")
    ap.add_argument("--score")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--test2000_run", default=os.path.join(
        REPO_ROOT, "experiments", "W07_B1_zeroshot_qwen25vl_vitextvqa_test2000_bf16_L4_seed42"))
    ap.add_argument("--exp_root", default=os.path.join(REPO_ROOT, "experiments"))
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if a.selftest:
        _selftest()
        return 0
    if not a.out:
        ap.error("cần --out")
    if a.make_sheet:
        rows = make_sheet(_load(os.path.join(a.test2000_run, "predictions.json")), a.out)
        print(f"✅ phiếu {len(rows)} câu (seed {SHEET_SEED}) -> {a.out}")
        return 0
    if a.score:
        recs = _load(os.path.join(a.test2000_run, "predictions.json"))
        s = score(read_labels(a.score), {r["question_id"]: r["question"] for r in recs})
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(s, fh, ensure_ascii=False, indent=2)
        print(f"accuracy {100 * s['accuracy']:.1f}% trên {s['n']} câu")
        for t, v in s["per_type"].items():
            p_ = "—" if v["precision"] is None else f"{100 * v['precision']:.0f}%"
            r_ = "—" if v["recall"] is None else f"{100 * v['recall']:.0f}%"
            print(f"  {t:16s} máy {v['n_auto']:3d} | người {v['n_human']:3d} | precision {p_:>4s} | recall {r_:>4s} | {v['flag']}")
        print("->", a.out)
        return 0
    if a.table:
        flags = None
        if a.labels:
            recs = _load(os.path.join(a.test2000_run, "predictions.json"))
            sc = score(read_labels(a.labels), {r["question_id"]: r["question"] for r in recs})
            flags = {t: v["flag"] for t, v in sc["per_type"].items()}
        res = build_table(a.exp_root, default_runs(), flags)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "question_types_table.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        with open(os.path.join(a.out, "question_types_table.md"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(to_markdown(res))
        print(to_markdown(res))
        print("->", a.out)
        return 0
    ap.error("chọn --make_sheet / --score / --table / --selftest")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
