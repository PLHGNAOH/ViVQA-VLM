"""
infer.py — Script suy luận dòng lệnh (không cần giao diện) cho hệ thống ViVQA-VLM đã đánh giá (W14, bước 5).

Vấn đề: gói tái lập (master §10) cần một "inference script" chạy được mà không phải mở giao diện Gradio.
Script này dùng ĐÚNG đường của prototype đã kiểm (R-PAR, R-OCR v2, R-LIVE v2):
    prepare_image (xoay EXIF, ngưỡng 6.000 px) -> OCR ở tiến trình riêng (PaddleOCR + VietOCR, engine = cache)
    -> Backend.ask (Qwen2.5-VL-3B 4-bit + adapter SH, ngưỡng τ* cho câu có/không, mức tin cậy).
Kết quả in ra JSON (một dòng / câu hỏi), có thể ghi file.

Chạy (Colab L4, sau khi cài thư viện như notebook 28 mục 2):
    python -m src.prototype.infer --adapter <…/W11_SH_…/epoch_1> --image anh.jpg \
        --question "cửa hàng này tên là gì?" --question "trong ảnh có xe máy không?" [--out ketqua.jsonl]
    # adapter + sha256 kiểm theo configs/official_models.json (mặc định: kiểm SH):
    python -m src.prototype.infer --official sh --drive_root /content/drive/MyDrive/ViVQA-VLM --image … --question …
Thử không cần GPU (backend giả, cùng đường xử lý ảnh + định dạng đầu ra):
    python -m src.prototype.infer --mock --image anh.jpg --question "…"
Kiểm thử offline:
    python -m src.prototype.infer --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST_REL = os.path.join("configs", "official_models.json")
OUT_KEYS = ("answer", "mode", "level", "confidence", "ocr_texts", "sec_ocr", "sec", "sec_total")


def official_adapter(model_id: str, drive_root: str, manifest_path: Optional[str] = None) -> Dict[str, str]:
    """Đường dẫn + sha256 adapter của model chính thức (configs/official_models.json)."""
    with open(manifest_path or os.path.join(REPO_ROOT, MANIFEST_REL), encoding="utf-8") as f:
        man = json.load(f)
    m = {x["id"]: x for x in man["models"]}.get(model_id)
    if m is None:
        raise SystemExit(f"Không có model '{model_id}' trong manifest (có: {[x['id'] for x in man['models']]})")
    if not m.get("adapter"):
        raise SystemExit(f"'{model_id}' là zero-shot, không có adapter -> dùng src.eval.run_baseline")
    # drive_path trong manifest luôn dùng "/" -> tách ra rồi nối bằng dấu phân cách của hệ điều hành (Windows: "\\")
    return {"path": os.path.join(drive_root, *m["adapter"]["drive_path"].split("/")),
            "sha256": m["adapter"]["weights_sha256"]}


def ask_all(backend, image: str, questions: List[str], tmp_dir: str, image_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Mỗi câu hỏi: prepare_image (như giao diện) -> backend.ask. Trả bản ghi gọn, không chứa đường dẫn ảnh tạm."""
    from src.prototype.app import prepare_image
    out = []
    for q in questions:
        used, info = prepare_image(image, tmp_dir)
        try:
            r = backend.ask(used, q)
        finally:
            if used != image and os.path.exists(used):
                os.remove(used)
        rec = {"image": image_name or os.path.basename(image), "question": q}
        rec.update({k: r.get(k) for k in OUT_KEYS})
        rec["image_info"] = {k: info[k] for k in ("orig_size", "used_size", "rotated", "resized")}
        out.append(rec)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image")
    ap.add_argument("--question", action="append", default=[], help="lặp lại để hỏi nhiều câu trên cùng ảnh")
    ap.add_argument("--adapter", help="thư mục adapter (…/epoch_1)")
    ap.add_argument("--expect_adapter_sha", default=None)
    ap.add_argument("--official", default=None, help="lấy adapter + sha256 từ configs/official_models.json (vd sh)")
    ap.add_argument("--drive_root", default="/content/drive/MyDrive/ViVQA-VLM")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "qwen_lora.yaml"))
    ap.add_argument("--ocr_device", default="gpu")
    ap.add_argument("--ocr_log", default=None)
    ap.add_argument("--out", default=None, help="ghi JSONL (mỗi dòng một câu hỏi)")
    ap.add_argument("--mock", action="store_true", help="backend giả (CPU), để thử định dạng")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
        return 0
    if not a.image or not a.question:
        ap.error("cần --image và ít nhất một --question")
    if not os.path.exists(a.image):
        ap.error(f"không thấy ảnh {a.image}")
    from src.prototype.app import MockBackend, load_ocr_selection, load_ui_settings, make_real_backend
    st, ocr_cfg = load_ui_settings(), load_ocr_selection()
    client = None
    if a.mock:
        backend = MockBackend(st, ocr_cfg)
        start = {"mock": True}
    else:
        adapter, sha = a.adapter, a.expect_adapter_sha
        if a.official:
            o = official_adapter(a.official, a.drive_root)
            adapter, sha = adapter or o["path"], sha or o["sha256"]
        if not adapter:
            ap.error("cần --adapter hoặc --official (hoặc --mock)")
        backend, client, start = make_real_backend(adapter, a.config, st, ocr_cfg, ocr_device=a.ocr_device,
                                                   ocr_log=a.ocr_log, expect_adapter_sha=sha)
    tmp = tempfile.mkdtemp(prefix="vivqa_infer_")
    try:
        recs = ask_all(backend, a.image, a.question, tmp)
    finally:
        if client is not None:
            client.close()
        shutil.rmtree(tmp, ignore_errors=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in recs]
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"startup": {k: start.get(k) for k in ("sec_start_total", "mock") if k in start}},
                     ensure_ascii=False), file=sys.stderr)
    return 0


def _selftest() -> None:
    import contextlib
    import io
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "a.jpg")
        ex = Image.Exif()
        ex[274] = 6
        Image.new("RGB", (400, 300), "white").save(p, exif=ex)
        out = os.path.join(td, "o", "r.jsonl")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["--mock", "--image", p, "--question", "cửa hàng tên gì?", "--question",
                       "trong ảnh có con mèo không?", "--out", out])
        assert rc == 0
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        assert len(rows) == 2 and rows[0]["question"] == "cửa hàng tên gì?" and rows[0]["image"] == "a.jpg"
        assert rows[0]["image_info"]["rotated"] and rows[0]["image_info"]["used_size"] == [300, 400]
        assert all(set(OUT_KEYS) <= set(r) for r in rows) and rows[0]["level"] in ("Cao", "Trung bình", "Thấp")
        assert rows[0]["mode"] == "open" and rows[1]["mode"] == "yesno"
        assert len(buf.getvalue().strip().splitlines()) == 2
        # manifest: SH có adapter + sha; zero-shot báo lỗi rõ
        o = official_adapter("sh", "/drive")
        want = os.path.join("/drive", "adapters", "W11_SH_")                     # đúng trên cả Linux lẫn Windows
        assert o["path"].startswith(want) and o["path"].endswith(os.path.join("", "epoch_1")) and len(o["sha256"]) == 64
        try:
            official_adapter("b1", "/drive")
            raise AssertionError("zero-shot không có adapter phải báo lỗi")
        except SystemExit:
            pass
    print("✅ selftest infer OK (đường ảnh như giao diện, nhiều câu hỏi, JSONL, manifest adapter)")


if __name__ == "__main__":
    sys.exit(main())
