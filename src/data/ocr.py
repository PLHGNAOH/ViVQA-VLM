"""
ocr.py — Trích & CACHE văn bản trong ảnh (OCR) cho OCR-enhanced prompting (W08).

VÌ SAO TÁCH RIÊNG & CACHE:
    - Đề cương yêu cầu ablation "with / without OCR-enhanced prompting". OCR chậm và kết quả
      có thể đổi theo phiên bản thư viện -> OCR MỌI ảnh MỘT LẦN, lưu ra JSON, rồi mọi run
      (eval lẫn train) chỉ TRA CỨU cache -> cùng đầu vào OCR cho run baseline và run cải tiến.
    - Chạy OCR ở phiên Colab RIÊNG (thư viện OCR không cần có trong phiên chạy VLM).

BACKEND (PaddleOCR 3.x — `pip install paddleocr` hiện chỉ còn bản 3.x, API khác 2.x):
    - "paddleocr"        : pipeline PaddleOCR(lang="vi") = dò chữ + đọc chữ (model Latin).
    - "paddledet_vietocr": PaddleOCR chỉ DÒ hộp chữ (cùng model dò) + VietOCR ĐỌC tiếng Việt.
    - "mock"             : engine giả để kiểm thử offline (không cần cài gì).

ĐỊNH DẠNG CACHE (v2) — một file / backend, vd <drive>/data/vitextvqa_official/ocr/paddleocr.json:
    {"format": "vivqa-vlm/ocr-cache/v2", "backend": ..., "engine_info": {phiên bản, tên model, thiết bị},
     "images": {"<tên_ảnh>": {"w":..,"h":..,"lines":[{"text","conf","box":[x0,y0,x1,y1]}]}},
     "timing": {"images": n, "sec": tổng}}
    Lưu cả conf + box -> sau này lọc theo ngưỡng tin cậy / phân tích "sai vùng" KHÔNG cần OCR lại.
    Cache key = TÊN FILE ẢNH (ViTextVQA: 16.762 ảnh, tên không trùng giữa các split).

Chạy (Colab, phiên riêng cho OCR):
    python -m src.data.ocr --dataset vitextvqa_official --backend paddleocr --device gpu \
        --files dev_subset300_seed42.json --limit_images 100        # đo tốc độ trước
Kiểm thử offline:
    python -m src.data.ocr --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

CACHE_FORMAT = "vivqa-vlm/ocr-cache/v2"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/ViVQA-VLM"
DATASET_DIRS = {"vivqa": "data/vivqa", "vitextvqa_official": "data/vitextvqa_official"}
OCR_SUBDIR = "ocr"
# Mặc định cho prompt (chọn lại trên dev ở W08 bước 5; giá trị dùng thật ghi vào run_signature)
OCR_DEFAULTS = {"template": "v1", "max_chars": 600, "min_conf": 0.0}
MAX_ERROR_FRAC = 0.01        # > 1% ảnh OCR lỗi -> dừng (giống ngưỡng loại mẫu khi train)
FAIL_FAST_N = 20             # W11: N ảnh ĐẦU của một lần chạy đều lỗi -> lỗi cài đặt -> dừng, KHÔNG lưu ảnh lỗi


def default_cache_path(drive_root: str, dataset: str, backend: str) -> str:
    return os.path.join(drive_root, DATASET_DIRS[dataset], OCR_SUBDIR, f"{backend}.json")


# =============================================================================
# 1) HÀM THUẦN (kiểm thử được không cần thư viện OCR)
# =============================================================================
def poly_to_box(poly: Any) -> List[int]:
    """Đa giác 4 điểm (list hoặc numpy) -> hộp thẳng [x0, y0, x1, y1] (số nguyên)."""
    pts = [(float(p[0]), float(p[1])) for p in poly]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [int(min(xs)), int(min(ys)), int(round(max(xs))), int(round(max(ys)))]


def sort_reading_order(boxes: List[List[int]]) -> List[int]:
    """
    Thứ tự đọc (trên -> dưới, trái -> phải) cho các hộp chữ: gom hộp có tâm dọc gần nhau
    (lệch < nửa chiều cao trung vị) vào cùng một dòng, rồi xếp từng dòng theo x.
    Trả danh sách CHỈ SỐ theo thứ tự đọc.
    """
    if not boxes:
        return []
    heights = sorted(max(1, b[3] - b[1]) for b in boxes)
    tol = heights[len(heights) // 2] / 2.0
    order = sorted(range(len(boxes)), key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2.0, boxes[i][0]))
    rows: List[List[int]] = []
    row_y: List[float] = []
    for i in order:
        yc = (boxes[i][1] + boxes[i][3]) / 2.0
        if rows and abs(yc - row_y[-1]) <= tol:
            rows[-1].append(i)
            row_y[-1] = sum((boxes[j][1] + boxes[j][3]) / 2.0 for j in rows[-1]) / len(rows[-1])
        else:
            rows.append([i])
            row_y.append(yc)
    return [i for row in rows for i in sorted(row, key=lambda j: boxes[j][0])]


def lines_from_paddle3(res: Any) -> List[Dict[str, Any]]:
    """
    Kết quả 1 ảnh của PaddleOCR 3.x (OCRResult, truy cập như dict):
        rec_texts (list str), rec_scores (list float), rec_polys (list đa giác) — đã theo thứ tự đọc.
    """
    texts = list(res["rec_texts"]) if res is not None else []
    scores = list(res["rec_scores"]) if texts else []
    polys = list(res["rec_polys"]) if texts else []
    if not (len(texts) == len(scores) == len(polys)):
        raise ValueError(f"PaddleOCR trả số phần tử lệch nhau: {len(texts)}/{len(scores)}/{len(polys)}")
    out = []
    for t, c, p in zip(texts, scores, polys):
        t = str(t).strip()
        if t:
            out.append({"text": t, "conf": round(float(c), 4), "box": poly_to_box(p)})
    return out


def lines_to_texts(entry: Dict[str, Any], max_chars: Optional[int], min_conf: float = 0.0) -> Tuple[List[str], bool]:
    """
    Các dòng OCR của 1 ảnh -> list chuỗi đưa vào prompt.
    Lọc conf < min_conf (conf None = giữ), giữ thứ tự đọc, CẮT khi độ dài ghép (" | ") vượt max_chars.
    Trả (texts, đã_bị_cắt).
    """
    from src.prompting.builder import OCR_JOINER
    kept = [ln["text"].strip() for ln in entry.get("lines", [])
            if ln.get("text", "").strip() and (ln.get("conf") is None or float(ln["conf"]) >= min_conf)]
    if not max_chars:
        return kept, False
    out: List[str] = []
    used = 0
    for t in kept:
        add = len(t) + (len(OCR_JOINER) if out else 0)
        if used + add > max_chars:
            if not out:                      # một dòng duy nhất đã quá dài -> cắt dòng đó
                out.append(t[:max_chars])
            return out, True
        out.append(t)
        used += add
    return out, False


# =============================================================================
# 2) CACHE: đọc / ghi an toàn / tóm tắt
# =============================================================================
def _normalize_entry(v: Any) -> Dict[str, Any]:
    """Chấp nhận định dạng cũ (list chuỗi) -> {lines:[{text, conf=None}]}."""
    if isinstance(v, list):
        return {"lines": [{"text": str(t), "conf": None} for t in v if str(t).strip()]}
    return v


def load_ocr_cache(cache_path: str) -> Dict[str, Any]:
    """Đọc cache (v2 hoặc cũ {ảnh: [chuỗi]}); không có file -> cache rỗng."""
    if not cache_path or not os.path.exists(cache_path):
        return {"format": CACHE_FORMAT, "backend": None, "engine_info": {}, "images": {}, "timing": {}}
    with open(cache_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("format") == CACHE_FORMAT:
        raw.setdefault("timing", {})
        return raw
    return {"format": "legacy-v1", "backend": None, "engine_info": {},
            "images": {k: _normalize_entry(v) for k, v in raw.items()}, "timing": {}}


def save_ocr_cache(cache: Dict[str, Any], cache_path: str) -> None:
    """Ghi .tmp rồi đổi tên -> Colab ngắt giữa lúc ghi cũng không làm hỏng cache."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cache, f, ensure_ascii=False, indent=0)
    os.replace(tmp, cache_path)


def cache_fingerprint(cache: Dict[str, Any], keys: Iterable[str]) -> str:
    """sha256 của đúng phần cache mà run dùng (các ảnh trong keys) -> cache lớn thêm không đổi chữ ký run cũ."""
    imgs = cache["images"]
    part = {k: imgs[k].get("lines", []) for k in sorted(set(keys))}
    return hashlib.sha256(json.dumps(part, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# =============================================================================
# 3) GẮN OCR VÀO MẪU (dùng chung cho run_baseline và train_qlora)
# =============================================================================
def attach_ocr(samples: List[Dict[str, Any]], cache: Dict[str, Any], max_chars: Optional[int],
               min_conf: float = 0.0) -> Dict[str, Any]:
    """
    Gắn s["ocr_texts"] cho từng mẫu. Ảnh không có trong cache -> BÁO LỖI (không âm thầm
    chạy như zero-shot). Ảnh OCR lỗi > 1% -> báo lỗi. Trả thống kê để ghi vào log.
    """
    imgs = cache["images"]
    keys = [os.path.basename(s["image_path"]) for s in samples]
    missing = sorted({k for k in keys if k not in imgs})
    if missing:
        raise KeyError(f"{len(missing)} ảnh chưa có trong OCR cache, vd {missing[:3]} -> chạy OCR cho các file này trước")
    uniq = sorted(set(keys))
    n_err = sum(1 for k in uniq if imgs[k].get("error"))
    if uniq and n_err / len(uniq) > MAX_ERROR_FRAC:
        raise ValueError(f"{n_err}/{len(uniq)} ảnh OCR bị lỗi (> {MAX_ERROR_FRAC:.0%}) -> kiểm tra cache")
    with_text, truncated, chars = 0, 0, []
    for s, k in zip(samples, keys):
        texts, cut = lines_to_texts(imgs[k], max_chars, min_conf)
        s["ocr_texts"] = texts
        with_text += bool(texts)
        truncated += cut
        chars.append(sum(len(t) for t in texts))
    n = max(len(samples), 1)
    return {"n_samples": len(samples), "n_images": len(uniq), "n_ocr_errors": n_err,
            "pct_samples_with_text": round(100 * with_text / n, 2),
            "pct_samples_truncated": round(100 * truncated / n, 2),
            "mean_chars": round(sum(chars) / n, 1),
            "subset_sha256": cache_fingerprint(cache, keys)}


def setup_ocr_prompting(cfg: Dict[str, Any], samples: List[Dict[str, Any]], cache_path: str,
                        template: str = OCR_DEFAULTS["template"], max_chars: Optional[int] = OCR_DEFAULTS["max_chars"],
                        min_conf: float = OCR_DEFAULTS["min_conf"]) -> Dict[str, Any]:
    """
    Nạp cache + gắn OCR vào mẫu + ghi thiết lập OCR vào cfg["prompting"]["ocr"].
    Vì nằm trong cfg["prompting"], mọi thứ ảnh hưởng prompt (backend, model OCR, phần cache dùng,
    template, ngưỡng) tự vào run_signature -> đổi bất kỳ thứ gì cũng thành run khác.
    """
    from src.prompting.builder import OCR_TEMPLATES
    if template not in OCR_TEMPLATES:
        raise ValueError(f"--ocr_template phải là một trong {sorted(OCR_TEMPLATES)}")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Không thấy OCR cache {cache_path} -> chạy `python -m src.data.ocr` trước")
    cache = load_ocr_cache(cache_path)
    stats = attach_ocr(samples, cache, max_chars, min_conf)
    info = cache.get("engine_info") or {}
    cfg["prompting"]["ocr"] = {
        "backend": cache.get("backend"), "cache_format": cache.get("format"),
        "engine": {k: info.get(k) for k in ("paddleocr", "det_model", "rec_model", "lang", "vietocr_config")},
        "cache_file": os.path.basename(cache_path), "subset_sha256": stats["subset_sha256"],
        "template": template, "max_chars": max_chars, "min_conf": min_conf,
    }
    return stats


# =============================================================================
# 4) ENGINE OCR (import trễ: máy không cài paddle vẫn import được module này)
# =============================================================================
class BaseOCREngine:
    backend = "base"
    info: Dict[str, Any] = {}

    def read(self, image_path: str) -> Dict[str, Any]:
        """Trả {"w", "h", "lines": [{"text", "conf", "box"}]} theo thứ tự đọc."""
        raise NotImplementedError


class MockOCREngine(BaseOCREngine):
    """Engine giả: chữ cố định theo tên file -> kiểm thử vòng cache/resume offline."""
    backend = "mock"

    def __init__(self, **_kw):
        self.info = {"engine": "mock"}

    def read(self, image_path: str) -> Dict[str, Any]:
        name = os.path.basename(image_path)
        return {"w": 100, "h": 50, "lines": [{"text": f"MOCK {name}", "conf": 0.9, "box": [0, 0, 50, 10]},
                                             {"text": "dòng hai", "conf": 0.3, "box": [0, 20, 50, 30]}]}


def _require_paddleocr3():
    import paddleocr
    major = int(str(paddleocr.__version__).split(".")[0])
    if major < 3:
        raise RuntimeError(f"Cần paddleocr >= 3 (đang có {paddleocr.__version__}) -> pip install -U paddleocr")
    return paddleocr


def _paddle_version() -> Optional[str]:
    try:
        import paddle
        return paddle.__version__
    except Exception:
        return None


def _make_text_detection(po: Any, det_model: str, device: str) -> Any:
    """
    Model dò chữ. W11: trên CPU, Paddle 3.3 + oneDNN (MKL-DNN) lỗi với mọi ảnh
    ("ConvertPirAttribute2RuntimeAttribute not support", notebook 19 lần 1) -> CPU thì TẮT oneDNN.
    GPU: giữ nguyên lời gọi cũ (cùng cấu hình với cache ViTextVQA W08–W10).
    """
    if device.startswith("gpu"):
        return po.TextDetection(model_name=det_model, device=device)
    try:
        return po.TextDetection(model_name=det_model, device=device, enable_mkldnn=False)
    except TypeError as e:
        raise RuntimeError(f"paddleocr {po.__version__} không nhận enable_mkldnn -> không chạy CPU an toàn được: {e}")


def _image_size(image_path: str) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(image_path) as im:
        return im.size


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR 3.x pipeline đầy đủ, lang='vi' (dò + đọc). Tắt xoay/nắn tài liệu: ảnh cảnh thật, không phải scan."""
    backend = "paddleocr"

    def __init__(self, device: str = "gpu", lang: str = "vi"):
        po = _require_paddleocr3()
        self._ocr = po.PaddleOCR(lang=lang, device=device, use_doc_orientation_classify=False,
                                 use_doc_unwarping=False, use_textline_orientation=False)
        params = getattr(self._ocr, "_params", {}) or {}
        self.info = {"engine": "PaddleOCR pipeline", "paddleocr": po.__version__, "paddlepaddle": _paddle_version(),
                     "lang": lang, "device": device,
                     "det_model": params.get("text_detection_model_name"),
                     "rec_model": params.get("text_recognition_model_name"),
                     "textline_orientation": False}

    def read(self, image_path: str) -> Dict[str, Any]:
        w, h = _image_size(image_path)
        res = self._ocr.predict(image_path)
        return {"w": w, "h": h, "lines": lines_from_paddle3(res[0] if res else None)}


class PaddleDetVietOCREngine(BaseOCREngine):
    """
    PaddleOCR 3.x CHỈ dò hộp chữ (cùng model dò với backend "paddleocr" -> chỉ khác phần ĐỌC)
    + VietOCR (vgg_transformer) đọc từng hộp. Cài VietOCR bằng `pip install --no-deps vietocr`
    + `pip install gdown einops` (bản PyPI khoá pillow==10.2.0 sẽ hạ cấp Pillow của Colab).
    """
    backend = "paddledet_vietocr"

    def __init__(self, device: str = "gpu", det_model: Optional[str] = None, lang: str = "vi"):
        po = _require_paddleocr3()
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
        if det_model is None:                           # lấy đúng model dò mà pipeline lang='vi' dùng
            det_model, _ = po.PaddleOCR._get_ocr_model_names(None, lang, None)
        self._det = _make_text_detection(po, det_model, device)
        vcfg = Cfg.load_config_from_name("vgg_transformer")
        vcfg["device"] = "cuda:0" if device.startswith("gpu") else "cpu"
        vcfg["predictor"]["beamsearch"] = False
        self._rec = Predictor(vcfg)
        self.info = {"engine": "PaddleOCR TextDetection + VietOCR", "paddleocr": po.__version__,
                     "paddlepaddle": _paddle_version(), "lang": lang,
                     "device": device, "det_model": det_model, "rec_model": "vietocr/vgg_transformer",
                     "vietocr_config": "vgg_transformer, beamsearch=False"}
        if not device.startswith("gpu"):                # chỉ ghi khi chạy CPU -> info bản GPU giữ nguyên như W08–W10
            self.info["det_mkldnn"] = False

    def read(self, image_path: str) -> Dict[str, Any]:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        det = self._det.predict(image_path)
        polys = list(det[0]["dt_polys"]) if det else []
        boxes = [poly_to_box(p) for p in polys]
        boxes = [[max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])] for b in boxes]
        boxes = [b for b in boxes if b[2] - b[0] >= 4 and b[3] - b[1] >= 4]
        boxes = [boxes[i] for i in sort_reading_order(boxes)]
        if not boxes:
            return {"w": w, "h": h, "lines": []}
        crops = [img.crop(tuple(b)) for b in boxes]
        texts, probs = self._rec.predict_batch(crops, return_prob=True)
        lines = [{"text": str(t).strip(), "conf": round(float(p), 4), "box": b}
                 for t, p, b in zip(texts, probs, boxes) if str(t).strip()]
        return {"w": w, "h": h, "lines": lines}


ENGINES = {"mock": MockOCREngine, "paddleocr": PaddleOCREngine, "paddledet_vietocr": PaddleDetVietOCREngine}


def get_ocr_engine(backend: str = "paddleocr", device: str = "gpu") -> BaseOCREngine:
    if backend not in ENGINES:
        raise ValueError(f"backend OCR không hợp lệ: '{backend}' (có: {sorted(ENGINES)})")
    return ENGINES[backend](device=device)


# =============================================================================
# 5) XÂY CACHE (resume được: bỏ qua ảnh đã có)
# =============================================================================
def build_ocr_cache(images: Dict[str, str], engine: BaseOCREngine, cache_path: str,
                    save_every: int = 50, limit_images: Optional[int] = None) -> Dict[str, Any]:
    """
    OCR các ảnh {tên_ảnh: đường_dẫn} còn thiếu trong cache rồi lưu. Lưu tạm mỗi `save_every` ảnh.
    Cache đã có của backend/model KHÁC -> báo lỗi (không trộn hai bộ OCR trong một file).
    """
    cache = load_ocr_cache(cache_path)
    if cache["format"] != CACHE_FORMAT:
        raise ValueError(f"{cache_path} là định dạng cũ -> dùng file mới cho cache v2")
    if cache["images"]:
        old = cache.get("engine_info") or {}
        for k in ("paddleocr", "det_model", "rec_model", "lang"):
            if old.get(k) != engine.info.get(k):
                raise ValueError(f"{cache_path} được tạo bằng {k}={old.get(k)}, engine hiện tại {engine.info.get(k)} "
                                 f"-> dùng file cache khác")
    cache["backend"] = engine.backend
    cache["engine_info"] = engine.info
    todo = [(k, p) for k, p in sorted(images.items()) if k not in cache["images"]]
    if limit_images is not None:
        todo = todo[:limit_images]
    print(f"[OCR] {engine.backend}: {len(images)} ảnh cần có | đã có {len(images) - len(todo) if limit_images is None else 'n/a'} "
          f"| sẽ OCR {len(todo)}", flush=True)
    timing = cache.setdefault("timing", {})
    t_run = time.time()
    n_err_run = 0
    for i, (key, path) in enumerate(todo, 1):
        t0 = time.time()
        try:
            cache["images"][key] = engine.read(path)
        except Exception as e:                       # ghi lỗi để không OCR lại vô hạn; attach_ocr chặn nếu > 1%
            cache["images"][key] = {"lines": [], "error": f"{type(e).__name__}: {e}"[:300]}
            print(f"[OCR] lỗi {key}: {e}", flush=True)
            n_err_run += 1
            if n_err_run == i == min(FAIL_FAST_N, len(todo)):
                raise RuntimeError(f"{i} ảnh đầu đều lỗi -> lỗi cài đặt OCR, KHÔNG lưu vào cache. Lỗi cuối: {e}")
        timing["images"] = timing.get("images", 0) + 1
        timing["sec"] = round(timing.get("sec", 0.0) + time.time() - t0, 2)
        if i % save_every == 0:
            save_ocr_cache(cache, cache_path)
            el = time.time() - t_run
            print(f"[OCR] {i}/{len(todo)} | {el / i:.2f} s/ảnh | còn ~{el / i * (len(todo) - i) / 60:.0f} phút", flush=True)
    save_ocr_cache(cache, cache_path)
    sec = timing.get("sec", 0.0) / max(timing.get("images", 1), 1)
    print(f"[OCR] xong. Cache: {cache_path} ({len(cache['images'])} ảnh, trung bình {sec:.2f} s/ảnh)", flush=True)
    return cache


def images_of(files: List[str], image_dir: str) -> Dict[str, str]:
    """Tập ảnh duy nhất được tham chiếu bởi các file dữ liệu."""
    from src.data.vivqa_dataset import load_vivqa
    out: Dict[str, str] = {}
    for f in files:
        for s in load_vivqa(f, image_dir=image_dir):
            out.setdefault(os.path.basename(s["image_path"]), s["image_path"])
    return out


# =============================================================================
# 6) KIỂM THỬ OFFLINE
# =============================================================================
def _selftest() -> None:
    import tempfile
    # hàm thuần
    assert poly_to_box([[10, 5], [40, 6], [41, 20], [9, 19.6]]) == [9, 5, 41, 20]
    boxes = [[60, 12, 90, 22], [0, 10, 50, 20], [0, 40, 30, 50], [35, 41, 80, 52]]
    assert sort_reading_order(boxes) == [1, 0, 2, 3], sort_reading_order(boxes)
    fake = {"rec_texts": ["CẤM ĐỖ XE", " ", "24/24"], "rec_scores": [0.98, 0.5, 0.71],
            "rec_polys": [[[0, 0], [9, 0], [9, 4], [0, 4]]] * 3}
    ls = lines_from_paddle3(fake)
    assert [x["text"] for x in ls] == ["CẤM ĐỖ XE", "24/24"] and ls[0]["box"] == [0, 0, 9, 4]
    assert lines_from_paddle3({"rec_texts": [], "rec_scores": [], "rec_polys": []}) == []
    try:
        lines_from_paddle3({"rec_texts": ["a"], "rec_scores": [], "rec_polys": []})
        raise AssertionError("không chặn kết quả lệch")
    except ValueError:
        pass
    e = {"lines": [{"text": "AAAA", "conf": 0.9}, {"text": "BB", "conf": 0.2}, {"text": "CCCC", "conf": None}]}
    assert lines_to_texts(e, None) == (["AAAA", "BB", "CCCC"], False)
    assert lines_to_texts(e, None, min_conf=0.5) == (["AAAA", "CCCC"], False)
    assert lines_to_texts(e, 9) == (["AAAA", "BB"], True)              # 4 + 3 + 2 = 9 vừa đủ, CCCC bị cắt
    assert lines_to_texts(e, 2) == (["AA"], True)
    assert lines_to_texts({"lines": []}, 600) == ([], False)

    with tempfile.TemporaryDirectory() as td:
        img = os.path.join(td, "img")
        os.makedirs(img)
        recs = []
        for i in range(6):
            open(os.path.join(img, f"{i % 4}.jpg"), "wb").close()
            recs.append({"question_id": f"q{i}", "question": f"hỏi {i}?", "answers": ["a"], "image": f"{i % 4}.jpg"})
        data = os.path.join(td, "d.json")
        json.dump(recs, open(data, "w", encoding="utf-8"), ensure_ascii=False)
        ims = images_of([data], img)
        assert sorted(ims) == ["0.jpg", "1.jpg", "2.jpg", "3.jpg"]
        cpath = os.path.join(td, "ocr", "mock.json")
        eng = MockOCREngine()
        c = build_ocr_cache(ims, eng, cpath, save_every=2, limit_images=3)
        assert len(c["images"]) == 3 and c["backend"] == "mock" and c["format"] == CACHE_FORMAT
        c = build_ocr_cache(ims, eng, cpath, save_every=2)               # resume: chỉ OCR ảnh còn thiếu
        assert len(c["images"]) == 4 and c["timing"]["images"] == 4
        other = MockOCREngine(); other.info = {"engine": "mock", "rec_model": "khác"}
        try:
            build_ocr_cache(ims, other, cpath)
            raise AssertionError("không chặn trộn engine khác vào cùng cache")
        except ValueError:
            pass
        # gắn OCR
        from src.data.vivqa_dataset import load_vivqa
        samples = load_vivqa(data, image_dir=img)
        cfg = {"prompting": {"mode": "ocr"}}
        st = setup_ocr_prompting(cfg, samples, cpath, template="v1", max_chars=600, min_conf=0.5)
        assert samples[0]["ocr_texts"] == ["MOCK 0.jpg"] and st["pct_samples_with_text"] == 100.0
        assert cfg["prompting"]["ocr"]["backend"] == "mock" and cfg["prompting"]["ocr"]["min_conf"] == 0.5
        sha1 = st["subset_sha256"]
        # cache lớn thêm ảnh KHÔNG dùng -> dấu vân tay của run không đổi
        cc = load_ocr_cache(cpath)
        cc["images"]["9.jpg"] = {"lines": [{"text": "x", "conf": 1.0}]}
        save_ocr_cache(cc, cpath)
        st2 = setup_ocr_prompting({"prompting": {"mode": "ocr"}}, load_vivqa(data, image_dir=img), cpath)
        assert st2["subset_sha256"] == sha1
        # ảnh thiếu trong cache -> lỗi
        cc["images"].pop("3.jpg")
        save_ocr_cache(cc, cpath)
        try:
            attach_ocr(load_vivqa(data, image_dir=img), load_ocr_cache(cpath), 600)
            raise AssertionError("không chặn ảnh thiếu OCR")
        except KeyError:
            pass
        # ảnh lỗi > 1% -> chặn
        cc["images"]["3.jpg"] = {"lines": [], "error": "boom"}
        save_ocr_cache(cc, cpath)
        try:
            attach_ocr(load_vivqa(data, image_dir=img), load_ocr_cache(cpath), 600)
            raise AssertionError("không chặn tỉ lệ lỗi OCR cao")
        except ValueError:
            pass
        # cache định dạng cũ {ảnh: [chuỗi]} vẫn đọc được
        legacy = os.path.join(td, "legacy.json")
        json.dump({"0.jpg": ["A", " "]}, open(legacy, "w", encoding="utf-8"))
        assert load_ocr_cache(legacy)["images"]["0.jpg"]["lines"] == [{"text": "A", "conf": None}]
        try:
            setup_ocr_prompting({"prompting": {"mode": "ocr"}}, [], cpath, template="v9")
            raise AssertionError("không chặn template lạ")
        except ValueError:
            pass
    # W11: engine hỏng (mọi ảnh lỗi) -> dừng sau FAIL_FAST_N ảnh, cache trên đĩa KHÔNG có ảnh lỗi nào
    class _Broken(MockOCREngine):
        def read(self, image_path):
            raise RuntimeError("ConvertPirAttribute2RuntimeAttribute not support")
    with tempfile.TemporaryDirectory() as td:
        ims = {}
        for i in range(30):
            p = os.path.join(td, f"{i}.jpg")
            open(p, "wb").close()
            ims[f"{i}.jpg"] = p
        cp = os.path.join(td, "ocr", "x.json")
        try:
            build_ocr_cache(ims, _Broken(), cp, save_every=50)
            raise AssertionError("không dừng khi engine hỏng")
        except RuntimeError:
            pass
        assert not load_ocr_cache(cp)["images"], "ảnh lỗi bị ghi vào cache"
        build_ocr_cache(dict(list(ims.items())[:3]), MockOCREngine(), cp)      # ít ảnh hơn N, engine tốt -> chạy bình thường
        assert len(load_ocr_cache(cp)["images"]) == 3
    print("ocr fail-fast OK", flush=True)
    assert sorted(ENGINES) == ["mock", "paddledet_vietocr", "paddleocr"]
    print("ocr SELFTEST OK", flush=True)


# =============================================================================
# 7) CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Xây OCR cache (v2) cho ViTextVQA / ViVQA")
    ap.add_argument("--dataset", choices=sorted(DATASET_DIRS))
    ap.add_argument("--files", nargs="+", help="Tên file dữ liệu trong data/<dataset>/ (vd dev_subset300_seed42.json)")
    ap.add_argument("--backend", default="paddleocr", choices=sorted(ENGINES))
    ap.add_argument("--device", default="gpu", help="gpu | gpu:0 | cpu")
    ap.add_argument("--drive_root", default=DEFAULT_DRIVE_ROOT)
    ap.add_argument("--image_dir", default=None, help="Thư mục ảnh (mặc định: chép/giải nén về /content như run_baseline)")
    ap.add_argument("--cache", default=None, help="Ghi đè đường dẫn cache (mặc định data/<dataset>/ocr/<backend>.json)")
    ap.add_argument("--limit_images", type=int, default=None, help="Chỉ OCR N ảnh còn thiếu (đo tốc độ)")
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.dataset or not a.files:
        ap.error("cần --dataset và --files (trừ khi --selftest)")
    if a.image_dir is None:
        from src.eval.run_baseline import prepare_images
        a.image_dir = prepare_images(a.dataset, a.drive_root)
    files = [f if os.path.isabs(f) else os.path.join(a.drive_root, DATASET_DIRS[a.dataset], f) for f in a.files]
    imgs = images_of(files, a.image_dir)
    missing = [p for p in imgs.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ảnh không có trên đĩa, vd {missing[0]}")
    engine = get_ocr_engine(a.backend, device=a.device)
    print(f"[OCR] engine: {json.dumps(engine.info, ensure_ascii=False)}", flush=True)
    build_ocr_cache(imgs, engine, a.cache or default_cache_path(a.drive_root, a.dataset, a.backend),
                    save_every=a.save_every, limit_images=a.limit_images)


if __name__ == "__main__":
    main()
