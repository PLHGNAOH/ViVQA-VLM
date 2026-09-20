"""
ocr.py — Trích & CACHE văn bản trong ảnh (OCR) cho ViTextVQA.

VÌ SAO TÁCH RIÊNG & CACHE:
    - PDF yêu cầu ablation "with/without OCR-enhanced prompting". OCR rất chậm, nếu
      chạy lại mỗi lần eval sẽ tốn hàng giờ và KHÔNG tái lập được (kết quả OCR có thể
      đổi giữa các phiên bản lib).
    - Giải pháp: OCR TẤT CẢ ảnh MỘT LẦN -> lưu {tên_ảnh: [text1, text2, ...]} ra JSON.
      Sau đó `run_eval.py` chỉ tra cứu cache (nhanh, cố định) qua tham số `ocr_lookup`.
      => cùng OCR input cho cả run baseline lẫn run cải tiến -> so sánh công bằng.

BACKEND HỖ TRỢ (chọn trong config: data.ocr.backend):
    - "paddleocr"        : PaddleOCR det+rec, lang='vi'. Đơn giản, chạy được ngay.
    - "paddledet_vietocr": PaddleOCR chỉ để DÒ hộp chữ + VietOCR để ĐỌC (nhận dạng
                           tiếng Việt tốt hơn). Combo mạnh nhất cho scene-text VN.
    - "mock"             : engine giả để test pipeline offline (không cần cài gì).

Cache key = TÊN FILE ẢNH (basename), khớp cách `run_eval.py` tra cứu.

Chạy:
    python -m src.data.ocr --config configs/qwen_lora.yaml --split test
"""
from __future__ import annotations
import os
import json
import argparse
from typing import List, Dict, Any, Optional


# =============================================================================
# 1) CÁC ENGINE OCR (mỗi engine expose 1 phương thức: read(image_path) -> List[str])
# =============================================================================
class BaseOCREngine:
    """Giao diện chung. Mọi engine trả về LIST các đoạn text đọc được trong ảnh."""
    def read(self, image_path: str) -> List[str]:
        raise NotImplementedError


class MockOCREngine(BaseOCREngine):
    """
    Engine GIẢ để smoke-test vòng cache offline (không phụ thuộc lib nặng).
    Trả về text cố định dựa trên tên file -> kiểm tra logic đọc/ghi/resume cache.
    """
    def read(self, image_path: str) -> List[str]:
        return [f"MOCK_TEXT::{os.path.basename(image_path)}"]


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR đầy đủ (dò + đọc), tiếng Việt (lang='vi').
    Import trễ để máy không cài paddleocr vẫn import được module này.
    """
    def __init__(self, use_gpu: bool = True, lang: str = "vi"):
        from paddleocr import PaddleOCR
        # use_angle_cls: xoay chữ nghiêng; show_log=False cho đỡ ồn output
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu,
                              show_log=False)

    def read(self, image_path: str) -> List[str]:
        result = self._ocr.ocr(image_path, cls=True)
        return _parse_paddle_result(result)


class PaddleDetVietOCREngine(BaseOCREngine):
    """
    Combo: PaddleOCR DÒ hộp chữ -> cắt ảnh -> VietOCR NHẬN DẠNG (đọc) tiếng Việt.
    Cho độ chính xác đọc tiếng Việt cao hơn rec mặc định của Paddle.
    """
    def __init__(self, use_gpu: bool = True):
        from paddleocr import PaddleOCR
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg

        # Paddle chỉ bật DETECTION (rec=False) -> lấy toạ độ hộp chữ
        self._det = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=use_gpu,
                              show_log=False)
        # VietOCR làm RECOGNIZER
        vcfg = Cfg.load_config_from_name("vgg_transformer")
        vcfg["device"] = "cuda:0" if use_gpu else "cpu"
        vcfg["predictor"]["beamsearch"] = False
        self._rec = Predictor(vcfg)

    def read(self, image_path: str) -> List[str]:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        det = self._det.ocr(image_path, det=True, rec=False)
        boxes = _parse_paddle_boxes(det)

        texts: List[str] = []
        for box in boxes:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            crop = img.crop((min(xs), min(ys), max(xs), max(ys)))
            try:
                txt = self._rec.predict(crop)
                if txt and txt.strip():
                    texts.append(txt.strip())
            except Exception:
                continue
        return texts


# =============================================================================
# 2) PARSE KẾT QUẢ PADDLE (chống đổi API giữa các version)
# =============================================================================
def _parse_paddle_result(result: Any) -> List[str]:
    """
    Rút danh sách text từ kết quả PaddleOCR.ocr(). PaddleOCR 2.x trả:
        [[ [box], (text, conf) ], ... ]  (bọc thêm 1 lớp list theo số ảnh)
    Hàm viết phòng thủ để không vỡ khi cấu trúc lồng khác nhau.
    """
    texts: List[str] = []
    if not result:
        return texts
    # Bóc lớp "theo ảnh" nếu có
    lines = result[0] if (len(result) == 1 and isinstance(result[0], list)) else result
    for line in lines or []:
        try:
            # line = [box, (text, conf)]
            payload = line[1]
            text = payload[0] if isinstance(payload, (list, tuple)) else payload
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        except (IndexError, TypeError):
            continue
    return texts


def _parse_paddle_boxes(result: Any) -> List[list]:
    """Rút danh sách hộp (mỗi hộp = list 4 điểm [x,y]) từ kết quả detection-only."""
    if not result:
        return []
    lines = result[0] if (len(result) == 1 and isinstance(result[0], list)) else result
    boxes = []
    for item in lines or []:
        # detection-only: item có thể là box trực tiếp, hoặc [box, ...]
        box = item[0] if (isinstance(item, (list, tuple)) and item and
                          isinstance(item[0], (list, tuple)) and
                          isinstance(item[0][0], (list, tuple))) else item
        boxes.append(box)
    return boxes


# =============================================================================
# 3) FACTORY: chọn engine theo config
# =============================================================================
def get_ocr_engine(backend: str = "paddleocr", use_gpu: bool = True) -> BaseOCREngine:
    backend = (backend or "paddleocr").lower()
    if backend == "mock":
        return MockOCREngine()
    if backend == "paddleocr":
        return PaddleOCREngine(use_gpu=use_gpu)
    if backend == "paddledet_vietocr":
        return PaddleDetVietOCREngine(use_gpu=use_gpu)
    raise ValueError(f"backend OCR không hợp lệ: '{backend}'")


# =============================================================================
# 4) XÂY CACHE (có RESUME: bỏ qua ảnh đã OCR)
# =============================================================================
def load_ocr_cache(cache_path: str) -> Dict[str, List[str]]:
    """Đọc cache OCR nếu tồn tại; trả dict rỗng nếu chưa có."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_ocr_cache(
    samples: List[Dict[str, Any]],
    engine: BaseOCREngine,
    cache_path: str,
    save_every: int = 50,
) -> Dict[str, List[str]]:
    """
    OCR toàn bộ ảnh DUY NHẤT trong `samples` và lưu {tên_ảnh: [text,...]} ra JSON.

    Tính năng RESUME: nếu cache đã có ảnh nào thì bỏ qua -> chạy lại an toàn khi
    Colab bị ngắt giữa chừng. Lưu định kỳ mỗi `save_every` ảnh để chống mất dữ liệu.

    Args:
        samples: list sample chuẩn (có 'image_path') từ vivqa_dataset.load_vivqa().
        engine:  engine OCR từ get_ocr_engine().
        cache_path: đường dẫn file JSON cache.
        save_every: số ảnh mỗi lần ghi tạm ra đĩa.

    Returns:
        dict cache đầy đủ.
    """
    cache = load_ocr_cache(cache_path)

    # Danh sách ảnh duy nhất (nhiều câu hỏi có thể trỏ cùng 1 ảnh)
    unique_imgs: Dict[str, str] = {}
    for s in samples:
        key = os.path.basename(s["image_path"])
        unique_imgs.setdefault(key, s["image_path"])

    todo = [(k, p) for k, p in unique_imgs.items() if k not in cache]
    print(f"[OCR] Tổng ảnh duy nhất: {len(unique_imgs)} | đã có cache: "
          f"{len(unique_imgs) - len(todo)} | cần OCR: {len(todo)}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    for i, (key, path) in enumerate(todo, 1):
        try:
            cache[key] = engine.read(path)
        except Exception as e:
            print(f"[OCR] Lỗi ảnh {key}: {e}")
            cache[key] = []   # ghi rỗng để không OCR lại vô hạn

        if i % save_every == 0:
            _save_cache(cache, cache_path)
            print(f"[OCR] {i}/{len(todo)} ảnh -> đã lưu tạm.")

    _save_cache(cache, cache_path)
    print(f"[OCR] Hoàn tất. Cache: {cache_path} ({len(cache)} ảnh).")
    return cache


def _save_cache(cache: Dict[str, List[str]], cache_path: str) -> None:
    if not cache_path:
        return
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


# =============================================================================
# 5) CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Xây cache OCR cho ViTextVQA/ViVQA")
    parser.add_argument("--config", default="configs/qwen_lora.yaml")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"],
                        help="split nào trong config sẽ được OCR")
    parser.add_argument("--backend", default=None,
                        help="ghi đè data.ocr.backend (paddleocr|paddledet_vietocr|mock)")
    parser.add_argument("--cpu", action="store_true", help="ép chạy CPU")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from src.data.vivqa_dataset import load_vivqa

    data_cfg = cfg["data"]
    path = data_cfg[f"{args.split}_path"]
    ocr_cfg = data_cfg.get("ocr", {})
    backend = args.backend or ocr_cfg.get("backend", "paddleocr")
    cache_path = data_cfg.get("ocr_cache", "data/ocr_cache.json")

    samples = load_vivqa(
        data_path=path,
        image_dir=data_cfg["image_dir"],
        field_map=ocr_cfg.get("field_map"),
        image_pattern=ocr_cfg.get("image_pattern"),
        max_samples=args.max_samples,
    )
    print(f"[OCR] Split '{args.split}': {len(samples)} mẫu | backend='{backend}'")

    engine = get_ocr_engine(backend=backend, use_gpu=not args.cpu)
    build_ocr_cache(samples, engine, cache_path)


if __name__ == "__main__":
    main()
