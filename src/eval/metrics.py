"""
metrics.py — Bộ chỉ số ĐÁNH GIÁ CHÍNH cho ViVQA-VLM.

Theo yêu cầu PDF (mục "Experimental Evaluation"):
    - Primary metrics: Exact Match (EM), VQA Accuracy, ANLS.
    - BLEU/CIDEr KHÔNG dùng làm primary (chỉ báo phụ cho câu trả lời sinh tự do).

Vì sao tự viết mà không tái dùng metrics.py của repo ViVQA-X cũ?
    - Repo cũ đo BLEU/METEOR/CIDEr/SPICE/BERTScore cho việc SINH GIẢI THÍCH dài.
    - Capstone của ta đo độ ĐÚNG của CÂU TRẢ LỜI (ngắn) + đọc chữ trong ảnh (ANLS).
    => Khác bài toán, khác metric. Module này viết mới hoàn toàn, không phụ thuộc
       pycocoevalcap (vốn cần Java, rất khó cài trên Windows/Colab).

Thiết kế: mọi hàm nhận (pred: str, gts: List[str]) và trả về float trong [0, 1].
'gts' là DANH SÁCH đáp án đúng có thể chấp nhận (ground truths). Nếu dataset chỉ
có 1 đáp án thì list có 1 phần tử — công thức vẫn đúng.
"""
from __future__ import annotations
import re
import unicodedata
from typing import List, Dict, Callable


# =============================================================================
# 1) CHUẨN HÓA VĂN BẢN TIẾNG VIỆT
# =============================================================================
def normalize_vi(text: str) -> str:
    """
    Chuẩn hóa 1 chuỗi trước khi so khớp.

    Các bước:
        - unicode NFC: gộp ký tự + dấu thành 1 code point thống nhất (tránh trường
          hợp "ề" lưu 2 kiểu khác nhau -> so sánh bị lệch). RẤT quan trọng cho
          tiếng Việt vì dữ liệu web hay lẫn NFC/NFD.
        - về chữ thường.
        - bỏ dấu câu NHƯNG GIỮ dấu thanh tiếng Việt (\\w trong regex unicode của
          Python đã bao gồm chữ cái có dấu -> ta chỉ loại ký tự không phải chữ/số/space).
        - gộp khoảng trắng thừa.

    LƯU Ý: ta KHÔNG bỏ dấu tiếng Việt. "con mèo" != "con meo" — bỏ dấu sẽ làm sai
    lệch, và chính "diacritics" là một trong các thách thức PDF muốn ta phân tích.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)  # bỏ dấu câu
    text = re.sub(r"\s+", " ", text).strip()                # gộp space
    return text


# =============================================================================
# 2) EXACT MATCH (EM)
# =============================================================================
def exact_match(pred: str, gts: List[str]) -> float:
    """
    EM = 1.0 nếu câu trả lời (sau chuẩn hóa) TRÙNG KHÍT với ít nhất 1 đáp án đúng,
    ngược lại 0.0. Đây là chỉ số nghiêm khắc nhất.
    """
    p = normalize_vi(pred)
    return 1.0 if p in {normalize_vi(g) for g in gts} else 0.0


# =============================================================================
# 3) VQA ACCURACY (soft accuracy kiểu VQA v2)
# =============================================================================
def vqa_accuracy(pred: str, gts: List[str]) -> float:
    """
    Công thức VQA gốc: acc = min(#annotator_đồng_ý / 3, 1.0).

    Ý nghĩa: nếu >= 3 người gán cùng đáp án mà model đoán trúng -> điểm tối đa 1.0.
    Bản chất "khoan dung" hơn EM: chấp nhận việc nhiều người có thể trả lời khác nhau.

    Trường hợp dataset chỉ có 1 đáp án/câu (như ViVQA gốc): list gts có 1 phần tử,
    khi đó min(1/3,1)=0.33 nếu trúng — hơi thấp. Vì vậy khi chỉ có 1 GT, ta coi
    như trúng = 1.0 (đồng nhất với EM). Đội eval nên GHI RÕ quy ước này trong báo cáo.
    """
    p = normalize_vi(pred)
    norm_gts = [normalize_vi(g) for g in gts]
    matches = sum(1 for g in norm_gts if g == p)
    if len(norm_gts) <= 1:                     # dataset 1-đáp-án -> quy về đúng/sai
        return 1.0 if matches >= 1 else 0.0
    return min(matches / 3.0, 1.0)             # dataset nhiều-annotator -> soft acc


# =============================================================================
# 4) ANLS (Average Normalized Levenshtein Similarity)
# =============================================================================
def _levenshtein(a: str, b: str) -> int:
    """
    Khoảng cách chỉnh sửa (edit distance) thuần Python — không cần thư viện ngoài.
    Dùng quy hoạch động 1 hàng để tiết kiệm bộ nhớ.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1,        # xóa
                           cur[j - 1] + 1,      # chèn
                           prev[j - 1] + cost)) # thay
        prev = cur
    return prev[-1]


def anls(pred: str, gts: List[str], threshold: float = 0.5) -> float:
    """
    ANLS = độ tương đồng chuỗi chuẩn hóa, đặc biệt hợp cho ViTextVQA (đọc chữ trong
    ảnh) vì cho điểm một phần khi OCR/đáp án chỉ SAI VÀI KÝ TỰ (vd thiếu 1 dấu).

    Với mỗi đáp án đúng g:
        NL(pred,g) = 1 - edit_distance(pred,g) / max(len(pred), len(g))
    Lấy giá trị tốt nhất trên tất cả gts. Nếu < threshold (mặc định 0.5) -> tính 0
    (coi như sai hẳn). Đây đúng theo định nghĩa ANLS trong các benchmark TextVQA.
    """
    p = normalize_vi(pred)
    best = 0.0
    for g in gts:
        g = normalize_vi(g)
        denom = max(len(p), len(g))
        nl = 1.0 - _levenshtein(p, g) / denom if denom > 0 else 0.0
        best = max(best, nl)
    return best if best >= threshold else 0.0


# =============================================================================
# 5) TỔNG HỢP TRÊN CẢ TẬP TEST
# =============================================================================
# Cho phép bật/tắt từng metric qua config. Chữ ký thống nhất (pred, gts, **kw).
_METRIC_FUNCS: Dict[str, Callable] = {
    "exact_match": lambda p, g, **kw: exact_match(p, g),
    "vqa_accuracy": lambda p, g, **kw: vqa_accuracy(p, g),
    "anls": lambda p, g, **kw: anls(p, g, threshold=kw.get("anls_threshold", 0.5)),
}


def evaluate_predictions(
    predictions: List[str],
    references: List[List[str]],
    metric_names: List[str] = ("exact_match", "vqa_accuracy", "anls"),
    anls_threshold: float = 0.5,
    question_types: List[str] | None = None,
) -> Dict[str, float]:
    """
    Tính điểm trung bình toàn tập, và (tùy chọn) tách theo loại câu hỏi.

    Args:
        predictions: list câu trả lời của model, mỗi phần tử là 1 string.
        references:  list song song, mỗi phần tử là LIST các đáp án đúng.
        metric_names: metric nào cần tính (khớp key trong _METRIC_FUNCS).
        anls_threshold: ngưỡng cho ANLS.
        question_types: (tùy chọn) nhãn loại câu hỏi để phục vụ ERROR ANALYSIS
                        mà PDF bắt buộc (yes/no, counting, text-reading, reasoning).

    Returns:
        dict {metric: điểm_trung_bình}, kèm các khóa "{metric}__{qtype}" nếu có
        question_types.
    """
    assert len(predictions) == len(references), "pred và ref phải cùng độ dài"

    n = len(predictions)
    totals = {m: 0.0 for m in metric_names}
    # Gộp điểm theo loại câu hỏi -> phục vụ error analysis
    by_type: Dict[str, Dict[str, float]] = {}
    by_type_count: Dict[str, int] = {}

    for i in range(n):
        pred, gts = predictions[i], references[i]
        qtype = question_types[i] if question_types else None
        if qtype is not None:
            by_type.setdefault(qtype, {m: 0.0 for m in metric_names})
            by_type_count[qtype] = by_type_count.get(qtype, 0) + 1
        for m in metric_names:
            score = _METRIC_FUNCS[m](pred, gts, anls_threshold=anls_threshold)
            totals[m] += score
            if qtype is not None:
                by_type[qtype][m] += score

    results = {m: (totals[m] / n if n else 0.0) for m in metric_names}
    # Bổ sung điểm theo từng loại câu hỏi
    for qtype, sums in by_type.items():
        c = by_type_count[qtype]
        for m in metric_names:
            results[f"{m}__{qtype}"] = sums[m] / c if c else 0.0
    return results


# --- Smoke test nhanh: chạy `python -m src.eval.metrics` để kiểm tra logic ---
if __name__ == "__main__":
    preds = ["con mèo", "hai người", "biển báo dừng lại"]
    refs = [["con mèo", "mèo"], ["2 người", "hai người"], ["dừng lại"]]
    qtypes = ["what", "counting", "text-reading"]
    out = evaluate_predictions(preds, refs, question_types=qtypes)
    for k, v in out.items():
        print(f"{k:35s} = {v:.3f}")
