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
# -----------------------------------------------------------------------------
# Lớp này áp ĐỐI XỨNG cho pred lẫn gold. Chỉ gộp CÁCH VIẾT CÙNG NGHĨA
# (hoa/thường, tiền tố "màu ", số đơn lẻ chữ <-> chữ số).
# KHÔNG alias màu. KHÔNG thay số nằm lẫn trong câu nhiều từ.
# Phần gần đúng (vd "xanh lá cây" vs "xanh lá") để ANLS lo — EM giữ chặt.
# =============================================================================

# Tiền tố từ chỉ loại — CHỈ bỏ khi nằm ở ĐẦU chuỗi (sau khi lower + gộp space).
_TYPE_PREFIXES = ("màu ",)

# Số 0..20 và chục tròn 30,40,...,100: cả chữ lẫn chữ số map về CÙNG dạng (chữ số).
# CHỈ dùng khi TOÀN BỘ chuỗi (sau prefix+strip) khớp một khóa — kể cả cụm 2 từ
# như "mười một". Không thay token số nằm trong câu dài.
_NUMBER_WORD_TO_DIGIT: Dict[str, str] = {
    "không": "0",
    "một": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4",
    "tư": "4",
    "năm": "5",
    "lăm": "5",
    "sáu": "6",
    "bảy": "7",
    "tám": "8",
    "chín": "9",
    "mười": "10",
    "mười một": "11",
    "mười hai": "12",
    "mười ba": "13",
    "mười bốn": "14",
    "mười lăm": "15",
    "mười sáu": "16",
    "mười bảy": "17",
    "mười tám": "18",
    "mười chín": "19",
    "hai mươi": "20",
    "ba mươi": "30",
    "bốn mươi": "40",
    "năm mươi": "50",
    "sáu mươi": "60",
    "bảy mươi": "70",
    "tám mươi": "80",
    "chín mươi": "90",
    "một trăm": "100",
    "trăm": "100",
}
# Chữ số cũng là khóa hợp lệ → giữ nguyên (hai chiều cùng về digit).
for _d in list(_NUMBER_WORD_TO_DIGIT.values()):
    _NUMBER_WORD_TO_DIGIT.setdefault(_d, _d)


def _strip_type_prefix(text: str) -> str:
    """Bỏ tiền tố từ chỉ loại ở ĐẦU chuỗi (vd 'màu đỏ' → 'đỏ'); không đụng nếu nằm giữa."""
    for prefix in _TYPE_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _canonical_number(token: str) -> str:
    """Map 1 token/cụm số (toàn chuỗi) về chữ số; không phải số đứng một mình thì giữ nguyên."""
    return _NUMBER_WORD_TO_DIGIT.get(token, token)


def normalize_vi(text: str) -> str:
    """
    Chuẩn hóa 1 chuỗi trước khi so khớp (áp đối xứng pred và gold).

    Các bước:
        - unicode NFC: gộp ký tự + dấu thành 1 code point thống nhất.
        - về chữ thường (.lower): "Đỏ" → "đỏ" TRƯỚC mọi so khớp.
        - bỏ dấu câu NHƯNG GIỮ dấu thanh tiếng Việt.
        - gộp khoảng trắng thừa.
        - bỏ tiền tố "màu " ở ĐẦU chuỗi ("màu đỏ" → "đỏ").
        - nếu TOÀN BỘ chuỗi còn lại là một số duy nhất (0..20 hoặc chục tròn
          30..100, chữ hoặc chữ số) thì map về digit. Không đụng số lẫn trong câu
          (tránh "chụp năm nào" → "chụp 5 nào").

    Không alias màu: "xanh lá cây" ≠ "xanh lá" ở EM; ANLS lo phần gần đúng.

    LƯU Ý: ta KHÔNG bỏ dấu tiếng Việt. "con mèo" != "con meo".
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower().strip()  # "Đỏ" → "đỏ" trước mọi so khớp
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)  # bỏ dấu câu
    text = re.sub(r"\s+", " ", text).strip()                # gộp space
    text = _strip_type_prefix(text)
    text = _canonical_number(text)  # no-op trừ khi cả chuỗi là một số
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
    # EM chặt: chỉ MATCH khi cùng nghĩa sau NFC/lower/prefix/số-đơn.
    must_em_1 = [
        ("Đỏ", "màu đỏ"),
        ("2", "hai"),
        ("mười một", "11"),
    ]
    must_em_0 = [
        ("màu vàng", "phòng"),
        ("xanh lá cây", "màu xanh lá"),
        ("con ngựa nhỏ", "con ngựa"),
        ("chụp năm nào", "chụp 5 nào"),  # không đổi số lẫn trong câu
    ]
    must_anls_ge_half = [
        ("xanh lá cây", "màu xanh lá"),
        ("con ngựa nhỏ", "con ngựa"),
    ]

    for pred, gold in must_em_1:
        em = exact_match(pred, [gold])
        assert em == 1.0, (
            f"EXPECT EM=1: {pred!r} vs {gold!r} -> {em} "
            f"(norm={normalize_vi(pred)!r}/{normalize_vi(gold)!r})"
        )
        print(f"EM=1  {pred!r:22s} ~ {gold!r:22s}  norm={normalize_vi(pred)!r}")

    for pred, gold in must_em_0:
        em = exact_match(pred, [gold])
        assert em == 0.0, f"EXPECT EM=0: {pred!r} vs {gold!r} -> {em}"
        print(f"EM=0  {pred!r:22s} / {gold!r:22s}")

    for pred, gold in must_anls_ge_half:
        em = exact_match(pred, [gold])
        nl = anls(pred, [gold])
        assert em == 0.0 and nl >= 0.5, (
            f"EXPECT EM=0 ANLS>=0.5: {pred!r} vs {gold!r} -> EM={em} ANLS={nl}"
        )
        print(f"ANLS  {pred!r:22s} / {gold!r:22s}  EM={em:.0f} ANLS={nl:.3f}")

    preds = ["con mèo", "hai người", "biển báo dừng lại"]
    refs = [["con mèo", "mèo"], ["2 người", "hai người"], ["dừng lại"]]
    qtypes = ["what", "counting", "text-reading"]
    out = evaluate_predictions(preds, refs, question_types=qtypes)
    for k, v in out.items():
        print(f"{k:35s} = {v:.3f}")
