"""
core.py — Lõi của prototype ViVQA-VLM (W13): MỘT đường suy luận dùng chung cho demo web và cho kiểm chứng.

Vấn đề: prototype phải trả lời GIỐNG HỆT pipeline đã báo cáo (prompt, OCR, adapter, cách sinh), nếu không thì số
liệu trong khoá luận không còn đúng cho bản demo; đồng thời phải hiện độ tin cậy và xử lý thiên lệch "có" của SH.

Thành phần (đều kiểm offline được):
  - build_inputs / generate_from_inputs: y hệt src.eval.run_eval.generate_answer (cùng messages, processor,
    model.generate(max_new_tokens, do_sample=False)), chỉ bật thêm output_scores để lấy log-prob từng token.
  - open_confidence: exp(trung bình log-prob các token trả lời, bỏ token đặc biệt) — theo phân phối mà greedy
    thật sự dùng (scores sau repetition penalty).
  - is_yes_no_question (detector yesno-detector-v1, thiết kế CHỈ trên ViVQA train + ViTextVQA dev, chốt trước test)
    + normalize_yes_no: câu có/không -> thêm chỉ dẫn POPE-vi -> quyết định bằng margin token đầu so với τ* (R-YB v3).
  - Backend: gói tất cả cho giao diện (bước 7): nạp model + adapter SH, OCR, trả lời + độ tin cậy + văn bản OCR.
"""
from __future__ import annotations

import math
import re
import time
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Hằng số chốt (đổi = run khác; ghi vào chữ ký run)
# --------------------------------------------------------------------------- #
CORE_VERSION = "prototype-core-v1"
DETECTOR_VERSION = "yesno-detector-v1"
YESNO_INSTRUCTION = 'Trả lời "có" hoặc "không".'          # đúng đuôi câu của POPE-vi (src.eval.pope_vi.QUESTION_TEMPLATE)
TAU_STAR_SH = 4.85                                        # R-YB v3 (27/10/2026), chọn trên calibration, SH
SH_REF_RUN = "W11_SH_eval_qlora_ocr_vietocr_v1_r16_shared_ep1_vivqa_test_bf16_L4_seed42"

# Detector v1 — xây trên câu hỏi ViVQA train + ViTextVQA dev (KHÔNG nhìn test):
#   câu kết thúc bằng một đuôi có/không, KHÔNG chứa từ để hỏi, và "không" cuối câu không thuộc danh từ ghép
#   ("hàng không", "trên không", "hư không", ...).
_YN_ENDINGS = ("phải không", "đúng không", "hay không", "không", "chưa")
_NOUN_KHONG_PREV = {"hàng", "trên", "hư", "khoảng", "chân", "tầng", "giữa"}
_WH_WORDS = ("bao nhiêu", "bao giờ", "bao lâu", "thế nào", "như thế nào", "khi nào", "tại sao", "vì sao",
             "làm sao", "gì", "nào", "đâu", "ai", "mấy", "sao")


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFC", str(text or "")).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE).split())


def strip_instruction(question: str) -> str:
    q = unicodedata.normalize("NFC", str(question or "")).strip()
    if q.endswith(YESNO_INSTRUCTION):
        q = q[: -len(YESNO_INSTRUCTION)].rstrip()
    return q


def is_yes_no_question(question: str) -> bool:
    """yesno-detector-v1 (xem chú thích ở trên). Bỏ đuôi chỉ dẫn trước khi xét."""
    words = _norm(strip_instruction(question)).split()
    if not words:
        return False
    text = " ".join(words)
    if not any(text == e or text.endswith(" " + e) for e in _YN_ENDINGS):
        return False
    if words[-1] == "không" and len(words) >= 2 and words[-2] in _NOUN_KHONG_PREV:
        return False
    padded = f" {text} "
    if any(f" {w} " in padded for w in _WH_WORDS):
        return False
    return True


def normalize_yes_no(question: str) -> str:
    """Câu có/không -> '<câu>? Trả lời "có" hoặc "không".' (giữ nguyên nếu đã có đuôi chỉ dẫn)."""
    q = strip_instruction(question).rstrip(" ?.")
    return f"{q}? {YESNO_INSTRUCTION}"


# --------------------------------------------------------------------------- #
# Sinh câu trả lời kèm log-prob (y hệt generate_answer)
# --------------------------------------------------------------------------- #
def build_inputs(processor, image_path: str, prompt_text: str, device=None):
    """Đúng các bước 1–3 của src.eval.run_eval.generate_answer."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from src.prompting.builder import to_chat_messages
    image = Image.open(image_path).convert("RGB")
    messages = to_chat_messages(image, prompt_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return inputs.to(device) if device is not None else inputs


def generate_from_inputs(model, inputs, max_new_tokens: int = 32) -> Tuple[List[int], List[float]]:
    """model.generate như generate_answer + output_scores. Trả (token mới sinh, log-prob của từng token đó
    theo scores ĐÃ qua bộ xử lý logit — chính phân phối greedy dùng, kể cả repetition penalty)."""
    import torch
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    n_in = inputs["input_ids"].shape[1]
    new_ids = out.sequences[0][n_in:].tolist()
    lps = []
    for step, tok in enumerate(new_ids):
        lps.append(float(torch.log_softmax(out.scores[step][0].float(), dim=-1)[tok]))
    return new_ids, lps


def open_confidence(token_ids: Sequence[int], logprobs: Sequence[float], special_ids: Sequence[int]) -> Dict[str, Any]:
    """conf = exp(trung bình log-prob các token KHÔNG đặc biệt). Không còn token nào -> conf 0."""
    sp = set(int(i) for i in special_ids)
    kept = [lp for t, lp in zip(token_ids, logprobs) if int(t) not in sp]
    if not kept:
        return {"conf": 0.0, "n_tokens": 0, "mean_logp": None}
    m = sum(kept) / len(kept)
    return {"conf": math.exp(m), "n_tokens": len(kept), "mean_logp": m}


def yes_no_confidence(margin: float, tau: float) -> float:
    """Độ tin cậy của quyết định có/không: sigmoid(|margin − τ|) ∈ [0,5; 1)."""
    return 1.0 / (1.0 + math.exp(-abs(margin - tau)))


def conf_level(conf: float, cuts: Tuple[float, float]) -> str:
    lo, hi = cuts
    return "Cao" if conf >= hi else ("Trung bình" if conf >= lo else "Thấp")


def _cuda_sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def answer_one(model, processor, image_path: str, question: str, ocr_texts: Optional[List[str]],
               prompting: Dict[str, Any], yn_tokens: Dict[str, Any], penalty: float, tau: float = TAU_STAR_SH,
               special_ids: Sequence[int] = (), max_new_tokens: int = 32) -> Dict[str, Any]:
    """Một câu hỏi qua đúng đường của prototype. Câu thường: greedy như eval. Câu có/không: margin vs τ."""
    from src.prompting.builder import prompt_for_sample
    from src.eval.yesno_calib import score_sample
    sample = {"question": question, "ocr_texts": list(ocr_texts or []), "question_id": "proto"}
    t0 = time.perf_counter()
    if is_yes_no_question(question):
        _cuda_sync()
        t_pre0 = time.perf_counter()
        prompt = prompt_for_sample({**sample, "question": normalize_yes_no(question)}, prompting)
        t_preprocess = time.perf_counter() - t_pre0
        _cuda_sync()
        t_gen0 = time.perf_counter()
        r = score_sample(model, processor, image_path, prompt, yn_tokens, penalty)
        _cuda_sync()
        t_model = time.perf_counter() - t_gen0
        t_post0 = time.perf_counter()
        ans = "có" if r["margin"] > tau else "không"
        conf = yes_no_confidence(r["margin"], tau)
        t_postprocess = time.perf_counter() - t_post0
        return {"mode": "yesno", "answer": ans, "confidence": conf,
                "margin": r["margin"], "prompt": prompt, "sec": time.perf_counter() - t0,
                "sec_preprocess": t_preprocess, "sec_model": t_model, "sec_postprocess": t_postprocess,
                "n_tokens": 0}
    _cuda_sync()
    t_pre0 = time.perf_counter()
    prompt = prompt_for_sample(sample, prompting)
    inputs = build_inputs(processor, image_path, prompt, device=model.device)
    _cuda_sync()
    t_preprocess = time.perf_counter() - t_pre0
    _cuda_sync()
    t_gen0 = time.perf_counter()
    ids, lps = generate_from_inputs(model, inputs, max_new_tokens)
    _cuda_sync()
    t_model = time.perf_counter() - t_gen0
    t_post0 = time.perf_counter()
    ans = processor.batch_decode([ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    c = open_confidence(ids, lps, special_ids)
    t_postprocess = time.perf_counter() - t_post0
    return {"mode": "open", "answer": ans, "confidence": c["conf"], "n_tokens": c["n_tokens"],
            "prompt": prompt, "sec": time.perf_counter() - t0,
            "sec_preprocess": t_preprocess, "sec_model": t_model, "sec_postprocess": t_postprocess}


# --------------------------------------------------------------------------- #
# Backend cho giao diện (bước 7) — nạp một lần, trả lời nhiều lần
# --------------------------------------------------------------------------- #
class Backend:
    def __init__(self, adapter_dir: str, config_path: str, ocr_fn: Callable[[str], List[Dict[str, Any]]],
                 ocr_template: str = "v1", ocr_max_chars: int = 600, ocr_min_conf: float = 0.0,
                 tau: float = TAU_STAR_SH, conf_cuts_open: Tuple[float, float] = (0.0, 1.0),
                 conf_cuts_yn: Tuple[float, float] = (0.5, 1.0)):
        from peft import PeftModel
        from src.eval.run_baseline import load_config, load_model
        from src.eval.yesno_calib import candidate_tokens, generation_penalty
        self.cfg = load_config(config_path, mode="ocr")
        self.cfg["prompting"]["ocr"] = {"template": ocr_template}
        model, self.processor = load_model(self.cfg)
        self.model = PeftModel.from_pretrained(model, adapter_dir).eval()
        self.penalty, _ = generation_penalty(self.model)
        self.yn_tokens = candidate_tokens(self.processor.tokenizer)
        self.special_ids = list(self.processor.tokenizer.all_special_ids)
        self.ocr_fn, self.ocr_max_chars, self.ocr_min_conf = ocr_fn, ocr_max_chars, ocr_min_conf
        self.tau, self.cuts_open, self.cuts_yn = tau, conf_cuts_open, conf_cuts_yn

    def ask(self, image_path: str, question: str,
            ocr_lines: Optional[List[Dict[str, Any]]] = None,
            ocr_timing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """ocr_lines/ocr_timing: dùng lại OCR prefetch/cache — cùng answer_one, không đổi kết quả."""
        from src.data.ocr import lines_to_texts
        t0 = time.perf_counter()
        ocr_timing = dict(ocr_timing or {})
        if ocr_lines is not None:
            lines = ocr_lines
            t_ocr_det = float(ocr_timing.get("sec_det", 0.0) or 0.0)
            t_ocr_rec = float(ocr_timing.get("sec_rec", 0.0) or 0.0)
            t_ocr = float(ocr_timing.get("sec_ocr", t_ocr_det + t_ocr_rec) or 0.0)
            ocr_source = str(ocr_timing.get("ocr_source", "cache"))
        else:
            t_ocr0 = time.perf_counter()
            read_fn = getattr(self, "ocr_read", None)
            if callable(read_fn):
                raw = read_fn(image_path)
                lines = list(raw.get("lines") or [])
                t_ocr_det = float(raw.get("sec_det") or 0.0)
                t_ocr_rec = float(raw.get("sec_rec") or 0.0)
                t_ocr = float(raw.get("sec") or (time.perf_counter() - t_ocr0))
            else:
                lines = self.ocr_fn(image_path)
                t_ocr_det = t_ocr_rec = 0.0
                t_ocr = time.perf_counter() - t_ocr0
            ocr_source = "fresh"
        texts, _ = lines_to_texts({"lines": lines}, self.ocr_max_chars, self.ocr_min_conf)
        r = answer_one(self.model, self.processor, image_path, question, texts, self.cfg["prompting"],
                       self.yn_tokens, self.penalty, self.tau, self.special_ids,
                       self.cfg["prompting"].get("max_new_tokens", 32))
        cuts = self.cuts_yn if r["mode"] == "yesno" else self.cuts_open
        t_post0 = time.perf_counter()
        level = conf_level(r["confidence"], cuts)
        t_level = time.perf_counter() - t_post0
        return {**r, "level": level, "ocr_lines": lines, "ocr_texts": texts,
                "sec_ocr": t_ocr, "sec_ocr_det": t_ocr_det, "sec_ocr_rec": t_ocr_rec,
                "sec_preprocess": float(r.get("sec_preprocess") or 0.0),
                "sec_model": float(r.get("sec_model") or 0.0),
                "sec_postprocess": float(r.get("sec_postprocess") or 0.0) + t_level,
                "ocr_source": ocr_source, "n_gen_tokens": int(r.get("n_tokens") or 0),
                "n_ocr_lines": len(lines), "sec_total": time.perf_counter() - t0}
