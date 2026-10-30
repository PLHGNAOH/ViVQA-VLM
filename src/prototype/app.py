"""
app.py — Giao diện web (Gradio) của prototype ViVQA-VLM (W13 bước 7).

Người dùng tải ảnh + gõ câu hỏi tiếng Việt -> câu trả lời, MỨC tin cậy, chữ OCR làm bằng chứng, cảnh báo.

Nguyên tắc (từ kết quả đã kiểm ở W13, docs/W13/README.md):
  - Trả lời bằng ĐÚNG đường đã kiểm parity (src.prototype.core.Backend, R-PAR 100%): cùng prompt, OCR, adapter SH,
    cách sinh. Ảnh dataset đi vào NGUYÊN FILE (không mã hoá lại); chỉ ảnh có cờ xoay EXIF hoặc quá lớn mới được
    xoay/thu nhỏ và lưu PNG (không mất thêm chất lượng).
  - Độ tin cậy: ECE > 0,05 (R-CONF) -> CHỈ hiện mức Cao / Trung bình / Thấp, KHÔNG hiện con số, không ghi "% đúng".
    Ngưỡng mức đọc từ prototype_check_summary.json (chọn trên nửa A tập test), không gõ tay.
  - OCR chạy ở tiến trình riêng (src.prototype.ocr_worker) với ĐÚNG engine đã dựng cache; khác engine -> không chạy.
  - Không lưu ảnh / câu hỏi / câu trả lời. Log (tuỳ chọn) chỉ ghi số đo: thời gian, mức, số dòng OCR, kích thước ảnh.

Chạy:
  Backend giả, CPU (xem/duyệt giao diện, 0 GPU):   python -m src.prototype.app --mock
  Model thật (L4, notebook 24):                      python -m src.prototype.app --adapter <SH/epoch_1> --share
  Kiểm thử offline:                                  python -m src.prototype.app --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

APP_VERSION = "prototype-app-v1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_REL = "experiments/W13_prototype_check_L4_seed42/prototype_check_summary.json"
OCR_SELECTION_REL = "experiments/W08_ocr_select_dev300_L4_seed42/ocr_selection.json"
ECE_SHOW_NUMBER = 0.05          # luật R-CONF: chỉ hiện con số khi ECE ≤ 0,05
MAX_SIDE = 2048                 # ảnh dài hơn -> thu nhỏ (ảnh ViVQA/ViTextVQA đều nhỏ hơn, nên đi nguyên file)
MAX_QUESTION_CHARS = 300
LEVELS = ("Cao", "Trung bình", "Thấp")
LEVEL_COLORS = {"Cao": ("#0f5132", "#d1e7dd"), "Trung bình": ("#664d03", "#fff3cd"), "Thấp": ("#842029", "#f8d7da")}

DISCLAIMER = (
    "<b>Prototype nghiên cứu</b> (khoá luận ViVQA-VLM, ĐH FPT). Model có thể trả lời sai hoặc mô tả thứ không có "
    "trong ảnh (hallucination). Không dùng kết quả cho quyết định quan trọng (y tế, pháp lý, tài chính, an toàn). "
    "Ảnh và câu hỏi chỉ được xử lý trong phiên chạy này; nhóm không thu thập hay lưu lại."
)
CONF_EXPLAIN = (
    "**Mức tin cậy** lấy từ xác suất model gán cho chính câu trả lời của nó, rồi chia thành 3 mức theo ngưỡng chọn "
    "trên một nửa tập test (chia theo ảnh). Trên nửa còn lại, câu ở mức **Cao** đúng nhiều nhất và câu ở mức "
    "**Thấp** sai nhiều nhất, ở cả ViVQA lẫn ViTextVQA.\n\n"
    "Giao diện **không hiện con số** vì model thường *tự tin quá mức*: một con số như \"97%\" sẽ khiến người dùng "
    "tin nhiều hơn mức đáng tin.\n\n"
    "**Câu hỏi có/không** (vd \"Trong ảnh có con mèo không?\") đi đường riêng: hệ thống thêm chỉ dẫn "
    "`Trả lời \"có\" hoặc \"không\".` và quyết định bằng một ngưỡng đã hiệu chỉnh, để giảm xu hướng trả lời "
    "\"có\" khi vật không có trong ảnh."
)
EMPTY_ANSWER = "<div class='vv-card vv-empty'>Tải ảnh lên, gõ câu hỏi rồi bấm <b>Trả lời</b>.</div>"
CSS = """
.vv-card{border:1px solid #d0d7de;border-radius:12px;padding:16px 18px;background:#fff;color:#1f2328}
.vv-empty{color:#57606a}
.vv-ans{font-size:1.6em;font-weight:700;margin:4px 0 10px;line-height:1.3;word-break:break-word}
.vv-badge{display:inline-block;border-radius:999px;padding:3px 12px;font-weight:600;font-size:.9em;margin-right:8px}
.vv-mode{color:#57606a;font-size:.9em}
.vv-mock{background:#fff3cd;color:#664d03;border:1px solid #ffda6a;border-radius:8px;padding:8px 12px;margin:6px 0}
.vv-disc{color:#57606a;font-size:.85em;border-top:1px solid #d0d7de;padding-top:8px;margin-top:8px}
.vv-ocr{width:100%;border-collapse:collapse;font-size:.92em}
.vv-ocr td{border-bottom:1px solid #eaeef2;padding:4px 6px;vertical-align:top}
.vv-ocr tr.ev td{background:#d1e7dd}
.vv-ocr tr.cut td{color:#8c959f}
@media (prefers-color-scheme: dark){.vv-card{background:#161b22;color:#e6edf3;border-color:#30363d}
 .vv-ocr td{border-color:#30363d}.vv-ocr tr.ev td{background:#0f5132;color:#fff}}
"""


# =============================================================================
# 1) THIẾT LẬP ĐỌC TỪ KẾT QUẢ ĐÃ KIỂM (không gõ tay ngưỡng)
# =============================================================================
def load_ui_settings(summary_path: Optional[str] = None) -> Dict[str, Any]:
    """Ngưỡng mức tin cậy + quyền hiện con số, đọc từ prototype_check_summary.json (notebook 23)."""
    from src.prototype.core import TAU_STAR_SH
    p = summary_path or os.path.join(REPO_ROOT, SUMMARY_REL)
    with open(p, encoding="utf-8") as f:
        s = json.load(f)
    if s.get("errors"):
        raise ValueError(f"summary có lỗi: {s['errors']}")
    if abs(float(s["rules"]["tau_star_sh"]) - TAU_STAR_SH) > 1e-9:
        raise ValueError(f"τ* trong summary ({s['rules']['tau_star_sh']}) ≠ core ({TAU_STAR_SH})")
    ece_open = float(s["confidence"]["ece_B_pooled"])
    ece_yn = float(s["yesno_confidence"]["ece_popevi"])
    return {
        "summary_path": os.path.relpath(p, REPO_ROOT) if p.startswith(REPO_ROOT) else p,
        "verdicts": dict(s["verdicts"]),
        "tau": float(s["rules"]["tau_star_sh"]),
        "cuts_open": tuple(float(x) for x in s["confidence"]["cuts_open"]),
        "cuts_yn": tuple(float(x) for x in s["yesno_confidence"]["cuts_yn"]),
        "ece_open": ece_open, "ece_yn": ece_yn,
        "show_number_open": bool(s["confidence"].get("ui_shows_number")) and ece_open <= ECE_SHOW_NUMBER,
        "show_number_yn": ece_yn <= ECE_SHOW_NUMBER,
    }


def require_verified(settings: Dict[str, Any]) -> None:
    """Model thật chỉ chạy khi cả 3 luật W13 bước 4–6 ĐẠT (parity, mức tin cậy, detector có/không)."""
    bad = {k: v for k, v in settings["verdicts"].items() if v != "ĐẠT"}
    need = {"R-PAR", "R-CONF", "R-YB-det"} - set(settings["verdicts"])
    if bad or need:
        raise SystemExit(f"❌ Không chạy model thật: luật chưa ĐẠT {bad or ''} {sorted(need) or ''}")


def load_ocr_selection(path: Optional[str] = None) -> Dict[str, Any]:
    with open(path or os.path.join(REPO_ROOT, OCR_SELECTION_REL), encoding="utf-8") as f:
        return json.load(f)["decision"]


# =============================================================================
# 2) ẢNH ĐẦU VÀO
# =============================================================================
def prepare_image(path: str, out_dir: str, max_side: int = MAX_SIDE) -> Tuple[str, Dict[str, Any]]:
    """
    Ảnh bình thường (không cờ xoay EXIF, ≤ max_side, JPEG/PNG/WEBP) -> dùng NGUYÊN file (giống lúc eval).
    Ảnh điện thoại có cờ xoay / quá lớn -> xoay đúng chiều, thu nhỏ, lưu PNG tạm. Trả (đường dẫn, thông tin).
    """
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        with Image.open(path) as im:
            fmt, size, orient = im.format, im.size, im.getexif().get(274, 1)
            if orient in (None, 1) and max(size) <= max_side and fmt in ("JPEG", "PNG", "WEBP"):
                return path, {"orig_size": list(size), "used_size": list(size), "rotated": False,
                              "resized": False, "reencoded": False, "format": fmt}
            im2 = ImageOps.exif_transpose(im) if orient not in (None, 1) else im.copy()
    except UnidentifiedImageError:
        raise ValueError("Không đọc được ảnh (định dạng chưa hỗ trợ, vd HEIC của iPhone). Hãy dùng JPG hoặc PNG.")
    im2 = im2.convert("RGB")
    resized = max(im2.size) > max_side
    if resized:
        im2.thumbnail((max_side, max_side), Image.LANCZOS)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"vv_{uuid.uuid4().hex}.png")
    im2.save(out)
    return out, {"orig_size": list(size), "used_size": list(im2.size), "rotated": orient not in (None, 1),
                 "resized": resized, "reencoded": True, "format": fmt}


# =============================================================================
# 3) HIỂN THỊ KẾT QUẢ (hàm thuần, kiểm offline được)
# =============================================================================
def evidence_lines(answer: str, lines: List[Dict[str, Any]]) -> List[int]:
    """Chỉ số các dòng OCR chứa câu trả lời (hoặc nằm trong câu trả lời), sau chuẩn hoá chữ thường/dấu câu."""
    from src.prototype.core import _norm
    a = _norm(answer)
    if len(a) < 2:
        return []
    out = []
    for i, ln in enumerate(lines):
        t = _norm(ln.get("text", ""))
        if t and (f" {a} " in f" {t} " or (len(t) >= 4 and f" {t} " in f" {a} ")):
            out.append(i)
    return out


def draw_ocr_boxes(image_path: str, lines: List[Dict[str, Any]], highlight: List[int]):
    """Ảnh có khung + số thứ tự đọc cho từng dòng OCR (xanh = dòng khớp câu trả lời)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(im.size) / 300))
    try:
        font = ImageFont.load_default(size=max(14, round(max(im.size) / 22)))
    except Exception:                                     # Pillow cũ / không có FreeType
        font = ImageFont.load_default()
    for i, ln in enumerate(lines):
        x0, y0, x1, y1 = [int(v) for v in ln["box"]]
        col = (25, 135, 84) if i in highlight else (253, 126, 20)
        d.rectangle([x0, y0, x1, y1], outline=col, width=lw)
        tag = str(i + 1)
        tb = d.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y0 - th - 2 * lw if y0 - th - 2 * lw >= 0 else y1
        d.rectangle([x0, ty, x0 + tw + 2 * lw, ty + th + 2 * lw], fill=col)
        d.text((x0 + lw, ty + lw - tb[1]), tag, fill=(255, 255, 255), font=font)
    return im


def _badge(level: str) -> str:
    fg, bg = LEVEL_COLORS[level]
    return f"<span class='vv-badge' style='color:{fg};background:{bg}'>Độ tin cậy: {html.escape(level)}</span>"


def render_result(r: Dict[str, Any], img_info: Dict[str, Any], settings: Dict[str, Any],
                  ocr_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Kết quả Backend.ask -> các phần hiển thị. KHÔNG có con số độ tin cậy khi luật không cho phép."""
    from src.data.ocr import lines_to_texts
    lines = r.get("ocr_lines") or []
    _, truncated = lines_to_texts({"lines": lines}, ocr_cfg["max_chars"], ocr_cfg["min_conf"])
    n_used = len(r.get("ocr_texts") or [])
    ev = evidence_lines(r["answer"], lines)
    show_num = settings["show_number_yn"] if r["mode"] == "yesno" else settings["show_number_open"]
    num = f" <span class='vv-mode'>({100 * r['confidence']:.0f}%)</span>" if show_num else ""
    mode_txt = "Câu hỏi có/không · quyết định bằng ngưỡng đã hiệu chỉnh" if r["mode"] == "yesno" else "Câu hỏi mở"
    answer_html = (f"<div class='vv-card'><div class='vv-mode'>Câu trả lời</div>"
                   f"<div class='vv-ans'>{html.escape(r['answer']) or '<i>(model không trả lời)</i>'}</div>"
                   f"{_badge(r['level'])}{num}<span class='vv-mode'>{mode_txt}</span></div>")

    warns = []
    if r["level"] == "Thấp":
        warns.append("⚠️ **Độ tin cậy thấp** — câu trả lời này dễ sai, hãy tự đối chiếu với ảnh.")
    if not r["answer"].strip():
        warns.append("⚠️ Model không đưa ra câu trả lời nào.")
    if not lines:
        warns.append("ℹ️ **Không đọc được chữ nào trong ảnh.** Nếu câu hỏi liên quan tới chữ (biển hiệu, giá, tên, "
                     "số điện thoại…), câu trả lời có thể không đáng tin.")
    elif ev:
        warns.append(f"✅ Câu trả lời khớp với chữ đọc được trong ảnh (dòng {', '.join(str(i + 1) for i in ev)}).")
    if truncated:
        warns.append(f"ℹ️ Ảnh có nhiều chữ: chỉ {n_used}/{len(lines)} dòng đầu (≤ {ocr_cfg['max_chars']} ký tự theo "
                     "thứ tự đọc) được đưa cho model.")
    if r["mode"] == "yesno":
        warns.append("ℹ️ Câu hỏi có/không: hệ thống thêm chỉ dẫn `Trả lời \"có\" hoặc \"không\".` và dùng ngưỡng "
                     "đã hiệu chỉnh để giảm xu hướng trả lời \"có\" khi vật không có trong ảnh.")
    if img_info.get("rotated") or img_info.get("resized"):
        what = " và ".join(x for x, ok in (("xoay theo EXIF", img_info.get("rotated")),
                                           (f"thu nhỏ về cạnh dài {MAX_SIDE} px", img_info.get("resized"))) if ok)
        warns.append(f"ℹ️ Ảnh đã được {what} trước khi xử lý.")

    if lines:
        rows = []
        for i, ln in enumerate(lines):
            cls = "ev" if i in ev else ("cut" if i >= n_used else "")
            note = " · khớp câu trả lời" if i in ev else (" · không đưa vào model (vượt giới hạn)" if i >= n_used else "")
            rows.append(f"<tr class='{cls}'><td>{i + 1}</td><td>{html.escape(ln['text'])}"
                        f"<span class='vv-mode'>{note}</span></td></tr>")
        ocr_html = ("<table class='vv-ocr'><tr><td><b>#</b></td><td><b>Chữ đọc được (thứ tự đọc: trên → dưới, "
                    "trái → phải)</b></td></tr>" + "".join(rows) + "</table>")
    else:
        ocr_html = "<div class='vv-mode'>Không có dòng chữ nào.</div>"

    def sec(k):
        return f"{r[k]:.2f} s".replace(".", ",") if isinstance(r.get(k), (int, float)) else "—"
    details = "\n".join([
        f"- **Chế độ:** `{r['mode']}` · mức `{r['level']}`",
        f"- **Thời gian:** OCR {sec('sec_ocr')} · model {sec('sec')} · tổng {sec('sec_total')}",
        f"- **Ảnh:** {img_info['orig_size'][0]}×{img_info['orig_size'][1]} px"
        + (f" → {img_info['used_size'][0]}×{img_info['used_size'][1]} px" if img_info.get("reencoded") else " (dùng nguyên file)"),
        f"- **OCR:** {ocr_cfg['backend']} · template {ocr_cfg['template']} · ≤ {ocr_cfg['max_chars']} ký tự · "
        f"{len(lines)} dòng, {n_used} dòng vào prompt",
        f"- **Ngưỡng câu có/không:** τ* = {str(settings['tau']).replace('.', ',')} (R-YB v3)",
        f"- **Phiên bản:** {APP_VERSION} · ngưỡng mức tin cậy từ `{settings['summary_path']}`",
        "", "**Prompt gửi cho model:**", "```text", r.get("prompt", ""), "```",
    ])
    return {"answer_html": answer_html, "warnings_md": "\n\n".join(warns), "ocr_html": ocr_html,
            "evidence": ev, "details_md": details, "truncated": truncated}


# =============================================================================
# 4) BACKEND GIẢ (CPU, duyệt giao diện) — cùng định dạng kết quả với core.Backend.ask
# =============================================================================
class MockBackend:
    """KHÔNG phải model: câu trả lời cố định theo câu hỏi, chỉ để dựng/duyệt giao diện. Prompt dựng bằng code thật."""
    is_mock = True

    def __init__(self, settings: Dict[str, Any], ocr_cfg: Dict[str, Any]):
        self.settings, self.ocr_cfg = settings, ocr_cfg

    def ask(self, image_path: str, question: str) -> Dict[str, Any]:
        from PIL import Image
        from src.data.ocr import lines_to_texts
        from src.prompting.builder import prompt_for_sample
        from src.prototype.core import conf_level, is_yes_no_question, normalize_yes_no
        t0 = time.time()
        with Image.open(image_path) as im:
            w, h = im.size
        lines = [{"text": "CỬA HÀNG MẪU", "conf": 0.97, "box": [int(.08 * w), int(.10 * h), int(.62 * w), int(.22 * h)]},
                 {"text": "ĐIỆN THOẠI · PHỤ KIỆN", "conf": 0.91, "box": [int(.08 * w), int(.25 * h), int(.70 * w), int(.33 * h)]},
                 {"text": "0123 456 789", "conf": 0.88, "box": [int(.55 * w), int(.80 * h), int(.92 * w), int(.88 * h)]}]
        if "không chữ" in question.lower():
            lines = []
        texts, _ = lines_to_texts({"lines": lines}, self.ocr_cfg["max_chars"], self.ocr_cfg["min_conf"])
        prompting = {"mode": "ocr", "ocr": {"template": self.ocr_cfg["template"]}}
        k = int(hashlib.sha256(question.encode("utf-8")).hexdigest(), 16)
        if is_yes_no_question(question):
            prompt = prompt_for_sample({"question": normalize_yes_no(question), "ocr_texts": texts}, prompting)
            conf = (0.9995, 0.99, 0.80)[k % 3]
            ans, mode, cuts = ("có" if k % 2 else "không"), "yesno", self.settings["cuts_yn"]
        else:
            prompt = prompt_for_sample({"question": question, "ocr_texts": texts}, prompting)
            conf = (0.98, 0.90, 0.60)[k % 3]
            q = question.lower()
            ans = "cửa hàng mẫu" if any(x in q for x in ("tên", "biển", "chữ")) else "màu đỏ"
            mode, cuts = "open", self.settings["cuts_open"]
        sec_ocr = 0.01
        return {"mode": mode, "answer": ans, "confidence": conf, "level": conf_level(conf, cuts), "prompt": prompt,
                "ocr_lines": lines, "ocr_texts": texts, "sec_ocr": sec_ocr, "sec": time.time() - t0,
                "sec_total": time.time() - t0 + sec_ocr}


def make_real_backend(adapter: str, config: str, settings: Dict[str, Any], ocr_cfg: Dict[str, Any],
                      ocr_device: str = "gpu", ocr_python: Optional[str] = None, ocr_log: Optional[str] = None,
                      expect_adapter_sha: Optional[str] = None, allow_ocr_mismatch: bool = False):
    """Model thật: kiểm luật + adapter + engine OCR rồi mới nạp. Trả (backend, ocr_client, thông tin khởi động)."""
    from src.eval.run_baseline import adapter_fingerprint
    from src.prototype.core import Backend
    from src.prototype.ocr_worker import OCRWorkerClient, check_engine_matches_cache
    require_verified(settings)
    t0 = time.time()
    fp = adapter_fingerprint(adapter)
    if expect_adapter_sha and fp["weights_sha256"] != expect_adapter_sha:
        raise SystemExit(f"❌ adapter {fp['weights_sha256'][:12]} ≠ adapter đã kiểm {expect_adapter_sha[:12]}")
    client = OCRWorkerClient(backend=ocr_cfg["backend"], device=ocr_device, python=ocr_python, log_path=ocr_log)
    chk = check_engine_matches_cache(client.info)
    if not chk["ok"] and not allow_ocr_mismatch:
        client.close()
        raise SystemExit(f"❌ engine OCR khác engine đã dựng cache: {chk['diff']}")
    t_ocr = time.time() - t0
    backend = Backend(adapter, config, ocr_fn=client.lines, ocr_template=ocr_cfg["template"],
                      ocr_max_chars=ocr_cfg["max_chars"], ocr_min_conf=ocr_cfg["min_conf"], tau=settings["tau"],
                      conf_cuts_open=settings["cuts_open"], conf_cuts_yn=settings["cuts_yn"])
    backend.is_mock = False
    info = {"adapter": fp, "ocr_engine": client.info, "ocr_check": chk, "sec_start_ocr": round(t_ocr, 1),
            "sec_start_total": round(time.time() - t0, 1)}
    return backend, client, info


# =============================================================================
# 5) GIAO DIỆN
# =============================================================================
class Handler:
    """Xử lý một lượt hỏi: kiểm đầu vào -> chuẩn bị ảnh -> backend -> hiển thị. Ghi log số đo (tuỳ chọn)."""

    def __init__(self, backend, settings: Dict[str, Any], ocr_cfg: Dict[str, Any], log_jsonl: Optional[str] = None,
                 tmp_dir: Optional[str] = None):
        self.backend, self.settings, self.ocr_cfg, self.log_jsonl = backend, settings, ocr_cfg, log_jsonl
        self.tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="vivqa_proto_")

    def _log(self, rec: Dict[str, Any]) -> None:
        if self.log_jsonl:
            with open(self.log_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def __call__(self, image_path: Optional[str], question: Optional[str]):
        from PIL import Image
        q = (question or "").strip()
        err = None
        if not image_path:
            err = "Chưa có ảnh: hãy tải một ảnh lên."
        elif not q:
            err = "Chưa có câu hỏi: hãy gõ câu hỏi bằng tiếng Việt."
        elif len(q) > MAX_QUESTION_CHARS:
            err = f"Câu hỏi quá dài ({len(q)} ký tự, tối đa {MAX_QUESTION_CHARS})."
        if err:
            return f"<div class='vv-card'>⚠️ {html.escape(err)}</div>", "", None, "", ""
        used = None
        rec: Dict[str, Any] = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "app": APP_VERSION}
        try:
            used, info = prepare_image(image_path, self.tmp_dir)
            rec.update({"img_orig": info["orig_size"], "img_used": info["used_size"], "rotated": info["rotated"],
                        "resized": info["resized"]})
            r = self.backend.ask(used, q)
            out = render_result(r, info, self.settings, self.ocr_cfg)
            boxes = draw_ocr_boxes(used, r.get("ocr_lines") or [], out["evidence"]) if r.get("ocr_lines") \
                else Image.open(used).convert("RGB")
            rec.update({"ok": True, "mode": r["mode"], "level": r["level"], "n_ocr_lines": len(r.get("ocr_lines") or []),
                        "ocr_truncated": out["truncated"], "sec_ocr": r.get("sec_ocr"), "sec_vlm": r.get("sec"),
                        "sec_total": r.get("sec_total")})
            self._log(rec)
            return out["answer_html"], out["warnings_md"], boxes, out["ocr_html"], out["details_md"]
        except Exception as e:                            # lỗi hiện cho người dùng, không làm sập app
            rec.update({"ok": False, "error": type(e).__name__})
            self._log(rec)
            return (f"<div class='vv-card'>⚠️ Không xử lý được: {html.escape(str(e)[:400])}</div>", "", None, "", "")
        finally:
            if used and used != image_path and os.path.exists(used):
                os.remove(used)                           # không giữ bản ảnh tạm


def build_demo(handler: Handler, examples: Optional[List[List[str]]] = None):
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")      # không gửi số liệu sử dụng ra ngoài
    import gradio as gr
    is_mock = getattr(handler.backend, "is_mock", False)
    header = ("<h2 style='margin:0'>ViVQA-VLM · Hỏi đáp hình ảnh tiếng Việt</h2>"
              "<div class='vv-mode'>Qwen2.5-VL-3B (4-bit) + adapter QLoRA dùng chung (SH) · đọc chữ trong ảnh bằng "
              "PaddleOCR + VietOCR · trả lời ngắn bằng tiếng Việt</div>")
    if is_mock:
        header += ("<div class='vv-mock'>⚠️ <b>Chế độ thử giao diện (backend giả)</b>: câu trả lời và chữ OCR là "
                   "dữ liệu mẫu cố định, <b>không</b> phải kết quả của model.</div>")
    with gr.Blocks(title="ViVQA-VLM · Hỏi đáp hình ảnh tiếng Việt", analytics_enabled=False) as demo:
        gr.HTML(header)
        with gr.Row():
            with gr.Column(scale=5):
                img_in = gr.Image(type="filepath", image_mode=None, sources=["upload", "clipboard"], label="Ảnh",
                                  height=380)
                q_in = gr.Textbox(label="Câu hỏi (tiếng Việt)", lines=2, max_lines=4,
                                  placeholder="Ví dụ: cửa hàng có biển màu vàng là nơi nào?  ·  Trong ảnh có con mèo không?")
                with gr.Row():
                    btn = gr.Button("Trả lời", variant="primary")
                    clr = gr.Button("Xoá")
                if examples:
                    gr.Examples(examples=examples, inputs=[img_in, q_in], label="Ví dụ")
            with gr.Column(scale=6):
                ans_out = gr.HTML(EMPTY_ANSWER)
                warn_out = gr.Markdown()
                with gr.Accordion("Chữ đọc được trong ảnh (OCR) — bằng chứng", open=True):
                    ocr_img = gr.Image(type="pil", label="Vùng chữ, đánh số theo thứ tự đọc (xanh = khớp câu trả lời)",
                                       interactive=False, height=300, format="png")
                    ocr_out = gr.HTML()
                with gr.Accordion("Chi tiết kỹ thuật", open=False):
                    det_out = gr.Markdown()
                with gr.Accordion("Về mức tin cậy", open=False):
                    gr.Markdown(CONF_EXPLAIN)
        gr.HTML(f"<div class='vv-disc'>{DISCLAIMER}</div>")
        outs = [ans_out, warn_out, ocr_img, ocr_out, det_out]
        btn.click(handler, [img_in, q_in], outs)
        q_in.submit(handler, [img_in, q_in], outs)
        clr.click(lambda: (None, "", EMPTY_ANSWER, "", None, "", ""), None, [img_in, q_in] + outs)
    return demo


def load_examples(path: Optional[str]) -> Optional[List[List[str]]]:
    """File JSON [{"image": đường dẫn, "question": ...}] -> ví dụ cho giao diện (bỏ ảnh không tồn tại)."""
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    ex = [[it["image"], it["question"]] for it in items if os.path.exists(it["image"])]
    return ex or None


# =============================================================================
# 6) KIỂM THỬ OFFLINE
# =============================================================================
def _selftest() -> None:
    import re
    from PIL import Image
    tmp = tempfile.mkdtemp(prefix="vvapp_")
    st = load_ui_settings()
    ocr_cfg = load_ocr_selection()
    # (1) thiết lập đọc đúng từ kết quả notebook 23
    assert st["verdicts"] == {"R-PAR": "ĐẠT", "R-CONF": "ĐẠT", "R-YB-det": "ĐẠT"}, st["verdicts"]
    assert abs(st["cuts_open"][0] - 0.8393393371447103) < 1e-12 and abs(st["cuts_open"][1] - 0.9581924825031203) < 1e-12
    assert abs(st["cuts_yn"][0] - 0.97547745352307) < 1e-12 and st["tau"] == 4.85
    assert st["show_number_open"] is False and st["show_number_yn"] is False     # ECE 0,198 và 0,0505 > 0,05
    require_verified(st)
    try:
        require_verified({**st, "verdicts": {**st["verdicts"], "R-PAR": "KHÔNG ĐẠT"}})
        raise AssertionError("luật chưa ĐẠT phải chặn model thật")
    except SystemExit:
        pass
    assert ocr_cfg["backend"] == "paddledet_vietocr" and ocr_cfg["max_chars"] == 600
    # (2) ảnh: file thường đi nguyên; cờ xoay EXIF -> xoay; ảnh lớn -> thu nhỏ; định dạng lạ -> lỗi rõ
    p_ok = os.path.join(tmp, "ok.jpg")
    Image.new("RGB", (640, 480), (200, 30, 30)).save(p_ok, quality=90)
    raw = open(p_ok, "rb").read()
    used, info = prepare_image(p_ok, tmp)
    assert used == p_ok and not info["reencoded"] and open(used, "rb").read() == raw
    p_rot = os.path.join(tmp, "rot.jpg")
    ex = Image.Exif()
    ex[274] = 6                                            # chụp dọc, cần xoay 90°
    Image.new("RGB", (400, 300), "white").save(p_rot, exif=ex)
    used, info = prepare_image(p_rot, tmp)
    assert info["rotated"] and Image.open(used).size == (300, 400) and used.endswith(".png")
    p_big = os.path.join(tmp, "big.jpg")
    Image.new("RGB", (4032, 3024), "white").save(p_big)
    used, info = prepare_image(p_big, tmp)
    assert info["resized"] and max(Image.open(used).size) == MAX_SIDE
    p_bad = os.path.join(tmp, "bad.jpg")
    open(p_bad, "wb").write(b"khong phai anh")
    try:
        prepare_image(p_bad, tmp)
        raise AssertionError("file hỏng phải báo lỗi")
    except ValueError as e:
        assert "JPG hoặc PNG" in str(e)
    # (3) bằng chứng OCR
    L = [{"text": "ĐẠI LÝ XỔ SỐ MIỀN BẮC", "box": [0, 0, 1, 1]}, {"text": "Collee Juice", "box": [0, 0, 1, 1]},
         {"text": "PHỞ", "box": [0, 0, 1, 1]}]
    assert evidence_lines("đại lý xổ số miền bắc", L) == [0]
    assert evidence_lines("phở", L) == [2]                  # khớp nguyên dòng
    assert evidence_lines("quán phở bò", L) == []           # dòng ngắn (< 4 ký tự) không tính khớp ngược
    assert evidence_lines("collee juice và phở", L) == [1]
    assert evidence_lines("màu đỏ", L) == [] and evidence_lines("", L) == []
    # (4) hiển thị: KHÔNG có con số độ tin cậy, đúng mức, cảnh báo đúng
    base = {"answer": "màu đỏ", "confidence": 0.9731, "prompt": "Câu hỏi: x", "ocr_lines": [], "ocr_texts": [],
            "sec_ocr": 0.4, "sec": 0.5, "sec_total": 0.9}
    info0 = {"orig_size": [640, 480], "used_size": [640, 480], "rotated": False, "resized": False, "reencoded": False}
    for mode, conf in (("open", 0.9731), ("yesno", 0.9994)):
        r = {**base, "mode": mode, "confidence": conf, "level": "Cao"}
        out = render_result(r, info0, st, ocr_cfg)
        visible = re.sub(r"<[^>]+>", " ", out["answer_html"] + out["warnings_md"])
        assert "%" not in visible and "97" not in visible and "0.97" not in visible, visible
        assert "Độ tin cậy: Cao" in out["answer_html"]
        assert "Không đọc được chữ" in out["warnings_md"]
    out = render_result({**base, "mode": "yesno", "answer": "không", "level": "Thấp"}, info0, st, ocr_cfg)
    assert "Độ tin cậy thấp" in out["warnings_md"] and "có/không" in out["warnings_md"]
    out = render_result({**base, "mode": "open", "answer": "<b>x</b>", "level": "Trung bình"}, info0, st, ocr_cfg)
    assert "&lt;b&gt;x&lt;/b&gt;" in out["answer_html"]      # chữ của model được escape
    many = [{"text": f"dòng chữ số {i} " + "x" * 40, "conf": 0.9, "box": [0, 0, 5, 5]} for i in range(20)]
    from src.data.ocr import lines_to_texts
    texts, cut = lines_to_texts({"lines": many}, 600, 0.0)
    out = render_result({**base, "mode": "open", "level": "Cao", "ocr_lines": many, "ocr_texts": texts},
                        info0, st, ocr_cfg)
    assert cut and out["truncated"] and f"chỉ {len(texts)}/20 dòng" in out["warnings_md"]
    shown = render_result({**base, "mode": "open", "level": "Cao"}, info0,
                          {**st, "show_number_open": True}, ocr_cfg)["answer_html"]
    assert "97%" in shown                                  # nếu sau này ECE ≤ 0,05 thì mới hiện số
    # (5) chạy trọn một lượt với backend giả (prompt dựng bằng builder thật)
    log = os.path.join(tmp, "q.jsonl")
    htmp = tempfile.mkdtemp(prefix="vvapp_h_")
    h = Handler(MockBackend(st, ocr_cfg), st, ocr_cfg, log_jsonl=log, tmp_dir=htmp)
    a, w, im, o, d = h(p_ok, "tên cửa hàng là gì")
    assert "cửa hàng mẫu" in a and "khớp với chữ" in w and im is not None and "CỬA HÀNG MẪU" in o
    assert "Văn bản đọc được trong ảnh: CỬA HÀNG MẪU | ĐIỆN THOẠI · PHỤ KIỆN | 0123 456 789" in d
    a, w, im, o, d = h(p_ok, "Trong ảnh có con mèo không?")
    assert 'Trả lời "có" hoặc "không".' in d and "có/không" in a
    a, w, im, o, d = h(p_rot, "ảnh không chữ này màu gì")
    assert "Không đọc được chữ" in w and "xoay theo EXIF" in w
    assert not os.listdir(htmp), "ảnh tạm phải bị xoá sau mỗi lượt"
    a, *_ = h(None, "câu hỏi")
    assert "Chưa có ảnh" in a
    a, *_ = h(p_ok, "   ")
    assert "Chưa có câu hỏi" in a
    a, *_ = h(p_bad, "màu gì")
    assert "Không xử lý được" in a
    recs = [json.loads(x) for x in open(log, encoding="utf-8")]
    assert len(recs) == 4 and [r["ok"] for r in recs] == [True, True, True, False]
    flat = json.dumps(recs, ensure_ascii=False)
    assert "con mèo" not in flat and "cửa hàng" not in flat, "log không được chứa câu hỏi / câu trả lời"
    # (6) dựng được giao diện Gradio (không mở cổng)
    try:
        import gradio  # noqa: F401
        has_gr = True
    except ImportError:
        has_gr = False
        print("⚠ chưa cài gradio -> bỏ qua kiểm dựng giao diện (notebook 24 cài gradio và kiểm lại)")
    if has_gr:
        demo = build_demo(h, examples=[[p_ok, "tên cửa hàng là gì"]])
        assert demo is not None
    print("✅ selftest app OK (ngưỡng từ summary, ảnh nguyên file / xoay / thu nhỏ, bằng chứng OCR, "
          "không hiện số độ tin cậy, cảnh báo, backend giả, log không chứa nội dung"
          + (", dựng Gradio)" if has_gr else ")"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mock", action="store_true", help="backend giả (CPU) để duyệt giao diện")
    ap.add_argument("--adapter", help="thư mục adapter SH (…/W11_SH_…/epoch_1)")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "qwen_lora.yaml"))
    ap.add_argument("--expect_adapter_sha", help="sha256 trọng số adapter phải trùng (lấy từ meta run W11)")
    ap.add_argument("--summary", default=None, help=f"mặc định {SUMMARY_REL}")
    ap.add_argument("--ocr_device", default="gpu")
    ap.add_argument("--ocr_python", default=None, help="python cho tiến trình OCR (mặc định: python hiện tại)")
    ap.add_argument("--ocr_log", default=None)
    ap.add_argument("--allow_ocr_mismatch", action="store_true")
    ap.add_argument("--examples_json", default=None)
    ap.add_argument("--log_jsonl", default=None, help="ghi số đo từng lượt (không ghi ảnh / câu hỏi / câu trả lời)")
    ap.add_argument("--share", action="store_true", help="tạo link công khai *.gradio.live (dùng trên Colab)")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    st = load_ui_settings(a.summary)
    ocr_cfg = load_ocr_selection()
    client = None
    if a.mock:
        backend = MockBackend(st, ocr_cfg)
    else:
        if not a.adapter:
            raise SystemExit("Cần --adapter (hoặc --mock để thử giao diện)")
        backend, client, info = make_real_backend(a.adapter, a.config, st, ocr_cfg, a.ocr_device, a.ocr_python,
                                                  a.ocr_log, a.expect_adapter_sha, a.allow_ocr_mismatch)
        print("Khởi động:", json.dumps(info, ensure_ascii=False))
    demo = build_demo(Handler(backend, st, ocr_cfg, a.log_jsonl), load_examples(a.examples_json))
    try:
        demo.queue(default_concurrency_limit=1).launch(share=a.share, server_name="0.0.0.0", server_port=a.port,
                                                       css=CSS, show_error=True)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
