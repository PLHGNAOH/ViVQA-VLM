"""
train_qlora.py — Huấn luyện LoRA/QLoRA cho Qwen2.5-VL trên ViVQA / ViTextVQA.

RÀNG BUỘC CỨNG CỦA ĐỀ TÀI: chỉ PEFT, KHÔNG full fine-tuning.
Trọng số gốc bị đóng băng (và lượng tử hoá 4-bit NF4); chỉ LoRA adapter được học.
Bằng chứng tuân thủ được in ra và ghi vào train_log.json: số tham số trainable / tổng.

Script này KHÔNG tự nạp model — nó dùng lại `src.models.vlm_loader.load_vlm_for_training`
để mọi lựa chọn (lượng tử hoá, dtype, target_modules) chỉ nằm ở MỘT chỗ: configs/*.yaml.

Ba việc script này làm mà chưa có ở đâu trong repo:
  1. Dựng mẫu huấn luyện: (ảnh + câu hỏi) -> prompt Y HỆT lúc eval, cộng câu trả lời đúng.
     Prompt lúc train phải trùng prompt lúc eval, nếu không là lệch phân phối.
  2. CHE NHÃN (label masking): chỉ tính loss trên phần ĐÁP ÁN. Nếu không che, model
     học cách sinh lại câu hỏi và ảnh -> loss giảm đẹp nhưng chất lượng trả lời không lên.
  3. Đo chi phí thật: giây/step và VRAM đỉnh -> dữ kiện để chốt `max_pixels` (W7).

Cách chạy (Colab, GPU):
  # smoke run 30 step ở độ phân giải 512 token
  !python -u -m src.adaptation.train_qlora --dataset vivqa \
      --run_name W06_smoke_qlora_qwen25vl_vivqa_mp512_seed42 \
      --limit 200 --max_steps 30 --max_pixels 401408

  # cùng vậy nhưng 1280 token -> so sánh chi phí
  !python -u -m src.adaptation.train_qlora --dataset vivqa \
      --run_name W06_smoke_qlora_qwen25vl_vivqa_mp1280_seed42 \
      --limit 200 --max_steps 30 --max_pixels 1003520

  # W08: QLoRA + OCR-enhanced prompting (B3) trên tập con cố định; B2-sub = cùng lệnh, bỏ --mode/--ocr_*
  !python -u -m src.adaptation.train_qlora --dataset vitextvqa_official \
      --data_file <drive>/data/vitextvqa_official/train_subset8000_byimage_seed42.json \
      --run_name W08_B3_qlora_ocr_... --epochs 1 --mode ocr --ocr_backend paddleocr

  # W09: lặp lại B2-sub / B3 với seed huấn luyện khác (tập con train GIỮ NGUYÊN seed 42)
  !python -u -m src.adaptation.train_qlora ... --seed 43

  # W10: ablation cấu hình LoRA trên công thức B3 (H3: vị trí gắn, H4: rank)
  !python -u -m src.adaptation.train_qlora ... --lora_targets all_linear_llm --lora_r 4   # A2, cùng ngân sách
  !python -u -m src.adaptation.train_qlora ... --lora_r 64                                # A3, alpha = 128

  # W11: adapter DÙNG CHUNG — train trộn nhiều dataset (thay --dataset/--data_file bằng --mix, lặp lại được)
  !python -u -m src.adaptation.train_qlora --run_name W11_SH_qlora_... --epochs 1 \
      --mix vivqa=data/vivqa/train.json \
      --mix vitextvqa_official=data/vitextvqa_official/train_subset8000_byimage_seed42.json \
      --mode ocr --ocr_backend paddledet_vietocr
  (mỗi dataset tự dùng OCR cache mặc định của nó; thiết lập OCR phải trùng nhau, nếu không -> dừng)

Chạy tiếp sau khi Colab ngắt: chạy lại ĐÚNG lệnh cũ (cùng --run_name) -> tự resume.

Kiểm thử offline (không GPU, không mạng, không cần torch):
  python -m src.adaptation.train_qlora --selftest
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/ViVQA-VLM"

# File train của từng dataset (đường dẫn tương đối so với drive_root).
DATASETS = {
    "vivqa": {"train_rel": "data/vivqa/train.json"},
    "vitextvqa_official": {"train_rel": "data/vitextvqa_official/train.json"},
}
OCR_MODES = {"ocr", "ocr_rag"}   # khớp src.prompting.builder.OCR_MODES (giữ module này import được không cần gì)

IGNORE_INDEX = -100          # quy ước của PyTorch: vị trí có nhãn -100 bị BỎ QUA khi tính loss
QWEN_TURN_END = "<|im_end|>"  # token kết thúc lượt nói của Qwen2.5; phải có để model biết dừng


# --------------------------------------------------------------------------- #
# Tiện ích chung (giữ đồng bộ với src/eval/run_baseline.py)
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: str, obj: Any, indent: Optional[int] = 2) -> None:
    """Ghi .tmp rồi đổi tên -> không bao giờ để lại file dở dang."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# 1) CHE NHÃN — hàm thuần Python, tách riêng để kiểm thử được KHÔNG cần GPU
# --------------------------------------------------------------------------- #
def build_labels(input_id_rows: List[List[int]], prompt_lens: List[int],
                 attention_rows: List[List[int]]) -> List[List[int]]:
    """
    Dựng nhãn cho loss: chỉ giữ phần ĐÁP ÁN, che phần prompt và phần padding.

    Vì sao cần: model sinh văn bản học bằng cách đoán token kế tiếp trên TOÀN chuỗi.
    Nếu không che, nó dành phần lớn sức học để sinh lại câu hỏi và các token ảnh —
    loss trông đẹp nhưng không giúp trả lời đúng hơn.

    Giả định: padding bên PHẢI (padding_side='right'), nên phần prompt luôn nằm ở
    đầu chuỗi: vị trí [0, prompt_len) là prompt, sau đó là đáp án, cuối là padding.

    Args:
        input_id_rows:  input_ids của từng mẫu (đã pad) dưới dạng list số nguyên.
        prompt_lens:    số token của riêng phần prompt (đã tính cả token ảnh) mỗi mẫu.
        attention_rows: attention_mask tương ứng (1 = token thật, 0 = padding).

    Returns:
        List nhãn cùng kích thước, vị trí bị bỏ qua mang giá trị IGNORE_INDEX.
    """
    if not (len(input_id_rows) == len(prompt_lens) == len(attention_rows)):
        raise ValueError("input_ids, prompt_lens, attention_mask phải cùng số mẫu")

    labels: List[List[int]] = []
    for ids, p_len, attn in zip(input_id_rows, prompt_lens, attention_rows):
        if len(ids) != len(attn):
            raise ValueError("input_ids và attention_mask phải cùng độ dài")
        if p_len > len(ids):
            raise ValueError(f"prompt_len ({p_len}) dài hơn cả chuỗi ({len(ids)})")
        row = [
            IGNORE_INDEX if (i < p_len or attn[i] == 0) else ids[i]
            for i in range(len(ids))
        ]
        if all(v == IGNORE_INDEX for v in row):
            raise ValueError("Một mẫu bị che toàn bộ -> không còn gì để học (đáp án rỗng?)")
        labels.append(row)
    return labels


# --------------------------------------------------------------------------- #
# 2) Dựng văn bản huấn luyện — cũng thuần Python, kiểm thử được offline
# --------------------------------------------------------------------------- #
def build_target_text(answer: str, turn_end: str = QWEN_TURN_END) -> str:
    """
    Phần model phải sinh ra: đáp án + token kết thúc lượt.

    Không có token kết thúc thì model không học được khi nào NÊN DỪNG — đúng lỗi
    'trả lời lan man' đã thấy ở baseline W5.
    """
    answer = " ".join(str(answer).split())          # gọn khoảng trắng, giữ nguyên dấu tiếng Việt
    if not answer:
        raise ValueError("Đáp án rỗng — mẫu này phải bị loại trước khi vào batch")
    return answer + turn_end


def plan_schedule(n_samples: int, batch_size: int, grad_accum: int, epochs: float,
                  warmup_ratio: float, max_steps: Optional[int] = None) -> Dict[str, int]:
    """
    Quy đổi lịch huấn luyện sang SỐ STEP.

    Vì sao cần: `TrainingArguments` của transformers 5.x chỉ nhận `warmup_steps`,
    không còn `warmup_ratio`. Nhưng config của dự án khai báo theo TỈ LỆ (hợp lý hơn
    vì không phụ thuộc kích thước dataset), nên phải quy đổi ở đây — một chỗ duy nhất.

    Trả về: {"steps_per_epoch", "total_steps", "warmup_steps"}.
    """
    import math
    eff_batch = max(int(batch_size) * int(grad_accum), 1)
    steps_per_epoch = max(1, math.ceil(max(int(n_samples), 1) / eff_batch))
    total_steps = int(max_steps) if max_steps else max(1, int(round(steps_per_epoch * float(epochs))))
    ratio = float(warmup_ratio or 0.0)
    warmup_steps = max(1, int(round(ratio * total_steps))) if ratio > 0 else 0
    return {"steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "warmup_steps": warmup_steps}


def pick_answer(sample: Dict[str, Any]) -> str:
    """
    ViVQA mỗi câu 1 đáp án; ViTextVQA có thể nhiều. Quy ước: lấy đáp án ĐẦU TIÊN.
    Ghi vào log để tái lập; đổi quy ước phải đổi ở đây, một chỗ duy nhất.
    """
    answers = sample.get("answers") or []
    if not answers:
        raise ValueError(f"Mẫu {sample.get('question_id')} không có đáp án")
    return str(answers[0])


def filter_trainable(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Loại mẫu không train được (đáp án rỗng, thiếu ảnh). Trả (mẫu_giữ, id_bị_loại)."""
    keep, dropped = [], []
    for s in samples:
        answers = [a for a in (s.get("answers") or []) if str(a).strip()]
        if not answers:
            dropped.append(str(s.get("question_id")))
            continue
        if not os.path.exists(s["image_path"]):
            dropped.append(str(s.get("question_id")))
            continue
        keep.append(s)
    return keep, dropped


MAX_DROP_FRAC = 0.01   # loại > 1% mẫu = gần như chắc chắn thiếu ảnh (chép dở), không phải dữ liệu xấu


def check_drop_ratio(n_keep: int, n_drop: int, image_dir: str = "", max_frac: float = MAX_DROP_FRAC) -> None:
    """Dừng hẳn nếu bị loại quá nhiều mẫu: train âm thầm trên dữ liệu thiếu sẽ làm sai lệch kết quả."""
    total = n_keep + n_drop
    if total and n_drop / total > max_frac:
        raise ValueError(f"Bị loại {n_drop}/{total} mẫu (> {max_frac:.0%}) — gần như chắc chắn thiếu ảnh ở "
                         f"{image_dir or 'thư mục ảnh'}. Chạy lại (ảnh sẽ được chép bù) hoặc kiểm tra dữ liệu.")


# --------------------------------------------------------------------------- #
# 3) Config
# --------------------------------------------------------------------------- #
def load_config(path: str, max_pixels: Optional[int] = None, dtype: Optional[str] = None,
                mode: Optional[str] = None, batch_size: Optional[int] = None,
                grad_accum: Optional[int] = None,
                grad_checkpointing: Optional[bool] = None,
                epochs: Optional[float] = None,
                seed: Optional[int] = None,
                lora_r: Optional[int] = None,
                lora_alpha: Optional[int] = None,
                lora_targets: Optional[str] = None) -> Dict[str, Any]:
    """Đọc config + cho phép ghi đè vài tham số qua CLI (phục vụ ablation, không sửa file gốc)."""
    import yaml
    with open(path, encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    if max_pixels is not None:
        cfg["model"]["max_pixels"] = int(max_pixels)
    if dtype is not None:
        cfg["model"]["torch_dtype"] = dtype
        cfg.setdefault("quantization", {})["bnb_4bit_compute_dtype"] = dtype
    if mode is not None:
        cfg["prompting"]["mode"] = mode
    # Ba núm vặn TỐC ĐỘ: quan trọng khi VRAM còn dư mà GPU chưa được dùng hết công suất.
    if batch_size is not None:
        cfg["training"]["per_device_batch_size"] = int(batch_size)
    if grad_accum is not None:
        cfg["training"]["gradient_accumulation_steps"] = int(grad_accum)
    if grad_checkpointing is not None:
        cfg["training"]["gradient_checkpointing"] = bool(grad_checkpointing)
    # Số epoch quyết định CẢ lịch learning rate (warmup + decay trải trên toàn bộ step),
    # nên 1 epoch của lịch 3 epoch KHÁC một run 1 epoch -> ghi đè ở đây, có trong chữ ký.
    if epochs is not None:
        cfg["training"]["epochs"] = float(epochs)
    # W09: seed huấn luyện (khởi tạo LoRA, thứ tự batch, dropout) để đo độ ổn định theo seed.
    # cfg["run"]["seed"] nằm trong chữ ký run -> seed khác = run khác (không resume nhầm).
    # None -> giữ seed của file config (42): chữ ký các run cũ KHÔNG đổi.
    if seed is not None:
        cfg["run"]["seed"] = int(seed)
    # W10: ablation cấu hình LoRA (H3: vị trí gắn, H4: rank). Ba khoá đều nằm trong cfg["lora"]
    # -> có trong chữ ký run. None -> giữ file config: chữ ký các run W07–W09 KHÔNG đổi.
    # Đổi r mà không truyền alpha -> alpha = 2·r (giữ hệ số scale alpha/r = 2 như cấu hình gốc r16/α32),
    # để so sánh rank không bị lẫn với thay đổi độ lớn bước cập nhật.
    if lora_targets is not None:
        from src.adaptation.lora_targets import preset_target_modules
        cfg["lora"]["target_modules"] = preset_target_modules(lora_targets)
    if lora_r is not None:
        cfg["lora"]["r"] = int(lora_r)
        if lora_alpha is None:
            cfg["lora"]["lora_alpha"] = 2 * int(lora_r)
    if lora_alpha is not None:
        cfg["lora"]["lora_alpha"] = int(lora_alpha)
    return cfg


# --------------------------------------------------------------------------- #
# 3b) Chạy tiếp (resume) an toàn — thuần Python, kiểm thử được không cần GPU
# --------------------------------------------------------------------------- #
CKPT_PREFIX = "checkpoint-"
SIGNATURE_FILE = "run_signature.json"
# Khoá chỉ ảnh hưởng CÁCH LƯU, không ảnh hưởng phép tính -> không đưa vào chữ ký
_SIG_EXCLUDE_TRAINING = ("save_strategy", "save_steps", "save_total_limit", "logging_steps")


def find_resume_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Checkpoint HOÀN CHỈNH mới nhất (checkpoint-N có trainer_state.json), hoặc None."""
    if not os.path.isdir(ckpt_dir):
        return None
    best, best_step = None, -1
    for name in os.listdir(ckpt_dir):
        if not name.startswith(CKPT_PREFIX):
            continue
        try:
            step = int(name[len(CKPT_PREFIX):])
        except ValueError:
            continue
        path = os.path.join(ckpt_dir, name)
        # Checkpoint đang ghi dở khi Colab đứt sẽ thiếu trainer_state.json -> bỏ qua
        if os.path.isfile(os.path.join(path, "trainer_state.json")) and step > best_step:
            best, best_step = path, step
    return best


def session_losses(hf_train_loss: float, global_step: int, session_steps: int,
                   log_history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Loss báo cáo đúng cả khi chạy tiếp từ checkpoint.

    transformers 5.x tính train_loss = (tổng loss các step CỦA PHIÊN NÀY) / global_step,
    mà global_step gồm cả các step đã chạy trước khi resume -> loss bị kéo thấp
    (vd. 7 step/12 -> thấp ~42%). Hiệu chỉnh: nhân lại global_step / session_steps.
    Không resume thì session_steps == global_step -> giữ nguyên giá trị cũ.
    """
    nan = float("nan")
    mean = (float(hf_train_loss) * global_step / session_steps) if session_steps else nan
    logged = [float(h["loss"]) for h in (log_history or []) if "loss" in h]
    return {"final_loss": round(mean, 4),
            "last_logged_loss": round(logged[-1], 4) if logged else nan,
            "hf_train_loss_raw": round(float(hf_train_loss), 4)}


def run_signature(cfg: Dict[str, Any], data_sha256: str, n_samples: int,
                  limit: Optional[int], max_steps: Optional[int]) -> Dict[str, Any]:
    """Dấu vân tay của MỘT run: mọi thứ ảnh hưởng phép tính. Khác chữ ký = run khác."""
    c = copy.deepcopy(cfg)
    c.get("run", {}).pop("name", None)
    for k in _SIG_EXCLUDE_TRAINING:
        c.get("training", {}).pop(k, None)
    payload = {"cfg": c, "data_sha256": data_sha256, "n_samples": n_samples,
               "limit": limit, "max_steps": max_steps}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"sha256": digest, **payload}


def check_or_write_signature(out_dir: str, sig: Dict[str, Any]) -> bool:
    """
    Lần đầu: ghi chữ ký. Lần sau (chạy tiếp): chữ ký PHẢI trùng, nếu không báo lỗi —
    tránh nối checkpoint của cấu hình/dữ liệu cũ vào một run mới (trộn kết quả).
    Trả True nếu chữ ký đã tồn tại và khớp.
    """
    path = os.path.join(out_dir, SIGNATURE_FILE)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fp:
            old = json.load(fp)
        if old.get("sha256") != sig["sha256"]:
            raise ValueError(f"{path} khác cấu hình/dữ liệu hiện tại -> dùng run_name mới "
                             f"(hoặc xoá thư mục run cũ nếu thật sự muốn train lại từ đầu)")
        return True
    _write_json(path, sig)
    return False


def attach_ocr_for_training(cfg: Dict[str, Any], samples: List[Dict[str, Any]], drive_root: str, dataset: str,
                            ocr_backend: Optional[str], ocr_cache: Optional[str], ocr_template: Optional[str],
                            ocr_max_chars: Optional[int], ocr_min_conf: Optional[float]) -> Optional[Dict[str, Any]]:
    """
    W08: mode OCR -> gắn OCR vào mẫu + ghi thiết lập vào cfg["prompting"]["ocr"] (nằm trong chữ ký run).
    Phải gọi TRƯỚC run_signature. Không phải mode OCR -> cfg giữ nguyên (chữ ký run cũ không đổi).
    """
    from src.data.ocr import OCR_DEFAULTS, default_cache_path, setup_ocr_prompting
    use_ocr = cfg["prompting"]["mode"] in OCR_MODES
    extra = [ocr_template, ocr_max_chars, ocr_min_conf]
    if (ocr_backend or ocr_cache or any(v is not None for v in extra)) and not use_ocr:
        raise ValueError("--ocr_* chỉ dùng với --mode ocr")
    if not use_ocr:
        return None
    if not (ocr_backend or ocr_cache):
        raise ValueError("--mode ocr cần --ocr_backend (vd paddleocr) hoặc --ocr_cache")
    path = ocr_cache or default_cache_path(drive_root, dataset, ocr_backend)
    return setup_ocr_prompting(
        cfg, samples, path,
        template=ocr_template if ocr_template is not None else OCR_DEFAULTS["template"],
        max_chars=ocr_max_chars if ocr_max_chars is not None else OCR_DEFAULTS["max_chars"],
        min_conf=ocr_min_conf if ocr_min_conf is not None else OCR_DEFAULTS["min_conf"])


# --------------------------------------------------------------------------- #
# 3c) W11 — train TRỘN nhiều dataset (adapter dùng chung). Thuần Python, kiểm thử offline được.
# --------------------------------------------------------------------------- #
# Thiết lập OCR phải TRÙNG giữa các dataset (cùng engine, cùng template/ngưỡng) — chỉ phần ảnh (subset) khác nhau.
MIX_COMMON_OCR_KEYS = ("backend", "cache_format", "engine", "cache_file", "template", "max_chars", "min_conf")
MIX_ORDER_NOTE = "ghép các phần theo thứ tự --mix, xáo bằng random.Random(seed), rồi RandomSampler(seed) của Trainer"


def parse_mix(items: Optional[List[str]]) -> List[Tuple[str, str]]:
    """['vivqa=data/vivqa/train.json', ...] -> [('vivqa', 'data/vivqa/train.json'), ...]. Cần >= 2 dataset khác nhau."""
    parts: List[Tuple[str, str]] = []
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--mix '{it}' sai dạng, cần DATASET=FILE (vd vivqa=data/vivqa/train.json)")
        ds, path = (x.strip() for x in it.split("=", 1))
        if ds not in DATASETS:
            raise ValueError(f"--mix: dataset '{ds}' không hợp lệ (có: {list(DATASETS)})")
        if not path:
            raise ValueError(f"--mix: thiếu file cho {ds}")
        parts.append((ds, path))
    if len(parts) < 2:
        raise ValueError("--mix cần ít nhất 2 phần (2 dataset)")
    if len({d for d, _ in parts}) != len(parts):
        raise ValueError("--mix: mỗi dataset chỉ được xuất hiện một lần")
    return parts


def merge_ocr_settings(by_ds: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Gộp cfg['prompting']['ocr'] của từng phần thành MỘT mục (vào chữ ký run). Khác thiết lập -> lỗi."""
    first_ds, first = next(iter(by_ds.items()))
    for ds, o in by_ds.items():
        diff = {k: (first.get(k), o.get(k)) for k in MIX_COMMON_OCR_KEYS if first.get(k) != o.get(k)}
        if diff:
            raise ValueError(f"OCR của '{ds}' khác '{first_ds}' ở {diff} -> các dataset phải dùng CÙNG engine "
                             f"và thiết lập OCR (adapter dùng chung = một luật prompt cho mọi câu)")
    merged = {k: copy.deepcopy(first.get(k)) for k in MIX_COMMON_OCR_KEYS}
    merged["subset_sha256_by_dataset"] = {ds: o["subset_sha256"] for ds, o in by_ds.items()}
    return merged


def mixed_data_sha(parts: List[Tuple[str, str]]) -> str:
    """Dấu vân tay dữ liệu của run trộn: sha256 của [(dataset, sha256 file), ...] theo thứ tự --mix."""
    return hashlib.sha256(json.dumps([list(p) for p in parts], ensure_ascii=False).encode("utf-8")).hexdigest()


def shuffle_mixed(samples: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """Xáo đều các dataset với nhau (tái lập được). Trainer cũng xáo lại -> đây là lớp bảo đảm thứ hai."""
    import random
    out = list(samples)
    random.Random(seed).shuffle(out)
    return out


def prepare_mixed(cfg: Dict[str, Any], mix: List[Tuple[str, str]], drive_root: str, limit: Optional[int],
                  ocr_backend: Optional[str], ocr_template: Optional[str], ocr_max_chars: Optional[int],
                  ocr_min_conf: Optional[float], image_dirs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Nạp + lọc + gắn OCR cho TỪNG phần như một run đơn (cùng hàm, cùng ngưỡng loại mẫu), rồi ghép + xáo.
    Ghi cfg['prompting']['ocr'] (đã gộp) TRƯỚC khi tính chữ ký run. image_dirs: chỉ dùng khi kiểm thử offline.
    """
    from src.data.vivqa_dataset import load_vivqa
    samples: List[Dict[str, Any]] = []
    dropped: List[str] = []
    parts_meta: List[Dict[str, Any]] = []
    ocr_by_ds: Dict[str, Dict[str, Any]] = {}
    stats_by_ds: Dict[str, Dict[str, Any]] = {}
    for ds, rel in mix:
        f = rel if os.path.isabs(rel) else os.path.join(drive_root, rel)
        if image_dirs is not None:
            idir = image_dirs[ds]
        else:
            from src.eval.run_baseline import prepare_images
            idir = prepare_images(ds, drive_root)
        part = load_vivqa(f, image_dir=idir, max_samples=limit)
        part, drop = filter_trainable(part)
        if not part:
            raise ValueError(f"{ds}: không còn mẫu nào train được sau khi lọc")
        check_drop_ratio(len(part), len(drop), idir)
        for s in part:
            s["source_dataset"] = ds
        st = attach_ocr_for_training(cfg, part, drive_root, ds, ocr_backend, None,
                                     ocr_template, ocr_max_chars, ocr_min_conf)
        if st:
            ocr_by_ds[ds] = copy.deepcopy(cfg["prompting"]["ocr"])
            stats_by_ds[ds] = st
        parts_meta.append({"dataset": ds,
                           "data_file": os.path.relpath(f, drive_root) if f.startswith(drive_root) else f,
                           "data_file_sha256": _sha256(f), "num_samples": len(part), "num_dropped": len(drop)})
        samples += part
        dropped += drop
    if ocr_by_ds:
        cfg["prompting"]["ocr"] = merge_ocr_settings(ocr_by_ds)
    seed = int(cfg["run"]["seed"])
    return {"samples": shuffle_mixed(samples, seed), "dropped": dropped, "parts": parts_meta,
            "ocr_stats": stats_by_ds or None,
            "data_sha": mixed_data_sha([(p["dataset"], p["data_file_sha256"]) for p in parts_meta])}


# --------------------------------------------------------------------------- #
# 4) Dataset + Collator (cần torch/transformers -> import trễ bên trong)
# --------------------------------------------------------------------------- #
def make_collate_fn(processor, cfg: Dict[str, Any]):
    """
    Trả về hàm gom nhiều mẫu thành 1 batch đúng định dạng Qwen2.5-VL.

    Với VLM đây là chỗ hay sai nhất, vì độ dài prompt PHỤ THUỘC ẢNH: mỗi ảnh nở ra
    một số token khác nhau tuỳ độ phân giải. Nên prompt_len phải đo bằng cách cho
    processor xử lý riêng phần prompt + đúng ảnh đó, không đoán bằng độ dài chữ.
    """
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from src.prompting.builder import prompt_for_sample, to_chat_messages

    prompting = cfg["prompting"]
    # Padding bên PHẢI: build_labels giả định prompt nằm ở đầu chuỗi.
    processor.tokenizer.padding_side = "right"

    def collate(batch: List[Dict[str, Any]]):
        texts, prompt_texts, images_per_sample = [], [], []
        for s in batch:
            image = Image.open(s["image_path"]).convert("RGB")
            # Cùng hàm dựng prompt với lúc eval (run_eval) -> train/eval không lệch prompt (kể cả dòng OCR)
            messages = to_chat_messages(image, prompt_for_sample(s, prompting))
            prompt_only = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompt_texts.append(prompt_only)
            texts.append(prompt_only + build_target_text(pick_answer(s)))
            img_inputs, _ = process_vision_info(messages)
            images_per_sample.append(img_inputs)

        flat_images = [img for imgs in images_per_sample for img in imgs]
        inputs = processor(text=texts, images=flat_images, padding=True, return_tensors="pt")

        # Đo độ dài prompt của TỪNG mẫu (riêng lẻ, không pad) -> mốc để che nhãn.
        prompt_lens = []
        for p_text, imgs in zip(prompt_texts, images_per_sample):
            enc = processor(text=[p_text], images=imgs, padding=False, return_tensors="pt")
            prompt_lens.append(int(enc["input_ids"].shape[1]))

        labels = build_labels(inputs["input_ids"].tolist(), prompt_lens,
                              inputs["attention_mask"].tolist())
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs

    return collate


# --------------------------------------------------------------------------- #
# 5) Quy trình huấn luyện
# --------------------------------------------------------------------------- #
def train(dataset: str, run_name: str, drive_root: str = DEFAULT_DRIVE_ROOT,
          data_file: Optional[str] = None, out_root: Optional[str] = None,
          adapter_root: Optional[str] = None,
          config: str = os.path.join(REPO_ROOT, "configs/qwen_lora.yaml"),
          limit: Optional[int] = None, max_steps: Optional[int] = None,
          max_pixels: Optional[int] = None, dtype: Optional[str] = None,
          mode: Optional[str] = None, image_dir: Optional[str] = None,
          batch_size: Optional[int] = None, grad_accum: Optional[int] = None,
          grad_checkpointing: Optional[bool] = None, save_steps: Optional[int] = None,
          debug_crash_at_step: Optional[int] = None,
          epochs: Optional[float] = None, seed: Optional[int] = None,
          ocr_backend: Optional[str] = None, ocr_cache: Optional[str] = None,
          ocr_template: Optional[str] = None, ocr_max_chars: Optional[int] = None,
          ocr_min_conf: Optional[float] = None,
          lora_r: Optional[int] = None, lora_alpha: Optional[int] = None,
          lora_targets: Optional[str] = None,
          mix: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
    """
    Nạp data -> nạp model + LoRA -> train -> lưu adapter + train_log.json.

    Chịu được việc Colab ngắt giữa chừng: checkpoint lưu mỗi `save_steps` step vào
    <out_dir>/checkpoints/; chạy lại ĐÚNG lệnh cũ -> tự chạy tiếp từ checkpoint mới nhất
    (sau khi kiểm tra chữ ký cấu hình + dữ liệu khớp). Adapter cuối mỗi epoch lưu riêng
    ở <adapter_dir>/epoch_<k>/ để chọn epoch tốt nhất trên tập dev.
    """
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from src.data.vivqa_dataset import load_vivqa
    from src.models.vlm_loader import load_vlm_for_training

    if mix:
        # W11: run trộn — mỗi phần tự khai file + dùng OCR cache mặc định của dataset đó
        if dataset or data_file or image_dir or ocr_cache:
            raise ValueError("--mix không dùng cùng --dataset / --data_file / image_dir / --ocr_cache")
        dataset_label = "mixed:" + "+".join(d for d, _ in mix)
    elif dataset not in DATASETS:
        raise ValueError(f"--dataset phải là một trong {list(DATASETS)}")
    else:
        dataset_label = dataset

    cfg = copy.deepcopy(load_config(config, max_pixels=max_pixels, dtype=dtype, mode=mode,
                                    batch_size=batch_size, grad_accum=grad_accum,
                                    grad_checkpointing=grad_checkpointing, epochs=epochs, seed=seed,
                                    lora_r=lora_r, lora_alpha=lora_alpha, lora_targets=lora_targets))
    cfg["run"]["name"] = run_name
    seed = cfg["run"]["seed"]
    set_seed(seed)
    _log(f"   LoRA: r={cfg['lora']['r']} alpha={cfg['lora']['lora_alpha']} "
         f"target_modules={cfg['lora']['target_modules']}")

    if not mix:
        data_file = data_file or os.path.join(drive_root, DATASETS[dataset]["train_rel"])
    out_dir = os.path.join(out_root or os.path.join(drive_root, "experiments"), run_name)
    adapter_dir = os.path.join(adapter_root or os.path.join(drive_root, "adapters"), run_name)

    _log(f"== {run_name} | dataset={dataset_label} | mode={cfg['prompting']['mode']} | seed={seed}")
    for ds_, f_ in (mix or [(dataset, data_file)]):
        _log(f"   data : {ds_}: {f_}")
    _log(f"   out  : {out_dir}")
    _log(f"   adapt: {adapter_dir}")

    # --- Dữ liệu ---
    parts_meta = None
    if mix:
        mixed = prepare_mixed(cfg, mix, drive_root, limit, ocr_backend, ocr_template, ocr_max_chars, ocr_min_conf)
        samples, dropped, parts_meta = mixed["samples"], mixed["dropped"], mixed["parts"]
        ocr_stats, data_sha = mixed["ocr_stats"], mixed["data_sha"]
        for p in parts_meta:
            _log(f"   {p['dataset']}: {p['num_samples']} mẫu (loại {p['num_dropped']}) | sha256 {p['data_file_sha256'][:12]}")
        _log(f"   TỔNG {len(samples)} mẫu train, đã xáo lẫn ({MIX_ORDER_NOTE})")
        if ocr_stats:
            o = cfg["prompting"]["ocr"]
            _log(f"   OCR: {o['cache_file']} ({o['backend']}) | template {o['template']} | max_chars {o['max_chars']} | "
                 f"min_conf {o['min_conf']} | % mẫu có dòng OCR: "
                 + ", ".join(f"{d} {s['pct_samples_with_text']}%" for d, s in ocr_stats.items()))
    else:
        if image_dir is None:
            from src.eval.run_baseline import prepare_images
            image_dir = prepare_images(dataset, drive_root)
        samples = load_vivqa(data_file, image_dir=image_dir, max_samples=limit)
        samples, dropped = filter_trainable(samples)
        if not samples:
            raise ValueError("Không còn mẫu nào train được sau khi lọc")
        check_drop_ratio(len(samples), len(dropped), image_dir)
        _log(f"   {len(samples)} mẫu train (loại {len(dropped)} mẫu thiếu ảnh/đáp án)")
        ocr_stats = attach_ocr_for_training(cfg, samples, drive_root, dataset, ocr_backend, ocr_cache,
                                            ocr_template, ocr_max_chars, ocr_min_conf)
        if ocr_stats:
            o = cfg["prompting"]["ocr"]
            _log(f"   OCR: {o['cache_file']} ({o['backend']}) | template {o['template']} | max_chars {o['max_chars']} | "
                 f"min_conf {o['min_conf']} | {ocr_stats['pct_samples_with_text']}% mẫu có chữ")
        data_sha = _sha256(data_file)

    # --- Chạy tiếp hay chạy mới? (kiểm tra TRƯỚC khi nạp model 4 phút) ---
    t = cfg["training"]
    if save_steps is not None:
        t["save_steps"] = int(save_steps)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    sig = run_signature(cfg, data_sha, len(samples), limit, max_steps)
    sig_existed = check_or_write_signature(out_dir, sig)
    resume_from = find_resume_checkpoint(ckpt_dir)
    if resume_from:
        _log(f"   ↻ CHẠY TIẾP từ {os.path.basename(resume_from)} (chữ ký cấu hình khớp)")
    elif sig_existed:
        _log("   chữ ký khớp nhưng chưa có checkpoint hoàn chỉnh -> train lại từ đầu")

    # --- Model + LoRA (PEFT-only; ràng buộc nằm trong vlm_loader) ---
    _log("   nạp model + gắn LoRA...")
    model, processor = load_vlm_for_training(cfg)
    model.config.use_cache = False                  # bắt buộc khi bật gradient checkpointing
    # W10: kiểm LoRA gắn đúng chỗ (không lọt vision tower, đủ lớp mỗi hậu tố, số tham số = tính tay)
    # -> lỗi ở đây chỉ mất ≈ 4 phút nạp model, không phải 3 giờ train sai cấu hình.
    from src.adaptation.lora_targets import audit_lora_modules
    lora_audit = audit_lora_modules(model, cfg)
    _log(f"   LoRA audit: {lora_audit['n_modules']} lớp {lora_audit['by_tower']} | {lora_audit['by_suffix']} | "
         f"tham số LoRA {lora_audit['lora_params_counted']:,} (tính tay {lora_audit['lora_params_expected']})")
    # Đếm tham số: với model 4-bit, p.numel() đếm THIẾU vì 2 giá trị được đóng gói
    # trong 1 byte. PEFT có hàm riêng xử lý đúng -> ưu tiên dùng, chỉ tự đếm khi không có.
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
        count_method = "peft.get_nb_trainable_parameters (4-bit aware)"
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        count_method = "p.numel() (CẢNH BÁO: đếm thiếu nếu model 4-bit)"

    # --- Tham số huấn luyện ---
    compute_dtype = cfg["model"].get("torch_dtype", "float16")
    sched = plan_schedule(len(samples), t["per_device_batch_size"],
                          t["gradient_accumulation_steps"], t["epochs"],
                          t.get("warmup_ratio", 0.0), max_steps)
    _log(f"   lịch: {sched['total_steps']} step tổng "
         f"({sched['steps_per_epoch']} step/epoch), warmup {sched['warmup_steps']} step")

    ta_kwargs = dict(
        output_dir=os.path.join(out_dir, "checkpoints"),
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_steps=sched["warmup_steps"],          # transformers 5.x bỏ warmup_ratio
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        # smoke run (max_steps) mặc định KHÔNG lưu, trừ khi truyền --save_steps (để thử resume)
        save_strategy="no" if (max_steps and save_steps is None) else t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 200),
        save_total_limit=t.get("save_total_limit", 2),
        gradient_checkpointing=t["gradient_checkpointing"],
        num_train_epochs=t["epochs"],
        max_steps=max_steps if max_steps else -1,
        fp16=(compute_dtype == "float16"),
        bf16=(compute_dtype == "bfloat16"),
        seed=seed,
        remove_unused_columns=False,                # mẫu của ta là dict tự định nghĩa
        report_to="none",
        dataloader_num_workers=2,
    )
    # Lớp phòng vệ: API TrainingArguments đổi theo phiên bản. Bỏ tham số không được
    # hỗ trợ và BÁO RÕ, thay vì để script chết giữa chừng sau khi đã nạp model 4 phút.
    import dataclasses
    supported = {f.name for f in dataclasses.fields(TrainingArguments)}
    unsupported = sorted(set(ta_kwargs) - supported)
    if unsupported:
        _log(f"   ⚠️ transformers hiện tại không nhận: {unsupported} -> bỏ qua (ghi vào log)")
    args = TrainingArguments(**{k: v for k, v in ta_kwargs.items() if k in supported})

    class EpochAdapterSaver(TrainerCallback):
        """Lưu adapter (chỉ LoRA, vài chục MB) cuối mỗi epoch -> chọn epoch trên dev."""
        def on_epoch_end(self, _args, state, _control, model=None, **_kw):
            k = int(round(state.epoch or 0))
            if model is not None and k >= 1:
                path = os.path.join(adapter_dir, f"epoch_{k}")
                model.save_pretrained(path)
                _log(f"   💾 adapter epoch {k} -> {path}")

    class DebugCrash(TrainerCallback):
        """CHỈ để kiểm thử resume: giả lập Colab đứt tại 1 step cố định."""
        def on_step_end(self, _args, state, _control, **_kw):
            if debug_crash_at_step and state.global_step >= debug_crash_at_step:
                raise RuntimeError(f"[debug] giả lập ngắt tại step {state.global_step}")

    callbacks = [EpochAdapterSaver()] + ([DebugCrash()] if debug_crash_at_step else [])
    trainer = Trainer(model=model, args=args, train_dataset=samples,
                      data_collator=make_collate_fn(processor, cfg), callbacks=callbacks)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    # --- Lưu adapter (CHỈ LoRA, vài chục MB — không phải toàn bộ model) ---
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    # --- Log tái lập ---
    # Số step chạy TRONG PHIÊN NÀY (khi chạy tiếp, trừ phần đã làm trước checkpoint) -> s/step đúng
    start_step = int(os.path.basename(resume_from)[len(CKPT_PREFIX):]) if resume_from else 0
    steps = max(int(trainer.state.global_step) - start_step, 0) or int(max_steps or 0)
    meta = {
        "run_name": run_name,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_label,
        "data_file": (None if mix else
                      os.path.relpath(data_file, drive_root) if data_file.startswith(drive_root) else data_file),
        "data_file_sha256": data_sha,
        "num_samples": len(samples),
        "num_dropped": len(dropped),
        "answer_policy": "đáp án đầu tiên trong trường answers",
        "prompting_mode": cfg["prompting"]["mode"],
        "ocr": (None if not ocr_stats else
                {**cfg["prompting"]["ocr"], "stats_by_dataset": ocr_stats} if mix else
                {**cfg["prompting"]["ocr"], "stats": ocr_stats}),
        "model_id": cfg["model"]["model_id"],
        "max_pixels": cfg["model"]["max_pixels"],
        "min_pixels": cfg["model"]["min_pixels"],
        "torch_dtype": compute_dtype,
        "quantization": cfg["quantization"],
        "lora": cfg["lora"],
        "lora_audit": lora_audit,
        "training": {**t, "max_steps": max_steps, "limit": limit, **sched,
                     "unsupported_training_args": unsupported},
        "run_signature_sha256": sig["sha256"],
        **({"data_parts": parts_meta, "mix_order": MIX_ORDER_NOTE} if mix else {}),
        "resumed_from": os.path.basename(resume_from) if resume_from else None,
        "cost_note": ("cost chỉ tính PHIÊN CUỐI (đã chạy tiếp từ checkpoint)" if resume_from
                      else "cost tính trọn run"),
        "epoch_adapters": sorted(d for d in (os.listdir(adapter_dir) if os.path.isdir(adapter_dir) else [])
                                 if d.startswith("epoch_")),
        "peft_only": True,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / max(total, 1), 4),
        "param_count_method": count_method,
        "seed": seed,
        "git_commit": git_commit(),
        "config_file": os.path.relpath(config, REPO_ROOT),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            "platform": platform.platform(),
        },
        "cost": {
            "train_runtime_sec": round(result.metrics.get("train_runtime", elapsed), 1),
            "steps": steps,
            "global_step": int(trainer.state.global_step),
            "sec_per_step": round(result.metrics.get("train_runtime", elapsed) / max(steps, 1), 3),
            "peak_vram_gb": round(peak_vram_gb, 2),
            **session_losses(result.metrics.get("train_loss", float("nan")),
                             int(trainer.state.global_step), steps, trainer.state.log_history),
        },
        "adapter_dir": adapter_dir,
    }
    try:
        import transformers, peft
        meta["versions"] = {"transformers": transformers.__version__,
                            "torch": torch.__version__, "peft": peft.__version__}
    except Exception:
        meta["versions"] = "unavailable"

    _write_json(os.path.join(out_dir, "train_log.json"),
                {"meta": meta, "log_history": trainer.state.log_history})

    c = meta["cost"]
    eff_batch = t["per_device_batch_size"] * t["gradient_accumulation_steps"]
    _log(f"== XONG: {c['steps']} step | {c['sec_per_step']}s/step | "
         f"{c['sec_per_step'] / max(eff_batch, 1):.2f}s/mẫu | batch hiệu dụng {eff_batch} | "
         f"grad_ckpt={t['gradient_checkpointing']} | VRAM đỉnh {c['peak_vram_gb']} GB | "
         f"loss {c['final_loss']}")
    _log(f"   trainable {trainable:,} / {total:,} ({meta['trainable_pct']}%) -> PEFT-only OK")
    _log(f"   adapter -> {adapter_dir}")
    _log(f"   log     -> {out_dir}/train_log.json")
    return {"meta": meta}


# --------------------------------------------------------------------------- #
# 6) Kiểm thử offline — KHÔNG cần torch, GPU hay mạng
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # (a) che nhãn: prompt bị che, đáp án giữ lại, padding bị che
    ids = [[10, 11, 12, 20, 21, 0, 0],
           [10, 11, 12, 13, 30, 31, 32]]
    attn = [[1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1]]
    labels = build_labels(ids, prompt_lens=[3, 4], attention_rows=attn)
    assert labels[0] == [-100, -100, -100, 20, 21, -100, -100], labels[0]
    assert labels[1] == [-100, -100, -100, -100, 30, 31, 32], labels[1]

    # (b) prompt dài hơn chuỗi -> phải báo lỗi, không im lặng cắt bớt
    try:
        build_labels([[1, 2]], [5], [[1, 1]])
        raise AssertionError("không chặn prompt_len quá dài")
    except ValueError:
        pass

    # (c) che sạch cả mẫu (đáp án rỗng) -> phải báo lỗi
    try:
        build_labels([[1, 2, 3]], [3], [[1, 1, 1]])
        raise AssertionError("không chặn mẫu bị che toàn bộ")
    except ValueError:
        pass

    # (d) số mẫu lệch nhau -> báo lỗi
    try:
        build_labels([[1, 2]], [1, 1], [[1, 1]])
        raise AssertionError("không chặn số mẫu lệch")
    except ValueError:
        pass

    # (d2) quy đổi lịch huấn luyện: tỉ lệ warmup -> số step
    sc = plan_schedule(n_samples=200, batch_size=2, grad_accum=8, epochs=3,
                       warmup_ratio=0.03)
    assert sc["steps_per_epoch"] == 13, sc          # ceil(200/16)
    assert sc["total_steps"] == 39, sc              # 13 * 3
    assert sc["warmup_steps"] == 1, sc              # round(0.03*39)=1, sàn tối thiểu 1
    sc = plan_schedule(200, 2, 8, 3, 0.03, max_steps=30)
    assert sc["total_steps"] == 30 and sc["warmup_steps"] == 1, sc
    sc = plan_schedule(100000, 2, 8, 3, 0.03)
    # round(0.03*18750) = round(562.5) = 562 — Python làm tròn .5 về số CHẴN
    assert sc["total_steps"] == 18750 and sc["warmup_steps"] == 562, sc
    assert plan_schedule(200, 2, 8, 3, 0.0)["warmup_steps"] == 0   # tắt warmup

    # (e) văn bản đích luôn có token kết thúc lượt
    assert build_target_text("  màu   đỏ ") == "màu đỏ" + QWEN_TURN_END
    try:
        build_target_text("   ")
        raise AssertionError("không chặn đáp án rỗng")
    except ValueError:
        pass

    # (f) chọn đáp án + lọc mẫu không train được
    assert pick_answer({"answers": ["hai", "2"]}) == "hai"
    here = os.path.abspath(__file__)
    keep, dropped = filter_trainable([
        {"question_id": "a", "answers": ["đỏ"], "image_path": here},          # giữ
        {"question_id": "b", "answers": [" "], "image_path": here},           # đáp án rỗng
        {"question_id": "c", "answers": ["xanh"], "image_path": "/khong/co"},  # thiếu ảnh
    ])
    assert [s["question_id"] for s in keep] == ["a"], keep
    assert sorted(dropped) == ["b", "c"], dropped

    # (g) chạy tiếp: chọn checkpoint HOÀN CHỈNH mới nhất, bỏ qua bản ghi dở / tên lạ
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, "checkpoints")
        assert find_resume_checkpoint(ck) is None                  # chưa có thư mục
        for name, complete in [("checkpoint-200", True), ("checkpoint-400", True),
                               ("checkpoint-600", False), ("checkpoint-abc", True), ("runs", True)]:
            os.makedirs(os.path.join(ck, name))
            if complete:
                open(os.path.join(ck, name, "trainer_state.json"), "w").close()
        assert os.path.basename(find_resume_checkpoint(ck)) == "checkpoint-400"

        # (h) chữ ký run: cùng cấu hình -> khớp; đổi phép tính -> chặn; đổi cách lưu -> vẫn khớp
        cfg = {"run": {"name": "a", "seed": 42}, "model": {"max_pixels": 401408},
               "training": {"epochs": 3, "learning_rate": 2e-4, "save_steps": 200}}
        sig = run_signature(cfg, "sha", 100, None, None)
        assert check_or_write_signature(td, sig) is False          # lần đầu: ghi
        assert check_or_write_signature(td, sig) is True           # lần sau: khớp
        cfg2 = copy.deepcopy(cfg); cfg2["run"]["name"] = "b"; cfg2["training"]["save_steps"] = 50
        assert run_signature(cfg2, "sha", 100, None, None)["sha256"] == sig["sha256"]
        for bad in [run_signature({**cfg, "training": {**cfg["training"], "learning_rate": 1e-4}},
                                  "sha", 100, None, None),
                    run_signature(cfg, "sha_khac", 100, None, None),
                    run_signature(cfg, "sha", 100, None, 30)]:
            try:
                check_or_write_signature(td, bad)
                raise AssertionError("không chặn chạy tiếp với cấu hình/dữ liệu khác")
            except ValueError:
                pass

    # loại quá 1% mẫu -> dừng (W07: thư mục ảnh chép dở từng làm loại 100% mẫu)
    check_drop_ratio(10799, 0); check_drop_ratio(10799, 100)
    for bad in [(10000, 799), (0, 10799)]:
        try:
            check_drop_ratio(*bad)
            raise AssertionError(f"không chặn {bad}")
        except ValueError:
            pass
    _log("drop guard OK")

    # --epochs ghi đè lịch: B2 sơ bộ 1 epoch ViVQA = ceil(10.799 / 16) = 675 step; có trong chữ ký
    cfg1 = load_config(os.path.join(REPO_ROOT, "configs/qwen_lora.yaml"), epochs=1)
    assert cfg1["training"]["epochs"] == 1.0
    assert plan_schedule(10799, 2, 8, 1, 0.03)["total_steps"] == 675, plan_schedule(10799, 2, 8, 1, 0.03)
    cfg3 = load_config(os.path.join(REPO_ROOT, "configs/qwen_lora.yaml"))
    assert run_signature(cfg1, "sha", 10799, None, None)["sha256"] != run_signature(cfg3, "sha", 10799, None, None)["sha256"]
    _log("epochs override OK")

    # W09 --seed: ghi đè run.seed; seed khác -> chữ ký khác; không truyền -> chữ ký cũ giữ nguyên
    cfg_s42 = load_config(os.path.join(REPO_ROOT, "configs/qwen_lora.yaml"), epochs=1)
    cfg_s43 = load_config(os.path.join(REPO_ROOT, "configs/qwen_lora.yaml"), epochs=1, seed=43)
    assert cfg_s42["run"]["seed"] == 42 and cfg_s43["run"]["seed"] == 43
    assert run_signature(cfg_s42, "sha", 8000, None, None)["sha256"] == run_signature(cfg1, "sha", 8000, None, None)["sha256"]
    assert run_signature(cfg_s43, "sha", 8000, None, None)["sha256"] != run_signature(cfg_s42, "sha", 8000, None, None)["sha256"]
    assert run_signature(load_config(os.path.join(REPO_ROOT, "configs/qwen_lora.yaml"), epochs=1, seed=42),
                         "sha", 8000, None, None)["sha256"] == run_signature(cfg_s42, "sha", 8000, None, None)["sha256"]
    _log("seed override OK")

    # W10 --lora_r / --lora_alpha / --lora_targets: không truyền -> chữ ký cũ giữ nguyên;
    # đổi r -> alpha = 2r; preset all_linear_llm -> regex; preset attn r16 == cấu hình cũ (cùng phép tính)
    from src.adaptation.lora_targets import ALL_LINEAR_LLM_REGEX
    cp = os.path.join(REPO_ROOT, "configs/qwen_lora.yaml")
    sig = lambda c: run_signature(c, "sha", 8000, None, None)["sha256"]
    c_old = load_config(cp, epochs=1)
    assert c_old["lora"]["r"] == 16 and c_old["lora"]["lora_alpha"] == 32
    assert sig(load_config(cp, epochs=1, lora_targets="attn", lora_r=16)) == sig(c_old)
    c_r64 = load_config(cp, epochs=1, lora_r=64)
    assert (c_r64["lora"]["r"], c_r64["lora"]["lora_alpha"]) == (64, 128) and sig(c_r64) != sig(c_old)
    c_al4 = load_config(cp, epochs=1, lora_targets="all_linear_llm", lora_r=4)
    assert c_al4["lora"]["target_modules"] == ALL_LINEAR_LLM_REGEX and c_al4["lora"]["lora_alpha"] == 8
    assert len({sig(c_old), sig(c_r64), sig(c_al4)}) == 3
    assert load_config(cp, lora_r=8, lora_alpha=32)["lora"]["lora_alpha"] == 32
    assert load_config(cp)["lora"]["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    _log("lora override OK")

    # loss sau resume: transformers chia tổng loss 7 step cho global_step 12 -> phải hiệu chỉnh
    sl = session_losses(0.8226, 12, 7, [{"loss": 1.6}, {"loss": "1.451"}, {"train_loss": 0.8}])
    assert abs(sl["final_loss"] - 1.4102) < 1e-3 and sl["last_logged_loss"] == 1.451, sl
    assert session_losses(1.2, 30, 30, [])["final_loss"] == 1.2   # không resume -> giữ nguyên
    _log("session_losses OK")

    _selftest_ocr()
    _selftest_mix()

    _log("SELFTEST OK")


def _selftest_ocr() -> None:
    """W08: B3 (OCR) và B2-sub (không OCR) khác chữ ký; prompt train == prompt eval; dùng sai -> lỗi."""
    import tempfile
    from src.data.ocr import MockOCREngine, build_ocr_cache, images_of
    from src.data.vivqa_dataset import load_vivqa
    from src.prompting.builder import prompt_for_sample
    from src.eval.run_eval import run_evaluation
    with tempfile.TemporaryDirectory() as td:
        img = os.path.join(td, "img")
        os.makedirs(img)
        recs = [{"question_id": f"q{i}", "question": f"hỏi {i}?", "answers": ["đáp"], "image": f"{i}.jpg"}
                for i in range(3)]
        for r in recs:
            open(os.path.join(img, r["image"]), "wb").close()
        data = os.path.join(td, "train.json")
        _write_json(data, recs)
        build_ocr_cache(images_of([data], img), MockOCREngine(),
                        os.path.join(td, "data/vitextvqa_official/ocr/mock.json"))
        cfg_path = os.path.join(REPO_ROOT, "configs/qwen_lora.yaml")
        base = load_config(cfg_path, epochs=1)
        s_plain = load_vivqa(data, image_dir=img)
        assert attach_ocr_for_training(copy.deepcopy(base), s_plain, td, "vitextvqa_official",
                                       None, None, None, None, None) is None
        sig_plain = run_signature(base, "sha", 3, None, None)["sha256"]
        assert sig_plain == run_signature(load_config(cfg_path, epochs=1), "sha", 3, None, None)["sha256"]
        cfg_ocr = load_config(cfg_path, epochs=1, mode="ocr")
        s_ocr = load_vivqa(data, image_dir=img)
        st = attach_ocr_for_training(cfg_ocr, s_ocr, td, "vitextvqa_official", "mock", None, None, None, None)
        assert st["n_samples"] == 3 and cfg_ocr["prompting"]["ocr"]["template"] == "v1"
        assert run_signature(cfg_ocr, "sha", 3, None, None)["sha256"] != sig_plain
        # prompt lúc train (collate) == prompt lúc eval (run_evaluation) cho cùng mẫu
        p_train = prompt_for_sample(s_ocr[0], cfg_ocr["prompting"])
        assert p_train.startswith("Văn bản đọc được trong ảnh: MOCK 0.jpg") and p_train.endswith("Câu hỏi: hỏi 0?")
        seen = []
        ecfg = {**cfg_ocr, "data": {"dataset_name": "x"}, "eval": {"primary_metrics": ["exact_match"]}}
        run_evaluation(None, None, s_ocr[:1], ecfg, verbose_every=0,
                       generate_fn=lambda m, p, i, t, max_new_tokens=32: seen.append(t) or "đáp")
        assert seen == [p_train], (seen, p_train)
        for bad in [dict(mode=None, backend="mock"), dict(mode="ocr", backend=None)]:
            try:
                attach_ocr_for_training(load_config(cfg_path, mode=bad["mode"]), load_vivqa(data, image_dir=img),
                                        td, "vitextvqa_official", bad["backend"], None, None, None, None)
                raise AssertionError(f"không chặn {bad}")
            except ValueError:
                pass
    _log("ocr (train) OK")


def _selftest_mix() -> None:
    """W11: run trộn — mỗi phần nhận ĐÚNG OCR như khi train riêng; ảnh không chữ -> prompt = zero-shot;
    thiết lập OCR lệch giữa các dataset -> dừng; xáo lẫn tái lập được; chữ ký khác run đơn."""
    import tempfile
    from src.data.ocr import MockOCREngine, build_ocr_cache, images_of, load_ocr_cache, save_ocr_cache
    from src.data.vivqa_dataset import load_vivqa
    from src.prompting.builder import build_prompt, prompt_for_sample

    # (a) parse_mix
    assert parse_mix(["vivqa=a.json", "vitextvqa_official=b.json"]) == [("vivqa", "a.json"), ("vitextvqa_official", "b.json")]
    for bad in (["vivqa=a.json"], ["vivqa=a.json", "vivqa=b.json"], ["xyz=a.json", "vivqa=b.json"],
                ["vivqa", "vitextvqa_official=b.json"], ["vivqa=", "vitextvqa_official=b.json"]):
        try:
            parse_mix(bad)
            raise AssertionError(f"không chặn --mix {bad}")
        except ValueError:
            pass

    # (b) xáo: tái lập được, trộn đều, không mất/thừa mẫu
    a = [{"question_id": f"a{i}", "source_dataset": "A"} for i in range(100)]
    b = [{"question_id": f"b{i}", "source_dataset": "B"} for i in range(100)]
    s1, s2 = shuffle_mixed(a + b, 42), shuffle_mixed(a + b, 42)
    assert [x["question_id"] for x in s1] == [x["question_id"] for x in s2]
    assert sorted(x["question_id"] for x in s1) == sorted(x["question_id"] for x in a + b)
    assert 20 <= sum(x["source_dataset"] == "A" for x in s1[:64]) <= 44, "64 mẫu đầu không trộn đều"
    assert [x["question_id"] for x in shuffle_mixed(a + b, 43)] != [x["question_id"] for x in s1]

    with tempfile.TemporaryDirectory() as td:
        img = {}
        files = {}
        for ds, n in (("vivqa", 4), ("vitextvqa_official", 3)):
            img[ds] = os.path.join(td, f"img_{ds}")
            os.makedirs(img[ds])
            recs = [{"question_id": f"{ds}_{i}", "question": f"{ds} hỏi {i}?", "answers": ["đáp"],
                     "image": f"{ds}_{i}.jpg"} for i in range(n)]
            for r in recs:
                open(os.path.join(img[ds], r["image"]), "wb").close()
            files[ds] = os.path.join(td, "data", ds, "train.json")
            _write_json(files[ds], recs)
            cache_path = os.path.join(td, "data", ds, "ocr", "mock.json")
            build_ocr_cache(images_of([files[ds]], img[ds]), MockOCREngine(), cache_path)
            if ds == "vivqa":               # ảnh vivqa_0 KHÔNG có chữ (như ảnh COCO thường)
                c = load_ocr_cache(cache_path)
                c["images"]["vivqa_0.jpg"]["lines"] = []
                save_ocr_cache(c, cache_path)
        cp = os.path.join(REPO_ROOT, "configs/qwen_lora.yaml")
        mix = [("vivqa", "data/vivqa/train.json"), ("vitextvqa_official", "data/vitextvqa_official/train.json")]

        cfg_mix = load_config(cp, epochs=1, mode="ocr")
        r = prepare_mixed(cfg_mix, mix, td, None, "mock", None, None, None, image_dirs=img)
        assert len(r["samples"]) == 7 and not r["dropped"]
        assert [(p["dataset"], p["num_samples"]) for p in r["parts"]] == [("vivqa", 4), ("vitextvqa_official", 3)]
        assert {s["source_dataset"] for s in r["samples"]} == {"vivqa", "vitextvqa_official"}
        o = cfg_mix["prompting"]["ocr"]
        assert set(o["subset_sha256_by_dataset"]) == {"vivqa", "vitextvqa_official"} and o["template"] == "v1"

        # (c) phần ViTextVQA nhận ĐÚNG OCR như run đơn (B3) -> cùng dấu vân tay; dữ liệu đơn khác chữ ký
        for ds in ("vivqa", "vitextvqa_official"):
            cfg_one = load_config(cp, epochs=1, mode="ocr")
            st = attach_ocr_for_training(cfg_one, load_vivqa(files[ds], image_dir=img[ds]), td, ds,
                                         "mock", None, None, None, None)
            assert o["subset_sha256_by_dataset"][ds] == st["subset_sha256"], ds
            assert r["ocr_stats"][ds]["subset_sha256"] == st["subset_sha256"]
            assert run_signature(cfg_mix, r["data_sha"], 7, None, None)["sha256"] != \
                   run_signature(cfg_one, _sha256(files[ds]), 7, None, None)["sha256"]
        assert r["data_sha"] == mixed_data_sha([("vivqa", _sha256(files["vivqa"])),
                                                ("vitextvqa_official", _sha256(files["vitextvqa_official"]))])

        # (d) luật S2: ảnh không chữ -> prompt TRÙNG zero-shot (= đầu vào của C1); ảnh có chữ -> thêm dòng OCR
        by_id = {s["question_id"]: s for s in r["samples"]}
        assert prompt_for_sample(by_id["vivqa_0"], cfg_mix["prompting"]) == build_prompt("vivqa hỏi 0?", mode="zero_shot")
        assert prompt_for_sample(by_id["vivqa_1"], cfg_mix["prompting"]).startswith("Văn bản đọc được trong ảnh: MOCK vivqa_1.jpg")

        # (e) chạy lại cùng lệnh -> cùng chữ ký (resume được); OCR lệch giữa 2 dataset -> dừng
        cfg_again = load_config(cp, epochs=1, mode="ocr")
        r2 = prepare_mixed(cfg_again, mix, td, None, "mock", None, None, None, image_dirs=img)
        assert run_signature(cfg_again, r2["data_sha"], 7, None, None)["sha256"] == \
               run_signature(cfg_mix, r["data_sha"], 7, None, None)["sha256"]
        c = load_ocr_cache(os.path.join(td, "data", "vivqa", "ocr", "mock.json"))
        c["engine_info"] = {**c.get("engine_info", {}), "det_model": "KHAC"}
        save_ocr_cache(c, os.path.join(td, "data", "vivqa", "ocr", "mock.json"))
        try:
            prepare_mixed(load_config(cp, epochs=1, mode="ocr"), mix, td, None, "mock", None, None, None, image_dirs=img)
            raise AssertionError("không chặn engine OCR lệch giữa 2 dataset")
        except ValueError:
            pass
    _log("mix (W11) OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train LoRA/QLoRA cho Qwen2.5-VL (PEFT-only).")
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--run_name", help="Tên thư mục kết quả, vd W06_smoke_qlora_qwen25vl_vivqa_mp512_seed42")
    ap.add_argument("--drive_root", default=DEFAULT_DRIVE_ROOT)
    ap.add_argument("--data_file", default=None, help="Ghi đè file train mặc định")
    ap.add_argument("--out_root", default=None, help="Mặc định <drive_root>/experiments")
    ap.add_argument("--adapter_root", default=None, help="Mặc định <drive_root>/adapters")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs/qwen_lora.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="Chỉ lấy N mẫu đầu (smoke run)")
    ap.add_argument("--max_steps", type=int, default=None, help="Dừng sau N step (smoke run)")
    ap.add_argument("--max_pixels", type=int, default=None, help="Ghi đè model.max_pixels")
    ap.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--mode", default=None, help="Ghi đè prompting.mode")
    ap.add_argument("--batch_size", type=int, default=None, help="Ghi đè training.per_device_batch_size")
    ap.add_argument("--grad_accum", type=int, default=None, help="Ghi đè training.gradient_accumulation_steps")
    ap.add_argument("--grad_checkpointing", default=None, choices=["on", "off"],
                    help="Bật/tắt gradient checkpointing (tắt = nhanh hơn, tốn VRAM hơn)")
    ap.add_argument("--save_steps", type=int, default=None,
                    help="Ghi đè training.save_steps; với smoke run (--max_steps) thì BẬT lưu checkpoint")
    ap.add_argument("--debug_crash_at_step", type=int, default=None,
                    help="CHỈ để kiểm thử resume: giả lập Colab đứt tại step N")
    ap.add_argument("--epochs", type=float, default=None,
                    help="Ghi đè training.epochs (vd 1 cho B2 sơ bộ); đổi cả lịch learning rate")
    ap.add_argument("--seed", type=int, default=None,
                    help="Ghi đè run.seed (mặc định theo config = 42); W09 đo độ ổn định theo seed")
    ap.add_argument("--ocr_backend", default=None, help="Backend OCR cache (paddleocr | paddledet_vietocr); cần --mode ocr")
    ap.add_argument("--ocr_cache", default=None, help="Ghi đè đường dẫn OCR cache")
    ap.add_argument("--ocr_template", default=None, help="Mẫu dòng OCR (v1 | v2); mặc định v1")
    ap.add_argument("--ocr_max_chars", type=int, default=None, help="Cắt văn bản OCR quá N ký tự; mặc định 600")
    ap.add_argument("--ocr_min_conf", type=float, default=None, help="Bỏ dòng OCR có độ tin cậy < ngưỡng; mặc định 0")
    ap.add_argument("--lora_r", type=int, default=None,
                    help="W10 ablation H4: ghi đè lora.r; không truyền --lora_alpha thì alpha = 2·r")
    ap.add_argument("--lora_alpha", type=int, default=None, help="Ghi đè lora.lora_alpha")
    ap.add_argument("--lora_targets", default=None, choices=["attn", "all_linear_llm"],
                    help="W10 ablation H3: attn = q,k,v,o (mặc định cũ); all_linear_llm = thêm gate,up,down "
                         "CHỈ ở LLM (regex, không lọt vision)")
    ap.add_argument("--mix", action="append", default=None, metavar="DATASET=FILE",
                    help="W11 adapter dùng chung: lặp lại cho từng dataset, vd --mix vivqa=data/vivqa/train.json "
                         "--mix vitextvqa_official=data/vitextvqa_official/train_subset8000_byimage_seed42.json "
                         "(FILE tương đối so với --drive_root). Thay cho --dataset/--data_file")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return
    mix = parse_mix(a.mix) if a.mix else None
    if not a.run_name or not (a.dataset or mix):
        ap.error("cần --run_name và (--dataset hoặc --mix) (trừ khi --selftest)")
    if mix and (a.dataset or a.data_file):
        ap.error("--mix không dùng cùng --dataset / --data_file")
    train(a.dataset, a.run_name, mix=mix, drive_root=a.drive_root, data_file=a.data_file,
          out_root=a.out_root, adapter_root=a.adapter_root, config=a.config,
          limit=a.limit, max_steps=a.max_steps, max_pixels=a.max_pixels,
          dtype=a.dtype, mode=a.mode, batch_size=a.batch_size, grad_accum=a.grad_accum,
          grad_checkpointing=None if a.grad_checkpointing is None else (a.grad_checkpointing == "on"),
          save_steps=a.save_steps, debug_crash_at_step=a.debug_crash_at_step, epochs=a.epochs, seed=a.seed,
          ocr_backend=a.ocr_backend, ocr_cache=a.ocr_cache, ocr_template=a.ocr_template,
          ocr_max_chars=a.ocr_max_chars, ocr_min_conf=a.ocr_min_conf,
          lora_r=a.lora_r, lora_alpha=a.lora_alpha, lora_targets=a.lora_targets)


if __name__ == "__main__":
    main()
