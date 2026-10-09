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
                epochs: Optional[float] = None) -> Dict[str, Any]:
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
    from src.prompting.builder import build_prompt, to_chat_messages

    mode = cfg["prompting"]["mode"]
    # Padding bên PHẢI: build_labels giả định prompt nằm ở đầu chuỗi.
    processor.tokenizer.padding_side = "right"

    def collate(batch: List[Dict[str, Any]]):
        texts, prompt_texts, images_per_sample = [], [], []
        for s in batch:
            image = Image.open(s["image_path"]).convert("RGB")
            messages = to_chat_messages(image, build_prompt(s["question"], mode=mode))
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
          epochs: Optional[float] = None) -> Dict[str, Any]:
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

    if dataset not in DATASETS:
        raise ValueError(f"--dataset phải là một trong {list(DATASETS)}")

    cfg = copy.deepcopy(load_config(config, max_pixels=max_pixels, dtype=dtype, mode=mode,
                                    batch_size=batch_size, grad_accum=grad_accum,
                                    grad_checkpointing=grad_checkpointing, epochs=epochs))
    cfg["run"]["name"] = run_name
    seed = cfg["run"]["seed"]
    set_seed(seed)

    data_file = data_file or os.path.join(drive_root, DATASETS[dataset]["train_rel"])
    out_dir = os.path.join(out_root or os.path.join(drive_root, "experiments"), run_name)
    adapter_dir = os.path.join(adapter_root or os.path.join(drive_root, "adapters"), run_name)

    _log(f"== {run_name} | dataset={dataset} | mode={cfg['prompting']['mode']}")
    _log(f"   data : {data_file}")
    _log(f"   out  : {out_dir}")
    _log(f"   adapt: {adapter_dir}")

    # --- Dữ liệu ---
    if image_dir is None:
        from src.eval.run_baseline import prepare_images
        image_dir = prepare_images(dataset, drive_root)
    samples = load_vivqa(data_file, image_dir=image_dir, max_samples=limit)
    samples, dropped = filter_trainable(samples)
    if not samples:
        raise ValueError("Không còn mẫu nào train được sau khi lọc")
    check_drop_ratio(len(samples), len(dropped), image_dir)
    _log(f"   {len(samples)} mẫu train (loại {len(dropped)} mẫu thiếu ảnh/đáp án)")

    # --- Chạy tiếp hay chạy mới? (kiểm tra TRƯỚC khi nạp model 4 phút) ---
    t = cfg["training"]
    if save_steps is not None:
        t["save_steps"] = int(save_steps)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    data_sha = _sha256(data_file)
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
        "dataset": dataset,
        "data_file": os.path.relpath(data_file, drive_root) if data_file.startswith(drive_root) else data_file,
        "data_file_sha256": data_sha,
        "num_samples": len(samples),
        "num_dropped": len(dropped),
        "answer_policy": "đáp án đầu tiên trong trường answers",
        "prompting_mode": cfg["prompting"]["mode"],
        "model_id": cfg["model"]["model_id"],
        "max_pixels": cfg["model"]["max_pixels"],
        "min_pixels": cfg["model"]["min_pixels"],
        "torch_dtype": compute_dtype,
        "quantization": cfg["quantization"],
        "lora": cfg["lora"],
        "training": {**t, "max_steps": max_steps, "limit": limit, **sched,
                     "unsupported_training_args": unsupported},
        "run_signature_sha256": sig["sha256"],
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

    # loss sau resume: transformers chia tổng loss 7 step cho global_step 12 -> phải hiệu chỉnh
    sl = session_losses(0.8226, 12, 7, [{"loss": 1.6}, {"loss": "1.451"}, {"train_loss": 0.8}])
    assert abs(sl["final_loss"] - 1.4102) < 1e-3 and sl["last_logged_loss"] == 1.451, sl
    assert session_losses(1.2, 30, 30, [])["final_loss"] == 1.2   # không resume -> giữ nguyên
    _log("session_losses OK")

    _log("SELFTEST OK")


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
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return
    if not a.dataset or not a.run_name:
        ap.error("cần --dataset và --run_name (trừ khi --selftest)")
    train(a.dataset, a.run_name, drive_root=a.drive_root, data_file=a.data_file,
          out_root=a.out_root, adapter_root=a.adapter_root, config=a.config,
          limit=a.limit, max_steps=a.max_steps, max_pixels=a.max_pixels,
          dtype=a.dtype, mode=a.mode, batch_size=a.batch_size, grad_accum=a.grad_accum,
          grad_checkpointing=None if a.grad_checkpointing is None else (a.grad_checkpointing == "on"),
          save_steps=a.save_steps, debug_crash_at_step=a.debug_crash_at_step, epochs=a.epochs)


if __name__ == "__main__":
    main()
