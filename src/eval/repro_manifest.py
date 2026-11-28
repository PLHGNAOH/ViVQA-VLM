"""
repro_manifest.py — Biên bản các model chính thức của khoá luận (W14, bước 5, gói tái lập).

Vấn đề: số liệu nằm rải ở ~80 thư mục experiments/, adapter nằm trên Drive. Gói tái lập cần MỘT chỗ ghi:
model nào là chính thức, adapter nào (sha256), notebook nào tạo ra, run eval nào cho ra con số trong báo cáo,
và thư viện nào đã dùng. Script này ĐỌC log đã có (không tính lại gì) rồi ghi configs/official_models.json.

  --build              đọc experiments/ -> configs/official_models.json (+ bảng Markdown docs/W14/official_models.md)
  --check_adapters D   kiểm từng adapter trên Drive (D = thư mục gốc ViVQA-VLM trên Drive, vd G:\\My Drive\\ViVQA-VLM):
                       đủ file + sha256 trọng số + sha256 config trùng biên bản
  --selftest

Chạy:  python -m src.eval.repro_manifest --build
       python -m src.eval.repro_manifest --check_adapters "G:\\My Drive\\ViVQA-VLM"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_JSON = os.path.join("configs", "official_models.json")
OUT_MD = os.path.join("docs", "W14", "official_models.md")
FORMAT = "vivqa-vlm/official-models/v1"

# id, vai trò, nhãn, train run (None = zero-shot), epoch adapter, notebook tạo ra, eval runs {tập: run}
OFFICIAL: List[Dict[str, Any]] = [
    {"id": "r0", "role": "tái lập baseline", "label": "R0 — Qwen2.5-VL-3B zero-shot trên TextVQA val (so số công bố 79,3)",
     "train_run": None, "epoch": None, "notebooks": ["06_r0_textvqa_repro_qwen25vl.ipynb"],
     "evals": {"textvqa_val": "W07_R0_textvqa_val_qwen25vl3b_paper_bf16_L4"}},
    {"id": "b1", "role": "baseline", "label": "B1 — zero-shot Qwen2.5-VL-3B-Instruct (4-bit NF4)",
     "train_run": None, "epoch": None, "notebooks": ["02_run_baseline_qwen25vl.ipynb", "08_fulltest_vitextvqa_b1_b2.ipynb"],
     "evals": {"vivqa_test": "W07_B1_zeroshot_qwen25vl_vivqa_test_bf16_L4_seed42",
               "vitextvqa_testfull": "W07_B1_zeroshot_qwen25vl_vitextvqa_testfull_bf16_L4_seed42"}},
    {"id": "b2_vitextvqa", "role": "bước thích nghi", "label": "B2 — QLoRA r16, ViTextVQA train đầy đủ, 1 epoch, không OCR",
     "train_run": "W07_B2_qlora_r16_qwen25vl_vitextvqa_ep1_bf16_L4_seed42", "epoch": "epoch_1",
     "notebooks": ["04_qlora_qwen25vl.ipynb", "08_fulltest_vitextvqa_b1_b2.ipynb"],
     "evals": {"vitextvqa_testfull": "W07_B2_eval_qlora_r16_ep1_vitextvqa_testfull_bf16_L4_seed42"}},
    {"id": "c3_vivqa", "role": "chính thức ViVQA", "label": "C3 — QLoRA r16, ViVQA train, lịch 2 epoch, adapter sau epoch 2",
     "train_run": "W09_B2_qlora_r16_qwen25vl_vivqa_ep2_bf16_L4_seed42", "epoch": "epoch_2",
     "notebooks": ["16_qlora_vivqa_2epoch_select_qwen25vl.ipynb"],
     "evals": {"vivqa_test": "W09_C3_eval_qlora_r16_vivqa_test_bf16_L4_seed42"}},
    {"id": "b3full", "role": "chính thức ViTextVQA (đóng góp chính)",
     "label": "B3-full — QLoRA nhận biết OCR (VietOCR · v1 · 600), ViTextVQA train đầy đủ, 1 epoch",
     "train_run": "W10_B3full_qlora_ocr_vietocr_v1_r16_qwen25vl_vitextvqa_full_ep1_bf16_L4_seed42", "epoch": "epoch_1",
     "notebooks": ["18_qlora_ocr_b3full_vs_b2_qwen25vl.ipynb"],
     "evals": {"vitextvqa_testfull": "W10_B3full_eval_qlora_ocr_vietocr_v1_r16_full_ep1_vitextvqa_testfull_bf16_L4_seed42",
               "vitextvqa_testfull_exiffix (độ nhạy W14)":
                   "W14_B3full_eval_exiffix_qlora_ocr_vietocr_v1_r16_full_ep1_vitextvqa_testfull_bf16_L4_seed42"}},
    {"id": "sh", "role": "prototype (adapter dùng chung)",
     "label": "SH — QLoRA nhận biết OCR dùng chung (ViVQA + ViTextVQA 8k), 1 epoch; prototype dùng τ* = +4,85",
     "train_run": "W11_SH_qlora_ocr_vietocr_v1_r16_qwen25vl_shared_vivqa_sub8k_ep1_bf16_L4_seed42", "epoch": "epoch_1",
     "notebooks": ["20_qlora_shared_adapter_qwen25vl.ipynb", "28_prototype_v2_live_demo_sh_qwen25vl.ipynb"],
     "evals": {"vivqa_test": "W11_SH_eval_qlora_ocr_vietocr_v1_r16_shared_ep1_vivqa_test_bf16_L4_seed42",
               "vitextvqa_test2000": "W11_SH_eval_qlora_ocr_vietocr_v1_r16_shared_ep1_vitextvqa_test2000_bf16_L4_seed42"}},
]
METRIC_KEYS = ("exact_match", "anls", "token_f1", "vqa_accuracy")


def _load(path: str) -> Optional[Dict[str, Any]]:
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def _drive_rel(abs_path: Optional[str]) -> Optional[str]:
    """/content/drive/MyDrive/ViVQA-VLM/adapters/X/epoch_1 -> adapters/X/epoch_1"""
    if not abs_path:
        return None
    p = abs_path.replace("\\", "/")
    k = p.find("ViVQA-VLM/")
    return p[k + len("ViVQA-VLM/"):] if k >= 0 else p


def build(exp_root: str) -> Dict[str, Any]:
    models = []
    for spec in OFFICIAL:
        m: Dict[str, Any] = {k: spec[k] for k in ("id", "role", "label", "notebooks")}
        m["train_run"], m["evals"], m["adapter"] = spec["train_run"], {}, None
        for tag, run in spec["evals"].items():
            mj = _load(os.path.join(exp_root, run, "metrics.json"))
            if mj is None:
                raise FileNotFoundError(f"thiếu experiments/{run}/metrics.json")
            meta = mj.get("meta", {})
            e = {"run": run, "metrics_pct": {k: round(100 * v, 2) for k, v in mj["metrics"].items() if k in METRIC_KEYS},
                 "n": meta.get("num_samples"), "data_file": meta.get("data_file"),
                 "data_file_sha256": meta.get("data_file_sha256"), "git_commit": meta.get("git_commit"),
                 "finished_utc": meta.get("finished_utc"), "sec_per_sample": meta.get("sec_per_sample"),
                 "peak_vram_gb": meta.get("peak_vram_gb_this_session"), "prompting_mode": meta.get("prompting_mode"),
                 "gpu": (meta.get("hardware") or {}).get("gpus_seen")}
            if meta.get("ocr"):
                o = meta["ocr"]
                e["ocr"] = {k: o.get(k) for k in ("backend", "cache_file", "template", "max_chars", "min_conf")}
            if "verdict" in mj:
                e["verdict"] = mj["verdict"]
            a = meta.get("adapter")
            if a:
                ad = {"drive_path": _drive_rel(a.get("path")), "weights_sha256": a.get("weights_sha256"),
                      "config_sha256": a.get("config_sha256"), "weights_mb": a.get("weights_mb")}
                if m["adapter"] is None:
                    m["adapter"] = ad
                elif m["adapter"]["weights_sha256"] != ad["weights_sha256"]:
                    raise ValueError(f"{spec['id']}: các run eval dùng adapter KHÁC nhau ({run})")
            m["evals"][tag] = e
        if spec["train_run"]:
            tl = _load(os.path.join(exp_root, spec["train_run"], "train_log.json"))
            if tl is None:
                raise FileNotFoundError(f"thiếu experiments/{spec['train_run']}/train_log.json")
            t = tl["meta"]
            cost = t.get("cost") or {}
            m["train"] = {"data_file": t.get("data_file"), "data_file_sha256": t.get("data_file_sha256"),
                          "num_samples": t.get("num_samples"), "seed": t.get("seed"), "lora": t.get("lora"),
                          "trainable_params": t.get("trainable_params"), "trainable_pct": t.get("trainable_pct"),
                          "training": {k: (t.get("training") or {}).get(k) for k in
                                       ("epochs", "per_device_batch_size", "gradient_accumulation_steps",
                                        "learning_rate", "warmup_ratio")},
                          "prompting_mode": t.get("prompting_mode"), "quantization": t.get("quantization"),
                          "max_pixels": t.get("max_pixels"), "train_hours": round((cost.get("train_runtime_sec") or 0) / 3600, 2),
                          "peak_vram_gb": cost.get("peak_vram_gb"), "gpu": (t.get("hardware") or {}).get("gpu"),
                          "git_commit": t.get("git_commit"), "versions": t.get("versions"),
                          "epoch_adapters": t.get("epoch_adapters"), "official_epoch": spec["epoch"]}
            if m["adapter"] and spec["epoch"] and not (m["adapter"]["drive_path"] or "").endswith(spec["epoch"]):
                raise ValueError(f"{spec['id']}: adapter eval {m['adapter']['drive_path']} không phải {spec['epoch']}")
        models.append(m)
    return {"format": FORMAT, "what": "Model chính thức của khoá luận ViVQA-VLM — sinh từ experiments/ bằng "
                                      "src/eval/repro_manifest.py (không gõ tay). Adapter nằm trên Drive, không lên git.",
            "base_model": "Qwen/Qwen2.5-VL-3B-Instruct", "models": models}


def to_markdown(man: Dict[str, Any]) -> str:
    L = ["# Model chính thức (sinh tự động từ `configs/official_models.json`)", "",
         "| id | Vai trò | Run eval | EM | ANLS | F1 | Adapter (Drive) | sha256 (12) | Notebook |", "|---|---|---|---|---|---|---|---|---|"]
    def f(x):
        return f"{x:.2f}".replace(".", ",") if isinstance(x, (int, float)) else str(x)
    for m in man["models"]:
        ad = m.get("adapter") or {}
        for tag, e in m["evals"].items():
            mp = {k: f(v) for k, v in e["metrics_pct"].items()}
            L.append(f"| {m['id']} | {m['role']} | {tag}: `{e['run']}` | {mp.get('exact_match', mp.get('vqa_accuracy', '—'))} | "
                     f"{mp.get('anls', '—')} | {mp.get('token_f1', '—')} | `{ad.get('drive_path') or '— (zero-shot)'}` | "
                     f"{(ad.get('weights_sha256') or '—')[:12]} | {', '.join(m['notebooks'])} |")
    return "\n".join(L) + "\n"


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_adapters(man: Dict[str, Any], drive_root: str) -> List[Dict[str, Any]]:
    out = []
    for m in man["models"]:
        ad = m.get("adapter")
        if not ad:
            continue
        d = os.path.join(drive_root, *ad["drive_path"].split("/"))
        w, c = os.path.join(d, "adapter_model.safetensors"), os.path.join(d, "adapter_config.json")
        r = {"id": m["id"], "dir": d, "exists": os.path.isfile(w) and os.path.isfile(c)}
        if r["exists"]:
            r["weights_ok"] = sha256_file(w) == ad["weights_sha256"]
            r["config_ok"] = sha256_file(c) == ad["config_sha256"]
        r["ok"] = r["exists"] and r.get("weights_ok", False) and r.get("config_ok", False)
        out.append(r)
    return out


def _selftest() -> None:
    import tempfile
    man = build(os.path.join(REPO_ROOT, "experiments"))
    ids = [m["id"] for m in man["models"]]
    assert ids == ["r0", "b1", "b2_vitextvqa", "c3_vivqa", "b3full", "sh"], ids
    by = {m["id"]: m for m in man["models"]}
    assert by["b3full"]["evals"]["vitextvqa_testfull"]["metrics_pct"]["exact_match"] == 53.35
    assert by["c3_vivqa"]["evals"]["vivqa_test"]["metrics_pct"]["exact_match"] == 76.64
    assert by["b2_vitextvqa"]["evals"]["vitextvqa_testfull"]["metrics_pct"]["exact_match"] == 50.26
    assert by["c3_vivqa"]["adapter"]["drive_path"].endswith("epoch_2") and by["b1"]["adapter"] is None
    assert by["sh"]["adapter"]["weights_sha256"].startswith("c9791e961aaa")
    assert by["b3full"]["train"]["trainable_params"] == 7372800 and by["b3full"]["train"]["num_samples"] == 35159
    assert "TÁI LẬP" in by["r0"]["evals"]["textvqa_val"]["verdict"]
    assert "53.35" in to_markdown(man) or "53,35" in to_markdown(man)
    with tempfile.TemporaryDirectory() as td:                      # kiểm adapter: đúng / sai sha / thiếu file
        fake = {"models": [{"id": "x", "adapter": {"drive_path": "adapters/X/epoch_1"}},
                           {"id": "y", "adapter": {"drive_path": "adapters/Y/epoch_1"}}, {"id": "z", "adapter": None}]}
        d = os.path.join(td, "adapters", "X", "epoch_1")
        os.makedirs(d)
        open(os.path.join(d, "adapter_model.safetensors"), "wb").write(b"w")
        open(os.path.join(d, "adapter_config.json"), "wb").write(b"c")
        fake["models"][0]["adapter"].update({"weights_sha256": sha256_file(os.path.join(d, "adapter_model.safetensors")),
                                             "config_sha256": "sai"})
        r = check_adapters(fake, td)
        assert len(r) == 2 and r[0]["exists"] and r[0]["weights_ok"] and not r[0]["config_ok"] and not r[0]["ok"]
        assert not r[1]["exists"] and not r[1]["ok"]
    print("✅ selftest repro_manifest OK (6 model, số khớp báo cáo, adapter đúng epoch, kiểm sha256 adapter)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check_adapters", metavar="DRIVE_ROOT")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if a.build:
        man = build(os.path.join(REPO_ROOT, "experiments"))
        with open(os.path.join(REPO_ROOT, OUT_JSON), "w", encoding="utf-8", newline="\n") as f:
            json.dump(man, f, ensure_ascii=False, indent=1)
            f.write("\n")
        with open(os.path.join(REPO_ROOT, OUT_MD), "w", encoding="utf-8", newline="\n") as f:
            f.write(to_markdown(man))
        print(f"-> {OUT_JSON}, {OUT_MD} ({len(man['models'])} model)")
    if a.check_adapters:
        man = json.load(open(os.path.join(REPO_ROOT, OUT_JSON), encoding="utf-8"))
        res = check_adapters(man, a.check_adapters)
        for r in res:
            print(f"{'✅' if r['ok'] else '❌'} {r['id']:14s} {r['dir']} | có file {r['exists']} | "
                  f"trọng số {r.get('weights_ok')} | config {r.get('config_ok')}")
        if not all(r["ok"] for r in res):
            raise SystemExit("❌ có adapter thiếu hoặc khác biên bản")
        print(f"✅ {len(res)}/{len(res)} adapter trùng biên bản")
    if not (a.build or a.check_adapters):
        ap.print_help()


if __name__ == "__main__":
    main()
