"""
lora_targets.py — Chọn lớp gắn LoRA cho Qwen2.5-VL và KIỂM rằng LoRA không lọt vào vision encoder.

Vấn đề (W10, ablation H3/H4):
  Qwen2.5-VL có HAI khối dùng tên lớp giống nhau:
    - LLM decoder : ...layers.<i>.self_attn.{q,k,v,o}_proj  và  ...layers.<i>.mlp.{gate,up,down}_proj
    - Vision tower: ...visual.blocks.<i>.attn.{qkv,proj}     và  ...visual.blocks.<i>.mlp.{gate,up,down}_proj
  PEFT khớp `target_modules` dạng LIST theo HẬU TỐ tên lớp. Viết ["gate_proj", "up_proj", "down_proj"]
  sẽ gắn LoRA vào CẢ MLP của vision tower -> ablation "all-linear" không còn so sánh sạch với
  attention-only (khác cả vị trí lẫn ngân sách tham số). Cấu hình attention-only hiện tại an toàn
  (vision dùng `qkv`/`proj`, không có q_proj), nhưng mọi cấu hình mới phải đi qua module này.

Giải pháp:
  1. PRESETS: tên preset -> target_modules. Preset có MLP dùng REGEX (PEFT khớp bằng re.fullmatch)
     chỉ nhận lớp `layers.<i>.(self_attn|mlp)...` và loại mọi tên chứa "visual".
  2. `audit_lora_modules(model)`: sau khi gắn LoRA, liệt kê lớp đã gắn theo khối (LLM / vision) và
     theo hậu tố, đếm tham số LoRA thật. Có lớp vision -> lỗi (dừng trước khi tốn 3 giờ GPU).
  3. `expected_lora_params(...)`: số tham số tính tay từ kích thước kiến trúc (để đối chiếu log và
     ghi vào thesis: "cùng ngân sách tham số").

Module thuần Python ở phần preset/đếm (không cần torch) -> kiểm thử offline được:
    python -m src.adaptation.lora_targets --selftest
"""
from __future__ import annotations

import argparse
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

ATTN_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_SUFFIXES = ("gate_proj", "up_proj", "down_proj")

# Regex khớp ĐÚNG lớp tuyến tính của LLM decoder (không lớp nào của vision tower, không lm_head).
ALL_LINEAR_LLM_REGEX = (r"^(?!.*visual).*layers\.\d+\."
                        r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$")

# preset -> (target_modules đưa vào LoraConfig, các hậu tố phải được gắn)
PRESETS: Dict[str, Dict[str, Any]] = {
    # Giữ NGUYÊN list cũ (W07–W09) -> chữ ký và adapter_config các run cũ không đổi.
    "attn": {"target_modules": list(ATTN_SUFFIXES), "suffixes": ATTN_SUFFIXES},
    "all_linear_llm": {"target_modules": ALL_LINEAR_LLM_REGEX, "suffixes": ATTN_SUFFIXES + MLP_SUFFIXES},
}

# Kích thước LLM decoder của Qwen2.5-VL-3B-Instruct (config.json trên HF).
QWEN25VL_3B_LLM = {"model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "num_layers": 36, "hidden": 2048,
                   "intermediate": 11008, "num_heads": 16, "num_kv_heads": 2}


def preset_target_modules(name: str) -> Union[List[str], str]:
    if name not in PRESETS:
        raise ValueError(f"--lora_targets phải là một trong {list(PRESETS)}")
    tm = PRESETS[name]["target_modules"]
    return list(tm) if isinstance(tm, (list, tuple)) else tm


def preset_of(target_modules: Union[Sequence[str], str]) -> Optional[str]:
    """Ngược lại: target_modules trong config -> tên preset (None nếu không khớp preset nào)."""
    for name, p in PRESETS.items():
        tm = p["target_modules"]
        if isinstance(tm, str) and isinstance(target_modules, str) and tm == target_modules:
            return name
        if not isinstance(tm, str) and not isinstance(target_modules, str) and list(tm) == list(target_modules):
            return name
    return None


def peft_matches(key: str, target_modules: Union[Sequence[str], str]) -> bool:
    """Mô phỏng luật khớp của PEFT (peft.tuners.tuners_utils.check_target_module_exists):
    chuỗi -> re.fullmatch; list -> key == t hoặc key kết thúc bằng '.' + t."""
    if isinstance(target_modules, str):
        return re.fullmatch(target_modules, key) is not None
    return any(key == t or key.endswith("." + t) for t in target_modules)


def tower_of(name: str) -> str:
    return "vision" if "visual" in name else "llm"


def suffix_of(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def expected_lora_params(r: int, suffixes: Iterable[str], arch: Dict[str, int] = QWEN25VL_3B_LLM) -> int:
    """Số tham số LoRA = sum_lớp r * (in + out). Chỉ LLM decoder (vision không gắn)."""
    h, inter = arch["hidden"], arch["intermediate"]
    kv = arch["num_kv_heads"] * (h // arch["num_heads"])
    shapes = {"q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h),
              "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h)}
    per_layer = sum(r * (shapes[s][0] + shapes[s][1]) for s in suffixes)
    return per_layer * arch["num_layers"]


def summarize_targets(names: Iterable[str]) -> Dict[str, Any]:
    """Gom danh sách tên lớp đã gắn LoRA thành bảng đếm theo khối và hậu tố."""
    names = list(names)
    by_tower: Dict[str, int] = {}
    by_suffix: Dict[str, int] = {}
    for n in names:
        by_tower[tower_of(n)] = by_tower.get(tower_of(n), 0) + 1
        by_suffix[suffix_of(n)] = by_suffix.get(suffix_of(n), 0) + 1
    return {"n_modules": len(names), "by_tower": dict(sorted(by_tower.items())),
            "by_suffix": dict(sorted(by_suffix.items())),
            "vision_examples": [n for n in names if tower_of(n) == "vision"][:5]}


def check_summary(summary: Dict[str, Any], expected_suffixes: Optional[Sequence[str]] = None,
                  num_layers: Optional[int] = None, allow_vision: bool = False) -> None:
    """Dừng nếu LoRA lọt vào vision, hoặc (khi biết preset) thiếu/thừa hậu tố, hoặc số lớp/hậu tố lệch."""
    if not allow_vision and summary["by_tower"].get("vision", 0):
        raise ValueError(f"LoRA đã gắn vào {summary['by_tower']['vision']} lớp của VISION tower "
                         f"(vd {summary['vision_examples']}) -> sửa target_modules (dùng preset all_linear_llm)")
    if summary["n_modules"] == 0:
        raise ValueError("Không lớp nào được gắn LoRA -> target_modules không khớp tên lớp của model")
    if expected_suffixes is not None:
        got = set(summary["by_suffix"])
        if got != set(expected_suffixes):
            raise ValueError(f"Hậu tố được gắn {sorted(got)} ≠ kỳ vọng {sorted(expected_suffixes)}")
        if num_layers is not None:
            bad = {s: c for s, c in summary["by_suffix"].items() if c != num_layers}
            if bad:
                raise ValueError(f"Mỗi hậu tố phải có đúng {num_layers} lớp (1 lớp/layer), lệch: {bad}")


def audit_lora_modules(model, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Sau get_peft_model: lớp nào đã gắn LoRA, bao nhiêu tham số LoRA thật, có lọt vision không.

    Đếm tham số LoRA trực tiếp từ lora_A/lora_B (không phụ thuộc cách PEFT đếm model 4-bit).
    Lỗi -> raise trước khi train (mất ≈ 4 phút nạp model thay vì 3 giờ train sai cấu hình).
    """
    names, lora_params = [], 0
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and len(getattr(mod, "lora_A")):
            names.append(name)
            for sub in list(mod.lora_A.values()) + list(mod.lora_B.values()):
                lora_params += sum(p.numel() for p in sub.parameters())
    summary = summarize_targets(names)
    lcfg = cfg["lora"]
    preset = preset_of(lcfg["target_modules"])
    expected_suffixes = PRESETS[preset]["suffixes"] if preset else None
    arch = QWEN25VL_3B_LLM if cfg["model"].get("model_id") == QWEN25VL_3B_LLM["model_id"] else None
    check_summary(summary, expected_suffixes, arch["num_layers"] if arch else None,
                  allow_vision=bool(lcfg.get("allow_vision", False)))
    expected = expected_lora_params(int(lcfg["r"]), expected_suffixes) if (arch and expected_suffixes) else None
    if expected is not None and expected != lora_params:
        raise ValueError(f"Tham số LoRA đếm được {lora_params:,} ≠ tính tay {expected:,} -> kiểm kiến trúc/target_modules")
    return {**summary, "preset": preset, "lora_params_counted": lora_params,
            "lora_params_expected": expected, "r": int(lcfg["r"]), "lora_alpha": lcfg["lora_alpha"]}


# --------------------------------------------------------------------------- #
# Kiểm thử offline — tên lớp lấy theo bố cục transformers 5.x VÀ bố cục cũ (4.49–4.51)
# --------------------------------------------------------------------------- #
def _fake_module_names(layout: str, n_layers: int = 36, n_blocks: int = 32) -> List[str]:
    if layout == "v5":
        llm, vis = "model.language_model.layers", "model.visual.blocks"
        merger = "model.visual.merger.mlp"
    else:   # bố cục cũ: model.layers / visual.blocks
        llm, vis, merger = "model.layers", "visual.blocks", "visual.merger.mlp"
    names = []
    for i in range(n_layers):
        names += [f"{llm}.{i}.self_attn.{s}" for s in ATTN_SUFFIXES]
        names += [f"{llm}.{i}.mlp.{s}" for s in MLP_SUFFIXES]
    for i in range(n_blocks):
        names += [f"{vis}.{i}.attn.qkv", f"{vis}.{i}.attn.proj"]
        names += [f"{vis}.{i}.mlp.{s}" for s in MLP_SUFFIXES]
    names += [f"{merger}.0", f"{merger}.2", "lm_head", "model.embed_tokens"]
    return names


class _P:
    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n


class _Lin:
    """Lớp tuyến tính giả: nếu được gắn LoRA thì có lora_A/lora_B (dict adapter -> module có parameters())."""
    def __init__(self, fan_in=0, fan_out=0, r=0):
        if r:
            self.lora_A = {"default": _Mod(r * fan_in)}
            self.lora_B = {"default": _Mod(fan_out * r)}


class _Mod:
    def __init__(self, n):
        self._n = n

    def parameters(self):
        return [_P(self._n)]


class _FakePeft:
    def __init__(self, names, target_modules, r, arch=QWEN25VL_3B_LLM):
        h, inter = arch["hidden"], arch["intermediate"]
        kv = arch["num_kv_heads"] * (h // arch["num_heads"])
        shp = {"q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h),
               "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h)}
        self._mods = [("", object())]
        for n in names:
            hit = peft_matches(n, target_modules)
            fi, fo = shp.get(suffix_of(n), (1280, 1280))
            self._mods.append(("base_model.model." + n, _Lin(fi, fo, r if hit else 0)))

    def named_modules(self):
        return iter(self._mods)


def _selftest_audit() -> None:
    names = _fake_module_names("v5")
    for preset, r, n_mod in (("attn", 16, 144), ("attn", 64, 144), ("all_linear_llm", 4, 252)):
        cfg = {"model": {"model_id": QWEN25VL_3B_LLM["model_id"]},
               "lora": {"r": r, "lora_alpha": 2 * r, "target_modules": preset_target_modules(preset)}}
        a = audit_lora_modules(_FakePeft(names, cfg["lora"]["target_modules"], r), cfg)
        assert a["by_tower"] == {"llm": n_mod} and a["preset"] == preset, a
        assert a["lora_params_counted"] == a["lora_params_expected"] == expected_lora_params(r, PRESETS[preset]["suffixes"])
    # list hậu tố ngây thơ -> lọt vision -> audit phải dừng
    naive = list(ATTN_SUFFIXES + MLP_SUFFIXES)
    cfg = {"model": {"model_id": QWEN25VL_3B_LLM["model_id"]}, "lora": {"r": 4, "lora_alpha": 8, "target_modules": naive}}
    try:
        audit_lora_modules(_FakePeft(names, naive, 4), cfg)
        raise AssertionError("audit không chặn LoRA lọt vision")
    except ValueError:
        pass
    print("lora audit OK")


def _selftest() -> None:
    # (a) số tham số: attention r=16 phải khớp đúng số đã log ở W07–W09 (7.372.800)
    assert expected_lora_params(16, ATTN_SUFFIXES) == 7_372_800
    assert expected_lora_params(64, ATTN_SUFFIXES) == 29_491_200
    assert expected_lora_params(4, ATTN_SUFFIXES + MLP_SUFFIXES) == 7_483_392   # cùng ngân sách (+1,5%)
    assert expected_lora_params(16, ATTN_SUFFIXES + MLP_SUFFIXES) == 29_933_568

    for layout in ("v5", "old"):
        names = _fake_module_names(layout)
        # (b) preset attn (list cũ): đúng 36×4 lớp LLM, 0 lớp vision
        s = summarize_targets(n for n in names if peft_matches(n, preset_target_modules("attn")))
        assert s["by_tower"] == {"llm": 144} and s["by_suffix"] == {k: 36 for k in ATTN_SUFFIXES}, s
        check_summary(s, ATTN_SUFFIXES, 36)
        # (c) preset all_linear_llm (regex): đúng 36×7 lớp LLM, 0 lớp vision, không lm_head/merger
        s = summarize_targets(n for n in names if peft_matches(n, preset_target_modules("all_linear_llm")))
        assert s["by_tower"] == {"llm": 252}, s
        assert s["by_suffix"] == {k: 36 for k in ATTN_SUFFIXES + MLP_SUFFIXES}, s
        check_summary(s, ATTN_SUFFIXES + MLP_SUFFIXES, 36)
        # (d) BẪY: list hậu tố MLP ngây thơ -> lọt 32×3 lớp vision -> phải bị chặn
        naive = list(ATTN_SUFFIXES + MLP_SUFFIXES)
        s = summarize_targets(n for n in names if peft_matches(n, naive))
        assert s["by_tower"] == {"llm": 252, "vision": 96}, s
        try:
            check_summary(s)
            raise AssertionError("không chặn LoRA lọt vào vision")
        except ValueError:
            pass

    # (e) thiếu hậu tố / sai số lớp -> chặn
    for bad in [{"n_modules": 108, "by_tower": {"llm": 108}, "by_suffix": {"q_proj": 36, "k_proj": 36, "v_proj": 36}},
                {"n_modules": 140, "by_tower": {"llm": 140},
                 "by_suffix": {"q_proj": 36, "k_proj": 36, "v_proj": 36, "o_proj": 32}},
                {"n_modules": 0, "by_tower": {}, "by_suffix": {}}]:
        try:
            check_summary({**bad, "vision_examples": []}, ATTN_SUFFIXES, 36)
            raise AssertionError(f"không chặn {bad}")
        except ValueError:
            pass

    # (e2) audit_lora_modules trên model giả (cùng giao diện named_modules / lora_A / lora_B như PEFT)
    _selftest_audit()

    # (f) preset_of nhận lại preset (để chữ ký run cũ không bị gán nhầm)
    assert preset_of(["q_proj", "k_proj", "v_proj", "o_proj"]) == "attn"
    assert preset_of(ALL_LINEAR_LLM_REGEX) == "all_linear_llm"
    assert preset_of(["q_proj", "v_proj"]) is None
    try:
        preset_target_modules("all_linear")
        raise AssertionError("không chặn preset lạ")
    except ValueError:
        pass
    print("lora targets OK")
    print("SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preset target_modules LoRA cho Qwen2.5-VL + kiểm không lọt vision.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", action="store_true", help="In bảng preset + số tham số tính tay theo r")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    for name, p in PRESETS.items():
        for r in (4, 16, 64):
            print(f"{name:<15} r={r:<3} target_modules={p['target_modules']!r:.60} "
                  f"params={expected_lora_params(r, p['suffixes']):,}")


if __name__ == "__main__":
    main()
