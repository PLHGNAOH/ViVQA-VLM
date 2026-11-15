"""
bench_latency.py — đo latency end-to-end prototype (không UI).

Load pipeline 1 lần → warm-up 1 câu → mỗi example × 3 lần:
  t_preprocess, t_ocr_det, t_ocr_rec (+n dòng), t_model (+n token), t_postprocess, t_total.

Chạy (venv vivqa-demo, từ gốc repo):
  python -m src.prototype.bench_latency \\
    --adapter D:\\vivqa_models\\SH\\epoch_1 \\
    --examples_json D:\\vivqa_models\\measure_2026-11-13\\examples_local.json \\
    --ocr_device cpu --out_dir docs/04_eval/latency

Ghi: latency_baseline_<YYYY-MM-DD>.csv + reference_outputs.json (nếu --write_reference).
So khớp: --compare_reference reference_outputs.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import tempfile
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGE_KEYS = (
    "t_preprocess", "t_ocr_det", "t_ocr_rec", "t_ocr", "t_model", "t_postprocess", "t_total",
)


def _median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def _norm_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ln in lines or []:
        out.append({
            "text": ln.get("text", ""),
            "conf": round(float(ln.get("conf", 0.0)), 4),
            "box": [int(x) for x in (ln.get("box") or [])],
        })
    return out


def _ref_record(ex: Dict[str, Any], r: Dict[str, Any], matched: List[int]) -> Dict[str, Any]:
    return {
        "image": ex["image"],
        "question": ex["question"],
        "answer": r["answer"],
        "confidence": float(r["confidence"]),
        "level": r["level"],
        "ocr_lines": _norm_lines(r.get("ocr_lines") or []),
        "matched_line": list(matched),
    }


def _refs_equal(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, str]:
    for k in ("answer", "level"):
        if a.get(k) != b.get(k):
            return False, f"{k}: {a.get(k)!r} ≠ {b.get(k)!r}"
    if abs(float(a["confidence"]) - float(b["confidence"])) > 1e-9:
        return False, f"confidence: {a['confidence']} ≠ {b['confidence']}"
    if _norm_lines(a.get("ocr_lines") or []) != _norm_lines(b.get("ocr_lines") or []):
        return False, "ocr_lines mismatch"
    if list(a.get("matched_line") or []) != list(b.get("matched_line") or []):
        return False, f"matched_line: {a.get('matched_line')} ≠ {b.get('matched_line')}"
    return True, "ok"


def run_once(backend, image_path: str, question: str, tmp_dir: str,
             ocr_lines=None, ocr_timing=None) -> Dict[str, Any]:
    from src.prototype.app import prepare_image, evidence_lines
    used, info = prepare_image(image_path, tmp_dir)
    try:
        r = backend.ask(used, question, ocr_lines=ocr_lines, ocr_timing=ocr_timing)
        matched = evidence_lines(r["answer"], r.get("ocr_lines") or [])
        return {
            "r": r,
            "matched": matched,
            "img_info": info,
            "used": used,
            "t_preprocess": float(r.get("sec_preprocess") or 0.0),
            "t_ocr_det": float(r.get("sec_ocr_det") or 0.0),
            "t_ocr_rec": float(r.get("sec_ocr_rec") or 0.0),
            "t_ocr": float(r.get("sec_ocr") or 0.0),
            "t_model": float(r.get("sec_model") or 0.0),
            "t_postprocess": float(r.get("sec_postprocess") or 0.0),
            "t_total": float(r.get("sec_total") or 0.0),
            "n_ocr_lines": int(r.get("n_ocr_lines") or len(r.get("ocr_lines") or [])),
            "n_gen_tokens": int(r.get("n_gen_tokens") or r.get("n_tokens") or 0),
            "ocr_source": r.get("ocr_source", "fresh"),
        }
    finally:
        if used and used != image_path and os.path.exists(used):
            try:
                os.remove(used)
            except OSError:
                pass


def load_pipeline(adapter: str, config: str, ocr_device: str, expect_sha: Optional[str]):
    from src.prototype.app import load_ui_settings, load_ocr_selection, make_real_backend
    st, ocr_cfg = load_ui_settings(), load_ocr_selection()
    backend, client, info = make_real_backend(
        adapter, config, st, ocr_cfg, ocr_device=ocr_device, expect_adapter_sha=expect_sha,
    )
    return backend, client, info, st, ocr_cfg


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=r"D:\vivqa_models\SH\epoch_1")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "qwen_lora.yaml"))
    ap.add_argument("--examples_json", default=r"D:\vivqa_models\measure_2026-11-13\examples_local.json")
    ap.add_argument("--ocr_device", default="cpu")
    ap.add_argument("--expect_adapter_sha",
                    default="c9791e961aaa58fdcd7d9baafede3bd7bd8c902acc9cc43917306f1e0fecc354")
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "docs", "04_eval", "latency"))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--tag", default="baseline", help="tên file: latency_<tag>_<date>.csv")
    ap.add_argument("--write_reference", action="store_true")
    ap.add_argument("--compare_reference", default=None)
    ap.add_argument("--scenario", choices=["plain", "a", "b", "c", "all_scenarios"], default="plain",
                    help="plain=baseline; a/b/c=kịch bản sau tối ưu; all_scenarios=a+b+c")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.examples_json, encoding="utf-8") as f:
        examples = json.load(f)
    examples = [ex for ex in examples if os.path.exists(ex["image"])]
    if not examples:
        raise SystemExit("Không có example hợp lệ")

    print("Loading pipeline…")
    backend, client, info, st, ocr_cfg = load_pipeline(
        args.adapter, args.config, args.ocr_device, args.expect_adapter_sha,
    )
    print("Start info:", json.dumps({k: info[k] for k in info if k != "ocr_engine"}, ensure_ascii=False))
    tmp = tempfile.mkdtemp(prefix="vivqa_bench_")

    try:
        warm = examples[0]
        print(f"Warm-up: {os.path.basename(warm['image'])}")
        run_once(backend, warm["image"], warm["question"], tmp)

        ref_path = args.compare_reference or os.path.join(args.out_dir, "reference_outputs.json")
        reference = None
        if args.compare_reference or (not args.write_reference and os.path.exists(ref_path) and args.tag != "baseline"):
            if args.compare_reference:
                with open(args.compare_reference, encoding="utf-8") as f:
                    reference = json.load(f)

        scenarios = ["plain"]
        if args.scenario == "all_scenarios":
            scenarios = ["a", "b", "c"]
        elif args.scenario != "plain":
            scenarios = [args.scenario]

        all_rows: List[Dict[str, Any]] = []
        ref_out: List[Dict[str, Any]] = []
        parity_ok = parity_n = 0

        for scen in scenarios:
            print(f"\n=== scenario={scen} ===")
            for ei, ex in enumerate(examples):
                trials_data = []
                # scenario helpers: OCR cache via backend.ask(ocr_lines=...)
                cached_lines = None
                cached_timing = None
                if scen in ("b", "c"):
                    # prefetch OCR on prepared image (như A1), optionally wait 5s (b)
                    from src.prototype.app import prepare_image
                    used_pf, _ = prepare_image(ex["image"], tmp)
                    try:
                        raw = backend.ocr_read(used_pf) if hasattr(backend, "ocr_read") else {
                            "lines": backend.ocr_fn(used_pf), "sec_det": 0.0, "sec_rec": 0.0, "sec": 0.0,
                        }
                    finally:
                        if used_pf and used_pf != ex["image"] and os.path.exists(used_pf):
                            try:
                                os.remove(used_pf)
                            except OSError:
                                pass
                    cached_lines = list(raw.get("lines") or [])
                    cached_timing = {
                        "sec_det": float(raw.get("sec_det") or 0.0),
                        "sec_rec": float(raw.get("sec_rec") or 0.0),
                        "sec_ocr": float(raw.get("sec") or 0.0),
                        "ocr_source": "prefetch",
                    }
                    if scen == "b":
                        time.sleep(5.0)
                        cached_timing["ocr_source"] = "prefetch_ready"

                for trial in range(args.trials):
                    if scen == "a":
                        # first question on new image, click immediately → fresh OCR
                        out = run_once(backend, ex["image"], ex["question"], tmp)
                    elif scen in ("b", "c"):
                        # reuse OCR; c = second question → still cached (same lines)
                        src = "cache" if scen == "c" or trial > 0 else cached_timing["ocr_source"]
                        timing = {**cached_timing, "ocr_source": src, "sec_ocr": 0.0,
                                  "sec_det": 0.0, "sec_rec": 0.0}
                        # wall-clock OCR wait counted 0 when already ready; for trial0 of b, already waited
                        out = run_once(backend, ex["image"], ex["question"], tmp,
                                       ocr_lines=cached_lines, ocr_timing=timing)
                    else:
                        out = run_once(backend, ex["image"], ex["question"], tmp)
                    trials_data.append(out)
                    print(f"  [{scen}] ex{ei} trial{trial}: total={out['t_total']:.3f}s "
                          f"ocr={out['t_ocr']:.3f} model={out['t_model']:.3f} "
                          f"lines={out['n_ocr_lines']} tok={out['n_gen_tokens']} src={out['ocr_source']}")

                # median across trials
                med = {k: _median([t[k] for t in trials_data]) for k in STAGE_KEYS}
                last = trials_data[-1]
                row = {
                    "scenario": scen,
                    "example_idx": ei,
                    "image": os.path.basename(ex["image"]),
                    "question": ex["question"],
                    "n_ocr_lines": last["n_ocr_lines"],
                    "n_gen_tokens": last["n_gen_tokens"],
                    **{f"med_{k}": round(med[k], 4) for k in STAGE_KEYS},
                    "answer": last["r"]["answer"],
                    "level": last["r"]["level"],
                }
                all_rows.append(row)

                if scen == "plain" or scen == "a":
                    # reference from first successful plain/a run (trial 0 of first scenario writing ref)
                    pass
                rec = _ref_record(ex, last["r"], last["matched"])
                if args.write_reference and scen in ("plain", "a") and args.trials >= 1:
                    # use trial 0 for reference stability
                    rec = _ref_record(ex, trials_data[0]["r"], trials_data[0]["matched"])
                    if scen == scenarios[0]:
                        ref_out.append(rec)

                if reference is not None:
                    # match by image+question
                    key = (ex["image"], ex["question"])
                    ref_item = next((x for x in reference if (x["image"], x["question"]) == key), None)
                    parity_n += 1
                    if ref_item is None:
                        print(f"  PARITY FAIL: missing reference for {key}")
                    else:
                        ok, msg = _refs_equal(rec, ref_item)
                        if ok:
                            parity_ok += 1
                        else:
                            print(f"  PARITY FAIL ex{ei}: {msg}")

        day = date.today().isoformat()
        csv_path = os.path.join(args.out_dir, f"latency_{args.tag}_{day}.csv")
        fieldnames = list(all_rows[0].keys()) if all_rows else []
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote {csv_path}")

        # summary table by scenario
        print("\n=== MEDIAN by stage (across examples) ===")
        for scen in scenarios:
            rows = [r for r in all_rows if r["scenario"] == scen]
            if not rows:
                continue
            print(f"scenario={scen} (n={len(rows)} examples)")
            for k in STAGE_KEYS:
                vals = [r[f"med_{k}"] for r in rows]
                print(f"  {k:16s}  median-of-medians={_median(vals):.3f}s  "
                      f"mean={statistics.mean(vals):.3f}s")

        if args.write_reference:
            # rebuild reference from a dedicated plain pass if empty
            if not ref_out:
                for ex in examples:
                    out = run_once(backend, ex["image"], ex["question"], tmp)
                    ref_out.append(_ref_record(ex, out["r"], out["matched"]))
            ref_file = os.path.join(args.out_dir, "reference_outputs.json")
            with open(ref_file, "w", encoding="utf-8") as f:
                json.dump(ref_out, f, ensure_ascii=False, indent=2)
            print(f"Wrote {ref_file} ({len(ref_out)} items)")

        if reference is not None:
            print(f"\nG1 PARITY: {parity_ok}/{parity_n}")
            if parity_ok != parity_n:
                return 2

        # also write baseline alias if tag=baseline
        if args.tag == "baseline":
            alias = os.path.join(args.out_dir, f"latency_baseline_{day}.csv")
            if os.path.abspath(alias) != os.path.abspath(csv_path):
                import shutil
                shutil.copy2(csv_path, alias)
                print(f"Alias {alias}")

    finally:
        client.close()
        try:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
