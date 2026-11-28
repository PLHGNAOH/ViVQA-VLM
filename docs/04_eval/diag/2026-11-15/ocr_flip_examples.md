# OCR flip examples (team reference — NOT for thesis)

Status: **N/A** — no SH-without-OCR predictions on ViVQA test in repo.

- SH+OCR preds: `experiments/W11_SH_eval_qlora_ocr_vietocr_v1_r16_shared_ep1_vivqa_test_bf16_L4_seed42/predictions.json`
- SH−OCR preds: missing
- Reported +0.15 EM pts (C1→SH, with-OCR subset): 0.1506 (match |Δ−0.15|≤0.01 = True)
- McNemar p (C1 vs SH, with-OCR): 0.9035973456835125

No R2W seed-42 sample list (requires paired SH±OCR).
