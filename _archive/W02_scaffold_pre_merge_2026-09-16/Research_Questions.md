# Research Questions — ViVQA-VLM

> **[W3-PREP SCAFFOLD — does NOT block Week-2 100%]** Template + TODOs. To be finalized for Review 1 (W3).
> Constraint of record: **PEFT-only** (LoRA/QLoRA, no full fine-tuning). Metrics: EM / VQA-Acc / ANLS.

## Primary RQ (measurable, PEFT-only)
- **RQ0 (draft):** To what extent can **PEFT-only** adaptation (LoRA/QLoRA) of a modern open VLM improve Vietnamese VQA over a frozen zero-shot baseline, measured by EM / VQA-Acc (ViVQA) and ANLS (ViTextVQA) under identical split/seed/hardware?
  - [ ] TODO: fix baseline model (recommended Qwen2.5-VL-3B) and datasets/splits.

## Secondary RQs
- **RQ1 (draft):** Does **OCR-enhanced prompting** improve ANLS on ViTextVQA scene-text questions vs. no OCR? `❌ TODO: refine`
- **RQ2 (draft):** How do **LoRA target modules / rank / quantization (4-bit vs 8-bit)** trade off accuracy vs. efficiency (%trainable params, VRAM, time)? `❌ TODO`
- **RQ3 (draft):** Does PEFT change **hallucination** (POPE-vi) relative to zero-shot? `❌ TODO`

## Links
- Evidence gaps: `docs/W02/Research_Gap.md` (G1–G4).
- Methods: `docs/W02/Candidate_Methods.md`. Hypotheses: `docs/W02/Hypotheses.md`.
