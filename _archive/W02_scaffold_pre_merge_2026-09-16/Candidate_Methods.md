# Candidate Methods — ViVQA-VLM

> **[W3-PREP SCAFFOLD — does NOT block Week-2 100%]** Template + TODOs. Finalize method at W6.
> Constraint: **PEFT-only** (LoRA/QLoRA). No full fine-tuning.

## Baseline candidates (choose exactly ONE to reproduce)
| Model | Size | Vietnamese | Scene-text/OCR | QLoRA feasibility | Note |
|-------|------|-----------|----------------|-------------------|------|
| BLIP-2 | ~2.7B+ | weak | weak | high | fallback (lightest) |
| Florence-2 | 0.23B/0.77B | limited | native OCR task | very high | unified vision |
| **Qwen2.5-VL-3B** | 3B | **strong** | **strong native** | feasible on Colab | **recommended primary** |

- [ ] TODO: confirm GPU + pin versions (transformers/accelerate/bitsandbytes/peft).

## Improvement directions (PEFT-only)
1. **QLoRA (4-bit NF4)** on LLM linear layers (+ optional merger) — core contribution. `❌ TODO: config`
2. **OCR-enhanced prompting** (PaddleOCR vs VietOCR) for ViTextVQA. `❌ TODO`
3. **Vietnamese response-format prompt** (evidence L2 LLaVA). `❌ TODO`
4. (Optional) RAG — only if time permits.

## Ablation grid (draft)
- zero-shot vs prompt-format vs LoRA/QLoRA · ±OCR · target-modules · rank · 4/8/16-bit · POPE-vi before/after.
- Full rationale + starting config: see `docs/literature/SV3_Paper_Review.md §5`.
