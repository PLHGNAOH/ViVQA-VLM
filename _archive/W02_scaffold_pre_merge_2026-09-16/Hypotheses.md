# Hypotheses — ViVQA-VLM

> **[W3-PREP SCAFFOLD — does NOT block Week-2 100%]** Template + TODOs. Finalize for Review 1 (W3).
> Every hypothesis must state the **metric** and the **baseline** it is compared against.

| ID | Hypothesis (draft) | Metric | Baseline / condition | Status |
|----|--------------------|--------|----------------------|--------|
| H1 | QLoRA adaptation improves VQA-Accuracy on ViVQA by ≥ X pts vs zero-shot | VQA-Acc / EM | frozen zero-shot, same test split & seed | `❌ TODO: set X` |
| H2 | OCR-enhanced prompting improves ANLS on ViTextVQA vs no-OCR | ANLS (thr 0.5) | same PEFT model, ±OCR | `❌ TODO` |
| H3 | All-linear LoRA targets > attention-only at equal param budget | EM/ANLS + %params | attention-only LoRA | `❌ TODO` (evidence: QLoRA review, in-repo) |
| H4 | PEFT does not increase (may reduce) hallucination | POPE-vi F1 | zero-shot | `❌ TODO` (evidence: L2 LLaVA) |

- [ ] TODO: turn each into a falsifiable statement with a threshold once baseline zero-shot numbers exist (W4–W5).
