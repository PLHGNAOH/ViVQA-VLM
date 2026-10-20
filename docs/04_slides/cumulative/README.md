# Cumulative review deck

One slide deck that grows at every milestone, so the final defense deck already contains every part.

| Part | Content | Status |
|---|---|---|
| A | Review 1 (24/09/2026), revised after the 10 reviewer comments | done (Review 2) |
| B | Review 2 (W07–W09): baseline, QLoRA B2, OCR-aware QLoRA B3 | done (Review 2) |
| C | Review 3 / faculty review (W13) | to add |
| D | Final defense (W15) | to add |

Rules
- Single source of truth: `build_cumulative_deck.js`. Never edit the generated `.pptx` by hand.
- At each new review: revise earlier parts after the feedback, then append the new part to the same builder.
- Hypotheses use one numbering everywhere: H1 QLoRA, H2 OCR, H3 Vietnamese format prompt, H4 hallucination.
  Old Review 1 numbering: H6→H2, H7→H3, H8→H4; old H2 (4-bit) → efficiency measurement; old H3/H4 → ablations A1/A2; old H5 → optional.
- Slide references inside Part B are computed (`OFF`), so Part A can grow without breaking them.

Build
```
SKILL_DIR=<pptx skill dir> node docs/04_slides/cumulative/build_cumulative_deck.js en docs/04_slides/cumulative/ViVQA-VLM_Review2_Cumulative_EN.pptx
SKILL_DIR=<pptx skill dir> node docs/04_slides/cumulative/build_cumulative_deck.js vi docs/04_slides/cumulative/ViVQA-VLM_Review2_Cumulative_TV.pptx
```
Figures are read from `docs/W07/figures/`.
