# Research Gap — ViVQA-VLM (Week 2, evidence-linked)

> Master §3. Each gap is linked to a **Literature_Matrix row ID** (`docs/W02/Literature_Matrix.md`, rows L1–L10)
> or explicitly marked `❌ TODO: cite`. Feasibility is judged against the capstone constraints
> (**PEFT-only**, single small GPU, metrics = EM / VQA-Acc / ANLS). Not copied verbatim from the register.
>
> ⚠️ Because no reference-paper PDFs are in the repo, **quantitative** evidence for most rows is still `❌ TODO`
> (see the matrix). Where a claim needs a number to be defensible, that is stated explicitly.

## Gaps

### G1 — English-centric pretraining under-serves Vietnamese vision-language alignment
- **Claim:** Open VLMs are pretrained predominantly on English data; Vietnamese (a lower-resource, tone-marked language) is served mainly by *emergent* multilingual behavior, not by targeted alignment.
- **Evidence:**
  - **L2 (LLaVA-1.5, Liu et al.)** — multilinguality is documented as an *emergent* effect of text-only ShareGPT data, still error-prone on non-English (secondary in-repo review `docs/02_references/Reading_Notes/SV3_Paper_Review.md §P4 §8`). ✅ qualitative / ⚠️ secondary.
  - **L10 (VLM Survey, Zhang et al., TPAMI 2024)** — framing for English-centric pretraining. `❌ TODO: cite specific page/claim from PDF`.
  - **L1 / L6 (BLIP-2 / Qwen2.5-VL)** — English-heavy pretraining corpora. `❌ TODO: cite`.
- **Feasible in Capstone scope? Partial** — we can *measure* the Vietnamese gap (zero-shot baseline) and *reduce* it with PEFT, but cannot re-pretrain; the alignment deficit itself is only partially closable.

### G2 — Vietnamese VQA is weak on scene-text (OCR, diacritics, code-switching)
- **Claim:** Reading Vietnamese text in images (signage, brands, URLs, phone numbers) with tone marks and Vietnamese↔English code-switching is a distinct, hard sub-problem that general VQA models handle poorly.
- **Evidence:**
  - **L4 (ViTextVQA, Nguyen et al., 2024)** — dedicated Vietnamese scene-text VQA; our own run measured **45,187 QA / 15,086 images**, answers with mean 4.19 tokens and **29,202 unique** open strings (`experiments/W02_dataset_stats/vitextvqa_stats.json`). ✅ source-read (measured).
  - **Real samples** — addresses/phones/brands with diacritics + code-switching (`experiments/W02_dataset_stats/samples/vitextvqa__Text-reading.json`, e.g. `"đc : 594 nguyễn trãi - tp . bắc ninh"`, `"02422 119 111"`, `"kenh14 . vn"`). ✅ source-read.
  - **L2 (LLaVA-1.5)** — resolution is a lever for text-rich images; TextVQA is English-only (highlighting the Vietnamese gap). ⚠️ secondary.
- **Feasible in Capstone scope? Yes** — addressable with OCR-enhanced prompting + higher resolution + ANLS; datasets are in hand.

### G3 — PEFT for a *Vietnamese* VLM is largely unexplored (**focus gap**)
- **Claim:** LoRA/QLoRA and PEFT-for-VQA are validated almost exclusively on English; no work verifies PEFT-only adaptation of a modern VLM for Vietnamese VQA (esp. tone-marked scene-text) on a single small GPU.
- **Evidence:**
  - **L2 (LLaVA-1.5)** — public evidence **LoRA ≈ full fine-tuning** on 7B/13B VLMs (e.g. LoRA-13B POPE F1 = 86.7 vs full-FT 85.9), all English (`docs/02_references/Reading_Notes/SV3_Paper_Review.md §P4 §8`). ⚠️ secondary — justifies PEFT-only.
  - **Supporting (outside the 10-row matrix, but read in-repo):** LoRA (Hu et al., 2022), QLoRA (Dettmers et al., 2023), VL-Adapter (Sung et al., 2022) — all English/text-only; VL-Adapter reports 4.18% params ≈ full-FT on VQA (`docs/02_references/Reading_Notes/SV3_Paper_Review.md §P1–P3`). ⚠️ secondary.
  - **Gap point:** none of the above evaluate Vietnamese or Vietnamese scene-text. `✅ absence-of-evidence` (established by the collective English-only scope of L2 + supporting reviews).
- **Feasible in Capstone scope? Yes (focus)** — squarely inside PEFT-only; measurable with EM/VQA-Acc/ANLS vs. a frozen zero-shot baseline under identical split/seed/hardware.

### G4 — Hallucination in Vietnamese VQA is unmeasured
- **Claim:** VLMs hallucinate objects/text; there is no Vietnamese object-hallucination probe, and text-hallucination (answer contradicting image OCR) is unquantified for Vietnamese.
- **Evidence:**
  - **L9 (POPE, Li et al., EMNLP 2023)** — object-hallucination protocol (balanced yes/no existence polling, F1). Metric values `❌ TODO: cite from PDF`; protocol is usable now. ⚠️ method-level.
  - **L2 (LLaVA-1.5)** — LoRA does not increase (may reduce) POPE hallucination vs full-FT; resolution reduces hallucination (`docs/02_references/Reading_Notes/SV3_Paper_Review.md §P4`). ⚠️ secondary.
- **Feasible in Capstone scope? Partial** — we can build **POPE-vi** (Vietnamese existence probing) and a text-vs-OCR mismatch check on a labeled subset; a full multi-annotator study is time-boxed.

## Focus decision
- **Primary focus:** **G3** (PEFT for Vietnamese VLM) — best evidence-to-feasibility ratio, fully inside PEFT-only, directly measurable.
- **Analysis axes (secondary):** **G2** (scene-text/OCR, via ablation) and **G4** (hallucination, POPE-vi).
- **Partial/deprioritized:** **G1** — measured and partially mitigated, but not the contribution headline (cannot re-pretrain).

## One-paragraph gap statement (for Review 1)
> LoRA/QLoRA are shown to match full fine-tuning for English LLMs and VLMs (L2 + in-repo PEFT reviews), and Vietnamese
> VQA resources exist for both general (L3 ViVQA) and scene-text (L4 ViTextVQA) settings, yet **no study verifies
> PEFT-only adaptation of a modern VLM for Vietnamese VQA — especially tone-marked scene text — under a single-GPU
> budget**, nor quantifies its hallucination behavior in Vietnamese (L9). This capstone targets that gap directly,
> measured with EM / VQA-Accuracy / ANLS against a frozen zero-shot baseline under identical split, seed, and hardware.

## Checklist
- [x] Each gap linked to ≥1 matrix row ID (or explicit `❌ TODO: cite`).
- [x] Feasibility column (Yes / Partial / No + one-line reason).
- [x] Focus gap chosen with justification.
- [ ] Replace `❌ TODO: cite` quantitative cells once primary PDFs are read (see Literature_Matrix TODO list).
