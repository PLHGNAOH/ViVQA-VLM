# Literature Matrix — ViVQA-VLM (Week 2)

> Scope: the **10 project references** listed in `docs/master_instruction.md` (§References).
> Language: English (thesis-bound). Inline notes may be Vietnamese; technical terms in English.
>
> **Verification legend (per Hard Rule #2):**
> - `✅ source-read (file/page)` — value extracted from a source actually read.
> - `⚠️ secondary / partial` — value from an in-repo team review or from title/abstract only; primary PDF **not** verified.
> - `❌ TODO` — no source read; value deliberately left empty rather than invented.
>
> **Numbers policy (Hard Rule #1 & #3):** No paper metric (EM / VQA-Acc / ANLS / F1 / …) is filled unless it
> comes from a source in this repo that was read. **No reference-paper PDFs exist in the repo** (searched
> `docs/`, `refs/`, `assets/`, `docs/02_references/` on 2026-09-14 — 0 PDFs found; the `docs/02_references/Reading_Notes/`
> files are empty templates). Therefore every paper-reported metric is `❌ TODO` **except** LLaVA-1.5, for which a
> detailed, table-cited in-repo review exists (`docs/literature/SV3_Paper_Review.md` §P4) — tagged `⚠️ secondary`.
> Dataset **sizes** for ViVQA / ViTextVQA are `✅ source-read` because they were *measured* by our own run
> (`experiments/W02_dataset_stats/*.json`, seed 42), but the paper-reported figures still need PDF confirmation.

## Matrix

| ID | Citation (Author, Year, Venue) | Category | Core idea | Method | Dataset(s) | Key results (metric=value) | What we can use for ViVQA-VLM | Verified? |
|----|--------------------------------|----------|-----------|--------|------------|----------------------------|-------------------------------|-----------|
| L1 | Li et al., 2023, ICML | VLM foundation | Bridge a **frozen** image encoder and a **frozen** LLM with a lightweight querying module, so vision-language alignment is learned cheaply. | BLIP-2: **Q-Former** (learnable query tokens) trained in 2 stages (representation learning + generative learning) between frozen ViT and frozen LLM (OPT/Flan-T5). | (paper) COCO, VQAv2, OK-VQA, GQA, NoCaps, Flickr30k. | EM/VQA-Acc = `❌ TODO: verify from source` (no PDF in repo) | Candidate baseline (light, HF-available). Q-Former is where SV3 can attach LoRA. English-centric → motivates G1. | ❌ TODO (metrics); ⚠️ method from register/general |
| L2 | Liu et al., 2023, NeurIPS (Visual Instruction Tuning) + LLaVA-1.5, Liu et al., 2024, CVPR | VLM foundation | Visual instruction tuning: connect a vision encoder to an LLM via a simple projector and train on GPT-generated multimodal instructions; LLaVA-1.5 shows a strong, reproducible recipe. | ViT-L/14 (336px) → **MLP projector** → Vicuna LLM; VQA academic data + a response-format prompt; public **LoRA** and full-FT checkpoints. | (paper/eval) VQAv2, GQA, VizWiz, ScienceQA, TextVQA, POPE, MME, MMBench, SEED, MM-Vet. | LoRA-13B: VQAv2=80.0, GQA=63.3, TextVQA=60.2, POPE(F1)=86.7, MME=1541.7, MMBench=68.5; full-FT-13B: VQAv2=80.0, TextVQA=61.3, POPE=85.9 `⚠️ secondary (SV3_Paper_Review.md §P4 §8; from LLaVA Model Zoo, primary PDF not in repo)` | Reference architecture for Qwen2.5-VL; evidence LoRA≈full-FT on VLMs (justifies PEFT-only); Vietnamese response-format prompt; POPE hallucination probing; resolution lever. | ⚠️ secondary (numbers from in-repo review, not primary PDF) |
| L3 | Tran et al., 2021, PACLIC | Vietnamese VQA (dataset) | First general Vietnamese VQA benchmark; questions/answers in Vietnamese over natural images. | Dataset construction (Vietnamese VQA pairs over COCO-style images) + baseline VQA models. | **ViVQA**. Measured from HF mirror `minhnguyent546/ViVQA`: 15,000 QA / 10,328 images (train 11,999 + test 3,001) `✅ source-read (experiments/W02_dataset_stats/vivqa_stats.json)`. Paper-reported #images/#QA + official split = `❌ TODO: verify from source (PDF)`. | Baseline VQA-Acc from paper = `❌ TODO: verify from source` | Primary **general** dataset. Answers are short (≥99% ≤3 words) → EM/VQA-Acc feasible. Diacritics/counting cues drive our question-type heuristic. Evidence for G2. | ✅ sizes (measured); ❌ TODO (paper metrics/official split) |
| L4 | Nguyen et al., 2024, arXiv:2404.10652 (ext. Expert Systems w/ Applications, 2025) | Vietnamese scene-text VQA (dataset) | Vietnamese **scene-text** VQA: answering requires reading text embedded in the image. | Dataset of images with Vietnamese scene text + QA; OCR-augmented baselines; ANLS as the headline metric. | **ViTextVQA**. Measured from HF mirror `nhonhoccode/ViTextVQA`: 45,187 QA / 15,086 images (train 35,159 + test 10,028; **validation split empty in this mirror**) `✅ source-read (experiments/W02_dataset_stats/vitextvqa_stats.json)`. Paper-reported #images/#QA + official split = `❌ TODO: verify from source (PDF)`. | Baseline ANLS from paper = `❌ TODO: verify from source` | Primary **scene-text** dataset; drives OCR (PaddleOCR/VietOCR) and **ANLS**. Long, brand/URL/phone answers with code-switching → central to G2. Evidence dataset. | ✅ sizes (measured); ❌ TODO (paper metrics/official split) |
| L5 | Xiao et al., 2024, CVPR | VLM baseline (unified vision) | A single **unified** vision foundation model handling detection, captioning, grounding, and OCR via a sequence-to-sequence, prompt-driven interface. | Florence-2: DaViT vision encoder + transformer encoder-decoder; trained on large multi-task "FLD-5B" annotations; task specified by text prompt. | (paper) FLD-5B; COCO det/caption, OCR, grounding benchmarks. | EM/VQA-Acc = `❌ TODO: verify from source` | Very small (0.23B/0.77B) → highest PEFT/GPU feasibility; native OCR task useful for ViTextVQA. Candidate baseline. | ❌ TODO (metrics); ⚠️ method from register/general |
| L6 | Bai et al., 2025, Technical Report (arXiv:2502.13923) | VLM baseline (open, multilingual) | Open multilingual VLM with strong native OCR/scene-text and dynamic resolution. | Qwen2.5-VL: ViT vision encoder + merger + Qwen2.5 LLM; dynamic resolution (`min_pixels/max_pixels`); instruction-tuned; 3B/7B/72B. | (paper) DocVQA, TextVQA, OCRBench, MMMU, etc. | EM/VQA-Acc/ANLS = `❌ TODO: verify from source` | **Recommended primary baseline** (3B): best Vietnamese + scene-text among candidates; fits QLoRA on Colab. LoRA target = LLM linear layers + optional merger. | ❌ TODO (metrics); ⚠️ method from register/general |
| L7 | Nguyen et al., 2024, Computers & Electrical Engineering, vol. 119 (Vietnamese VQA with Transformer & Convolutional Integration) | Vietnamese VQA (method) | Combine transformer and convolutional features for Vietnamese VQA. | Hybrid Transformer + CNN visual/text integration for Vietnamese VQA. | (paper) Vietnamese VQA benchmark(s) incl. ViVQA. | EM/VQA-Acc = `❌ TODO: verify from source` | Non-VLM Vietnamese VQA point of comparison; positions our PEFT-VLM approach; possible related-work contrast. | ❌ TODO (metrics); ⚠️ citation from register |
| L8 | Chen et al., 2025, CVPR (Florence-VL) | VLM method | Enrich an LLM-based VLM with **Florence-2** generative vision features (depth-breadth fusion) for better perception. | Florence-VL: Florence-2 features + depth-breadth fusion into an LLM decoder. | (paper) standard VLM/VQA + OCR/chart benchmarks. | EM/VQA-Acc = `❌ TODO: verify from source` | Shows unified-vision features (Florence-2) help VQA/OCR → supports a Florence-2-backed ablation and OCR-aware design. | ❌ TODO (metrics); ⚠️ citation from register |
| L9 | Li et al., 2023, EMNLP (POPE) | Hallucination (evaluation) | Measure **object hallucination** via balanced yes/no polling about object existence, avoiding caption-parsing bias. | POPE: sample present/absent objects (random/popular/adversarial) and ask existence questions; report Accuracy/Precision/Recall/**F1**. | (paper) MSCOCO (+ A-OKVQA, GQA) object sets. | POPE F1 = `❌ TODO: verify from source` | Blueprint for **POPE-vi** (Vietnamese yes/no existence probing) in W12 hallucination analysis (SV4). Evidence for G4. | ❌ TODO (metrics); ⚠️ method from register/general |
| L10 | Zhang et al., 2024, IEEE TPAMI (Vision-Language Models for Vision Tasks: A Survey) | Survey | Systematize VLM pretraining objectives, transfer (incl. PEFT), and evaluation across vision tasks. | Survey/taxonomy of datasets, architectures, pretraining objectives, transfer/PEFT, benchmarks. | (survey) aggregates many datasets/benchmarks. | N/A (survey) — specific claims = `❌ TODO: verify from source` | Framing for related work; cite for English-centric pretraining (G1) and for positioning PEFT/eval choices. | ❌ TODO (specific claims); ⚠️ citation from register |

## What-we-can-use rationale (master §14) — expanded

- **L1 BLIP-2 / L5 Florence-2 / L6 Qwen2.5-VL** are the three candidate baselines. Per the master plan we reproduce **exactly one** (recommended: **Qwen2.5-VL-3B-Instruct**) and may use a second only as an ablation *reference* backbone.
- **L2 LLaVA-1.5** supplies the strongest in-repo-documented evidence that **LoRA ≈ full fine-tuning** on 7B/13B VLMs, directly justifying the PEFT-only constraint, plus the response-format-prompt and resolution levers.
- **L3 ViVQA / L4 ViTextVQA** are the two datasets we analyze in D2–D4 (general vs. scene-text). ViTextVQA anchors the **ANLS** metric and the OCR pipeline.
- **L7 / L8** are recent-method comparators (non-VLM Vietnamese VQA; Florence-2-enriched VLM).
- **L9 POPE** defines the hallucination protocol we localize to Vietnamese (POPE-vi).
- **L10 Survey** frames the gaps (English-centric pretraining, evaluation).

## TODO — papers still unread (primary PDFs NOT in repo)

All 10 primary PDFs are missing from the repo; the following need to be obtained and read to replace `❌ TODO` metric cells:

- [ ] **L1 BLIP-2** (Li et al., ICML 2023) — extract VQAv2/OK-VQA/GQA numbers + page cites.
- [ ] **L3 ViVQA** (Tran et al., PACLIC 2021) — paper-reported #images/#QA, official split, baseline VQA-Acc (compare to our measured 15,000 QA).
- [ ] **L4 ViTextVQA** (Nguyen et al., 2024) — paper-reported size/split, baseline ANLS (compare to our measured 45,187 QA; note validation split missing in the HF mirror).
- [ ] **L5 Florence-2** (Xiao et al., CVPR 2024) — task-wise results + page cites.
- [ ] **L6 Qwen2.5-VL** (Bai et al., 2025) — DocVQA/TextVQA/OCRBench numbers for 3B/7B.
- [ ] **L7 VN-VQA Transformer+Conv** (Nguyen et al., CEE 2024) — method + results.
- [ ] **L8 Florence-VL** (Chen et al., CVPR 2025) — fusion details + results.
- [ ] **L9 POPE** (Li et al., EMNLP 2023) — F1 across random/popular/adversarial + page cites.
- [ ] **L10 VLM Survey** (Zhang et al., TPAMI 2024) — pull specific English-centric / PEFT claims with page cites.
- [x] **L2 LLaVA-1.5** — numbers covered *secondarily* by `docs/literature/SV3_Paper_Review.md` §P4; still verify against the primary PDF / Model Zoo page.

> Note: `docs/literature/SV3_Paper_Review.md` also contains fully-worked, table-cited reviews of **LoRA (Hu et al., 2022)**, **QLoRA (Dettmers et al., 2023)**, and **VL-Adapter (Sung et al., 2022)**. These are the *core PEFT techniques of record* but are **not** among the 10 project references above, so they are intentionally excluded from this matrix (see that file for their verified metrics).

## Ma trận Gap × Paper (từ weekly/W02 đã archive)

Cột `#1`–`#10` theo **bảng weekly cũ** (không phải L1–L10). Weekly #1=BLIP-2, #2=Florence-2, #3=Qwen2.5-VL, #4=ViVQA, #5=ViTextVQA, #6=LoRA, #7=QLoRA, #8=POPE, #9=VL-Adapter, #10=LLaVA-1.5.

TODO (W3): remap cột sang L1–L10. Thứ tự weekly ≠ L1–L10 (weekly #2 = Florence-2 vs L2 = LLaVA; weekly #6/#7/#9 = LoRA/QLoRA/VL-Adapter không nằm trong 10 ref chính thức).

| Gap | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 |
|-----|----|----|----|----|----|----|----|----|----|-----|
| G1 English-centric pretraining | | | | | | ✓ | ✓ | | ✓ | ✓ |
| G2 Vietnamese VQA yếu, scene-text | | | | | | | | | ✓ | ✓ |
| G3 PEFT cho VLM tiếng Việt chưa có | | | | | | ✓ | ✓ | | ✓ | ✓ |
| G4 Hallucination | | | | | | | | | | ✓ |
