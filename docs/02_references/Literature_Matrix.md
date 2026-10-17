# Literature Matrix — ViVQA-VLM (canonical sống)

> Scope: the **10 project references** listed in `docs/master_instruction.md` (§References).
> Language: English (thesis-bound). Inline notes may be Vietnamese; technical terms in English.
> **Bản sống duy nhất.** Pointer W02: `docs/W02/Literature_Matrix.md`.
>
> **Verification legend (per Hard Rule #2):**
> - `✅ source-read (file/page)` — value extracted from a source actually read.
> - `⚠️ secondary / partial` — value from an in-repo team review or from title/abstract only; primary PDF **not** verified.
> - `❌ TODO` — no source read; value deliberately left empty rather than invented.
>
> **Numbers policy:** No paper metric is filled unless it comes from a source in this repo that was read.
> **No reference-paper PDFs exist in the repo** (re-checked 2026-09-17: 0 PDFs under `docs/02_references/` for L1–L10).
> Per-paper templates in `docs/02_references/Reading_Notes/` (BLIP2, Florence2, Qwen25VL, ViVQA, ViTextVQA, POPE, LoRA, QLoRA) are still empty `_…_` frames.
> Dataset **sizes** for ViVQA / ViTextVQA are `✅ source-read` (measured, `experiments/W02_dataset_stats/*.json`, seed 42).
> W03 (2026-09-17): remapped Gap×Paper to **L1–L10 by paper name**; added metric skeleton; fixed dead Reading Notes paths (folder `literature` cũ).

## Matrix

| ID | Citation (Author, Year, Venue) | Category | Core idea | Method | Dataset(s) | Key results (metric=value) | What we can use for ViVQA-VLM | Verified? |
|----|--------------------------------|----------|-----------|--------|------------|----------------------------|-------------------------------|-----------|
| L1 | Li et al., 2023, ICML | VLM foundation | Bridge a **frozen** image encoder and a **frozen** LLM with a lightweight querying module, so vision-language alignment is learned cheaply. | BLIP-2: **Q-Former** (learnable query tokens) trained in 2 stages (representation learning + generative learning) between frozen ViT and frozen LLM (OPT/Flan-T5). | (paper) COCO, VQAv2, OK-VQA, GQA, NoCaps, Flickr30k. | EM/VQA-Acc = `❌ TODO: verify from source` (no PDF in repo) | Candidate baseline (light, HF-available). Q-Former is where SV3 can attach LoRA. English-centric → motivates G1. | ❌ TODO (metrics); ⚠️ method from register/general |
| L2 | Liu et al., 2023, NeurIPS (Visual Instruction Tuning) + LLaVA-1.5, Liu et al., 2024, CVPR | VLM foundation | Visual instruction tuning: connect a vision encoder to an LLM via a simple projector and train on GPT-generated multimodal instructions; LLaVA-1.5 shows a strong, reproducible recipe. | ViT-L/14 (336px) → **MLP projector** → Vicuna LLM; VQA academic data + a response-format prompt; public **LoRA** and full-FT checkpoints. | (paper/eval) VQAv2, GQA, VizWiz, ScienceQA, TextVQA, POPE, MME, MMBench, SEED, MM-Vet. | LoRA-13B: VQAv2=80.0, GQA=63.3, TextVQA=60.2, POPE(F1)=86.7, MME=1541.7, MMBench=68.5; full-FT-13B: VQAv2=80.0, TextVQA=61.3, POPE=85.9 `⚠️ secondary (Reading_Notes/SV3_Paper_Review.md §P4; from LLaVA Model Zoo, primary PDF not in repo)` | Reference architecture for Qwen2.5-VL; evidence LoRA≈full-FT on VLMs (justifies PEFT-only); Vietnamese response-format prompt; POPE hallucination probing; resolution lever. | ⚠️ secondary (numbers from in-repo review, not primary PDF) |
| L3 | Tran et al., 2021, PACLIC | Vietnamese VQA (dataset) | First general Vietnamese VQA benchmark; questions/answers in Vietnamese over natural images. | Dataset construction (Vietnamese VQA pairs over COCO-style images) + baseline VQA models. | **ViVQA**. Measured from HF mirror `minhnguyent546/ViVQA`: 15,000 QA / 10,328 images (train 11,999 + test 3,001) `✅ source-read (experiments/W02_dataset_stats/vivqa_stats.json)`. Paper-reported #images/#QA + official split = `❌ TODO: verify from source (PDF)`. | Baseline VQA-Acc from paper = `❌ TODO: verify from source` | Primary **general** dataset. Answers are short (≥99% ≤3 words) → EM/VQA-Acc feasible. Diacritics/counting cues drive our question-type heuristic. Evidence for G2. | ✅ sizes (measured); ❌ TODO (paper metrics/official split) |
| L4 | Nguyen et al., 2024, arXiv:2404.10652 (ext. Expert Systems w/ Applications, 2025) | Vietnamese scene-text VQA (dataset) | Vietnamese **scene-text** VQA: answering requires reading text embedded in the image. | Dataset of images with Vietnamese scene text + QA; OCR-augmented baselines; ANLS as the headline metric. | **ViTextVQA**. Measured from HF mirror `nhonhoccode/ViTextVQA`: 45,187 QA / 15,086 images (train 35,159 + test 10,028; **validation split empty in this mirror**) `✅ source-read (experiments/W02_dataset_stats/vitextvqa_stats.json)`. Paper-reported #images/#QA + official split = `❌ TODO: verify from source (PDF)`. | Baseline ANLS from paper = `❌ TODO: verify from source` | Primary **scene-text** dataset; drives OCR (PaddleOCR/VietOCR) and **ANLS**. Long, brand/URL/phone answers with code-switching → central to G2. Evidence dataset. | ✅ sizes (measured); ❌ TODO (paper metrics/official split) |
| L5 | Xiao et al., 2024, CVPR | VLM baseline (unified vision) | A single **unified** vision foundation model handling detection, captioning, grounding, and OCR via a sequence-to-sequence, prompt-driven interface. | Florence-2: DaViT vision encoder + transformer encoder-decoder; trained on large multi-task "FLD-5B" annotations; task specified by text prompt. | (paper) FLD-5B; COCO det/caption, OCR, grounding benchmarks. | EM/VQA-Acc = `❌ TODO: verify from source` | Very small (0.23B/0.77B) → highest PEFT/GPU feasibility; native OCR task useful for ViTextVQA. Candidate baseline / **backup**. | ❌ TODO (metrics); ⚠️ method from register/general |
| L6 | Bai et al., 2025, Technical Report (arXiv:2502.13923) | VLM baseline (open, multilingual) | Open multilingual VLM with strong native OCR/scene-text and dynamic resolution. | Qwen2.5-VL: ViT vision encoder + merger + Qwen2.5 LLM; dynamic resolution (`min_pixels/max_pixels`); instruction-tuned; 3B/7B/72B. | (paper) DocVQA, TextVQA, OCRBench, MMMU, etc. | EM/VQA-Acc/ANLS = `❌ TODO: verify from source` | **Proposed primary baseline** (3B) — see `docs/W03/Baseline_Decision.md` (chưa CHỐT). LoRA target = LLM linear layers + optional merger. | ❌ TODO (metrics); ⚠️ method from register/general |
| L7 | Nguyen et al., 2024, Computers & Electrical Engineering, vol. 119 (Vietnamese VQA with Transformer & Convolutional Integration) | Vietnamese VQA (method) | Combine transformer and convolutional features for Vietnamese VQA. | Hybrid Transformer + CNN visual/text integration for Vietnamese VQA. | (paper) Vietnamese VQA benchmark(s) incl. ViVQA. | EM/VQA-Acc = `❌ TODO: verify from source` | Non-VLM Vietnamese VQA point of comparison; positions our PEFT-VLM approach; possible related-work contrast. | ❌ TODO (metrics); ⚠️ citation from register |
| L8 | Chen et al., 2025, CVPR (Florence-VL) | VLM method | Enrich an LLM-based VLM with **Florence-2** generative vision features (depth-breadth fusion) for better perception. | Florence-VL: Florence-2 features + depth-breadth fusion into an LLM decoder. | (paper) standard VLM/VQA + OCR/chart benchmarks. | EM/VQA-Acc = `❌ TODO: verify from source` | Shows unified-vision features (Florence-2) help VQA/OCR → supports a Florence-2-backed ablation and OCR-aware design. | ❌ TODO (metrics); ⚠️ citation from register |
| L9 | Li et al., 2023, EMNLP (POPE) | Hallucination (evaluation) | Measure **object hallucination** via balanced yes/no polling about object existence, avoiding caption-parsing bias. | POPE: sample present/absent objects (random/popular/adversarial) and ask existence questions; report Accuracy/Precision/Recall/**F1**. | (paper) MSCOCO (+ A-OKVQA, GQA) object sets. | POPE F1 = `❌ TODO: verify from source` | Blueprint for **POPE-vi** (Vietnamese yes/no existence probing) in W12 hallucination analysis (SV4). Evidence for G4. | ❌ TODO (metrics); ⚠️ method from register/general |
| L10 | Zhang et al., 2024, IEEE TPAMI (Vision-Language Models for Vision Tasks: A Survey) | Survey | Systematize VLM pretraining objectives, transfer (incl. PEFT), and evaluation across vision tasks. | Survey/taxonomy of datasets, architectures, pretraining objectives, transfer/PEFT, benchmarks. | (survey) aggregates many datasets/benchmarks. | N/A (survey) — specific claims = `❌ TODO: verify from source` | Framing for related work; cite for English-centric pretraining (G1) and for positioning PEFT/eval choices. | ❌ TODO (specific claims); ⚠️ citation from register |

## What-we-can-use rationale (master §14) — expanded

- **L1 BLIP-2 / L5 Florence-2 / L6 Qwen2.5-VL** are the three candidate baselines. Per the master plan we reproduce **exactly one** (proposed: **Qwen2.5-VL-3B-Instruct**, not CHỐT — `docs/W03/Baseline_Decision.md`) and may use a second only as an ablation *reference* backbone.
- **L2 LLaVA-1.5** supplies the strongest in-repo-documented evidence that **LoRA ≈ full fine-tuning** on 7B/13B VLMs, directly justifying the PEFT-only constraint, plus the response-format-prompt and resolution levers.
- **L3 ViVQA / L4 ViTextVQA** are the two datasets we analyze in D2–D4 (general vs. scene-text). ViTextVQA anchors the **ANLS** metric and the OCR pipeline.
- **L7 / L8** are recent-method comparators (non-VLM Vietnamese VQA; Florence-2-enriched VLM).
- **L9 POPE** defines the hallucination protocol we localize to Vietnamese (POPE-vi).
- **L10 Survey** frames the gaps (English-centric pretraining, evaluation).

## Metric skeleton (W03) — khung, không suy số từ tiêu đề

Cột chuẩn: **Paper | Dataset | Metric | Số báo cáo | Nguồn (trang/section) | Verified?**

Quy tắc: ô số trống = `[TODO-human: đọc PDF điền + ghi trang]`. Số kéo từ Reading Notes → tag `from reading notes — verify page`. **Không** suy từ title/abstract.

| Paper | Dataset | Metric | Số báo cáo | Nguồn (trang/section) | Verified? |
|-------|---------|--------|------------|----------------------|-----------|
| L1 BLIP-2 (Li et al., 2023) | VQAv2 | VQA-Acc (zero-shot) | [TODO-human: đọc PDF điền + ghi trang] | [TODO-human: trang/section] | ❌ |
| L1 BLIP-2 | OK-VQA | VQA-Acc | [TODO-human: đọc PDF điền + ghi trang] | [TODO-human] | ❌ |
| L1 BLIP-2 | GQA | VQA-Acc | [TODO-human: đọc PDF điền + ghi trang] | [TODO-human] | ❌ |
| L2 LLaVA-1.5-13B LoRA (Liu et al., 2024) | VQAv2 | VQA-Acc | 80.0 | `Reading_Notes/SV3_Paper_Review.md` §P4 bảng Model Zoo (không ghi số trang PDF) | ⚠️ from reading notes — verify page |
| L2 LLaVA-1.5-13B LoRA | GQA | VQA-Acc | 63.3 | SV3 §P4 bảng | ⚠️ from reading notes — verify page |
| L2 LLaVA-1.5-13B LoRA | TextVQA | VQA-Acc | 60.2 | SV3 §P4 bảng | ⚠️ from reading notes — verify page |
| L2 LLaVA-1.5-13B LoRA | POPE | F1 | 86.7 | SV3 §P4 bảng | ⚠️ from reading notes — verify page |
| L2 LLaVA-1.5-13B LoRA | MME | score | 1541.7 | SV3 §P4 bảng | ⚠️ from reading notes — verify page |
| L2 LLaVA-1.5-13B LoRA | MMBench | Acc | 68.5 | SV3 §P4 bảng | ⚠️ from reading notes — verify page |
| L2 LLaVA-1.5-13B full-FT | VQAv2 | VQA-Acc | 80.0 | SV3 §P4 bảng | ⚠️ from reading notes — verify page |
| L2 LLaVA-1.5-13B full-FT | TextVQA | VQA-Acc | 61.3 | SV3 §P4 bảng | ⚠️ from reading notes — verify page |
| L2 LLaVA-1.5-13B full-FT | POPE | F1 | 85.9 | SV3 §P4 bảng | ⚠️ from reading notes — verify page |
| L2 LLaVA-1.5-13B full-FT | GQA / MME / MMBench | — | GQA 63.3; MME 1531.3; MMBench 67.7 (SV3 bảng full-FT — **khác** một số ô từng ghi ở hàng L2 cũ) | SV3 §P4 bảng | ⚠️ from reading notes — verify page · ⚠️REVIEW nếu lệch bản Model Zoo |
| L3 ViVQA (Tran et al., 2021) | ViVQA | #QA / #images (paper) | [TODO-human: đọc PDF điền + ghi trang] | [TODO-human] | ❌ paper; ✅ measured 15,000 / 10,328 (JSON, seed 42) |
| L3 ViVQA | ViVQA | baseline VQA-Acc | [TODO-human: đọc PDF điền + ghi trang] | [TODO-human] | ❌ |
| L4 ViTextVQA (Nguyen et al., 2024) | ViTextVQA | #QA / #images (paper) | [TODO-human: đọc PDF điền + ghi trang] | [TODO-human] | ❌ paper; ✅ measured 45,187 / 15,086 (JSON, seed 42) |
| L4 ViTextVQA | ViTextVQA | ANLS (baseline paper) | [TODO-human: đọc PDF điền + ghi trang] | [TODO-human] | ❌ |
| L5 Florence-2 (Xiao et al., 2024) | COCO / OCR / TextVQA (paper) | task metrics | [TODO-human: đọc PDF điền + ghi trang] | [TODO-human] | ❌ |
| L6 Qwen2.5-VL-3B (Bai et al., 2025) | DocVQA / TextVQA / OCRBench | ANLS hoặc Acc (paper) | [TODO-human: đọc PDF điền + ghi trang] | [TODO-human] | ❌ |
| L6 Qwen2.5-VL-7B | DocVQA / TextVQA / OCRBench | ANLS hoặc Acc (paper) | [TODO-human: đọc PDF điền + ghi trang] | [TODO-human] | ❌ |
| L7 Nguyen et al., CEE 2024 | ViVQA (paper) | VQA-Acc | [TODO-human: đọc PDF điền + ghi trang] | [TODO-human] | ❌ |
| L8 Florence-VL (Chen et al., 2025) | VQA / OCR benchmarks (paper) | Acc / ANLS | [TODO-human: đọc PDF điền + ghi trang] | [TODO-human] | ❌ |
| L9 POPE (Li et al., 2023) | MSCOCO (random / popular / adversarial) | F1 | [TODO-human: đọc PDF điền + ghi trang] | [TODO-human] | ❌ |
| L10 Zhang et al., TPAMI 2024 | (survey) | N/A — claim English-centric / PEFT | [TODO-human: trích claim + trang] | [TODO-human] | ❌ |

Reading Notes **không** có số cho L1, L3–L10 (template `_…_`). W01 `SV1_Paper_Review.md` có BLIP-2 zero-shot VQAv2 **≈ 65.0** — **không** kéo vào bảng này vì không nằm `Reading_Notes/` và đã đánh ≈; ⚠️REVIEW nếu leader muốn port sau khi đối trang PDF.

PEFT ngoài 10 ref (LoRA / QLoRA / VL-Adapter): số liệu trong `docs/02_references/Reading_Notes/SV3_Paper_Review.md` §P1–P3 — **cố ý không** thành hàng Lx.

## TODO — papers still unread (primary PDFs NOT in repo)

- [ ] **L1 BLIP-2** — VQAv2/OK-VQA/GQA + page cites.
- [ ] **L3 ViVQA** — paper-reported #images/#QA, official split, baseline VQA-Acc (vs measured 15,000 QA).
- [ ] **L4 ViTextVQA** — paper size/split, baseline ANLS (vs measured 45,187 QA; val missing in HF mirror).
- [ ] **L5 Florence-2** — task-wise results + page cites.
- [ ] **L6 Qwen2.5-VL** — DocVQA/TextVQA/OCRBench for 3B/7B.
- [ ] **L7 VN-VQA Transformer+Conv** — method + results.
- [ ] **L8 Florence-VL** — fusion details + results.
- [ ] **L9 POPE** — F1 random/popular/adversarial + page cites.
- [ ] **L10 VLM Survey** — English-centric / PEFT claims + page cites.
- [x] **L2 LLaVA-1.5** — numbers *secondarily* from `docs/02_references/Reading_Notes/SV3_Paper_Review.md` §P4; still verify primary PDF / Model Zoo.

> `docs/02_references/Reading_Notes/SV3_Paper_Review.md` also contains fully-worked reviews of **LoRA (Hu et al., 2022)**, **QLoRA (Dettmers et al., 2023)**, and **VL-Adapter (Sung et al., 2022)**. Core PEFT techniques of record but **not** among the 10 project references — excluded from L1–L10.

## Ma trận Gap × Paper (W03 remap — nhãn Lx theo TÊN PAPER)

### Map cột weekly #1–#10 → Lx (theo tên, không theo số)

| Weekly # | Paper (weekly) | Canonical Lx? |
|----------|----------------|---------------|
| #1 | BLIP-2 (Li et al.) | **L1** |
| #2 | Florence-2 (Xiao et al.) | **L5** (không phải L2) |
| #3 | Qwen2.5-VL (Bai et al.) | **L6** |
| #4 | ViVQA (Tran et al.) | **L3** |
| #5 | ViTextVQA (Nguyen et al.) | **L4** |
| #6 | LoRA (Hu et al.) | **không** trong L1–L10 |
| #7 | QLoRA (Dettmers et al.) | **không** trong L1–L10 |
| #8 | POPE (Li et al.) | **L9** |
| #9 | VL-Adapter (Sung et al.) | **không** trong L1–L10 |
| #10 | LLaVA-1.5 (Liu et al.) | **L2** |

Weekly gốc (archive, không dùng số nữa):

| Gap | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 |
|-----|----|----|----|----|----|----|----|----|----|-----|
| G1 | | | | | | ✓ | ✓ | | ✓ | ✓ |
| G2 | | | | | | | | | ✓ | ✓ |
| G3 | | | | | | ✓ | ✓ | | ✓ | ✓ |
| G4 | | | | | | | | | | ✓ |

### Ma trận canonical G1–G4 × L1–L10

**✓** = remap được từ weekly (cùng *tên paper*). Ô trống = weekly có paper đó và **không** tick. **⚠️REVIEW** = không truy được cột weekly nào tương ứng.

| Gap | L1 BLIP-2 | L2 LLaVA-1.5 | L3 ViVQA | L4 ViTextVQA | L5 Florence-2 | L6 Qwen2.5-VL | L7 VN-VQA CNN+Tr. | L8 Florence-VL | L9 POPE | L10 Survey |
|-----|-----------|--------------|----------|--------------|---------------|---------------|-------------------|----------------|---------|------------|
| G1 English-centric pretraining | | ✓ | | | | | ⚠️REVIEW | ⚠️REVIEW | | ⚠️REVIEW |
| G2 Vietnamese VQA yếu, scene-text | | ✓ | | | | | ⚠️REVIEW | ⚠️REVIEW | | ⚠️REVIEW |
| G3 PEFT cho VLM tiếng Việt chưa có | | ✓ | | | | | ⚠️REVIEW | ⚠️REVIEW | | ⚠️REVIEW |
| G4 Hallucination | | ✓ | | | | | ⚠️REVIEW | ⚠️REVIEW | | ⚠️REVIEW |

**Supporting (ngoài L1–L10)** — weekly ✓, giữ evidence, không gán Lx:

| Gap | LoRA (Hu 2022) | QLoRA (Dettmers 2023) | VL-Adapter (Sung 2022) |
|-----|----------------|-----------------------|------------------------|
| G1 | ✓ (weekly #6) | ✓ (weekly #7) | ✓ (weekly #9) |
| G2 | | | ✓ (weekly #9) |
| G3 | ✓ | ✓ | ✓ |
| G4 | | | |

### ⚠️REVIEW — leader xác nhận

1. **L7, L8, L10:** không có cột weekly. Không tự tick. `docs/W02/Research_Gap.md` đã *dẫn* L10 (và L1, L6) cho G1; L4 cho G2; L9 cho G4 — đó là narrative evidence, **không** phải ✓ weekly. Có cộng ✓ vào ma trận không? **Chờ leader.**
2. **L9 POPE × G4:** weekly #8 **trống** mọi gap (kể cả G4). Remap trung thực = để trống. Research_Gap lại gọi L9 là protocol G4. Thêm ✓ G4-L9? **Chờ leader** — không tự quyết.
3. **L3 / L4 × G2:** weekly #4/#5 trống. Dataset tiếng Việt hiển nhiên liên quan G2 nhưng weekly không tick. Không tự tick.
4. **L1 / L6 × G1:** weekly trống; Research_Gap ghi `❌ TODO: cite`. Không tự tick.

## Changelog

| Ngày | Thay đổi |
|------|----------|
| 2026-09-17 (W03) | Remap Gap×Paper sang L1–L10 theo tên paper. Thêm khung metric ≥10 paper. Sửa path review SV3 → `docs/02_references/Reading_Notes/SV3_Paper_Review.md`. Ô không truy được → ⚠️REVIEW. |
| 2026-09-17 (literature consolidation) | File trở thành bản sống tại `docs/02_references/`. |
| 2026-09-16 | Append ma trận weekly #1–#10 (lỗi đánh số so với L1–L10). |
| 2026-09-14 | Hàng L1–L10 + sizes đo JSON seed 42. |
