# Dataset Analysis — ViVQA & ViTextVQA (Week 2)

> **PROVENANCE (hand-edit forbidden for numbers):** all quantitative values below come from
> `python src/data/analyze_dataset.py` — run **2026-09-14**, **seed = 42**,
> hardware **Windows 11 / Intel Core i5-12500H / 15.7 GB RAM (CPU-only; no GPU needed for descriptive stats)**,
> **Python 3.12.3**, `datasets` (HuggingFace). Source JSON: `experiments/W02_dataset_stats/{vivqa,vitextvqa,vivqa_x}_stats.json`.
> If you change a number here, re-run the script instead — numbers must match the JSON.
>
> **Data sources actually loaded (real):**
> - ViVQA ← HuggingFace `minhnguyent546/ViVQA` (mirror of Tran et al., PACLIC 2021).
> - ViTextVQA ← HuggingFace `nhonhoccode/ViTextVQA` (mirror of Nguyen et al., 2024).
> - ViVQA-X ← local `data/ViVQA-X/data/final/*.json` — a **RELATED but DIFFERENT** dataset (translated VQA-X with
>   explanations, COCO images). Reported here as supplementary context, **never as a substitute for ViVQA**.
>
> ⚠️ **Heuristic disclaimer:** the question-type distribution comes from a rule-based Vietnamese cue-word
> classifier (`QUESTION_TYPE_RULES` in the script). It is **exploratory only** and **not** a validated annotation.
> Concrete false positives are documented in §3.

---

## 1. Overview (from stats JSON)

| Dataset | Source (loaded) | Type | #Images | #QA | Split sizes | Unique answers | Language | License |
|---------|-----------------|------|--------:|----:|-------------|---------------:|----------|---------|
| **ViVQA** | HF `minhnguyent546/ViVQA` | General VQA | 10,328 | 15,000 | train 11,999 / test 3,001 (no val in mirror) | 351 | vi | `❌ TODO: verify from source (paper/repo)` |
| **ViTextVQA** | HF `nhonhoccode/ViTextVQA` | Scene-text VQA | 15,086 | 45,187 | train 35,159 / test 10,028 (**validation empty in mirror**) | 29,202 | vi (+ code-switch) | `❌ TODO: verify from source (paper/repo)` |
| ViVQA-X *(related, not ViVQA)* | local `data/ViVQA-X` | VQA + explanations | 28,180 | 32,886 | train 29,459 / val 1,459 / test 1,968 | 1,504 | vi | `❌ TODO: verify` |

**Question / answer length (whitespace tokens):**

| Dataset | Q mean | Q median | Q p90 | Q max | A mean | A median | A p90 | A max | % answer 1 word | % answer ≤3 words |
|---------|-------:|---------:|------:|------:|-------:|---------:|------:|------:|----------------:|------------------:|
| ViVQA | 9.514 | 9 | 15 | 26 | 1.78 | 2 | 3 | 4 | 34.05 | 99.83 |
| ViTextVQA | 9.599 | 9 | 13 | 31 | 4.194 | 3 | 9 | 52 | 15.91 | 57.98 |
| ViVQA-X | 7.072 | 7 | 10 | 27 | 1.365 | 1 | 2 | 6 | 67.89 | 99.32 |

**Reading of the numbers (exploratory):**
- ViVQA answers are **very short** (99.83% ≤ 3 words, max 4) → **Exact Match / VQA-Accuracy** are the natural metrics.
- ViTextVQA answers are **much longer and more variable** (mean 4.19 tokens, max 52; 29,202 unique answers) → EM will be harsh; **ANLS** is the appropriate headline metric, matching the master constraint.
- The large ViTextVQA answer vocabulary (29,202 unique vs. 351 for ViVQA) reflects open scene-text strings (brands, URLs, phone numbers, addresses).

**Top answers (from JSON `top_answers`, top 5 shown; full 30 in the JSON):**
- **ViVQA:** `hai` (671), `ba` (612), `bốn` (427), `màu trắng` (395), `màu đỏ` (376) → counting + colors dominate.
- **ViTextVQA:** `vinmart` (184), `chợ` (132), `kenh14 . vn` (127), `hà nội` (119), `việt nam` (91) → brands, URLs, place names (scene text).
- **ViVQA-X:** `không` (7,645), `có` (6,171), `đúng` (4,297) → strongly yes/no-skewed.

---

## 2. Question-type distribution (heuristic — exploratory only)

Counts and percentages from `question_type_distribution` in each stats JSON. Method:
`rule_based_vi_cue_words`. **Not a validated annotation.**

| Type | ViVQA count | ViVQA % | ViTextVQA count | ViTextVQA % | ViVQA-X count | ViVQA-X % |
|------|------------:|--------:|----------------:|------------:|--------------:|----------:|
| Yes/No | 58 | 0.39 | 48 | 0.11 | 17,159 | 52.18 |
| Counting | 2,090 | 13.93 | 2,642 | 5.85 | 18 | 0.05 |
| Object-recognition | 8,092 | 53.95 | 12,640 | 27.97 | 2,966 | 9.02 |
| Text-reading | 244 | 1.63 | 9,191 | 20.34 | 119 | 0.36 |
| Reasoning | 63 | 0.42 | 2,260 | 5.00 | 2,230 | 6.78 |
| Other | 4,453 | 29.69 | 18,406 | 40.73 | 10,394 | 31.61 |

**Reading (exploratory):**
- **Text-reading** is ~12× more frequent in ViTextVQA (20.34%) than ViVQA (1.63%) — consistent with ViTextVQA being a scene-text dataset. Even so, 20% is almost certainly an **under-count** (see §3 false-negatives), because many scene-text questions phrase themselves as "…tên gì?" / "…là nơi nào?" and fall into `Object-recognition`/`Other`.
- ViVQA is dominated by **Object-recognition + Counting** — general natural-image VQA, as expected.
- ViVQA-X is dominated by **Yes/No** (52%) — a different task profile, another reason not to treat it as ViVQA.
- The large **Other** bucket (29–41%) is the honest cost of a shallow cue-word ruleset; it is the priority target for the manual annotation pass in W12.

---

## 3. Vietnamese-specific challenges (each with a REAL sample from experiments/.../samples/)

### 3.1 Diacritics & Unicode (NFC) — scene text with tone marks
Real sample — `experiments/W02_dataset_stats/samples/vitextvqa__Text-reading.json` (image `3212.jpg`):
> Q: *"địa chỉ của trung tâm này là gì ?"* → A: *"đc : 594 nguyễn trãi - tp . bắc ninh"*

The answer packs tone marks (`nguyễn trãi`, `bắc ninh`) plus abbreviations/punctuation. If prediction and gold differ only by NFC vs. NFD encoding of `ễ`/`ắ`, a naïve string EM fails despite being semantically correct → **NFC normalization is mandatory** before scoring, and **ANLS** (character-level) is more forgiving here than EM.

### 3.2 Code-switching (Vietnamese ↔ English/brands/URLs)
Real samples — `vitextvqa__Text-reading.json` / `vitextvqa__Other.json` and `top_answers`:
> A: *"kenh14 . vn"*, *"vinmart"*, *"ssd adata legend 710 512gb"* (Q: *"sản phẩm nào giảm 40 % ?"*, image `14794.jpg`)

Answers mix Latin brand names, URLs, model codes, and numbers with Vietnamese. Tokenizer coverage and answer normalization (spacing around `.` and `%`, casing) directly affect EM/ANLS. The mirror already inserts spaces around punctuation (`kenh14 . vn`, `50 %`), which our normalization must handle consistently.

### 3.3 Scene-text / OCR reading
Real sample — `vitextvqa__Text-reading.json` (image `6778.jpg`):
> Q: *"số điện thoại của cửa hàng phía sau người phụ nữ áo đỏ là gì ?"* → A: *"02422 119 111"*

Answering requires **reading text in the image** (a phone number), often on small/angled/blurred signage → motivates an OCR stage (PaddleOCR vs. VietOCR) and OCR-enhanced prompting.

### 3.4 Heuristic classifier limitations (honest, from real output)
Real sample — `vivqa__Text-reading.json` (image `514607`):
> Q: *"những gì trên bãi biển cát"* → A: *"chiếc ô"* — **mislabeled Text-reading**.

The cue word **`biển`** is polysemous: `biển` = *sea/beach* (`bãi biển`) **and** `biển` = *sign board* (`biển hiệu`). The ruleset over-triggers on beach questions. Symmetrically, ViTextVQA scene-text questions like *"cửa hàng di động bên phải tên gì ?"* (image `8803.jpg`) fall into **Other** (false-negative). These concrete errors are why the distribution in §2 is flagged exploratory and must be human-reviewed before any thesis claim.

---

## 4. Preprocessing plan

1. **Unicode normalization — NFC (mandatory).** All questions/answers normalized with `unicodedata.normalize("NFC", …)` at load time (already implemented in `analyze_dataset.py::nfc` and in `src/eval/metrics.py::normalize_vi`). Never store/compare NFD.
2. **Answer normalization for scoring.** Lowercase, trim, collapse whitespace, normalize punctuation spacing (`" . " → "."`, `" % " → "%"`), and normalize number/word forms where appropriate (e.g. `2` ↔ `hai`) — **keep Vietnamese tone marks** (do NOT strip diacritics; `mèo` ≠ `meo`). Use EM/VQA-Acc on ViVQA, **ANLS (threshold 0.5)** on ViTextVQA.
3. **OCR (scene text).** Compare **PaddleOCR (`lang='vi'`)** vs. **VietOCR** on a sample; keep the stronger engine, other as backup; report CER/WER; store OCR as `{text, bbox, confidence}` for prompt injection (`Văn bản trong ảnh: "{ocr_text}"`).
4. **Fixed split policy (seed = 42).** Use each dataset's official split when confirmed from the paper/repo. **Current blockers to record:** the ViVQA mirror ships **no validation split** and the ViTextVQA mirror ships an **empty validation split** — a val split must be carved deterministically (seed 42) from train, or the official split obtained. Commit the resulting split files under `data/splits/` so runs are reproducible.
5. **Unified schema.** Convert both datasets to the common record used by `src/data/vivqa_dataset.py` (`question_id, question, answers[], image_path, question_type`) so eval/train code is dataset-agnostic.

> Images themselves (COCO for ViVQA/ViVQA-X; scene images for ViTextVQA) are **not** downloaded yet — only annotations. `image_path` in the sample files points to the expected filenames (e.g. `COCO_train2014_000000013920.jpg`, `3212.jpg`); fetching image binaries is a W4 data-bring-up task.
