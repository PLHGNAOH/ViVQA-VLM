# WEEK 2 STATUS — ViVQA-VLM (self-audit)

> Reviewer reads this first. Scope: Literature Matrix + Dataset Analysis (master §9).
> Run of record: `python src/data/analyze_dataset.py` on **2026-09-14**, **seed 42**,
> Windows 11 / i5-12500H / 15.7 GB RAM / Python 3.12.3 (CPU-only). No fabricated numbers.

## Definition-of-Done audit

| DoD item | Status | Evidence (file path / line) |
|----------|:------:|-----------------------------|
| D1 Literature_Matrix: 10 rows, all columns present | ✅ | `docs/W02/Literature_Matrix.md` — rows **L1–L10**, 9 columns per master spec |
| D1: every numeric cell tagged | ✅ | Verification legend + per-cell `✅/⚠️/❌` tags; e.g. L2 `⚠️ secondary`, L3/L4 sizes `✅ source-read` |
| D1: no invented numbers | ✅ | All paper metrics are `❌ TODO` (no PDFs) except L2 LLaVA (`⚠️ secondary`, from `docs/literature/SV3_Paper_Review.md §P4`) |
| D1: unread-paper TODO list present | ✅ | `docs/W02/Literature_Matrix.md` → "TODO — papers still unread" (9 PDFs) |
| D2 analyze_dataset.py: runnable | ✅ | `src/data/analyze_dataset.py`; exit 0, SUMMARY all `OK` (see run below) |
| D2: NFC normalization | ✅ | `analyze_dataset.py::nfc` (`unicodedata.normalize("NFC", …)`) |
| D2: fixed seed = 42 (logged) | ✅ | `SEED=42`; written to every JSON `"seed": 42` |
| D2: VN-cue question-type classifier (editable dict) | ✅ | `QUESTION_TYPE_RULES` + `classify_question`; disclaimer printed |
| D2: PENDING_RUN fallback (no fake output) | ✅ | `pending_stats()`; verified live (earlier ViTextVQA run wrote `PENDING_RUN` with exact errors before fix) |
| D3: REAL stats for BOTH ViVQA & ViTextVQA (JSON) | ✅ | `experiments/W02_dataset_stats/vivqa_stats.json` (`status:OK`, 15,000 QA), `vitextvqa_stats.json` (`status:OK`, 45,187 QA) |
| D3: per-type samples exist | ✅ | `experiments/W02_dataset_stats/samples/` — 18 files, 5 samples × 6 types × 3 datasets |
| D4 Dataset_Analysis: overview + distribution + VN challenges (real samples) + preprocessing | ✅ | `docs/W02/Dataset_Analysis.md` §1–§4; each challenge cites a real sample file |
| D4: numbers sourced from JSON (not hardcoded), provenance header | ✅ | `docs/W02/Dataset_Analysis.md` top provenance block; values match JSON |
| D5 Research_Gap: G1–G4 evidence-linked + feasibility | ✅ | `docs/W02/Research_Gap.md` — G1–G4 cite L1/L2/L4/L9/L10; Feasible? Yes/Partial |
| Gate: no fabricated metric/size anywhere | ✅ | Unverified values left as `❌ TODO`; dataset sizes are *measured* and cite JSON |
| Gate: verification tags present | ✅ | Legend used across D1/D4/D5 |
| Gate: PEFT-only respected; metrics = EM/VQA-Acc/ANLS | ✅ | No full-FT proposed; metrics stated in D4 §4, D5, scaffolds |
| Gate: only allowed paths touched | ✅ | Created only under `docs/W02/`, `src/data/`, `experiments/W02_dataset_stats/`, `WEEK2_STATUS.md` |
| D6 WEEK2_STATUS present with per-item evidence + honest % | ✅ | this file |
| Optional W3-prep scaffolds | ✅ | `docs/W02/{Research_Questions,Hypotheses,Candidate_Methods}.md` (marked W3-PREP) |

## Honest completion: **90%**

Every DoD checkbox passes and **D3 has REAL stats for BOTH target datasets** (no PENDING_RUN remaining), so
Week 2 is *substantially* complete. It is **not a true 100%** because of the research-debt items below.

## Remaining ❌ TODO
1. **Reference-paper PDFs (0/10 in repo).** All paper-reported metrics in the Literature Matrix are `❌ TODO`
   (only L2 LLaVA is `⚠️ secondary` from an in-repo review). Must obtain & read the 9 PDFs to fill metric cells
   with page cites — see `docs/W02/Literature_Matrix.md` TODO list.
2. **Dataset provenance is community-mirror, not official.** ViVQA (`minhnguyent546/ViVQA`) and ViTextVQA
   (`nhonhoccode/ViTextVQA`) were loaded from HuggingFace mirrors. Paper-reported #images/#QA, official splits,
   and **license** are still `❌ TODO: verify from source`.
3. **Validation splits missing.** ViVQA mirror has **no val split**; ViTextVQA mirror's **validation split is empty**.
   A deterministic (seed 42) val split must be carved or the official split obtained (D4 §4.4).
4. **Question-type labels are heuristic.** Rule-based only; ~30–41% land in `Other`, with documented false
   positives/negatives (D4 §3.4). Needs a human-reviewed annotation pass (planned W12).
5. **Images not downloaded** (annotations only) — image bring-up is a W4 task.

## Blockers (make Week 2 partial, not 100%)
- **B1 (Medium):** No primary PDFs → literature metrics unverifiable this week. *Mitigation:* fetch PDFs into
  `docs/02_references/` and re-run extraction.
- **B2 (Medium):** Official dataset releases + licenses unconfirmed; mirrors lack a usable validation split.
  *Mitigation:* confirm official ViVQA/ViTextVQA sources; commit fixed splits under `data/splits/`.

## Reproduce
```
python src/data/analyze_dataset.py
```
(Optional flags: `--data_root data`, `--top_k 30`, `--only vivqa vitextvqa`, `--vivqa_hf_id <id>`.)
Outputs: `experiments/W02_dataset_stats/{vivqa,vitextvqa,vivqa_x}_stats.json` + `.../samples/*.json`.
