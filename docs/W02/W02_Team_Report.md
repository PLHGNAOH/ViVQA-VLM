# W02 — Team Weekly Report

Nguồn: `docs/W02/WEEK2_STATUS.md` (self-audit 2026-09-14) + khung `docs/templates/Weekly_Report_Template.md`. Không thêm số liệu mới. Ô không có trong nguồn → `TODO (W3)`.

## Metadata

| Trường | Nội dung |
|--------|----------|
| **Tuần** | W02 — Literature matrix + Dataset analysis |
| **Nhóm** | ViVQA-VLM (4 SV) |
| **GVHD** | Assoc.Prof. Đặng Ngọc Minh Đức |
| **Ngày** | 2026-09-14 (run of record trong WEEK2_STATUS; ngày nộp GVHD = TODO (W3)) |

## Mục tiêu

Literature Matrix + Dataset Analysis (master §9). Run of record: `python src/data/analyze_dataset.py` on **2026-09-14**, **seed 42**, Windows 11 / i5-12500H / 15.7 GB RAM / Python 3.12.3 (CPU-only). No fabricated numbers.

## Đã làm (per-SV)

WEEK2_STATUS không gán việc theo từng SV. Không suy diễn owner.

| SV | Vai trò | Đã làm | Evidence |
|----|---------|--------|----------|
| SV1 | VLM / Baseline | TODO (W3) | TODO (W3) |
| SV2 | Dataset / OCR | TODO (W3) | TODO (W3) |
| SV3 | PEFT / Training | TODO (W3) | TODO (W3) |
| SV4 | Evaluation / Deployment | TODO (W3) | TODO (W3) |

**Team DoD (nguyên từ WEEK2_STATUS):** honest completion **90%**. Evidence tập trung `docs/W02/{Literature_Matrix,Dataset_Analysis,Research_Gap}.md`, `src/data/analyze_dataset.py`, `experiments/W02_dataset_stats/`.

| DoD item | Status | Evidence |
|----------|:------:|----------|
| D1 Literature_Matrix: 10 rows, all columns present | ✅ | `docs/W02/Literature_Matrix.md` — L1–L10 |
| D1: every numeric cell tagged | ✅ | `✅/⚠️/❌` tags |
| D1: no invented numbers | ✅ | paper metrics `❌ TODO` except L2 LLaVA `⚠️ secondary` |
| D2 analyze_dataset.py runnable | ✅ | `src/data/analyze_dataset.py`; exit 0 |
| D3 REAL stats ViVQA & ViTextVQA | ✅ | `experiments/W02_dataset_stats/vivqa_stats.json` (15,000 QA), `vitextvqa_stats.json` (45,187 QA) |
| D3 per-type samples | ✅ | `experiments/W02_dataset_stats/samples/` — 18 files |
| D4 Dataset_Analysis | ✅ | `docs/W02/Dataset_Analysis.md` §1–§4 |
| D5 Research_Gap G1–G4 | ✅ | `docs/W02/Research_Gap.md` |
| D6 WEEK2_STATUS | ✅ | `docs/W02/WEEK2_STATUS.md` |

## Kết quả / số liệu

| Mục | Giá trị |
|-----|---------|
| Dataset split + seed | seed **42**. ViVQA 15,000 QA; ViTextVQA 45,187 QA (WEEK2_STATUS D3). Split chi tiết / val missing → xem Dataset_Analysis; official split = TODO (W3) |
| Model / baseline | TODO (W3) — W2 chưa reproduce baseline (WEEK2_STATUS scope: literature + dataset) |
| Method | descriptive stats (`analyze_dataset.py`); chưa LoRA/QLoRA/OCR-run |
| **Exact Match (EM)** | TODO (W3) — chưa chạy model eval |
| **VQA Accuracy** | TODO (W3) — chưa chạy model eval |
| **ANLS** | TODO (W3) — chưa chạy model eval |
| BLEU / CIDEr (phụ) | TODO (W3) |
| Train time | không train trong W2 |
| Inference time | không inference VLM trong W2 |
| Hardware | Windows 11 / Intel Core i5-12500H / 15.7 GB RAM / Python 3.12.3 (CPU-only) |

## Blocker

| Blocker | Impact | Owner | Next step |
|---------|--------|-------|-----------|
| B1: No primary PDFs (0/10) → literature metrics unverifiable | Medium | TODO (W3) | fetch PDFs into `docs/02_references/` and re-run extraction |
| B2: Official dataset releases + licenses unconfirmed; mirrors lack usable validation split | Medium | TODO (W3) | confirm official ViVQA/ViTextVQA sources; commit splits under `data/splits/` |

Remaining ❌ TODO (WEEK2_STATUS): unread PDFs; community-mirror provenance; heuristic question-types (~30–41% Other); images not downloaded (W4).

## Kế hoạch tuần sau

WEEK2_STATUS không liệt kê task per-SV cho W3.

- [ ] SV1: TODO (W3)
- [ ] SV2: TODO (W3)
- [ ] SV3: TODO (W3)
- [ ] SV4: TODO (W3)

## Link experiment log

- Template: `docs/templates/Experiment_Log_Template.md`
- Folder: `experiments/W02_dataset_stats/`
- Reproduce: `python src/data/analyze_dataset.py` (optional `--data_root data`, `--top_k 30`, `--only vivqa vitextvqa`)
- Outputs: `experiments/W02_dataset_stats/{vivqa,vitextvqa,vivqa_x}_stats.json` + `samples/*.json`
- Full audit: `docs/W02/WEEK2_STATUS.md`

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.
