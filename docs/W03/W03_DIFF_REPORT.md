# W03 DIFF REPORT

Ngày: 2026-09-17. Cwd thực tế: `D:\FPT_university\termno9\ViVQA-VLM` (`D:\ViVQA-VLM` không tồn tại). Không sửa `src/**/*.py`, `data/`, `adapters/`, `configs/*.json`, nội dung `_archive/` cũ. Không xóa file (chỉ move). Không train / không tải model.

Backup trước sửa: `_archive/W03_pre_edit_2026-09-17/` (giữ cây con).

## A. File CREATED

| File | Vai trò |
|------|---------|
| `docs/W03/README.md` | index tuần + pack Review 1 |
| `docs/W03/RQ_Hypothesis_Reconciliation.md` | map scaffold → canonical RQ/H |
| `docs/W03/Baseline_Decision.md` | so sánh 3B/7B/Florence-2; đề xuất chưa CHỐT |
| `docs/W03/W03_Team_Report.md` | per-SV, feedback chờ review, kế hoạch W4 |
| `docs/W03/W03_DIFF_REPORT.md` | file này |
| `docs/W03/W03_CLOSEOUT_LOG.md` | log + self-check |
| `docs/04_slides/review1/review1_outline.md` | outline 9 mục pack; không render pptx |
| `docs/00_admin/CONVENTIONS.md` | run-name `W0X_<method>_<model>_seed42` |
| `docs/W01/SV3_Paper_Review.md` | POINTER 1 dòng (sau khi move bản identical) |

## B. File EDITED

| File | Thay đổi |
|------|----------|
| `docs/W02/Research_Questions.md` | Canonical RQ1–RQ4; changelog; bỏ W3-PREP numbering |
| `docs/W02/Hypotheses.md` | Canonical H1–H8; H9–H11 → out-of-scope; changelog |
| `docs/02_references/Literature_Matrix.md` | Remap Gap×Paper L1–L10; khung metric; path Reading_Notes |
| `docs/W02/Literature_Matrix.md` | Pointer: ghi W03 đã remap |
| `docs/W02/README.md` | Index RQ/H canonical |
| `docs/W02/Candidate_Methods.md` | Path SV3; trỏ Baseline_Decision; không chốt Trạng thái |
| `docs/W02/Research_Gap.md` | 4 path `literature` → Reading_Notes |
| `docs/W02/WEEK2_STATUS.md` | Path SV3 + ghi chú W03 (không sửa DoD W2) |
| `docs/W01/W01_Team_Report.md` | Evidence path `docs/weekly/W01/` → `docs/W01/` (+ pointer SV3) |

## C. File MOVED (không xóa)

| Từ | Đến | Lý do |
|----|-----|-------|
| `docs/W01/SV3_Paper_Review.md` | `_archive/W03_identical_SV3_2026-09-17/SV3_Paper_Review.md` | SHA256 identical với `docs/02_references/Reading_Notes/SV3_Paper_Review.md` |

Copy backup (không phải move nguồn): `_archive/W03_pre_edit_2026-09-17/docs/...`

## D. File KHÔNG đụng (cố ý)

- `src/models/` (2 file), `src/prompting/` (2 file), mọi `src/**/*.py`
- `data/` (kể cả không tách val)
- `adapters/`, `configs/*.json`
- `_archive/` nội dung cũ (chỉ **thêm** folder W03)
- `docs/W01/SV1_Paper_Review.md`, `SV2_Paper_Review.md`, `SV4_Paper_Review.md` — **không identical** với Reading_Notes → giữ nguyên, ⚠️REVIEW
- Historical: `PLACEMENT_AUDIT.md`, `docs/W02_DIFF_REPORT.md`, `docs/W02_CLOSEOUT_LOG.md`, `docs/MIGRATION_REPORT.md`, `docs/LITERATURE_CONSOLIDATION_LOG.md` (còn chuỗi path cũ như nhật ký)

## E. Link đã sửa (Phase 3)

| File | Cũ | Mới |
|------|----|-----|
| `docs/02_references/Literature_Matrix.md` | `docs/literature/SV3_Paper_Review.md` (3 chỗ, bản pre-edit) | `docs/02_references/Reading_Notes/SV3_Paper_Review.md` |
| `docs/W02/Candidate_Methods.md` | `docs/literature/SV3_Paper_Review.md` | Reading_Notes |
| `docs/W02/Research_Gap.md` | 4× `docs/literature/SV3_Paper_Review.md` | Reading_Notes |
| `docs/W02/WEEK2_STATUS.md` | `docs/literature/SV3_Paper_Review.md` | Reading_Notes |
| `docs/W01/W01_Team_Report.md` | `docs/weekly/W01/SVx_Paper_Review.md` | `docs/W01/...` / Reading_Notes (SV3) |

`grep docs/literature/` trên **canonical Literature_Matrix** = 0 (self-check).

Còn chuỗi trong nhật ký lịch sử (không sửa): `docs/W02_DIFF_REPORT.md`, `docs/W02_CLOSEOUT_LOG.md`, `docs/LITERATURE_CONSOLIDATION_LOG.md`, `docs/MIGRATION_REPORT.md`, `PLACEMENT_AUDIT.md`.

## F. Mọi ô ⚠️REVIEW

| ID | Nội dung | File |
|----|----------|------|
| R1 | L7, L8, L10 không có cột weekly — không tự tick Gap×Paper | Literature_Matrix |
| R2 | L9 POPE × G4: weekly #8 trống; Research_Gap vẫn dẫn L9 | Literature_Matrix |
| R3 | Có cộng ✓ L3/L4×G2, L1/L6×G1 từ narrative Research_Gap? | Literature_Matrix |
| R4 | Bảng L2 full-FT MME 1531.3 vs hàng L2 cũ (LoRA MME 1541.7) — đối Model Zoo | Literature_Matrix metric |
| R5 | Có port BLIP-2 VQAv2 ≈65.0 từ W01 SV1 (không phải Reading_Notes, đã ≈)? | Literature_Matrix |
| R6 | SV1/SV2/SV4: W01 ≠ Reading_Notes (size 14637/12895/11959 vs 2964/3256/3167) — giữ cả hai | docs/W01 vs Reading_Notes |
| R7 | VRAM 3B 2–2.5 vs 2–3 GB; peak train / P100 **chưa đo** | Baseline_Decision |
| R8 | Licence Qwen 3B vs 7B — chỉ có trong SV1, chưa mở card | Baseline_Decision |
| R9 | GPU thật (Kaggle P100 vs Colab vs trường) | Baseline_Decision |
| R10 | Tỉ lệ val carve 10% — gợi ý, không phải official | W03_Team_Report |
| R11 | ViVQA-X `LICENSE` = MIT **repo code**, không chứng minh licence ViVQA/ViTextVQA | Phase 6 |
| R12 | `experiments.zip` / `docs/W02.zip` không còn root **và** không có trong `_archive/` | Phase 6 |

RQ/H numbering: **không** ⚠️REVIEW (task chốt canonical = weekly).

## G. Mọi [TODO-human]

Mọi ô số paper trong khung metric L1, L3–L10 (+ trang PDF L2): `[TODO-human: đọc PDF điền + ghi trang]`.
Paper-reported size/split ViVQA & ViTextVQA; licence dataset; CHỐT baseline; xác nhận GPU; tách val nếu được phép; render deck.

## H. Dataset split (chỉ kiểm tra)

| Dataset | Trên đĩa `data/` | Official train/val/test? | Licence file |
|---------|------------------|--------------------------|--------------|
| ViVQA | Không (chỉ HF mirror lúc W02 stats) | Mirror: train+test, **không val** (`vivqa_stats.json`) | Không |
| ViTextVQA | Không | Mirror: train+test, **val trống** | Không |
| ViVQA-X | Có `data/ViVQA-X/data/final/{train,val,test}.json` | Có 3 split (JSON W02: 29459 / 1459 / 1968) | `data/ViVQA-X/LICENSE` = MIT (repo); ⚠️REVIEW dataset licence |
| `data/splits/` | Chỉ `.gitkeep` | Chưa freeze split nhóm | — |

**Không tách val.**

## I. Zip loose-end

| Zip | 14/9 (PLACEMENT_AUDIT) | W02 closeout | 2026-09-17 |
|-----|------------------------|--------------|------------|
| `experiments.zip` | root, byte-identical `experiments/` | MOVE → archive **SKIP source missing** | Không ở root, không trong `_archive/` |
| `docs/W02.zip` | snapshot 6 file W02 | SKIP missing | Không thấy |
| `src/data.zip` | — | MOVE OK | `_archive/src_data_2026-09-14.zip` |

Kết luận: hai zip đầu **không còn** trong workspace. Bản sống `experiments/` và `docs/W02/` vẫn đủ; zip là dup lạc chỗ đã mất snapshot. Không phục hồi (không bịa).
