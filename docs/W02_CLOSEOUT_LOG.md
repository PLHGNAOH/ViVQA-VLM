# W02 CLOSEOUT LOG

Thời điểm: 2026-09-16. Idempotent. Không hard-delete work files. Không viết RQ/số liệu/baseline mới.

Nguồn weekly giàu: `_archive/docs_weekly_W02_pre_2026-09-14/` (đã archive phiên trước). Backup scaffold: `_archive/W02_scaffold_pre_merge_2026-09-16/`.

## Thao tác

| Timestamp | Thao tác | Từ → Đến | Kết quả |
|-----------|----------|----------|---------|
| 2026-09-16 21:15:59 | COUNT_BEFORE | src/models n=2; src/prompting n=2; docs/literature n=10 | OK |
| 2026-09-16 21:15:59 | MKDIR | `_archive` | SKIP already exists |
| 2026-09-16 21:15:59 | MOVE | `experiments.zip` → `_archive/experiments_2026-09-14.zip` | SKIP source missing |
| 2026-09-16 21:15:59 | MOVE | `docs/W02.zip` → `_archive/docs_W02_2026-09-14.zip` | SKIP source missing |
| 2026-09-16 21:15:59 | MOVE | `src/data.zip` → `_archive/src_data_2026-09-14.zip` | OK |
| 2026-09-16 21:15:59 | MOVE | `WEEK2_STATUS.md` → `docs/W02/WEEK2_STATUS.md` | OK |
| 2026-09-16 21:15:59 | RM_PYCACHE | `src/data/__pycache__` (chỉ *.pyc) | OK |
| 2026-09-16 21:15:59 | WRITE | `.gitignore` (`__pycache__/`, `*.pyc`) | OK created |
| 2026-09-16 21:15:59 | MKDIR | `_archive/W02_scaffold_pre_merge_2026-09-16` | OK |
| 2026-09-16 21:15:59 | COPY_BACKUP | 3 scaffold + Literature_Matrix → folder trên | OK |
| 2026-09-16 21:15:59 | COPY_BACKUP | `README.md` → `_archive/README_pre_2026-09-16.md` | OK |
| 2026-09-16 21:17 | MERGE | weekly RQ/H/Methods → `docs/W02/{Research_Questions,Hypotheses,Candidate_Methods}.md` | OK (weekly substance + W3-PREP note; không nhân đôi bảng) |
| 2026-09-16 21:17 | APPEND | Gap×Paper weekly → cuối `docs/W02/Literature_Matrix.md` | OK (backup trước; L1–L10 không sửa) |
| 2026-09-16 21:17 | WRITE | `docs/W02/W02_Team_Report.md` từ WEEK2_STATUS + template | OK |
| 2026-09-16 21:17 | WRITE | `docs/W02/README.md` index | OK |
| 2026-09-16 21:17 | WRITE | `experiments/W02_dataset_stats/README.md` | OK (chưa có) |
| 2026-09-16 21:17 | PATCH | `README.md` (bỏ `docs/weekly/`, trỏ W01/W02 + archive W03–W15) | OK |
| 2026-09-16 21:18:06 | MOVE | `docs/weekly/W02` → `_archive/docs_weekly_W02_pre_2026-09-14` | SKIP already archived |
| 2026-09-16 21:18:06 | MOVE | `docs/weekly/W01` → `docs/W01` | OK |
| 2026-09-16 21:18:06 | MKDIR | `_archive/docs_weekly_legacy_2026-09-16` | OK |
| 2026-09-16 21:18:06–07 | MOVE | weekly W03–W15 + PROGRESS.md → legacy archive | OK |
| 2026-09-16 21:18:07 | RMDIR_EMPTY | `docs/weekly` | OK |
| 2026-09-16 21:18 | COUNT_AFTER | src/models n=2; src/prompting n=2; docs/literature n=10 | OK khớp trước |
| 2026-09-16 21:18 | SKIP | `docs/W02/{Dataset_Analysis,Research_Gap}.md` | SKIP không đụng (9343 B / 6159 B) |
| 2026-09-16 21:18 | SKIP | `src/models/`, `src/prompting/`, `docs/literature/` | SKIP không đụng |

## Sanity E

| Check | Kết quả |
|-------|---------|
| `docs/W02/` đủ 6 deliverable + Team Report + README + WEEK2_STATUS | OK (9 file) |
| grep H8 / A9 / NF4 | OK — Hypotheses: H8+NF4; Candidate_Methods: A9+H8+NF4; Research_Questions: NF4 (RQ2c) |
| `docs/weekly/` biến mất | OK |
| `docs/W01` tồn tại | OK (11 file) |
| protected counts | OK 2 / 2 / 10 |

## CẦN NGƯỜI KIỂM TRA (W3)

1. **RQ IDs:** weekly **RQ1–RQ4** (+ RQ2a–e) giữ nguyên; scaffold cũ dùng **RQ0–RQ3**. Không remap.
2. **Hypothesis IDs:** weekly **H1–H8** giữ nguyên; scaffold H1–H4 **khác nghĩa** (scaffold H2 = OCR; weekly H2 = 4-bit vs 16-bit). Không trộn.
3. **Ma trận Gap × Paper:** cột `#1`–`#10` theo weekly cũ, **không** khớp L1–L10 (weekly #2 = Florence-2 vs L2 = LLaVA; weekly #6/#7/#9 = LoRA/QLoRA/VL-Adapter ngoài 10 ref). Cần remap tay.
4. **Team Report per-SV / kế hoạch W3 / EM·VQA-Acc·ANLS / baseline:** không có trong WEEK2_STATUS → để `TODO (W3)`.
5. **Baseline chưa chốt:** bảng weekly Trạng thái `_…_`; scaffold ghi Qwen2.5-VL-3B recommended — chỉ giữ note, không chọn hộ.
6. **H9–H11** vẫn stub `_…_` (có sẵn trong weekly).
7. **`experiments.zip` / `docs/W02.zip`:** không còn ở root lúc closeout; không thấy trong `_archive/` phiên này (SKIP). `src/data.zip` đã archive.

Không làm nội dung tuần 3.
