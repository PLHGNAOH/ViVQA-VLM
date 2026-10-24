# W03 CLOSEOUT LOG

Thời điểm: 2026-09-17. Không hard-delete. Không train. Không sửa `src/**/*.py`.

Cwd: `D:\FPT_university\termno9\ViVQA-VLM` (không có `D:\ViVQA-VLM`).

## Thao tác

| Timestamp | Thao tác | Từ → Đến | Kết quả |
|-----------|----------|----------|---------|
| 2026-09-17 | PREFLIGHT | cwd ≠ `D:\ViVQA-VLM`; repo khớp workspace | OK tiếp tục trên path thực tế |
| 2026-09-17 | COUNT_BEFORE | `src/models` n=2; `src/prompting` n=2 | OK |
| 2026-09-17 | MKDIR | `_archive/W03_pre_edit_2026-09-17` | OK |
| 2026-09-17 | COPY_BACKUP | 10 file sắp sửa → archive (giữ cây `docs/...`) | OK |
| 2026-09-17 | MKDIR | `docs/W03/`, `docs/04_slides/review1/` | OK |
| 2026-09-17 | WRITE | Canonical RQ/H; reconciliation; Baseline_Decision; CONVENTIONS; outline; Team Report; README W03 | OK |
| 2026-09-17 | PATCH | Literature_Matrix remap + metric skeleton + path Reading_Notes | OK |
| 2026-09-17 | PATCH | Candidate_Methods, Research_Gap, WEEK2_STATUS, W02 pointer/README, W01_Team_Report | OK |
| 2026-09-17 | HASH | W01 SV3 vs Reading_Notes SV3 | IDENTICAL |
| 2026-09-17 | MOVE | `docs/W01/SV3_Paper_Review.md` → `_archive/W03_identical_SV3_2026-09-17/` | OK |
| 2026-09-17 | WRITE | POINTER `docs/W01/SV3_Paper_Review.md` | OK |
| 2026-09-17 | HASH | W01 SV1/SV2/SV4 vs Reading_Notes | KHÁC → giữ nguyên, ⚠️REVIEW |
| 2026-09-17 | SKIP | Tách val; train; xóa file; `src/`; `data/` nội dung | SKIP |
| 2026-09-17 | SKIP | Sửa nhật ký W02_DIFF / PLACEMENT_AUDIT / MIGRATION / CONSOLIDATION | SKIP (lịch sử) |
| 2026-09-17 | COUNT_AFTER | `src/models` n=2; `src/prompting` n=2 | OK khớp trước |

## Self-check (chạy 2026-09-17, terminal)

| Check | Kết quả |
|-------|---------|
| grep `docs/literature/` trong canonical Literature_Matrix | **PASS** (0) |
| `src/models/` n=2 (`vlm_loader.py`, `__init__.py`); `src/prompting/` n=2 (`builder.py`, `__init__.py`) | **PASS** |
| không `src/**/*.py` sửa ngày 2026-09-17 | **PASS** |
| không file bị xóa (SV3 chỉ move archive) | **PASS** |
| RQ/H một hệ: RQ1–4 table; H1–8 table; RQ0 không còn hàng; H9 không còn hàng giả thuyết | **PASS** |
| `docs/W03/` đủ README, Team_Report, Reconciliation, Baseline_Decision, DIFF, CLOSEOUT | **PASS** |
| `docs/04_slides/review1/review1_outline.md` | **PASS** |
| Dòng `CHỐT bởi nhóm: ____` để trống | **PASS** (file có dòng; regex PowerShell lỗi encoding UTF-8 — xác nhận bằng read file) |

**Tổng: PASS** (1 false-fail encoding trên CHỐT, đã đối file).


## ⚠️REVIEW (rút gọn)

Xem bảng đầy đủ `docs/W03/W03_DIFF_REPORT.md` §F. Leader cần: Gap×Paper L7/L8/L10/L9; SV1–2–4 W01 vs Reading_Notes; VRAM/P100/licence; tỉ lệ val; zip mất snapshot.

## [TODO-human]

Xem DIFF §G + khối "VIỆC CÒN LẠI CHO CON NGƯỜI" cuối `W03_DIFF_REPORT` / tin nhắn closeout.

## Sanity (thủ công lúc viết log)

| Check | Kết quả lúc viết |
|-------|------------------|
| `docs/W03/` 6 file listed | viết xong cùng batch |
| Không `docs/literature/` trên matrix | đã grep = 0 trước khi chốt log |
| Pointer SV3 1 dòng | OK |
