# LITERATURE CONSOLIDATION LOG

Thời điểm: 2026-09-17. Canonical sống = `docs/02_references/`. Không hard-delete work files. Không sửa nội dung matrix (Gap×Paper #1–#10 vs L1–L10 để nguyên).

## Bước 1 — so sánh Literature_Matrix

| Bản | Size | Dòng | Gap×Paper | Ghi chú |
|-----|------|------|-----------|---------|
| `docs/W02/Literature_Matrix.md` | 12156 B | 58 | có | **CHỌN canonical** (nhiều nội dung nhất + Gap matrix) |
| `docs/literature/Literature_Matrix.md` | 3713 B | 28 | có | bảng weekly/SV, nhỏ hơn |
| `docs/02_references/Literature_Matrix.md` | — | — | — | chưa tồn tại |

## Thao tác

| Timestamp | Thao tác | Từ → Đến | Kết quả |
|-----------|----------|----------|---------|
| 2026-09-17 09:33:57 | COUNT_BEFORE | docs/notes n=5; docs/templates n=4; src n=24 | OK |
| 2026-09-17 09:33:57 | MKDIR | `_archive/literature_consolidation_2026-09-16` | OK |
| 2026-09-17 09:33:57 | COPY | `docs/literature/` → `.../docs_literature/` | OK backup đầy đủ |
| 2026-09-17 09:33:57 | COPY | `docs/W02/Literature_Matrix.md` → `.../W02_Literature_Matrix.md` | OK |
| 2026-09-17 09:33:57 | CHOOSE | W02 matrix = canonical | OK |
| 2026-09-17 09:33:57 | COPY | W02 matrix → `docs/02_references/Literature_Matrix.md` | OK bản sống |
| 2026-09-17 09:33:57 | MOVE | `docs/literature/Literature_Matrix.md` → `.../literature_LM.md` | OK |
| 2026-09-17 09:34:29 | WRITE | `docs/W02/Literature_Matrix.md` = pointer | OK (bản đầy đủ đã backup) |
| 2026-09-17 09:34:29 | PATCH | `docs/W02/README.md` link → `../02_references/Literature_Matrix.md` | OK (bản cũ: `.../W02_README_pre.md`) |
| 2026-09-17 09:34:29 | MOVE | SV1–SV4_Paper_Review.md → `docs/02_references/Reading_Notes/` | OK (Reading_Notes chưa có tên đó) |
| 2026-09-17 09:34:29 | MOVE | `*.from_AIP491.md` (5 file) → archive | OK |
| 2026-09-17 09:34:29 | RMDIR_EMPTY | `docs/literature` | OK |
| 2026-09-17 09:34 | COUNT_AFTER | notes=5; templates=4; src=24 | OK khớp |
| 2026-09-17 09:34 | SKIP | `docs/W01/SVx_Paper_Review.md`, `docs/notes/`, `docs/templates/`, `src/` | SKIP không đụng |

## Kiểm tra

| Check | Kết quả |
|-------|---------|
| `docs/02_references/Literature_Matrix.md` tồn tại + grep Gap | OK (12156 B, `## Ma trận Gap × Paper`) |
| Reading_Notes chứa SV1–SV4_Paper_Review | OK (2964 / 3256 / 49702 / 3167 B) |
| `docs/literature/` biến mất | OK |
| `docs/W02/Literature_Matrix.md` pointer (grep `02_references`) | OK |
| notes / templates / src counts | OK 5 / 4 / 24 |

## Ghi chú (không sửa matrix)

Bản sống vẫn trích `docs/literature/SV3_Paper_Review.md` (và path cũ). File review đã chuyển `docs/02_references/Reading_Notes/`. **Không** sửa matrix (luật 5). Remap path + Gap×Paper IDs = việc W3.

`docs/W01/SVx_Paper_Review.md` cố ý giữ nguyên (luật 4).
