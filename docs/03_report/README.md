# 03_report — Báo cáo / Thesis

Báo cáo (thesis) được **viết trên Overleaf** (LaTeX). Folder này là **bản đồng bộ local**:

- Giữ bản `.tex` đồng bộ từ Overleaf (source of truth vẫn là project Overleaf).
- Chứa **hình/bảng xuất ra** (figures, tables) dùng trong báo cáo.
- Khung nội dung theo chương: `R1_Introduction`, `R3_Existing_Systems`, `R4_Methodology`, `R5_Implementation`, `R6_Results`, `R7_Discussion_Conclusion` (bản `.md` nháp; nội dung chính thức nằm ở `.tex` trên Overleaf).

**Nguồn trích dẫn** (PDF paper, `.bib`, reading notes) nằm ở [`../02_references/`](../02_references/).

## Quy trình đồng bộ

1. Viết/chỉnh nội dung trên Overleaf.
2. Định kỳ export/pull `.tex` + assets về đây để backup local và version cùng code.
3. Khi trích dẫn, thêm entry vào `.bib` và đặt PDF nguồn trong `../02_references/`.
