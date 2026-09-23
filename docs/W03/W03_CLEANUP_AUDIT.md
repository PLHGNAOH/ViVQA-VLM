# W03 cleanup audit — 2026-09-23

Quét toàn cây trước khi xoá/move. `data/` và `adapters/` không đụng (vẫn match `.gitignore`). Không commit, không push.

## A. Rác an toàn xoá

| Đường dẫn | Lý do | Hành động |
|-----------|--------|-----------|
| `src/__pycache__/` (`__init__.cpython-312.pyc`) | bytecode | HARD-DELETE |
| `src/data/__pycache__/` (3 file `.pyc`: `__init__`, `prepare_vivqa`, `prepare_vitextvqa`) | bytecode | HARD-DELETE |
| `src/eval/__pycache__/` (2 file `.pyc`: `__init__`, `metrics`) | bytecode | HARD-DELETE |

Không thấy `.DS_Store`, `Thumbs.db`, `.ipynb_checkpoints/`, `*.swp`/`*~`.

## B. Trùng lặp

| Đường dẫn | Lý do | Hành động |
|-----------|--------|-----------|
| `notebooks/01_zeroshot_baseline.ipynb` (2 761 518 B, 11 code cell, 35 output, import `src.eval.metrics` + `src.data.prepare_vitextvqa`) | Bản Colab đã chạy (W04 closeout đã dời từ `docs/W04/`) | **GIỮ** |
| `notebooks/01_zeroshot_baseline_vivqa.ipynb` (10 953 B, 0 output, scaffold; import `src.data.vivqa_dataset` / `src.models.vlm_loader` / `src.eval.run_eval`) | Scaffold chưa chạy. Tên `01_baseline_zeroshot_qwen25vl.ipynb` **không tồn tại** | MOVE → `_archive/cleanup_2026-09-23/notebooks/` |
| `experiments.zip`, `docs/W02.zip` | Không có trên đĩa | không làm |
| `_archive/src_data_2026-09-14.zip` (33 230 B) | Zip cũ, đã nằm `_archive/`, `*.zip` đã ignore, **không** bị track | để nguyên |
| `docs/W01/SV1_Paper_Review.md` (14 637 B) vs `docs/02_references/Reading_Notes/SV1_Paper_Review.md` (2 964 B) | **Khác hash** — bản W01 dài hơn, không phải copy | GIỮ cả hai |
| `docs/W01/SV2_Paper_Review.md` (12 895 B) vs Reading_Notes (3 256 B) | Khác hash | GIỮ cả hai |
| `docs/W01/SV3_Paper_Review.md` (118 B) | Đã là pointer 1 dòng sang Reading_Notes (49 702 B). Bản identical đã ở `_archive/W03_identical_SV3_2026-09-17/` | GIỮ pointer |
| `docs/W01/SV4_Paper_Review.md` (11 959 B) vs Reading_Notes (3 167 B) | Khác hash | GIỮ cả hai |
| `docs/W02/Literature_Matrix.md` (315 B) | Pointer sang `docs/02_references/Literature_Matrix.md`, không phải bản copy | GIỮ |
| `docs/W01/ViVQA-VLM_Week1_Progress.pptx` ≡ `docs/04_slides/ViVQA-VLM_Week1_Progress.pptx` (SHA256 `64e0be5e8e92…`, 596 085 B) | Byte-identical. Cả hai chỗ đều hợp quy ước (pack tuần vs `04_slides`) | **Nhóm F** — không move |

## C. Lạc chỗ

| Đường dẫn | Lý do | Hành động |
|-----------|--------|-----------|
| `1.jpg` / `2.jpg` ở gốc repo | **Không còn.** Ảnh nằm `data/samples/1.jpg`, `data/samples/2.jpg` (trong `data/`, gitignore) | không đụng |
| `src/README.from_AIP491.md` | README khóa "Mother of AI / arXiv Paper Curator", không phải ViVQA. `src/README.md` là README đúng. PLACEMENT_AUDIT đã ghi nên đưa `_archive/` | MOVE → `_archive/cleanup_2026-09-23/src/` |
| `docs/W04/W04_verify_baseline.ipynb` | Notebook self-contained xác nhận số W04, đặt cùng deliverable tuần. Runner thật là `notebooks/01_zeroshot_baseline.ipynb` | GIỮ chỗ |
| `AI_Capstone_Review123_Template.xlsx` (gốc) | Template, chưa rõ `docs/templates/` hay giữ gốc | **Nhóm F** |
| `PLACEMENT_AUDIT.md` (gốc) | Audit cũ 2026-09, vẫn được nhật ký W03 trích | **Nhóm F** |

## D. Link chết (markdown, cây đang dùng — không tính `_archive/`)

| File:dòng | Path hiện tại | Path đúng | Hành động |
|-----------|----------------|-----------|-----------|
| `docs/W01/README.md:53` | `../../Master_Instruction.md` (ra gốc repo, sai cả case) | `../master_instruction.md` | sửa href |
| `docs/W01/README.md:49` | backtick `docs/Master_Instruction.md` | `docs/master_instruction.md` | sửa path |
| `docs/W01/README.md:54` | `../../templates/Weekly_Report_Template.md` | `../templates/Weekly_Report_Template.md` | sửa href |
| `docs/W01/README.md:55` | `../../templates/Experiment_Log_Template.md` | `../templates/Experiment_Log_Template.md` | sửa href |
| `src/README.md:36` | `notebooks/01_zeroshot_baseline_vivqa.ipynb` (scaffold) | `notebooks/01_zeroshot_baseline.ipynb` | sửa path sau khi archive scaffold |
| `README.md:25` | cây thư mục còn `docs/literature/` | reviews nằm `docs/02_references/Reading_Notes/` | sửa 1 dòng cây |
| `docs/02_references/Literature_Matrix.md` | — | đã trỏ `Reading_Notes/`, grep `docs/literature/` = 0 | không sửa |
| Nhật ký lịch sử (`docs/W02_DIFF_REPORT.md`, `docs/W02_CLOSEOUT_LOG.md`, `docs/W03/W03_*`, `docs/MIGRATION_REPORT.md`, `docs/LITERATURE_CONSOLIDATION_LOG.md`, `PLACEMENT_AUDIT.md`, `docs/W04/W04_CLOSEOUT_LOG.md`) | còn chuỗi path cũ | nhật ký sự kiện đã xảy ra | **không sửa** (W03 đã SKIP các file này) |

## E. File lớn / nhạy đang track

`git ls-files`: không có `*.zip`, `*.safetensors`, `*.bin`, `*.pt`, `*.jpg`, `*.png`, `__pycache__` nào đang track.

| Đường dẫn | Kích thước | Ghi chú | Hành động |
|-----------|------------|---------|-----------|
| `docs/02_references/[Description]_Project_VQA.pdf` | 17 683 031 B | PDF tham chiếu, **không** nằm gitignore | **Nhóm F** — không `git rm --cached` |
| `docs/02_references/smart_farm.pdf` | 17 369 775 B | PDF lớn, không phải weight | **Nhóm F** |
| `notebooks/01_zeroshot_baseline.ipynb` | 2 761 518 B | Output Colab nhúng — đây là bản giữ | GIỮ |
| `hand_off_to_claude.md` | ~2.6 KB, đang track | Scratch phối hợp; protocol yêu cầu file ở gốc | **Nhóm F** — không untrack |
| `_archive/production-agentic-rag-course/**` | cả khóa RAG khác + gif 4.3 MB + `uv.lock` | Đã nằm `_archive/`, vẫn track | **Nhóm F** |
| `_archive/Đề tài đã sắp xếp/**` | nhiều `.docx`/`.pdf` đề tài nhóm khác | Đã archive, vẫn track; cân nhắc trước khi repo public | **Nhóm F** |
| `docs/01_proposal/…DucDNM2.pdf` | 242 455 B | Trùng kích thước với bản trong `_archive/Đề tài đã sắp xếp/02_Kha_Thi/` | GIỮ bản `docs/01_proposal/`; bản archive đã có sẵn |

`git check-ignore`: `data/` (kể cả `data/samples/*.jpg`), `adapters/` (kể cả README/.gitkeep), `*.zip`, `*.png` (kể cả `docs/00_admin/TimeLine.png` — ảnh docs **không** lên git).

## F. Cần hoàng quyết — để nguyên

1. Hai bản `ViVQA-VLM_Week1_Progress.pptx` giống hệt (`docs/W01/` và `docs/04_slides/`). Chọn một bản canonical rồi archive bản kia.
2. Hai PDF ~17 MB trong `docs/02_references/` (`smart_farm.pdf`, `[Description]_Project_VQA.pdf`) — giữ trong git hay chuyển chỗ khác / Git LFS.
3. Toàn bộ `_archive/production-agentic-rag-course/` vẫn tracked — có bỏ khỏi git (`git rm -r --cached`) trước khi push public không.
4. `_archive/Đề tài đã sắp xếp/` (đề tài SV khác) vẫn tracked.
5. `hand_off_to_claude.md` đang tracked. Protocol repo yêu cầu file này; untrack chỉ khi nhóm quyết scratch không lên GitHub.
6. `AI_Capstone_Review123_Template.xlsx` và `PLACEMENT_AUDIT.md` ở gốc repo.
7. `.gitignore` có `*.png` / `*.jpg` nên ảnh trong `docs/00_admin/` không được track. Bỏ rule ảnh nếu muốn timeline/tiêu đề lên GitHub (đừng bỏ `/data/`).
8. `docs/W03/README.md` **đã dirty từ trước task này**: nội dung index tuần bị thay bằng snippet `npx vercel --prod` (40 dòng → 6 dòng). Cleanup **không** sửa file này. `git add -A` sẽ stage thay đổi đó — xem lại trước khi commit.
9. `docs/W03/index.html` + `docs/W03/vendor/chart.umd.min.js` là file mới (presentation), không sửa nội dung; `git add -A` sẽ thêm chúng.

## Đã xử lý

### Đã xoá
- `src/__pycache__/`
- `src/data/__pycache__/`
- `src/eval/__pycache__/`
- (bytecode do `py_compile` sinh thêm ở `src/models/__pycache__/` và `src/prompting/__pycache__/` đã xoá lại)

### Đã move
- `notebooks/01_zeroshot_baseline_vivqa.ipynb` → `_archive/cleanup_2026-09-23/notebooks/01_zeroshot_baseline_vivqa.ipynb` (scaffold, 0 output)
- **Giữ** `notebooks/01_zeroshot_baseline.ipynb` (bản Colab đã chạy)
- `src/README.from_AIP491.md` → `_archive/cleanup_2026-09-23/src/README.from_AIP491.md`

### Đã sửa link / path
- `docs/W01/README.md` — 3 href + backtick master instruction
- `src/README.md` — pointer notebook sang bản giữ
- `README.md` — bỏ dòng cây `literature/` (folder không còn; canonical ở `docs/02_references/Reading_Notes/`)

### Import
Không move file trong `src/**/*.py`. `py_compile` các module `src/` OK. `experiments/W02_dataset_stats/` còn. `run_eval.py` vẫn import `src.prompting.builder`; `ocr.py` vẫn import `src.data.vivqa_dataset`. Notebook giữ vẫn import `src.eval.metrics` và `src.data.prepare_vitextvqa`. Quét link markdown cây active: 0 link chết.

### .gitignore
Đã thêm `.DS_Store` và `Thumbs.db`. Đã có sẵn: `/data/`, `/adapters/`, `*.pyc`, `__pycache__/`, `.ipynb_checkpoints/`, `*.zip`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`.

### Không đụng
`data/`, `adapters/`, `docs/W03/index.html`, mọi `.pptx`, nội dung deliverable ngoài các path link ở trên. Không `git commit`, không `git push`.
