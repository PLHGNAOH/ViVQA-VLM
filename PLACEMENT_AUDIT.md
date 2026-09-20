# PLACEMENT AUDIT — ViVQA-VLM

Ngày audit: 2026-09-16. Phạm vi: mục root **Date modified = 14/09/2026** (`docs/`, `experiments/`, `src/`, `experiments.zip`, `WEEK2_STATUS.md`). Read-only; không move/sửa gì ngoài file này.

Chuẩn đối chiếu: `docs/{W01..W15}/` (không lớp `weekly/`), `src/{data,baseline,adaptation,eval,prototype}/`, `experiments/Wxx_*/`.

Thời gian: Windows `CreationTime` / `LastWriteTime` (local +07). File **mới 14/9** = CreationTime 14/9. File **cũ sửa 14/9** = CreationTime trước đó, LastWriteTime 14/9.

---

## Root ngoài phạm vi (1 dòng)

| Mục | Ghi chú |
|-----|---------|
| `adapters/` | Tạo 12/09/2026; 2 file. Không quét. |
| `configs/` | Tạo 12/09/2026; 3 file. Không quét. |
| `data/` | Tạo 12/09/2026; 56 file (gồm dataset ViVQA-X). Không quét. |
| `notebooks/` | Tạo 12/09/2026; 1 notebook. Không quét. |
| `_archive/` | Tạo 12/09/2026, sửa 12/09 tối; ~195 file (migration dump). Không quét. |
| `README.md` | Tạo/sửa 12/09/2026. |
| `requirements.txt` | Tạo/sửa 12/09/2026. |

---

## A. CÂY THƯ MỤC (đệ quy) — `docs/`, `src/`, `experiments/`

Bỏ nội dung nhị phân lớn (pdf/pptx/png/docx/zip/pyc); giữ tên + kích thước.

```
docs/
  00_admin/
    tieu_de_moi_tuan.png  [omit 112353 B]   ← tạo 16/09, không thuộc 14/9
    TimeLine.png          [omit 294396 B]
  01_proposal/
    FA26AI29_…_DucDNM2.pdf  [omit 242455 B]
  02_references/
    Reading_Notes/
      BLIP2_Li_ICML2023.md
      Florence2_Xiao_CVPR2024.md
      LoRA_Hu_ICLR2022.md
      POPE_Li_EMNLP2023.md
      QLoRA_Dettmers_2023.md
      Qwen25VL_Bai_2025.md
      README.md
      ViTextVQA_Nguyen_2024.md
      ViVQA_Tran_PACLIC2021.md
    templates/
      Phan_Tich_Template_Capstone_AIP491_Agentic_RAG.pdf  [omit]
      Template_AIP491_CP_StudentsGuide.docx               [omit]
    [Description]_Project_VQA.pdf  [omit ~17 MB]
    smart_farm.pdf                 [omit ~17 MB]
  03_report/
    Capstone_Report_1_Project_Introduction.docx  [omit]
    R1_Introduction.md
    R3_Existing_Systems.md
    R4_Methodology.md
    R5_Implementation.md
    R6_Results.md
    R7_Discussion_Conclusion.md
    README.md
  04_slides/
    faculty_review/   (trống)
    final_defense/    (trống)
    review1/          (trống)
    review2/          (trống)
    ViVQA-VLM_Week1_Progress.pptx  [omit 596085 B]
  literature/
    Literature_Matrix.md  + .from_AIP491.md
    SV1_Paper_Review.md   + .from_AIP491.md
    SV2_Paper_Review.md   + .from_AIP491.md
    SV3_Paper_Review.md   + .from_AIP491.md
    SV4_Paper_Review.md   + .from_AIP491.md
  notes/
    Pretrain_VLM.txt
    README.md
    SFT_Large_Vision_Language_Model.txt
    ViVQA-VLM_Ke_hoach_15_tuan.md
    ViVQA-VLM_Ke_hoach_15_tuan.pdf  [omit]
  templates/
    Experiment_Log_Template.md
    Paper_Review_Template.md
    Review_Checklist_Template.md
    Weekly_Report_Template.md
  W02/                          ← MỚI 14/9; khớp chuẩn docs/W02/
    Candidate_Methods.md
    Dataset_Analysis.md
    Hypotheses.md
    Literature_Matrix.md
    Research_Gap.md
    Research_Questions.md
  weekly/                       ← lớp THỪA vs chuẩn docs/{W01..W15}/
    PROGRESS.md
    W01/ … W15/                 (scaffold + deliverable; xem chi tiết dưới)
  master_instruction.md         ← chuẩn: Master_Instruction.md (hoa chữ)
  MIGRATION_REPORT.md
  W02.zip                       [omit 15534 B]  ← MỚI 14/9; snapshot của docs/W02/

docs/weekly/W01/
  Project_Plan_15_Weeks.md      (sửa 14/9; tạo 06/9)
  README.md
  Research_Orientation.md       (sửa 14/9)
  SV1_Paper_Review.md           (sửa 14/9)
  SV2_Paper_Review.md           (sửa 14/9)
  SV3_Paper_Review.md
  SV4_Paper_Review.md           (sửa 14/9)
  tracker.md                    (sửa 14/9)
  ViVQA-VLM_Week1_Progress.pptx [omit]
  W01_Progress_Report.md        (tạo+sửa 14/9)
  W01_Team_Report.md            (sửa 14/9)

docs/weekly/W02/
  Candidate_Methods.md          (sửa 14/9; HASH KHÁC docs/W02/)
  Dataset_Analysis.md           (không sửa 14/9; HASH KHÁC)
  Hypotheses.md                 (sửa 14/9; HASH KHÁC)
  Literature_Matrix.md          (sửa 14/9; HASH KHÁC)
  README.md
  Research_Gap.md               (sửa 14/9; HASH KHÁC)
  Research_Questions.md         (sửa 14/9; HASH KHÁC)
  SV1..SV4_Paper_Review.md      (stub ~1.1 KB, 03/9)
  tracker.md
  W02_Team_Report.md
  WEEK_02_Literature_and_Dataset.txt

docs/weekly/W03..W15/ — scaffold cũ (README, tracker, Wxx_Team_Report, WEEK_xx_*.txt, vài file chuyên đề). Không sửa 14/9.

src/
  adaptation/   .gitkeep, README.md
  baseline/     .gitkeep, README.md
  data/
    __pycache__/                [omit bytecode; tạo 14/9]
    .gitkeep, __init__.py, ocr.py, README.md, vivqa_dataset.py
    analyze_dataset.py          ← MỚI 14/9
  eval/         .gitkeep, __init__.py, metrics.py, README.md, run_eval.py
  models/       __init__.py, vlm_loader.py          ← NGOÀI bộ 5 module chuẩn
  prompting/    __init__.py, builder.py             ← NGOÀI bộ 5 module chuẩn
  prototype/    .gitkeep, README.md
  __init__.py
  data.zip                    [omit 33230 B]  ← MỚI 14/9; snapshot src/data/
  README.from_AIP491.md
  README.md

experiments/
  .gitkeep, README.md
  W01_Pilot/    README.md
  W02_dataset_stats/            ← MỚI 14/9; khớp chuẩn experiments/Wxx_*/
    vivqa_stats.json
    vitextvqa_stats.json
    vivqa_x_stats.json
    samples/
      vivqa__{Counting,Object-recognition,Other,Reasoning,Text-reading,Yes-No}.json
      vitextvqa__{…6 types}.json
      vivqa_x__{…6 types}.json
```

---

## B. BẢNG AUDIT — mọi file/thư mục tạo hoặc sửa 14/09/2026

| Đường dẫn thực tế | Loại | Tạo | Sửa 14/9 | ĐÚNG/SAI | Vị trí đúng đề xuất | Rủi ro khi di chuyển |
|---|---|---|---|---|---|---|
| `WEEK2_STATUS.md` | file .md | 14/9 22:02 | 14/9 22:02 | **SAI** | `docs/W02/WEEK2_STATUS.md` | Thấp (.md). Cập nhật 1 đường dẫn trong README nếu có link. |
| `experiments.zip` | zip root | 14/9 23:01 | 14/9 23:01 | **SAI** | `_archive/experiments_2026-09-14.zip` | Thấp. Bản sao byte-identical với `experiments/` (24/24 SHA256 khớp). |
| `docs/` | dir | 12/9 | 14/9 22:55 (W02.zip) | ĐÚNG (thư mục chuẩn) | — | — |
| `docs/W02/` | dir | 14/9 21:58 | 14/9 22:01 | **ĐÚNG** | `docs/W02/` | — |
| `docs/W02/Literature_Matrix.md` | .md | 14/9 21:58 | 21:58 | **ĐÚNG** | giữ | Thấp |
| `docs/W02/Dataset_Analysis.md` | .md | 14/9 22:00 | 22:00 | **ĐÚNG** | giữ | Thấp |
| `docs/W02/Research_Gap.md` | .md | 14/9 22:01 | 22:01 | **ĐÚNG** | giữ | Thấp |
| `docs/W02/Research_Questions.md` | .md | 14/9 22:01 | 22:01 | **ĐÚNG** | giữ | Thấp |
| `docs/W02/Hypotheses.md` | .md | 14/9 22:01 | 22:01 | **ĐÚNG** | giữ | Thấp |
| `docs/W02/Candidate_Methods.md` | .md | 14/9 22:01 | 22:01 | **ĐÚNG** | giữ | Thấp |
| `docs/W02.zip` | zip | 14/9 22:55 | 22:55 | **SAI** | `_archive/docs_W02_2026-09-14.zip` | Thấp. Snapshot đúng 6 file `docs/W02/`. |
| `docs/weekly/` | dir | 12/9 | 14/9 22:55 | **SAI** (lớp thừa) | flatten → `docs/W01`…`docs/W15` | Trung bình: trùng tên với `docs/W02/` **khác hash**. |
| `docs/weekly/W01/` | dir | 12/9 | 14/9 21:56 | **SAI** | `docs/W01/` | Thấp (.md) nếu chưa có `docs/W01/`. |
| `docs/weekly/W01/W01_Progress_Report.md` | .md mới | 14/9 15:41 | 17:17 | **SAI** | `docs/W01/W01_Progress_Report.md` | Thấp |
| `docs/weekly/W01/Project_Plan_15_Weeks.md` | .md cũ sửa | 06/9 | 17:17 | **SAI** | `docs/W01/Project_Plan_15_Weeks.md` | Thấp |
| `docs/weekly/W01/SV1_Paper_Review.md` | .md cũ sửa | 06/9 | 17:17 | **SAI** | `docs/W01/SV1_Paper_Review.md` | Thấp |
| `docs/weekly/W01/SV2_Paper_Review.md` | .md cũ sửa | 06/9 | 17:17 | **SAI** | `docs/W01/SV2_Paper_Review.md` | Thấp |
| `docs/weekly/W01/SV4_Paper_Review.md` | .md cũ sửa | 06/9 | 16:06 | **SAI** | `docs/W01/SV4_Paper_Review.md` | Thấp |
| `docs/weekly/W01/W01_Team_Report.md` | .md cũ sửa | 06/9 | 16:06 | **SAI** | `docs/W01/W01_Team_Report.md` | Thấp |
| `docs/weekly/W01/Research_Orientation.md` | .md cũ sửa | 12/9 | 16:07 | **SAI** | `docs/W01/` (ngoài list tên chuẩn, vẫn thuộc W01) | Thấp |
| `docs/weekly/W01/tracker.md` | .md cũ sửa | 12/9 | 16:08 | **SAI** | `docs/W01/tracker.md` | Thấp |
| `docs/weekly/W02/` | dir | 12/9 | 14/9 16:09 | **SAI** | không merge đè `docs/W02/` | **Cao nội dung:** cùng tên, khác SHA256/size. |
| `docs/weekly/W02/Literature_Matrix.md` | .md cũ sửa | 06/9 | 16:08 | **SAI** | `_archive/weekly_W02/` (bản cũ 3713 B vs 11320 B ở `docs/W02/`) | Thấp nếu archive; **mất dữ liệu nếu overwrite** |
| `docs/weekly/W02/Research_Gap.md` | .md cũ sửa | 06/9 | 16:08 | **SAI** | như trên (3129 vs 6159 B) | như trên |
| `docs/weekly/W02/Research_Questions.md` | .md cũ sửa | 06/9 | 16:08 | **SAI** | như trên (2538 vs 1173 B — bản weekly DÀI hơn) | So sánh tay trước khi bỏ |
| `docs/weekly/W02/Hypotheses.md` | .md cũ sửa | 06/9 | 16:08 | **SAI** | như trên (3262 vs 1041 B — weekly DÀI hơn) | So sánh tay |
| `docs/weekly/W02/Candidate_Methods.md` | .md cũ sửa | 06/9 | 16:08 | **SAI** | như trên (4170 vs 1309 B — weekly DÀI hơn) | So sánh tay |
| `src/` | dir | 12/9 | 14/9 23:01 (`data.zip`) | ĐÚNG (thư mục chuẩn) | — | extra module: xem mục 3 |
| `src/data/` | dir | 12/9 | 14/9 22:02 | **ĐÚNG** | `src/data/` | — |
| `src/data/analyze_dataset.py` | .py mới | 14/9 21:49 | 21:55 | **ĐÚNG** | giữ | Stdlib only; hardcode `experiments/W02_dataset_stats`. Move file → sửa `DEFAULT_OUT_DIR` + lệnh `python src/data/analyze_dataset.py`. |
| `src/data.zip` | zip | 14/9 23:01 | 23:01 | **SAI** | `_archive/src_data_2026-09-14.zip` | Thấp. Snapshot `src/data/` **kèm** `__pycache__/*.pyc`. |
| `src/data/__pycache__/` | dir | 14/9 22:02 | 22:02 | **SAI** | xóa / gitignore; không thuộc cấu trúc chuẩn | Thấp (artifact) |
| `src/data/__pycache__/analyze_dataset.cpython-312.pyc` | bytecode | 14/9 22:02 | 22:02 | **SAI** | xóa | Thấp |
| `experiments/` | dir | 12/9 | 14/9 21:50 | **ĐÚNG** | — | — |
| `experiments/W02_dataset_stats/` | dir | 14/9 21:50 | 21:50 | **ĐÚNG** | giữ | — |
| `experiments/W02_dataset_stats/samples/` | dir | 14/9 21:50 | 21:55 | **ĐÚNG** | giữ | — |
| `experiments/W02_dataset_stats/vivqa_stats.json` | json | 14/9 21:50 | 21:55 | **ĐÚNG** | giữ | Thấp |
| `experiments/W02_dataset_stats/vitextvqa_stats.json` | json | 14/9 21:50 | 21:55 | **ĐÚNG** | giữ | Thấp |
| `experiments/W02_dataset_stats/vivqa_x_stats.json` | json | 14/9 21:50 | 21:55 | **ĐÚNG** | giữ | Thấp |
| `experiments/W02_dataset_stats/samples/*.json` (18 file, tất cả tạo+sửa 14/9) | json | 14/9 21:50–21:55 | 21:55 | **ĐÚNG** | giữ | Thấp. 6 type × {vivqa, vitextvqa, vivqa_x}. |

`docs/weekly/W02/Dataset_Analysis.md` **không** sửa 14/9 (LastWrite 06/9) nên không vào bảng trên; vẫn là bản trùng tên/khác nội dung (3407 B vs 9343 B).

---

## Kiểm riêng 4 điểm

### 1. `experiments.zip` vs `experiments/`

**Kết luận: backup/export lạc chỗ, trùng lặp 100%.**

- Zip tạo 14/9 23:01:18, 23552 B, nằm root.
- 24 entry file; SHA256 từng file **khớp** bản trên đĩa. Không entry nào chỉ có trong zip hoặc chỉ có trên folder.
- Nội dung zip (không giải nén):

```
experiments/.gitkeep
experiments/README.md
experiments/W01_Pilot/README.md
experiments/W02_dataset_stats/vivqa_stats.json
experiments/W02_dataset_stats/vitextvqa_stats.json
experiments/W02_dataset_stats/vivqa_x_stats.json
experiments/W02_dataset_stats/samples/vivqa__Counting.json
experiments/W02_dataset_stats/samples/vivqa__Object-recognition.json
experiments/W02_dataset_stats/samples/vivqa__Other.json
experiments/W02_dataset_stats/samples/vivqa__Reasoning.json
experiments/W02_dataset_stats/samples/vivqa__Text-reading.json
experiments/W02_dataset_stats/samples/vivqa__Yes-No.json
experiments/W02_dataset_stats/samples/vitextvqa__Counting.json
experiments/W02_dataset_stats/samples/vitextvqa__Object-recognition.json
experiments/W02_dataset_stats/samples/vitextvqa__Other.json
experiments/W02_dataset_stats/samples/vitextvqa__Reasoning.json
experiments/W02_dataset_stats/samples/vitextvqa__Text-reading.json
experiments/W02_dataset_stats/samples/vitextvqa__Yes-No.json
experiments/W02_dataset_stats/samples/vivqa_x__Counting.json
experiments/W02_dataset_stats/samples/vivqa_x__Object-recognition.json
experiments/W02_dataset_stats/samples/vivqa_x__Other.json
experiments/W02_dataset_stats/samples/vivqa_x__Reasoning.json
experiments/W02_dataset_stats/samples/vivqa_x__Text-reading.json
experiments/W02_dataset_stats/samples/vivqa_x__Yes-No.json
```

**Đề xuất:** chuyển `_archive/` (giữ snapshot). **Không xoá** cho đến khi archive xong. **Không** giải nén đè folder. Giữ `experiments/` làm bản sống.

Cùng kiểu (14/9, lạc chỗ): `docs/W02.zip`, `src/data.zip` → `_archive/`.

### 2. `WEEK2_STATUS.md` (root)

**SAI.** Báo cáo tuần 2; tự trích dẫn `docs/W02/`, `src/data/analyze_dataset.py`, `experiments/W02_dataset_stats/`.

**Đề xuất:** `docs/W02/WEEK2_STATUS.md` (không nhầm `docs/weekly/W02/`). `.md` an toàn. File `docs/weekly/W02/W02_Team_Report.md` là stub 03/9 (2704 B) — không ghi đè bằng status.

### 3. `src/` — mục ngoài `{data, baseline, adaptation, eval, prototype}`

| Mục | Thuộc 14/9? | Đánh giá |
|-----|-------------|----------|
| `src/models/` (`vlm_loader.py`) | không (11/9) | **Ngoài chuẩn.** Notebook import `src.models.vlm_loader`. |
| `src/prompting/` (`builder.py`) | không (11/9) | **Ngoài chuẩn.** `src/eval/run_eval.py` import `src.prompting.builder`. |
| `src/data.zip` | có | Zip lạc chỗ. |
| `src/data/__pycache__/` | có | Artifact. |
| `src/README.from_AIP491.md` | không | File migration; nên `_archive/` hoặc gộp README. |
| `src/__init__.py` | không | Package marker; chấp nhận được. |

Module **đúng chuẩn:** `data/` (gồm `analyze_dataset.py` mới 14/9), `baseline/`, `adaptation/`, `eval/`, `prototype/`.

Nếu sau này nhét `models`/`prompting` vào `baseline` hoặc `prototype`: **vỡ import** — phải sửa `src/eval/run_eval.py`, `notebooks/01_zeroshot_baseline_vivqa.ipynb`, `src/README.md`. `analyze_dataset.py` **không** import nội bộ `src.*`.

### 4. `docs/` — file tuần 2 đã đúng `docs/W02/`?

**Một phần — đang song song 2 cây.**

| Deliverable chuẩn | `docs/W02/` (mới 14/9 tối) | `docs/weekly/W02/` | `docs/` root |
|---|---|---|---|
| Literature_Matrix.md | **có (canonical, 11320 B)** | có, **khác hash**, 3713 B | `docs/literature/` bản W1 |
| Dataset_Analysis.md | **có (9343 B)** | có, khác hash, 3407 B, không sửa 14/9 | — |
| Research_Gap.md | **có** | có, khác hash | — |
| Research_Questions.md | **có (ngắn hơn weekly)** | có, dài hơn | — |
| Hypotheses.md | **có (ngắn hơn weekly)** | có, dài hơn | — |
| Candidate_Methods.md | **có (ngắn hơn weekly)** | có, dài hơn | — |
| SVx_Paper_Review.md | **thiếu** | stub 03/9 | bản đầy đủ W1: `docs/weekly/W01/` + `docs/literature/` |
| Project_Plan_15_Weeks.md | **thiếu** | không | nằm `docs/weekly/W01/` |
| W02_Team_Report.md | **thiếu** (status ở root) | stub 03/9 | `WEEK2_STATUS.md` root |
| Master_Instruction.md | — | — | `docs/master_instruction.md` (sai hoa chữ) |

Không có file W2 nào nhầm vào `docs/W01/` (không tồn tại `docs/W01/`; W01 nằm dưới `weekly/`).

**Canonical W2 deliverable 14/9 tối = `docs/W02/`.** `docs/weekly/W02/` là scaffold cũ (sửa 14/9 **chiều**, trước khi tạo `docs/W02/` tối). Không xoá weekly trước khi diff: 3 file weekly **dài hơn** bản mới (RQ, Hypotheses, Candidate_Methods) — có thể còn nội dung scaffold cần giữ.

---

## C. MOVE PLAN (đề xuất — CHƯA thực hiện)

Thứ tự an toàn: archive zip → status → pycache → weekly W02 (diff trước) → flatten W01. PowerShell (`Rename-Item`/`Move-Item`); repo không nhất thiết là git.

```powershell
# 1. Snapshot zip lạc root/docs/src → _archive (giữ, đừng xoá)
Move-Item .\experiments.zip  .\_archive\experiments_2026-09-14.zip
Move-Item .\docs\W02.zip     .\_archive\docs_W02_2026-09-14.zip
Move-Item .\src\data.zip     .\_archive\src_data_2026-09-14.zip

# 2. Báo cáo tuần 2 vào đúng docs/W02/
Move-Item .\WEEK2_STATUS.md  .\docs\W02\WEEK2_STATUS.md

# 3. Artifact bytecode
Remove-Item -Recurse -Force .\src\data\__pycache__

# 4. KHÔNG merge docs/weekly/W02 → docs/W02 (khác hash).
#    Diff rồi archive bản weekly:
#    Compare-Object (Get-Content docs\W02\X.md) (Get-Content docs\weekly\W02\X.md)
Move-Item .\docs\weekly\W02  .\_archive\docs_weekly_W02_pre_2026-09-14

# 5. Flatten W01 (chưa có docs/W01/ → ít đụng độ)
Move-Item .\docs\weekly\W01  .\docs\W01

# 6. (Tùy chọn, ngoài 14/9) flatten W03–W15 khi rảnh:
#    Move-Item .\docs\weekly\W03 .\docs\W03   # … W15
#    rồi xóa docs/weekly/ nếu trống.

# 7. (Tùy chọn) KHÔNG move src/models hay src/prompting nếu chưa sửa import.
```

**Import sau move**

| File | Cần sửa? |
|------|----------|
| `src/data/analyze_dataset.py` | Không, nếu giữ nguyên path. |
| `src/eval/run_eval.py` | Chỉ khi move `src/prompting/` → đổi `from src.prompting.builder import …` |
| `src/data/ocr.py` | `from src.data.vivqa_dataset import load_vivqa` — vỡ nếu tách `ocr.py` khỏi `src/data/`. |
| `notebooks/01_zeroshot_baseline_vivqa.ipynb` | `src.models.vlm_loader`, `src.eval.run_eval`, `src.data.vivqa_dataset` — vỡ nếu dời 3 module đó. |
| `.md` / `.json` / `.zip` | Không.

Sau bước 2: sửa link `WEEK2_STATUS.md` trong README (nếu có) → `docs/W02/WEEK2_STATUS.md`.

---

## D. TÓM TẮT

| Nhóm | Số |
|------|----|
| File **ĐÚNG chỗ** (14/9) | **28** — 6 md `docs/W02/` + `analyze_dataset.py` + 3 stats + 18 sample json |
| File **SAI chỗ** (14/9) | **18** — `WEEK2_STATUS.md`, 3 zip, 1 pyc, 8 file `docs/weekly/W01/`, 5 file `docs/weekly/W02/` |
| Dir ĐÚNG (14/9) | `docs/W02/`, `src/data/`, `experiments/W02_dataset_stats/` + `samples/` |
| Dir SAI (14/9) | `docs/weekly/` (+ W01/W02), `src/data/__pycache__/` |
| **Cần xử lý** | 3 zip → `_archive/`; status → `docs/W02/`; xóa pycache; **diff rồi** archive `docs/weekly/W02/` (không overwrite); flatten `weekly/W01` → `docs/W01/`; `src/models` + `src/prompting` ngoài chuẩn (không đụng 14/9, import nội bộ) |

W2 deliverable mới nằm đúng `docs/W02/` + `experiments/W02_dataset_stats/` + `src/data/analyze_dataset.py`. Lệch chính: status/zip ở root, cây `docs/weekly/` song song, bytecode, và hai module `src` thừa so với bộ 5 thư mục chuẩn.
