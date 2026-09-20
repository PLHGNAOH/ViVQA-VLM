# MIGRATION_REPORT — tái cấu trúc Capstone ViVQA-VLM

Ngày: 2026-09-03  
Workspace: `D:\FPT_university\termno9\AIP491`  
Quy tắc: **không xoá / không ghi đè** file đang có. Chỉ CREATE thư mục+file mới, MOVE, RENAME.

Root **không phải git repo** → dùng `Rename-Item` / `Move-Item` (không `git mv`).

---

## Tóm tắt số lượng

| Hành động | Số lượng |
|-----------|----------|
| RENAME thư mục | 1 (`04_SourceCode` → `src`) |
| MOVE (file/thư mục) | 5 |
| CREATE thư mục mới | 34 |
| CREATE file mới | 109 (108 scaffold + file báo cáo này) |
| DELETE / OVERWRITE | **0** |

---

## Nguồn → Đích → Hành động

### MOVE / RENAME (bắt buộc)

| Nguồn | Đích | Hành động |
|-------|------|-----------|
| `04_SourceCode/` | `src/` | RENAME (giữ nguyên mọi file cũ bên trong, gồm `production-agentic-rag-course/`, `README.md`, `production-agentic-rag_link_github.txt`) |
| `ViVQA-VLM_Master_Instruction.md` | `docs/Master_Instruction.md` | MOVE + đổi tên file |
| `message.txt` | `docs/notes/message.txt` | MOVE |
| `Pretrain_VLM.txt` | `docs/notes/Pretrain_VLM.txt` | MOVE |
| `SFT_Large_Vision_Language_Model.txt` | `docs/notes/SFT_Large_Vision_Language_Model.txt` | MOVE |
| `Đề tài đã sắp xếp/` | `_archive/Đề tài đã sắp xếp/` | MOVE nguyên thư mục |

`00_Admin/`, `01_Proposal/`, `02_References/` (PDF + `templates/`), `03_Report/` (docx/pdf/txt) — **giữ nguyên vị trí và nội dung**. Chỉ **thêm** file/thư mục mới bên trong khi được yêu cầu.

### CREATE — thư mục

| Nguồn | Đích | Hành động |
|-------|------|-----------|
| _(mới)_ | `docs/` | CREATE |
| _(mới)_ | `docs/notes/` | CREATE |
| _(mới)_ | `docs/templates/` | CREATE |
| _(mới)_ | `docs/weekly/` | CREATE |
| _(mới)_ | `docs/weekly/W01/` … `docs/weekly/W15/` | CREATE (15 thư mục) |
| _(mới)_ | `data/` | CREATE |
| _(mới)_ | `data/raw/` | CREATE |
| _(mới)_ | `data/processed/` | CREATE |
| _(mới)_ | `data/ocr/` | CREATE |
| _(mới)_ | `data/splits/` | CREATE |
| _(mới)_ | `experiments/` | CREATE |
| _(mới)_ | `adapters/` | CREATE |
| _(mới)_ | `configs/` | CREATE |
| _(mới)_ | `_archive/` | CREATE |
| _(mới)_ | `02_References/Reading_Notes/` | CREATE |
| _(mới)_ | `src/data/` | CREATE |
| _(mới)_ | `src/baseline/` | CREATE |
| _(mới)_ | `src/adaptation/` | CREATE |
| _(mới)_ | `src/eval/` | CREATE |
| _(mới)_ | `src/prototype/` | CREATE |

### CREATE — file root & ignore

| Nguồn | Đích | Hành động |
|-------|------|-----------|
| _(mới)_ | `README.md` | CREATE |
| _(mới)_ | `.gitignore` | CREATE |
| _(mới)_ | `MIGRATION_REPORT.md` | CREATE (file này) |

### CREATE — .gitkeep (thư mục dữ liệu / artifact rỗng)

| Nguồn | Đích | Hành động |
|-------|------|-----------|
| _(mới)_ | `data/raw/.gitkeep` | CREATE |
| _(mới)_ | `data/processed/.gitkeep` | CREATE |
| _(mới)_ | `data/ocr/.gitkeep` | CREATE |
| _(mới)_ | `data/splits/.gitkeep` | CREATE |
| _(mới)_ | `src/data/.gitkeep` | CREATE |
| _(mới)_ | `src/baseline/.gitkeep` | CREATE |
| _(mới)_ | `src/adaptation/.gitkeep` | CREATE |
| _(mới)_ | `src/eval/.gitkeep` | CREATE |
| _(mới)_ | `src/prototype/.gitkeep` | CREATE |
| _(mới)_ | `experiments/.gitkeep` | CREATE |
| _(mới)_ | `adapters/.gitkeep` | CREATE |
| _(mới)_ | `configs/.gitkeep` | CREATE |
| _(mới)_ | `_archive/.gitkeep` | CREATE |

### CREATE — docs/templates

| Nguồn | Đích | Hành động |
|-------|------|-----------|
| _(mới)_ | `docs/templates/Weekly_Report_Template.md` | CREATE |
| _(mới)_ | `docs/templates/Paper_Review_Template.md` | CREATE |
| _(mới)_ | `docs/templates/Experiment_Log_Template.md` | CREATE |
| _(mới)_ | `docs/templates/Review_Checklist_Template.md` | CREATE |

### CREATE — 03_Report (khung thesis; không đụng file docx/pdf/txt cũ)

| Nguồn | Đích | Hành động |
|-------|------|-----------|
| _(mới)_ | `03_Report/R1_Introduction.md` | CREATE |
| _(mới)_ | `03_Report/R3_Existing_Systems.md` | CREATE |
| _(mới)_ | `03_Report/R4_Methodology.md` | CREATE |
| _(mới)_ | `03_Report/R5_Implementation.md` | CREATE |
| _(mới)_ | `03_Report/R6_Results.md` | CREATE |
| _(mới)_ | `03_Report/R7_Discussion_Conclusion.md` | CREATE |

### CREATE — 02_References/Reading_Notes

| Nguồn | Đích | Hành động |
|-------|------|-----------|
| _(mới)_ | `02_References/Reading_Notes/README.md` | CREATE |
| _(mới)_ | `02_References/Reading_Notes/BLIP2_Li_ICML2023.md` | CREATE |
| _(mới)_ | `02_References/Reading_Notes/Florence2_Xiao_CVPR2024.md` | CREATE |
| _(mới)_ | `02_References/Reading_Notes/Qwen25VL_Bai_2025.md` | CREATE |
| _(mới)_ | `02_References/Reading_Notes/ViVQA_Tran_PACLIC2021.md` | CREATE |
| _(mới)_ | `02_References/Reading_Notes/ViTextVQA_Nguyen_2024.md` | CREATE |
| _(mới)_ | `02_References/Reading_Notes/LoRA_Hu_ICLR2022.md` | CREATE |
| _(mới)_ | `02_References/Reading_Notes/QLoRA_Dettmers_2023.md` | CREATE |
| _(mới)_ | `02_References/Reading_Notes/POPE_Li_EMNLP2023.md` | CREATE |

### CREATE — README module (hướng dẫn chỗ viết code mới)

| Nguồn | Đích | Hành động |
|-------|------|-----------|
| _(mới)_ | `data/README.md` | CREATE |
| _(mới)_ | `experiments/README.md` | CREATE |
| _(mới)_ | `adapters/README.md` | CREATE |
| _(mới)_ | `configs/README.md` | CREATE |
| _(mới)_ | `docs/notes/README.md` | CREATE |
| _(mới)_ | `src/data/README.md` | CREATE |
| _(mới)_ | `src/baseline/README.md` | CREATE |
| _(mới)_ | `src/adaptation/README.md` | CREATE |
| _(mới)_ | `src/eval/README.md` | CREATE |
| _(mới)_ | `src/prototype/README.md` | CREATE |

**Không** ghi đè `src/README.md` (file cũ của khóa Agentic RAG vẫn nguyên).

### CREATE — docs/weekly (mọi tuần: README + Team Report)

| Nguồn | Đích | Hành động |
|-------|------|-----------|
| _(mới)_ | `docs/weekly/W01/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W01/W01_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W02/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W02/W02_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W03/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W03/W03_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W04/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W04/W04_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W05/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W05/W05_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W06/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W06/W06_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W07/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W07/W07_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W08/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W08/W08_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W09/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W09/W09_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W10/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W10/W10_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W11/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W11/W11_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W12/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W12/W12_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W13/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W13/W13_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W14/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W14/W14_Team_Report.md` | CREATE |
| _(mới)_ | `docs/weekly/W15/README.md` | CREATE |
| _(mới)_ | `docs/weekly/W15/W15_Team_Report.md` | CREATE |

### CREATE — file đặc thù theo tuần

| Nguồn | Đích | Hành động |
|-------|------|-----------|
| _(mới)_ | `docs/weekly/W01/Project_Plan_15_Weeks.md` | CREATE |
| _(mới)_ | `docs/weekly/W01/Research_Orientation.md` | CREATE |
| _(mới)_ | `docs/weekly/W02/Literature_Matrix.md` | CREATE |
| _(mới)_ | `docs/weekly/W02/Dataset_Analysis.md` | CREATE |
| _(mới)_ | `docs/weekly/W02/Research_Gap.md` | CREATE |
| _(mới)_ | `docs/weekly/W02/Research_Questions.md` | CREATE |
| _(mới)_ | `docs/weekly/W02/Hypotheses.md` | CREATE |
| _(mới)_ | `docs/weekly/W02/Candidate_Methods.md` | CREATE |
| _(mới)_ | `docs/weekly/W02/SV1_Paper_Review.md` | CREATE |
| _(mới)_ | `docs/weekly/W02/SV2_Paper_Review.md` | CREATE |
| _(mới)_ | `docs/weekly/W02/SV3_Paper_Review.md` | CREATE |
| _(mới)_ | `docs/weekly/W02/SV4_Paper_Review.md` | CREATE |
| _(mới)_ | `docs/weekly/W03/Review1_Slides_Outline.md` | CREATE |
| _(mới)_ | `docs/weekly/W03/Review1_Checklist.md` | CREATE |
| _(mới)_ | `docs/weekly/W04/Environment_Setup.md` | CREATE |
| _(mới)_ | `docs/weekly/W04/Baseline_Plan.md` | CREATE |
| _(mới)_ | `docs/weekly/W05/Baseline_Reproduce.md` | CREATE |
| _(mới)_ | `docs/weekly/W06/Methodology.md` | CREATE |
| _(mới)_ | `docs/weekly/W07/Review2_Slides_Outline.md` | CREATE |
| _(mới)_ | `docs/weekly/W07/Preliminary_Results.md` | CREATE |
| _(mới)_ | `docs/weekly/W07/Review2_Checklist.md` | CREATE |
| _(mới)_ | `docs/weekly/W08/Improvement_v1_PEFT_OCR_RAG.md` | CREATE |
| _(mới)_ | `docs/weekly/W09/Improvement_Final.md` | CREATE |
| _(mới)_ | `docs/weekly/W10/Experiments_Main_Results.md` | CREATE |
| _(mới)_ | `docs/weekly/W11/Evaluation_Full_Metrics.md` | CREATE |
| _(mới)_ | `docs/weekly/W12/Ablation_Study.md` | CREATE |
| _(mới)_ | `docs/weekly/W12/Error_Analysis.md` | CREATE |
| _(mới)_ | `docs/weekly/W12/Hallucination_Analysis.md` | CREATE |
| _(mới)_ | `docs/weekly/W13/Faculty_Review_Package.md` | CREATE |
| _(mới)_ | `docs/weekly/W13/Faculty_Review_Checklist.md` | CREATE |
| _(mới)_ | `docs/weekly/W14/Prototype_Notes.md` | CREATE |
| _(mới)_ | `docs/weekly/W14/Thesis_Draft_Status.md` | CREATE |
| _(mới)_ | `docs/weekly/W15/Final_Defense_Slides_Outline.md` | CREATE |
| _(mới)_ | `docs/weekly/W15/Reproducibility_Package_Checklist.md` | CREATE |

---

## Không đụng tới (xác nhận)

- Toàn bộ nội dung `00_Admin/`, `01_Proposal/`, `02_References/*.pdf`, `02_References/templates/`
- `03_Report/Capstone_Report_1_Project_Introduction.docx`, `03_Report/link_report_docs.txt`
- Mọi file trong `src/production-agentic-rag-course/` và `src/README.md` cũ
- Nội dung `docs/Master_Instruction.md` sau khi MOVE (không edit thêm, dù bên trong vẫn ghi `docs/Wxx/` theo bản gốc)

---

## Giả định đã dùng

1. **Không có git ở root AIP491** → MOVE/RENAME bằng filesystem, không `git mv`. Nested `.git` trong `production-agentic-rag-course` giữ nguyên.
2. **Code Agentic RAG cũ** nằm trong `04_SourceCode` nên **ở lại trong `src/`** sau RENAME (đúng mapping bắt buộc). Không đưa vào `_archive` vì lệnh yêu cầu giữ mọi file cũ bên trong `04_SourceCode`. Pipeline ViVQA-VLM viết mới vào `src/{data,baseline,adaptation,eval,prototype}/`.
3. **`Đề tài đã sắp xếp/`** là kho đề tài khảo sát (không phải working set ViVQA) → MOVE nguyên khối vào `_archive/` như yêu cầu.
4. **Không tạo `R2_*.md`:** spec chỉ liệt kê R1, R3–R7. Related work đặt ở `R3_Existing_Systems.md`. Thesis_Draft_Status ghi chú giả định này.
5. **Reading_Notes:** tạo khung 8 paper cốt lõi (phụ lục Master Instruction) theo template 10 câu hỏi; PDF gốc không move.
6. **`02_References/templates/`** (docx/pdf khoa) khác **`docs/templates/`** (markdown nhóm) — không trộn, không xoá cái cũ.
7. **Chỉ tạo `.md` / `.txt` / `.gitignore` / `.gitkeep`** — không tạo yaml/json mẫu trong `configs/` (README giải thích convention `Wxx_<run>.yaml` khi bắt đầu train).
8. **Nhắc PEFT-only trên tuần có chạy/train model:** W04, W05, W08, W09, W10. Các tuần khác vẫn có mục ràng buộc rút gọn.
9. **Tên thành viên** không có trong repo → dùng SV1–SV4 theo vai trò Master Instruction.
10. **Root `README.md` và `.gitignore` chưa tồn tại** → CREATE hợp lệ (không ghi đè).
11. File note `message.txt` / tutorial links **giữ nguyên tên** khi MOVE vào `docs/notes/`.
12. `_archive/.gitkeep` vẫn tạo dù thư mục không rỗng (vô hại; giúp nhận diện thư mục trên một số tool).

---

## Local merge 2026-09-12

Gộp **local** 2 folder rời rạc → 1 monorepo hoàn chỉnh. Quy tắc: **chỉ ĐỌC + COPY**, không đụng tới `AIP491/` và `ViVQA-VLM/` (2 nguồn nguyên vẹn). Không thao tác git.

- **SRC_A** = `AIP491/` (nguồn bổ sung tài liệu)
- **SRC_B** = `ViVQA-VLM/` (CANONICAL cho code trùng — bản mới hơn)
- **DEST**  = `ViVQA-VLM_monorepo/` (output; đổi tên thành `ViVQA-VLM` sau khi kiểm tra)

### Tóm tắt số lượng

| Hạng mục | Số lượng |
|----------|----------|
| File COPY từ 2 nguồn vào DEST | **412** |
| File quản lý SINH MỚI (README, PROGRESS, 15 tracker, requirements, 03_report/README) | 19 |
| **Tổng file trong DEST** | 431 |
| File XUNG ĐỘT `.from_AIP491` cần merge tay | **8** |
| Xoá/ghi đè trong 2 nguồn | **0** |
| Dung lượng DEST | ~69 MB |

### Quyết định cho tình huống mơ hồ (an toàn, không phá dữ liệu)

1. **`.venv` (2.35 GB, ~51k file) của `production-agentic-rag-course` → KHÔNG copy.** Đây là virtualenv (env ephemeral, đường dẫn tuyệt đối vô nghĩa khi copy), không phải data/adapters/weights. Copy sẽ phá vỡ mục tiêu "sạch, dễ quản lý". Có thể tái tạo bằng `pip install`.
2. **`production-agentic-rag-course/` (code khóa Agentic-RAG cũ, đã bỏ theo proposal) → đưa vào `_archive/production-agentic-rag-course/`** thay vì `src/` (giữ `src/` sạch, chỉ chứa pipeline ViVQA). Đã loại `.venv`, `.git`, `__pycache__`. File `production-agentic-rag_link_github.txt` cũng để trong `_archive/`.
3. **Nested `.git/` (trong `ViVQA-X` và course) + `__pycache__/` + `*.pyc` → KHÔNG copy** (artifact VCS/build, tái tạo được, tránh nested-repo bẩn).
4. **`docs/notes/` + `docs/templates/` của SRC_A → giữ lại trong `docs/`** (không nằm trong spec cây thư mục nhưng là tài liệu có giá trị; bảo toàn để không mất dữ liệu).
5. **Slide fixtures rác** (`default.pptx` của lib `python-pptx`, `test.key` của lib `tornado`) vô tình khớp glob `*.pptx/*.key` bên trong `.venv` → đã **loại khỏi `docs/04_slides/`** (chỉ giữ `ViVQA-VLM_Week1_Progress.pptx` là slide thật). Bản gốc vẫn còn trong nguồn (không đụng).
6. **Weekly:** `W01` lấy SRC_B làm canonical, trộn thêm file của SRC_A (khác nội dung → `.from_AIP491`); `W02..W15` chỉ SRC_A có nên copy nguyên. `tracker.md` sinh mới ở mỗi tuần (tên khác README nên không đè).
7. **`literature/`:** gom paper review (SV1-4) + Literature_Matrix; bản SRC_B (W01) là canonical, bản SRC_A (W02) khác nội dung để cạnh dạng `.from_AIP491` (đồng thời vẫn còn trong `weekly/`).

### Cây thư mục DEST cuối cùng

```text
ViVQA-VLM_monorepo/
|-- _archive/                         # tài liệu khảo sát cũ + code khóa Agentic-RAG (tham khảo)
|   |-- Đề tài đã sắp xếp/            # kho đề tài khảo sát (từ SRC_A/_archive)
|   `-- production-agentic-rag-course/ # code khóa cũ (ĐÃ loại .venv/.git/__pycache__)
|-- adapters/                         # (2) placeholder weights (từ SRC_A)
|-- configs/                          # (3) qwen_lora.yaml (SRC_B) + README/.gitkeep (SRC_A)
|-- data/                             # raw/processed/ocr/splits (SRC_A) + ViVQA-X/ (SRC_B)
|   `-- ViVQA-X/                       # dataset ViVQA-X đầy đủ (data/final/*.json)
|-- docs/
|   |-- 00_admin/  01_proposal/  02_references/  03_report/
|   |-- 04_slides/                    # {review1,review2,faculty_review,final_defense} + Week1 pptx
|   |-- literature/                   # (10) paper reviews SV1-4 + Literature_Matrix (+ .from_AIP491)
|   |-- notes/  templates/            # tài liệu & template nhóm (bảo toàn từ SRC_A)
|   |-- weekly/                        # PROGRESS.md + W01..W15/ (tracker.md + nội dung cũ)
|   |-- master_instruction.md
|   `-- MIGRATION_REPORT.md            # file này
|-- experiments/                      # README (SRC_A) + W01_Pilot/ (SRC_B)
|-- notebooks/                        # 01_zeroshot_baseline_vivqa.ipynb (SRC_B)
|-- src/                               # BASE = SRC_B/src
|   |-- data/ (ocr.py, vivqa_dataset.py)  models/ (vlm_loader.py)  prompting/ (builder.py)
|   |-- eval/ (metrics.py, run_eval.py)   baseline/  adaptation/  prototype/ (scaffold)
|   `-- README.md + README.from_AIP491.md (xung đột)
|-- README.md                         # sinh mới
`-- requirements.txt                  # sinh mới (placeholder)
```

> Cây chi tiết đầy đủ (mọi thư mục + số file): xem phần Appendix bên dưới.

### Danh sách file XUNG ĐỘT (`.from_AIP491`) — CẦN MERGE TAY (8)

Bản chính là bản SRC_B (không hậu tố); bản `.from_AIP491` là của SRC_A đặt cạnh để bạn tự đối chiếu & merge:

| # | File cần merge (đường dẫn trong DEST) |
|---|----------------------------------------|
| 1 | `src/README.from_AIP491.md` ↔ `src/README.md` |
| 2 | `docs/weekly/W01/Project_Plan_15_Weeks.from_AIP491.md` ↔ `Project_Plan_15_Weeks.md` |
| 3 | `docs/weekly/W01/W01_Team_Report.from_AIP491.md` ↔ `W01_Team_Report.md` |
| 4 | `docs/literature/SV1_Paper_Review.from_AIP491.md` ↔ `SV1_Paper_Review.md` |
| 5 | `docs/literature/SV2_Paper_Review.from_AIP491.md` ↔ `SV2_Paper_Review.md` |
| 6 | `docs/literature/SV3_Paper_Review.from_AIP491.md` ↔ `SV3_Paper_Review.md` |
| 7 | `docs/literature/SV4_Paper_Review.from_AIP491.md` ↔ `SV4_Paper_Review.md` |
| 8 | `docs/literature/Literature_Matrix.from_AIP491.md` ↔ `Literature_Matrix.md` |

### Checklist việc cần làm tay sau migration

- [ ] Mở 8 cặp file xung đột ở trên, merge nội dung → giữ bản đúng, **xoá file `.from_AIP491`** sau khi merge.
- [ ] Rà `_archive/production-agentic-rag-course/` — quyết định giữ để tham khảo hay xoá hẳn (code khóa cũ, không thuộc ViVQA).
- [ ] Nếu cần chạy course cũ: tự tạo lại `.venv` bằng `pip install` (venv gốc **không** được copy).
- [ ] Điền `requirements.txt` (pin transformers/peft/bitsandbytes/torch/paddleocr...).
- [ ] Cập nhật `docs/weekly/PROGRESS.md` + `Wxx/tracker.md` theo tiến độ thật.
- [ ] Đồng bộ bản `.tex` từ Overleaf về `docs/03_report/` (xem `docs/03_report/README.md`).
- [ ] Kiểm tra `data/ViVQA-X/` đủ (train/val/test JSON) trước khi train.
- [ ] Sau khi xác nhận DEST OK: **xoá 2 folder cũ** `AIP491/` + `ViVQA-VLM/` → **đổi tên** `ViVQA-VLM_monorepo/` thành `ViVQA-VLM/`.

### Appendix — cây thư mục đầy đủ (mọi folder + số file)

```text
ViVQA-VLM_monorepo/
|-- _archive/ (2 files)
|   |-- Đề tài đã sắp xếp/
|   |   |-- 00_Tong_Hop_Bao_Cao/ (1 files)
|   |   |-- 01_Rat_Kha_Thi/ (2 files)
|   |   |-- 02_Kha_Thi/ (8 files)
|   |   |-- 03_Trung_Binh/ (8 files)
|   |   |-- 04_Kho_Nang/ (7 files)
|   |   `-- 05_De_Tai_Y_Khoa/ (7 files)
|   `-- production-agentic-rag-course/ (13 files)
|       |-- airflow/ (4 files)
|       |   |-- dags/ (2 files)
|       |   |   `-- arxiv_ingestion/ (6 files)
|       |   `-- plugins/
|       |-- notebooks/
|       |   |-- week1/ (2 files) ... week7/ (2 files)
|       |-- src/ (7 files)
|       |   |-- db/ · models/ · repositories/ · routers/ (5) · schemas/ · services/
|       |-- static/ (9 files)
|       `-- tests/ (2 files)
|-- adapters/ (2 files)
|-- configs/ (3 files)
|-- data/ (1 files)
|   |-- ocr/ (1)  processed/ (1)  raw/ (1)  splits/ (1)
|   `-- ViVQA-X/ (4 files)
|       |-- assets/ (1)
|       |-- data/final/ (3 files)          # ViVQA-X_train/val/test.json
|       |-- scripts/ (4)
|       `-- src/
|           |-- models/ (baseline_model, heuristic_model)
|           `-- pipeline/ (post_processing, selection/evaluators, translation/translators)
|-- docs/ (2 files)
|   |-- 00_admin/ (3)  01_proposal/ (2, +_archive/ 2)  02_references/ (3, +Reading_Notes/ 9, +templates/ 2)
|   |-- 03_report/ (8 files)
|   |-- 04_slides/ (1 file: Week1 pptx)  {faculty_review, final_defense, review1, review2}/ (empty)
|   |-- literature/ (10 files)
|   |-- notes/ (6)  templates/ (4)
|   `-- weekly/ (1 file: PROGRESS.md)
|       |-- W01/ (18)  W02/ (13)  W03/ (5)  W04/ (5)  W05/ (4)  W06/ (4)  W07/ (6)
|       |-- W08/ (4)  W09/ (4)  W10/ (4)  W11/ (4)  W12/ (6)  W13/ (5)  W14/ (5)  W15/ (5)
|-- experiments/ (2 files)
|   `-- W01_Pilot/ (1 file)
|-- notebooks/ (1 file)
`-- src/ (3 files: README.md, README.from_AIP491.md, __init__.py)
    |-- adaptation/ (2)  baseline/ (2)  data/ (5)  eval/ (5)
    |-- models/ (2)  prompting/ (2)  prototype/ (2)
+ root files: README.md, requirements.txt
```

