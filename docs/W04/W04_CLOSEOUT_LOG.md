# W04 CLOSEOUT LOG — Placement audit & dọn repo

Thời điểm: 2026-09-21 (00:31–00:39 UTC+7).  
Cwd: `D:\FPT_university\termno9\ViVQA-VLM`.  
Nguồn: báo cáo agent closeout Week 4 (audit + đặt chỗ + `.gitignore` + đề xuất dọn notebook).

Ràng buộc đã tuân: **không git**; **không xóa** `data/`, `adapters/`, JSON log; **không sửa** nội dung `.md` người dùng đã tạo; chỉ di chuyển khi đủ file / đã duyệt.

---

## Nhiệm vụ 1 — Audit

`docs/W04/` **có**. Hiện chỉ chứa notebook Colab + log thí nghiệm (giải nén Drive), **không** có `README.md` / `W04_Team_Report.md` / `W04_Baseline_Report.md`.

`notebooks/01_zeroshot_baseline.ipynb` **không có**. Trong `notebooks/` chỉ có `01_zeroshot_baseline_vivqa.ipynb` (scaffold cũ). Notebook Colab thật nằm ở `docs/W04/01_zeroshot_baseline.ipynb`.

`experiments/W04_zeroshot_qwen25vl_seed42/` **không có**. Bốn JSON **có đủ** nhưng đang lồng sai chỗ (thư mục zip Google Drive):

`docs/W04/experiments/W04_zeroshot_qwen25vl_seed42-20260920T171044Z-1-001/W04_zeroshot_qwen25vl_seed42/`  
→ `env.json`, `zeroshot_sample.json`, `zeroshot_vitextvqa_n20.json`, `zeroshot_vitextvqa_n300_seed42.json`

Đã quét repo, Downloads, Desktop, Documents, OneDrive, `D:\FPT_university\termno9`. **Không** tìm thấy `W04_README.md`, `W04_Team_Report.md`, `W04_Baseline_Report.md` (bản vừa viết). Trong `_archive/` chỉ có **template trống** W04 ngày 3/9 — không dùng.

| File | Vị trí hiện tại | Vị trí đúng | Khớp/lệch |
|---|---|---|---|
| `W04_README.md` | **không thấy** (ngoài template archive) | `docs/W04/README.md` | **thiếu** |
| `W04_Team_Report.md` | **không thấy** (ngoài template archive) | `docs/W04/W04_Team_Report.md` | **thiếu** |
| `W04_Baseline_Report.md` | **không thấy** | `docs/W04/W04_Baseline_Report.md` | **thiếu** |
| `Glossary.md` | `docs/Glossary.md` | `docs/Glossary.md` | **khớp** |
| `Review_W01-W04_and_Defense_Prep.md` | `docs/Review_W01-W04_and_Defense_Prep.md` | `docs/Review_W01-W04_and_Defense_Prep.md` | **khớp** |
| `01_zeroshot_baseline.ipynb` | `docs/W04/01_zeroshot_baseline.ipynb` | `notebooks/01_zeroshot_baseline.ipynb` | **lệch** |
| `env.json` + 3 log zeroshot | `docs/W04/experiments/.../W04_zeroshot_qwen25vl_seed42/` | `experiments/W04_zeroshot_qwen25vl_seed42/` | **lệch** |

---

## Kế hoạch đã chốt (trước khi sửa)

1. **Không** di chuyển 2 file đã đúng chỗ. **Không** lấy template archive. **Không** tạo `.md` thay thế.
2. **Không** dời notebook/log thí nghiệm — không nằm trong danh sách N2; chờ duyệt.
3. **Chỉ** bổ sung dòng còn thiếu trong `.gitignore`.
4. Đọc notebook ở `docs/W04/` để **đề xuất** dọn code; **không refactor**.

---

## Nhiệm vụ 2 — Đặt đúng chỗ

Bỏ qua: `Glossary.md`, `Review_W01-W04_and_Defense_Prep.md` (đã đúng).  
Không làm được: 3 file W04 `.md` — chưa có trên máy.

## Nhiệm vụ 3 — `.gitignore`

Trước chỉ có `__pycache__/`, `*.pyc`. Đã **thêm**: `data/`, `adapters/`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.zip`, `.ipynb_checkpoints/`.

## Nhiệm vụ 4 — Dọn code (đề xuất, chưa làm)

Notebook canonical `notebooks/01_zeroshot_baseline.ipynb` **chưa có**. Đã đọc bản Colab tại `docs/W04/01_zeroshot_baseline.ipynb` (11 cell).

Trùng lặp rõ: `normalize` / `lev` / `em` / `anls` copy nguyên ở cell 8 (n=20) và cell 10 (n=300). Cell 2 và 3 gần như cùng logic scene-text. Cell 6 join COCO-style `images`+`annotations` → `test_flat.json`. `src/eval/metrics.py` **đã có** `normalize_vi`, `exact_match`, `anls`, `evaluate_predictions`.

**Kế hoạch refactor (chờ duyệt):**

- Gom `normalize`/`em`/`anls`/`lev` → import `src.eval.metrics` (không giữ bản copy trong notebook).
- Gom join + flatten ViTextVQA → `src/data/` (vd. `prepare_vitextvqa.py`); notebook chỉ gọi hàm.
- Notebook mỏng: setup Colab + load model + loop infer + gọi metric + ghi JSON.
- **Không** đụng `notebooks/01_zeroshot_baseline_vivqa.ipynb` trừ khi được bảo (scaffold cũ, URL `ViVOA-VLM`).

Chưa refactor.

---

## Nhiệm vụ 5 — Báo cáo cuối

**`docs/W04/`** (không đổi sau dọn):

```
docs/W04/
  01_zeroshot_baseline.ipynb
  experiments/
    W04_zeroshot_qwen25vl_seed42-20260920T171044Z-1-001.zip
    W04_zeroshot_qwen25vl_seed42-20260920T171044Z-1-001/
      W04_zeroshot_qwen25vl_seed42/
        env.json
        zeroshot_sample.json
        zeroshot_vitextvqa_n20.json
        zeroshot_vitextvqa_n300_seed42.json
```

**`notebooks/`:** `01_zeroshot_baseline_vivqa.ipynb`  
**`experiments/`:** `W01_Pilot/`, `W02_dataset_stats/` — **chưa có** `W04_*`

| Đã làm | Đã bỏ qua | Còn thiếu (cần xử lý tay) |
|---|---|---|
| Audit vị trí file | Di chuyển 3 file W04 `.md` (không có) | Thả `W04_README.md`, `W04_Team_Report.md`, `W04_Baseline_Report.md` vào repo |
| Bỏ qua 2 `.md` đã đúng chỗ | Dời notebook + JSON log (chưa duyệt) | Duyệt: copy/move 4 JSON → `experiments/W04_zeroshot_qwen25vl_seed42/` |
| Cập nhật `.gitignore` | Refactor notebook | Duyệt: `docs/W04/01_zeroshot_baseline.ipynb` → `notebooks/` |
| Đề xuất dọn code | Dùng template `_archive` | Duyệt kế hoạch refactor N4 |
| | Git | Zip Drive trong `docs/W04/experiments/` — giữ hay dời? |

**Xác nhận:** không chạy git; không xóa `data/`, `adapters/`, hay JSON log. Chỉ sửa `.gitignore`.

---

## Cần duyệt tiếp (từ báo cáo gốc)

1. Dời notebook: `docs/W04/01_zeroshot_baseline.ipynb` → `notebooks/01_zeroshot_baseline.ipynb`
2. Dời 4 JSON log → `experiments/W04_zeroshot_qwen25vl_seed42/`
3. Refactor notebook theo kế hoạch N4
4. Cung cấp 3 file W04 `.md` còn thiếu rồi đặt vào `docs/W04/`

---

# Lần 2 — Đã duyệt dời file + đối chiếu metric (2026-09-21, ~00:45–00:53 UTC+7)

Đã xong các bước được duyệt (trừ thay metric — logic **không hoàn toàn giống** nên đã dừng đúng luật). Không git, không xóa nguồn JSON.

## Trước / sau (ngắn)

**Duyệt 1 — notebook.** Không trùng tên với `01_zeroshot_baseline.ipynb`. Scaffold `01_zeroshot_baseline_vivqa.ipynb` (10 953 bytes) **giữ nguyên**.  
Đã dời: `docs/W04/01_zeroshot_baseline.ipynb` → `notebooks/01_zeroshot_baseline.ipynb`.

**Duyệt 2 — COPY 4 JSON** (SHA256 nguồn = đích; nguồn **còn** trong `docs/W04/experiments/...`):

| File | Bytes |
|---|---|
| `env.json` | 289 |
| `zeroshot_sample.json` | 569 |
| `zeroshot_vitextvqa_n20.json` | 3979 |
| `zeroshot_vitextvqa_n300_seed42.json` | 53846 |

Đích: `experiments/W04_zeroshot_qwen25vl_seed42/`.

**Bạn muốn xóa folder zip Google Drive + thư mục lồng trong `docs/W04/experiments/` không?** Nguồn vẫn nguyên; chỉ xóa khi bạn nói có.

**Duyệt 3 — 3 file `.md`**

| File | Có? |
|---|---|
| `docs/W04/W04_Team_Report.md` | có |
| `docs/W04/W04_Baseline_Report.md` | có |
| `docs/W04/README.md` | **thiếu** — không tự tạo |

**Duyệt 4 — đối chiếu metric → DỪNG thay import**

Trên log W4 thật: **n=20 và n=300, EM/ANLS notebook = `src.eval.metrics` (0 mẫu lệch)**.  
EM n=300 = 38.3%, ANLS = 55.9% — khớp báo cáo.

Khác ở **2 biên**, nên chưa thay `from src.eval.metrics import ...`:

| Case | Chuẩn hóa | EM nb / src | ANLS nb / src |
|---|---|---|---|
| Câu tiếng Việt thường, punct, dấu, NFC | giống | giống | giống |
| Ngưỡng ANLS 0.5 | giống | — | giống |
| `pred=""`, `gold=[""]` | `""` / `""` | 1 / 1 | **1.0 / 0.0** |
| `pred=None` | `"none"` / `""` | **1 / 0** (nếu gold `"none"`) | khác hệ quả |

Notebook: `max(len, 1)` → hai chuỗi rỗng coi là khớp ANLS.  
`src`: `denom=0` → similarity 0.  
`None`: notebook `str(None)` → `"none"`; `src` trả `""`.

**Giữ bản nào?** (a) notebook cell 8/10, (b) `src/eval/metrics.py`, rồi mới thay import.

Phần flatten **đã làm**: `src/data/prepare_vitextvqa.py` (smoke test OK). Notebook cell 0 thêm `sys.path`; cell 6 gọi `find_image_root` + `flatten_coco_file`. Cell 8/10 vẫn copy `normalize/lev/em/anls`. Không chạy được cell Colab/GPU trên máy này; đã parse syntax cell 6/8/10.

---

## Cây sau khi dời

```
notebooks/
  01_zeroshot_baseline.ipynb          ← Colab W4
  01_zeroshot_baseline_vivqa.ipynb    ← scaffold cũ, không đụng

experiments/
  W01_Pilot/
  W02_dataset_stats/
  W04_zeroshot_qwen25vl_seed42/
    env.json
    zeroshot_sample.json
    zeroshot_vitextvqa_n20.json
    zeroshot_vitextvqa_n300_seed42.json

docs/W04/
  W04_Team_Report.md
  W04_Baseline_Report.md
  W04_CLOSEOUT_LOG.md
  experiments/                        ← BẢN SAO nguồn, chưa xóa
    W04_zeroshot_qwen25vl_seed42-...-001.zip
    W04_zeroshot_qwen25vl_seed42-...-001/
      W04_zeroshot_qwen25vl_seed42/
        (4 json giống trên)
```

| Đã làm | Chờ bạn |
|---|---|
| Dời notebook; không ghi đè scaffold | Thả `docs/W04/README.md` |
| COPY 4 JSON + hash khớp | Có/không xóa zip + thư mục lồng `docs/W04/experiments/` |
| Xác nhận 2/3 file `.md` | (a) hay (b) cho EM/ANLS rồi mới thay import |
| Tách flatten; notebook gọi module | Chạy lại notebook trên Colab (GPU) |

**Xác nhận:** không git; không xóa `data/`, `adapters/`, JSON log.

---

# Lần 3 — Metric (b) + dọn cruft + kiểm README (2026-09-21, ~01:02 UTC+7)

Đã duyệt: chọn **(b)** `src/eval/metrics.py` làm nguồn DUY NHẤT; xóa bản thừa trong `docs/W04/experiments/`. Không git; không sửa `src/eval/metrics.py`; không xóa `data/`, `adapters/`, hay 4 JSON ở `experiments/W04_zeroshot_qwen25vl_seed42/`. Không chạy GPU / không chạy notebook.

## 1) METRIC — chọn (b)

Notebook `notebooks/01_zeroshot_baseline.ipynb`: bỏ copy `normalize` / `lev` / `em` / `anls` ở **cell 8** (n=20) và **cell 10** (n=300). Thay bằng:

```python
from src.eval.metrics import normalize_vi, exact_match, anls, evaluate_predictions
```

Không sửa `src/eval/metrics.py`.

**2 khác biệt biên — KNOWN, không xảy ra trên dữ liệu thật:**

| Case | Notebook copy (cũ) | `src.eval.metrics` (b) |
|---|---|---|
| `pred=""`, `gold=[""]` (rỗng-vs-rỗng) | ANLS **1.0** (`max(len, 1)` → coi khớp) | ANLS **0.0** (`denom=0` → similarity 0) |
| `pred=None` | `str(None)` → `"none"` → EM **1** nếu gold `"none"` | `normalize_vi(None)` → `""` → EM **0** |

Trên log W4 thật: **n=300, lệch 0 mẫu** (không có pred rỗng / pred=None). → **baseline 38.3 / 55.9 không đổi.**

Kiểm trên máy này: import `src.eval.metrics` OK; `ast.parse` cell 8 và cell 10 OK. **Không chạy GPU / không chạy notebook** → không khẳng định “đã chạy được”. **Cần chạy lại notebook trên Colab** để xác nhận cell 8/10 với `sys.path` + JSON log.

## 2) DỌN CRUFT

Hash SHA256 4 JSON `experiments/W04_zeroshot_qwen25vl_seed42/` = bản lồng `docs/W04/experiments/...-001/` (khớp từng file). Đã xóa **chỉ** trong `docs/W04/experiments/`:

- `W04_zeroshot_qwen25vl_seed42-20260920T171044Z-1-001.zip`
- thư mục lồng `W04_zeroshot_qwen25vl_seed42-20260920T171044Z-1-001/`

Không đụng `experiments/` ở root. 4 JSON canonical còn nguyên (289 / 569 / 3979 / 53846 bytes).

## 3) KIỂM README

`docs/W04/README.md` — **thiếu** (không tự tạo).  
Có `docs/W04/W04_README.md` (index tuần 4, vừa thả). Đổi tên `W04_README.md` → `README.md` khi bạn muốn.

## Trước / sau

**Trước** `docs/W04/` còn `experiments/` (zip + thư mục lồng 4 JSON). Notebook cell 8/10 copy hàm metric.

**Sau:**

```
docs/W04/
  W04_Baseline_Report.md
  W04_CLOSEOUT_LOG.md
  W04_README.md              ← có; README.md thì không
  W04_Team_Report.md

notebooks/
  01_zeroshot_baseline.ipynb          ← cell 8/10 import src.eval.metrics
  01_zeroshot_baseline_vivqa.ipynb    ← không đụng

experiments/W04_zeroshot_qwen25vl_seed42/
  env.json
  zeroshot_sample.json
  zeroshot_vitextvqa_n20.json
  zeroshot_vitextvqa_n300_seed42.json
```

| Đã làm | Không làm / chờ bạn |
|---|---|
| Cell 8/10 → import (b); syntax + import OK | Chạy lại notebook trên **Colab** (GPU) để xác nhận |
| Xóa zip + thư mục lồng `docs/W04/experiments/` | `docs/W04/README.md` vẫn thiếu — có `W04_README.md` |
| Ghi log chọn (b) + 2 biên KNOWN | Git; sửa `metrics.py`; xóa `data/` / `adapters/` / JSON root |

**Xác nhận:** không git; không xóa `data/`, `adapters/`, hay 4 JSON trong `experiments/W04_zeroshot_qwen25vl_seed42/`.


---

# Lần 3 — Metric (b) + dọn cruft + kiểm README (2026-09-21, ~01:02 UTC+7)

Đã duyệt: chọn **(b)** `src/eval/metrics.py` làm nguồn DUY NHẤT; xóa bản thừa trong `docs/W04/experiments/`. Không git; không sửa `src/eval/metrics.py`; không xóa `data/`, `adapters/`, hay 4 JSON ở `experiments/W04_zeroshot_qwen25vl_seed42/`. Không chạy GPU / không chạy notebook.

## 1) METRIC — chọn (b)

Notebook `notebooks/01_zeroshot_baseline.ipynb`: bỏ copy `normalize` / `lev` / `em` / `anls` ở **cell 8** (n=20) và **cell 10** (n=300). Thay bằng:

```python
from src.eval.metrics import normalize_vi, exact_match, anls, evaluate_predictions
```

Không sửa `src/eval/metrics.py`.

**2 khác biệt biên — KNOWN, không xảy ra trên dữ liệu thật:**

| Case | Notebook copy (cũ) | `src.eval.metrics` (b) |
|---|---|---|
| `pred=""`, `gold=[""]` (rỗng-vs-rỗng) | ANLS **1.0** (`max(len, 1)` → coi khớp) | ANLS **0.0** (`denom=0` → similarity 0) |
| `pred=None` | `str(None)` → `"none"` → EM **1** nếu gold `"none"` | `normalize_vi(None)` → `""` → EM **0** |

Trên log W4 thật: **n=300, lệch 0 mẫu** (không có pred rỗng / pred=None). → **baseline 38.3 / 55.9 không đổi.**

Kiểm trên máy này: import `src.eval.metrics` OK; `ast.parse` cell 8 và cell 10 OK. **Không chạy GPU / không chạy notebook** → không khẳng định “đã chạy được”. **Cần chạy lại notebook trên Colab** để xác nhận cell 8/10 với `sys.path` + JSON log.

## 2) DỌN CRUFT

Hash SHA256 4 JSON `experiments/W04_zeroshot_qwen25vl_seed42/` = bản lồng `docs/W04/experiments/...-001/` (khớp từng file). Đã xóa **chỉ** trong `docs/W04/experiments/`:

- `W04_zeroshot_qwen25vl_seed42-20260920T171044Z-1-001.zip`
- thư mục lồng `W04_zeroshot_qwen25vl_seed42-20260920T171044Z-1-001/`

Không đụng `experiments/` ở root. 4 JSON canonical còn nguyên (289 / 569 / 3979 / 53846 bytes).

## 3) KIỂM README

`docs/W04/README.md` — **thiếu** (không tự tạo).  
Có `docs/W04/W04_README.md` (index tuần 4, vừa thả). Đổi tên `W04_README.md` → `README.md` khi bạn muốn.

## Trước / sau

**Trước** `docs/W04/` còn `experiments/` (zip + thư mục lồng 4 JSON). Notebook cell 8/10 copy hàm metric.

**Sau:**

```
docs/W04/
  W04_Baseline_Report.md
  W04_CLOSEOUT_LOG.md
  W04_README.md              ← có; README.md thì không
  W04_Team_Report.md

notebooks/
  01_zeroshot_baseline.ipynb          ← cell 8/10 import src.eval.metrics
  01_zeroshot_baseline_vivqa.ipynb    ← không đụng

experiments/W04_zeroshot_qwen25vl_seed42/
  env.json
  zeroshot_sample.json
  zeroshot_vitextvqa_n20.json
  zeroshot_vitextvqa_n300_seed42.json
```

| Đã làm | Không làm / chờ bạn |
|---|---|
| Cell 8/10 → import (b); syntax + import OK | Chạy lại notebook trên **Colab** (GPU) để xác nhận |
| Xóa zip + thư mục lồng `docs/W04/experiments/` | `docs/W04/README.md` vẫn thiếu — có `W04_README.md` |
| Ghi log chọn (b) + 2 biên KNOWN | Git; sửa `metrics.py`; xóa `data/` / `adapters/` / JSON root |

**Xác nhận:** không git; không xóa `data/`, `adapters/`, hay 4 JSON trong `experiments/W04_zeroshot_qwen25vl_seed42/`.

