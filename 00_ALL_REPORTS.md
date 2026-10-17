# 00_ALL_REPORTS — ViVQA-VLM progress dossier (auto-generated)

Built from `main` @ `25a343d` on 2026-10-17T08:28Z. Each section starts with its source path.

---

# FILE: docs/00_admin/AI_setup/Gemini_Context_Pack.md

# ViVQA-VLM — Context Pack cho AI trợ lý (cập nhật 17/10/2026, sau khi đóng W09)

> File kiến thức dùng chung cho Gemini (và bất kỳ AI nào). Khi có handoff mới hơn, handoff thắng.
> Chỉ chứa thông tin đã chốt; mục nào chưa chắc ghi "(cần xác minh)".

## 1. Người dùng & dự án
- Hoàng (Phạm Lê Huy Hoàng), SV ĐH FPT, trưởng nhóm 4 người; vai trò chính thức SV3 PEFT/Training nhưng tự làm toàn bộ pipeline rồi chuyển giao cho nhóm.
- Đề tài: "Efficient Vietnamese Visual Question Answering using Vision-Language Models" (ViVQA-VLM), KLTN AIP491, chuyên ngành NLP.
- GVHD: PGS. Đặng Ngọc Minh Đức (thầy Đức) và thầy Phạm Minh Trí.
- Vai trò nhóm (để hiểu bối cảnh, không để phân việc): VLM/Baseline · Dataset/OCR · PEFT/Training (Hoàng) · Evaluation/Deploy.

## 2. Ràng buộc đề tài (từ Master Instruction + đăng ký đề tài)
- Reproduce ĐÚNG 1 baseline trước khi cải tiến → **Qwen2.5-VL-3B-Instruct** (GVHD xác nhận 07/10/2026).
- Cải tiến chính đã kiểm chứng: **OCR-aware QLoRA (B3)** — QLoRA train với dòng OCR trong prompt, dùng OCR ở CẢ lúc train và lúc test. Chỉ thêm OCR vào prompt lúc suy luận thì KHÔNG giúp (bảng 2×2, W09).
- CHỈ PEFT (LoRA/QLoRA). Không full fine-tuning.
- Metric chính EM, VQA-Acc, ANLS (token-F1 là phụ). Bắt buộc ablation, error analysis, hallucination analysis.
- Datasets: ViVQA (general) và ViTextVQA (scene-text, dùng repo HF CHÍNH THỨC có gating `minhquan6203/ViTextVQA`; đáp án test lấy từ `ViTextVQA_test_gt.json`).
- Tiền xử lý tiếng Việt: chuẩn hoá Unicode NFC.

## 3. Lịch thực
- Tuần thực 1 = T2 07/09/2026. Tuần ảo trong chat/repo chạy trước lịch (W09 ảo đã đóng 17/10, trong tuần thực 6).
- Review 1 xong. **Review 2 = tuần thực 7: 19–25/10/2026.** Hội đồng Khoa = 30/11–06/12. Final = 14–20/12.
- Review 1, 2 do 2 GV phản biện (không phải GVHD). Slide tiếng Anh, nói tiếng Việt. Sau mỗi review điền checklist AI_Capstone_Review123, gửi mail GVHD + phản biện, upload Drive.
- Slide Review 2: 20 trang (EN + TV) trong `docs/W07/` (Part 06 = OCR-aware QLoRA).

## 4. Cấu hình chốt
- Qwen2.5-VL-3B-Instruct · bf16 · 4-bit NF4 + double quant · max_pixels 401.408 (≈512 visual token) · greedy, max_new_tokens 32 · seed 42 · GPU L4 (Colab Pro).
- QLoRA: r=16, α=32, target q/k/v/o, 7.372.800 tham số train (0,196%), lr 2e-4, batch hiệu dụng 16, warmup 3%, checkpoint mỗi 200 step. ViTextVQA: 1 epoch. **ViVQA: B2 chính thức = C3 (lịch 2 epoch, lấy epoch 2)**.
- OCR đóng băng: PP-OCRv6_medium_det + VietOCR vgg_transformer · template v1 ("Văn bản đọc được trong ảnh: …") · max_chars 600 · min_conf 0. Cache `data/vitextvqa_official/ocr/paddledet_vietocr.json` (4.422 ảnh W08 + 1.867 ảnh full test W09; phủ toàn bộ 3.353 ảnh test).
- Tập ViTextVQA: test2000 (2.000 câu, seed 42) cho ablation; full test 10.028 câu cho kết quả chính; dev300 để chọn cấu hình; train subset 8.000 câu / 2.670 ảnh (seed 42). Seed train kiểm ổn định: 42/43/44.

## 5. Kết quả đã có (KHÔNG làm tròn lại, không suy diễn thêm)
| Run | Tập | EM | ANLS | Ghi chú |
|---|---|---|---|---|
| B1 zero-shot | ViVQA test 3.001 | 0,2209 | 0,2861 | |
| B2 QLoRA 1 ep (W07) | ViVQA test 3.001 | 0,7378 | 0,7566 | train 3,03 h, VRAM 6,62 GB |
| **B2 ViVQA C3 (2 ep, W09)** | ViVQA test 3.001 | **0,7664** | +0,024 so với W07 | +2,86 EM, p=4,6e-9; chọn trên val sạch; không do nhớ ảnh |
| B1 / B2 | ViTextVQA full 10.028 | 29,12 / 50,26 | 47,60 / 72,02 | |
| B2-sub / **B3** | ViTextVQA full 10.028 | 46,56 / **49,10** | 68,90 / **70,47** | +2,54 EM, p=1,1e-14; B3 > ViTextBLIP-2 (EM 25,48, F1 53,95) |
| B2-sub / B3, 3 seed | test2000 | 45,45±0,66 / 49,02±0,23 | ΔANLS +0,0155±0,0040 | ΔEM +3,57±0,76; 3/3 seed p≤1,4e-4 → CẢI THIỆN ỔN ĐỊNH |
| 2×2 OCR train×test | test2000 seed 42 | chỉ test −0,65 · chỉ train −0,55 · cả hai +4,35 | | tương tác +4,9 EM CI95 [+3,2; +6,6] |
| B3 bỏ OCR lúc test | test2000 | 0,4500 | | −4,25 EM, p=3,8e-9 → B3 phụ thuộc OCR |
| B1 / B1+OCR (inference only) | test2000 | 0,2880 / 0,2935 | 0,4753 / 0,4802 | p=0,547, không khác biệt |
| R0 / R0b | TextVQA val 5.000 | VQA-Acc 79,60 / 75,21 | | tái lập đạt (paper 79,3) |
| R1 Qwen2-VL-7B | ViTextVQA 2.000 | 16,7 | | kém 3B do trả lời dài |
| FS5 few-shot | ViVQA / ViTextVQA | 20,03 / 27,60 | | không giúp |

- **Đính chính W08:** B3 thấp hơn B2 dữ liệu đầy đủ 1,16 EM trên full test (p=0,0025). Phát biểu đúng: với 23% dữ liệu, 25% thời gian train, OCR bù ≈ 69% khoảng cách EM do cắt dữ liệu. KHÔNG nói "B3 ngang B2 full".
- H6 (OCR tăng ANLS ≥ +3 và cộng hưởng QLoRA): **ủng hộ một phần** (cộng hưởng mạnh; ANLS chỉ +1,55).
- Rò rỉ ViVQA đã kiểm (tập clean 1.219 câu). Error audit 200 lỗi B2: đúng ý 54%, hallucination 7% [4,2; 11,4], gold nhiễu 12,5%; ViTextVQA: lỗi OCR 28%, nhầm vùng chữ 23%, dấu thanh 8%.
- OCR dev300: PaddleOCR làm rơi dấu tiếng Việt; VietOCR giữ dấu tốt hơn (McNemar p=0,0008) → bằng chứng gap G2.

## 6. Việc đang mở (đầu W10 ảo · tuần thực 7)
1. Review 2: tập nói slide 20 trang (≈ 22–26 phút); sau buổi review điền checklist, gửi mail, upload Drive.
2. B3 trên TOÀN BỘ train ViTextVQA (≈ 13 h L4, ≈ 20 CU) — OCR-aware có vượt B2 khi đủ dữ liệu không.
3. Ablation cấu hình LoRA: rank, module gắn LoRA (H3, H4).
4. R6 Results bản tiếng Việt cho phần OCR.
- Giới hạn cần nhớ: B3 phụ thuộc OCR → prototype phải chạy OCR trước, hiển thị OCR, cảnh báo khi OCR rỗng.
- Colab: L4 ≈ 1,54 CU/giờ; còn ≈ 84 CU (17/10, ước tính).

## 7. Repo & quy ước
- Local: `D:\FPT_university\termno9\ViVQA-VLM` · GitHub public `github.com/PLHGNAOH/ViVQA-VLM` · Drive `G:\My Drive\ViVQA-VLM` = `/content/drive/MyDrive/ViVQA-VLM` trên Colab.
- `docs/W0X/` (README + W0X_Team_Report.md), `docs/02_references/` (Literature_Matrix bản sống duy nhất), `docs/03_report/` (R1..R7), `docs/04_slides/`, `docs/00_admin/AI_setup/` (file setup AI; `handoffs/` bị gitignore), `src/{data,baseline,adaptation,eval,prototype,models,prompting}`, `experiments/W0X_*`, `adapters/`, `configs/`, `notebooks/NN_topic_model.ipynb`.
- Quy trình: AI soạn → Cursor kiểm/chép/selftest/commit/push, ghi `hand_off_to_claude.md` → AI kiểm lại trên GitHub → Colab (mở notebook từ GitHub, Run all) → Cursor chép kết quả từ Drive vào `experiments/`.
- Luật kết luận ghi vào README và commit TRƯỚC khi có số. Không chạy thêm seed để "câu" kết quả.
- **Đóng tuần** chỉ xong khi đã dọn Downloads + thư mục Drive dự án (qua prompt Cursor, xoá vào thùng rác, chỉ xoá thứ đã có bản ở nơi khác).
- Bài học: paddlepaddle-gpu ghi đè NCCL của torch → phải "Disconnect and delete runtime"; VietOCR cài `--no-deps`; PowerShell không phân biệt hoa/thường tên biến; git pathspec wildcard kết thúc `/*`; không mở menu runtime khi đang train; console Windows chạy với `PYTHONIOENCODING=utf-8`.

## 8. Khuôn handoff (cả Claude và Gemini đều viết theo khuôn này)
```
# Wxx — Bàn giao (<MỞ/ĐÃ ĐÓNG> dd/mm/yyyy hh:mm) — viết bởi <Claude|Gemini>
## Lịch tuần thực
## Trạng thái repo (GitHub main = <hash>)
| Commit | Nội dung |
## Cấu hình chốt (chỉ ghi nếu đổi)
## Kết quả mới (bảng; ghi rõ run, tập, seed, GPU, thời gian)
## Quyết định đã chốt trong phiên
## Việc mang sang phiên sau (có ước lượng giờ)
## Nguồn gốc (AI nào sinh số/nhãn/code nào)
```

---

# FILE: docs/W01/README.md

# W01 — Orientation & planning

## 1. Objective

Định hướng nghiên cứu và lập kế hoạch 15 tuần: hiểu bài toán Vietnamese VQA, ràng buộc PEFT-only, phân công SV1–SV4, và chốt milestone Review 1 / Review 2 / Faculty Review / Final Defense.

## 2. Deliverable / Milestone

W01 Team Report · Project_Plan_15_Weeks.md · Research_Orientation.md · phân công cá nhân rõ ràng.

File đặc thù tuần này: `Project_Plan_15_Weeks.md, Research_Orientation.md` · báo cáo nhóm: `W01_Team_Report.md`.

## 3. Task theo từng SV

### SV1 — VLM / Baseline
*Vai trò:* Hiểu & reproduce 1 VLM baseline (BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL); kiến trúc vision encoder → VL interface → LM.
- [ ] Tóm tắt kiến trúc VLM (vision encoder, projector/Q-Former, LM) và 3 ứng viên baseline: BLIP-2, Florence-2, Qwen2.5-VL.
- [ ] Đề xuất 1 baseline sẽ reproduce (kèm lý do VRAM / giấy phép / code có sẵn).

### SV2 — Dataset / OCR
*Vai trò:* ViVQA + ViTextVQA; normalize; PaddleOCR / VietOCR; question-type tagging; split cố định (ghi seed).
- [ ] Rà soát ViVQA và ViTextVQA (licence, kích thước, split gốc, đặc thù scene-text).
- [ ] Phác thảo pipeline OCR (PaddleOCR vs VietOCR) và nơi lưu `data/raw`, `data/ocr`.

### SV3 — PEFT / Training
*Vai trò:* LoRA / QLoRA, prompting, OCR-enhanced prompt, RAG; log hyperparameter + thời gian train.
- [ ] Đọc LoRA/QLoRA; liệt kê ràng buộc: không full FT, 4/8-bit, Colab Pro/Premium hoặc GPU trường.
- [ ] Phác thảo không gian phương pháp: PEFT · OCR-prompt · RAG · prompt-opt.

### SV4 — Evaluation / Deployment
*Vai trò:* EM, VQA Accuracy, ANLS; hallucination; prototype web; experiment log.
- [ ] Chốt metric chính: EM, VQA Accuracy, ANLS; BLEU/CIDEr chỉ phụ.
- [ ] Phác thảo experiment log (seed, split, hardware, thời gian) và yêu cầu prototype (upload ảnh + hỏi tiếng Việt).

## 4. Checklist

- [ ] Cả nhóm đọc `docs/Master_Instruction.md` (Mục 4 ràng buộc + Mục 7 evaluation).
- [ ] Đã điền Project_Plan_15_Weeks.md (gantt logic, owner từng deliverable).
- [ ] Đã điền Research_Orientation.md (bài toán, gap sơ bộ, rủi ro ≥ 5).
- [ ] Phân công SV1–SV4 không chồng lấn trách nhiệm kỹ thuật.
- [ ] Nộp W01_Team_Report.md.

## 5. Definition of Done

Nhóm giải thích được bài toán, ràng buộc PEFT-only, 4 milestone, và mỗi SV có task cá nhân + deliverable tuần 1.

## 6. Ràng buộc (nhắc lại)

Tuần này trọng tâm không phải train. Khi có huấn luyện ở tuần khác: **CHỈ PEFT (LoRA/QLoRA 4/8-bit), KHÔNG full fine-tuning.** Cải tiến luôn so baseline cùng split/seed/hardware protocol. Chi tiết: `docs/master_instruction.md` Mục 4 và Mục 7.

## Liên kết

- Master: [`docs/master_instruction.md`](../master_instruction.md)
- Template báo cáo: [`docs/templates/Weekly_Report_Template.md`](../templates/Weekly_Report_Template.md)
- Experiment log: [`docs/templates/Experiment_Log_Template.md`](../templates/Experiment_Log_Template.md)

---

# FILE: docs/W01/W01_Team_Report.md

# W01 Team Report — Research Orientation & Planning

> Điền theo `docs/templates/Weekly_Report_Template.md`. Không để trống các mục số liệu nếu tuần có thí nghiệm.

## Metadata

| Trường | Nội dung |
|--------|----------|
| **Tuần** | W01 — Research orientation & planning |
| **Nhóm** | ViVQA-VLM (4 SV) |
| **GVHD** | Assoc.Prof. Đặng Ngọc Minh Đức |
| **Ngày nộp** | 2026-09-14 |

## 1. Mục tiêu tuần

Định hướng nghiên cứu và lập kế hoạch 15 tuần: hiểu bài toán Vietnamese VQA với VLM, ràng buộc **PEFT-only**, phân công SV1–SV4, mỗi SV đọc ≥ 3 paper theo mảng (10 câu hỏi + "liên quan gì đến ViVQA-VLM"), dựng khung docs, và chốt milestone Review 1 / Review 2 / Faculty Review / Final Defense.

## 2. Đã làm (per-SV)

| SV | Vai trò | Công việc đã hoàn thành | Evidence (file / commit / log) |
|----|---------|-------------------------|--------------------------------|
| SV1 | VLM / Baseline | Review 3 paper (BLIP-2, Florence-2, Qwen2.5-VL) đủ 10 câu; lập bảng so sánh 3 baseline, đề xuất Qwen2.5-VL-3B làm baseline chính | `docs/W01/SV1_Paper_Review.md`; `Literature_Matrix.md` (weekly #1–#3 lúc W01; canonical L1/L5/L6) |
| SV2 | Dataset / OCR | Review 3 paper (ViVQA, ViTextVQA, TextVQA/LoRRA) đủ 10 câu; bảng thống kê nhanh 2 dataset; đề xuất PaddleOCR + đối chứng VietOCR | `docs/W01/SV2_Paper_Review.md`; `Literature_Matrix.md` (weekly #4–#5; canonical L3/L4) |
| SV3 | PEFT / Training | Review 4 paper (LoRA, QLoRA, VL-Adapter, LLaVA-1.5) đủ 10 câu; đề xuất recipe QLoRA, 9 ablation, 8 giả thuyết PEFT; điền 4 hàng Literature Matrix | Canonical review: `docs/02_references/Reading_Notes/SV3_Paper_Review.md` (W01 = pointer); `Hypotheses.md` H1–H8; `Candidate_Methods.md` |
| SV4 | Evaluation / Deployment | Review 3 paper (POPE, VQAv2, Survey TPAMI) đủ 10 câu; chốt định nghĩa EM/VQA-Acc/ANLS/POPE-vi; giao thức chuẩn hoá câu trả lời tiếng Việt | `docs/W01/SV4_Paper_Review.md`; `Literature_Matrix.md` (weekly #8; canonical L9) |

## 3. Team deliverables

- [x] Cấu trúc thư mục `ViVQA-VLM/` + khung docs `docs/weekly/W01/`
- [x] `Project_Plan_15_Weeks.md` — 4 milestone đã **gắn ngày dự kiến** (W03/W07/W13/W15)
- [x] Paper review cá nhân SV1–SV4 (mỗi người ≥ 3 paper, đủ 10 câu + mục liên quan)
- [x] `Research_Orientation.md` — bài toán, scope, gap sơ bộ, risk, evaluation (mục 7 đã trả lời)
- [x] Deliverable synthesis (Literature_Matrix, Research_Gap, RQ, Hypotheses, Candidate_Methods, Dataset_Analysis) — **thuộc W02**, bản nháp đã có tại `docs/weekly/W02/`
- [x] `ViVQA-VLM_Week1_Progress.pptx` (deck tiến độ, 12 slide)

## 4. Kết quả / số liệu

| Mục | Giá trị | Ghi chú |
|-----|---------|---------|
| Dataset / split | ViVQA / ViTextVQA · split `chốt W02` · seed `42` | Số liệu chi tiết → W02 |
| Baseline model | **Qwen2.5-VL-3B** (đề xuất) · BLIP-2 đối chứng · Florence-2 ablation | Reproduce **đúng 1**, chốt W03 |
| Method | Đề xuất: QLoRA NF4 all-linear + format-prompt tiếng Việt + OCR-prompting | PEFT-only |
| Exact Match (EM) | _—_ (đo từ W05) | Metric chính |
| VQA Accuracy | _—_ (đo từ W05) | Metric chính |
| ANLS | _—_ (đo từ W05, ViTextVQA) | Metric chính scene-text |
| BLEU / CIDEr | _—_ | **Phụ**, không dùng làm metric chính |
| Train time | _—_ (đo pilot W04) | Nếu có PEFT |
| Inference time | _—_ (đo W05) | Batch size / hardware |
| Hardware | Colab Pro/Premium / GPU trường | GPU/VRAM ghi khi chạy |

> Tuần 1 **không có thí nghiệm chính thức** (đúng phạm vi orientation); các ô metric để "—" và sẽ đo từ W04–W05.

## 5. Blocker

| Blocker | Impact | Owner | Hướng xử lý |
|---------|--------|-------|-------------|
| Chưa tải PDF/dataset về local | Low | Cả nhóm | Tải dataset + PDF vào W02; đối chiếu số liệu "≈" trong review |
| Chưa cấp phát GPU trường / hạn mức Colab | Medium | SV3 | Đăng ký Colab Pro/Premium; queue GPU trường cho pilot W04 |

## 6. Kế hoạch tuần sau (W02)

- [ ] SV1: hoàn tất so sánh baseline (bảng số) BLIP-2 / Florence-2 / Qwen2.5-VL; hỗ trợ chốt baseline
- [ ] SV2: tải ViVQA, ViTextVQA; thống kê thật (#ảnh/#QA/loại câu hỏi); thử PaddleOCR/VietOCR trên 50 ảnh (CER/WER)
- [ ] SV3: đọc bổ sung DoRA / LLaMA-Adapter V2; viết notebook pilot QLoRA (Qwen2.5-VL-3B, ~200 mẫu) cho `experiments/W01_Pilot/`
- [ ] SV4: hiện thực `src/eval/` (EM, VQA-Acc, ANLS, POPE-vi) + script chuẩn hoá câu trả lời tiếng Việt

## 7. Link experiment log

- Template: `docs/templates/Experiment_Log_Template.md`
- Run folder: `experiments/W01_Pilot/` (pilot QLoRA dự kiến W04)
- Config: `configs/` (pin phiên bản ở W04)

## 8. Ràng buộc nhắc lại

> **Ràng buộc cứng (PEFT-only):** CHỈ được dùng PEFT — LoRA hoặc QLoRA (4-bit / 8-bit).
> **KHÔNG** full fine-tuning toàn bộ VLM. Mọi cải tiến phải so với baseline đã reproduce, **cùng dataset split, cùng seed, cùng hardware protocol**.
> Runtime mục tiêu: Google Colab Pro/Premium hoặc GPU server trường.

---

# FILE: docs/W02/README.md

# W02 — Literature & dataset (index)

Tuần 2: **Literature matrix + Dataset analysis**. Canonical dir: `docs/W02/`. Không phải tuần train.

## Deliverable

| File | Vai trò |
|------|---------|
| [Literature_Matrix.md](../02_references/Literature_Matrix.md) | canonical sống tại `docs/02_references/Literature_Matrix.md` (W02 chỉ còn pointer) |
| [Dataset_Analysis.md](Dataset_Analysis.md) | thống kê thật từ JSON, seed 42 |
| [Research_Gap.md](Research_Gap.md) | G1–G4 evidence-linked, focus G3 |
| [Research_Questions.md](Research_Questions.md) | **Canonical RQ1–RQ4** (+ sub-RQ; chốt numbering W03) |
| [Hypotheses.md](Hypotheses.md) | **Canonical H1–H8** (H9–H11 out-of-scope; chốt W03) |
| [Candidate_Methods.md](Candidate_Methods.md) | baseline table, recipe NF4, A1–A9 |
| [W02_Team_Report.md](W02_Team_Report.md) | báo cáo tuần (từ WEEK2_STATUS + template) |
| [WEEK2_STATUS.md](WEEK2_STATUS.md) | self-audit DoD, 90% |

## Stats & script

- Output: `experiments/W02_dataset_stats/` (`vivqa` / `vitextvqa` / `vivqa_x` stats + `samples/` 6 loại câu hỏi × 3 dataset).
- Script: `src/data/analyze_dataset.py`

```
python src/data/analyze_dataset.py
```

Run of record: 2026-09-14, seed 42. Optional: `--data_root data`, `--top_k 30`, `--only vivqa vitextvqa`.

---

# FILE: docs/W02/W02_Team_Report.md

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

---

# FILE: docs/W03/README.md

# W03 — Review 1 (index)

Tuần 3: **đóng gói deliverable Review 1** + trả nợ kỹ thuật W2 (RQ/H numbering, Gap×Paper, link chết). Canonical dir: `docs/W03/`. Không phải tuần train.

> Review 1 **không** yêu cầu model chạy. Đánh giá: research understanding, gap/RQ, kế hoạch.

## Deliverable tuần này

| File | Vai trò |
|------|---------|
| [W03_Team_Report.md](W03_Team_Report.md) | báo cáo tuần, per-SV, kế hoạch W4 |
| [RQ_Hypothesis_Reconciliation.md](RQ_Hypothesis_Reconciliation.md) | map scaffold RQ0–3 / H1–4 → canonical RQ1–4 / H1–8 |
| [Baseline_Decision.md](Baseline_Decision.md) | so sánh 3B vs 7B vs Florence-2; **đề xuất**, chưa CHỐT |
| [W03_DIFF_REPORT.md](W03_DIFF_REPORT.md) | mọi file created/edited/moved |
| [W03_CLOSEOUT_LOG.md](W03_CLOSEOUT_LOG.md) | log thao tác + self-check |

## Deliverable pack Review 1 (nguồn slide)

| # | Mục | File nguồn |
|---|---------|------------|
| 1 | Bài toán + motivation | `docs/master_instruction.md` §1–2; `docs/W01/Research_Orientation.md` |
| 2 | Research Gap G1–G4 + Gap×Paper | `docs/W02/Research_Gap.md`; `docs/02_references/Literature_Matrix.md` |
| 3 | RQ1–RQ4 | `docs/W02/Research_Questions.md` |
| 4 | Hypotheses H1–H8 | `docs/W02/Hypotheses.md` |
| 5 | Dataset Analysis (stats seed 42) | `docs/W02/Dataset_Analysis.md`; `experiments/W02_dataset_stats/` |
| 6 | Candidate Methods + baseline đề xuất | `docs/W02/Candidate_Methods.md`; `docs/W03/Baseline_Decision.md` |
| 7 | Kế hoạch 15 tuần + milestones | `docs/W01/Project_Plan_15_Weeks.md` |
| 8 | Risk matrix | `docs/W01/Project_Plan_15_Weeks.md` §6; `docs/master_instruction.md` §12 |
| 9 | (optional) zero-shot sanity | *chưa có run* — slide ghi "W4–W5"; không bịa số |

Outline slide: [`docs/04_slides/review1/review1_outline.md`](../04_slides/review1/review1_outline.md)
→ render `.pptx` bằng Claude skill (navy/teal/amber), **không** render ở Cursor.

## Canonical literature

Bản sống duy nhất: [`docs/02_references/Literature_Matrix.md`](../02_references/Literature_Matrix.md)

---

# FILE: docs/W03/W03_Team_Report.md

# W03 — Team Weekly Report

Nguồn: công việc automatable W03 (2026-09-17) + khung `docs/templates/Weekly_Report_Template.md` + plan `docs/W01/Project_Plan_15_Weeks.md`. Không thêm số liệu model. Ô chờ hội đồng → `[CHỜ SAU BUỔI REVIEW]`.

## Metadata

| Trường | Nội dung |
|--------|----------|
| **Tuần** | W03 — REVIEW 1 |
| **Nhóm** | ViVQA-VLM (4 SV) |
| **GVHD** | Assoc.Prof. Đặng Ngọc Minh Đức |
| **Ngày** | 2026-09-17 (đóng pack tài liệu; ngày họp hội đồng = ____) |

## Mục tiêu

Tổng hợp W1–W2 thành câu chuyện Review 1: gap G1–G4, **RQ1–RQ4**, **H1–H8**, dataset stats seed 42, candidate methods, **đề xuất** baseline (chưa CHỐT), kế hoạch 15 tuần, risk. Trả nợ kỹ thuật W2 (numbering, Gap×Paper, link chết). **Không train. Không tải model/dataset.**

## Đã làm (per-SV)

| SV | Vai trò | Việc | Trạng thái | Blocker |
|----|---------|------|------------|---------|
| SV1 | VLM / Baseline | Bảng so sánh 3B vs 7B vs Florence-2; khuyến nghị 3B (nhãn đề xuất) | **Xong khung** — `docs/W03/Baseline_Decision.md` | Chưa CHỐT; GPU/P100 chưa xác nhận với thầy; licence 3B ⚠️REVIEW |
| SV2 | Dataset / OCR | Kiểm tra split/licence trên đĩa (không tách val) | **Xong kiểm tra** | ViVQA & ViTextVQA: không có file local + không có val trên mirror; licence paper ❌ TODO. ViVQA-X có train/val/test + `LICENSE` (MIT *repo*, không phải licence ViVQA) |
| SV3 | PEFT / Training (**leader**) | Canonical RQ/H; reconciliation; remap Gap×Paper; khung metric; conventions run-name; pack W03 | **Xong automatable** | Ô Gap×Paper ⚠️REVIEW (L7/L8/L10, L9×G4); metric paper TODO-human; H1 ngưỡng +5 chưa đo |
| SV4 | Evaluation / Deploy | Outline slide Review 1 (9 mục pack); nhắc EM/VQA-Acc/ANLS, không số model | **Xong outline** | Deck `.pptx` **không** render ở Cursor (Claude skill). Zero-shot sanity chưa có run |

**Team:** pack `docs/W03/` + `docs/04_slides/review1/review1_outline.md` + canonical literature.

## Feedback hội đồng Review 1

[CHỜ SAU BUỔI REVIEW]

| Câu hỏi / góp ý | Ai nói | Việc follow-up | Owner |
|-----------------|--------|----------------|-------|
| [CHỜ SAU BUỔI REVIEW] | | | |

## Kết quả buổi review

[CHỜ SAU BUỔI REVIEW]

- Baseline CHỐT? ____
- Gap/RQ được chấp nhận? ____
- Thay đổi kế hoạch W4–W7? ____

## Kết quả / số liệu

Không có thí nghiệm model trong W03. Số dataset = run W02 seed 42 (không đo lại).

| Mục | Giá trị |
|-----|---------|
| Dataset split + seed | seed **42**. ViVQA train 11,999 / test 3,001 (không val). ViTextVQA train 35,159 / test 10,028 (val trống). Official split paper = TODO-human |
| Model / baseline | **ĐỀ XUẤT** Qwen2.5-VL-3B-Instruct; backup Florence-2; **CHỐT bởi nhóm: ____** |
| Method | tài liệu + remap; chưa LoRA/QLoRA |
| **Exact Match (EM)** | chưa chạy (W5) |
| **VQA Accuracy** | chưa chạy (W5) |
| **ANLS** | chưa chạy (W5) |
| Hardware | W03 = CPU docs only |

## Blocker

| Blocker | Impact | Owner | Next step |
|---------|--------|-------|-----------|
| Baseline chưa CHỐT + GPU chưa xác nhận | High | SV1 + leader | Thầy Đức; điền `Baseline_Decision.md` |
| Mirror ViTextVQA (và ViVQA) thiếu val | Medium | SV2 | **Không tự tách.** Đề xuất lệnh seed=42 — mục dưới. Chờ xác nhận tỉ lệ |
| 0/10 PDF paper | Medium | cả nhóm | điền metric TODO-human |
| Ô Gap×Paper ⚠️REVIEW | Medium | leader | xác nhận L7/L8/L10/L9 |
| `experiments.zip` / `docs/W02.zip` không còn ở root **và** không thấy trong `_archive/` | Low | SV3 | snapshot 14/9 đã mất khỏi workspace; bản sống `experiments/` + `docs/W02/` vẫn đủ |

## Đề xuất tách val (KHÔNG chạy ở W03)

Mirror ViTextVQA: JSON chỉ `train`/`test` (`vitextvqa_stats.json`). Dataset_Analysis: validation empty. **Không** tạo file dưới `data/splits/` trong W03.

Tỉ lệ val: ⚠️REVIEW (không có official ratio trên đĩa). Gợi ý *chờ xác nhận*: 10% **image_id** của train, seed=42, toàn bộ QA của ảnh val đi cùng (tránh leakage).

```text
# ĐỀ XUẤT — copy chạy ở W04 SAU KHI nhóm chốt tỉ lệ. Không execute W03.
# Input: danh sách image_id unique của split train (từ HF mirror nhonhoccode/ViTextVQA).
# Output đề xuất: data/splits/vitextvqa_val_image_ids_seed42.txt  (+ jsonl QA)

python -c "import random; random.seed(42); ids=sorted(set(train_image_ids)); random.shuffle(ids); n=max(1,int(round(0.10*len(ids)))); val=set(ids[:n]); print(len(val))"
```

Cùng protocol cho ViVQA nếu paper không có val (mirror `minhnguyent546/ViVQA`: train/test only).

## Kế hoạch W4

Bám `docs/W01/Project_Plan_15_Weeks.md` (W04 Environment + baseline startup):

- [ ] SV1: env (Colab / GPU trường / Kaggle P100 — **sau khi thầy xác nhận**); pin `transformers`/`peft`/`bitsandbytes`; zero-shot 1 batch trên baseline đề xuất
- [ ] SV2: xác nhận nguồn official + licence; **nếu** được phép — tách val seed=42, commit `data/splits/`; chưa tải ảnh hàng loạt nếu chưa có GPU
- [ ] SV3: pilot QLoRA ~200 mẫu (`experiments/W01_Pilot/` theo plan); đặt tên run `W04_qlora_qwen3b_seed42` (nếu baseline 3B được chốt)
- [ ] SV4: script eval EM/VQA-Acc/ANLS dry-run trên 10 mẫu; log template

## Link experiment log

- W02 stats (vẫn sống): `experiments/W02_dataset_stats/`
- W03: không có run model
- Convention: `docs/00_admin/CONVENTIONS.md`

> **Ràng buộc cứng (PEFT-only):** CHỈ LoRA hoặc QLoRA (4-bit / 8-bit). **KHÔNG** full fine-tuning. Cùng split, seed, hardware so với baseline.

---

# FILE: docs/W04/README.md

# Week 4 — Reproduce Baseline (khởi động)

Index tuần 4 của ViVQA-VLM. Mục tiêu W4: hiểu kiến trúc VLM · dựng data pipeline · **baseline zero-shot đo được** trên ViTextVQA. *Chưa train* (QLoRA để sau).

## Deliverables tuần này
- `W04_Team_Report.md` — báo cáo tuần (tóm tắt + trạng thái).
- `W04_Baseline_Report.md` — báo cáo baseline chi tiết (số, protocol, phát hiện, failure cases, lever cải tiến → W5).
- (xuyên tuần) `../Glossary.md` — sổ tay thuật ngữ.
- (xuyên tuần) `../Review_W01-W04_and_Defense_Prep.md` — ôn tập W1–W4 + luyện bảo vệ.

## Kết quả chính
- Baseline: **Qwen2.5-VL-3B-Instruct**, zero-shot.
- **EM 38.3% / ANLS 55.9%** (ViTextVQA mirror, test, n=300, seed 42).

## Artifacts (log thí nghiệm)
- `../../experiments/W04_zeroshot_qwen25vl_seed42/` — `env.json`, `zeroshot_sample.json`, `zeroshot_vitextvqa_n20.json`, `zeroshot_vitextvqa_n300_seed42.json`.
- `../../notebooks/01_zeroshot_baseline.ipynb` — notebook chạy (runner).

## Trạng thái
✅ Đủ điều kiện sang W5 (baseline đo được + tái lập). Việc còn mở xem cuối `W04_Team_Report.md`.

---

# FILE: docs/W04/W04_Team_Report.md

# W04 Report — Reproduce Baseline (khởi động)

**Tuần:** 4 · **Milestone khung:** Baseline (môi trường + baseline khởi động)
**Baseline:** Qwen2.5-VL-3B-Instruct · **Compute:** Google Colab (Tesla T4 15GB)

---

## 1. Mục tiêu tuần
Hiểu kiến trúc VLM end-to-end · dựng data pipeline ViTextVQA · zero-shot baseline **đo được** (EM/ANLS). *Chưa train QLoRA.*

## 2. Đã làm
- Hiểu forward pass Qwen VL (Vision Encoder → Merger → LLM; DeepStack; dynamic resolution).
- **Chốt baseline Qwen2.5-VL-3B** (đổi từ Qwen3-VL-4B: tooling chín + đúng danh sách duyệt + nhẹ, hợp T4 free; Qwen3-VL-4B giữ làm reference). *Còn: xác nhận với thầy Đức.*
- Môi trường Colab tái lập: mount Drive, pin version, log GPU, seed 42, ghi `env.json`.
- Data ViTextVQA (mirror `nhonhoccode`): tải `images.zip` (16.762 ảnh) → giải nén Drive → join COCO-style → flatten 10.028 record test.
- Zero-shot + eval harness EM/ANLS.

## 3. Kết quả
| Metric | Giá trị |
|---|---|
| Exact Match (EM) | **38.3%** |
| ANLS | **55.9%** |
| EM↔ANLS gap | 17.6 điểm |
| Câu trả lời dài (nghi thừa chữ) | 87/300 (29%) |

Protocol: n=300, seed 42, shuffle seed 42, max_pixels=1024·28·28, greedy, max_new_tokens=32. Chi tiết đầy đủ ở `W04_Baseline_Report.md`.

## 4. Phát hiện lõi
EM↔ANLS gap (đúng-ý-nhưng-lệch-format) · anchoring (đoán lì 1 đáp án) · đọc sai chữ số · hallucination địa danh · 3 failure case (general OK / hallucination infographic / degenerate chữ cách điệu). → 3 lever cải tiến có bằng chứng: siết prompt · OCR-enhanced prompting · đo hallucination.

## 5. Artifacts
- Notebook: `notebooks/01_zeroshot_baseline.ipynb`
- Log: `experiments/W04_zeroshot_qwen25vl_seed42/*.json`
- Docs: `W04_Baseline_Report.md`, `../Glossary.md`

## 6. Việc còn mở → W5
- [ ] Đổi **mirror → official split** (`minhquan6203/ViTextVQA`, gated — cần xin quyền + verify train/val/test).
- [ ] Mở rộng n (→ vài trăm/full) cho baseline chính thức.
- [ ] Phân loại `question_type` (Yes-No/counting/text-reading/reasoning) cho error analysis.
- [ ] Baseline ViVQA (general) — cần ảnh COCO.
- [ ] Xác nhận baseline với thầy Đức.
- [ ] Thử lever "prompt ngắn" như ablation đầu tiên.

## 7. Trạng thái
✅ **PASS** — baseline đo được + tái lập được. Sẵn sàng W5.

---

# FILE: docs/W05/README.md

# Week 5 — Baseline reproduce hoàn chỉnh

Index tuần 5 của ViVQA-VLM. Mục tiêu W5: **baseline zero-shot chính thức, tái lập được** trên cả hai dataset của đề tài (ViVQA + ViTextVQA), dữ liệu đúng nguồn chính thức, metric và config chốt trước khi chạy test. *Chưa train* (QLoRA để W8–W9).

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Bản tiếng Anh cho thesis (R4 §2–3, §6 và R6 §2) sẽ chuyển từ bản này sau khi nhóm duyệt xong.

## Deliverables tuần này

| File | Vai trò |
|------|---------|
| [W05_Baseline_Report.md](W05_Baseline_Report.md) | Báo cáo baseline chi tiết: số liệu, protocol, dữ liệu, error analysis, so sánh W4↔W5, vấn đề mở |
| [W05_Team_Report.md](W05_Team_Report.md) | Báo cáo tuần theo khuôn nhóm (per-SV, DoD, blocker, kế hoạch W6) |

## Kết quả chính (zero-shot, Qwen2.5-VL-3B-Instruct, 4-bit NF4)

| Dataset | Tập đánh giá | EM | VQA-Acc | ANLS |
|---------|--------------|:--:|:-------:|:----:|
| ViVQA | test đầy đủ, 3.001 câu | **0.225** | 0.225 | **0.288** |
| ViTextVQA (chính thức) | tập con test 2.000 câu, seed 42 | **0.285** | 0.285 | **0.468** |

⚠️ Không so sánh với số W4 (EM 38.3% / ANLS 55.9%): khác cấu hình ở 4 điểm — xem `W05_Baseline_Report.md` §8.

## Artifacts

- Kết quả: `../../experiments/W05_zeroshot_qwen25vl_vivqa_test_seed42/`, `../../experiments/W05_zeroshot_qwen25vl_vitextvqa_test2000_seed42/` (`metrics.json` + `predictions.json`)
- Script: `../../src/data/prepare_vivqa.py`, `../../src/data/prepare_vitextvqa.py`, `../../src/eval/run_baseline.py`, `../../src/eval/metrics.py`
- Config chuẩn: `../../configs/qwen_lora.yaml` (commit `37da80a`)
- Notebook: `../../notebooks/01_baseline_zeroshot_qwen25vl.ipynb` (ViVQA), `../../notebooks/02_run_baseline_qwen25vl.ipynb` (runner chung)

## Trạng thái

✅ Baseline đã reproduce trên cả hai dataset, log tái lập đầy đủ. Việc còn mở xem cuối `W05_Team_Report.md`.

---

# FILE: docs/W05/W05_Team_Report.md

# W05 — Team Weekly Report

Nguồn: `docs/W05/W05_Baseline_Report.md` + log trong `experiments/W05_*` + lịch sử commit. Khung theo `docs/templates/Weekly_Report_Template.md`. Không có số liệu ngoài log. Ô chưa có nguồn → `TODO (nhóm bổ sung)`.

## Metadata

| Trường | Nội dung |
|--------|----------|
| **Tuần** | W05 — Baseline reproduce hoàn chỉnh |
| **Nhóm** | ViVQA-VLM (4 SV) |
| **GVHD** | Assoc.Prof. Đặng Ngọc Minh Đức · Phạm Minh Trí |
| **Thời gian** | 2026-09-22 → 2026-09-27 |

## Mục tiêu

Baseline zero-shot **chính thức, tái lập được** trên cả ViVQA và ViTextVQA (master §9: W5 = baseline reproduce xong), với dữ liệu đúng nguồn chính thức, metric và config chốt trước khi chạy test.

## Đã làm

1. **Dữ liệu ViVQA gốc:** phát hiện dữ liệu trên máy là ViVQA-X (dataset khác) → lấy ViVQA từ repo gốc, viết `src/data/prepare_vivqa.py` (tải 10.328 ảnh COCO, cắt val 10% seed 42).
2. **Dữ liệu ViTextVQA chính thức:** xin quyền kho gated `minhquan6203/ViTextVQA`, phát hiện `test.json` chỉ là mẫu nộp Kaggle (dùng `test_gt.json`), viết lại `src/data/prepare_vitextvqa.py` (kiểm tra tự động, tập con 2.000 câu cố định).
3. **Metric chặt:** sửa `src/eval/metrics.py` (chữ thường thật sự, bỏ tiền tố `màu `, đổi số chỉ khi cả chuỗi là một số, không bảng đồng nghĩa), có test.
4. **Config chuẩn:** `max_pixels = 512·28·28`, float16 cho T4 (commit `37da80a`).
5. **Script chạy chung** `src/eval/run_baseline.py`: chạy theo lô, resume, chống trộn kết quả, ghi log tái lập — đáp ứng mục "inference script" của reproducibility package.
6. **Chạy baseline chính thức** trên cả hai dataset + error analysis (xem Baseline Report §6–§8).
7. **Dọn dẹp:** repo sạch (không file chưa commit, không nhánh thừa), Drive gọn (gom dữ liệu thừa, archive notebook W4), `.gitignore` chặn file cá nhân khỏi repo public.

### Per-SV

| SV | Vai trò | Đã làm | Evidence |
|----|---------|--------|----------|
| SV1 | VLM / Baseline | TODO (nhóm bổ sung) | |
| SV2 | Dataset / OCR | TODO (nhóm bổ sung) | |
| SV3 | PEFT / Training | Thực hiện các mục 1–7 ở trên | commit `37da80a` → `10cc915` |
| SV4 | Evaluation / Deployment | TODO (nhóm bổ sung) | |

### Definition of Done

| DoD item | Status | Evidence |
|----------|:------:|----------|
| Model chạy được trên Colab (4-bit NF4, T4) | ✅ | `metrics.json` meta (GPU, version) |
| Dữ liệu ViVQA đúng nguồn, đủ ảnh | ✅ | `data/vivqa/prep_report.json` — 10.328/10.328 ảnh |
| Dữ liệu ViTextVQA chính thức, đủ ảnh, không dùng nhầm mẫu Kaggle | ✅ | `data/vitextvqa_official/prep_report.json` — thiếu 0 ảnh |
| Metric EM / VQA-Acc / ANLS có test, chốt trước test | ✅ | `src/eval/metrics.py` |
| Baseline ViVQA test đầy đủ | ✅ | `experiments/W05_zeroshot_qwen25vl_vivqa_test_seed42/` |
| Baseline ViTextVQA (tập con 2.000) | ✅ | `experiments/W05_zeroshot_qwen25vl_vitextvqa_test2000_seed42/` |
| Log tái lập (split, seed, GPU, version, config, commit, thời gian) | ✅ | `meta` trong mỗi `metrics.json` |
| Error analysis sơ bộ | ✅ | Baseline Report §6–§7 |
| Baseline được GVHD xác nhận | ❌ | xem Blocker B1 |

## Kết quả / số liệu

| Mục | ViVQA | ViTextVQA |
|-----|-------|-----------|
| Tập đánh giá | test đầy đủ, 3.001 câu | tập con test 2.000 câu, seed 42 |
| Model / baseline | Qwen2.5-VL-3B-Instruct, zero-shot, 4-bit NF4 | như bên trái |
| **Exact Match (EM)** | **0.225** | **0.285** |
| **VQA Accuracy** | 0.225 | 0.285 |
| **ANLS** | **0.288** | **0.468** |
| BLEU / CIDEr (phụ) | không dùng (đáp án ngắn) | không dùng |
| Train time | không train | không train |
| Inference time | 2.532 s (0,84 s/mẫu) | 2.702 s (1,35 s/mẫu) |
| Hardware | Colab Tesla T4 15 GB | Colab Tesla T4 15 GB |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Owner | Bước tiếp |
|---|--------|-----|-------|-----------|
| B1 | Chưa có xác nhận của GVHD cho lựa chọn baseline Qwen2.5-VL-3B (mở từ W4) | Cao | TODO | Trình thầy kèm kết quả W5 |
| B2 | 23 output hỏng `!!!!` ở ViTextVQA — nghi tràn số fp16; rủi ro cho train QLoRA trên T4 | Cao | SV3 | Chạy lại 23 câu với compute fp32 (W6) |
| B3 | Số ViTextVQA giảm ~10 điểm so với W4 — nghi do `max_pixels` 512 và 4-bit | Trung bình | TODO | Ablation trên dev: 512 vs 1024 token, 4-bit vs fp16 (W6) |
| B4 | ViTextVQA không có nhãn loại câu hỏi → chưa phân tích lỗi theo loại được | Trung bình | TODO | Gán loại câu hỏi (heuristic hoặc tay trên mẫu) |
| B5 | Val ViVQA là tự cắt (bản gốc không có) | Thấp | — | Đã ghi rõ trong protocol (seed 42, 10%) |

## Kế hoạch tuần sau (W06 — Baseline → Method, Report 4)

- [ ] Kiểm chứng vấn đề fp16 (B2) — **ưu tiên**, vì ảnh hưởng trực tiếp việc train QLoRA
- [ ] Ablation độ phân giải / lượng tử hoá trên dev (B3), quyết định config cuối
- [ ] Prompt optimization trên VAL/dev (siết trả lời ngắn, tiếng Việt, không lặp câu hỏi)
- [ ] Dựng pipeline OCR (PaddleOCR / VietOCR) cho ViTextVQA
- [ ] Gán loại câu hỏi ViTextVQA (B4)
- [ ] Trình GVHD xác nhận baseline (B1)
- [ ] Viết Methodology (Report 4) — bản tiếng Việt trước
- [ ] Chuẩn bị dàn ý Review 2 (W7): baseline + phương pháp đề xuất + kết quả sơ bộ

Phân công per-SV: TODO (nhóm thống nhất).

## Link experiment log

- `experiments/W05_zeroshot_qwen25vl_vivqa_test_seed42/` (`metrics.json`, `predictions.json`)
- `experiments/W05_zeroshot_qwen25vl_vitextvqa_test2000_seed42/` (`metrics.json`, `predictions.json`)
- Reproduce: xem `W05_Baseline_Report.md` §11

---

# FILE: docs/W06/README.md

# Week 6 — Baseline → Method (Report 4)

Index tuần 6 của ViVQA-VLM. Mục tiêu W6: **chẩn đoán và chốt những gì ảnh hưởng tới tính công bằng của mọi so sánh về sau**, dựng đường ray huấn luyện QLoRA, và viết Methodology (R4). Nguyên tắc: *chẩn đoán dtype → ablation cấu hình → đóng băng cấu hình → mới huấn luyện*.

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Deliverables tuần này

| File | Vai trò |
|------|---------|
| [W06_Team_Report.md](W06_Team_Report.md) | Báo cáo tuần: việc đã làm, số liệu, blocker, kế hoạch W7 |
| [../03_report/R4_Methodology.md](../03_report/R4_Methodology.md) | **Deliverable chính** — Methodology bản tiếng Việt (7 mục + phụ lục quyết định còn mở) |

## Ba phát hiện chính

| # | Phát hiện | Con số then chốt |
|---|-----------|------------------|
| 1 | **Tràn số fp16** là nguyên nhân 23 output suy biến `!!!!` của baseline | fp16: 23 câu hỏng → fp32: 0; đối chứng 50 câu không thoái lui câu nào |
| 2 | **Độ phân giải là đánh đổi**, không phải cải thiện đơn thuần | 1.280 vs 512 token: ANLS +0,039 (CI95 [+0,012; +0,066]), EM +0,027 (McNemar p = 0,134), chậm 1,46× |
| 3 | **T4 bão hoà tính toán ở fp32** khi huấn luyện | 4,93 s/mẫu; tăng batch 2 → 6 không nhanh hơn; ~14,77 giờ/epoch ViVQA |

Bằng chứng PEFT-only: **7.372.800 / 3.761.995.776 = 0,196%** tham số được huấn luyện, khớp chính xác LoRA r=16 trên q/k/v/o × 36 lớp decoder ngôn ngữ.

## Artifacts

- **Dtype probe:** `../../experiments/W06_dtype_probe_seed42/` (`failed_ids.json`, `probe_summary.json`, `probe_breakdown.json`)
- **Ablation độ phân giải:** `../../experiments/W06_ablation_config_dev300_seed42/` (`ablation_summary.json`, `paired_A1_vs_A2.json`)
- **Smoke run QLoRA:** `../../experiments/W06_smoke_qlora_seed42/smoke_summary.json` + `train_log.json` của từng run
- **Script mới:** `../../src/adaptation/train_qlora.py`, `../../src/eval/find_degraded.py`, `../../src/eval/probe_breakdown.py`, `../../src/eval/compare_runs.py`
- **Notebook:** `../../notebooks/03_ablation_config_qwen25vl.ipynb`, `../../notebooks/04_qlora_qwen25vl.ipynb`

## Trạng thái

✅ Phần thực nghiệm cấu hình và R4 bản tiếng Việt hoàn tất. ⏳ Năm quyết định cấu hình được dời có chủ đích sang W7 (đo trên GPU L4) — xem cuối `W06_Team_Report.md`.

---

# FILE: docs/W06/W06_Team_Report.md

# W06 — Team Weekly Report

> Bản tiếng Việt (bản làm việc).

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W06 — Baseline → Method (Report 4) |
| Kết thúc | 2026-09-29 |
| Mốc kế tiếp | **W07 — Review 2** (baseline + phương pháp đề xuất + kết quả sơ bộ) |
| Phần cứng dùng trong tuần | Tesla T4 (Colab miễn phí) |
| Commit cuối tuần | xem mục "Dấu vết commit" |

## Mục tiêu

1. Chẩn đoán 23 output suy biến của baseline W5 trước khi huấn luyện.
2. Ablation cấu hình suy luận (độ phân giải, kiểu số) trên **dev**, không đụng test.
3. Dựng và kiểm chứng đường ray huấn luyện QLoRA; đo chi phí thật.
4. Viết Methodology (R4) bản tiếng Việt.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | Dtype probe (23 câu hỏng + 50 đối chứng) | ✅ fp16 là nguyên nhân; fp32 sửa được, không thoái lui |
| 2 | Ablation độ phân giải trên dev 300 câu + kiểm định bắt cặp | ✅ ANLS cải thiện có ý nghĩa, EM không; chi phí 1,46× |
| 3 | `train_qlora.py`: che nhãn chỉ đáp án, đo `prompt_len` theo từng ảnh, log chi phí | ✅ selftest offline; đã huấn luyện thật 30 step trên T4 |
| 4 | Smoke run: batch 2 vs 6, bật/tắt gradient checkpointing | ✅ T4 bão hoà tính toán; tắt checkpointing → hết VRAM |
| 5 | Sửa lỗi đếm tham số với trọng số 4-bit (đếm thiếu một nửa) | ✅ dùng hàm đếm của PEFT: **0,196%** |
| 6 | Tương thích `transformers` 5.17.0 (bỏ `warmup_ratio`) | ✅ quy đổi sang `warmup_steps` + lọc tham số theo API thật |
| 7 | R4 Methodology bản tiếng Việt | ✅ 23/23 con số đối chiếu khớp dữ liệu repo |
| 8 | Xác nhận GPU của trường | ✅ Trường **không có GPU** → chuyển Colab Pro (L4) từ W7 |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| VLM / Baseline | Dtype probe (1), ablation độ phân giải (2) |
| Dataset / OCR | Đếm mẫu train thật (10.799), chính sách chọn đáp án |
| PEFT / Training | `train_qlora.py` (3), smoke run (4), đếm tham số (5), tương thích thư viện (6) |
| Evaluation | `find_degraded.py`, `probe_breakdown.py`, `compare_runs.py` (McNemar + bootstrap) |

### Definition of Done

- [x] Nguyên nhân output suy biến được xác nhận bằng thí nghiệm một biến
- [x] Mọi so sánh cấu hình thực hiện trên dev, có kiểm định thống kê
- [x] Đường ray huấn luyện chạy hết, loss giảm, tỉ lệ tham số < 1%
- [x] Chi phí huấn luyện đo được (s/mẫu, VRAM đỉnh, giờ/epoch)
- [x] R4 bản tiếng Việt, mọi con số truy được về file trong repo
- [ ] Đóng băng cấu hình → **dời W7** (có lý do, xem Blocker)
- [ ] Prompt optimization, gán `question_type` ViTextVQA → **dời W7**

## Kết quả / số liệu

**Dtype probe** (73 câu ViTextVQA test: 23 hỏng + 50 đối chứng):

| | fp16 | fp32 |
|---|---|---|
| Câu suy biến | 23 | 0 |
| Nhóm hỏng — EM / ANLS | 0,000 / 0,000 | 0,348 / 0,517 |
| Nhóm đối chứng — EM | 0,280 | 0,280 |
| Thoái lui | — | 0 |
| s/câu | 1,965 | 2,739 |

**Ablation độ phân giải** (dev 300 câu, fp32, 4-bit):

| | 512 token | 1.280 token |
|---|---|---|
| EM | 0,3333 | 0,3600 |
| ANLS | 0,4654 | 0,5041 |
| s/câu | 2,605 | 3,807 |

Bảng chéo EM: 93 cùng đúng · 7 chỉ 512 đúng · 15 chỉ 1.280 đúng · 185 cùng sai.

**Smoke run QLoRA** (200 mẫu ViVQA, 512 token, fp32):

| Batch × tích luỹ | Checkpointing | s/mẫu | VRAM đỉnh | Kết quả |
|---|---|---|---|---|
| 2 × 8 | Bật | 4,93 | 5,93 GB | ✅ loss 1,61 → 0,88 |
| 6 × 3 | Tắt | — | 14,46 / 14,56 GB | ❌ hết VRAM |
| 6 × 3 | Bật | 5,10 | 11,6 GB | ✅ |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|--------|-----|-------------|
| B1 | Baseline Qwen2.5-VL-3B chưa được GVHD xác nhận (treo từ W4) | Trung bình | Trao đổi trực tiếp với hai GVHD tại buổi họp offline |
| B2 | T4 không có bf16, fp32 quá chậm: 14,77 giờ/epoch | **Cao** | Colab Pro, GPU L4 từ W7 |
| B3 | Cấu hình baseline W5 dùng fp16 (đã chứng minh gây lỗi) | Cao | Chạy lại baseline **một lần** sau khi đóng băng cấu hình ở W7 |
| B4 | `run_baseline.py` kiểm tra cứng `torch_dtype = float16` (`CANONICAL`) | Trung bình | Cập nhật cùng lúc với việc đóng băng cấu hình ở W7 |
| B5 | Review 2 cần kết quả sơ bộ; chưa có số EM/ANLS của mô hình đã huấn luyện | Cao | Run QLoRA ngắn trên L4 nếu kịp; không dùng adapter smoke run |

**Vì sao dời đóng băng cấu hình sang W7:** phần cứng đổi từ T4 sang L4. Chốt trên T4 rồi đổi GPU nghĩa là chạy lại baseline hai lần.

## Kế hoạch tuần sau (W07 — Review 2)

1. Nâng Colab Pro; xác nhận được cấp L4 và `bf16_supported = True`.
2. Probe ba nhánh fp16 / fp32 / bf16 trên L4 (tái dùng 73 câu probe).
3. Smoke run huấn luyện 512 vs 1.280 token trên L4 → chốt `max_pixels`; đo thêm trục 4-bit vs không lượng tử hoá.
4. Đóng băng cấu hình: cập nhật `configs/qwen_lora.yaml` + `CANONICAL` trong `run_baseline.py`.
5. Chạy lại baseline trên cả hai dataset, một lần.
6. Run QLoRA sơ bộ + đánh giá trên dev → kết quả sơ bộ cho Review 2 (nếu kịp).
7. Prompt optimization trên dev; gán `question_type` cho ViTextVQA.
8. Slide Review 2 (tiếng Anh, trình bày tiếng Việt); sau review điền checklist AI_Capstone_Review123.

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `26af132` | `find_degraded.py` + danh sách 23 câu hỏng |
| `23e0421` | Notebook 03 (dtype probe + ablation cấu hình) |
| `43aa272` | Kết quả dtype probe + phân tích theo nhóm |
| `6cb7c18` | Ablation độ phân giải + kiểm định McNemar |
| `c9ab5f7` | `train_qlora.py` |
| `0c93f78` | Notebook 04 (smoke run) |
| `e7f9a40` | Tương thích `transformers` 5.17 |
| `d22595c` | Đếm tham số 4-bit đúng + tham số batch/checkpointing qua CLI |
| `5e7e2e6` | Kết quả smoke run |
| `ebe5d91` | Số mẫu train thật + chú thích 4-bit và 10 step |
| `357e166` | R4 Methodology bản tiếng Việt |

## Link experiment log

- `experiments/W06_dtype_probe_seed42/`
- `experiments/W06_probe_fp16_qwen25vl_vitextvqa73_seed42/`, `experiments/W06_probe_fp32_qwen25vl_vitextvqa73_seed42/`
- `experiments/W06_ablation_config_dev300_seed42/`
- `experiments/W06_ablation_A1_mp512_4bit_qwen25vl_vitextvqa_dev300_seed42/`, `experiments/W06_ablation_A2_mp1280_4bit_qwen25vl_vitextvqa_dev300_seed42/`
- `experiments/W06_smoke_qlora_seed42/`
- `experiments/W06_smoke_qlora_qwen25vl_vivqa_mp512_seed42/`, `experiments/W06_smoke_qlora_qwen25vl_vivqa_mp512_bs6_seed42/`

---

# FILE: docs/W07/README.md

# Week 7 — Review 2 (baseline đã tái lập + cải tiến đầu tiên)

Index tuần 7 của ViVQA-VLM. Mục tiêu W7, theo góp ý GVHD (07/10/2026): **có baseline rõ ràng trước khi bàn cải tiến**. Cụ thể: sơ đồ kiến trúc baseline, hiểu dataset và nhiễu, chạy lại baseline với giao thức đánh giá nêu rõ, rồi mới thử cải tiến đầu tiên (QLoRA). Nguyên tắc: *chốt cấu hình → baseline một lần → kiểm chứng pipeline → cải tiến → phân tích lỗi*.

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Slide Review 2 bằng tiếng Anh, thuyết trình bằng tiếng Việt.

## Deliverables tuần này

| File | Vai trò |
|------|---------|
| [W07_Team_Report.md](W07_Team_Report.md) | Báo cáo tuần: việc đã làm, toàn bộ số liệu, blocker, kế hoạch W8 |
| [ViVQA-VLM_Review2_EN.pptx](ViVQA-VLM_Review2_EN.pptx) | **Deliverable chính** — slide Review 2 tiếng Anh (16 slide) |
| [ViVQA-VLM_Review2_TV.pptx](ViVQA-VLM_Review2_TV.pptx) | Bản tiếng Việt của slide (để đọc hiểu, không trình chiếu) |
| [ViVQA-VLM_Review2_Script.txt](ViVQA-VLM_Review2_Script.txt) | Kịch bản thuyết trình tiếng Việt theo từng slide |
| [Review2_carryover_from_Review1.md](Review2_carryover_from_Review1.md) | Nội dung W4–W6 + góp ý GVHD chuyển từ báo cáo Review 1 sang Review 2 |
| [figures/](figures/) | 3 sơ đồ theo yêu cầu GVHD, bản EN + TV (xem dưới) |
| [audit/](audit/) | Error audit 200 lỗi của B2: nhãn, phân xử, script tính số liệu |

## Bốn phát hiện chính

| # | Phát hiện | Con số then chốt |
|---|-----------|------------------|
| 1 | **Pipeline đáng tin**: tái lập đúng số công bố của Qwen2.5-VL-3B | TextVQA val 79,60 so với 79,3 (ngưỡng ±2,0) |
| 2 | **QLoRA 0,196% tham số tăng EM mạnh** trên cả hai dataset, không do rò rỉ | ViVQA 0,221 → 0,738; ViTextVQA full 0,291 → 0,503 (McNemar p ≈ 0) |
| 3 | **B2 vượt SOTA của paper ViTextVQA** trên full test 10.028 câu | EM 50,26 [49,27; 51,23] / F1 71,16 so với ViTextBLIP-2 25,48 / 53,95 |
| 4 | **Điểm yếu còn lại là đọc chữ** — căn cứ cho cải tiến W8 | ViTextVQA: OCR 28% + sai vùng 23% + dấu thanh 8% lỗi; 54% lỗi EM thực ra đúng ý |

Hai đối chứng loại trừ giải thích thay thế: backbone lớn hơn (Qwen2-VL-7B) **kém** 3B khoảng 12 điểm EM; few-shot 5 ví dụ **không** sửa được định dạng câu trả lời → mức tăng của B2 là do QLoRA.

## Sơ đồ (yêu cầu GVHD 07/10)

| Sơ đồ | Nội dung |
|-------|----------|
| `figures/Fig6A_ViVQA_pipeline_noise_{EN,TV}.png` | Quy trình tạo ViVQA (COCO-QA → dịch máy) và các điểm nhiễu N1–N4 |
| `figures/Fig7A_Qwen25VL3B_architecture_{EN,TV}.png` | Kiến trúc Qwen2.5-VL-3B-Instruct và vị trí LoRA (q/k/v/o) |
| `figures/Fig7B_evaluation_protocol_{EN,TV}.png` | Giao thức đánh giá: split/seed cố định, cấu hình khoá, kiểm định ghép cặp |

## Artifacts

- **Chốt cấu hình L4:** `../../experiments/W07_config_freeze_L4_seed42/`
- **B1 (baseline chính thức):** `../../experiments/W07_B1_summary_L4_seed42/`
- **B2 (QLoRA 1 epoch):** `../../experiments/W07_B2_summary_L4_seed42/`
- **Full test ViTextVQA:** `../../experiments/W07_fulltest_summary_L4_seed42/`
- **R0 (tái lập TextVQA):** `../../experiments/W07_R0_summary_L4/`
- **R1 (Qwen2-VL-7B):** `../../experiments/W07_R1_summary_L4_seed42/`
- **FS5 (few-shot):** `../../experiments/W07_FS_summary_L4_seed42/`
- **Rò rỉ train → test:** `../../experiments/W07_data_checks/`
- **Error audit:** `audit/W07_error_audit_final.json` — tái tạo: `python docs/W07/audit/compute_error_audit.py`
- **Script mới:** `../../src/eval/paper_consistency.py`, `../../src/eval/answer_format.py`, `../../src/eval/stratify_by_overlap.py`, `../../src/data/check_split_overlap.py`, `../../src/data/make_audit_sheet.py`, `../../src/eval/run_textvqa_r0.py`
- **Notebook:** `../../notebooks/02` (B1), `04` (QLoRA), `05` (chốt cấu hình), `06` (R0), `07` (R1), `08` (full test), `09` (few-shot)
- **Adapter B2** (không lên git, trên Drive): `adapters/W07_B2_qlora_r16_qwen25vl_{vivqa,vitextvqa}_ep1_bf16_L4_seed42/`

## Trạng thái

✅ Thực nghiệm W7 hoàn tất; slide Review 2 sẵn sàng. ⏳ Sau buổi Review 2 (tuần thực 19–25/10): điền checklist AI_Capstone_Review123 (sheet Review2) theo góp ý reviewer. Việc dời sang W8: QLoRA nhiều epoch, `data_audit` nhãn train, `question_type` ViTextVQA — xem cuối `W07_Team_Report.md`.

---

# FILE: docs/W07/W07_Team_Report.md

# W07 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W07 — Review 2 (baseline đã tái lập + cải tiến đầu tiên) |
| Kết thúc | 2026-10-12 |
| Mốc kế tiếp | **Review 2** — tuần thực 7 (19–25/10/2026); ngày cụ thể bộ môn báo qua email trong tuần đó |
| Phần cứng dùng trong tuần | NVIDIA L4 22 GB (Colab Pro), bf16 native |
| Commit cuối tuần | `2f19ca4` (xem mục "Dấu vết commit") |

## Mục tiêu

Theo yêu cầu của GVHD Phạm Minh Trí (họp 07/10/2026): **chưa bàn cải tiến khi chưa có baseline rõ ràng**. Cụ thể tuần này phải:

1. Có sơ đồ kiến trúc baseline và hiểu từng thành phần; giải thích dataset, nhiễu và cách xử lý.
2. Chốt cấu hình trên L4 rồi chạy lại baseline **một lần** với giao thức đánh giá nêu rõ (B1).
3. Chứng minh pipeline đánh giá đáng tin bằng cách tái lập một con số đã công bố.
4. Chạy cải tiến đầu tiên (QLoRA, B2), so với B1 dưới điều kiện y hệt, có kiểm định thống kê.
5. Error analysis + hallucination analysis trên lỗi của B2.
6. Slide Review 2 (tiếng Anh, trình bày tiếng Việt).

Baseline **Qwen2.5-VL-3B-Instruct** được GVHD xác nhận ngày 07/10/2026.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | Chốt cấu hình trên L4: dtype probe fp16 / fp32 / bf16 (73 câu) + so 512 vs 1.280 token khi train | ✅ **bf16** (0 câu hỏng) · **512 token** (1.280 train chậm 1,91× > ngưỡng 1,5×) |
| 2 | Checkpoint / resume cho huấn luyện + chữ ký run (sha256 dữ liệu + cấu hình) | ✅ resume thử trên L4; chạy tiếp với cấu hình khác bị chặn |
| 3 | **B1** — baseline zero-shot chính thức trên L4 | ✅ 0 câu hỏng trên cả hai dataset |
| 4 | **R0** — tái lập TextVQA val của Qwen2.5-VL-3B (kiểm chứng pipeline) | ✅ 79,60 so với 79,3 công bố (ngưỡng ±2,0) |
| 5 | **B2** — QLoRA 1 epoch trên ViVQA và ViTextVQA | ✅ EM tăng mạnh trên cả hai, McNemar p < 0,05 |
| 6 | Kiểm tra rò rỉ train/dev → test + tách kết quả ViVQA theo mức trùng | ✅ mức tăng của B2 không do rò rỉ |
| 7 | Full test ViTextVQA (10.028 câu) cho B1 và B2, so với SOTA của paper | ✅ B2 vượt ViTextBLIP-2 ở cả EM và F1 (theo CI95) |
| 8 | **R1** — Qwen2-VL-7B zero-shot (đối chiếu Table 7 paper ViTextVQA) | ✅ 7B kém 3B; nguyên nhân chính là trả lời dài |
| 9 | **FS5** — đối chứng few-shot 5 ví dụ | ✅ few-shot không sửa được định dạng; B2 vượt FS5 → QLoRA học hơn định dạng |
| 10 | Error audit 200 lỗi EM của B2 (4 người gán nhãn + nhãn gợi ý AI, có phân xử) | ✅ xem mục "Error audit" |
| 11 | 3 sơ đồ theo yêu cầu GVHD (quy trình + nhiễu ViVQA, kiến trúc baseline, giao thức đánh giá) | ✅ `figures/` (bản EN + TV) |
| 12 | Slide Review 2 EN + TV (16 slide) + kịch bản thuyết trình | ✅ |
| 13 | Đưa báo cáo Review 1 (`docs/W03/index_TV/TA`) về đúng nội dung mốc Review 1 | ✅ tag `review1-report`; phần W4–W6 chuyển sang Review 2 |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| VLM / Baseline | Chốt cấu hình (1), B1 (3), R0 (4), R1 (8), sơ đồ kiến trúc |
| Dataset / OCR | Kiểm tra rò rỉ (6), sơ đồ quy trình + nhiễu ViVQA, số mẫu ViTextVQA chính thức |
| PEFT / Training | Checkpoint/resume (2), B2 (5), đếm tham số train |
| Evaluation | Full test (7), FS5 (9), error audit (10), `paper_consistency.py`, `answer_format.py`, kiểm định McNemar + bootstrap |

### Definition of Done

- [x] Sơ đồ kiến trúc baseline + sơ đồ dataset/nhiễu + sơ đồ giao thức đánh giá (yêu cầu GVHD 07/10)
- [x] Cấu hình chốt bằng luật đặt trước, ghi vào `configs/qwen_lora.yaml`
- [x] Baseline B1 chạy một lần trên cấu hình chốt, 0 câu hỏng
- [x] Pipeline tái lập được số công bố (R0)
- [x] Cải tiến B2 so với B1 cùng điều kiện, có McNemar + CI95 bootstrap
- [x] Error + hallucination analysis có độ tin cậy giữa người gán nhãn (κ)
- [x] Slide Review 2 EN + kịch bản TV
- [ ] `data_audit` 200 nhãn train (đo nhiễu nhãn để làm ablation làm sạch dữ liệu) → **dời W8**
- [ ] Gán `question_type` cho ViTextVQA (để tách kết quả theo loại câu hỏi) → **dời W8**
- [ ] QLoRA chính thức nhiều epoch, chọn epoch trên dev → **dời W8**

## Kết quả / số liệu

### Cấu hình chốt (áp dụng cho mọi run từ B1)

Qwen2.5-VL-3B-Instruct · bf16 · 4-bit NF4 + double quant · `max_pixels` 401.408 (≤ 512 token ảnh) · greedy, `max_new_tokens` 32 · seed 42 · L4.
QLoRA: r = 16, α = 32, dropout 0,05, target q/k/v/o · **7.372.800 tham số train = 0,196%** · lr 2e-4 · batch hiệu dụng 16 (2 × 8) · warmup 3% · checkpoint mỗi 200 step.

| Quyết định | Luật đặt trước | Số đo | Kết luận |
|---|---|---|---|
| dtype | bf16 nếu 0 câu hỏng và thoái lui đối chứng ≤ 2/50 | fp16: 23/23 câu hỏng (tái hiện đúng lỗi T4) · bf16: 0 câu hỏng, 1/50 thoái lui so với fp32 | **bf16** |
| Độ phân giải | 1.280 token nếu train chậm ≤ 1,5× so với 512 | ViTextVQA: 34,77 vs 18,18 s/step = **1,91×** | **512 token** (1.280 để làm ablation) |

### B1 — baseline chính thức (zero-shot)

| Tập | N | EM | ANLS | token-F1 (phụ) | s/câu |
|---|---|---|---|---|---|
| ViVQA test | 3.001 | 0,2209 | 0,2861 | 0,3805 | 0,51 |
| ViTextVQA test2000 | 2.000 | 0,2880 | 0,4753 | 0,5377 | 0,86 |
| ViTextVQA full test | 10.028 | 0,2912 | 0,4760 | 0,5389 | 0,88 |

VQA-Acc = EM vì mỗi câu chỉ có 1 đáp án gold. B0 (W05, fp16 trên T4) chỉ dùng **tham khảo**: fp16 gây 23 câu hỏng trên ViTextVQA nên không so trực tiếp với B1.

### R0 — kiểm chứng pipeline bằng số đã công bố

| Run | N | VQA-Acc (%) | Công bố | VRAM đỉnh |
|---|---|---|---|---|
| R0 (cấu hình paper: bf16, không lượng tử, độ phân giải mặc định) | 5.000 | **79,60** | 79,3 (Bai et al., 2025, Table 5) | 10,9 GB |
| R0b (cấu hình B1: 4-bit, 512 token) | 5.000 | 75,21 | — | 2,54 GB |

R0 nằm trong ngưỡng ±2,0 điểm đặt trước → **pipeline tái lập được**. Cấu hình tiết kiệm của nhóm mất 4,39 điểm (CI95 [−5,35; −3,45]) nhưng giảm VRAM 4,3×.

### B2 — QLoRA 1 epoch so với B1

| Tập | N | EM B1 → B2 | ANLS B1 → B2 | McNemar (chỉ B1 đúng / chỉ B2 đúng) | ΔANLS CI95 |
|---|---|---|---|---|---|
| ViVQA test | 3.001 | 0,2209 → **0,7378** | 0,2861 → 0,7566 | 35 / 1.586, p ≈ 0 | [+0,453; +0,489] |
| ViTextVQA test2000 | 2.000 | 0,2880 → **0,4970** | 0,4753 → 0,7191 | 86 / 504, p ≈ 7 × 10⁻⁷³ | — |
| ViTextVQA full | 10.028 | 0,2912 → **0,5026** | 0,4760 → 0,7202 | 423 / 2.543, p ≈ 0 | [+0,235; +0,253] |

| Chi phí train | ViVQA | ViTextVQA |
|---|---|---|
| Số mẫu train | 10.799 | 35.159 |
| Step / thời gian 1 epoch | 675 / **3,03 h** | 2.198 / **11,99 h** |
| VRAM đỉnh | 6,62 GB | 6,88 GB |
| Loss cuối | 0,476 | 0,618 |

Độ dài câu trả lời (số từ trung bình) trên ViVQA: gold 1,57 · B1 2,88 · B2 1,55.

**So với SOTA của paper ViTextVQA** (full test; luật: "cao hơn" nếu cận dưới CI95 bootstrap > số công bố):

| | EM (CI95) | F1 (CI95) | Kết luận |
|---|---|---|---|
| ViTextBLIP-2 (Nguyen et al., 2025, Table 6) | 25,48 | 53,95 | — |
| B1 zero-shot | 29,12 [28,23; 29,99] | 53,89 [53,11; 54,66] | EM cao hơn · F1 không phân biệt được |
| B2 QLoRA | **50,26** [49,27; 51,23] | **71,16** [70,42; 71,89] | **Cao hơn ở cả EM và F1** |

Kiểm tra phụ: dự đoán trên 2.000 câu chung trùng 100% giữa run full và run test2000 (tất định); EM của test2000 nằm trong CI95 của full test → test2000 đại diện tốt.

### Rò rỉ train → test

| | ViVQA | ViTextVQA |
|---|---|---|
| Ảnh train trùng ảnh test | 1.274 | **0** |
| Câu test có ảnh đã gặp trong train | 1.356 / 3.001 (45,2%) | 0 |
| Cặp hỏi–đáp test đã gặp trong train | 541 | 407 (khác ảnh) |

Trên tập **clean** ViVQA (1.219 câu không trùng ảnh, không trùng cặp hỏi–đáp): B1 → B2 = 0,250 → 0,691, ΔEM CI95 [+0,41; +0,47]. Mức tăng của B2 **không do rò rỉ**. ViTextVQA không có ảnh trùng nên là phép thử tổng quát hoá sạch hơn.

### R1 — backbone lớn hơn (Qwen2-VL-7B, zero-shot)

| Run | EM | F1 (phụ) | ANLS | s/câu | VRAM đỉnh |
|---|---|---|---|---|---|
| B1 · Qwen2.5-VL-3B · 4-bit · 512 token | **28,8** | 53,77 | 47,53 | 0,86 | — |
| R1 · Qwen2-VL-7B · cấu hình paper | 16,7 | 43,50 | 29,94 | 2,85 | 20,27 GB |
| R1b · Qwen2-VL-7B · cấu hình B1 | 16,2 | 41,31 | 29,04 | 1,03 | 6,2 GB |
| Paper Table 7 "QwenVL-7b" (100 câu) | 28,0 | 53,8 | — | — | — |

ViTextVQA test2000. R1 lệch Table 7 (|z| > 1,96); B1 lại nhất quán với dòng đó. Backbone lớn hơn **không** tự động tốt hơn: 7B kém 3B khoảng 12 điểm EM, chủ yếu vì trả lời dài.

### FS5 — đối chứng few-shot (tách lợi ích định dạng khỏi QLoRA)

| Tập | EM B1 | EM FS5 | EM B2 | Few-shot sửa định dạng? | B2 > FS5? |
|---|---|---|---|---|---|
| ViVQA | 22,09 | 20,03 | 73,78 | **Không** — EM giảm (p < 0,05) | Có (p < 0,05) |
| ViTextVQA test2000 | 28,80 | 27,60 | 49,70 | Chưa kết luận (p ≥ 0,05) | Có (p < 0,05) |

5 ví dụ chỉ có chữ, lấy từ train (seed 42). Kết luận: mức tăng của B2 không thể đạt được chỉ bằng prompt; QLoRA học nhiều hơn định dạng câu trả lời.

### Error audit — 200 câu B2 sai EM (100 mỗi dataset)

| Chỉ số | Toàn bộ | ViVQA | ViTextVQA |
|---|---|---|---|
| B2 **đúng ý** (người đọc chấp nhận) | **54,0%** [47,1; 60,8] | 62,0% | 46,0% |
| Hallucination | **7,0%** [4,2; 11,4] | 5,0% | 9,0% |

Phân loại nguyên nhân chính (%):

| Loại lỗi | Toàn bộ | ViVQA | ViTextVQA |
|---|---|---|---|
| Định dạng / độ dài (ý đúng) | 21,0 | 9,0 | **33,0** |
| Đồng nghĩa / khác từ (ý đúng) | 17,5 | **34,0** | 1,0 |
| Sai vùng / không có trong ảnh | 15,5 | 8,0 | 23,0 |
| Đọc chữ sai (OCR) | 14,0 | — | **28,0** |
| Khác | 12,5 | 25,0 | — |
| Gold sai / nhiễu | 12,5 | 19,0 | 6,0 |
| Sai dấu thanh | 4,0 | — | 8,0 |
| Đếm sai | 2,5 | 5,0 | — |
| Code-switching | 0,5 | — | 1,0 |

Khoảng tin cậy 95% theo Wilson. Độ tin cậy gán nhãn: κ Cohen giữa hai người trên 20 dòng chung = 0,62 (loại lỗi) / 0,57 (đúng ý); κ người–AI ở hallucination = 0,30 (thấp → nhãn hallucination dựa vào người + phân xử). Bỏ 42 dòng có hỗ trợ AI: đúng ý 53,8%, hallucination 7,0% → **kết luận không đổi**.

**Ý nghĩa:**
- Hơn một nửa "lỗi" EM của B2 thực ra đúng ý → EM đánh giá thấp mô hình; cần báo cáo kèm ANLS/F1 và phân tích định tính.
- Trên ViTextVQA, nhóm lỗi đọc chữ (OCR 28 + sai vùng 23 + dấu thanh 8 = 59%) là điểm yếu chính → căn cứ chọn **OCR-enhanced prompting** làm cải tiến ở W8.
- Gold nhiễu 19% ở ViVQA (dataset dịch máy) → cần `data_audit` và ablation làm sạch nhãn.

Giới hạn: người phân xử đã thấy nhãn AI (có thể bị neo theo AI — anchoring); so sánh hallucination B1 vs B2 (H8) cần audit thêm lỗi của B1 (dự kiến W11).

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|--------|-----|-------------|
| B1 | 1 epoch ViTextVQA mất 12 h L4 → không đủ ngân sách Colab để train nhiều epoch hoặc nhiều biến thể trên full train | **Cao** | Nhiều epoch chỉ trên ViVQA; B3 (QLoRA + OCR) và B2-sub train trên **tập con cố định ≈ 8.000 mẫu, seed 42** (chốt 12/10, xem `../W08/README.md`) |
| B2 | OCR-enhanced prompting (ablation bắt buộc theo đề cương) chưa nối vào `run_baseline.py` / `train_qlora.py` | **Cao** | W8: nối `mode="ocr"` với OCR cache + selftest |
| B3 | EM đánh giá thấp mô hình (54% lỗi EM là đúng ý) | Trung bình | Báo cáo song song ANLS + token-F1 + error audit; không đổi metric chính |
| B4 | Nhiễu nhãn ViVQA chưa đo trực tiếp trên train | Trung bình | W8: `data_audit` 200 nhãn train |
| B5 | Few-shot / prompt không cải thiện → hướng "prompt optimization" trong kế hoạch W06 bị thay bằng đối chứng FS5 | Thấp | Giữ FS5 làm bằng chứng; không đầu tư thêm vào prompt thuần |

## Kế hoạch tuần sau (W08 — Improvement: OCR-enhanced prompting)

1. Hồ sơ Review 2: Team Report + README W07 (file này); sau buổi Review 2 điền checklist AI_Capstone_Review123 (sheet Review2) theo góp ý reviewer.
2. Nối OCR vào `run_baseline.py` và `train_qlora.py` (`--mode ocr` đọc OCR cache) + selftest.
3. Tạo OCR cache ViTextVQA (dev → test → train), đo tốc độ trên 100 ảnh trước.
4. Chọn backend OCR (`paddleocr` vs `paddledet_vietocr`) và mẫu prompt trên dev300.
5. Ablation OCR chỉ ở inference trên test2000: B1 + OCR, B2 + OCR.
6. QLoRA ViVQA 2 epoch, đánh giá val sau mỗi epoch để chọn epoch.

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `72e3a79`, `22877ec`, `7729301` | Notebook 05 + chốt cấu hình L4 (bf16, 512 token, token-F1 phụ, checkpoint/resume, chữ ký run) |
| `280a543`, `e0984e8` | Sửa `final_loss` sau resume; bằng chứng resume trên L4 |
| `e9acd3c`, `b5a58dd` | 3 sơ đồ theo góp ý GVHD; đưa báo cáo Review 1 về đúng mốc (tag `review1-report`) |
| `c96e3f3`, `c8cf419` | Notebook 02 + kết quả B1 |
| `3fcb1f4`, `fefa9ed` | `--adapter`, `--epochs`; R0 notebook 06 |
| `5e4154f` | B2 ViVQA |
| `026e3b1`, `53a89ed` | Kiểm tra rò rỉ; tách kết quả theo mức trùng |
| `22b44a3` | Kết quả R0 / R0b |
| `997f9c3` | B2 ViTextVQA |
| `e8d8fa2`, `884d964` | `--variant` Qwen2-VL-7B + `paper_consistency.py`; kết quả R1 / R1b |
| `b7dc647`, `8fdf8d2` | `--few_shot_k`, `answer_format.py`, `make_audit_sheet.py` |
| `ce54e31` | Full test ViTextVQA 10.028 câu (B1, B2) |
| `60ec444` | Kết quả FS5 |
| `f562e07` | Slide Review 2 EN + TV + kịch bản + sơ đồ EN |
| `4fcc084`, `665b966`, `2f19ca4` | Error audit: thiết lập, kết quả, nguồn gốc nhãn AI |
| `3b41b14` | Script thu hồi chia sẻ Drive (bản dùng thật `thuHoiNhanh` v2 commit ở `5d161c4`, W8) |

## Link experiment log

- Chốt cấu hình: `experiments/W07_config_freeze_L4_seed42/`, `experiments/W07_probe_*_L4_seed42/`, `experiments/W07_smoke_qlora_*_L4_seed42/`, `experiments/W07_resume_test_qlora_qwen25vl_vivqa_bf16_L4_seed42/`
- B1: `experiments/W07_B1_summary_L4_seed42/` + 3 run `W07_B1_zeroshot_*`
- B2: `experiments/W07_B2_summary_L4_seed42/` + 2 run train `W07_B2_qlora_*` + 3 run eval `W07_B2_eval_*`
- Full test: `experiments/W07_fulltest_summary_L4_seed42/`
- R0: `experiments/W07_R0_summary_L4/`, `experiments/W07_R0_textvqa_val_qwen25vl3b_paper_bf16_L4/`, `experiments/W07_R0b_textvqa_val_qwen25vl3b_b1cfg_L4/`
- R1: `experiments/W07_R1_summary_L4_seed42/` + 2 run `W07_R1*`
- FS5: `experiments/W07_FS_summary_L4_seed42/` + 2 run `W07_FS5_*`
- Rò rỉ: `experiments/W07_data_checks/`
- Error audit: `docs/W07/audit/W07_error_audit_final.json` (tái tạo bằng `docs/W07/audit/compute_error_audit.py`)

---

# FILE: docs/W08/README.md

# Week 8 — Improvement (OCR-enhanced prompting)

Index tuần 8 (tuần ảo) của ViVQA-VLM. Mục tiêu W8: **nối OCR-enhanced prompting vào pipeline và chạy ablation "có / không OCR"** trên ViTextVQA, đồng thời chạy QLoRA ViVQA nhiều epoch có chọn epoch trên val. Hồ sơ Review 2 (Team Report W07 + README W07) phải xong trước tuần thực 7 (19–25/10).

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Vì sao chọn OCR-enhanced prompting

Error audit W07 (200 lỗi EM của B2): trên ViTextVQA, nhóm lỗi đọc chữ chiếm phần lớn — sai OCR 28%, nhầm vùng chữ 23%, dấu thanh 8% (nguồn: `../W07/audit/W07_error_audit_final.json`). Đây là gap G2 (scene-text tiếng Việt). Ablation "with vs. without OCR-enhanced prompting" cũng là yêu cầu bắt buộc của đề cương (mục 7.3).

Rủi ro đã biết: Qwen2.5-VL vốn tự đọc chữ tốt (R0 TextVQA val VQA-Acc 79,60), nên text OCR ngoài có thể gây nhiễu. Kết quả âm vẫn hợp lệ nếu ablation đầy đủ.

## Kế hoạch tuần

| # | Việc | Trạng thái |
|---|------|-----------|
| 1 | Dọn tồn W07: commit `thuHoiNhanh` v2, `compute_error_audit.py` ghi JSON với `newline="\n"`, tạo `docs/W08/` | ✅ `5d161c4` |
| 2 | Hồ sơ Review 2: `docs/W07/W07_Team_Report.md` + `docs/W07/README.md` (bản TV); checklist Review2 điền SAU buổi review | ✅ `3f736eb` |
| 3 | Nối OCR vào `src/eval/run_baseline.py` và `src/adaptation/train_qlora.py` (`--mode ocr` đọc OCR cache) + selftest; script tập con train | ✅ |
| 4 | Tạo OCR cache ViTextVQA: notebook `10_ocr_cache_vitextvqa.ipynb` — Phần A: pilot 100 ảnh + dev300 × 2 backend + chỉ số answer-in-OCR; Phần B (sau bước 5): test2000 + tập con train | ✅ 13/10 — Phần A (dev300, 2 backend) + Phần B (VietOCR: test2000 1.486 ảnh + train 2.670 ảnh, 0 lỗi, 0,97 s/ảnh) |
| 5 | Chọn backend OCR (`paddleocr` vs `paddledet_vietocr`) + mẫu prompt trên dev300: notebook `11_ocr_select_dev300_qwen25vl.ipynb` + `src/eval/select_ocr_config.py` (áp luật tự động) | ✅ 13/10 — chốt **VietOCR · v1** |
| 6 | Ablation OCR chỉ ở inference trên test2000: B1+OCR, B2+OCR — notebook `12_ocr_ablation_test2000_qwen25vl.ipynb` | ✅ 13/10 — không khác biệt có ý nghĩa (cả B1 lẫn B2) |
| 7 | **B3 (QLoRA + OCR) vs B2-sub (QLoRA, không OCR)** trên tập con 8.000 câu, eval test2000 — notebook `13_qlora_ocr_b3_vs_b2sub_qwen25vl.ipynb` (thay QLoRA ViVQA 2 epoch, dời sang W9) | ✅ 14/10 — **B3 cải thiện có ý nghĩa** (EM +3,70 điểm, p = 1,7 × 10⁻⁶) |
| 8 | Đóng tuần: tổng kết + bàn giao W08 — `W08_Team_Report.md` | ✅ 14/10 |

## Quyết định đã chốt

| Ngày | Quyết định | Lý do |
|------|-----------|-------|
| 12/10/2026 | **B3 (QLoRA + OCR) train trên tập con cố định của ViTextVQA train (≈ 8.000 mẫu, seed 42)**; train lại **B2-sub** (không OCR) trên đúng tập con đó để so sánh cùng điều kiện. Không train B3 trên full train. | 1 epoch full ViTextVQA tốn 43.156 s ≈ 12,0 h trên L4 (`train_log.json` của B2) → full B3 ≈ 60–70 CU, vượt ngân sách Colab còn lại. Tập con giữ so sánh công bằng (chỉ khác yếu tố OCR) với ≈ 30 CU. |
| 13/10/2026 | Tập con **chọn theo ảnh** (`src/data/make_train_subset.py`): xáo tên ảnh bằng seed 42, lấy dần tới khi ≥ 8.000 câu | Mọi câu hỏi của một ảnh đi cùng nhau; ~3 câu/ảnh nên OCR cache cho tập con rẻ hơn ~3 lần. Số câu/ảnh thật + sha256: `../../experiments/W08_data_subsets/` |
| 13/10/2026 | Prompt `ocr` = prompt zero-shot + **đúng một dòng OCR** đặt trước câu hỏi; ảnh không có chữ → trùng hệt zero-shot | Bản trước W08 thêm cả chỉ dẫn "Trả lời ngắn gọn…" → ablation lẫn 2 biến. Train và eval dùng chung `prompt_for_sample` |
| 13/10/2026 | GPU còn lại của W8 dùng cho **B3 + B2-sub** (phép thử chính của hướng OCR); **QLoRA ViVQA 2 epoch dời sang W9** | B3 vs B2-sub trả lời câu hỏi nghiên cứu chính; ViVQA 2 epoch chỉ tinh chỉnh B2, không liên quan OCR. Colab còn 146,75 CU (13/10, 20:27) |
| 13/10/2026 | OCR chạy ở **phiên Colab riêng** với PaddleOCR 3.x; VietOCR cài `--no-deps` | `pip install paddleocr` chỉ còn bản 3.x (API khác 2.x). Gói VietOCR trên PyPI khoá `pillow==10.2.0`, sẽ hạ cấp Pillow của môi trường chạy VLM |

## Luật chọn cấu hình OCR (đặt TRƯỚC khi có số liệu — 13/10/2026)

Chọn trên **dev300** (không nhìn test), với B1 (zero-shot, cấu hình chốt W07) + OCR:

1. **Backend:** chọn backend có EM(B1 + OCR) cao hơn trên dev300. Nếu McNemar exact p ≥ 0,05 (không phân biệt được), chọn backend có `answer_in_ocr_pct` cao hơn (`experiments/W08_ocr_dev300_coverage/`). Nếu vẫn hoà, chọn backend nhanh hơn (s/ảnh).
2. **Template** (`v1` trung tính / `v2` báo OCR có thể sai): cùng luật như trên với backend đã chọn. Hoà → giữ `v1` (đơn giản hơn).
3. `max_chars = 600`, `min_conf = 0` giữ mặc định. Chỉ đổi nếu chênh `answer_in_ocr_pct` − `answer_in_ocr_capped_pct` > 5 điểm (cắt làm mất đáp án).
4. Cấu hình chọn xong thì **đóng băng** cho B1+OCR / B2+OCR trên test2000 và B3 (train). Không chỉnh thêm sau khi nhìn test.

`answer_in_ocr` (đáp án gold nằm nguyên cụm từ trong văn bản OCR) chỉ là **chẩn đoán / tiêu chí phụ**: ViTextVQA không có nhãn chữ thật nên không tính được CER của OCR.

## Kết quả bước 4 — OCR dev300 (266 ảnh, 300 câu)

Nguồn: `../../experiments/W08_ocr_dev300_coverage/ocr_coverage.json`. Môi trường: Colab T4, paddleocr 3.7.0, paddlepaddle 3.3.1 (GPU).

| Backend | Model đọc | % ảnh có chữ | dòng/ảnh | conf TB | đáp án ∈ OCR (giữ dấu) | đáp án ∈ OCR (bỏ dấu) | token recall | s/ảnh |
|---|---|---|---|---|---|---|---|---|
| `paddleocr` | PP-OCRv6_medium_rec (Latin) | 99,62 | 13,34 | 0,889 | 27,0% | **57,67%** | 0,490 | **0,34** |
| `paddledet_vietocr` | VietOCR vgg_transformer | 100,0 | 12,87 | 0,758 | **37,67%** | 48,67% | **0,573** | 0,95 |

Ghép cặp (đáp án ∈ OCR, giữ dấu): chỉ `paddleocr` có đáp án 28 câu, chỉ `paddledet_vietocr` có 60 câu, McNemar exact p = 0,0008. Cắt ở 600 ký tự không làm mất đáp án ở cả 2 backend.

**Phát hiện:** mô hình đọc Latin của PaddleOCR **làm rơi ký tự có dấu tiếng Việt** ("Kiểu Thẩm" → "Kiu Thm", "KIẾN THỨC" → "IENTHUC"): chênh giữ dấu / bỏ dấu tới 30,7 điểm. VietOCR giữ dấu tốt hơn (chênh 11,0 điểm) nhưng đọc sót mặt chữ nhiều hơn. Đây là bằng chứng trực tiếp cho gap G2 (OCR có sẵn chưa xử lý tốt dấu thanh tiếng Việt), khớp nhóm lỗi "sai dấu thanh" 8% trong error audit W07. Lựa chọn backend vẫn theo luật EM ở bước 5.

**Sự cố đã xử lý:** `paddlepaddle-gpu` (CUDA 12.6) ghi đè thư viện NCCL của torch (CUDA 13) → `import torch` hỏng (`undefined symbol: ncclCommWindowDeregister`). Notebook 10 mục 2 chụp và khôi phục thư viện `nvidia-*` của torch, kiểm torch + Paddle GPU ở tiến trình riêng, CPU fallback giữ torch nguyên (commit `833abd8`).

## Kết quả bước 5 — chọn cấu hình OCR trên dev300 (L4, bf16, 4-bit, seed 42)

Nguồn: `../../experiments/W08_ocr_select_dev300_L4_seed42/ocr_selection.json` + 5 run `W08_B1*_dev300_*`.

| Run | EM | ANLS | token-F1 (phụ) | Chỉ B1 đúng / chỉ OCR đúng | McNemar p (so B1) | s/câu |
|---|---|---|---|---|---|---|
| B1 zero-shot | 0,3267 | 0,4615 | 0,5327 | — | — | 0,859 |
| B1 + VietOCR · v1 **(chốt)** | 0,3267 | 0,4600 | 0,5385 | 16 / 16 | 1 | 0,878 |
| B1 + VietOCR · v2 | 0,3233 | 0,4696 | 0,5434 | 17 / 16 | 1 | 0,867 |
| B1 + PaddleOCR · v1 | 0,3200 | 0,4606 | 0,5384 | 20 / 18 | 0,871 | 0,853 |
| B1 + PaddleOCR · v2 | 0,3067 | 0,4700 | 0,5410 | 23 / 17 | 0,43 | 0,858 |

**Áp luật (đúng như ghi trước):** Luật 1: EM v1 hai backend không phân biệt được (p = 0,856) → xét answer-in-OCR: VietOCR 37,67% > PaddleOCR 27,0% → **VietOCR**. Luật 2: v1 = 0,3267 vs v2 = 0,3233, p = 1 → **v1**. Luật 3: cắt 600 ký tự mất 0 điểm → giữ `max_chars 600`, `min_conf 0`. Cấu hình này **đóng băng** cho test2000 và B3.

**Phát hiện:** chèn OCR vào prompt lúc suy luận **không cải thiện B1 zero-shot** (EM giữ nguyên, 16 câu được sửa đúng nhưng 16 câu bị làm sai). Phù hợp rủi ro đã nêu: Qwen2.5-VL vốn tự đọc chữ tốt (R0 TextVQA 79,60). Đây là kết quả âm hợp lệ cho ablation "có / không OCR". Phép thử chính vẫn là **B3 (train cùng OCR) so với B2-sub**: model zero-shot chưa được dạy dùng dòng OCR; train cùng OCR có thể dạy model dùng OCR để sửa dấu. **Phân tích lật (dev300, VietOCR · v1 so với B1):** 167/300 câu trả lời **đổi** khi thêm OCR → model **có dùng** dòng OCR, không bỏ qua. 16 câu OCR sửa đúng, chủ yếu khi model tự đọc sai: số điện thoại (`0914.414.227` → `0949.002.339`), tên riêng (`Lê Xuân Khanh` → `Lê Xuân Khoa`), dấu (`Dây nổ` → `Dây nịt`). 16 câu OCR làm sai do model **chép nguyên lỗi OCR** (`CHĂN` → `CHÂN`, `Thấm` → `Thắm`, `Số` → `Sô`), **dán nguyên chuỗi OCR** (`Thăng Long | 0919467868 | …`) hoặc trả lời dài hơn. 31 câu chỉ khác nhau ở dấu. → Model zero-shot dùng OCR **không phân biệt** (không biết khi nào tin ảnh, khi nào tin OCR, không biết trả lời ngắn) — đúng thứ train cùng OCR (B3) có thể dạy.

## Luật kết luận bước 6 và B3 (đặt TRƯỚC khi chạy test — 13/10/2026)

Cấu hình OCR dùng cho mọi run dưới đây = cấu hình đóng băng ở bước 5 (VietOCR · v1 · 600 · 0). Tất cả trên **test2000** (seed 42), L4, bf16, 4-bit, cùng prompt zero-shot ± đúng 1 dòng OCR.

| So sánh | Vai trò | "Cải thiện" khi |
|---|---|---|
| B1+OCR vs B1 | Ablation bắt buộc "có / không OCR" (zero-shot) | EM tăng **và** McNemar exact p < 0,05 |
| B2+OCR vs B2 | Chẩn đoán: B2 chưa từng thấy dòng OCR khi train (lệch train/test) | như trên — **không** dùng làm kết quả cải tiến |
| **B3 vs B2-sub** | **Phép thử chính:** cùng tập con train 8.000 câu (`train_subset8000_byimage_seed42`), cùng siêu tham số QLoRA (r 16, 1 epoch), chỉ khác B3 có dòng OCR khi train **và** khi test | EM tăng **và** McNemar exact p < 0,05 |
| B3 vs B2 (full train) | Chỉ báo cáo: tập con 8.000 câu + OCR có bù được 77% dữ liệu train bị bỏ không | — (không kết luận cải tiến) |

ANLS luôn báo cáo kèm CI95 bootstrap ghép cặp (10.000 lần, seed 42). Kết quả âm (p ≥ 0,05) vẫn được báo cáo đầy đủ trong bảng ablation.

## Kết quả bước 6 — ablation "có / không OCR" trên test2000 (L4, bf16, 4-bit, seed 42)

Nguồn: `../../experiments/W08_ocr_ablation_test2000_L4_seed42/` (`ocr_ablation_summary.json`, 2 file `paired_*.json`) + 2 run `W08_B1ocr_…_test2000_…`, `W08_B2ocr_…_test2000_…`. Cấu hình OCR đóng băng từ bước 5 (VietOCR · v1 · 600 · 0); 99,85% câu có chữ OCR, trung bình 144 ký tự, 3,55% bị cắt.

| Run | EM | ANLS | token-F1 (phụ) | s/câu |
|---|---|---|---|---|
| B1 zero-shot (W07) | 0,2880 | 0,4753 | 0,5377 | 0,862 |
| B1 + OCR | 0,2935 | 0,4802 | 0,5462 | 0,906 |
| B2 QLoRA 1 epoch (W07) | 0,4970 | 0,7191 | 0,7101 | 1,011 |
| B2 + OCR (chỉ lúc suy luận) | 0,5070 | 0,7214 | 0,7124 | 0,997 |

| So sánh | Chỉ base đúng / chỉ +OCR đúng | McNemar p | ΔANLS (CI95) | Kết luận theo luật |
|---|---|---|---|---|
| B1 → B1 + OCR | 132 / 143 | 0,547 | +0,0050 [−0,0097; +0,0203] | Không khác biệt có ý nghĩa |
| B2 → B2 + OCR | 121 / 141 | 0,24 | +0,0023 [−0,0102; +0,0148] | Không khác biệt có ý nghĩa |

**Diễn giải:**
- Ablation bắt buộc "with / without OCR-enhanced prompting" ở **suy luận** cho **kết quả âm**, nhất quán với dev300. Báo cáo đầy đủ như một dòng ablation.
- Tín hiệu (chưa có ý nghĩa thống kê, **không** dùng làm kết luận): B2 — dù chưa từng thấy dòng OCR khi train — nhích nhiều hơn B1 (+1,00 so với +0,55 điểm EM). Gợi ý mô hình đã fine-tune khai thác OCR tốt hơn zero-shot → củng cố lý do kiểm định B3 (train cùng OCR) so với B2-sub.
- **Độ nhạy (tính bằng McNemar exact, giả định ~260 câu lật như bước 6):** ngưỡng để đạt p < 0,05 là chênh ≈ 34 câu = **1,7 điểm EM**; xác suất phát hiện (mô phỏng 2.000 lần) khi chênh thật là 1,5 / 2,0 / 2,3 / 2,5 điểm lần lượt ≈ 42% / 67% / 79% / 85%. → B3 vs B2-sub chỉ phát hiện ổn định nếu lợi ích thật ≳ 2,3 điểm; kết quả âm khi lợi ích nhỏ hơn không có nghĩa là "OCR vô dụng".

## Kết quả B3 vs B2-sub — phép thử chính (test2000, L4, bf16, 4-bit, seed 42)

Nguồn: `../../experiments/W08_B3_vs_B2sub_test2000_L4_seed42/` (`b3_vs_b2sub_summary.json` + 3 file `paired_*.json`), train log `W08_B2sub_qlora_…`, `W08_B3_qlora_ocr_…`, eval `W08_B2sub_eval_…`, `W08_B3_eval_…`. Notebook 13 chạy một mạch 13–14/10 (không resume).

| Run | Train | OCR | EM | ANLS | token-F1 (phụ) | s/câu |
|---|---|---|---|---|---|---|
| B1 zero-shot | — | không | 0,2880 | 0,4753 | 0,5377 | 0,862 |
| B2 (W07) | full 35.159 câu | không | 0,4970 | 0,7191 | 0,7101 | 1,011 |
| B2-sub | 8.000 câu (2.670 ảnh) | không | 0,4555 | 0,6848 | 0,6692 | 1,021 |
| **B3** | **cùng 8.000 câu** | **VietOCR · v1 (train + test)** | **0,4925** | **0,7038** | **0,6981** | 1,001 |

| So sánh | Chỉ A đúng / chỉ B đúng | McNemar p | ΔANLS (CI95) | Vai trò |
|---|---|---|---|---|
| **B2-sub → B3** | **81 / 155** | **1,67 × 10⁻⁶** | **+0,0191 [+0,0073; +0,0309]** | **Phép thử chính → CẢI THIỆN** (EM +3,70 điểm, vượt ngưỡng độ nhạy ~2,3 điểm) |
| B2 → B3 | 156 / 147 | 0,646 | −0,0153 [−0,0288; −0,0017] | Chỉ báo cáo: EM không phân biệt được; ANLS B3 thấp hơn chút |
| B2 → B2-sub | 154 / 71 | 3,24 × 10⁻⁸ | −0,0343 [−0,0470; −0,0224] | Chỉ báo cáo: cắt 77% dữ liệu train làm giảm 4,15 điểm EM |

| Chi phí train | Mẫu | Step | Giờ train | s/step | VRAM đỉnh | Loss cuối | Tham số train |
|---|---|---|---|---|---|---|---|
| B2 (W07) | 35.159 | 2.198 | 11,99 | 19,63 | 6,88 GB | 0,618 | 0,196% |
| B2-sub | 8.000 | 500 | 2,76 | 19,89 | 6,85 GB | 0,716 | 0,196% |
| B3 | 8.000 | 500 | 2,98 | 21,48 | 8,25 GB | 0,661 | 0,196% |

**Kết luận theo luật đặt trước:** train QLoRA **cùng** OCR (B3) cải thiện có ý nghĩa so với cùng cấu hình **không** OCR (B2-sub): EM 0,4555 → 0,4925, p = 1,7 × 10⁻⁶; ANLS +0,019 (CI95 không chứa 0).

**Diễn giải:**
- Khác bước 6 (OCR chỉ lúc suy luận: không cải thiện), OCR **có tác dụng khi được train cùng** — khớp phân tích dev300: model zero-shot dùng OCR không phân biệt; train cùng OCR dạy model khi nào tin OCR.
- **Hiệu quả:** với 23% dữ liệu và 25% thời gian train (2,98 / 11,99 giờ), B3 đạt EM ngang B2 full (0,4925 vs 0,4970, p = 0,65). OCR bù ≈ 89% phần EM mất do cắt dữ liệu: (0,4925 − 0,4555) / (0,4970 − 0,4555) ≈ 0,89. Chi phí thêm: +8% s/step, +1,4 GB VRAM.

**Giới hạn (phải nêu trong báo cáo):**
1. Một seed (42), một tập con — chưa đo dao động theo seed.
2. Đánh giá trên test2000, chưa trên full test 10.028 câu.
3. ANLS B3 vẫn thấp hơn B2 full (−0,015, CI95 không chứa 0) → B3 "ngang" B2 ở EM, không phải mọi metric.
4. B3 dùng OCR cả train lẫn test (đúng thiết kế); chưa đo B3 khi bỏ OCR lúc test (model có phụ thuộc OCR không).

## Số liệu ngân sách (đo từ log W07, L4, bf16, 4-bit NF4)

| Hạng mục | Thời gian | Nguồn |
|----------|-----------|-------|
| QLoRA 1 epoch ViVQA (10.799 mẫu) | 10.917 s ≈ 3,03 h | `experiments/W07_B2_qlora_r16_qwen25vl_vivqa_ep1_bf16_L4_seed42/train_log.json` |
| QLoRA 1 epoch ViTextVQA (35.159 mẫu) | 43.156 s ≈ 12,0 h | `experiments/W07_B2_qlora_r16_qwen25vl_vitextvqa_ep1_bf16_L4_seed42/train_log.json` |
| Eval B2 ViTextVQA | ≈ 1,0 s/mẫu | `experiments/W07_B2_eval_qlora_r16_ep1_vitextvqa_test2000_bf16_L4_seed42/metrics.json` |
| Eval B1 ViVQA | ≈ 0,51 s/mẫu | `experiments/W07_B1_zeroshot_qwen25vl_vivqa_test_bf16_L4_seed42/metrics.json` |

## Deliverables tuần này

| File | Vai trò |
|------|---------|
| [W08_Team_Report.md](W08_Team_Report.md) | Báo cáo tuần: việc đã làm, toàn bộ số liệu, blocker, kế hoạch W9 |
| README này | Quyết định, luật đặt trước, kết quả từng bước |

## Artifacts

- **Code mới:** `../../src/data/ocr.py` (viết lại, PaddleOCR 3.x, cache v2), `../../src/data/make_train_subset.py`, `../../src/eval/ocr_coverage.py`, `../../src/eval/select_ocr_config.py`; `--mode ocr` trong `../../src/eval/run_baseline.py` và `../../src/adaptation/train_qlora.py`; `prompt_for_sample` trong `../../src/prompting/builder.py`
- **Notebook:** `../../notebooks/10_ocr_cache_vitextvqa.ipynb` (T4), `11_ocr_select_dev300_qwen25vl.ipynb`, `12_ocr_ablation_test2000_qwen25vl.ipynb`, `13_qlora_ocr_b3_vs_b2sub_qwen25vl.ipynb` (L4)
- **Kết quả:** xem mục "Link experiment log" trong `W08_Team_Report.md`
- **Trên Drive (không lên git):** OCR cache `data/vitextvqa_official/ocr/{paddleocr,paddledet_vietocr}.json` (sha256 trong `experiments/W08_ocr_cache/ocr_cache_manifest.json`); tập con `data/vitextvqa_official/train_subset8000_byimage_seed42.json`; adapter B2-sub, B3

## Trạng thái

✅ **Đóng tuần 14/10/2026.** Kết quả chính: **B3 (QLoRA + OCR) cải thiện có ý nghĩa so với B2-sub** (EM 0,4555 → 0,4925, p = 1,7 × 10⁻⁶) và đạt EM ngang B2 full với 23% dữ liệu. OCR chỉ lúc suy luận không cải thiện. Việc mang sang W9: thêm seed cho B3/B2-sub, B3 trên full test, B3 khi bỏ OCR lúc test, QLoRA ViVQA 2 epoch.

---

# FILE: docs/W08/W08_Team_Report.md

# W08 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W08 — Improvement: OCR-enhanced prompting |
| Thời gian thực | 12–14/10/2026 (tuần thực 6) |
| Mốc kế tiếp | **Review 2** — tuần thực 7 (19–25/10/2026); ngày cụ thể bộ môn báo qua email trong tuần đó |
| Phần cứng | Colab Pro: NVIDIA L4 (VLM, train/eval) · Tesla T4 (OCR) |
| Commit cuối tuần | `89c7d5e` (xem mục "Dấu vết commit") |

## Mục tiêu

Error audit W07 cho thấy trên ViTextVQA, lỗi đọc chữ chiếm phần lớn lỗi của B2 (sai OCR 28%, sai vùng 23%, sai dấu 8%). Đề cương bắt buộc ablation "with / without OCR-enhanced prompting". Tuần này:

1. Nối OCR vào pipeline (eval và train) sao cho ablation **chỉ khác đúng một dòng OCR** trong prompt.
2. Tạo OCR cache một lần, tái lập được (ghi rõ engine, phiên bản, model).
3. Chọn cấu hình OCR trên **dev**, theo luật **ghi trước khi có số liệu**.
4. Ablation trên test2000: OCR chỉ lúc suy luận (B1+OCR, B2+OCR).
5. **Phép thử chính:** QLoRA train cùng OCR (B3) so với QLoRA không OCR (B2-sub), cùng tập con train.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | Dọn tồn W07 + hồ sơ Review 2 (W07 Team Report, README) | ✅ `5d161c4`, `3f736eb` |
| 2 | OCR vào pipeline: `--mode ocr` cho `run_baseline` và `train_qlora`; hàm dựng prompt dùng chung `prompt_for_sample` | ✅ prompt `ocr` = zero-shot + đúng 1 dòng OCR; train = eval (selftest) |
| 3 | `ocr.py` viết lại theo PaddleOCR 3.7 (bản `pip` hiện có); cache v2 có conf + box + nguồn gốc | ✅ ảnh thiếu OCR → báo lỗi; thiết lập OCR nằm trong chữ ký run |
| 4 | Tập con train cố định **theo ảnh**: 8.000 câu / 2.670 ảnh, seed 42 | ✅ tất định (manifest có sha256) |
| 5 | Notebook 10 — OCR cache: dev300 × 2 backend, rồi test2000 + train bằng backend đã chọn | ✅ 4.422 ảnh VietOCR, 0 lỗi, 0 NaN; sự cố NCCL đã xử lý |
| 6 | Chỉ số thay thế khi không có nhãn chữ: đáp án có trong OCR không (`ocr_coverage.py`) | ✅ phát hiện PaddleOCR làm rơi dấu tiếng Việt |
| 7 | Notebook 11 — chọn backend × template trên dev300 bằng luật tự động (`select_ocr_config.py`) | ✅ chốt **VietOCR · v1 · 600 ký tự · min_conf 0** |
| 8 | Notebook 12 — ablation OCR lúc suy luận trên test2000 | ✅ không khác biệt có ý nghĩa (B1 lẫn B2) |
| 9 | Notebook 13 — **B3 vs B2-sub** (train cùng / không OCR, cùng tập con) | ✅ **B3 cải thiện có ý nghĩa: EM +3,70 điểm, p = 1,7 × 10⁻⁶** |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| Dataset / OCR | OCR cache (5), chỉ số answer-in-OCR (6), tập con train (4) |
| PEFT / Training | `--mode ocr` khi train (2), B2-sub + B3 (9) |
| Evaluation | Chọn cấu hình theo luật (7), ablation test2000 (8), McNemar + bootstrap, phân tích độ nhạy |
| VLM / Baseline | Prompt OCR sạch, B1 dev300 trên cấu hình chốt W07 |

### Definition of Done

- [x] Ablation "có / không OCR" có kiểm định, luật kết luận ghi trước khi chạy test
- [x] Cấu hình OCR chọn trên dev, đóng băng trước khi chạy test
- [x] OCR cache tái lập được (engine, phiên bản, model, sha256 trong biên bản)
- [x] Phép thử chính chỉ khác đúng yếu tố OCR (chữ ký train B2-sub và B3 chỉ khác khoá `prompting.ocr`)
- [x] Mọi số liệu truy được về file trong `experiments/`
- [ ] Độ ổn định theo seed (mới 1 seed) → **W9**
- [ ] B3 trên full test 10.028 câu, B3 khi bỏ OCR lúc test → **W9**
- [ ] QLoRA ViVQA 2 epoch (dời từ W8) → **W9**

## Kết quả / số liệu

### OCR trên dev300 (266 ảnh, 300 câu) — chỉ số thay thế

| Backend | Đáp án ∈ OCR (giữ dấu) | Đáp án ∈ OCR (bỏ dấu) | s/ảnh (T4) |
|---|---|---|---|
| PaddleOCR (PP-OCRv6, Latin) | 27,0% | **57,7%** | **0,34** |
| PaddleOCR dò + **VietOCR** đọc | **37,7%** | 48,7% | 0,95 |

McNemar (giữ dấu): 28 / 60 câu, p = 0,0008. PaddleOCR làm rơi ký tự có dấu ("Kiểu Thẩm" → "Kiu Thm"). Đây là bằng chứng cho gap G2: OCR có sẵn chưa xử lý tốt dấu thanh tiếng Việt.

### Chọn cấu hình trên dev300 (B1 + OCR)

| Run | EM | ANLS | Lật so với B1 (B1 đúng / OCR đúng) |
|---|---|---|---|
| B1 zero-shot | 0,3267 | 0,4615 | — |
| B1 + VietOCR · v1 **(chốt)** | 0,3267 | 0,4600 | 16 / 16, p = 1 |
| B1 + VietOCR · v2 | 0,3233 | 0,4696 | 17 / 16 |
| B1 + PaddleOCR · v1 | 0,3200 | 0,4606 | 20 / 18 |
| B1 + PaddleOCR · v2 | 0,3067 | 0,4700 | 23 / 17 |

Luật: EM 2 backend hoà (p = 0,856) → xét answer-in-OCR → VietOCR; v2 không hơn v1 → giữ v1.

### Ablation OCR lúc suy luận — test2000

| So sánh | EM | Lật | p | ΔANLS (CI95) | Kết luận |
|---|---|---|---|---|---|
| B1 → B1 + OCR | 0,2880 → 0,2935 | 132 / 143 | 0,547 | +0,005 [−0,010; +0,020] | Không khác biệt |
| B2 → B2 + OCR | 0,4970 → 0,5070 | 121 / 141 | 0,24 | +0,002 [−0,010; +0,015] | Không khác biệt |

Phân tích dev300: OCR làm đổi 167/300 câu trả lời. Model **có dùng** OCR, nhưng dùng không phân biệt (sửa đúng số điện thoại, tên riêng; nhưng cũng chép nguyên lỗi OCR và dán cả chuỗi OCR).

### Phép thử chính — B3 vs B2-sub (test2000)

| Run | Train | OCR | EM | ANLS | Giờ train | VRAM đỉnh |
|---|---|---|---|---|---|---|
| B2 (W07) | 35.159 câu | không | 0,4970 | 0,7191 | 11,99 | 6,88 GB |
| B2-sub | 8.000 câu | không | 0,4555 | 0,6848 | 2,76 | 6,85 GB |
| **B3** | **cùng 8.000 câu** | **có (train + test)** | **0,4925** | **0,7038** | 2,98 | 8,25 GB |

| So sánh | Lật | p | ΔANLS (CI95) | Vai trò |
|---|---|---|---|---|
| **B2-sub → B3** | **81 / 155** | **1,7 × 10⁻⁶** | **+0,019 [+0,007; +0,031]** | **Phép thử chính → CẢI THIỆN** |
| B2 → B3 | 156 / 147 | 0,646 | −0,015 [−0,029; −0,002] | Báo cáo: EM ngang nhau, ANLS B3 thấp hơn chút |
| B2 → B2-sub | 154 / 71 | 3,2 × 10⁻⁸ | −0,034 [−0,047; −0,022] | Báo cáo: cắt 77% dữ liệu mất 4,15 điểm EM |

**Ý nghĩa:** OCR không giúp khi chỉ chèn lúc suy luận, nhưng **giúp rõ khi được train cùng**. Với 23% dữ liệu và 25% thời gian train, B3 đạt EM ngang B2 full; OCR bù khoảng 89% phần EM mất đi do cắt dữ liệu. Chi phí thêm: +8% s/step, +1,4 GB VRAM.

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|--------|-----|-------------|
| B1 | Kết quả B3 mới **1 seed** | **Cao** (hội đồng sẽ hỏi) | W9: chạy thêm seed cho B2-sub và B3 (khoảng 6 h L4 mỗi seed) |
| B2 | Mới đánh giá trên test2000 | Trung bình | W9: B3 trên full test 10.028 câu (khoảng 2,8 h) để đặt cạnh B2 full test và ViTextBLIP-2 |
| B3 | Chưa biết B3 có **phụ thuộc OCR** không | Trung bình | W9: eval B3 khi **bỏ** dòng OCR lúc test (khoảng 35 phút) |
| B4 | ANLS B3 vẫn thấp hơn B2 full | Thấp | Ghi giới hạn; thử B3 với nhiều dữ liệu hơn nếu còn ngân sách |
| B5 | PaddleOCR GPU (CUDA 12) ghi đè NCCL của torch (CUDA 13) | Đã xử lý | Notebook 10 mục 2 chụp + khôi phục thư viện `nvidia-*` của torch (`833abd8`) |

**Ngân sách Colab:** còn 133 CU (14/10); L4 đo thật **1,54 CU/giờ** → khoảng 86 giờ L4, đủ cho phần còn lại của đồ án (ước tính cần 40–45 giờ).

## Kế hoạch tuần sau (W09)

1. **Củng cố B3:** thêm seed cho B2-sub và B3; B3 trên full test; B3 khi bỏ OCR lúc test.
2. QLoRA ViVQA 2 epoch, chọn epoch trên val (dời từ W8).
3. Sau buổi Review 2: điền checklist AI_Capstone_Review123 (sheet Review2), gửi mail, upload Drive.
4. Bắt đầu Results (R6) bản tiếng Việt cho phần OCR, sau khi số liệu đã ổn định.

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `5d161c4` | Bắt đầu W8: `thuHoiNhanh` v2, audit JSON LF, index `docs/W08/` |
| `3f736eb` | W07 Team Report + README (Review 2); `.gitignore` file admin |
| `4c71df2` | Pipeline OCR: PaddleOCR 3.x, cache v2, `--mode ocr`, `prompt_for_sample`, tập con train theo ảnh |
| `c31700e` | Notebook 10, `ocr_coverage.py`, luật chọn cấu hình OCR ghi trước |
| `833abd8` | Sửa notebook 10: bảo vệ NCCL của torch khi cài Paddle GPU |
| `dc76997` | Kết quả OCR dev300 + notebook 11 + `select_ocr_config.py` |
| `f63f346` | Kết quả chọn cấu hình (VietOCR · v1) |
| `50b794c` | Biên bản OCR cache + luật bước 6 và B3 ghi trước + notebook 12 |
| `c80374e` | Kết quả ablation OCR lúc suy luận (test2000) |
| `6b69ff2` | Quyết định A (B3 vs B2-sub) + notebook 13 |
| `89c7d5e` | **Kết quả chính: B3 vs B2-sub** |

## Link experiment log

- OCR: `experiments/W08_ocr_dev300_coverage/`, `experiments/W08_ocr_cache/ocr_cache_manifest.json`
- Tập con train: `experiments/W08_data_subsets/train_subset8000_byimage_seed42_manifest.json`
- Chọn cấu hình: `experiments/W08_ocr_select_dev300_L4_seed42/` + 5 run `W08_B1*_dev300_*`
- Ablation test2000: `experiments/W08_ocr_ablation_test2000_L4_seed42/` + `W08_B1ocr_…_test2000_…`, `W08_B2ocr_…_test2000_…`
- B3 vs B2-sub: `experiments/W08_B3_vs_B2sub_test2000_L4_seed42/` + train `W08_B2sub_qlora_…`, `W08_B3_qlora_ocr_…`, `W08_smoke_B3_…` + eval `W08_B2sub_eval_…`, `W08_B3_eval_…`
- Adapter (không lên git, trên Drive): `adapters/W08_B2sub_qlora_…/epoch_1`, `adapters/W08_B3_qlora_ocr_…/epoch_1`

---

# FILE: docs/W09/README.md

# Week 9 — Củng cố B3 + Review 2

Index tuần 9 (tuần ảo) của ViVQA-VLM. Tuần ảo W09 chạy trong **tuần thực 6–7** (15/10 → Review 2 ở tuần thực 7, 19–25/10).

Mục tiêu W9: **củng cố kết quả chính W08** (B3 = QLoRA + OCR thắng B2-sub) trước khi đưa lên Review 2 — thêm seed, kiểm phụ thuộc OCR, chạy full test — và **cập nhật slide Review 2** với phương pháp đề xuất + kết quả sơ bộ.

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Vì sao phải củng cố B3

Kết quả W08 (`../W08/README.md`): B3 vs B2-sub trên test2000 — EM 0,4555 → 0,4925, McNemar p = 1,7 × 10⁻⁶, ΔANLS +0,019 [+0,007; +0,031]. Còn 4 giới hạn đã ghi:

1. **Mới 1 seed (42)** → lợi ích có thể do khởi tạo LoRA / thứ tự batch "may mắn".
2. Mới test2000, chưa full test 10.028 câu.
3. Chưa biết B3 có **phụ thuộc OCR** không (train có OCR; lúc dùng thật OCR có thể hỏng).
4. ANLS B3 vẫn thấp hơn B2 full.

Giới hạn 1 và 3 quyết định độ tin cậy của đóng góp chính → làm trước.

## Kế hoạch tuần

| # | Việc | GPU | Trạng thái |
|---|------|-----|-----------|
| 0 | Code: `--seed` cho `train_qlora.py` (selftest), `src/eval/seed_stability.py`, notebook `14_b3_seed_stability_qwen25vl.ipynb`, luật ghi trước (README này) | — | ✅ 15/10 |
| 1 | **Phần A:** B3 seed 42 eval **bỏ OCR** lúc test (test2000) | ≈ 40 phút L4 | ✅ 15/10 — B3 PHỤ THUỘC OCR; bỏ OCR thì NGANG B2-sub (EM) |
| 2 | **Phần B:** seed 43 — train + eval B2-sub và B3 | ≈ 7 h L4 | ✅ 15/10 (B3 chạy tiếp từ checkpoint-100 sau khi runtime bị xoá) |
| 3 | **Phần B':** seed 44 — như trên | ≈ 7 h L4 | ✅ 16/10 — **CẢI THIỆN ỔN ĐỊNH** (3/3 seed, ΔEM +3,57 ± 0,76 điểm) |
| 4a | **Ô 2×2:** B2-sub seed 42 eval **có OCR** lúc test (test2000) → đủ bảng OCR lúc train × OCR lúc test — notebook `15_ocr_factorial_fulltest_qwen25vl.ipynb` Phần A | ≈ 40 phút L4 | ✅ 16/10 — **TƯƠNG TÁC: CẦN OCR Ở CẢ TRAIN LẪN TEST** (K: +4,35 điểm EM, p = 2,2 × 10⁻⁸) |
| 4b | OCR phần ảnh full test còn thiếu (≈ 1.867 ảnh) — notebook 10 **Phần C** | ≈ 0,5–1 h T4 | ✅ 16/10 — +1867 ảnh, lỗi 0, device gpu; phần cache cũ giữ nguyên (kiểm lại độc lập từ Drive); biên bản `experiments/W09_ocr_cache_fulltest/` |
| 4c | **Full test 10.028 câu:** B3 và B2-sub seed 42 — notebook 15 Phần B | ≈ 5,7 h L4 | ✅ 16/10 — **TÁI LẬP TRÊN FULL TEST** (B2-sub → B3: +2,54 điểm EM, p = 1,1 × 10⁻¹⁴); B3 vượt ViTextBLIP-2 (EM, F1) |
| 5 | **Review 2:** thêm Part 06 (phương pháp OCR + kết quả B3 + độ ổn định seed) vào slide `docs/W07/ViVQA-VLM_Review2_*.pptx`, sửa slide "Plan", cập nhật script thuyết trình | — | ✅ 17/10 — 20 slide (thêm 14–17, sửa 1, 2, 3, 13, 18, 19); script sinh từ ghi chú slide |
| 6 | QLoRA ViVQA lịch 2 epoch, **chọn epoch trên val sạch** (dời từ W8) — notebook `16_qlora_vivqa_2epoch_select_qwen25vl.ipynb` | ≈ 7–7,5 h L4 | ✅ 17/10 — chọn **C3** (2 epoch) trên val sạch 688 câu; test EM 0,7378 → 0,7664 (+2,86 điểm, p = 4,6 × 10⁻⁹) → **CẢI THIỆN** |
| 7 | Sau buổi Review 2: checklist AI_Capstone_Review123 (sheet Review2), gửi mail, upload Drive | — | ⏳ |
| 8 | Đóng tuần: `W09_Team_Report.md` | — | ✅ 17/10 |

## Quyết định đã chốt

| Ngày | Quyết định | Lý do |
|------|-----------|-------|
| 15/10/2026 | **3 seed huấn luyện (42 / 43 / 44)** cho B2-sub và B3 | 2 seed không ước lượng được độ lệch chuẩn có nghĩa; "mean ± std trên 3 seed" là mức tối thiểu hội đồng chấp nhận. Chi phí thêm ≈ 14 h L4 ≈ 22 CU (còn 133 CU ngày 14/10) |
| 15/10/2026 | **Giữ cố định** tập con train 8.000 câu (`train_subset8000_byimage_seed42`); chỉ đổi **seed huấn luyện** (`--seed`) | Tách riêng một nguồn dao động: khởi tạo LoRA + thứ tự batch + dropout. Đổi cả tập con thì không biết chênh lệch đến từ đâu |
| 15/10/2026 | Seed trong tên run eval (`…_seed43`) = **seed huấn luyện** của adapter | Suy luận greedy (không lấy mẫu) nên seed không ảnh hưởng eval |
| 15/10/2026 | **Đưa kết quả B3 vào slide Review 2** (Part 06) | Master §9: Review 2 hỏi "proposed method là gì, có preliminary results không". Kết quả có trước ngày review nên thuộc mốc Review 2 |
| 15/10/2026 | `--seed` không truyền → seed của config (42) → **chữ ký các run W07/W08 không đổi** | Selftest `seed override OK` kiểm điều này; run cũ resume/tái lập được như trước |
| 15/10/2026 | **Thêm ô 2×2** (B2-sub + OCR lúc test) | Phần A cho thấy B3 phụ thuộc OCR, nhưng chưa tách được "train cùng OCR" khỏi "có OCR lúc test" trên cùng tập con. Chỉ ≈ 1 CU |
| 16/10/2026 | Notebook 15 thêm cell **tự trả máy** (`runtime.unassign()`) ở cuối | Notebook 14 xong lúc 1:37 nhưng máy L4 bị giữ tới khi Colab tự ngắt → tốn thêm ≈ 2–3 CU |
| 17/10/2026 | **B2 ViVQA chính thức = C3** (QLoRA lịch 2 epoch, adapter sau epoch 2) thay cho B2 W07 (1 epoch) | Thắng theo luật chọn epoch ghi trước (val sạch) và cải thiện có ý nghĩa trên test (+2,86 EM). Số W07 vẫn giữ trong slide Review 2 (mốc W07) và báo cáo như lịch sử |
| 17/10/2026 | Slide và báo cáo dùng **số full test** làm số chính cho B3; đính chính câu W08 "B3 ngang B2 full" | Full test là tập paper dùng; test2000 có sai số mẫu |
| 15/10/2026 | **Full test chạy cả B3 lẫn B2-sub** (không chỉ B3) | So sánh chính B2-sub → B3 được lặp lại trên 10.028 câu (độ nhạy cao hơn test2000 ≈ 5 lần về số câu), cùng tập paper dùng để so ViTextBLIP-2. Thêm ≈ 2,8 h L4 ≈ 4 CU |

## Luật Phần A — B3 bỏ OCR lúc test (ghi TRƯỚC khi chạy, 15/10/2026)

Adapter B3 seed 42 (W08), eval test2000 với prompt **zero-shot** (không dòng OCR). Hai so sánh ghép cặp (McNemar exact trên EM, ANLS kèm CI95 bootstrap 10.000 lần, seed 42):

| So sánh | Câu hỏi | Kết luận |
|---|---|---|
| B3 (có OCR) → B3 bỏ OCR | B3 có dựa vào dòng OCR không? | EM giảm **và** p < 0,05 → **B3 PHỤ THUỘC OCR**; còn lại → không phụ thuộc có ý nghĩa |
| B2-sub → B3 bỏ OCR | Train cùng OCR có làm model yếu đi khi thiếu OCR? | EM giảm & p < 0,05 → **KÉM HƠN B2-sub**; EM tăng & p < 0,05 → **VẪN TỐT HƠN B2-sub**; còn lại → **NGANG B2-sub** |

Đây là **chẩn đoán**, không phải kết quả cải tiến. Hệ quả cho prototype: nếu "PHỤ THUỘC OCR" + "KÉM HƠN B2-sub" thì demo phải chạy OCR trước và báo rõ khi OCR lỗi.

## Luật độ ổn định theo seed (ghi TRƯỚC khi chạy seed 43/44, 15/10/2026)

Mỗi seed s ∈ {42, 43, 44}: B2-sub(s) và B3(s) cùng tập con, cùng siêu tham số, cùng cấu hình OCR đóng băng (VietOCR · v1 · 600 · 0), eval test2000, so sánh ghép cặp B2-sub(s) → B3(s). Áp bằng `src/eval/seed_stability.py`:

| Kết luận | Điều kiện |
|---|---|
| **CẢI THIỆN ỔN ĐỊNH** | Cả 3 seed: EM(B3) > EM(B2-sub) **và** McNemar exact p < 0,05 |
| **HƯỚNG NHẤT QUÁN, CHƯA ỔN ĐỊNH** | Cả 3 seed EM(B3) > EM(B2-sub), nhưng có seed p ≥ 0,05 |
| **KHÔNG ỔN ĐỊNH** | Có seed EM(B3) ≤ EM(B2-sub) |

- Báo cáo EM, ANLS, ΔEM, ΔANLS dạng **mean ± std** (độ lệch chuẩn mẫu, ddof = 1) trên 3 seed, kèm bảng từng seed (chỉ A đúng / chỉ B đúng, p, ΔANLS CI95).
- **Không** chạy thêm seed sau khi thấy kết quả để "câu" kết luận. Kết quả nào cũng báo cáo đầy đủ.
- Seed 42 tính lại từ predictions W08 phải khớp `../../experiments/W08_B3_vs_B2sub_test2000_L4_seed42/paired_B2sub_vs_B3.json` (notebook 14 tự kiểm).

## Luật ô 2×2 — OCR lúc train × OCR lúc test (ghi TRƯỚC khi chạy ô c01, 15/10/2026)

Cùng test2000, cùng tập con train seed 42, cùng cấu hình OCR đóng băng (VietOCR · v1 · 600 · 0):

| | Test **không** OCR | Test **có** OCR |
|---|---|---|
| Train **không** OCR | c00 = B2-sub (W08) | **c01 = B2-sub + OCR (mới)** |
| Train **có** OCR | c10 = B3 bỏ OCR (W09 Phần A) | c11 = B3 (W08) |

Áp bằng `src/eval/factorial_2x2.py`. So sánh quyết định **K = c01 → c11** (khi lúc test ĐÃ có OCR, train cùng OCR có thêm lợi ích không) và **tương tác** d = (c11 − c10) − (c01 − c00) tính theo từng câu, CI95 bootstrap ghép cặp (10.000, seed 42):

| Kết luận | Điều kiện |
|---|---|
| **TƯƠNG TÁC: CẦN OCR Ở CẢ TRAIN LẪN TEST** | K: EM tăng **và** p < 0,05, **và** cận dưới CI95 tương tác EM > 0 |
| **CỘNG GỘP: TRAIN CÙNG OCR CÓ LỢI RIÊNG** | K: EM tăng và p < 0,05, nhưng CI95 tương tác EM chứa 0 |
| **OCR LÚC TEST LÀ ĐỦ** | K: p ≥ 0,05 → train cùng OCR không thêm lợi ích có ý nghĩa (phải ghi thật vào báo cáo: đóng góp chính khi đó là OCR lúc test, không phải QLoRA + OCR) |
| **TRAIN CÙNG OCR KÉM HƠN** | K: EM giảm và p < 0,05 |

Báo cáo đủ 4 hiệu ứng đơn (c00→c01, c10→c11, c00→c10, c01→c11) kèm McNemar p và ΔANLS CI95, cùng tương tác ANLS.

## Luật full test 10.028 câu (ghi TRƯỚC khi chạy, 15/10/2026)

Run: B3 seed 42 (OCR train + test) và B2-sub seed 42 (không OCR) trên `data/vitextvqa_official/test.json` — cùng adapter, cùng cấu hình như test2000. OCR cache phải phủ đủ 3.353 ảnh test.

1. **Tái lập (điều kiện dừng):** trên 2.000 câu chung, dự đoán của run full phải **trùng ≥ 99%** với run test2000 tương ứng (W08). Dưới 99% → không dùng số liệu, báo Claude.
2. **So sánh chính lặp lại:** B2-sub → B3 trên full test: "TÁI LẬP TRÊN FULL TEST" nếu EM tăng **và** McNemar exact p < 0,05; còn lại báo cáo đúng như kết quả.
3. **So với ViTextBLIP-2** (Table 6, Nguyen et al., *ESWA* 2025: EM 25,48 · F1 53,95): chỉ viết "cao hơn" khi **cận dưới CI95 bootstrap** (10.000, seed 42) của B3 lớn hơn số công bố, xét riêng EM và F1 — như luật W07 (`notebooks/08_…`). Giới hạn: paper không công bố cách chuẩn hoá khi chấm.
4. **Chỉ báo cáo:** B2 full train (W07, full test) → B3; B3 dùng 23% dữ liệu train.
5. **Cache OCR mở rộng không được đổi phần cũ:** dấu vân tay (`cache_fingerprint`) của 4.422 ảnh đã có phải giữ nguyên sau khi OCR thêm (notebook 10 Phần C tự kiểm). Phiên bản `paddlepaddle` của đợt OCR mới được ghi vào biên bản `experiments/W09_ocr_cache_fulltest/`.

## Luật chọn epoch QLoRA ViVQA (ghi TRƯỚC khi chạy, 16/10/2026)

**Vấn đề:** B2 ViVQA (W07) mới 1 epoch. Chọn số epoch bằng cách nhìn test là rò rỉ; val ViVQA lại tách từ `train.csv` **theo câu hỏi**, nên nhiều câu val dùng chung ảnh với câu train → train càng lâu càng "nhớ ảnh", val thiên vị epoch nhiều hơn.

**Tập chọn epoch = val sạch:** các câu `val.json` có ảnh **không** nằm trong `train.json` (giữ thứ tự gốc, không ngẫu nhiên) → `data/vivqa/val_imgdisjoint.json`. Nếu < 300 câu → dùng toàn bộ `val.json` và ghi rõ. Biên bản: `experiments/W09_vivqa_epoch_select_L4_seed42/val_split_manifest.json`.

| Ứng viên | Mô hình | Step |
|---|---|---|
| C1 | B2 W07 (lịch 1 epoch, adapter đã chấm — kiểm sha256) | 675 |
| C2 | Run mới lịch 2 epoch (`--epochs 2`, còn lại y hệt B2), adapter sau epoch 1 | 675 |
| C3 | Cùng run, adapter sau epoch 2 | 1.350 |

1. Chọn ứng viên có **EM cao nhất** trên tập chọn; hoà EM → ít step hơn → C1.
2. Nếu ứng viên đó ≠ C1 nhưng so với C1 (ghép cặp, McNemar exact) **p ≥ 0,05** → **giữ C1** (rẻ hơn — efficiency là ràng buộc thiết kế).
3. **Chỉ** ứng viên được chọn (nếu ≠ C1) được chấm trên test 3.001 câu, **một lần**. So ghép cặp với B2 W07 trên test: "CẢI THIỆN" nếu EM tăng **và** p < 0,05; còn lại báo đúng như kết quả. Chọn C1 → không chạm test, B2 W07 (EM 0,7378) giữ nguyên làm B2 ViVQA chính thức.
4. Lưu ý so sánh: C2 ≠ C1 dù cùng 675 step — lịch learning rate của C2 trải trên 1.350 step nên lúc hết epoch 1 LR còn cao hơn.

## Thay đổi code

| File | Thay đổi | Kiểm thử |
|---|---|---|
| `src/adaptation/train_qlora.py` | `--seed` ghi đè `run.seed`; seed nằm trong chữ ký run (seed khác = run khác, không resume nhầm); dòng log đầu in `seed=` | selftest `seed override OK`: seed 42 mặc định trùng chữ ký cũ, seed 43 khác chữ ký |
| `src/eval/seed_stability.py` | Gộp `paired_*.json` qua các seed, mean ± std, áp luật trên | `--selftest` (4 tình huống luật + chặn số câu khác nhau) |
| `notebooks/14_b3_seed_stability_qwen25vl.ipynb` | Phần A (bỏ OCR) → Phần B seed 43 → B' seed 44 → Phần C gộp; bỏ qua phần đã xong; train lỗi thì dừng, không eval | Mô phỏng offline mục 4 và mục 8 bằng predictions W08 |
| `src/eval/factorial_2x2.py` | Bảng 2×2 OCR lúc train × OCR lúc test: 4 hiệu ứng đơn + tương tác (difference-in-differences theo câu) + luật trên | `--selftest` (4 tình huống luật + chặn lệch question_id) |
| `notebooks/10_ocr_cache_vitextvqa.ipynb` | Thêm **Phần C** (mục 11–12): OCR phần ảnh full test còn thiếu, kiểm dấu vân tay phần cache cũ, ghi biên bản W09 | Chạy trên T4 |
| `notebooks/15_ocr_factorial_fulltest_qwen25vl.ipynb` | Phần A: ô c01 + bảng 2×2. Phần B: B3 + B2-sub full test, kiểm tái lập ≥ 99%, so ViTextBLIP-2, so sánh chính lặp lại | Mô phỏng offline mục kết quả |
| `notebooks/16_qlora_vivqa_2epoch_select_qwen25vl.ipynb` | Tạo val sạch (ảnh không trùng train) → eval C1 → train lịch 2 epoch → eval C2, C3 → chọn theo luật → test chỉ khi chọn C2/C3 → tự trả máy | Mô phỏng offline mục chọn epoch (4 tình huống luật) |

## Ngân sách

| Hạng mục | GPU | CU (1,54 CU/h L4) |
|---|---|---|
| Phần A | ≈ 0,7 h | ≈ 1 |
| Seed 43 + 44 | ≈ 14 h | ≈ 22 |
| Ô 2×2 | ≈ 0,7 h | ≈ 1 |
| Full test B3 + B2-sub (+ OCR T4) | ≈ 6,5 h | ≈ 10 |
| ViVQA lịch 2 epoch + chọn epoch | ≈ 7–7,5 h | ≈ 11 |
| **Tổng W9** | **≈ 29 h** | **≈ 46** (còn ≈ 87 CU cho W10–W15) |

## Kết quả

### Phần A — B3 bỏ OCR lúc test (15/10/2026, L4, bf16, 4-bit, seed 42)

Nguồn: `../../experiments/W09_B3_ocr_dependency_test2000_L4_seed42/` + eval `W09_B3noocr_eval_qlora_ocr_vietocr_v1_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed42`.

| Run | OCR lúc train | OCR lúc test | EM | ANLS | token-F1 (phụ) | s/câu |
|---|---|---|---|---|---|---|
| B2-sub | không | không | 0,4555 | 0,6848 | 0,6692 | 1,021 |
| B3 | có | có | 0,4925 | 0,7038 | 0,6981 | 1,001 |
| B3 bỏ OCR | có | **không** | 0,4500 | 0,6746 | 0,6617 | 0,979 |

| So sánh | Chỉ A đúng / chỉ B đúng | McNemar p | ΔANLS (CI95) | Kết luận theo luật |
|---|---|---|---|---|
| B3 → B3 bỏ OCR | 147 / 62 | 3,8 × 10⁻⁹ | −0,0292 [−0,0401; −0,0185] | **B3 PHỤ THUỘC OCR** |
| B2-sub → B3 bỏ OCR | 63 / 52 | 0,351 | −0,0101 [−0,0194; −0,0007] | **NGANG B2-sub** (theo EM) |

**Diễn giải:**
- Bỏ OCR lúc test làm B3 mất 4,25 điểm EM — nhiều hơn toàn bộ phần lợi +3,70 điểm so với B2-sub. → Lợi ích của B3 đến từ việc model **thật sự dùng dòng OCR**, không phải do train "khéo hơn" nói chung: bằng chứng cơ chế cho kết quả chính W08.
- Thiếu OCR, B3 lùi về mức B2-sub (EM ngang, p = 0,35): suy giảm êm, không sụp.
- **Giới hạn nhỏ:** ANLS của B3 bỏ OCR thấp hơn B2-sub −0,010 (CI95 không chứa 0) → train cùng OCR có thể làm model hơi giảm khả năng tự đọc chữ chính xác từng ký tự. Luật kết luận theo EM nên ghi "NGANG", nhưng phải nêu trong báo cáo.
- Cùng với W08 bước 6 (chèn OCR chỉ lúc suy luận không giúp B1/B2), kết quả **gợi ý** OCR cần có ở **cả train lẫn test**. Chưa khẳng định được vì thiếu ô "B2-sub train không OCR + test có OCR" trên cùng tập con.
- Hệ quả prototype: luôn chạy OCR trước khi hỏi model, hiển thị văn bản OCR, cảnh báo khi OCR rỗng/lỗi (chất lượng khi đó ≈ B2-sub).

### Ô 2×2 — OCR lúc train × OCR lúc test (16/10/2026, L4, bf16, 4-bit, seed 42, test2000)

Nguồn: `../../experiments/W09_ocr_factorial_2x2_test2000_L4_seed42/factorial_summary.json` + eval `W09_B2subocr_eval_qlora_r16_sub8k_ep1_vitextvqa_test2000_bf16_L4_seed42` (ô c01, mới).

| EM | Test không OCR | Test có OCR |
|---|---|---|
| Train không OCR | c00 = 0,4555 | c01 = 0,4490 |
| Train có OCR | c10 = 0,4500 | c11 = 0,4925 |

| Hiệu ứng | So sánh | Chỉ A đúng / chỉ B đúng | McNemar p | ΔEM (điểm) | ΔANLS (CI95) |
|---|---|---|---|---|---|
| OCR lúc test, model train không OCR | c00 → c01 | 160 / 147 | 0,49 | −0,65 | −0,0070 [−0,0202; +0,0063] |
| OCR lúc test, model train có OCR | c10 → c11 | 62 / 147 | 3,8 × 10⁻⁹ | +4,25 | +0,0292 [+0,0185; +0,0401] |
| Train cùng OCR, test không OCR | c00 → c10 | 63 / 52 | 0,35 | −0,55 | −0,0101 [−0,0194; −0,0007] |
| **Train cùng OCR, test có OCR (K)** | **c01 → c11** | **77 / 164** | **2,2 × 10⁻⁸** | **+4,35** | **+0,0261 [+0,0141; +0,0383]** |

Tương tác (theo từng câu, CI95 bootstrap ghép cặp): EM +0,0490 [+0,0320; +0,0660] · ANLS +0,0362 [+0,0225; +0,0503].

**Kết luận theo luật ghi trước: TƯƠNG TÁC — CẦN OCR Ở CẢ TRAIN LẪN TEST.**

**Diễn giải:**
- Mỗi yếu tố đứng riêng không giúp: chèn OCR lúc test cho model chưa train cùng OCR (−0,65 điểm, p = 0,49 — khớp W08 bước 6), hay train cùng OCR mà test không có OCR (−0,55 điểm, p = 0,35).
- Chỉ khi có cả hai mới tăng +4,25–4,35 điểm EM. Đóng góp chính là **QLoRA nhận biết OCR (OCR-aware fine-tuning)**: model học cách dùng dòng OCR (khi nào tin OCR, khi nào tin ảnh), không phải "chỉ cần thêm OCR vào prompt".
- **Đối chiếu H6** (Review 1: OCR-enhanced prompting tăng ANLS ≥ +3 điểm và cộng hưởng với QLoRA; bác bỏ nếu ΔANLS < +1): cộng hưởng với QLoRA **được ủng hộ mạnh**; mức tăng ANLS của B3 so với B2-sub = +1,55 ± 0,40 điểm qua 3 seed — **dưới ngưỡng +3** nhưng trên ngưỡng bác bỏ +1. → **H6 ủng hộ một phần**: lợi ích rõ ở EM (+3,57 ± 0,76 điểm) hơn ở ANLS.

### Full test 10.028 câu — B3 và B2-sub (16/10/2026, L4, bf16, 4-bit, seed 42)

Nguồn: `../../experiments/W09_fulltest_B3_vs_B2sub_L4_seed42/` (`fulltest_summary.json` + 2 file `paired_*`) + eval `W09_B3_eval_…_testfull_…`, `W09_B2sub_eval_…_testfull_…`. **Tái lập:** trên 2.000 câu chung, dự đoán trùng 100% với run test2000 (cả B3 lẫn B2-sub).

| Run | Train | OCR | EM | ANLS | F1 (phụ) |
|---|---|---|---|---|---|
| B1 zero-shot (W07) | — | không | 29,12 | 47,60 | 53,89 |
| B2 (W07) | full 35.159 câu | không | 50,26 | 72,02 | 71,16 |
| B2-sub | 8.000 câu | không | 46,56 | 68,90 | 67,83 |
| **B3** | 8.000 câu | VietOCR · v1 | **49,10** | **70,47** | **70,03** |
| ViTextBLIP-2 (paper, Table 6) | | | 25,48 | — | 53,95 |

| So sánh | Chỉ A đúng / chỉ B đúng | McNemar p | ΔEM (điểm) | ΔANLS (CI95) | Kết luận |
|---|---|---|---|---|---|
| **B2-sub → B3** | **417 / 672** | **1,1 × 10⁻¹⁴** | **+2,54** | **+0,0157 [+0,0106; +0,0208]** | **TÁI LẬP TRÊN FULL TEST** |
| B2 full → B3 (chỉ báo cáo) | 784 / 668 | 0,0025 | −1,16 | −0,0155 [−0,0215; −0,0098] | B3 thấp hơn B2 full có ý nghĩa |

So với ViTextBLIP-2 (luật W07: cận dưới CI95 > số công bố): B3 EM 49,10 [48,11; 50,07], F1 70,03 [69,30; 70,78] → **CAO HƠN** cả EM lẫn F1. Giới hạn: paper không công bố cách chuẩn hoá đáp án.

**Diễn giải:**
- So sánh chính lặp lại trên tập lớn gấp 5 lần: B3 hơn B2-sub +2,54 điểm EM (p = 1,1 × 10⁻¹⁴), ANLS CI95 không chứa 0. Mức lợi ích nhỏ hơn test2000 seed 42 (+3,70) và nằm ở mép dưới dải 3 seed (+2,75 → +4,25) → báo cáo **số full test** làm số chính.
- **Đính chính W08:** trên test2000, B3 "ngang B2 full về EM" (p = 0,65). Trên full test, B3 **thấp hơn** B2 full 1,16 điểm EM (p = 0,0025). Phát biểu đúng: với 23% dữ liệu train, OCR-aware QLoRA bù được ≈ **69%** khoảng cách EM do cắt dữ liệu, (49,10 − 46,56) / (50,26 − 46,56); trên test2000 con số này là 89%.
- Tốc độ suy luận B3 ≈ B2-sub (0,98 s/câu, không tính OCR); OCR chạy trước và cache, ≈ 0,85 s/ảnh trên GPU.

### Độ ổn định theo seed (16/10/2026, L4, bf16, 4-bit, test2000)

Nguồn: `../../experiments/W09_B3_seed_stability_test2000_L4/` (`seed_stability_summary.json` + 3 file `paired_B2sub_vs_B3_seed{42,43,44}.json`); train `W09_B2sub_qlora_…_seed{43,44}`, `W09_B3_qlora_ocr_…_seed{43,44}`; eval `W09_B2sub_eval_…_seed{43,44}`, `W09_B3_eval_…_seed{43,44}`. Seed 42 = run W08, tính lại từ predictions khớp 81 / 155.

| Seed | EM B2-sub | EM B3 | ΔEM (điểm) | Chỉ B2-sub đúng / chỉ B3 đúng | McNemar p | ΔANLS (CI95) |
|---|---|---|---|---|---|---|
| 42 | 0,4555 | 0,4925 | +3,70 | 81 / 155 | 1,7 × 10⁻⁶ | +0,0191 [+0,0073; +0,0309] |
| 43 | 0,4475 | 0,4900 | +4,25 | 77 / 162 | 4,0 × 10⁻⁸ | +0,0161 [+0,0040; +0,0280] |
| 44 | 0,4605 | 0,4880 | +2,75 | 74 / 129 | 1,4 × 10⁻⁴ | +0,0112 [+0,0000; +0,0224] |
| **mean ± std** | 45,45 ± 0,66 | 49,02 ± 0,23 | **+3,57 ± 0,76** | | | +0,0155 ± 0,0040 |

ANLS mean ± std: B2-sub 0,6857 ± 0,0031 · B3 0,7011 ± 0,0024.

**Kết luận theo luật ghi trước: CẢI THIỆN ỔN ĐỊNH** — cả 3 seed EM(B3) > EM(B2-sub) và p ≤ 1,4 × 10⁻⁴.

**Diễn giải:**
- Lợi ích trung bình +3,57 điểm EM, lớn gấp ≈ 5 lần độ lệch chuẩn của B2-sub giữa các seed (0,66) → không phải do khởi tạo / thứ tự batch "may mắn".
- ANLS cùng chiều ở cả 3 seed, nhưng seed 44 có cận dưới CI95 = +0,0000 (sát 0) → lợi ích ANLS **yếu hơn** lợi ích EM. Luật kết luận dựa trên EM; ANLS phải báo cáo đúng như vậy.
- Độ lệch chuẩn EM của B3 (0,23) nhỏ hơn B2-sub (0,66). Chỉ 3 seed nên **không** kết luận B3 ổn định hơn — chỉ ghi nhận.

| Chi phí train | Seed | Giờ train | s/step | VRAM đỉnh (GB) | Loss cuối | Chạy tiếp từ |
|---|---|---|---|---|---|---|
| B2-sub | 42 | 2,76 | 19,89 | 6,85 | 0,7161 | — |
| B3 | 42 | 2,98 | 21,48 | 8,25 | 0,6605 | — |
| B2-sub | 43 | 2,69 | 19,40 | 6,84 | 0,7105 | — |
| B3 | 43 | 2,40\* | 21,56 | 8,24 | 0,6044\* | checkpoint-100 |
| B2-sub | 44 | 2,79 | 20,06 | 6,81 | 0,7162 | — |
| B3 | 44 | 3,01 | 21,66 | 8,25 | 0,6603 | — |

\* B3 seed 43: runtime bị xoá giữa chừng (15/10 ≈ 15:30), chạy tiếp từ checkpoint-100 (trạng thái optimizer + lịch learning rate + RNG khôi phục theo checkpoint). Giờ train và loss cuối chỉ tính **phiên cuối** (step 101–500) → không so trực tiếp với các run khác. Chi phí một run trọn: B2-sub ≈ 2,7–2,8 h, B3 ≈ 3,0 h (+8% s/step, +1,4 GB VRAM).

### QLoRA ViVQA — chọn epoch trên val sạch (17/10/2026, L4, bf16, 4-bit, seed 42)

Nguồn: `../../experiments/W09_vivqa_epoch_select_L4_seed42/` (`epoch_select_summary.json`, `val_split_manifest.json`, 3 file `paired_*`, `stratified_B2W07_vs_C3_vivqa_test.json`) + train `W09_B2_qlora_r16_qwen25vl_vivqa_ep2_bf16_L4_seed42` + eval `W09_C{1,2,3}_eval_…_valclean_…`, `W09_C3_eval_…_test_…`.

**Tập chọn epoch:** val 1.200 câu, trong đó 512 câu (43%) dùng chung ảnh với train → val sạch **688 câu / 660 ảnh** (≥ 300, dùng được).

| Ứng viên | Step | EM val sạch | ANLS val sạch | So với C1 (chỉ C1 đúng / chỉ ứng viên đúng, p) |
|---|---|---|---|---|
| C1 · B2 W07 (lịch 1 epoch) | 675 | 0,7311 | 0,7476 | — |
| C2 · lịch 2 epoch, sau epoch 1 | 675 | 0,7151 | 0,7317 | 30 / 19, p = 0,15 |
| **C3 · lịch 2 epoch, sau epoch 2** | **1.350** | **0,7703** | **0,7832** | **11 / 38, p = 1,4 × 10⁻⁴** |

**Chọn theo luật: C3** (EM cao nhất và hơn C1 có ý nghĩa). C2 thấp hơn C1 (không có ý nghĩa) — hợp lý vì sau epoch 1 của lịch 2 epoch, learning rate còn cao, chưa giảm hết.

**Test 3.001 câu (chạm một lần):** B2 W07 0,7378 → **C3 0,7664** (+2,86 điểm EM; 65 / 151; p = 4,6 × 10⁻⁹); ANLS 0,7566 → 0,7803 (+0,0238, CI95 [+0,0147; +0,0330]) → **CẢI THIỆN**. Chi phí train 1.350 step × 16,2 s = 6,1 h L4 (gấp đôi), VRAM đỉnh 6,62 GB (không đổi).

**Kiểm tra sau (post-hoc, không phải luật) — có phải do nhớ ảnh không?** `stratify_by_overlap.py`:

| Tập con test | n | EM B2 W07 | EM C3 | ΔEM (CI95) | McNemar p |
|---|---|---|---|---|---|
| Tất cả | 3.001 | 0,7378 | 0,7664 | +0,029 [+0,020; +0,038] | 4,6 × 10⁻⁹ |
| Ảnh đã thấy trong train | 1.356 | 0,7463 | 0,7751 | +0,029 [+0,015; +0,044] | 1,1 × 10⁻⁴ |
| Ảnh chưa thấy | 1.645 | 0,7307 | 0,7593 | +0,029 [+0,016; +0,041] | 1,7 × 10⁻⁵ |
| **Sạch** (ảnh và cặp hỏi–đáp chưa thấy) | **1.219** | **0,6907** | **0,7227** | **+0,032 [+0,016; +0,048]** | **1,5 × 10⁻⁴** |

→ Mức tăng **đều** giữa ảnh đã thấy và chưa thấy, giữ nguyên trên tập sạch → **không phải do nhớ ảnh**. B2 ViVQA chính thức từ W09 = C3.

## Link experiment log

- Phần A: `experiments/W09_B3_ocr_dependency_test2000_L4_seed42/` + eval `W09_B3noocr_eval_…_seed42`
- Seed: `experiments/W09_B3_seed_stability_test2000_L4/` + train `W09_B2sub_qlora_…_seed{43,44}`, `W09_B3_qlora_ocr_…_seed{43,44}` + eval `W09_B2sub_eval_…`, `W09_B3_eval_…`
- Ô 2×2: `experiments/W09_ocr_factorial_2x2_test2000_L4_seed42/` + eval `W09_B2subocr_eval_…_seed42`
- OCR full test: `experiments/W09_ocr_cache_fulltest/ocr_cache_manifest.json`
- Full test: `experiments/W09_fulltest_B3_vs_B2sub_L4_seed42/` + eval `W09_B3_eval_…_testfull_…`, `W09_B2sub_eval_…_testfull_…`
- ViVQA chọn epoch: `experiments/W09_vivqa_epoch_select_L4_seed42/` + train `W09_B2_qlora_r16_qwen25vl_vivqa_ep2_bf16_L4_seed42` + eval `W09_C{1,2,3}_eval_…_valclean_…`, `W09_C3_eval_…_test_…`
- Adapter (không lên git, trên Drive): `adapters/W09_*_seed{43,44}/epoch_1`, `adapters/W09_B2_qlora_r16_qwen25vl_vivqa_ep2_bf16_L4_seed42/epoch_{1,2}`

## Deliverables tuần này

| File | Vai trò |
|------|---------|
| [W09_Team_Report.md](W09_Team_Report.md) | Báo cáo tuần: việc đã làm, số liệu chính, blocker, kế hoạch W10 |
| README này | Quyết định, luật ghi trước, kết quả từng bước |
| [../W07/ViVQA-VLM_Review2_EN.pptx](../W07/ViVQA-VLM_Review2_EN.pptx) | **Slide Review 2 cập nhật** (20 slide, thêm Part 06 · OCR-aware QLoRA) |
| [../W07/ViVQA-VLM_Review2_TV.pptx](../W07/ViVQA-VLM_Review2_TV.pptx) | Bản tiếng Việt để đọc hiểu |
| [../W07/ViVQA-VLM_Review2_Script.txt](../W07/ViVQA-VLM_Review2_Script.txt) | Kịch bản nói, sinh từ ghi chú slide |

## Trạng thái

✅ **Đóng tuần 17/10/2026** (trừ việc 7: checklist Review 2, làm sau buổi review). Kết quả chính: **QLoRA nhận biết OCR (B3) cải thiện ổn định** — 3/3 seed (+3,57 ± 0,76 EM), tái lập trên full test (+2,54 EM, p = 1,1 × 10⁻¹⁴), và **cần OCR ở cả train lẫn test** (tương tác +4,9 EM). H6 chỉ ủng hộ một phần (ANLS +1,55 < +3). B3 thấp hơn B2 dữ liệu đầy đủ 1,16 EM (đính chính W08). ViVQA: B2 chính thức đổi sang 2 epoch (test +2,86 EM, không do nhớ ảnh).

---

# FILE: docs/W09/W09_Team_Report.md

# W09 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W09 — Củng cố B3 (QLoRA nhận biết OCR) + Review 2 |
| Thời gian thực | 15–17/10/2026 (tuần thực 6) |
| Mốc kế tiếp | **Review 2** — tuần thực 7 (19–25/10/2026); ngày cụ thể bộ môn báo qua email trong tuần đó |
| Phần cứng | Colab Pro: NVIDIA L4 (train/eval) · Tesla T4 (OCR) |
| Commit cuối tuần | commit đóng tuần sau `7701fd3` (xem mục "Dấu vết commit") |

## Mục tiêu

Kết quả chính W08 — B3 (QLoRA train cùng OCR) thắng B2-sub (cùng QLoRA, cùng 8.000 câu train, không OCR) — còn 4 lỗ hổng mà hội đồng sẽ hỏi. Tuần này lấp cả 4, rồi đưa phương pháp lên slide Review 2:

1. **Mới 1 seed** → chạy thêm seed 43, 44, kết luận theo luật ghi trước.
2. **Chưa biết B3 có dựa vào OCR không** → eval B3 khi bỏ OCR lúc test.
3. **Chưa tách được** "train cùng OCR" khỏi "có OCR lúc test" → hoàn chỉnh bảng 2×2.
4. **Mới test2000** → chạy lại trên full test 10.028 câu (tập paper dùng).
5. Việc dời từ W08: QLoRA ViVQA 2 epoch, **chọn epoch không chạm test**.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | `--seed` cho `train_qlora.py` (seed nằm trong chữ ký run; mặc định 42 giữ nguyên chữ ký W07/W08); `seed_stability.py`; notebook 14 | ✅ `c69794a` |
| 2 | Phần A: B3 seed 42 **bỏ OCR** lúc test | ✅ **B3 phụ thuộc OCR** (−4,25 EM, p = 3,8 × 10⁻⁹); bỏ OCR thì **ngang B2-sub** về EM |
| 3 | Seed 43, 44 cho B2-sub và B3 (tập con train giữ nguyên) | ✅ **Cải thiện ổn định**: 3/3 seed, +3,57 ± 0,76 EM |
| 4 | `factorial_2x2.py` + notebook 15 Phần A: ô "train không OCR, test có OCR" | ✅ **Tương tác**: cần OCR ở cả train lẫn test (+4,9 EM, CI95 [+3,2; +6,6]) |
| 5 | Notebook 10 Phần C: OCR thêm 1.867 ảnh full test (VietOCR, cùng engine W08) | ✅ 0 lỗi; dấu vân tay 4.422 ảnh cũ giữ nguyên (kiểm độc lập từ Drive) |
| 6 | Notebook 15 Phần B: B3 + B2-sub trên **full test 10.028 câu** | ✅ **Tái lập**: +2,54 EM, p = 1,1 × 10⁻¹⁴; trùng 100% với run test2000 trên 2.000 câu chung |
| 7 | Notebook 16: QLoRA ViVQA lịch 2 epoch, chọn epoch trên **val sạch** (ảnh không trùng train) | ✅ chọn C3; test 0,7378 → **0,7664** (+2,86, p = 4,6 × 10⁻⁹) |
| 8 | Slide Review 2: thêm Part 06 (4 slide), sửa slide 1, 2, 3, 13, 18, 19; script sinh từ ghi chú slide | ✅ 20 slide, bản EN + TV |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| Dataset / OCR | Mở rộng OCR cache sang full test (5); val sạch ViVQA (7) |
| PEFT / Training | `--seed` (1); seed 43/44 (3); QLoRA ViVQA 2 epoch (7) |
| Evaluation | Phần A (2), bảng 2×2 + tương tác (4), full test + so ViTextBLIP-2 (6), luật chọn epoch + phân tầng rò rỉ (7) |
| VLM / Baseline | B2 ViVQA chính thức đổi sang C3 (7) |

### Definition of Done

- [x] Độ ổn định theo seed: 3 seed, luật ghi trước, mean ± std
- [x] B3 khi bỏ OCR lúc test (phụ thuộc OCR) — luật ghi trước
- [x] Ablation 2×2 OCR lúc train × lúc test, có kiểm định tương tác
- [x] B3 trên full test 10.028 câu, kiểm tái lập ≥ 99%
- [x] QLoRA ViVQA 2 epoch, chọn epoch trên val sạch, test chạm một lần
- [x] Slide Review 2 có phương pháp đề xuất + kết quả sơ bộ
- [x] Mọi số liệu truy được về file trong `experiments/`; mọi luật kết luận commit lên git **trước** khi có số
- [ ] Checklist AI_Capstone_Review123 (sheet Review2) → **sau buổi Review 2**

## Kết quả / số liệu

### 1. Độ ổn định theo seed (test2000)

| Seed | EM B2-sub | EM B3 | ΔEM | p |
|---|---|---|---|---|
| 42 | 0,4555 | 0,4925 | +3,70 | 1,7 × 10⁻⁶ |
| 43 | 0,4475 | 0,4900 | +4,25 | 4,0 × 10⁻⁸ |
| 44 | 0,4605 | 0,4880 | +2,75 | 1,4 × 10⁻⁴ |
| **mean ± std** | 45,45 ± 0,66 | 49,02 ± 0,23 | **+3,57 ± 0,76** | |

ANLS: +0,0155 ± 0,0040 (seed 44 cận dưới CI95 sát 0). Kết luận theo luật: **CẢI THIỆN ỔN ĐỊNH**.

### 2. Bảng 2×2 — OCR lúc train × OCR lúc test (test2000, seed 42)

| EM | Test không OCR | Test có OCR |
|---|---|---|
| Train không OCR | 0,4555 | 0,4490 |
| Train có OCR | 0,4500 | **0,4925** |

Chỉ OCR lúc test: −0,65 (p = 0,49). Chỉ OCR lúc train: −0,55 (p = 0,35). Cả hai: +4,35 (p = 2,2 × 10⁻⁸). Tương tác **+4,9 EM**, CI95 [+3,2; +6,6]. → Đóng góp là **dạy model dùng OCR** (OCR-aware fine-tuning), không phải "thêm OCR vào prompt".

### 3. Full test 10.028 câu

| Run | Dữ liệu train | EM | ANLS | F1 (phụ) |
|---|---|---|---|---|
| B1 zero-shot | — | 29,12 | 47,60 | 53,89 |
| B2 (W07) | 100% | 50,26 | 72,02 | 71,16 |
| B2-sub | 23% | 46,56 | 68,90 | 67,83 |
| **B3** | 23% + OCR | **49,10** | **70,47** | **70,03** |
| ViTextBLIP-2 (paper Table 6) | 100% | 25,48 | — | 53,95 |

- B2-sub → B3: **+2,54 EM** (417 / 672, p = 1,1 × 10⁻¹⁴), ANLS +0,0157 [+0,0106; +0,0208] → **tái lập**.
- B3 cao hơn ViTextBLIP-2 cả EM và F1 (cận dưới CI95 48,11 / 69,30).
- **Đính chính W08:** B3 **thấp hơn** B2 dữ liệu đầy đủ 1,16 EM (p = 0,0025). Phát biểu đúng: với 23% dữ liệu và 25% thời gian train, OCR bù ≈ **69%** khoảng cách EM do cắt dữ liệu.

### 4. ViVQA — chọn epoch

| Ứng viên | EM val sạch (688 câu) |
|---|---|
| C1 · B2 W07, 1 epoch | 0,7311 |
| C2 · lịch 2 epoch, sau epoch 1 | 0,7151 (p = 0,15 so với C1) |
| **C3 · lịch 2 epoch, sau epoch 2** | **0,7703** (p = 1,4 × 10⁻⁴) |

Test: 0,7378 → **0,7664** (+2,86, p = 4,6 × 10⁻⁹). Kiểm sau: mức tăng đều giữa ảnh đã thấy (+2,9) và chưa thấy (+2,9), trên tập sạch +3,2 (p = 1,5 × 10⁻⁴) → **không phải nhớ ảnh**. B2 ViVQA chính thức = C3.

### 5. Đối chiếu giả thuyết

| Giả thuyết | Trạng thái | Lý do |
|---|---|---|
| H6 · OCR prompting tăng ANLS ≥ +3 và cộng hưởng với QLoRA | **Ủng hộ một phần** | Cộng hưởng mạnh (tương tác +4,9 EM); nhưng ANLS +1,55 ± 0,40 (< +3, > ngưỡng bác bỏ +1) |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|--------|-----|-------------|
| B1 | B3 vẫn thấp hơn B2 dữ liệu đầy đủ (−1,16 EM, full test) | Trung bình | W10: train B3 trên toàn bộ train ViTextVQA (≈ 13 h L4, ≈ 20 CU) — câu hỏi: OCR-aware có vượt B2 khi đủ dữ liệu không |
| B2 | Lợi ích ANLS nhỏ hơn EM (H6 một phần) | Thấp | Ghi giới hạn; phân tích lỗi B3 theo nhóm (dấu thanh, lệch ký tự) ở W11 |
| B3 | B3 phụ thuộc OCR: OCR hỏng → lùi về mức B2-sub | Thấp | Prototype: chạy OCR trước, hiển thị văn bản OCR, cảnh báo khi OCR rỗng |
| B4 | Colab mất kết nối tab / runtime bị xoá nhầm (15/10) | Đã xử lý | Notebook resume được từ checkpoint; notebook 15, 16 tự trả máy; quy tắc: không mở menu runtime khi đang train |
| B5 | Selftest crash trên console Windows (cp1252) | Thấp | Chạy với `PYTHONIOENCODING=utf-8`; sửa gốc (`sys.stdout.reconfigure`) khi tiện |

**Ngân sách Colab (ước tính):** W09 dùng ≈ 49 CU (notebook 14 ≈ 26, notebook 15 ≈ 11, notebook 16 ≈ 11, OCR T4 ≈ 2) → còn ≈ **84 CU ≈ 55 giờ L4** cho W10–W15 (kiểm lại số thật trong Colab → Resources).

## Kế hoạch tuần sau (W10)

1. **Review 2** (tuần thực 7): tập nói slide 20 trang (≈ 22–26 phút); sau buổi review điền checklist AI_Capstone_Review123 sheet Review2, gửi mail, upload Drive.
2. B3 trên toàn bộ train ViTextVQA (đóng blocker B1).
3. Ablation cấu hình LoRA: rank và module gắn LoRA (H3, H4).
4. Bắt đầu Results (R6) bản tiếng Việt cho phần OCR — số liệu W09 đã ổn định.

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `c69794a` | Bước 0: `--seed`, `seed_stability.py`, notebook 14, luật Phần A + luật seed ghi trước |
| `7697750` | Kết quả Phần A (B3 bỏ OCR) |
| `701f87f` | Bước 1: luật 2×2 + luật full test ghi trước; `factorial_2x2.py`; notebook 15; notebook 10 Phần C |
| `c9894f5` | Kết quả 3 seed; notebook 15 tự trả máy |
| `ab70867` | Biên bản OCR full test + kiểm độc lập |
| `f871872` | Bước 3: luật chọn epoch ViVQA ghi trước; notebook 16 |
| `1777b08` | Kết quả bảng 2×2 |
| `c3c3937` | Kết quả full test + đính chính W08 |
| `7701fd3` | Kết quả chọn epoch ViVQA + phân tầng rò rỉ |
| *(commit đóng tuần)* | Slide Review 2 (20 slide) + README + Team Report W09 |

## Link experiment log

- Phần A: `experiments/W09_B3_ocr_dependency_test2000_L4_seed42/`
- Seed: `experiments/W09_B3_seed_stability_test2000_L4/` + các run `W09_B2sub_*_seed{43,44}`, `W09_B3_*_seed{43,44}`
- Bảng 2×2: `experiments/W09_ocr_factorial_2x2_test2000_L4_seed42/`
- OCR full test: `experiments/W09_ocr_cache_fulltest/`
- Full test: `experiments/W09_fulltest_B3_vs_B2sub_L4_seed42/`
- ViVQA chọn epoch: `experiments/W09_vivqa_epoch_select_L4_seed42/`
- Adapter (không lên git, trên Drive): `adapters/W09_*`

---

# FILE: docs/03_report/R1_Introduction.md

# R1 — Introduction

> Khung thesis. Điền dần từ W01–W02 và hoàn thiện W14. Không thay file `.docx` có sẵn trong thư mục này.

## 1. Motivation

Visual Question Answering tiếng Việt: ảnh + câu hỏi → câu trả lời. VLM mã nguồn mở thiên về tiếng Anh; scene-text và hallucination còn khó.

## 2. Problem statement

```text
Vietnamese image + Vietnamese question → VLM → Vietnamese answer
```

## 3. Research gap (trỏ W02)

G1 English-centric · G2 Vietnamese VQA/scene-text · G3 PEFT cho VLM vi · G4 hallucination.

**Gap nhóm chọn:** _…_

## 4. Objectives & contribution

Reproduce **đúng 1** baseline (BLIP-2 / Florence-2 / Qwen2.5-VL). Đề xuất ≥ 1 cải tiến **PEFT-only** (LoRA/QLoRA 4/8-bit, OCR-prompt, RAG, prompt-opt), so sánh **cùng điều kiện**, kèm ablation, error analysis, hallucination analysis.

## 5. Scope

In: ViVQA, ViTextVQA, Colab Pro/Premium hoặc GPU trường. Out: full fine-tuning.

## 6. Thesis structure

R3 existing systems · R4 methodology · R5 implementation · R6 results · R7 discussion.

---

# FILE: docs/03_report/R3_Existing_Systems.md

# R3 — Existing Systems / Related Work

> Nguồn: literature matrix W02 + `02_References/Reading_Notes/`. (Không có R2 trong khung này — related work gộp chương này trừ khi khoa bắt buộc chương 2 riêng.)

## 1. Visual Question Answering

_…_

## 2. Vision-Language Models

### 2.1 BLIP-2
_…_

### 2.2 Florence-2
_…_

### 2.3 Qwen2.5-VL
_…_

### 2.4 LLaVA và hướng instruction tuning
_…_

## 3. Vietnamese VQA benchmarks

### 3.1 ViVQA
_…_

### 3.2 ViTextVQA (scene-text)
_…_

## 4. OCR cho tiếng Việt

PaddleOCR · VietOCR · lỗi diacritics.

## 5. Parameter-efficient fine-tuning

LoRA · QLoRA (4/8-bit) · **không** full FT trong đề tài này.

## 6. Hallucination trong LVLM

POPE và hạn chế khi chuyển sang VQA sinh câu.

## 7. Positioning

Bảng so sánh hệ thống hiện có vs đề tài (PEFT-only, vi, ablation).

---

# FILE: docs/03_report/R4_Methodology.md

# R4 — Phương pháp (Methodology)

> **Trạng thái:** bản tiếng Việt, viết cuối W6 (2026-09-29). Sẽ chuyển sang tiếng Anh khi nội dung đã chốt và không còn lỗi.
> **Ký hiệu:** ⏳ W7 = quyết định còn mở, chờ đo trên GPU L4 ở tuần 7. Được ghi tường minh kèm lý do, không viết như thể đã chốt.
> **Nguồn số liệu:** mọi con số trong tài liệu này truy được về một file trong `experiments/` hoặc `src/` của repo, ghi ở cuối mỗi mục.

---

## 1. Tổng quan

Đề tài kiểm chứng một giả thuyết kỹ thuật: **có thể thích nghi một VLM mã nguồn mở cho hỏi đáp trực quan tiếng Việt bằng tinh chỉnh hiệu quả tham số (PEFT), dưới ngân sách phần cứng của Colab**, và đo được cải thiện so với baseline zero-shot dưới điều kiện y hệt.

Quy trình gồm năm bước, mỗi bước chỉ bắt đầu khi bước trước có số liệu tái lập được:

```text
[1] Chuẩn bị dữ liệu        ViVQA + ViTextVQA (nguồn chính thức), split cố định, seed 42
        ↓
[2] Reproduce baseline      Qwen2.5-VL-3B-Instruct, zero-shot, 4-bit NF4
        ↓
[3] Cải tiến (PEFT-only)    QLoRA trên decoder ngôn ngữ  (+ OCR-prompting ở W8)
        ↓
[4] Đánh giá                EM · VQA-Acc · ANLS · kiểm định bắt cặp · ablation · error/hallucination
        ↓
[5] Prototype web           upload ảnh + hỏi tiếng Việt + hiển thị confidence / OCR
```

**Ràng buộc cứng (không thương lượng):**

- **Chỉ PEFT.** Không full fine-tuning. Trọng số gốc bị đóng băng và lượng tử hoá 4-bit; chỉ LoRA adapter được học.
- **Đúng một baseline**, reproduce trước mọi cải tiến.
- **So sánh dưới điều kiện y hệt:** cùng split, seed, prompt, độ phân giải, dtype, phần cứng.
- **Chạy được trên Colab.** Hiệu quả là một ràng buộc thiết kế tường minh, không phải hệ quả phụ.

---

## 2. Dữ liệu và phân chia

### 2.1. Hai bộ dữ liệu

| | ViVQA (Tran et al., PACLIC 2021) | ViTextVQA (Nguyen et al., 2024) |
|---|---|---|
| Loại câu hỏi | Tổng quát | Scene-text (đọc chữ trong ảnh) |
| Nguồn dùng | CSV gốc trên GitHub `kh4nh12/ViVQA` + ảnh COCO | Kho HuggingFace **chính thức, có kiểm soát truy cập** `minhquan6203/ViTextVQA` |
| Số đáp án mỗi câu | 1 | Có thể nhiều |

**Hai quyết định về nguồn dữ liệu, và lý do:**

1. **Loại bỏ ViVQA-X.** Dữ liệu cục bộ ban đầu là ViVQA-X — một bộ khác với bộ đề cương yêu cầu. Phát hiện ở W5, thay bằng ViVQA gốc.
2. **Bỏ bản mirror ViTextVQA không chính thức**, chuyển sang kho chính thức. Kho này có một cái bẫy: `ViTextVQA_test.json` là **mẫu nộp bài Kaggle** (đáp án đều là chuỗi giữ chỗ), không phải đáp án thật. Đáp án thật nằm ở `ViTextVQA_test_gt.json`. Dùng nhầm file mẫu sẽ cho ra điểm vô nghĩa mà không báo lỗi.

### 2.2. Phân chia (split)

| Tập | ViVQA | ViTextVQA | Dùng để |
|---|---|---|---|
| Train | **10.799** | 35.159 câu / 11.733 ảnh | Huấn luyện QLoRA |
| Val / Dev | 1.200 | 5.155 câu / 1.676 ảnh | Chọn cấu hình, tối ưu prompt |
| Test đầy đủ | 3.001 | 10.028 câu / 3.353 ảnh | — |
| **Test dùng để báo cáo** | **3.001 (đầy đủ)** | **2.000 (tập con, seed 42)** | Con số cuối cùng |
| Dev dùng cho ablation | — | 300 (tập con, seed 42) | Ablation cấu hình W6 |

- **ViVQA không có tập val gốc.** Val được cắt 10% từ train (xáo trộn seed 42): 15.000 − 3.001 = 11.999 câu train gốc → 1.200 val + 10.799 train. Đây là quyết định của đề tài, không phải của tác giả dataset — ghi nhận như một giới hạn.
- **ViTextVQA báo cáo trên tập con 2.000 câu** thay vì 10.028 câu vì chi phí suy luận (~1,35 s/câu trên T4 ở cấu hình baseline → tập đầy đủ ~3,8 giờ). Danh sách ID của tập con được lưu (`test_subset2000_seed42_ids.json`) nên tập con là cố định, không lấy lại ngẫu nhiên giữa các lần chạy.
- **Nguyên tắc tách vai trò:** mọi lựa chọn cấu hình (dtype, độ phân giải, prompt) được thực hiện trên **dev**, không bao giờ trên test. Test chỉ dùng để báo cáo. Chọn cấu hình trên test sẽ làm con số báo cáo bị thổi phồng.

### 2.3. Chuẩn hoá văn bản tiếng Việt

- **Unicode NFC** cho cả câu hỏi, dự đoán và đáp án. Tiếng Việt có hai cách mã hoá cùng một ký tự có dấu (dựng sẵn và tổ hợp); không chuẩn hoá thì hai chuỗi trông giống hệt nhau vẫn bị chấm khác nhau.
- Chữ thường, gọn khoảng trắng.
- **Chính sách chọn đáp án khi huấn luyện:** lấy đáp án **đầu tiên** trong danh sách. Ghi vào log mỗi lần chạy (`answer_policy`).

*Nguồn:* `src/data/prepare_vivqa.py`, `src/data/prepare_vitextvqa.py`, `src/eval/metrics.py` (`normalize_vi`), `docs/W05/`.

---

## 3. Baseline

### 3.1. Mô hình và cấu hình

| Thành phần | Giá trị |
|---|---|
| Mô hình | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Lượng tử hoá | 4-bit NF4, double quantization |
| Độ phân giải ảnh | `min_pixels` 200.704 (256 token) · `max_pixels` 401.408 (512 token) |
| Giải mã | Greedy (`do_sample=False`), `max_new_tokens` 32 |
| Prompt | Zero-shot; system prompt tiếng Việt yêu cầu trả lời ngắn gọn, chỉ đưa đáp án |
| Thư viện | `transformers` 5.17.0 (cố định phiên bản) |
| Phần cứng | Tesla T4 16 GB, khả dụng ~14,6 GiB (Colab) |

**Lý do chọn Qwen2.5-VL-3B:** nằm trong danh sách baseline được phép, công cụ hỗ trợ trưởng thành, đủ nhẹ để chạy 4-bit trên T4, yêu cầu `transformers` thấp hơn các bản mới → tái lập chắc chắn hơn. Qwen3-VL-4B được giữ làm **mô hình tham chiếu** (chỉ zero-shot, không huấn luyện) để tránh mở rộng phạm vi thành hai baseline song song.

### 3.2. "Reproduce" nghĩa là gì trong đề tài này

Paper gốc của Qwen2.5-VL **không báo cáo** trên ViVQA hay ViTextVQA. Nên "reproduce baseline" ở đây không có nghĩa là tái lập một con số đã công bố, mà là: **chạy mô hình pretrained ở chế độ zero-shot dưới một giao thức cố định, log đầy đủ, để tạo ra con số tham chiếu** mà mọi cải tiến sau này đo hơn/kém so với nó.

### 3.3. Kết quả baseline (W5)

| Dataset | n | EM | ANLS |
|---|---|---|---|
| ViVQA test | 3.001 | **0,225** | **0,288** |
| ViTextVQA test (tập con) | 2.000 | **0,285** | **0,468** |

*Tốc độ:* 1,351 s/câu trên ViTextVQA (T4, 4-bit, fp16).

**Không so sánh trực tiếp EM giữa hai dataset.** ViTextVQA có EM cao hơn dù khó hơn, vì đáp án thường chính là dòng chữ trong ảnh — model đọc được thì dễ khớp tuyệt đối; ViVQA đòi đúng từ vựng của đáp án gốc.

### 3.4. Phân tích lỗi baseline (định hướng cho cải tiến)

| Hiện tượng | Bằng chứng | Gap liên quan |
|---|---|---|
| **Chữ tiếng Việt có dấu khó hơn chữ Latin** | Trên ViTextVQA: EM 0,394 (đáp án Latin) so với 0,224 (đáp án tiếng Việt có dấu) | G1, G2 |
| Chuyển sang tiếng Anh (code-switching) | ~90 dự đoán trả lời bằng tiếng Anh | G1 |
| Lệch phong cách | Đáp án chuẩn thường bỏ loại từ (`con`, `chiếc`); model thì giữ | — |
| Thiên lệch màu | Dự đoán `đỏ` 117 lần, trong khi đáp án đúng chỉ có 75 lần | G4 |
| **Output suy biến** | 23 dự đoán toàn ký tự `!` — xem mục 5.1 | — |

⏳ W7: **baseline sẽ được chạy lại** dưới cấu hình cuối cùng (xem mục 5.4), vì cấu hình W5 dùng fp16 — đã chứng minh gây lỗi ở mục 5.1.

*Nguồn:* `experiments/W05_zeroshot_qwen25vl_vivqa_test_seed42/`, `experiments/W05_zeroshot_qwen25vl_vitextvqa_test2000_seed42/`, `docs/W05/W05_Baseline_Report.md`.

---

## 4. Phương pháp đề xuất: QLoRA

### 4.1. Cấu hình adapter

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Rank `r` | 16 | |
| `lora_alpha` | 32 | = 2r, theo thông lệ |
| `lora_dropout` | 0,05 | |
| Module đích | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention của decoder ngôn ngữ |
| Trọng số gốc | Đóng băng, 4-bit NF4 | |

### 4.2. Bằng chứng PEFT-only — đo được, không chỉ tuyên bố

```
trainable params: 7,372,800 || all params: 3,761,995,776 || trainable%: 0.1960
```

**0,196%** tham số được huấn luyện. Con số này còn **kiểm chứng được bằng phép tính**, cho biết adapter nằm ở đâu:

| Module | Kích thước (vào → ra) | Tham số LoRA = r × (vào + ra) |
|---|---|---|
| `q_proj` | 2048 → 2048 | 16 × 4096 = 65.536 |
| `k_proj` | 2048 → 256 | 16 × 2304 = 36.864 |
| `v_proj` | 2048 → 256 | 16 × 2304 = 36.864 |
| `o_proj` | 2048 → 2048 | 16 × 4096 = 65.536 |
| **Mỗi lớp decoder** | | **204.800** |
| **× 36 lớp** | | **7.372.800** ✔ |

Khớp chính xác với số PEFT báo cáo. Tức **toàn bộ adapter nằm trong decoder ngôn ngữ**; vision encoder và lớp nối ảnh–chữ không bị chạm tới. (Các lớp attention của vision encoder dùng tên module khác nên không khớp danh sách đích.)

> **Lưu ý về cách đếm:** trọng số 4-bit được đóng gói hai giá trị trong một byte, nên đếm bằng `p.numel()` cho ra **thiếu một nửa** (2,04 tỉ thay vì 3,76 tỉ → tỉ lệ sai 0,361%). Script dùng hàm đếm của PEFT, có xử lý đóng gói 4-bit, và ghi phương pháp đếm vào log (`param_count_method`). Lỗi này được phát hiện và sửa ở W6; run smoke đầu tiên còn mang số đếm sai và được chú thích rõ trong `smoke_summary.json`.

### 4.3. Che nhãn: chỉ tính loss trên đáp án

Mỗi mẫu huấn luyện là một chuỗi:

```text
[system] [ảnh → ~512 token] Câu hỏi: ... | <đáp án><|im_end|>
└──────────── nhãn = −100 (bỏ qua) ─────┘ └──── tính loss ───┘
```

**Vì sao bắt buộc:** mô hình sinh văn bản học bằng cách đoán token kế tiếp trên toàn chuỗi. Không che thì phần lớn tín hiệu học dồn vào việc sinh lại câu hỏi và token ảnh — hơn 500 token so với vài token đáp án. Loss vẫn giảm đẹp nhưng chất lượng trả lời không lên: một lỗi không tự báo.

**Ba chi tiết triển khai:**

1. **Độ dài prompt phụ thuộc ảnh.** Mỗi ảnh nở ra số token khác nhau theo độ phân giải, nên không thể đoán vị trí che bằng độ dài chữ. Độ dài prompt được đo riêng cho từng mẫu bằng cách cho processor xử lý phần prompt **cùng đúng ảnh đó**.
2. **Đệm bên phải** (`padding_side = "right"`), đặt tường minh — để prompt luôn nằm đầu chuỗi và vị trí che đúng.
3. **Đáp án kèm token kết thúc lượt `<|im_end|>`.** Không có nó, mô hình không học được khi nào dừng — nguyên nhân của lỗi trả lời lan man ở baseline.

**Prompt lúc huấn luyện trùng khít prompt lúc đánh giá.** Lệch prompt giữa hai giai đoạn là lệch phân phối, làm kết quả mất ý nghĩa.

*Nguồn:* `src/adaptation/train_qlora.py` (`build_labels`, `make_collate_fn`), `src/models/vlm_loader.py`, `configs/qwen_lora.yaml`.

---

## 5. Thiết lập huấn luyện và ba phát hiện thực nghiệm (W6)

Trước khi huấn luyện thật, W6 chạy ba thí nghiệm cấu hình. Mỗi thí nghiệm **cố định tiêu chí kết luận trước khi xem số**, để tránh diễn giải kết quả theo hướng có lợi.

### 5.1. Phát hiện 1 — Tràn số fp16 gây output suy biến

**Hiện tượng:** 23/2.000 dự đoán baseline ViTextVQA là chuỗi `!` dài đúng 32 ký tự.

**Chẩn đoán:** ba dấu hiệu cùng chỉ về tràn số:

- Độ dài bằng đúng `max_new_tokens = 32` → mô hình sinh cùng một token đến hết hạn mức, không bao giờ kết thúc.
- `!` là **token id 0** trong tokenizer Qwen2.
- Khi logits thành `NaN`/`Inf`, phép chọn giá trị lớn nhất rơi về chỉ số 0 ở mọi bước.

fp16 chỉ biểu diễn được tới 65.504; giá trị trung gian vượt ngưỡng thành vô cực.

**Thí nghiệm A/B:** chạy lại đúng 23 câu hỏng + **50 câu đối chứng** (chọn ngẫu nhiên seed 42 từ phần còn lại), chỉ đổi kiểu số khi tính toán:

| | fp16 | fp32 |
|---|---|---|
| Câu suy biến | **23** | **0** |
| Nhóm 23 câu hỏng — EM / ANLS | 0,000 / 0,000 | 0,348 / 0,517 |
| Nhóm 50 đối chứng — EM | 0,280 | 0,280 |
| Câu thoái lui (đúng → sai) | — | **0** |
| Thời gian | 1,965 s/câu | 2,739 s/câu (**1,39×**) |

**Kết luận:** tràn số fp16 là nguyên nhân, đã xác nhận nhân quả bằng thí nghiệm một biến. fp32 phục hồi 8/23 câu về đúng hoàn toàn, **không làm hỏng câu nào vốn đúng** (EM đối chứng trùng khít).

**Hai lưu ý diễn giải:**

- **Không trình bày đây là cải tiến điểm số.** Tập probe có 31,5% câu hỏng; trên test thật chỉ 1,15%. Tác động ước tính lên test đầy đủ: EM +0,004, ANLS +0,006. Giá trị của phát hiện nằm ở chỗ khác: gỡ một biến nhiễu khỏi mọi thí nghiệm sau, và **cảnh báo rủi ro khi huấn luyện** — gradient có dải giá trị rộng hơn activation, nên fp16 dễ gây NaN loss, hỏng toàn bộ adapter chứ không chỉ vài câu trả lời.
- **Tràn số là hiện tượng rời rạc, không phải nhiễu liên tục.** EM nhóm đối chứng trùng khít giữa hai dtype cho thấy fp16 hoặc hỏng hoàn toàn, hoặc không ảnh hưởng gì.

**Ràng buộc phần cứng:** giải pháp chuẩn là bf16 (cùng 16 bit nhưng dải rộng như fp32). **T4 không hỗ trợ bf16** (kiến trúc Turing, sm_75; bf16 cần từ Ampere sm_80). Trên T4 chỉ còn fp32.

*Nguồn:* `experiments/W06_dtype_probe_seed42/` (`failed_ids.json`, `probe_summary.json`, `probe_breakdown.json`), `src/eval/find_degraded.py`, `src/eval/probe_breakdown.py`, `notebooks/03_ablation_config_qwen25vl.ipynb`.

### 5.2. Phát hiện 2 — Độ phân giải: đánh đổi, không phải cải thiện đơn thuần

**Câu hỏi:** tăng số token ảnh từ 512 lên 1.280 có giúp đọc chữ tốt hơn không?

**Thiết lập:** 300 câu dev ViTextVQA (seed 42), fp32, 4-bit, chỉ đổi `max_pixels`.

| | A1 — 512 token | A2 — 1.280 token | Chênh |
|---|---|---|---|
| EM | 0,3333 | 0,3600 | +0,0267 |
| ANLS | 0,4654 | 0,5041 | +0,0387 |
| Thời gian | 2,605 s/câu | 3,807 s/câu | **1,46×** |

**Kiểm định bắt cặp** (cùng 300 câu nên so sánh từng câu được):

| | A2 đúng | A2 sai |
|---|---|---|
| **A1 đúng** | 93 | 7 |
| **A1 sai** | 15 | 185 |

- **EM:** McNemar chính xác, b = 7, c = 15 → **p = 0,134**. Không có ý nghĩa thống kê.
- **ANLS:** chênh +0,0387, khoảng tin cậy 95% bootstrap bắt cặp (10.000 lần) **[+0,012; +0,066]**. Không chứa 0 → có ý nghĩa.

**Diễn giải — hai metric không mâu thuẫn:** EM là thang nhị phân, chỉ thấy 22 câu đổi trạng thái đúng/sai — quá ít để kết luận. ANLS là thang liên tục, thấy cải thiện một phần trên cả 300 câu. Kết luận hợp lệ: *độ phân giải cao giúp nhận dạng ký tự chính xác hơn trên toàn tập, nhưng chưa đủ để chuyển thành nhiều câu đúng hoàn toàn hơn.* Nhất quán với phát hiện baseline: rào cản chính là **dấu thanh tiếng Việt**, không phải độ nét.

**Độ phân giải cao cũng có cái giá về chất lượng:** 7 câu A1 đúng mà A2 sai. Giả thuyết: ảnh nét hơn đưa thêm chữ vào tầm nhìn, mô hình chọn nhầm dòng chữ. Chỉ lộ ra nhờ kiểm định bắt cặp; so sánh bằng số tổng sẽ che mất.

**Ràng buộc:** độ phân giải dùng khi huấn luyện phải trùng độ phân giải khi đánh giá. Huấn luyện ở 512 rồi đánh giá ở 1.280 là lệch phân phối.

*Nguồn:* `experiments/W06_ablation_config_dev300_seed42/` (`ablation_summary.json`, `paired_A1_vs_A2.json`), `src/eval/compare_runs.py`.

### 5.3. Phát hiện 3 — Chi phí huấn luyện trên T4

**Thiết lập smoke run:** 200 mẫu ViVQA, 512 token, fp32, 4-bit.

| Cấu hình | Batch hiệu dụng | Gradient checkpointing | s/mẫu | VRAM đỉnh | Kết quả |
|---|---|---|---|---|---|
| Batch 2 × tích luỹ 8 | 16 | Bật | **4,93** | 5,93 GB | ✅ 30 step |
| Batch 6 × tích luỹ 3 | 18 | **Tắt** | — | 14,46 / 14,56 GB | ❌ Hết VRAM |
| Batch 6 × tích luỹ 3 | 18 | Bật | **5,10** | 11,6 GB | ✅ 10 step |

**Đường ray huấn luyện hoạt động đúng:** loss giảm đều 1,61 → 1,03 → 0,88 qua 30 step. Nếu che nhãn sai hoặc LoRA không gắn, loss sẽ đứng im.

**T4 bão hoà tính toán ở fp32:** gấp ba batch, gấp đôi VRAM, tốc độ không đổi (4,93 so với 5,10 s/mẫu; chênh 3% nằm trong nhiễu, và run 10 step chịu chi phí khởi động nặng hơn nên nếu có thì sai lệch làm batch 6 trông chậm hơn thực tế). **Không tắt được gradient checkpointing** trên T4 ở cấu hình này. Tối ưu phần mềm không cứu được; cần đổi GPU.

**Ngân sách thực tế:** 10.799 mẫu × 4,93 s/mẫu ≈ **14,77 giờ / epoch ViVQA** trên T4. Cấu hình mặc định 3 epoch ≈ 44 giờ. Không khả thi trên Colab miễn phí cho 8 ablation. Đây là căn cứ định lượng cho việc chuyển sang GPU L4 (Colab Pro) từ W7.

**Giữ batch 2 làm mặc định:** cùng tốc độ, dùng ít hơn một nửa VRAM, biên an toàn rộng.

*Nguồn:* `experiments/W06_smoke_qlora_seed42/smoke_summary.json`, `experiments/W06_smoke_qlora_qwen25vl_vivqa_mp512*_seed42/train_log.json`, `notebooks/04_qlora_qwen25vl.ipynb`.

### 5.4. Siêu tham số huấn luyện

| Tham số | Giá trị | Trạng thái |
|---|---|---|
| Learning rate | 2 × 10⁻⁴ | Chốt |
| Warmup | 3% tổng số step (quy đổi sang số step) | Chốt |
| Weight decay | 0 | Chốt |
| Batch × tích luỹ gradient | 2 × 8 = 16 | Chốt (mục 5.3) |
| Gradient checkpointing | Bật | Chốt (mục 5.3) |
| Số epoch | 3 | ⏳ W7 — tuỳ ngân sách L4 |
| **Kiểu số khi tính toán** | fp32 trên T4 | ⏳ **W7** — probe ba nhánh fp16 / fp32 / bf16 trên L4 |
| **`max_pixels`** | 512 token (thiên hướng) | ⏳ **W7** — cần chi phí huấn luyện trên L4 |
| Lượng tử hoá 4-bit vs không | 4-bit | ⏳ W7 — trên T4 bản không lượng tử hoá ở fp32 cần ~12,4 GB chỉ riêng trọng số |

**Vì sao chưa chốt dtype và độ phân giải ở W6:** phần cứng sẽ đổi ở W7 (T4 → L4). Chốt trên T4 rồi đổi GPU nghĩa là chạy lại baseline hai lần. Baseline chỉ có ý nghĩa khi đo cùng điều kiện với mô hình cải tiến, nên nó sẽ được chạy đúng một lần — trên phần cứng và cấu hình cuối cùng.

---

## 6. Giao thức đánh giá

### 6.1. Metric chính

| Metric | Định nghĩa | Ghi chú |
|---|---|---|
| **Exact Match (EM)** | Khớp tuyệt đối sau chuẩn hoá | Chuẩn hoá **chặt**: không gộp bí danh màu, chuẩn hoá số chỉ khi toàn chuỗi là số |
| **VQA Accuracy** | Theo chuẩn VQA | Với ViVQA (1 đáp án/câu), VQA-Acc ≈ EM sau chuẩn hoá — ghi rõ để tránh hiểu nhầm |
| **ANLS** | Độ giống chuỗi chuẩn hoá, ngưỡng 0,5 | Metric chính cho ViTextVQA; nhạy với câu đúng một phần |

BLEU/CIDEr **không** dùng làm metric chính, chỉ là chỉ số phụ nếu cần cho câu trả lời mở.

### 6.2. So sánh công bằng

- Mọi biến thể chạy qua **cùng một mã đánh giá** (`src/eval/run_baseline.py`), chỉ khác file cấu hình.
- **Mỗi thí nghiệm đổi đúng một biến.** Bài học từ W6: đổi hai biến cùng lúc (batch + checkpointing) dẫn tới hết VRAM mà không xác định được nguyên nhân.
- **Tiêu chí kết luận cố định trước khi xem số.**
- **Kiểm định thống kê**, không chỉ so số tổng:
  - EM: **McNemar chính xác** (nhị thức hai phía) trên các cặp bất đồng.
  - ANLS: **bootstrap bắt cặp** 10.000 lần, seed 42, khoảng tin cậy 95% theo bách phân vị.
- **Tập đối chứng** khi sửa một lỗi cụ thể, để phát hiện thoái lui.

### 6.3. Kế hoạch ablation (W8–W12)

| # | So sánh | Trả lời câu hỏi |
|---|---|---|
| 1 | Zero-shot vs prompt tối ưu vs QLoRA | Cải thiện đến từ đâu: prompt hay huấn luyện? |
| 2 | Có vs không OCR-enhanced prompting | OCR tiếng Việt có bù được điểm yếu dấu thanh? |
| 3 | Độ phân giải 512 vs 1.280 (huấn luyện + đánh giá cùng mức) | Đánh đổi chất lượng–chi phí khi đã có adapter |
| 4 | Qwen2.5-VL-3B vs Qwen3-VL-4B (tham chiếu, zero-shot) | Backbone mới hơn có thu hẹp khoảng cách không? |

Giả thuyết chi tiết: `docs/W02/Hypotheses.md`.

### 6.4. Phân tích lỗi và hallucination

- **Phân rã theo loại câu hỏi:** yes/no, đếm, đọc chữ, suy luận, nhận dạng đối tượng. ViVQA có sẵn nhãn loại; ViTextVQA cần gán (⏳ W7).
- **Case study tiếng Việt:** dấu thanh, code-switching, lỗi OCR.
- **Hallucination:** theo phương pháp POPE (Li et al., EMNLP 2023) — hỏi yes/no về sự tồn tại của đối tượng, đo tỉ lệ khẳng định sai.
- **Phát hiện output suy biến tự động** (`src/eval/find_degraded.py`) cho mọi run.

### 6.5. Tái lập

Mỗi lần chạy ghi: dataset split, seed (42), mã băm SHA-256 của file dữ liệu, commit git, loại GPU và khả năng bf16, phiên bản thư viện, toàn bộ cấu hình, thời gian suy luận/huấn luyện, VRAM đỉnh. Kết quả được lưu theo lô, chạy lại được khi mất kết nối.

Quy ước đặt tên: `experiments/W0X_<chế độ>_<model>_<dataset>_seed42/`.

---

## 7. Đạo đức và AI có trách nhiệm (CLO7)

- **Quyền riêng tư:** ảnh người dùng tải lên prototype có thể chứa khuôn mặt, giấy tờ, dữ liệu cá nhân. Thiết kế dự kiến (⏳ W14): không lưu ảnh sau khi xử lý.
- **Giấy phép dữ liệu:** ViTextVQA dùng qua kho có kiểm soát truy cập, tuân theo điều khoản của tác giả.
- **Thiên lệch:** mô hình pretrain chủ yếu trên dữ liệu tiếng Anh. Bằng chứng định lượng từ baseline: chênh 17 điểm EM giữa đáp án Latin và đáp án tiếng Việt có dấu; ~90 câu trả lời bằng tiếng Anh cho câu hỏi tiếng Việt.
- **Độ tin cậy:** câu trả lời trôi chảy nhưng sai là rủi ro chính. Prototype hiển thị confidence và văn bản OCR trích được để người dùng tự kiểm tra, không trình bày câu trả lời như sự thật tuyệt đối.

---

## Phụ lục — Các quyết định còn mở (⏳ W7)

| Quyết định | Căn cứ đã có | Còn thiếu |
|---|---|---|
| Kiểu số khi tính toán | fp16 gây tràn số (5.1); T4 không có bf16 | Probe fp16 / fp32 / bf16 trên L4 |
| `max_pixels` | +0,039 ANLS có ý nghĩa, EM không; chi phí suy luận 1,46× (5.2) | Chi phí huấn luyện trên L4 |
| Lượng tử hoá 4-bit vs không | — | Đo trên L4 (22,5 GB VRAM) |
| Số epoch | 14,77 giờ/epoch trên T4 (5.3) | Tốc độ thực trên L4 |
| Chạy lại baseline | Cấu hình W5 dùng fp16 | Chạy **một lần** sau khi chốt ba quyết định trên |

---

# FILE: docs/03_report/R5_Implementation.md

# R5 — Implementation

## 1. Repository layout

`src/data` preprocessing/OCR/tagging · `src/baseline` · `src/adaptation` PEFT/prompt/RAG · `src/eval` · `src/prototype`.  
`data/` · `configs/` · `adapters/` · `experiments/`.

## 2. Data pipeline

Raw → processed → OCR → splits (seed).

## 3. Baseline inference

Thư viện: PyTorch, HuggingFace Transformers. Version: `_`.

## 4. PEFT adaptation

PEFT library · bitsandbytes · target modules · lưu adapter (không full weights).

## 5. Evaluation scripts

EM · VQA Accuracy · ANLS · export pred.

## 6. Prototype

UI, GPU runtime, OCR/evidence visualization.

## 7. Reproducibility

Map sang checklist W15. Lưu ý: thư mục `src/production-agentic-rag-course/` là code đề tài cũ, không phải pipeline VQA.

---

# FILE: docs/03_report/R6_Results.md

# R6 — Results

Nguồn số: W10–W12. Mọi bảng nêu split, seed, hardware, N.

## 1. Experimental setup recap

## 2. Main results

EM · VQA-Acc · ANLS trên ViVQA và ViTextVQA. BLEU/CIDEr phụ (nếu có).

## 3. Efficiency

Train time (PEFT) · infer time · VRAM.

## 4. Ablation

Trỏ `docs/weekly/W12/Ablation_Study.md`.

## 5. Error analysis

Trỏ W12 Error_Analysis.

## 6. Hallucination analysis

Trỏ W12 Hallucination_Analysis.

## 7. Qualitative examples

---

# FILE: docs/03_report/R7_Discussion_Conclusion.md

# R7 — Discussion and Conclusion

## 1. Trả lời RQ / hypotheses

## 2. So với related work

## 3. Limitations

GPU · subset · OCR noise · EM chặt · 1 baseline.

## 4. Ethics and responsible AI

Privacy ảnh · bias · hallucination trên demo.

## 5. Future work

(Vẫn trong PEFT-only nếu tiếp tục đề tài.)

## 6. Conclusion

Đóng góp: reproduce 1 VLM + cải tiến kiểm chứng không full FT + phân tích bắt buộc.

---

# FILE: docs/03_report/README.md

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
