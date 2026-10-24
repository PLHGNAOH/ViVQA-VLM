# 00_ALL_REPORTS — ViVQA-VLM progress dossier (auto-generated)

Built from `main` @ `6b84e3d` on 2026-10-24T04:41Z. Each section starts with its source path.

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

# FILE: docs/W10/README.md

# Week 10 — Ablation cấu hình LoRA + B3 trên toàn bộ dữ liệu train

Index tuần 10 (tuần ảo) của ViVQA-VLM. Tuần ảo W10 chạy trong **tuần thực 7** (19–25/10/2026). Phần Review 2 của tuần thực 7 theo dõi riêng (checklist AI_Capstone_Review123), không nằm trong README này.

Mục tiêu W10: trả lời **hai câu hỏi còn mở** về công thức B3 (OCR-aware QLoRA) trước khi viết phần Results:

1. **Cấu hình LoRA đã hợp lý chưa?** — B3 dùng r = 16, chỉ 4 lớp attention; chưa từng kiểm (H3, H4).
2. **Đủ dữ liệu thì OCR-aware có vượt B2 không?** — W09: B3 (23% dữ liệu) thấp hơn B2 dữ liệu đầy đủ 1,16 EM trên full test.

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

> **Đánh số giả thuyết.** Luật ghi trước của tuần này (19/10) dùng đánh số **cũ** trong `docs/W02/Hypotheses.md`: H3 = vị trí gắn LoRA, H4 = rank. Ngày 20/10, deck cộng dồn Review 2 (`docs/04_slides/cumulative/`, commit `6359085`) chốt đánh số **chính thức mới** H1–H4 và chuyển cấu hình LoRA thành **ablation A1 (module gắn LoRA) và A2 (rank)**. Luật đã commit trước khi có số nên **giữ nguyên chữ**; bảng ánh xạ dùng cho báo cáo:
>
> | Luật tuần này (số cũ) | Tên chính thức (số mới) | Run (tên thư mục, không đổi) |
> |---|---|---|
> | H3 · all-linear so với attention-only, cùng ngân sách | **Ablation A1** · module gắn LoRA | `W10_A2_*` (all-linear r4) |
> | H4 · rank 64 so với 16 | **Ablation A2** · rank | `W10_A3_*` (attention r64) |
> | — | **H2** · OCR (≥ +3 ANLS, cộng hưởng với QLoRA) | `W10_B3full_*` |

## Kế hoạch tuần

| # | Việc | GPU | Trạng thái |
|---|------|-----|-----------|
| 0 | Code: `src/adaptation/lora_targets.py` (preset + chặn LoRA lọt vision), `--lora_r / --lora_alpha / --lora_targets` cho `train_qlora.py`, LoRA audit trong `train_log.json`, `src/eval/lora_ablation.py`, notebook `17_lora_ablation_rank_modules_qwen25vl.ipynb`, luật ghi trước (README này) | — | ✅ 19/10 (`8be932e`) |
| 1 | **A2** — all-linear r = 4 (cùng ngân sách tham số với B3), train + eval test2000 | ≈ 3,9 h L4 | ✅ 20/10 — **H3 KHÔNG ĐƯỢC ỦNG HỘ** (−0,55 EM, p = 0,34), chậm hơn 17,5% |
| 2 | **A3** — attention r = 64, train + eval test2000 | ≈ 3,7 h L4 | ✅ 20/10 — **ỦNG HỘ H4 (yếu)** (+1,05 EM, p = 0,071, CI95 [−0,05; +2,15]), adapter ×4 |
| 3 | OCR cho phần ảnh train còn thiếu (≈ 9.063 ảnh) — notebook 10 **Phần D** (mục 13–14) | ≈ 2–2,5 h T4 | ✅ 20/10 — +9.063 ảnh, 0 lỗi, 0,72 s/ảnh (≈ 1,9 h T4); 6.289 ảnh cũ giữ nguyên; cache 15.352 ảnh |
| 4 | **B3-full** — QLoRA + OCR trên toàn bộ train (35.159 câu), eval full test 10.028 câu — notebook `18_qlora_ocr_b3full_vs_b2_qwen25vl.ipynb`, kết luận bằng `src/eval/b3full.py` | ≈ 13 h + 2,8 h L4 | ✅ 21/10 — **OCR-AWARE VƯỢT B2 KHI ĐỦ DỮ LIỆU** (+3,09 EM, p = 6,7 × 10⁻²¹) |
| 5 | Results (R6) bản tiếng Việt — phần QLoRA nhận biết OCR (số liệu W08–W10) | — | ✅ 21/10 — `docs/03_report/R6_Results.md` mục 2–4 (bản nháp) |
| 6 | Đóng tuần: `W10_Team_Report.md`, dọn Downloads + Drive, handoff | — | ✅ 21/10 |

## Quyết định đã chốt

| Ngày | Quyết định | Lý do |
|------|-----------|-------|
| 19/10/2026 | Ablation LoRA chạy trên **công thức B3** (tập con 8k + OCR, seed 42), so với **B3 seed 42 (W08)** đã có | B3 là phương pháp đề xuất — câu hỏi "cấu hình LoRA có hợp lý không" phải trả lời trên chính nó. Dùng lại B3 làm mốc → chỉ tốn 2 run mới |
| 19/10/2026 | **A2 = all-linear r = 4** (7.483.392 tham số), không phải all-linear r = 16 | H3 hỏi "ở **cùng ngân sách tham số**". All-linear r = 16 có 29,9 triệu tham số (×4) → nếu tốt hơn cũng không biết do vị trí gắn hay do nhiều tham số. r = 4 lệch B3 chỉ +1,5% |
| 19/10/2026 | **A3 = attention r = 64** (29.491.200 tham số) | Đúng phát biểu H4 (r = 64 so với r = 16). Cùng vị trí gắn với B3 → chỉ khác rank |
| 19/10/2026 | **α = 2r** ở mọi run (r4/α8, r16/α32, r64/α128) | Giữ hệ số scale α/r = 2 như cấu hình gốc → so sánh rank không bị lẫn với thay đổi độ lớn bước cập nhật. `--lora_r` không kèm `--lora_alpha` tự đặt α = 2r |
| 19/10/2026 | Preset `all_linear_llm` dùng **regex** chỉ khớp `layers.<i>.(self_attn\|mlp)…` và loại tên chứa `visual` | Vision tower của Qwen2.5-VL cũng có `gate_proj/up_proj/down_proj` (`modeling_qwen2_5_vl.py`, transformers 5.17.0). List hậu tố sẽ gắn LoRA vào 32 × 3 lớp vision. `train_qlora` dừng nếu audit thấy lớp vision |
| 19/10/2026 | **1 seed (42)** cho ablation | Std EM của B3 qua 3 seed = 0,23 điểm (W09), nhỏ hơn nhiều mức test2000 phát hiện được (≈ 1,7 điểm). Thêm seed tốn ≈ 7 h L4 / seed / biến thể. Ghi là giới hạn |
| 19/10/2026 | **B3-full dùng nguyên công thức B3** (attention r = 16) bất kể kết quả ablation | B3-full so với B2 W07 (cũng attention r = 16). Đổi cấu hình LoRA theo kết quả ablation là chọn cấu hình bằng test → rò rỉ. Nếu A2/A3 thắng, ghi là hướng cải tiến tiếp |
| 20/10/2026 | **Review 2 dời sang thứ Năm 29/10/2026** (tuần thực 8), theo thông báo của bộ môn | Kết quả W10 có trước ngày review → theo quy ước W09 thuộc mốc Review 2; việc đưa vào deck cộng dồn làm trong tuần thực 8, trước buổi review |
| 21/10/2026 | **Mô hình ViTextVQA chính thức = B3-full** (thay B2 W07) cho bảng kết quả chính và prototype | Thắng B2 W07 theo luật ghi trước (cùng dữ liệu, cùng siêu tham số, chỉ khác OCR). Số B2, B2-sub, B3 vẫn báo cáo đủ trong bảng ablation. ViVQA giữ C3 (W09) |
| 21/10/2026 | **Adapter dùng chung 2 dataset: làm bản rẻ ở W11** (ViVQA đầy đủ + ViTextVQA tập con, 1 epoch); thiết kế + luật ghi trước chốt đầu W11 | Đã hứa trên slide Review 2 (kế hoạch tới Review 3); trả lời câu hỏi chuyển giao giữa hai loại câu hỏi; demo dùng 1 adapter nếu không kém. Nếu kém → demo chuyển adapter theo ảnh (PEFT hỗ trợ sẵn) |

## Luật ablation LoRA — H3 (vị trí gắn) và H4 (rank) (ghi TRƯỚC khi chạy, 19/10/2026)

Mọi thứ khác **y hệt B3 seed 42 (W08)**: `train_subset8000_byimage_seed42` (sha256 trong manifest W08), OCR VietOCR · v1 · 600 · 0, lr 2e-4, 1 epoch (500 step), batch hiệu dụng 16, warmup 3%, dropout 0,05, bf16, 4-bit NF4 + double quant, max_pixels 401.408, L4, eval test2000 greedy (max_new_tokens 32). `src/eval/lora_ablation.py` **tự kiểm** A2/A3 chỉ khác B3 ở mục LoRA.

| Run | Lớp gắn LoRA (chỉ LLM, 36 layer) | r | α | Tham số train |
|---|---|---|---|---|
| B3 (mốc, W08) | q, k, v, o | 16 | 32 | 7.372.800 |
| A2 | q, k, v, o, gate, up, down | 4 | 8 | 7.483.392 |
| A3 | q, k, v, o | 64 | 128 | 29.491.200 |

**Điều kiện hợp lệ (kiểm trước khi eval):** `train_log.json` có `lora_audit.by_tower` = `{'llm': 252}` (A2) / `{'llm': 144}` (A3), không có `vision`; số tham số train (PEFT đếm) = số tính tay ở bảng trên. Sai → không dùng run, báo Claude.

So sánh ghép cặp trên cùng 2.000 câu (`compare_runs.py`: McNemar exact trên EM, ΔANLS kèm CI95 bootstrap 10.000 lần, seed 42); thêm CI95 của ΔEM (bootstrap ghép cặp từ bảng chéo, seed 42):

| Giả thuyết | So sánh | Kết luận |
|---|---|---|
| **H3** — all-linear tốt hơn attention-only ở cùng ngân sách | B3 → A2 | EM(A2) > EM(B3) **và** p < 0,05 → **ỦNG HỘ H3** · EM(A2) < EM(B3) và p < 0,05 → **BÁC BỎ H3** (attention-only tốt hơn) · p ≥ 0,05 → **H3 KHÔNG ĐƯỢC ỦNG HỘ** (không khác biệt có ý nghĩa) |
| **H4** — bão hoà ở r ≤ 16; r = 64 không hơn quá 1 điểm | B3 → A3 | ΔEM > +1 điểm **và** p < 0,05 → **BÁC BỎ H4** · p ≥ 0,05 → **ỦNG HỘ H4** · A3 kém hơn và p < 0,05 → **ỦNG HỘ H4 (r = 64 kém hơn)** · p < 0,05 và 0 < ΔEM ≤ 1 → **ỦNG HỘ H4 (tăng ≤ 1 điểm)** |

- **Độ mạnh của "ủng hộ H4":** cận trên CI95 ΔEM < +1 điểm → *mạnh*; còn chứa giá trị > +1 → *yếu* (test2000 chưa đủ nhạy để loại trừ mức tăng 1 điểm).
- "Không khác biệt có ý nghĩa" nghĩa là **nhỏ hơn mức đo được** (≈ 1,7 điểm EM trên test2000), không phải "bằng nhau" — báo cáo kèm CI95.
- Báo cáo đủ: EM, ANLS, token-F1 (phụ), chi phí (giờ train, s/step, VRAM đỉnh, dung lượng adapter) của cả 3 run.
- **Không** chạy thêm seed hay cấu hình sau khi thấy kết quả để "câu" kết luận.

## Luật B3-full — OCR-aware QLoRA trên toàn bộ train (ghi TRƯỚC khi chạy, 19/10/2026)

Run **B3-full**: công thức B3 (attention r = 16, α = 32, OCR VietOCR · v1 · 600 · 0, lr 2e-4, 1 epoch, seed 42, L4) trên **toàn bộ** `data/vitextvqa_official/train.json` (35.159 câu, 11.733 ảnh, sha256 `dec0acc1ca99…` — cùng file với B2 W07). Eval full test 10.028 câu, có OCR.

0. **Điều kiện trước khi train:** OCR cache phủ đủ 11.733 ảnh train; dấu vân tay của các ảnh đã có trong cache **giữ nguyên** sau khi OCR thêm (như luật full test W09).
1. **So sánh chính:** B2 W07 (full train, không OCR, full test EM 50,26) → B3-full, ghép cặp trên 10.028 câu:
   - EM tăng **và** McNemar exact p < 0,05 → **OCR-AWARE VƯỢT B2 KHI ĐỦ DỮ LIỆU**
   - EM giảm **và** p < 0,05 → **OCR-AWARE KÉM B2 KHI ĐỦ DỮ LIỆU**
   - còn lại → **NGANG B2 KHI ĐỦ DỮ LIỆU** (đóng góp của OCR-aware khi đó chỉ là hiệu quả dữ liệu — ghi đúng như vậy vào thesis)
   - báo kèm ΔANLS + CI95 bootstrap.
2. **So với ViTextBLIP-2** (Table 6, Nguyen et al., *ESWA* 2025: EM 25,48 · F1 53,95): chỉ viết "cao hơn" khi cận dưới CI95 bootstrap của B3-full lớn hơn số công bố (như luật W09).
3. **Chỉ báo cáo** (không kết luận riêng): bảng 2 × 2 *lượng dữ liệu (23% / 100%) × OCR (không / có)* gồm B2-sub, B3, B2 W07, B3-full trên full test — lợi ích của OCR thay đổi thế nào khi tăng dữ liệu.
4. **Giới hạn đã biết:** B2 W07 train bằng code commit W07; B3-full train bằng code W10. Prompt zero-shot và cách train không đổi giữa hai commit (W08 chỉ thêm dòng OCR), nhưng vẫn ghi là giới hạn.

## Thay đổi code

| File | Thay đổi | Kiểm |
|------|----------|------|
| `src/adaptation/lora_targets.py` (mới) | Preset `attn` (list cũ, giữ nguyên) và `all_linear_llm` (regex chỉ LLM); `audit_lora_modules` đếm lớp LoRA theo khối (LLM / vision) và hậu tố, đếm tham số LoRA từ `lora_A/lora_B`; `expected_lora_params` tính tay theo kích thước Qwen2.5-VL-3B | `--selftest`: 7.372.800 khớp log W07–W09; regex đúng 252 lớp LLM, 0 lớp vision trên bố cục tên transformers 5.x **và** bố cục cũ; list hậu tố MLP ngây thơ bị chặn (lọt 96 lớp vision) |
| `src/adaptation/train_qlora.py` | `--lora_r`, `--lora_alpha`, `--lora_targets`; gọi `audit_lora_modules` ngay sau khi nạp model (dừng nếu sai); ghi `lora_audit` vào `train_log.json` | `--selftest` thêm `lora override OK`: không truyền → chữ ký W07–W09 giữ nguyên; `attn` r16 trùng chữ ký cũ; r64 → α128; all-linear r4 → regex + α8 |
| `src/eval/lora_ablation.py` (mới) | Luật H3/H4, CI95 ΔEM, kiểm "chỉ khác LoRA" | `--selftest` |
| `notebooks/17_lora_ablation_rank_modules_qwen25vl.ipynb` (mới) | A2 → A3 → kết luận; bỏ qua phần đã xong; tự trả máy L4 | — |
| `notebooks/10_ocr_cache_vitextvqa.ipynb` | Thêm **Phần D** (mục 13–14): OCR phần ảnh còn thiếu của `train.json`; cache phải trùng sha256 biên bản W09 trước khi mở rộng, dấu vân tay 6.289 ảnh cũ phải giữ nguyên; biên bản `experiments/W10_ocr_cache_fulltrain/` | — |
| `src/eval/b3full.py` (mới) | Luật B3-full (B2 → B3-full) + bảng 2×2 dữ liệu × OCR, 4 hiệu ứng đơn, tương tác theo câu (chỉ báo cáo); ghép cặp theo `question_id` | `--selftest`; chạy thử trên predictions thật W07/W09 (B3-full tạm thay bằng B2): B2-sub → B3 = 417 / 672, p = 1,06 × 10⁻¹⁴, ΔANLS CI95 [+0,0106; +0,0208] — **khớp** `W09_fulltest_B3_vs_B2sub_L4_seed42`; B3 → B2 = +1,16 EM, p = 0,0025 — **khớp** đính chính W09; ≈ 3 phút CPU |
| `notebooks/18_qlora_ocr_b3full_vs_b2_qwen25vl.ipynb` (mới) | Kiểm trước: `train.json` trùng sha256 file B2 W07, OCR cache trùng biên bản Phần D và phủ đủ train + test; train → kiểm `train_log` (35.159 câu, r16/α32/attention, LoRA chỉ LLM, OCR khớp, siêu tham số = B2 W07) → eval full test → kết luận; tự trả máy L4 | — |

Run cũ không bị ảnh hưởng: audit chỉ chạy khi train mới; với cấu hình attention r16, audit kỳ vọng đúng 144 lớp LLM và 7.372.800 tham số (như mọi run W07–W09).

## Ngân sách

| Việc | L4 | T4 | CU (ước tính, L4 = 1,54 CU/h) |
|---|---|---|---|
| A2 + A3 | ≈ 7,6 h | — | ≈ 12 |
| OCR train còn thiếu | — | ≈ 2,5 h | ≈ 3 |
| B3-full train + eval full test | ≈ 16 h | — | ≈ 25 |
| **Tổng W10 (ước tính trước)** | ≈ 23,6 h | ≈ 2,5 h | **≈ 40** (còn ≈ 84 trước W10 → ≈ 44 cho W11–W15) |
| **Thực tế** | ≈ 23,8 h (A2 3,51 + A3 2,90 train, 2 eval ≈ 1,2 h; B3-full 12,75 train + 2,78 eval; nạp/kiểm ≈ 0,7 h) | ≈ 2,1 h | số CU thật: đọc ở Colab → Resources khi đóng tuần (trước W10: 86,52 CU lúc 19/10 23:49) |

## Kết quả

### Ablation LoRA — H3 và H4 (20/10/2026, L4, bf16, 4-bit, seed 42, test2000)

Nguồn: `../../experiments/W10_lora_ablation_test2000_L4_seed42/` (`lora_ablation_summary.json`, `paired_B3_vs_A2.json`, `paired_B3_vs_A3.json`); train `W10_A2_qlora_ocr_vietocr_v1_alllinear_r4_…`, `W10_A3_qlora_ocr_vietocr_v1_r64_…`; eval `W10_A2_eval_…`, `W10_A3_eval_…`. Mốc B3 = run W08 seed 42. `lora_ablation.py` đã kiểm A2, A3 chỉ khác B3 ở mục LoRA.

| Run | Lớp gắn LoRA | r / α | Tham số train | EM | ΔEM so với B3 (CI95) | Chỉ B3 đúng / chỉ biến thể đúng | McNemar p | ΔANLS (CI95) |
|---|---|---|---|---|---|---|---|---|
| B3 (mốc) | q, k, v, o | 16 / 32 | 7.372.800 (0,196%) | 0,4925 | — | — | — | — |
| A2 | q, k, v, o, gate, up, down (252 lớp, chỉ LLM) | 4 / 8 | 7.483.392 (0,199%) | 0,4870 | −0,55 [−1,60; +0,45] | 60 / 49 | 0,34 | −0,0025 [−0,0116; +0,0068] |
| A3 | q, k, v, o (144 lớp, chỉ LLM) | 64 / 128 | 29.491.200 (0,779%) | 0,5030 | +1,05 [−0,05; +2,15] | 51 / 72 | 0,071 | +0,0049 [−0,0040; +0,0140] |

**Chi phí** (cùng L4, cùng 500 step):

| Run | Giờ train | s/step | VRAM đỉnh | Loss cuối | Adapter |
|---|---|---|---|---|---|
| B3 | 2,98 | 21,48 | 8,25 GB | 0,6605 | 28,2 MB |
| A2 | 3,51 | 25,24 (+17,5%) | 8,26 GB | 0,6495 | 28,6 MB |
| A3 | 2,90 | 20,85 | 8,58 GB | 0,6380 | 112,5 MB (×4) |

**Kết luận theo luật ghi trước:**
- **H3 — KHÔNG ĐƯỢC ỦNG HỘ** (không khác biệt có ý nghĩa). Ở cùng ngân sách tham số, gắn LoRA vào cả MLP không tốt hơn attention-only (ΔEM −0,55, CI95 [−1,60; +0,45]) nhưng train chậm hơn 17,5%. → Lựa chọn attention-only của B3 được giữ: **rẻ hơn, chất lượng không kém**. QLoRA paper (Dettmers et al., NeurIPS 2023) khuyên gắn LoRA vào mọi lớp tuyến tính, nhưng chính paper cũng nêu yếu tố quan trọng nhất là **tổng số tham số LoRA** — ở đây tổng đó được giữ bằng nhau (lệch 1,5%), và paper đo trên LLM thuần văn bản, không phải VLM tiếng Việt.
- **H4 — ỦNG HỘ (yếu)**: r = 64 không khác biệt có ý nghĩa so với r = 16 (p = 0,071). *Yếu* vì cận trên CI95 ΔEM (+2,15) còn vượt +1 điểm: test2000 chưa đủ nhạy để loại trừ mức tăng 1–2 điểm. r = 64 tốn adapter ×4 và VRAM +0,33 GB.

**Giới hạn và ghi chú trung thực:**
- 1 seed huấn luyện. EM của A3 (0,5030) cao hơn cả 3 seed B3 (0,4880–0,4925, std 0,23 điểm — W09), nhưng đây là quan sát **sau khi thấy số**, không nằm trong luật; theo luật không chạy thêm seed để đổi kết luận. Ghi r = 64 là **hướng cải tiến tiếp** (cần ≥ 3 seed hoặc full test để kết luận).
- B3-full vẫn dùng attention r = 16 (quyết định 19/10, chốt trước khi có kết quả ablation).

### OCR toàn bộ train — notebook 10 Phần D (20/10/2026, T4, GPU)

Nguồn: `../../experiments/W10_ocr_cache_fulltrain/ocr_cache_manifest.json`. Cache trên Drive trùng sha256 biên bản W09 trước khi mở rộng (6.289 ảnh) → OCR thêm **9.063 ảnh, 0 lỗi**, 99,6% ảnh mới có chữ, 0,72 s/ảnh → **15.352 ảnh** (sha256 `e7ce4f76d98c…`). Dấu vân tay 6.289 ảnh cũ **giữ nguyên**; engine trùng đợt W08/W09 (PaddleOCR 3.7.0 · paddlepaddle 3.3.1 · PP-OCRv6_medium_det + VietOCR vgg_transformer). Phủ đủ train 11.733/11.733 và full test 3.353/3.353 ảnh.

### B3-full — QLoRA nhận biết OCR trên toàn bộ train (21/10/2026, L4, bf16, 4-bit, seed 42, full test 10.028 câu)

Nguồn: `../../experiments/W10_B3full_vs_B2_testfull_L4_seed42/b3full_summary.json`; train `W10_B3full_qlora_ocr_vietocr_v1_r16_qwen25vl_vitextvqa_full_ep1_bf16_L4_seed42`; eval `W10_B3full_eval_qlora_ocr_vietocr_v1_r16_full_ep1_vitextvqa_testfull_bf16_L4_seed42`. Các run so sánh: B2 W07, B2-sub và B3 W09 (full test).

**Kiểm hợp lệ (notebook 18 tự kiểm, đều đạt):** `train.json` trùng sha256 file B2 W07; 35.159 câu, loại 0; r 16 / α 32 / attention, LoRA audit 144 lớp chỉ LLM, 7.372.800 tham số; OCR VietOCR · v1 · 600 · 0 (99,77% mẫu train có chữ); lr, epoch, batch, warmup trùng B2 W07; train chạy liền một phiên (không resume).

| Run | Dữ liệu train | OCR | EM | ANLS | F1 (phụ) | s/câu |
|---|---|---|---|---|---|---|
| B2-sub (W09) | 8.000 câu (23%) | không | 46,56 | 68,90 | 67,83 | 0,983 |
| B3 (W09) | 8.000 câu (23%) | có | 49,10 | 70,47 | 70,03 | 0,983 |
| B2 (W07) | 35.159 câu (100%) | không | 50,26 | 72,02 | 71,16 | 1,001 |
| **B3-full** | 35.159 câu (100%) | có | **53,35** | **73,89** | **73,73** | 0,996 |
| ViTextBLIP-2 (paper, Table 6) | 100% | SwinTextSpotter | 25,48 | — | 53,95 | — |

| Hiệu ứng (ghép cặp 10.028 câu) | So sánh | Chỉ A đúng / chỉ B đúng | McNemar p | ΔEM (điểm) | ΔANLS (CI95) |
|---|---|---|---|---|---|
| OCR-aware, 23% dữ liệu (W09) | B2-sub → B3 | 417 / 672 | 1,1 × 10⁻¹⁴ | +2,54 | +0,0157 [+0,0106; +0,0208] |
| **OCR-aware, 100% dữ liệu — SO SÁNH CHÍNH** | **B2 → B3-full** | **395 / 705** | **6,7 × 10⁻²¹** | **+3,09** | **+0,0187 [+0,0138; +0,0236]** |
| Thêm dữ liệu, không OCR | B2-sub → B2 | 355 / 726 | 6,1 × 10⁻³⁰ | +3,70 | +0,0312 [+0,0260; +0,0365] |
| Thêm dữ liệu, có OCR | B3 → B3-full | 341 / 767 | 3,0 × 10⁻³⁸ | +4,25 | +0,0343 [+0,0289; +0,0397] |

Tương tác dữ liệu × OCR (theo câu, chỉ báo cáo): EM +0,0055 [−0,0020; +0,0129] · ANLS +0,0031 [−0,0030; +0,0092].

So với ViTextBLIP-2 (cận dưới CI95 bootstrap > số công bố): EM 53,35 [52,39; 54,33] > 25,48 · F1 73,73 [73,02; 74,45] > 53,95 → **CAO HƠN** cả EM lẫn F1.

| Chi phí | Giờ train | s/step | VRAM đỉnh | Loss cuối | Adapter |
|---|---|---|---|---|---|
| B2 (W07) | 11,99 | 19,63 | 6,88 GB | 0,6179 | cùng cỡ (cùng r16 · attention) |
| B3-full | 12,75 (+6,3%) | 20,89 | 8,38 GB (+1,5) | 0,5598 | 29,53 MB |

Suy luận full test: 9.991 s (0,996 s/câu, chưa tính OCR; OCR chạy trước một lần và cache, 0,72–0,85 s/ảnh trên GPU).

**Kết luận theo luật ghi trước: OCR-AWARE VƯỢT B2 KHI ĐỦ DỮ LIỆU.**

**Diễn giải:**
- **Đính chính W09.** W09 chỉ nói được "hiệu quả dữ liệu" (với 23% dữ liệu, OCR bù ≈ 69% khoảng cách). Nay: **cùng dữ liệu, cùng siêu tham số, chỉ khác OCR**, QLoRA nhận biết OCR hơn QLoRA thường +3,09 EM (p = 6,7 × 10⁻²¹) — cải tiến thật so với baseline đã reproduce.
- **Lợi ích của OCR không mất đi khi thêm dữ liệu** (+2,54 ở 23% → +3,09 ở 100%; tương tác +0,55, CI95 chứa 0 → cộng gộp). Lo ngại "đủ dữ liệu thì model tự đọc chữ tốt, OCR thành thừa" không xảy ra trong phạm vi đo.
- Hai yếu tố cộng dồn: B2-sub → B3-full = +6,79 EM (46,56 → 53,35).
- **Đối chiếu H2** (số mới: OCR tăng ANLS ≥ +3 điểm và cộng hưởng với QLoRA): EM tăng mạnh, nhưng ΔANLS = +1,87 điểm, **vẫn dưới ngưỡng +3** (trên ngưỡng bác bỏ +1) → **H2 vẫn ủng hộ một phần**, như kết luận Review 2. Lợi ích chủ yếu ở việc trả lời **đúng nguyên chuỗi** (EM) hơn là gần đúng ký tự (ANLS).
- **Chi phí của cải tiến nhỏ:** +6% thời gian train, +1,5 GB VRAM (prompt dài hơn ≈ 140 ký tự OCR), tốc độ suy luận như cũ; cộng bước OCR chạy trước.

**Giới hạn:**
- B3-full 1 seed (42). Ở tập con 8k, độ lệch chuẩn EM của B3 qua 3 seed là 0,23 điểm và lợi ích của OCR ổn định 3/3 seed (W09) — nhưng chưa đo lại ở 100% dữ liệu.
- B2 W07 train bằng code commit `22b44a3`, B3-full bằng code W10. Giữa hai commit: `configs/qwen_lora.yaml` không đổi; `src/prompting/builder.py` chỉ thêm nhánh OCR; `train_qlora.py` thêm seed / OCR / resume / LoRA audit (không đổi phép tính khi không bật). B2-sub (code W08) và B2 W07 cùng prompt zero-shot.
- ViTextBLIP-2: paper không công bố cách chuẩn hoá đáp án → so sánh tuyệt đối mang tính tham khảo; khoảng cách (+27,9 EM) lớn hơn nhiều mọi khác biệt chuẩn hoá hợp lý.

## Link experiment log

- Ablation LoRA (A1, A2): `../../experiments/W10_lora_ablation_test2000_L4_seed42/`, `../../experiments/W10_A2_*`, `../../experiments/W10_A3_*`
- OCR cache toàn bộ train: `../../experiments/W10_ocr_cache_fulltrain/ocr_cache_manifest.json`
- B3-full: `../../experiments/W10_B3full_vs_B2_testfull_L4_seed42/` + train `W10_B3full_qlora_…` + eval `W10_B3full_eval_…`
- Adapter (không lên git, trên Drive): `adapters/W10_A2_*`, `adapters/W10_A3_*`, `adapters/W10_B3full_*` (`epoch_1`)

## Deliverables tuần này

| File | Vai trò |
|------|---------|
| [W10_Team_Report.md](W10_Team_Report.md) | Báo cáo tuần: việc đã làm, số liệu chính, blocker, kế hoạch W11 |
| README này | Quyết định, luật ghi trước, kết quả từng bước |
| [../03_report/R6_Results.md](../03_report/R6_Results.md) | R6 bản nháp tiếng Việt: kết quả chính + ablation QLoRA nhận biết OCR |
| `src/adaptation/lora_targets.py`, `src/eval/lora_ablation.py`, `src/eval/b3full.py` | Code mới, có selftest |
| Notebook 17, 18; notebook 10 Phần D | Chạy lại được toàn bộ W10 |

## Trạng thái

✅ **Đóng tuần 21/10/2026.** Kết quả chính: **QLoRA nhận biết OCR vượt QLoRA thường khi đủ dữ liệu** (B3-full 53,35 so với B2 50,26 EM, +3,09, p = 6,7 × 10⁻²¹; ANLS +1,87) — từ "hiệu quả dữ liệu" (W09) thành cải tiến thật; lợi ích OCR không giảm khi thêm dữ liệu. Ablation cấu hình LoRA: A1 (module) — all-linear không tốt hơn attention-only ở cùng ngân sách nhưng chậm hơn 17,5%; A2 (rank) — r = 64 không khác biệt có ý nghĩa (yếu: CI95 chạm +2,15). H2 vẫn ủng hộ một phần (ANLS +1,87 < +3).

---

# FILE: docs/W10/W10_Team_Report.md

# W10 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W10 — Ablation cấu hình LoRA + B3 trên toàn bộ dữ liệu train |
| Thời gian thực | 19–21/10/2026 (tuần thực 7) |
| Mốc kế tiếp | **Review 2 — thứ Năm 29/10/2026** (tuần thực 8; bộ môn dời lịch, báo ngày 20/10) |
| Phần cứng | Colab Pro: NVIDIA L4 (train/eval) · Tesla T4 (OCR) |
| Commit cuối tuần | commit đóng tuần sau `0cf9829` (xem mục "Dấu vết commit") |

## Mục tiêu

Sau W09, phương pháp đề xuất — QLoRA nhận biết OCR (B3) — còn hai câu hỏi hội đồng chắc chắn hỏi:

1. **Cấu hình LoRA (r = 16, chỉ 4 lớp attention) chọn từ W04 có hợp lý không?** QLoRA paper khuyên gắn mọi lớp tuyến tính; chưa kiểm rank. → ablation A1 (module) và A2 (rank).
2. **Đủ dữ liệu thì OCR-aware có vượt B2 không?** W09: B3 (23% dữ liệu) thấp hơn B2 (100%) 1,16 EM trên full test → đóng góp mới chỉ là "hiệu quả dữ liệu". → B3-full.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | `src/adaptation/lora_targets.py`: preset `attn` / `all_linear_llm` (regex chỉ LLM), audit LoRA theo khối + đếm tham số; `train_qlora.py` thêm `--lora_r / --lora_alpha / --lora_targets` + `lora_audit` trong `train_log.json`; `lora_ablation.py`; notebook 17; luật ghi trước | ✅ `8be932e` — selftest bắt được bẫy: list hậu tố MLP sẽ gắn LoRA vào 96 lớp của **vision encoder** |
| 2 | Ablation A1 — all-linear r4 (cùng ngân sách tham số với B3) | ✅ **không tốt hơn** attention-only (−0,55 EM, p = 0,34), **chậm hơn 17,5%** |
| 3 | Ablation A2 — attention r64 (×4 tham số) | ✅ **không khác biệt có ý nghĩa** (+1,05 EM, p = 0,071; CI95 [−0,05; +2,15] → kết luận yếu), adapter ×4 |
| 4 | Notebook 10 Phần D: OCR thêm 9.063 ảnh train | ✅ 0 lỗi, 0,72 s/ảnh; 6.289 ảnh cũ giữ nguyên dấu vân tay |
| 5 | `src/eval/b3full.py` + notebook 18: B3-full (công thức B3 trên 35.159 câu) + full test | ✅ **OCR-AWARE VƯỢT B2 KHI ĐỦ DỮ LIỆU**: +3,09 EM, p = 6,7 × 10⁻²¹ |
| 6 | R6 Results bản nháp tiếng Việt (mục 2–4) | ✅ `docs/03_report/R6_Results.md` |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| Dataset / OCR | OCR toàn bộ train + bảo vệ cache cũ (4) |
| PEFT / Training | Preset + audit LoRA (1); ablation A1, A2 (2, 3); B3-full (5) |
| Evaluation | Luật A1/A2 + CI95 ΔEM (1–3); luật B3-full + bảng 2×2 dữ liệu × OCR (5); R6 (6) |
| VLM / Baseline | Mô hình ViTextVQA chính thức đổi sang B3-full (5) |

### Definition of Done

- [x] Ablation module gắn LoRA ở **cùng ngân sách tham số**, LoRA không lọt vision (audit trong log)
- [x] Ablation rank, α/r giữ nguyên
- [x] OCR cache phủ toàn bộ train, phần cũ không đổi
- [x] B3-full so với B2 cùng dữ liệu, cùng siêu tham số — luật ghi trước, ghép cặp trên full test
- [x] Mọi số liệu truy được về file trong `experiments/`; mọi luật kết luận commit **trước** khi có số
- [x] R6 bản nháp tiếng Việt cho phần OCR-aware
- [x] Dọn Downloads + Drive

## Kết quả / số liệu

### 1. Ablation cấu hình LoRA (test2000, seed 42, công thức B3)

| Ablation | Run | Cấu hình | Tham số | EM | ΔEM (CI95) | p | s/step | Adapter |
|---|---|---|---|---|---|---|---|---|
| — | B3 (mốc) | attention · r16 | 7,37 tr | 0,4925 | — | — | 21,48 | 28,2 MB |
| **A1** module | `W10_A2_*` | all-linear · r4 | 7,48 tr | 0,4870 | −0,55 [−1,60; +0,45] | 0,34 | 25,24 | 28,6 MB |
| **A2** rank | `W10_A3_*` | attention · r64 | 29,49 tr | 0,5030 | +1,05 [−0,05; +2,15] | 0,071 | 20,85 | 112,5 MB |

→ Giữ **attention-only r16**: rẻ nhất, không kém có ý nghĩa. r64 có xu hướng tăng nhưng chưa đủ bằng chứng (1 seed, test2000) → hướng cải tiến tiếp.

### 2. B3-full — dữ liệu × OCR (full test 10.028 câu, seed 42)

| EM | Không OCR | Có OCR (OCR-aware) | Lợi ích OCR |
|---|---|---|---|
| 23% train | 46,56 (B2-sub) | 49,10 (B3) | +2,54 (p = 1,1 × 10⁻¹⁴) |
| 100% train | 50,26 (B2) | **53,35 (B3-full)** | **+3,09 (p = 6,7 × 10⁻²¹)** |

- ANLS B2 → B3-full: 72,02 → 73,89 (+0,0187, CI95 [+0,0138; +0,0236]); F1 71,16 → 73,73.
- Tương tác dữ liệu × OCR: +0,55 EM, CI95 [−0,20; +1,29] → lợi ích OCR **không giảm** khi thêm dữ liệu.
- So với ViTextBLIP-2 (EM 25,48 · F1 53,95): B3-full **cao hơn** cả hai (cận dưới CI95: 52,39 / 73,02).
- Chi phí: train 12,75 h so với 11,99 h (+6%), VRAM 8,38 so với 6,88 GB, suy luận ≈ 1 s/câu như nhau (chưa tính OCR).

### 3. Đối chiếu giả thuyết (đánh số chính thức từ deck cộng dồn Review 2)

| Mục | Trạng thái sau W10 | Lý do |
|---|---|---|
| H2 · OCR tăng ANLS ≥ +3, cộng hưởng với QLoRA | **Ủng hộ một phần** (không đổi) | Cộng hưởng mạnh và nay vượt cả B2 cùng dữ liệu (+3,09 EM); nhưng ΔANLS +1,87 < +3 |
| Ablation A1 · module gắn LoRA | Attention-only giữ nguyên | All-linear không tốt hơn ở cùng ngân sách, chậm hơn 17,5% |
| Ablation A2 · rank | r16 đủ (kết luận yếu) | r64 +1,05 EM, p = 0,071; CI95 chạm +2,15 |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|---|---|---|
| B1 | Kết quả W10 (B3-full, A1/A2) chưa có trong deck cộng dồn Review 2 | Cao (review 29/10) | Tuần thực 8, trước 29/10: thêm vào Phần B của `docs/04_slides/cumulative/build_cumulative_deck.js` (không sửa pptx tay) |
| B2 | `docs/W02/Hypotheses.md` còn đánh số cũ H1–H8 | Thấp | Cập nhật theo đánh số mới (H1–H4, A1/A2) khi viết R4 Methodology |
| B3 | B3-full mới 1 seed | Thấp | W09: std 0,23 EM ở 8k, 3/3 seed ổn định; ghi giới hạn. Chạy thêm seed chỉ khi reviewer yêu cầu (≈ 16 h L4 / seed) |
| B4 | A2 (rank) kết luận yếu | Thấp | Ghi hướng cải tiến tiếp; không chạy thêm để "câu" kết luận |
| B5 | H4 (hallucination) còn chờ | Trung bình | W12: đo kiểu POPE trên B1 / B2 / B3-full |
| B6 | Ngân sách Colab | Trung bình | Kiểm số CU thật khi đóng tuần; adapter dùng chung (W11) ≈ 15–19 CU |

## Kế hoạch tuần sau (W11)

1. **Bảng đánh giá đầy đủ** (EM, VQA-Acc, ANLS, F1 phụ; thời gian train/suy luận; VRAM) cho mọi run chính của cả hai dataset — từ file đã có, không tốn GPU.
2. **Adapter dùng chung 2 dataset** (bản rẻ): thiết kế + luật ghi trước, OCR ảnh ViVQA nếu cần, train + eval.
3. Bắt đầu chuẩn bị nội dung W10 cho deck cộng dồn Review 2 (thực hiện trước 29/10).

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `8be932e` | Bước 0: preset + audit LoRA, `--lora_*`, `lora_ablation.py`, notebook 17, luật A1/A2 (số cũ H3/H4) + luật B3-full ghi trước |
| `530e957` | Kết quả ablation A1/A2; notebook 10 Phần D, notebook 18, `b3full.py` |
| `6359085` | Deck cộng dồn Review 2 (phiên riêng): Phần A = Review 1 đã sửa, Phần B = Review 2; đánh số giả thuyết mới |
| `0cf9829` | Dọn file ghi chép một lần trong `docs/` |
| *(commit đóng tuần)* | Biên bản OCR toàn bộ train, kết quả B3-full, README + Team Report W10, R6 bản nháp |

## Link experiment log

- Ablation LoRA: `experiments/W10_lora_ablation_test2000_L4_seed42/` + run `W10_A2_*`, `W10_A3_*`
- OCR toàn bộ train: `experiments/W10_ocr_cache_fulltrain/`
- B3-full: `experiments/W10_B3full_vs_B2_testfull_L4_seed42/` + run `W10_B3full_*`
- Adapter (không lên git, trên Drive): `adapters/W10_*`

---

# FILE: docs/W11/README.md

# Week 11 — Bảng đánh giá đầy đủ + adapter dùng chung hai dataset

Index tuần 11 (tuần ảo) của ViVQA-VLM. Tuần ảo W11 bắt đầu ngày 21/10/2026, thuộc **tuần thực 7** (19–25/10). **Review 2** diễn ra **thứ Năm 29/10/2026** (tuần thực 8).

Mục tiêu W11 gồm ba phần:

1. **Bảng đánh giá đầy đủ** cho mọi run chính trên cả 2 dataset. Số liệu tính lại từ file có sẵn, không dùng GPU. Bảng này là nguồn duy nhất cho phần Results (R6) và slide.
2. **Adapter dùng chung** cho ViVQA + ViTextVQA (đã hứa trên slide Review 2). Câu hỏi cần trả lời: một adapter có phục vụ được cả hai loại câu hỏi mà không kém adapter chuyên biệt hay không?
3. **Phân rã theo loại câu hỏi** cho ViTextVQA, rồi đưa kết quả W10 vào deck cộng dồn trước ngày 29/10.

> Ngôn ngữ: bản **tiếng Việt** (bản làm việc). Bản tiếng Anh cho thesis sẽ chuyển từ bản này sau khi nội dung chốt.

## Kế hoạch tuần

| # | Việc | GPU | Trạng thái |
|---|------|-----|-----------|
| 1 | Luật ghi trước (README này): adapter dùng chung S2, quy ước VQA-Acc, luật bộ phân loại câu hỏi | — | ✅ 21/10 |
| 2 | `src/eval/full_eval_table.py`: bảng đầy đủ 21 dòng × 3 bảng + chi phí + ổn định theo seed + loại câu hỏi ViVQA; tự kiểm metrics.json bằng cách tính lại từ predictions.json | — | ✅ 21/10 (`experiments/W11_full_eval_table/`) |
| 3 | OCR cho ảnh ViVQA train + test (notebook `19_ocr_cache_vivqa.ipynb`) | ≈ 20 phút T4 | ✅ 21/10 (lần 3; lần 1–2 hỏng do cài Paddle, xem mục Sự cố) |
| 4 | Code train **trộn 2 dataset** (`train_qlora.py --mix`) + notebook `20_qlora_shared_adapter_qwen25vl.ipynb` + script kết luận `src/eval/shared_adapter.py` | — | ✅ 21/10 |
| 5 | **SH**: train adapter dùng chung + eval ViVQA test và ViTextVQA test2000 + kết luận theo luật (notebook 20) | 6,07 h + ≈ 1,1 h L4 | ✅ 21/10 — **ADAPTER DÙNG CHUNG ĐẠT** |
| 6 | Bộ phân loại câu hỏi ViTextVQA (theo luật) + bảng EM/ANLS theo loại | — | ✅ 21/10 (luật `qt-v1` đóng băng 15:17 `9b441f1`; 150 nhãn tay; bảng tính sau commit) |
| 7 | Đưa kết quả W10 + W11 vào deck cộng dồn | — | ✅ 22/10 — **bản nháp** `ViVQA-VLM_Review1-2_KLTN.pptx` (44 slide, có lời nói từng slide). Không làm qua builder: xem Quyết định 22/10 |
| 8 | Tập nói Review 2 | — | ➜ **chuyển W12** (trước 29/10): Hoàng nắm Phần A trước, rồi đọc và sửa Phần B |
| 9 | Đóng tuần: README, `W11_Team_Report.md`, R6 (mục 3, 4.5, 5), deck vào repo, dọn Downloads + Drive, handoff | — | ✅ 22/10 |

## Quyết định đã chốt

| Ngày | Quyết định | Lý do |
|------|-----------|-------|
| 21/10/2026 | **Adapter dùng chung theo phương án S2**: mọi câu hỏi của cả 2 dataset dùng chế độ `ocr` (VietOCR · v1 · 600 · 0). Ảnh không có chữ thì prompt tự trùng hệt zero-shot | (1) **Khớp prototype**: lúc demo, hệ thống không biết câu hỏi "thuộc dataset nào". S1 (ViVQA không OCR) buộc phải có bộ định tuyến, hoặc đưa vào dòng OCR mà adapter chưa từng thấy với câu hỏi thường. Khi đó điểm eval không phản ánh đúng demo. (2) **Tách hiệu ứng vẫn làm được mà không tốn GPU**: với câu ViVQA có ảnh không đọc ra chữ, đầu vào của SH trùng hệt C1, nên chênh lệch ở nhóm này chỉ do trộn dữ liệu. (3) **Trả lời thêm một câu hỏi nghiên cứu**: chữ không liên quan trong ảnh thường có làm hại câu hỏi chung không. Câu này nối sang phân tích hallucination ở W12. (4) Phía ViTextVQA thì S1 = S2, nên so với B3 chỉ khác đúng việc trộn dữ liệu |
| 21/10/2026 | **VQA-Acc ≡ EM** trên cả hai dataset; vẫn báo cột VQA-Acc (metric bắt buộc) kèm chú thích | `full_eval_table.py` đếm được **0** câu có hơn 1 đáp án gold ở cả ViVQA lẫn ViTextVQA. Công thức min(#khớp/3, 1) khi đó rút về đúng/sai, nên trùng EM ở **từng câu** |
| 22/10/2026 | **Deck cộng dồn: file pptx là nguồn duy nhất, sửa tay trong PowerPoint**; builder `build_cumulative_deck.js` và hai deck Review2_Cumulative chuyển vào `docs/04_slides/cumulative/archive_W10_builder/` (giữ làm lịch sử, không build lại) | Hoàng soạn lại slide theo cách nói của mình (Inter, bố cục riêng), và deck mới ghép slide của Hoàng với slide cũ. Builder JS không sinh lại được những slide đó. Hai nguồn song song sẽ lệch nhau, nên chỉ giữ một. Lời nói nằm trong speaker notes; bản `_Script_TV.txt` trích từ notes để đọc nhanh |
| 21/10/2026 | **H4 (QLoRA không làm tăng hallucination) dời sang W12** | Đúng mốc "W12 · Analysis" trong master. W11 dành GPU cho SH. Kết quả SH (dòng OCR trên ảnh COCO) cũng là đầu vào cho phân tích "chép nhầm chữ OCR" ở W12 |

## Luật adapter dùng chung — SH (ghi TRƯỚC khi chạy, 21/10/2026)

### Run

| Mục | Giá trị |
|---|---|
| Train | `W11_SH_qlora_ocr_vietocr_v1_r16_qwen25vl_shared_vivqa_sub8k_ep1_bf16_L4_seed42` |
| Dữ liệu train | ViVQA `data/vivqa/train.json` (10.799 câu, sha256 `892243fa732f…`, **cùng file với C1**) **+** ViTextVQA `train_subset8000_byimage_seed42.json` (8.000 câu, sha256 `093ab061edbb…`, **cùng file với B3**) = **18.799 câu**. Ghép hai danh sách rồi để bộ lấy mẫu ngẫu nhiên của Trainer (seed 42) **xáo lẫn** hai dataset. **Không** tăng hay giảm tỉ lệ mẫu (giữ tỉ lệ tự nhiên 57 : 43) |
| Prompt | Chế độ `ocr` cho **mọi** câu. VietOCR (`paddledet_vietocr`) · template v1 · cắt 600 ký tự · min_conf 0. Mỗi câu tra OCR trong cache **của dataset mình**: ViVQA = cache mới W11 (biên bản `experiments/W11_ocr_cache_vivqa/`), ViTextVQA = cache W10 (sha256 `e7ce4f76d98c…`) |
| Siêu tham số | **Y hệt B3 và C1**: LoRA r 16, α 32, q/k/v/o (chỉ LLM), dropout 0,05; lr 2e-4; 1 epoch; batch 2 × tích luỹ 8 = 16; warmup 3%; weight decay 0; bf16; 4-bit NF4 + double quant; min/max_pixels 200.704 / 401.408; seed 42; L4 |
| Eval | `W11_SH_eval_…_vivqa_test_…` (3.001 câu) và `W11_SH_eval_…_vitextvqa_test2000_…` (2.000 câu). Chế độ `ocr`, greedy, max_new_tokens 32, giống eval C1 / B3 |

### Điều kiện hợp lệ (kiểm trước khi eval; sai → không dùng run, báo Claude)

- `train_log.json`: `num_samples` = 18.799, tách theo dataset **10.799 / 8.000**, `num_dropped` = 0.
- `lora_audit.by_tower` = `{'llm': 144}`, không có `vision`; tham số train = **7.372.800**.
- Dấu vân tay OCR của phần ViTextVQA (`subset_sha256`) = **`c2d7cb1f75b8…`**, trùng `train_log` của B3. Điều này đảm bảo phần ViTextVQA nhận **đúng** chuỗi OCR mà B3 đã nhận.
- Cache ViVQA trên Drive trùng sha256 trong biên bản W11. Engine OCR (paddleocr, det_model, rec_model, vietocr_config, lang) trùng cache ViTextVQA.

### So sánh chính (ghép cặp trên cùng câu hỏi)

| Dataset | Mốc (cùng điều kiện) | Vì sao chọn mốc này |
|---|---|---|
| ViVQA test (3.001) | **C1** (QLoRA ViVQA, 1 epoch, W07): EM 73,78 | Cùng file train ViVQA, cùng 1 lần duyệt dữ liệu ViVQA, cùng siêu tham số |
| ViTextVQA test2000 | **B3** (QLoRA + OCR, tập con 8k, seed 42, W08): EM 49,25 | Cùng tập con 8k, cùng OCR, cùng siêu tham số |

Thống kê: ΔEM = SH − mốc, kèm **CI95 bootstrap ghép cặp** (10.000 lần, seed 42) và **McNemar exact** trên EM. ΔANLS kèm CI95.

**Biên "không kém" δ = 1,5 điểm EM**, áp cho **từng** dataset:

| Điều kiện | Kết luận cho dataset đó |
|---|---|
| Cận dưới CI95 của ΔEM > −1,5 | **KHÔNG KÉM** |
| … và thêm ΔEM > 0, p < 0,05 | **KHÔNG KÉM — TỐT HƠN** (chuyển giao dương giữa hai loại câu hỏi) |
| Cận trên CI95 của ΔEM < −1,5 | **KÉM** |
| Còn lại | **CHƯA KẾT LUẬN** (CI95 chứa −1,5) |

**Kết luận chung:**
- Cả hai dataset **KHÔNG KÉM** → **ADAPTER DÙNG CHUNG ĐẠT**. Prototype dùng **1 adapter (SH)**. Số chính thức trong bảng Results vẫn là C3 (ViVQA) và B3-full (ViTextVQA); khoảng cách giữa SH và hai mốc này báo cáo minh bạch.
- Còn lại → **ADAPTER DÙNG CHUNG KHÔNG ĐẠT**. Prototype dùng **2 adapter chuyên biệt** (C3, B3-full) và chọn adapter theo ảnh (PEFT hỗ trợ nạp nhiều adapter). Cách chọn adapter quyết định ở tuần prototype.

**Vì sao δ = 1,5 điểm:** con số này bằng khoảng một nửa mức lợi mà OCR-aware mang lại khi đủ dữ liệu (+3,09 EM, W10) và nhỏ hơn mức lợi C1 → C3 của epoch thứ hai (+2,87). Kém hơn mức này nghĩa là xoá một phần đáng kể đóng góp của đề tài. Độ chính xác cũng đủ: nửa độ rộng CI95 của ΔEM ghép cặp là khoảng **0,95** điểm trên ViVQA (C1 → C3) và khoảng **1,0** điểm trên test2000 (ablation A1 W10). Vì vậy, nếu thật sự không có chênh lệch, cận dưới vẫn nằm trên −1,5.

### Chỉ báo cáo (không có luật kết luận)

1. **Tách hiệu ứng trên ViVQA test:** chia câu theo việc prompt S2 có dòng OCR hay không.
   - Nhóm **không có dòng OCR**: đầu vào SH trùng hệt C1, nên ΔEM ở nhóm này chỉ do **trộn dữ liệu**.
   - Nhóm **có dòng OCR**: ΔEM là tác động gộp của **trộn dữ liệu + dòng OCR**.
   - Mỗi nhóm báo ΔEM kèm CI95 và số câu.
2. **So với mô hình chính thức** C3 (ViVQA, 2 epoch) và B3-full (ViTextVQA, toàn bộ train). Điều kiện **không** giống nhau (khác số epoch, khác lượng dữ liệu), nên chỉ mô tả, không kết luận.
3. **Chi phí:** 1 adapter (≈ 29,5 MB) so với 2; giờ train SH so với tổng C1 + B3; s/câu.
4. **Chẩn đoán chép OCR trên ViVQA:** tỉ lệ dự đoán sai (EM = 0) mà là chuỗi con của văn bản OCR. Đây là số đầu vào cho phân tích hallucination ở W12.

### Không được làm

- Không chạy lại với tỉ lệ trộn, số epoch hay seed khác sau khi thấy kết quả để đổi kết luận.
- Không đổi mốc so sánh (vd so SH với C3) để "đạt".

### Giới hạn đã biết

- 1 seed (42). Độ lệch chuẩn EM của B3 qua 3 seed là 0,23 điểm (W09), nhỏ hơn nhiều so với δ.
- Lịch learning rate dài hơn (≈ 1.175 step so với 675 của C1 và 500 của B3). Điều này đi kèm việc trộn dữ liệu, không tách riêng được.
- Cache OCR ViVQA tạo ở W11 (cùng engine với ViTextVQA, đã kiểm ở notebook 19).

## Luật bộ phân loại câu hỏi ViTextVQA (ghi trước, 21/10/2026)

ViTextVQA không có nhãn loại câu hỏi (`question_type` = `unknown` ở mọi run), trong khi master yêu cầu error analysis theo loại câu hỏi.

1. Bộ phân loại là **luật từ khoá** (không dùng model). Luật được thiết kế **chỉ trên câu hỏi dev**, không nhìn câu hỏi test và không nhìn dự đoán của model.
2. Luật được **commit (đóng băng) trước** khi tính bất kỳ số nào theo loại trên test.
3. Kiểm độ chính xác: Hoàng tự gán nhãn tay **150 câu** test2000 (chọn ngẫu nhiên, seed 42) mà **không xem** dự đoán. Báo độ chính xác tổng và theo loại. Loại nào có độ chính xác < 80% thì gắn cờ "nhãn kém tin cậy" trong bảng.
4. Loại có < 50 câu trên test2000 thì gộp vào "khác".
5. Bảng theo loại chỉ để **mô tả / error analysis**, không dùng để chọn mô hình.
6. *(Làm rõ ngày 21/10, trước khi có nhãn tay và trước khi tính số theo loại trên test.)*
   - "Độ chính xác theo loại" = **precision**: trong các câu máy gán loại T, bao nhiêu % Hoàng cũng gán T. Đây là thứ ảnh hưởng tới bảng, vì bảng nhóm câu theo nhãn máy. Recall chỉ báo kèm để tham khảo.
   - Loại có < 5 câu máy gán trong mẫu 150 thì ghi "n quá nhỏ", không gắn cờ.
   - Hướng dẫn gán nhãn: `question_types_guide.md`.

**Bộ phân loại `qt-v1` (đóng băng 21/10/2026):** 9 loại, xếp theo thứ tự ưu tiên: `source` · `contact` · `number_time` · `person` · `location` · `name` · `text_content` · `product_service` · `other`. Luật từ khoá được viết **chỉ** từ 300 câu hỏi dev (dev300). Chưa chạy bộ phân loại trên câu test nào trước khi commit. Phân bố trên dev300: name 83 · product_service 42 · other 31 · text_content 30 · location 30 · number_time 30 · source 23 · contact 20 · person 11. ViTextVQA gần như **không có câu có/không**, nên loại "yes/no" của master không áp dụng cho dataset này.

## Thay đổi code

| File | Thay đổi | Kiểm |
|------|----------|------|
| `src/eval/full_eval_table.py` (mới) | Danh sách 21 run chính (3 bảng: ViTextVQA full test, test2000, ViVQA test) → EM / VQA-Acc / ANLS / F1, ΔEM so với B1, chi phí lần ngược từ adapter → `train_log.json`, ổn định theo seed (B2-sub, B3), EM theo loại câu hỏi (ViVQA), kiểm quy ước VQA-Acc. **Tính lại mọi metric từ predictions.json**, dừng nếu lệch metrics.json (> 1e-9), nếu adapter không thuộc run train khai báo, hoặc nếu các run trong cùng bảng khác tập câu hỏi | `--selftest` (bắt được metrics.json bị sửa và adapter sai run). Chạy thật: 21/21 run khớp metrics.json; số khớp README W07–W10 (vd B3-full 53,35 / 73,89; B3 test2000 49,25; C3 76,64) |
| `notebooks/19_ocr_cache_vivqa.ipynb` (mới) | OCR VietOCR cho ảnh ViVQA train + test (T4). Bắt buộc engine trùng cache ViTextVQA W10. Thống kê % câu có dòng OCR, đáp án∈OCR; biên bản `experiments/W11_ocr_cache_vivqa/` | Cú pháp các cell đã kiểm; dùng lại `src/data/ocr.py` (đã hỗ trợ `--dataset vivqa`) |
| `src/adaptation/train_qlora.py` | `--mix DATASET=FILE` (lặp lại, ≥ 2 dataset) thay `--dataset/--data_file`. Mỗi phần nạp, lọc và gắn OCR **bằng đúng hàm của run đơn**, với OCR cache mặc định của dataset đó. Thiết lập OCR (engine, template, ngưỡng) **phải trùng** giữa các phần, nếu không thì dừng. Ghép rồi xáo bằng `random.Random(seed)` (Trainer xáo thêm lần nữa). Chữ ký run gồm sha256 từng file + dấu vân tay OCR từng phần. `train_log.json` có thêm `data_parts`, `ocr.stats_by_dataset`, `mix_order`. Run đơn **không đổi** chữ ký | `--selftest` thêm `mix (W11) OK`: phần ViTextVQA trong run trộn có **cùng dấu vân tay OCR** như run đơn (B3); ảnh không chữ cho prompt **trùng zero-shot**; chạy lại cùng lệnh thì cùng chữ ký (resume được); engine OCR lệch thì dừng; xáo tái lập được, trộn đều. Các selftest cũ vẫn qua |
| `src/eval/shared_adapter.py` (mới) | Luật SH: kiểm hợp lệ train_log (file = C1 + B3, siêu tham số = C1 = B3, LoRA 144 lớp LLM, 7.372.800 tham số, OCR ViTextVQA = dấu vân tay B3) và 2 run eval (cùng file test, cùng OCR với B3 trên test2000, adapter đúng run SH). Sau đó so sánh C1 → SH, B3 → SH (ΔEM + CI95, McNemar, ΔANLS) và kết luận theo δ = 1,5. Chỉ báo cáo: tách ViVQA theo dòng OCR, tỉ lệ chép chữ OCR, so với C3 / B3-full trên cùng câu, chi phí | `--selftest`. Chạy thử trên predictions thật (SH tạm thay bằng run có sẵn): B3 → A3 = +1,05 [−0,05; +2,15], p = 0,071, **khớp** README W10; C1 → C3 = +2,87, **khớp** W09 |
| `notebooks/20_qlora_shared_adapter_qwen25vl.ipynb` (mới) | Kiểm trước (sha256 2 file train = C1 / B3, 2 OCR cache = biên bản W11 / W10, phủ đủ ảnh, run mốc, selftest 5 script) → train SH → kiểm hợp lệ → eval ViVQA test → eval test2000 → kết luận. Bỏ qua phần đã xong, tự trả máy L4 | Cú pháp các cell đã kiểm |
| `src/data/ocr.py` | **Dừng ngay** khi 20 ảnh đầu của một lần chạy đều lỗi (không lưu ảnh lỗi nào). Chạy CPU thì **tắt oneDNN**; chạy GPU giữ nguyên như W08–W10 | `--selftest` thêm `ocr fail-fast OK` |
| `notebooks/19_ocr_cache_vivqa.ipynb` | Cách ly cache lỗi (đổi tên `.bad_<giờ>.json`). Mục 4 bắt buộc 0 ảnh lỗi và tự dừng nếu ước tính > 4 h. Tải Paddle GPU thử lại 3 lần, ghim `paddlepaddle-gpu==3.3.1` | Lần 3 chạy đúng (GPU, 0 lỗi) |
| `src/eval/question_types.py` (mới) | Bộ phân loại `qt-v1` (9 loại, luật từ khoá, dấu vân tay bộ luật `rules_sha256`). `--make_sheet` tạo phiếu 150 câu test2000 (seed 42, chỉ có câu hỏi). `--score` so nhãn tay với nhãn máy (accuracy, precision/recall theo loại, cờ). `--table` cho bảng EM/ANLS theo loại cho các run ViTextVQA trong `full_eval_table.RUNS` (test2000 + full test), gộp loại < 50 câu | `--selftest`: 16 câu mẫu (gồm bẫy "giảm giá", "đánh giá", "chữ … tên"), phiếu tái lập, nhãn lạ hoặc phiếu sai câu thì báo lỗi, bảng trên dữ liệu giả |
| `docs/W11/qt_labels_150.csv` (mới, 21/10 tối) | 150 nhãn tay (chuyển từ Excel; giữ nguyên câu hỏi gốc, kể cả câu stt 97 người gán đã sửa chữ trong phiếu); `experiments/W11_question_types/` = kết quả `--score` + `--table` | Kiểm: đủ 150 câu, đúng `question_id` và thứ tự của phiếu, nhãn hợp lệ, 26 câu có ghi chú |
| `docs/W11/question_types_guide.md`, `docs/W11/qt_label_sheet_150.csv` (mới) | Hướng dẫn gán nhãn và phiếu trống 150 câu (bản Excel có danh sách thả xuống được gửi riêng để gán) | Phiếu sinh lại từ script thì trùng byte |

## Ngân sách

| Việc | L4 | T4 | CU (ước tính; L4 ≈ 1,54 CU/h, T4 ≈ 1,2 CU/h) |
|---|---|---|---|
| OCR ViVQA (≈ 9–10 nghìn ảnh × ≈ 0,75 s) | — | ≈ 2 h | ≈ 2,5 |
| SH train (≈ C1 3,03 h + B3 2,98 h) | ≈ 6,1 h | — | ≈ 9,4 |
| SH eval ViVQA test + ViTextVQA test2000 (+ nạp model) | ≈ 1,3 h | — | ≈ 2 |
| **Thực tế W11** | ≈ 7,3 h (train 6,07 + eval ≈ 1,1 + kiểm/nạp) | ≈ 0,6 h (3 lần chạy notebook 19, lần 3 ≈ 20 phút) | **≈ 12** (ước tính từ giờ chạy: 7,3 × 1,54 + 0,6 × 1,2) |
| **Còn lại** | | | **≈ 37** (49,3 lúc 21/10 04:10 − 12). Số ước tính: Hoàng đọc số thật ở Colab → Resources đầu W12 và ghi vào README W12 |

## Sự cố

| Ngày | Sự cố | Xử lý |
|---|---|---|
| 21/10/2026 | Notebook 19 lần 1–2: tải wheel Paddle GPU bị timeout nên cell tự chuyển sang CPU. Paddle 3.3 CPU + oneDNN lỗi PIR với **mọi** ảnh, cache ghi 9.770 ảnh lỗi. Mục 6 chặn được (lỗi > 1%), chưa run nào dùng cache này | Cache lỗi được đổi tên thành `paddledet_vietocr.bad_….json` (giữ làm bằng chứng, xoá khi dọn Drive cuối tuần). Thêm fail-fast, 0 lỗi ở mục 4, tải lại Paddle GPU, tắt oneDNN trên CPU. Lần 3: GPU, 9.770 ảnh, 0 lỗi, 0,11 s/ảnh |

## Kết quả

### Bảng đánh giá đầy đủ (21/10/2026, không GPU)

Nguồn: `../../experiments/W11_full_eval_table/` (`full_eval_table.md` / `.csv` / `.json`). Bảng thesis lấy số **từ file này**, không chép tay từ README các tuần trước.

| Dataset · tập test | Zero-shot B1 | QLoRA không OCR | **Đề xuất** | ΔEM so với B1 |
|---|---|---|---|---|
| ViTextVQA · full test (10.028) | EM 29,12 · ANLS 47,60 | B2: 50,26 · 72,02 | **B3-full: 53,35 · 73,89** | +24,23 |
| ViVQA · test (3.001) | EM 22,09 · ANLS 28,61 | C1: 73,78 · 75,66 | **C3: 76,64 · 78,03** | +54,55 |
| *Adapter dùng chung SH* | | | ViVQA 73,51 · 75,45 · test2000 49,50 · 71,12 | (W11, cho prototype) |

EM ViVQA theo loại câu hỏi (nhãn có sẵn của dataset). Đây là số mô tả, phần giải thích để dành cho error analysis W12:

| Run | object (1.247) | number (444) | color (625) | location (685) |
|---|---|---|---|---|
| B1 | 5,69 | 69,14 | 44,80 | 0,73 |
| C1 | 73,54 | 77,03 | 83,04 | 63,65 |
| C3 | 76,66 | 77,03 | 84,32 | 69,34 |

### OCR ViVQA (notebook 19, lần 3, 21/10/2026, T4 GPU)

Biên bản `../../experiments/W11_ocr_cache_vivqa/` (trên Drive; chép vào git khi đóng tuần): 9.770 ảnh (train 8.255 + test 2.789, có ảnh dùng chung), 0 lỗi, 0,11 s/ảnh, sha256 `3f2605808c38…`. Engine **trùng** cache ViTextVQA (PaddleOCR 3.7.0, Paddle 3.3.1, PP-OCRv6_medium_det + VietOCR vgg_transformer, GPU).

| | ViVQA train | ViVQA test |
|---|---|---|
| % câu có dòng OCR trong prompt S2 | 44,8 | 44,2 |
| Trung bình ký tự OCR / câu | 8 | 8 |
| Đáp án gold nằm trong OCR | — | **0,0%** |

Với ViVQA, dòng OCR **chỉ là nhiễu**. Ảnh COCO chủ yếu có chữ tiếng Anh, VietOCR đọc thành chuỗi vô nghĩa, conf trung bình 0,648. Vì vậy SH còn là phép thử độ bền với nhiễu OCR (xem phần tách nhóm và chép chữ OCR trong luật SH).

### Loại câu hỏi ViTextVQA (bước 6, 21/10/2026, chỉ mô tả)

Nguồn: `../../experiments/W11_question_types/` (`question_types_table.md` / `.json`, `qt_label_check.json`), nhãn tay `qt_labels_150.csv`. Thứ tự đúng như luật:
- bộ luật `qt-v1` được commit (`9b441f1`, 21/10 15:17) **trước** khi chạy trên câu test nào;
- 150 câu được gán nhãn mù (chỉ thấy câu hỏi), người gán nhãn: **Phạm Lê Huy Hoàng** (đúng luật ghi trước);
- sau đó mới chấm và tính bảng.

**Kiểm nhãn (150 câu test2000):** accuracy **80,7%**.

| Loại | Máy gán | Người gán | Precision | Recall | Cờ |
|---|---|---|---|---|---|
| source | 16 | 15 | 94% | 100% | ok |
| contact | 11 | 11 | 100% | 100% | ok |
| number_time | 12 | 12 | 100% | 100% | ok |
| person | 10 | 13 | 100% | 77% | ok |
| location | 18 | 23 | 100% | 78% | ok |
| name | 23 | 26 | 70% | 62% | ⚠ nhãn kém tin cậy |
| text_content | 11 | 6 | 45% | 83% | ⚠ nhãn kém tin cậy |
| product_service | 15 | 16 | 87% | 81% | ok |
| other | 34 | 28 | 62% | 75% | ⚠ nhãn kém tin cậy |

Lỗi chính là nhầm giữa ba loại "mềm":
- người gán `name` nhưng máy gán `other` (8 câu), chủ yếu câu hỏi tên tổ chức kiểu "đơn vị nào / bộ nào / sở nào", mà luật không có từ khoá;
- người gán `other` nhưng máy gán `text_content` (4 câu): câu có "nói / nội dung" nhưng hỏi ý nghĩa, không đòi đọc lại chữ.

Theo luật, **không sửa `qt-v1` sau khi thấy test**. Ba loại bị gắn ⚠ chỉ dùng để mô tả, và ghi là giới hạn. Muốn có `qt-v2` thì phải có bộ nhãn mới, nằm ngoài kế hoạch.

**Sai lệch so với luật ghi trước:**
- câu stt 97 (`test_41608`, "cửa hàng bên trái **bên** gì ?") bị người gán sửa chữ thành "tên gì" trong phiếu. Bộ phân loại vẫn chạy trên câu gốc, nên không ảnh hưởng nhãn máy.

**EM % theo loại trên full test (10.028 câu)** — mọi loại ≥ 50 câu trên test2000 nên không gộp (đủ số kèm ANLS: `question_types_table.md`):

| Run | source (661) | contact (641) | number_time (1.041) | person (420) | location (969) | name ⚠ (1.971) | text_content ⚠ (777) | product_service (1.144) | other ⚠ (2.404) | Tất cả |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 24,5 | 47,9 | 41,2 | 40,7 | 25,3 | 34,9 | 27,4 | 16,5 | 21,5 | 29,1 |
| B2-sub | 49,8 | 52,0 | 54,7 | 49,8 | 38,5 | 51,4 | 45,2 | 39,5 | 43,2 | 46,6 |
| B3 | 50,2 | 59,0 | 56,3 | 57,1 | 42,9 | 52,3 | 47,6 | 40,9 | 45,9 | 49,1 |
| B2 | 57,2 | 55,5 | 55,7 | 54,8 | 44,0 | 55,7 | 48,6 | 42,5 | 46,1 | 50,3 |
| **B3-full** | **58,1** | **64,1** | **58,8** | **61,2** | **47,0** | **57,6** | **51,1** | **44,1** | **49,6** | **53,4** |
| Δ B2 → B3-full | +0,9 | **+8,6** | +3,1 | +6,4 | +3,0 | +1,9 | +2,5 | +1,6 | +3,5 | +3,1 |

Nhận xét **chỉ mô tả** (chưa kiểm định theo loại, không có luật; giải thích đầy đủ để dành cho error analysis W12):
1. OCR giúp nhiều nhất ở **`contact`** (+8,6 EM): số điện thoại, website, tức chuỗi phải chép đúng từng ký tự. Trên test2000, bỏ OCR lúc suy luận khỏi B3 làm `contact` giảm 64,4 → 53,0, mức giảm lớn nhất trong các loại.
2. OCR giúp ít ở **`source`** (+0,9): watermark nguồn ảnh thường nhỏ, mờ hoặc nằm ở góc. Riêng QLoRA đã tăng mạnh loại này (B1 24,5 → B2 57,2), có thể vì nguồn ảnh lặp lại giữa train và test.
3. Loại **khó nhất** sau B3-full là **`product_service`** (44,1), **`location`** (47,0) và **`other`** (49,6). Đây là các câu cần suy ra từ chữ ("cửa hàng bán gì", "ở tỉnh nào"), không chỉ chép lại chữ.
4. Mô hình 7B tham chiếu R1b yếu nhất ở `product_service` (3,0) và `name` (9,8) trên test2000. Nhiều khả năng do định dạng trả lời dài, cần kiểm ở W12.

### Adapter dùng chung — SH (21/10/2026, L4, bf16, 4-bit, seed 42)

Run: `W11_SH_qlora_ocr_vietocr_v1_r16_qwen25vl_shared_vivqa_sub8k_ep1_bf16_L4_seed42` (train), `W11_SH_eval_qlora_ocr_vietocr_v1_r16_shared_ep1_{vivqa_test,vitextvqa_test2000}_bf16_L4_seed42` (eval), `../../experiments/W11_shared_adapter_L4_seed42/shared_adapter_summary.json` (kết luận).

**Điều kiện hợp lệ: đạt hết** (`✔ kiểm hợp lệ`):
- 18.799 câu (ViVQA 10.799 = file C1, ViTextVQA 8.000 = file B3), không loại câu nào;
- LoRA 144 lớp, chỉ ở LLM, 7.372.800 tham số; siêu tham số = C1 = B3;
- OCR phần ViTextVQA trùng dấu vân tay B3; OCR test2000 trùng B3;
- 1.175 step, không chạy tiếp từ checkpoint nào; train bằng code `72286c4`.

| Dataset | Mốc → SH (EM) | ΔEM (CI95) | Chỉ mốc đúng / chỉ SH đúng | McNemar p | ΔANLS (CI95) | Kết luận |
|---|---|---|---|---|---|---|
| ViVQA test (3.001) | C1 73,78 → **73,51** | −0,27 [−1,13; +0,60] | 91 / 83 | 0,60 | −0,21 [−0,99; +0,60] | **KHÔNG KÉM** |
| ViTextVQA test2000 | B3 49,25 → **49,50** | +0,25 [−0,95; +1,45] | 75 / 80 | 0,75 | +0,73 [−0,33; +1,76] | **KHÔNG KÉM** |

➜ **KẾT LUẬN THEO LUẬT (δ = 1,5 điểm EM): ADAPTER DÙNG CHUNG ĐẠT.** Prototype dùng **1 adapter (SH)**. Không có bằng chứng chuyển giao dương: cả hai p đều ≥ 0,05.

**Chỉ báo cáo:**

| Mục | Kết quả |
|---|---|
| ViVQA · câu **không có** dòng OCR (1.673 câu, đầu vào SH = C1) | C1 73,58 → SH 72,98, ΔEM −0,60 [−1,79; +0,60]. Chênh lệch chỉ do trộn dữ liệu và nằm trong nhiễu |
| ViVQA · câu **có** dòng OCR vô nghĩa (1.328 câu) | C1 74,02 → SH 74,17, ΔEM +0,15 [−1,05; +1,36]. **Dòng OCR nhiễu không làm hại** |
| Chép chữ OCR vào đáp án sai (ViVQA, câu có dòng OCR) | C1 3/1.328 = 0,23% · SH 4/1.328 = 0,30%. Gần như không có (đầu vào cho phân tích hallucination W12) |
| So với mô hình chính thức (khác điều kiện, chỉ mô tả) | C3 76,64 → SH 73,51 (−3,13 [−4,13; −2,13]); B3-full trên cùng 2.000 câu 53,85 → SH 49,50 (−4,35 [−5,80; −2,90]). Lý do: SH chỉ có 1 epoch ViVQA và 23% dữ liệu ViTextVQA (bản rẻ, đã ghi trước) |
| Chi phí | Train 6,07 h (C1 + B3 = 6,01 h); 18,58 s/step; VRAM đỉnh 8,15 GB; **1 adapter 29,5 MB thay vì 2**; suy luận 0,57 s/câu (ViVQA), 1,00 s/câu (ViTextVQA), như các adapter chuyên biệt |

**Ý nghĩa cho thesis:** chia sẻ một adapter cho hai loại câu hỏi **không làm mất độ chính xác** so với adapter chuyên biệt cùng ngân sách. Model cũng học được cách bỏ qua dòng OCR vô nghĩa trên ảnh thường. Bảng kết quả chính vẫn dùng C3 (ViVQA) và B3-full (ViTextVQA).

**Hướng tiếp (chưa quyết, không chạy trong W11):** SH-full, tức SH trên ViVQA 2 epoch + ViTextVQA toàn bộ, để prototype đạt mức C3 / B3-full. Ước tính khoảng 19 h L4 ≈ 29 CU, gần hết ngân sách còn lại; cân nhắc sau Review 2.

`full_eval_table` (23 dòng) và bảng theo loại câu hỏi (test2000) đã có thêm dòng SH.

### Deck cộng dồn Review 1 + Review 2 (bước 7, 22/10/2026 — bản nháp)

File: `../04_slides/cumulative/ViVQA-VLM_Review1-2_KLTN.pptx` (44 slide, 20 × 11,25 in) + `ViVQA-VLM_Review1-2_KLTN_Script_TV.txt` (lời nói trích từ speaker notes). Dựng từ `KLTN.pptx` của Hoàng: giữ nguyên slide 1–8 Hoàng đã soạn, ghép phần còn lại từ deck Review 2 cũ, rồi cập nhật số W10–W11.

| Phần | Slide | Nội dung |
|---|---|---|
| Mở đầu | 1–2 | Bìa · nội dung |
| **A · Review 1 — Phạm vi bài toán** | 3–17 | Bài toán, thách thức tiếng Việt V1–V4, khoảng trống, **trong / ngoài phạm vi** (mới), RQ, giả thuyết, dữ liệu, baseline Qwen2.5-VL-3B và lý do chỉ dùng QLoRA, phương pháp đề xuất (pipeline), token & RAG, sản phẩm, kế hoạch (thêm cột "đầu ra · trạng thái"), rủi ro, góp ý R1 |
| **B · Review 2 — Existing System** | 18–27 | Tóm tắt số chính (79,6 · +21,1 · 53,4 · +3,1), góp ý GVHD, công trình trước & SOTA, dữ liệu benchmark, **kiến trúc baseline** (slide 3 cũ của Hoàng), giao thức đánh giá, R0 (slide 4 cũ), B1 zero-shot, ablation backbone R1 |
| **B · Review 2 — Methodology** | 28–41 | Lộ trình (mới), B2, đối chứng, SOTA, so với SOTA, phân tích lỗi, B3, ablation OCR 2×2, độ vững theo seed, **B3-full**, **ablation LoRA**, chọn epoch ViVQA, **adapter dùng chung SH**, **theo loại câu hỏi** (5 slide mới từ W10–W11) |
| Kết | 42–44 | Đối chiếu giả thuyết, bước tiếp W12–W13, cảm ơn |

Vị trí baseline: slide 3 cũ (kiến trúc Qwen2.5-VL) chuyển xuống **Existing System** ngay trước giao thức đánh giá; slide 4 cũ (R0) theo sau. Lý do: Review 1 chỉ cần nói *chọn* baseline nào và vì sao (slide 11); *cấu tạo* và *kết quả* của baseline là "hệ thống hiện có" mà Methodology cải tiến lên.

Chưa xong (chuyển W12): Hoàng tập nói Phần A, đọc và sửa Phần B theo cách nói của mình; tập nói toàn bộ trước 29/10.

## Đóng tuần (22/10/2026)

| Mục | Kết quả |
|---|---|
| Kết luận chính | Bảng đánh giá đầy đủ 23 dòng (tính lại từ predictions, khớp metrics.json) · **adapter dùng chung ĐẠT** (không kém C1 và B3, δ = 1,5) · OCR giúp nhiều nhất ở loại `contact` (+8,6 EM) |
| Giả thuyết | H1, H2, H3 không đổi so với W10. H4 chuyển W12. SH là mục tuỳ chọn cũ (H5 bản R1) nay **đạt** |
| Report | `W11_Team_Report.md`; R6 thêm mục 3 (chi phí đầy đủ), 4.5 (SH), 5 (EM theo loại câu hỏi) |
| Ngân sách | ≈ 12 CU, còn ≈ 37 (ước tính) |
| Chuyển W12 | Tập nói + sửa Phần B deck (trước 29/10) · H4 kiểu POPE · error analysis theo loại câu hỏi · `docs/W02/Hypotheses.md` đánh số cũ |

---

# FILE: docs/W11/W11_Team_Report.md

# W11 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W11 — Bảng đánh giá đầy đủ + adapter dùng chung hai dataset + phân rã theo loại câu hỏi |
| Thời gian thực | 21–22/10/2026 (tuần thực 7) |
| Mốc kế tiếp | **Review 2 — thứ Năm 29/10/2026** (tuần thực 8) |
| Phần cứng | Colab Pro: NVIDIA L4 (train/eval) · Tesla T4 (OCR) |
| Commit cuối tuần | commit đóng tuần sau `c406196` (xem mục "Dấu vết commit") |

## Mục tiêu

Sau W10, mô hình chính thức đã chốt (ViVQA: C3; ViTextVQA: B3-full). W11 trả lời ba câu hỏi còn lại trước Review 2:

1. **Mọi số liệu có nhất quán không?** Gom mọi run chính của hai dataset vào một bảng duy nhất, tính lại từ `predictions.json` thay vì chép tay từ README các tuần.
2. **Một adapter có phục vụ được cả hai loại câu hỏi không?** Prototype không biết câu hỏi "thuộc dataset nào", nên cần một adapter dùng chung (SH) mà không kém adapter chuyên biệt. Đây là điều đã hứa trên slide Review 2.
3. **OCR giúp loại câu hỏi nào?** ViTextVQA không có nhãn loại câu hỏi, trong khi master yêu cầu error analysis theo loại.

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | Luật ghi trước: SH (phương án S2, δ = 1,5 EM), quy ước VQA-Acc, luật bộ phân loại câu hỏi | ✅ `324caeb` — commit **trước** khi có số |
| 2 | `src/eval/full_eval_table.py`: bảng đầy đủ (EM · VQA-Acc · ANLS · F1 · chi phí · ổn định seed · loại câu hỏi ViVQA) | ✅ 23 dòng, 23/23 khớp `metrics.json`; phát hiện **VQA-Acc ≡ EM** (0 câu có > 1 đáp án gold) |
| 3 | OCR ảnh ViVQA (notebook 19) | ✅ lần 3: 9.770 ảnh, **0 lỗi**, 0,11 s/ảnh, engine trùng cache ViTextVQA. Lần 1–2 hỏng do cài Paddle (xem Blocker đã xử lý) |
| 4 | `train_qlora.py --mix`, `shared_adapter.py`, notebook 20 | ✅ `5076b76` — selftest: phần ViTextVQA trong run trộn giữ đúng dấu vân tay OCR của B3 |
| 5 | SH: train + eval 2 dataset + kết luận theo luật | ✅ **ADAPTER DÙNG CHUNG ĐẠT** (6,07 h train L4) |
| 6 | Bộ phân loại `qt-v1` (9 loại) + 150 nhãn tay mù + bảng EM/ANLS theo loại | ✅ đóng băng `9b441f1` trước khi chạy test; accuracy 80,7% |
| 7 | Deck cộng dồn Review 1 + Review 2 (44 slide, có lời nói) | ✅ **bản nháp**; Hoàng còn đọc và sửa Phần B |
| 8 | Tập nói Review 2 | ➜ chuyển W12 (trước 29/10) |
| 9 | Đóng tuần: README, Team Report, R6 (mục 3, 4.5, 5), dọn Downloads + Drive | ✅ |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| VLM / Baseline | Bảng đầy đủ: B1 và R1b trong cùng giao thức (2); slide kiến trúc baseline chuyển sang Existing System (7) |
| Dataset / OCR | OCR ảnh ViVQA + fail-fast `ocr.py` (3); gán nhãn tay 150 câu (6) |
| PEFT / Training | `--mix` + train SH (4, 5) |
| Evaluation / Deployment | `full_eval_table.py` (2); luật + kết luận SH (1, 5); `question_types.py` + bảng theo loại (6); R6 (9). SH là adapter cho prototype |

### Definition of Done

- [x] Một bảng duy nhất cho mọi run chính, tính lại từ predictions và đối chiếu với `metrics.json`
- [x] SH so với mốc **cùng điều kiện** (C1, B3): cùng file train, cùng siêu tham số, cùng OCR; luật ghi trước, ghép cặp trên cùng câu
- [x] Điều kiện hợp lệ của SH kiểm tự động trước khi kết luận
- [x] Bộ phân loại câu hỏi đóng băng trước khi chạy test, kiểm bằng nhãn tay mù, gắn cờ loại kém tin cậy
- [x] Mọi số truy được về file trong `experiments/`; sai lệch so với luật ghi trước được ghi lại (câu stt 97)
- [x] Deck có kết quả W10 + W11
- [ ] Tập nói Review 2 → W12
- [x] Dọn Downloads + Drive

## Kết quả / số liệu

### 1. Bảng chính (nguồn: `experiments/W11_full_eval_table/`)

| Dataset · tập test | Zero-shot B1 | QLoRA không OCR | **Đề xuất** | ΔEM so với B1 |
|---|---|---|---|---|
| ViTextVQA · full test (10.028) | EM 29,12 · ANLS 47,60 | B2: 50,26 · 72,02 | **B3-full: 53,35 · 73,89** | +24,23 |
| ViVQA · test (3.001) | EM 22,09 · ANLS 28,61 | C1: 73,78 · 75,66 | **C3: 76,64 · 78,03** | +54,55 |

VQA-Acc trùng EM ở từng câu vì cả hai dataset chỉ có một đáp án gold mỗi câu; cột VQA-Acc vẫn báo kèm chú thích.

### 2. Adapter dùng chung SH (ViVQA 10.799 + ViTextVQA 8.000 câu, prompt OCR cho mọi câu, 1 epoch, seed 42)

| Dataset | Mốc → SH (EM) | ΔEM (CI95) | McNemar p | Kết luận (δ = 1,5) |
|---|---|---|---|---|
| ViVQA test (3.001) | C1 73,78 → 73,51 | −0,27 [−1,13; +0,60] | 0,60 | KHÔNG KÉM |
| ViTextVQA test2000 | B3 49,25 → 49,50 | +0,25 [−0,95; +1,45] | 0,75 | KHÔNG KÉM |

- → **ADAPTER DÙNG CHUNG ĐẠT**: prototype dùng 1 adapter (29,5 MB) thay vì 2. Bảng kết quả chính vẫn dùng C3 và B3-full.
- Dòng OCR vô nghĩa trên ảnh ViVQA (44% câu, đáp án gold trong OCR 0,0%) **không làm hại**: ΔEM +0,15 [−1,05; +1,36]; chép chữ OCR vào đáp án sai 0,30% (C1: 0,23%).
- Chi phí: train 6,07 h ≈ C1 + B3 (6,01 h); VRAM đỉnh 8,15 GB; suy luận như adapter chuyên biệt.

### 3. EM theo loại câu hỏi — ViTextVQA full test (chỉ mô tả)

| | source | contact | number_time | person | location | name ⚠ | text_content ⚠ | product_service | other ⚠ |
|---|---|---|---|---|---|---|---|---|---|
| B2 | 57,2 | 55,5 | 55,7 | 54,8 | 44,0 | 55,7 | 48,6 | 42,5 | 46,1 |
| B3-full | 58,1 | 64,1 | 58,8 | 61,2 | 47,0 | 57,6 | 51,1 | 44,1 | 49,6 |
| Δ | +0,9 | **+8,6** | +3,1 | +6,4 | +3,0 | +1,9 | +2,5 | +1,6 | +3,5 |

⚠ = precision < 80% so với nhãn tay (bộ luật không sửa sau khi thấy test). OCR giúp nhiều nhất khi phải chép đúng chuỗi (số điện thoại, website, tên người); loại khó nhất là câu phải suy ra từ chữ (`product_service`, `location`).

### 4. Đối chiếu giả thuyết

| Mục | Trạng thái sau W11 | Lý do |
|---|---|---|
| H1 · QLoRA ≥ +5 EM | Ủng hộ (không đổi) | B1 → B2 +21,1 EM (ViTextVQA), +51,7 (ViVQA, C1) |
| H2 · OCR ≥ +3 ANLS, cộng hưởng với QLoRA | Ủng hộ một phần (không đổi) | Bảng theo loại cho thấy lợi ích tập trung ở loại chép chuỗi |
| H3 · prompt định dạng tiếng Việt | Không ủng hộ (không đổi) | |
| H4 · QLoRA không tăng hallucination | Chờ W12 | Đã có đầu vào: tỉ lệ chép OCR nhiễu 0,30% |
| Adapter dùng chung (mục tuỳ chọn) | **Đạt** | Không kém cả hai mốc |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|---|---|---|
| B1 | Deck Phần B là bản nháp Claude soạn, chưa theo cách nói của Hoàng; chưa tập nói | **Cao** (review 29/10) | W12, trước 29/10: Hoàng nắm Phần A → đọc Phần B, ghi yêu cầu sửa → tập nói toàn bộ, bấm giờ |
| B2 | H4 (hallucination) chưa đo | Trung bình | W12: kiểu POPE trên B1 / B2 / B3-full |
| B3 | `name`, `text_content`, `other` của `qt-v1` kém tin cậy (precision 45–70%) | Thấp | Chỉ dùng mô tả, ghi giới hạn; không sửa luật sau khi thấy test |
| B4 | SH thấp hơn mô hình chính thức 3–4 EM (bản rẻ) | Thấp | SH-full ≈ 19 h L4 ≈ 29 CU — chưa quyết, cân nhắc sau Review 2 |
| B5 | `docs/W02/Hypotheses.md` còn đánh số cũ | Thấp | Cập nhật khi viết R4 Methodology |
| B6 | Ngân sách Colab | Trung bình | Còn ≈ 37 CU (ước tính); đọc số thật đầu W12 |

**Đã xử lý trong tuần:** notebook 19 lần 1–2 tải wheel Paddle GPU bị timeout nên chuyển CPU; Paddle 3.3 CPU + oneDNN lỗi với mọi ảnh. Mục kiểm chặn được trước khi run nào dùng cache lỗi. Sửa: `ocr.py` dừng ngay nếu 20 ảnh đầu đều lỗi, tắt oneDNN trên CPU; notebook bắt buộc 0 lỗi ở bước thử, tải lại Paddle GPU 3 lần, ghim `paddlepaddle-gpu==3.3.1`.

## Kế hoạch tuần sau (W12 · Hallucination + error analysis + Review 2)

1. **Review 2 (ưu tiên số 1, trước 29/10):** Hoàng nắm Phần A, đọc Phần B và ghi yêu cầu sửa; Claude sửa; tập nói toàn bộ, bấm giờ; chuẩn bị câu hỏi phản biện.
2. **H4 — hallucination kiểu POPE tiếng Việt** trên B1 / B2 / B3-full: luật ghi trước, sinh câu hỏi có/không về vật thể có / không có trong ảnh.
3. **Error analysis B3-full theo loại câu hỏi và kiểu lỗi** (đọc chữ sai, sai vùng, sai dấu thanh, gold nhiễu), lấy mẫu theo loại; ví dụ định tính cho R6 mục 7.
4. Cập nhật `docs/W02/Hypotheses.md` theo đánh số mới.

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `324caeb` | Luật ghi trước (README W11), `full_eval_table.py` + kết quả, notebook 19 |
| `5076b76` | `train_qlora.py --mix`, `shared_adapter.py`, notebook 20 |
| `401a09d` | Sửa OCR lần 1: fail-fast, tắt oneDNN trên CPU, cách ly cache lỗi |
| `72286c4` | Notebook 19: tải lại Paddle GPU, ghim 3.3.1 |
| `9b441f1` | Đóng băng `qt-v1` + phiếu 150 câu + hướng dẫn gán nhãn |
| `14783da` | 150 nhãn tay, kiểm nhãn, bảng theo loại |
| `fadb7b5` | Run SH (train + 2 eval), kết luận SH, biên bản OCR ViVQA |
| `c406196` | README phần SH, bảng đầy đủ 23 dòng |
| *(commit đóng tuần)* | Deck `ViVQA-VLM_Review1-2_KLTN.pptx` + script, builder cũ vào `archive_W10_builder/`, README + Team Report W11, R6 mục 3 / 4.5 / 5 |

## Link experiment log

- Bảng đầy đủ: `experiments/W11_full_eval_table/`
- OCR ViVQA: `experiments/W11_ocr_cache_vivqa/`
- SH: `experiments/W11_SH_qlora_*` (train), `experiments/W11_SH_eval_*` (2 eval), `experiments/W11_shared_adapter_L4_seed42/`
- Loại câu hỏi: `experiments/W11_question_types/`, `docs/W11/qt_labels_150.csv`
- Adapter (không lên git, trên Drive): `adapters/W11_SH_*`

---

# FILE: docs/W12/README.md

# W12 · Hallucination + Error analysis

> Tuần ảo W12 (bắt đầu 23/10/2026, tuần thực 7). Mốc master: "W12 · Analysis — ablation + error + hallucination".
> Review 2 (29/10) làm ở phiên chat riêng, **không** thuộc phạm vi README này.
> Colab đầu tuần (23/10 10:15, đọc ở Resources): **34,96 CU** (W11 ước tính ≈ 37 → lệch −2).

## Việc trong tuần

| # | Việc | Ước tính | Trạng thái |
|---|---|---|---|
| 1 | Khởi động: README W12 + luật H4 ghi trước + từ điển `configs/pope_vi_objects.json` | 0,5 h | ✅ |
| 2 | Code `src/eval/pope_vi.py` (sinh câu hỏi, chấm, phán quyết) + selftest + notebook 21; commit luật TRƯỚC khi chạy | 1,5 h | ✅ |
| 3 | Colab L4: chạy POPE-vi 5 cấu hình | ≈ 4,5 h GPU (≈ 7 CU) | ✅ (23/10, ≈ 3 h 20 phút L4) |
| 4 | Error analysis C3 (ViVQA, 4 loại có sẵn) + B3-full (ViTextVQA, `qt-v1`), luật phân loại lỗi chốt trước, không cần GPU | 2,5 h | ✅ (4a tự động · 4b audit tay SV4, 24/10) |
| 5 | Kết luận H4 theo luật + ghi README | 1 h | ✅ |
| 6 | Ví dụ định tính 6–8 failure case | 1,5 h | ✅ (12 ví dụ, R6 §7) |
| 7 | R6 §5–§7 bản tiếng Việt | 2 h | ✅ (bản nháp; §5.6 chờ audit) |
| 8 | Sửa đánh số cũ trong `docs/W02/Hypotheses.md` (H8 cũ = H4 mới) | 0,3 h | ✅ (bảng đối chiếu, không đổi số cũ) |
| 9 | Đóng tuần: Team Report, dọn Downloads + Drive, commit + push | 1 h | ✅ 24/10 |

## Quyết định

| Ngày | Quyết định | Lý do |
|---|---|---|
| 23/10/2026 | **H4 đo theo kiểu POPE (Li et al., EMNLP 2023) bằng tiếng Việt: POPE-vi** trên ảnh test ViVQA (ảnh COCO 2014), nhãn vật thể lấy từ annotation COCO `instances_train2014/val2014` | ViVQA là ảnh COCO nên có nhãn vật thể thật, dựng được đúng 3 cách chọn câu "không" của POPE. ViTextVQA không có nhãn vật thể |
| 23/10/2026 | **5 cấu hình**: B1-plain, B1-ocr, C3, B3-full, SH | C3 và B3-full là hai adapter chính thức, SH là adapter của prototype. B1-ocr để so B3-full/SH với zero-shot **cùng prompt**, tách hiệu ứng QLoRA khỏi hiệu ứng dòng OCR |
| 23/10/2026 | **Quy mô**: 500 ảnh × (3 câu "có" dùng chung + 3 câu "không" cho mỗi cách random/popular/adversarial) = **6.000 câu / cấu hình** | Đúng khuôn POPE (500 ảnh, 50% có / 50% không mỗi cách chọn). Câu "có" dùng chung cho 3 cách nên chênh giữa 3 cách chỉ do câu "không"; tiết kiệm 1/3 GPU so với 9.000 câu. Cặp (ảnh, vật) trùng giữa các cách chọn chỉ hỏi **1 lần** (greedy ⇒ cùng prompt cùng đáp án) nên số câu chạy thật < 6.000; mỗi cách chọn vẫn chấm đủ 3.000 lượt (bổ sung 23/10, trước khi chạy) |
| 23/10/2026 | Ảnh chỉ lấy từ ViVQA test **không** xuất hiện trong ViVQA train | ViVQA train và test có ảnh dùng chung; C3/SH đã thấy ảnh train lúc huấn luyện |
| 23/10/2026 | ~~Không train thêm run nào từ W12 trở đi~~ → **bỏ** (cùng ngày): CU được cộng thêm khi gia hạn Google AI Pro (02/11); run mới được phép nếu có lý do khoa học | Ngân sách không còn là ràng buộc; hiệu quả (giờ GPU, VRAM) vẫn báo đầy đủ vì là luận điểm của đề tài |

## H4 — luật ghi trước (chốt trước khi chạy; commit trước notebook 21)

**Giả thuyết H4** (H8 cũ ở W02): QLoRA không làm tăng hallucination vật thể so với zero-shot. Điều kiện bác bỏ kế thừa W02: F1 POPE-vi giảm > 2 điểm.

**Câu hỏi.**
- Ảnh: ảnh ViVQA test có annotation COCO, ≥ 3 lớp vật thể khác nhau, không nằm trong ViVQA train. Chọn 500 ảnh, seed 42.
- Câu "có": 3 lớp có trong ảnh, chọn ngẫu nhiên (seed 42).
- Câu "không" (3 lớp không có trong ảnh, loại các lớp cùng nhóm dễ nhầm trong `confusable_groups`):
  - **random**: chọn ngẫu nhiên.
  - **popular**: các lớp xuất hiện nhiều nhất trong toàn bộ ảnh của pool, lấy theo thứ tự.
  - **adversarial**: các lớp đồng xuất hiện nhiều nhất với các lớp có trong ảnh, lấy theo thứ tự.
- Mẫu câu: `Trong ảnh có {tên tiếng Việt} không? Trả lời "có" hoặc "không".` Tên lấy từ `configs/pope_vi_objects.json` (`pope-vi-objects-v1`).
- File câu hỏi sinh một lần, ghi sha256 vào README trước khi chạy model. Thủ tục: notebook 21 Phần A (CPU) sinh file → ⏸ Điểm dừng A → chép file vào `experiments/W12_pope_vi_questions/` + ghi sha256 + commit → Phần B (L4) kiểm sha256 trùng bản commit rồi mới chạy model.
- sha256 file câu hỏi `pope_vi_questions_v1.json`: `53d04eb23ad41ecee372d6dee01201578de75d992d33dd2c5b94d4aa328928fd` (sinh 23/10/2026 trên Colab CPU, commit trước khi chạy model)

**Chạy.** Qwen2.5-VL-3B-Instruct 4-bit NF4, bf16, L4, greedy, `max_new_tokens` 32 (giữ đúng config chuẩn `configs/qwen_lora.yaml`, không thêm biến; sửa 23/10 trước khi chạy), seed 42. Chạy bằng `src.eval.run_baseline --dataset vivqa --data_file <file câu hỏi>` (theo lô, resume, chữ ký run). B1-plain và C3: prompt thường. B1-ocr, B3-full, SH: prompt `ocr` (VietOCR · v1 · 600 · 0, cache W11 `experiments/W11_ocr_cache_vivqa`). Adapter kiểm bằng `adapter_fingerprint` (khớp W10/W11).

**Đọc câu trả lời** (NFC, chữ thường, bỏ dấu câu; sửa 23/10 trước khi chạy): từ **đầu tiên** là "có"/"yes" → **có**; là "không"/"no" → **không**; nếu từ đầu không phải hai loại đó mà câu chứa "không có"/"không thấy" → **không**; còn lại → **không đọc được**. Câu không đọc được tính là **sai** theo cách nghiêm: ở câu "có" là FN, ở câu "không" là FP (không thể giúp adapter trông tốt hơn).

**Metric** (dương = "có"), cho từng cách chọn: Accuracy, Precision, Recall, **F1 (chính)**, yes-ratio, tỉ lệ không đọc được.

**So sánh** (ghép cặp cùng câu hỏi):
- C3 so với B1-plain.
- B3-full và SH so với B1-ocr.
- ΔF1 = F1(adapter) − F1(mốc). CI95 bằng bootstrap theo ảnh (mỗi ảnh lấy cả 6 câu của nó), 10.000 lần, seed 42.

**Phán quyết cho từng adapter:**
- **KHÔNG TĂNG HALLUCINATION**: cận dưới CI95 của ΔF1 > −2,0 ở **cả 3** cách chọn.
- **TĂNG HALLUCINATION**: cận trên CI95 < −2,0 ở **ít nhất 1** cách chọn.
- Còn lại: **CHƯA KẾT LUẬN**.

**H4 tổng:** **ỦNG HỘ** nếu C3 và B3-full đều KHÔNG TĂNG; **BÁC BỎ** nếu một trong hai TĂNG; còn lại **MỘT PHẦN**. SH chấm cùng luật, báo riêng như kiểm tra an toàn cho prototype (không tính vào H4).

**Kiểm hợp lệ** (kiểm trước khi đọc kết quả):
1. Mỗi cấu hình đủ 6.000 dự đoán, đúng sha256 file câu hỏi.
2. Fingerprint adapter khớp.
3. Tỉ lệ không đọc được ≤ 5% ở mỗi cấu hình. Nếu vượt, cấu hình đó không hợp lệ cho H4: báo như một phát hiện về lệch định dạng (format shift) và phán quyết của adapter đó = CHƯA KẾT LUẬN.

**Chỉ mô tả, không có luật:** B1-ocr so với B1-plain (dòng OCR có gây hallucination ở zero-shot không); yes-ratio theo adapter.

**Giới hạn đã biết:** annotation COCO không đầy đủ (vật có thật nhưng không được gán nhãn sẽ thành câu "không" sai); Qwen2.5-VL có thể đã thấy ảnh COCO khi pretrain. Cả 5 cấu hình dùng chung một backbone nên hai giới hạn này ảnh hưởng như nhau.

## Ngân sách

| Việc | GPU | CU (L4 ≈ 1,54 CU/h, T4 ≈ 1,2 CU/h) |
|---|---|---|
| POPE-vi 5 cấu hình × ≈ 5.000–5.900 câu sau khi bỏ trùng (≈ 0,57 s/câu như C3/SH trên ViVQA test) | ≈ 4,5 h L4 | ≈ 7 |
| Error analysis, ví dụ định tính | — | 0 |
| **Còn lại dự kiến cuối W12** | | **≈ 28** |

## File mới (bước 2)

| File | Nội dung | Kiểm |
|---|---|---|
| `src/eval/pope_vi.py` | `--build` sinh câu hỏi POPE-vi từ annotation COCO 2014 (ảnh ViVQA test chưa thấy lúc train, ≥ 3 lớp, 500 ảnh, seed 42, loại lớp dễ nhầm, bỏ câu trùng; file có khoá `data` nên `run_baseline --data_file` đọc thẳng) · `--score` kiểm hợp lệ (đủ câu, sha256, mode, adapter trùng run gốc, OCR cùng SH), metric theo cách chọn, bootstrap theo ảnh, phán quyết theo luật | `--selftest`: đọc câu trả lời (14 ca), build tất định, loại ảnh train, loại lớp dễ nhầm, phán quyết 3 nhánh + H4, bắt lỗi adapter / sha256 / mode / OCR, `load_vivqa` đọc được file câu hỏi |
| `notebooks/21_pope_vi_hallucination_qwen25vl.ipynb` | Phần A (CPU): tải annotation COCO về `/content` (không lên Drive), sinh file câu hỏi, ⏸ Điểm dừng A · Phần B (L4): kiểm OCR cache = biên bản W11, adapter = run gốc, chạy 5 cấu hình, chấm, tự trả máy | Cú pháp mọi cell; mỗi code cell có markdown mô tả + kỳ vọng |

## Điểm dừng A — file câu hỏi đã chốt (23/10/2026)

| Mục | Giá trị |
|---|---|
| File | `experiments/W12_pope_vi_questions/pope_vi_questions_v1.json` (bản Drive trùng byte) |
| Câu duy nhất | **4.985** (500 ảnh; mỗi cách chọn random / popular / adversarial đủ 3.000 lượt chấm; bỏ trùng tiết kiệm 1.015 câu) |
| Pool ảnh | 2.789 ảnh ViVQA test → 1.413 ảnh **không** có trong train/val → 618 ảnh có ≥ 3 lớp COCO → chọn 500 (seed 42) |
| popular top 10 | person, chair, dining table, bottle, cup, bowl, car, handbag, book, couch |
| Thời gian dự kiến Phần B | 5 × 4.985 câu × ≈ 0,57 s ≈ 4,2 h L4 ≈ 6,5 CU |

**Trùng ảnh train–test ViVQA (đối chiếu kiểm tra rò rỉ W07, `src/eval/stratify_by_overlap.py`):** W07 đã đo 1.274 / 2.789 ảnh test (46%) có trong train; tính cả val là 1.376 / 2.789 (**49%**) (khác câu hỏi, cùng ảnh). Không phải phát hiện mới; ghi cho R3/R6 và câu hỏi của thầy Trí về nhiễu dataset. Điểm ViVQA test vì vậy có phần ảnh model đã thấy lúc train. POPE-vi chỉ dùng ảnh chưa thấy nên không bị ảnh hưởng. Error analysis W12 nên tách EM ViVQA test theo nhóm ảnh đã thấy / chưa thấy (chỉ mô tả).

## Error analysis — luật ghi trước (23/10/2026, commit trước khi chạy)

**Mục tiêu:** biết lỗi của hai mô hình chính thức (B3-full, C3) đến từ đâu. **Chỉ mô tả**, không phán quyết, không dùng để chọn mô hình. Script `src/eval/error_analysis.py` (định nghĩa cố định trong code, có selftest). Không tốn GPU: chỉ đọc predictions đã có.

**Đầu vào:** ViTextVQA full test 10.028 câu (B1, B2, B3-full) · ViVQA test 3.001 câu (B1, C1, C3). Script kiểm sha256 file test trùng run gốc và thiết lập OCR của B3-full = VietOCR · v1 · 600 · 0.

**Định nghĩa:**
- Kiểu lỗi (câu sai EM): dùng nguyên `answer_format.error_bucket` (W07), thứ tự ưu tiên tone → truncated → extended → near_miss → other.
- Trạng thái OCR của câu ViTextVQA (văn bản OCR đúng phần đã vào prompt, cắt 600 ký tự): `has_answer` (đáp án là một cụm trong OCR) · `wrong_diacritics` (chỉ khớp khi bỏ dấu) · `answer_missing` (OCR có chữ nhưng không có đáp án) · `no_text`.
- Nguồn lỗi B3-full: `ocr_upstream` (OCR không đưa đáp án vào prompt) · `model_wrong_span` (OCR có đáp án, model chép sai đoạn OCR) · `model_outside_ocr` (OCR có đáp án, câu trả lời không nằm trong OCR).
- B2 → B3-full ghép cặp: fixed / broken / both_right / both_wrong, tách theo trạng thái OCR; % câu bị làm hỏng mà câu trả lời nằm trong OCR (proxy "OCR làm phân tâm").
- ViVQA: kiểu lỗi C3 × 4 loại câu hỏi; 10 cặp (gold → dự đoán) sai nhiều nhất theo loại; EM tách theo ảnh đã thấy / chưa thấy (train + val), B1 vs C3.
- Loại câu hỏi ViTextVQA: `qt-v1` (đóng băng W11), gộp loại như W11.
- Ứng viên ví dụ định tính (bước 6): chọn tất định seed 42, tối đa 5 câu / nhóm.

**Audit tay (bước 4b, sau khi có số tự động):** 100 lỗi B3-full (full test) + 50 lỗi C3, chọn ngẫu nhiên seed 42 trong các câu sai EM; gán theo đúng bộ nhãn W07 (F S O T H C N G K + "đúng ý" + "hallucination"); gán **mù** (không xem kiểu lỗi / nguồn lỗi tự động, không xem dự đoán của run khác). So với audit W07 của B2 (chỉ mô tả).

**Người gán (23/10/2026):** Văn Quang Trường (SV4 · Evaluation/Hallucination), một người gán, gán mù (chưa xem kết quả error analysis tự động). Hướng dẫn: `docs/W12/audit/HUONG_DAN_GAN_NHAN_W12.md`. Phiếu gốc: `audit/W12/W12_audit_150.xlsx` trên Drive (danh sách 150 câu đã khoá ở `docs/W12/audit/audit_manifest.json`, commit `e3e6168`). Giới hạn: một người gán nên không tính được κ trong W12; κ của bộ nhãn này đo ở W07 (Huân–Trường 0,62).

## Kết quả H4 — POPE-vi (23/10/2026)

Nguồn: `experiments/W12_pope_vi_L4_seed42/pope_vi_summary.md` (bảng đầy đủ 5 cấu hình × 3 cách chọn). 5 run: `experiments/W12_POPE_*`.

| So sánh (luật ghi trước) | ΔF1 random | ΔF1 popular | ΔF1 adversarial | Phán quyết |
|---|---|---|---|---|
| C3 vs B1-plain | — | — | — | **CHƯA KẾT LUẬN** (91,43% câu không đọc được > 5% → không hợp lệ) |
| B3-full vs B1-ocr | +5,50 [+4,22; +6,80] | +3,65 [+2,29; +5,02] | +2,47 [+1,08; +3,88] | **KHÔNG TĂNG HALLUCINATION** |
| SH vs B1-ocr (prototype) | −5,69 [−7,30; −4,06] | −9,16 [−10,73; −7,57] | −9,42 [−10,98; −7,82] | **TĂNG HALLUCINATION** |

**➜ H4 theo luật: MỘT PHẦN.** Không đổi phán quyết sau khi xem số.

**Phân tích sau (KHÔNG thuộc luật, chỉ mô tả):**
- Tỉ lệ báo sai (câu "không" bị trả lời "có", trên 1.500 câu "không" mỗi cách chọn; tính từ Precision/Recall trong bảng): B1-ocr 2,7 / 5,7 / 8,9% · **B3-full 8,7 / 16,1 / 22,4%** · SH 55,4 / 72,4 / 78,7% (random / popular / adversarial). F1 của B3-full tăng vì recall tăng (73,5 → 86,9%) nhưng precision giảm → **F1 (chọn từ W02 theo LLaVA) không bắt được việc B3-full nói "có" sai nhiều hơn**. Ghi vào Discussion như giới hạn của metric.
- C3 gần như không trả lời "có/không" (lệch định dạng sau QLoRA 2 epoch trên ViVQA — dataset không có câu có/không). Cần xem C3 trả lời gì thay thế.
- SH có thiên lệch "có" mạnh (yes-ratio 77–89%) → rủi ro cho prototype W13. Đang kiểm giả thuyết: dữ liệu có/không trong ViTextVQA train lệch về "có".
- B1-ocr vs B1-plain: dòng OCR làm F1 zero-shot giảm nhẹ (−0,6 đến −1,2 điểm), chủ yếu qua recall.

### H4 — phân tích bổ sung + quyết định (23/10/2026, KHÔNG thuộc luật)

- **C3 trả lời câu có/không bằng danh từ** ("bức ảnh" 525, "cái ghế" 434; chỉ 372 / 4.985 câu là "không") → quên kỹ năng tuân theo chỉ dẫn sau QLoRA 2 epoch trên ViVQA (không có câu có/không).
- **B3-full code-switching:** trả lời tiếng Anh "no" ở 881 / 4.985 câu (17,7%), gần như luôn đúng (802 ở câu "không").
- **Giả thuyết "SH thiên lệch 'có' do dữ liệu có/không lệch" bị bác:** đáp án "có" = 0 trong ViTextVQA train (35.159), tập con 8k (8.000) và ViVQA train (10.799); câu hỏi dạng có/không chỉ 18 / 5 / 7. Nguyên nhân thiên lệch của SH chưa rõ.
- **Quyết định (Hoàng, 23/10):** sửa SH để sang **đầu W13** (chẩn đoán rồi mới sửa, vì sửa SH là việc chuẩn bị prototype). W12 ghi giới hạn này trong R6 §6.
- Viết đầy đủ ở `docs/03_report/R6_Results.md` §6.

## Kết quả audit tay (bước 4b, 24/10/2026)

Nguồn: `experiments/W12_error_audit/audit_summary.md` (commit `3f3ce27`). Người gán: SV4, gán mù, mẫu khoá ở `e3e6168`.

| | B3-full (100 lỗi) | C3 (50 lỗi) |
|---|---|---|
| % lỗi EM mà đúng ý | 55,0 [45,2; 64,4] | 64,0 [50,1; 75,9] |
| % lỗi là hallucination | 5,0 [2,2; 11,2] | 16,0 [8,3; 28,5] |
| EM → EM theo người đọc | 53,35 → 79,01 | 76,64 → 91,59 |

- Trong 78 câu `ocr_upstream`, chỉ 43,6% là lỗi đọc chữ thật; 35,9% là đúng ý. Lỗi đọc chữ thật chiếm 36% số lỗi của B3-full (≈ 17 điểm EM), là trần thô cho việc nâng cấp OCR.
- Viết đầy đủ ở R6 §5.6.

## Tổng kết W12 (đóng tuần 24/10/2026)

| Mục tiêu | Kết quả |
|---|---|
| H4 (hallucination) | Theo luật: **MỘT PHẦN**. B3-full không tăng (ΔF1 +2,5 đến +5,5); C3 không đo được (91% câu không trả lời có/không); SH tăng (ΔF1 −5,7 đến −9,4, yes-ratio 77–89%) |
| Error analysis tự động | QLoRA chủ yếu sửa lỗi trả lời dài dòng; OCR sửa 705 / làm hỏng 395 câu (+3,09 EM); 11,5% câu có OCR sai dấu; C3 nhầm đếm ±1, màu gần nhau, đồng nghĩa |
| Error analysis tay | EM theo người đọc B3-full ≈ 79, C3 ≈ 92; hallucination 2,3% / 3,7% mọi câu trả lời |
| Report | R6 §5–§7 bản nháp tiếng Việt; bảng đối chiếu đánh số giả thuyết trong `docs/W02/Hypotheses.md` |

**Ngân sách:** đầu tuần 34,96 CU. Dùng ≈ 5 CU (POPE-vi ≈ 3 h 20 phút L4 + Phần A trên CPU). Còn ≈ 30 CU (ước tính); được cộng thêm khi gia hạn gói ngày 02/11.

**Chuyển W13 (prototype):**
1. Chẩn đoán và xử lý thiên lệch "có" của SH **trước** khi dùng cho prototype. Ứng viên: đo xác suất "có"/"không" (ngưỡng lúc suy luận), hoặc train lại có câu có/không cân bằng từ ảnh ViVQA train (template khác POPE-vi), rồi kiểm lại không-kém + POPE-vi.
2. Prototype web: upload ảnh, câu hỏi tiếng Việt, câu trả lời, confidence, văn bản OCR (master §11), chạy trên Colab L4.
3. Xử lý code-switching ("no" tiếng Anh) trong prototype; nếu có hậu xử lý thì phải báo cáo.

---

# FILE: docs/W12/W12_Team_Report.md

# W12 — Team Weekly Report

> Bản tiếng Việt (bản làm việc). Bản tiếng Anh cho thesis chuyển từ bản này sau khi nội dung chốt.

## Metadata

| Trường | Giá trị |
|--------|---------|
| Tuần | W12 — Hallucination (H4, POPE-vi) + error analysis của hai mô hình chính thức |
| Thời gian thực | 23–24/10/2026 (tuần thực 7) |
| Mốc kế tiếp | Review 2 (thứ Năm 29/10/2026, làm ở phiên riêng) · W13 = prototype web |
| Phần cứng | Colab Pro: NVIDIA L4 (POPE-vi) · CPU (sinh câu hỏi) · máy local (error analysis, audit) |
| Commit cuối tuần | commit đóng tuần sau `3f3ce27` (xem mục "Dấu vết commit") |

## Mục tiêu

Sau W11, hai mô hình chính thức đã chốt (ViVQA: C3; ViTextVQA: B3-full) và prototype dùng adapter chung SH. W12 trả lời hai câu hỏi bắt buộc của master ("W12 · Analysis"):

1. **QLoRA có làm mô hình "bịa" vật thể nhiều hơn không?** Đây là giả thuyết H4 (H8 cũ ở W02), đo kiểu POPE bằng tiếng Việt.
2. **Lỗi còn lại của mô hình chính thức đến từ đâu?** OCR không đưa được đáp án vào prompt, model chọn sai, hay lỗi thật ra đúng ý?

## Đã làm

| # | Việc | Kết quả |
|---|------|---------|
| 1 | Luật H4 ghi trước + từ điển 80 lớp COCO tiếng Việt + 9 nhóm dễ nhầm | ✅ `9209170`, commit **trước** khi sinh câu hỏi |
| 2 | `src/eval/pope_vi.py` + notebook 21 (Phần A CPU / Phần B L4) | ✅ selftest đạt trên cả Windows lẫn Colab |
| 3 | Sinh file câu hỏi POPE-vi, khoá vào git | ✅ `b09aff4`: 4.985 câu duy nhất, 500 ảnh test chưa thấy lúc train, 3.000 lượt chấm / cách chọn |
| 4 | Chạy POPE-vi 5 cấu hình (B1, B1-ocr, C3, B3-full, SH) | ✅ ≈ 3 h 20 phút L4; kết quả `a945ac5` |
| 5 | Error analysis tự động (luật trước `a523f35`, kết quả `9e15817`) | ✅ nguồn lỗi B3-full, B2 → B3-full ghép cặp, C3 theo loại, ảnh đã thấy / chưa thấy |
| 6 | Audit tay mù 150 lỗi (100 B3-full + 50 C3) | ✅ công cụ `de947a1`, mẫu khoá `e3e6168`, SV4 gán, kết quả `3f3ce27` |
| 7 | R6 §5–§7 bản nháp tiếng Việt, 12 ví dụ định tính | ✅ `2785d40` + đóng tuần |
| 8 | Bảng đối chiếu đánh số giả thuyết (`docs/W02/Hypotheses.md`) | ✅ `2785d40` |
| 9 | Đóng tuần: README, Team Report, R6 §5.6, dọn Downloads + Drive | ✅ |

### Phân loại theo mảng (gợi ý bàn giao)

| Mảng | Việc thuộc về |
|------|---------------|
| VLM / Baseline | B1 và B1-ocr làm mốc zero-shot cho POPE-vi (4) |
| Dataset / OCR | Từ điển COCO tiếng Việt + nhóm dễ nhầm (1); phát hiện 49% ảnh ViVQA test trùng train/val, khớp W07 (3); trạng thái OCR từng câu (5) |
| PEFT / Training | Ảnh hưởng của QLoRA lên hallucination và lên khả năng trả lời có/không (4) |
| Evaluation / Deployment | `pope_vi.py`, `error_analysis.py`, `audit_w12.py` (2, 5, 6); SV4 gán nhãn audit (6); R6 §5–§7 (7); rủi ro thiên lệch "có" của SH cho prototype |

### Definition of Done

- [x] Luật H4 và file câu hỏi khoá trên git **trước** khi chạy model; Phần B tự kiểm sha256 file câu hỏi trùng bản commit
- [x] Mỗi adapter so với zero-shot **cùng prompt** (C3 ↔ B1; B3-full, SH ↔ B1-ocr); adapter kiểm bằng dấu vân tay trọng số
- [x] Câu không đọc được tính sai theo cách nghiêm; cấu hình > 5% không đọc được bị loại khỏi kết luận
- [x] Phán quyết H4 giữ đúng luật; mọi phân tích sau đều ghi nhãn "không thuộc luật"
- [x] Error analysis: định nghĩa commit trước khi chạy; số tự kiểm khớp kết quả chính (705 − 395 = +3,09 EM)
- [x] Audit tay: mẫu khoá trước, gán mù, cùng bộ nhãn W07
- [x] Mọi số truy được về `experiments/W12_*`
- [x] Dọn Downloads + Drive

## Kết quả / số liệu

### 1. H4 — POPE-vi (nguồn: `experiments/W12_pope_vi_L4_seed42/`)

| Cấu hình | F1 random | F1 popular | F1 adversarial | Tỉ lệ báo sai adversarial | yes-ratio |
|---|---|---|---|---|---|
| B1 (zero-shot) | 84,60 | 83,17 | 81,19 | 10,9% | 40–43% |
| B1-ocr | 83,39 | 81,99 | 80,59 | 8,9% | 38–41% |
| C3 | — (91,43% không đọc được) | — | — | — | — |
| **B3-full** | **88,89** | **85,65** | **83,06** | 22,4% | 48–55% |
| SH | 77,70 | 72,83 | 71,17 | 78,7% | 77–89% |

- **Theo luật: H4 MỘT PHẦN.** B3-full KHÔNG TĂNG HALLUCINATION; C3 CHƯA KẾT LUẬN (lệch định dạng); SH TĂNG HALLUCINATION (không tính vào H4).
- **Phân tích sau:**
  - F1 tăng ở B3-full là nhờ recall, trong khi **tỉ lệ báo sai tăng 2,5 lần** (8,9% → 22,4% ở adversarial). Vì vậy F1 không đủ để kết luận về hallucination.
  - C3 trả lời câu có/không bằng danh từ: quên kỹ năng tuân theo chỉ dẫn.
  - B3-full chen tiếng Anh "no" ở 17,7% câu.
  - Train không có đáp án "có" nào, nên giả thuyết "SH học lệch từ dữ liệu" bị bác.

### 2. Error analysis tự động (nguồn: `experiments/W12_error_analysis/`)

- **QLoRA sửa lỗi trả lời dài dòng:** lỗi `extended` giảm từ 25,7% xuống 13,6% số lỗi (ViTextVQA) và từ 21,3% xuống 3,9% (ViVQA).
- **B3-full:** 81% lỗi nằm ở câu mà OCR không chứa nguyên văn đáp án (cận trên; xem mục 3). Khi OCR có đáp án, lỗi chủ yếu là **cắt sai ranh giới** (383/492). 11,5% câu có OCR đọc đúng chữ nhưng sai dấu.
- **OCR B2 → B3-full:** sửa 705, làm hỏng 395 câu (+3,09 EM). Tỉ lệ sửa/hỏng là 326 : 116 ở câu OCR có đáp án. OCR làm phân tâm khoảng 1,4% số câu.
- **C3:** lỗi `number` toàn bộ là đếm lệch; `color` nhầm màu gần nhau; `object` nhiều cặp đồng nghĩa. Ảnh trùng train chỉ làm phồng khoảng 2 điểm (77,66 so với 75,67).

### 3. Audit tay mù 150 lỗi (nguồn: `experiments/W12_error_audit/`)

| | B3-full | C3 |
|---|---|---|
| % lỗi EM mà đúng ý | 55,0 [45,2; 64,4] | 64,0 [50,1; 75,9] |
| EM → EM theo người đọc | 53,35 → **79,01** | 76,64 → **91,59** |
| Hallucination trên mọi câu trả lời | 2,33% | 3,74% |

Kiểu lỗi lớn nhất của B3-full là định dạng / độ dài (40%). Lỗi đọc chữ thật chiếm 36% (≈ 17 điểm EM, trần thô cho việc nâng cấp OCR). Với C3, 28% lỗi là gold sai / nhiễu.

### 4. Đối chiếu giả thuyết

| Mục | Trạng thái sau W12 | Lý do |
|---|---|---|
| H1 · QLoRA ≥ +5 EM | Ủng hộ (không đổi) | B1 → B2 +21,1 EM (ViTextVQA), +51,7 (ViVQA, C1) |
| H2 · OCR ≥ +3 ANLS, cộng hưởng với QLoRA | Ủng hộ một phần (không đổi) | Error analysis W12: OCR sửa 705 / hỏng 395 câu |
| H3 · prompt định dạng tiếng Việt | Không ủng hộ (không đổi) | |
| **H4 · QLoRA không tăng hallucination** | **Một phần** (theo luật) | B3-full không tăng theo F1; C3 không đo được; tỉ lệ báo sai tăng (phân tích sau) |
| Adapter dùng chung SH | Đạt ở W11; **rủi ro mới**: thiên lệch "có" | Xử lý ở W13 trước khi dùng cho prototype |

## Blocker / vấn đề mở

| # | Vấn đề | Mức | Hướng xử lý |
|---|---|---|---|
| B1 | SH (adapter prototype) thiên lệch "có": báo sai 55–79% | **Cao** (prototype) | Đầu W13: chẩn đoán (xác suất "có"/"không"), rồi sửa bằng ngưỡng lúc suy luận hoặc train lại có câu có/không cân bằng; kiểm lại không-kém + POPE-vi |
| B2 | F1 không bắt được việc tăng báo sai | Trung bình | Báo kèm tỉ lệ báo sai + yes-ratio; ghi vào Discussion (R7) như giới hạn metric |
| B3 | C3 quên cách trả lời có/không | Trung bình | Ghi là cái giá của fine-tune chuyên biệt; prototype dùng adapter chung nên không chặn |
| B4 | B3-full chen tiếng Anh "no" (17,7%) | Thấp | W13: xử lý trong prototype và báo cáo nếu có hậu xử lý |
| B5 | EM đánh giá thấp (EM theo người đọc 79 / 92) | Thấp | Báo cả EM theo người đọc ở R6/R7; không thay metric chính |
| B6 | Audit một người gán (không có κ W12) | Thấp | Ghi giới hạn; κ bộ nhãn đo ở W07 (0,62) |

## Kế hoạch tuần sau (W13 · Prototype web)

1. **Sửa SH trước khi dựng prototype** (B1): chẩn đoán rồi chọn cách sửa, có số kèm theo. Chạy lại POPE-vi và kiểm không-kém cho bản sửa.
2. **Prototype web hoàn chỉnh** theo master §11: upload ảnh, câu hỏi tiếng Việt, câu trả lời, confidence, văn bản OCR; 1 adapter; chạy trên Colab L4.
3. Xử lý code-switching trong prototype (B4).
4. Bản tiếng Anh R6 §5–§7 để dành W14 (viết report sau khi thí nghiệm chốt).

## Dấu vết commit

| Commit | Nội dung |
|--------|----------|
| `9209170` | Luật H4 (README W12), từ điển `configs/pope_vi_objects.json`, `src/eval/pope_vi.py`, notebook 21 |
| `b09aff4` | Điểm dừng A: khoá file câu hỏi POPE-vi v1 (sha256 `53d04eb2…`) |
| `a523f35` | Luật error analysis + `src/eval/error_analysis.py`; đính chính ghi chú trùng ảnh (W07: 46%, tính cả val: 49%); bỏ quyết định "không train thêm" |
| `9e15817` | Kết quả error analysis tự động |
| `de947a1` | Công cụ audit tay `src/eval/audit_w12.py` |
| `e3e6168` | Khoá danh sách 150 câu audit (seed 42) |
| `14a83c1` | Hướng dẫn gán nhãn, người gán = SV4 |
| `a945ac5` | Kết quả POPE-vi 5 cấu hình + README kết quả H4 |
| `2785d40` | Phân tích bổ sung H4, R6 §5–§7, đối chiếu đánh số giả thuyết |
| `3f3ce27` | Nhãn audit SV4 + kết quả audit |
| *(commit đóng tuần)* | R6 §5.6, README + Team Report W12 |

## Link experiment log

- POPE-vi: `experiments/W12_pope_vi_questions/` (file câu hỏi), `experiments/W12_POPE_*` (5 run), `experiments/W12_pope_vi_L4_seed42/` (phán quyết)
- Error analysis: `experiments/W12_error_analysis/`
- Audit tay: `experiments/W12_error_audit/`, `docs/W12/audit/` (manifest, hướng dẫn, nhãn)
- Notebook: `notebooks/21_pope_vi_hallucination_qwen25vl.ipynb`

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

# R6 — Results (bản nháp tiếng Việt)

> Bản nháp **tiếng Việt** (W10, 21/10/2026; cập nhật W11, 22/10/2026). Bản tiếng Anh viết từ bản này sau khi nội dung chốt; bản chính thức nằm trên Overleaf. Mục 2, 3, 4 dùng số đã chốt (W07–W11). Mục 5, 6, 7 viết ở W12 (23/10/2026) từ error analysis tự động, POPE-vi (H4) và ví dụ định tính; 5.6 audit tay 150 lỗi (SV4, 24/10/2026).
>
> Đánh số giả thuyết theo deck cộng dồn (`docs/04_slides/cumulative/`): H1 QLoRA · H2 OCR · H3 prompt định dạng tiếng Việt · H4 hallucination; cấu hình LoRA = ablation A1 (module), A2 (rank).
>
> Mọi số truy được về `experiments/<run>/metrics.json`, `predictions.json` và file tóm tắt ghi ở từng bảng. Từ W11, nguồn tổng hợp duy nhất là `experiments/W11_full_eval_table/` (`full_eval_table.py` tính lại mọi metric từ `predictions.json` và đối chiếu với `metrics.json`).

## 1. Experimental setup recap

| Hạng mục | Thiết lập chung (mọi run, trừ khi ghi khác) |
|---|---|
| Backbone | Qwen2.5-VL-3B-Instruct, lượng tử 4-bit NF4 (double quant), tính toán bf16 |
| PEFT | QLoRA: LoRA r 16, α 32, dropout 0,05, gắn q, k, v, o của LLM (7.372.800 tham số train = 0,196%) |
| Train | lr 2e-4, batch hiệu dụng 16 (2 × 8), warmup 3%, 1 epoch (ViVQA chính thức: 2 epoch, chọn trên val sạch), gradient checkpointing |
| Ảnh | min_pixels 200.704, max_pixels 401.408 (≤ 512 token ảnh) |
| Suy luận | greedy, max_new_tokens 32 |
| OCR (run "OCR") | PaddleOCR 3.7 dò chữ + VietOCR đọc chữ, template v1, cắt 600 ký tự, min_conf 0 — chọn trên dev300 (W08), đóng băng |
| Phần cứng | Colab Pro, NVIDIA L4 (train / eval), Tesla T4 (OCR) |
| Seed | 42 (độ ổn định: 42 / 43 / 44) |
| Tập đánh giá | ViVQA test 3.001 câu; ViTextVQA test chính thức **10.028 câu** (full test, số chính) và tập con **test2000** (seed 42, dùng cho ablation) |
| Metric | EM (chính), VQA-Acc (trùng EM ở mọi run), ANLS (ngưỡng 0,5), token-F1 (phụ, để so với paper) |
| Thống kê | McNemar exact (hai phía) trên EM; CI95 bootstrap ghép cặp 10.000 lần (seed 42) cho ΔANLS, ΔEM, tương tác |

## 2. Main results

### 2.1 ViTextVQA — full test 10.028 câu

| Mô hình | Train | OCR | EM | ANLS | F1 (phụ) |
|---|---|---|---|---|---|
| B1 · zero-shot | — | không | 29,12 | 47,60 | 53,89 |
| B2 · QLoRA | 35.159 câu | không | 50,26 | 72,02 | 71,16 |
| **B3-full · QLoRA nhận biết OCR (đề xuất)** | 35.159 câu | có | **53,35** | **73,89** | **73,73** |
| ViTextBLIP-2 (Nguyen et al., *ESWA* 2025, Table 6) | 100% | SwinTextSpotter | 25,48 | — | 53,95 |

Nguồn: `W07_B1_zeroshot_…_testfull_…`, `W07_B2_eval_…_testfull_…`, `W10_B3full_eval_…_testfull_…`; so sánh: `experiments/W10_B3full_vs_B2_testfull_L4_seed42/b3full_summary.json`.

- **B2 → B3-full** (cùng dữ liệu, cùng siêu tham số, chỉ khác OCR): +3,09 EM (chỉ B2 đúng 395 / chỉ B3-full đúng 705, p = 6,7 × 10⁻²¹); ΔANLS +0,0187, CI95 [+0,0138; +0,0236].
- **So với ViTextBLIP-2:** B3-full EM 53,35 [52,39; 54,33] và F1 73,73 [73,02; 74,45] — cận dưới CI95 cao hơn số công bố ở cả hai. Paper không công bố cách chuẩn hoá đáp án nên so sánh tuyệt đối mang tính tham khảo; nhóm công khai hàm chuẩn hoá (`src/eval/metrics.py`).
- B1 → B2: QLoRA tăng +21,1 EM (H1 được ủng hộ, xem Review 2).

### 2.2 ViVQA — test 3.001 câu

| Mô hình | EM | ANLS | F1 (phụ) |
|---|---|---|---|
| B1 · zero-shot | 22,09 | 28,61 | 38,05 |
| B2 · QLoRA 1 epoch (W07) | 73,78 | 75,66 | 76,20 |
| **B2 · QLoRA 2 epoch (C3, chính thức)** | **76,64** | **78,03** | **78,84** |

Nguồn: `W07_B1_zeroshot_…_vivqa_test_…`, `W07_B2_eval_…_vivqa_test_…`, `W09_C3_eval_…_vivqa_test_…`; chọn epoch: `experiments/W09_vivqa_epoch_select_L4_seed42/`.

- Số epoch chọn trên **val sạch** (688 câu có ảnh không nằm trong train), test chạm một lần: C3 hơn bản 1 epoch +2,86 EM (65 / 151, p = 4,6 × 10⁻⁹).
- Kiểm rò rỉ (sau): mức tăng như nhau trên ảnh đã thấy và chưa thấy trong train (+2,9 / +2,9 EM), trên tập sạch +3,2 (p = 1,5 × 10⁻⁴) → không phải do nhớ ảnh.
- ViVQA không dùng OCR (ảnh đời thường, câu hỏi không xoay quanh chữ trong ảnh).

## 3. Efficiency

Nguồn: `experiments/W11_full_eval_table/full_eval_table.md` (mục Cost). Giờ train và VRAM lấy từ `train_log.json`, suy luận từ `metrics.json`. Mọi run: 1 GPU NVIDIA L4 (24 GB), 4-bit NF4, bf16.

| Run | Dataset | Câu train | Epoch | Giờ train | s/step | VRAM train đỉnh | Suy luận (s/câu) | Adapter |
|---|---|---|---|---|---|---|---|---|
| B1 · zero-shot | ViTextVQA / ViVQA | — | — | — | — | — | 0,876 / 0,510 | — |
| C1 · QLoRA | ViVQA | 10.799 | 1 | 3,03 | 16,17 | 6,62 GB | 0,571 | 29,53 MB |
| **C3 · QLoRA (chính thức)** | ViVQA | 10.799 | 2 | 6,07 | 16,20 | 6,62 GB | 0,571 | 29,53 MB |
| B2 · QLoRA | ViTextVQA | 35.159 | 1 | 11,99 | 19,63 | 6,88 GB | 1,001 | 29,53 MB |
| **B3-full · QLoRA + OCR (chính thức)** | ViTextVQA | 35.159 | 1 | 12,75 | 20,89 | 8,38 GB | 0,996 (+ OCR 0,11–0,85 s/ảnh, chạy một lần) | 29,53 MB |
| SH · adapter dùng chung | cả hai | 18.799 | 1 | 6,07 | 18,58 | 8,15 GB | 0,574 / 1,002 | 29,53 MB (**1 thay vì 2**) |

- Tham số train ở mọi run chính: 7.372.800 (0,196% của 3,76 tỷ tham số). Mọi run đều vừa một GPU 24 GB với VRAM đỉnh ≤ 8,4 GB, tức chạy được trên Colab Pro (L4) — đúng ràng buộc hiệu quả của đề tài.
- OCR làm VRAM train tăng ≈ 1,5 GB (prompt dài hơn) và s/step tăng ≈ 6–8% (B2 → B3-full, B2-sub → B3); suy luận gần như không đổi. Chi phí OCR (PaddleOCR + VietOCR) chạy một lần cho mỗi ảnh và cache lại: 0,72–0,85 s/ảnh trên T4 ở W08–W10, 0,11 s/ảnh trên T4 cho ảnh ViVQA (ít chữ) ở W11.

## 4. Ablation

### 4.1 OCR lúc train × OCR lúc test (test2000, seed 42, 8.000 câu train)

| EM | Test không OCR | Test có OCR |
|---|---|---|
| Train không OCR | 45,55 | 44,90 |
| Train có OCR | 45,00 | **49,25** |

Nguồn: `experiments/W09_ocr_factorial_2x2_test2000_L4_seed42/factorial_summary.json`.

- Chỉ thêm OCR lúc test: −0,65 EM (p = 0,49). Chỉ train cùng OCR: −0,55 (p = 0,35). Cả hai: +4,35 so với chỉ OCR lúc test (p = 2,2 × 10⁻⁸).
- Tương tác +4,90 EM, CI95 [+3,20; +6,60]; ANLS +3,62 [+2,25; +5,03].
- → Lợi ích đến từ việc **dạy model dùng dòng OCR** (OCR-aware fine-tuning), không phải chỉ "thêm OCR vào prompt". Không có OCR lúc test, model đã train cùng OCR lùi về mức B2-sub (EM ngang, p = 0,35; ANLS −0,010).

### 4.2 Lượng dữ liệu × OCR (full test 10.028 câu, seed 42)

| EM | Không OCR | Có OCR | Lợi ích OCR |
|---|---|---|---|
| 23% train (8.000 câu) | 46,56 | 49,10 | +2,54 (p = 1,1 × 10⁻¹⁴) |
| 100% train (35.159 câu) | 50,26 | 53,35 | +3,09 (p = 6,7 × 10⁻²¹) |

Nguồn: `experiments/W10_B3full_vs_B2_testfull_L4_seed42/b3full_summary.json`.

- Tương tác dữ liệu × OCR: +0,55 EM, CI95 [−0,20; +1,29] → lợi ích của OCR **không giảm** khi tăng dữ liệu (cộng gộp).
- Thêm dữ liệu: +3,70 EM khi không OCR, +4,25 khi có OCR.
- Với 23% dữ liệu, OCR-aware bù ≈ 69% khoảng cách EM do cắt dữ liệu; với 100% dữ liệu, OCR-aware vượt B2.

### 4.3 Độ ổn định theo seed (test2000, 8.000 câu train)

| Seed | EM B2-sub | EM B3 | ΔEM | p |
|---|---|---|---|---|
| 42 | 45,55 | 49,25 | +3,70 | 1,7 × 10⁻⁶ |
| 43 | 44,75 | 49,00 | +4,25 | 4,0 × 10⁻⁸ |
| 44 | 46,05 | 48,80 | +2,75 | 1,4 × 10⁻⁴ |
| mean ± std | 45,45 ± 0,66 | 49,02 ± 0,23 | **+3,57 ± 0,76** | |

Nguồn: `experiments/W09_B3_seed_stability_test2000_L4/seed_stability_summary.json`. ΔANLS +0,0155 ± 0,0040 (seed 44: cận dưới CI95 sát 0).

### 4.4 Cấu hình LoRA — A1 (module) và A2 (rank) (test2000, seed 42, công thức B3)

| Ablation | Lớp gắn LoRA (chỉ LLM) | r / α | Tham số train | EM | ΔEM so với B3 (CI95) | p | s/step | Adapter |
|---|---|---|---|---|---|---|---|---|
| — (B3) | q, k, v, o | 16 / 32 | 7.372.800 | 49,25 | — | — | 21,48 | 28,2 MB |
| A1 | q, k, v, o, gate, up, down | 4 / 8 | 7.483.392 | 48,70 | −0,55 [−1,60; +0,45] | 0,34 | 25,24 | 28,6 MB |
| A2 | q, k, v, o | 64 / 128 | 29.491.200 | 50,30 | +1,05 [−0,05; +2,15] | 0,071 | 20,85 | 112,5 MB |

Nguồn: `experiments/W10_lora_ablation_test2000_L4_seed42/lora_ablation_summary.json` (run `W10_A2_*` = A1, `W10_A3_*` = A2).

- **A1:** ở cùng ngân sách tham số (lệch 1,5%), gắn LoRA vào cả MLP không tốt hơn attention-only, train chậm hơn 17,5%. QLoRA paper (Dettmers et al., NeurIPS 2023) khuyên gắn mọi lớp tuyến tính nhưng cũng nêu yếu tố quan trọng nhất là tổng số tham số LoRA; ở đây tổng đó được giữ bằng nhau.
- **A2:** r 64 không khác biệt có ý nghĩa so với r 16; CI95 còn chứa mức tăng tới +2,15 nên kết luận **yếu**. r 64 tốn adapter ×4.
- LoRA được kiểm không gắn vào vision encoder (`lora_audit` trong `train_log.json`: 252 / 144 lớp, đều thuộc LLM).

### 4.5 Adapter dùng chung cho hai dataset — SH (W11)

Một adapter train trên ViVQA (10.799 câu, file của C1) + ViTextVQA (8.000 câu, file của B3), mọi câu dùng prompt OCR (ảnh không có chữ thì prompt trùng zero-shot), siêu tham số y hệt C1 / B3. Luật ghi trước: **không kém** nếu cận dưới CI95 của ΔEM > −1,5 trên **từng** dataset.

| Dataset | Mốc → SH (EM) | ΔEM (CI95) | McNemar p | ΔANLS (CI95) | Kết luận |
|---|---|---|---|---|---|
| ViVQA test (3.001) | C1 73,78 → 73,51 | −0,27 [−1,13; +0,60] | 0,60 | −0,21 [−0,99; +0,60] | không kém |
| ViTextVQA test2000 | B3 49,25 → 49,50 | +0,25 [−0,95; +1,45] | 0,75 | +0,73 [−0,33; +1,76] | không kém |

Nguồn: `experiments/W11_shared_adapter_L4_seed42/shared_adapter_summary.json`.

- → **Adapter dùng chung đạt**: một adapter (29,5 MB) phục vụ cả hai loại câu hỏi, không mất độ chính xác so với adapter chuyên biệt cùng ngân sách. Không có bằng chứng chuyển giao dương (cả hai p ≥ 0,05).
- **Nhiễu OCR trên ảnh thường:** 44% câu ViVQA có dòng OCR (chữ tiếng Anh trên ảnh COCO, VietOCR đọc thành chuỗi vô nghĩa, đáp án gold nằm trong OCR 0,0%). Trên nhóm này ΔEM = +0,15 [−1,05; +1,36]; tỉ lệ chép chữ OCR vào đáp án sai 0,30% (C1: 0,23%) → model học được cách **bỏ qua** dòng OCR không liên quan.
- SH thấp hơn mô hình chính thức (C3: −3,13 EM; B3-full: −4,35 EM trên cùng 2.000 câu) vì chỉ 1 epoch ViVQA và 23% dữ liệu ViTextVQA (bản rẻ, đã ghi trước). Bảng kết quả chính vẫn dùng C3 và B3-full; SH dùng cho prototype.

### 4.6 Kết luận cho giả thuyết H2 (OCR)

Cộng hưởng với QLoRA: **ủng hộ mạnh** (tương tác ở 4.1; vượt B2 cùng dữ liệu ở 4.2; ổn định 3 seed ở 4.3). Ngưỡng ANLS ≥ +3: **chưa đạt** (+1,55 ± 0,40 ở 8.000 câu; +1,87 ở 100% dữ liệu), nhưng trên ngưỡng bác bỏ +1 → **H2 ủng hộ một phần**. Lợi ích chủ yếu ở việc trả lời đúng nguyên chuỗi (EM) hơn là gần đúng ký tự (ANLS).

## 5. Error analysis

Mục này trả lời câu hỏi: **lỗi còn lại của hai mô hình chính thức (B3-full, C3) đến từ đâu?** Mọi số trong mục chỉ để mô tả, không dùng để chọn mô hình. Định nghĩa được ghi trong `docs/W12/README.md` (commit `a523f35`) **trước** khi chạy `src/eval/error_analysis.py`. Nguồn số: `experiments/W12_error_analysis/error_analysis_summary.md`.

### 5.1 EM theo loại câu hỏi — ViTextVQA full test (W11)

ViTextVQA không có nhãn loại câu hỏi. Nhóm dùng bộ phân loại theo luật từ khoá `qt-v1` (9 loại), viết chỉ từ 300 câu dev và đóng băng (commit `9b441f1`) trước khi chạy trên test. Kiểm trên 150 câu test2000 gán nhãn mù: accuracy 80,7%. Ba loại `name`, `text_content`, `other` có precision < 80% nên đánh dấu ⚠ (nhãn kém tin cậy).

| Run | source | contact | number_time | person | location | name ⚠ | text_content ⚠ | product_service | other ⚠ | Tất cả |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 661 | 641 | 1.041 | 420 | 969 | 1.971 | 777 | 1.144 | 2.404 | 10.028 |
| B1 | 24,5 | 47,9 | 41,2 | 40,7 | 25,3 | 34,9 | 27,4 | 16,5 | 21,5 | 29,1 |
| B2 | 57,2 | 55,5 | 55,7 | 54,8 | 44,0 | 55,7 | 48,6 | 42,5 | 46,1 | 50,3 |
| **B3-full** | **58,1** | **64,1** | **58,8** | **61,2** | **47,0** | **57,6** | **51,1** | **44,1** | **49,6** | **53,4** |
| Δ B2 → B3-full | +0,9 | **+8,6** | +3,1 | +6,4 | +3,0 | +1,9 | +2,5 | +1,6 | +3,5 | +3,1 |

Nguồn: `experiments/W11_question_types/question_types_table.md`, `qt_label_check.json`.

- OCR giúp nhiều nhất ở loại phải **chép đúng từng ký tự** (`contact`: số điện thoại, website, +8,6 EM; `person` +6,4), và ít nhất ở `source` (+0,9: watermark nhỏ ở góc ảnh).
- Sau B3-full, loại khó nhất là câu cần **suy ra** từ chữ (`product_service` 44,1; `location` 47,0), không chỉ chép lại chữ.

### 5.2 Kiểu lỗi: QLoRA sửa lỗi gì?

Mỗi câu sai EM được xếp vào một kiểu theo thứ tự ưu tiên (`answer_format.error_bucket`, W07):
- `tone`: chỉ khác dấu thanh;
- `truncated`: câu trả lời thiếu chữ so với gold;
- `extended`: câu trả lời thừa chữ, gold nằm bên trong;
- `near_miss`: lệch ≤ 2 ký tự;
- `other`: còn lại.

| Run | Tỉ lệ sai | tone | truncated | extended | near_miss | other |
|---|---|---|---|---|---|---|
| ViTextVQA · B1 | 70,9% | 3,3 | 5,0 | **25,7** | 8,0 | 57,9 |
| ViTextVQA · B2 | 49,7% | 9,0 | 13,2 | 13,6 | 16,8 | 47,4 |
| ViTextVQA · B3-full | 46,6% | 9,6 | 15,5 | 14,5 | 16,5 | 43,9 |
| ViVQA · B1 | 77,9% | 0,0 | 3,6 | **21,3** | 6,4 | 68,6 |
| ViVQA · C1 | 26,2% | 0,5 | 2,8 | 3,2 | 15,1 | 78,4 |
| ViVQA · C3 | 23,4% | 0,0 | 2,3 | 3,9 | 16,0 | 77,9 |

(% trên số câu sai của từng run.)

QLoRA chủ yếu sửa lỗi **trả lời dài dòng**. Lỗi `extended` giảm từ 25,7% xuống 13,6% số lỗi trên ViTextVQA, và từ 21,3% xuống 3,9% trên ViVQA. Model học được định dạng đáp án của dataset, khớp kết luận audit W07. Các lỗi còn lại dịch về phía đọc sai ký tự (`near_miss`, `tone`) và sai nội dung (`other`).

### 5.3 Nguồn lỗi của B3-full: OCR hay model?

Với mỗi câu ViTextVQA, nhóm xét văn bản OCR **đúng phần đã vào prompt** (VietOCR, cắt 600 ký tự) và xếp câu vào một trong 4 trạng thái:

| Trạng thái OCR | Định nghĩa | % câu (10.028) |
|---|---|---|
| `has_answer` | đáp án gold là một cụm từ trong OCR | 33,1 |
| `wrong_diacritics` | chỉ khớp khi bỏ dấu (OCR đọc đúng chữ, sai dấu) | 11,5 |
| `answer_missing` | OCR có chữ nhưng không chứa nguyên văn đáp án | 55,1 |
| `no_text` | không đọc ra chữ | 0,3 |

Trên 4.678 câu B3-full sai:

| Nguồn lỗi | % | Kiểu lỗi chiếm nhiều nhất |
|---|---|---|
| **`ocr_upstream`**: OCR không đưa nguyên văn đáp án vào prompt | **81,0** (trong đó sai dấu 14,6, thiếu đáp án 84,8, không có chữ 0,6) | other 1.816 · near_miss 682 · truncated 537 |
| `model_wrong_span`: OCR có đáp án, model chép **sai đoạn** | 10,5 | extended 193 · truncated 190 (**cắt sai ranh giới**) |
| `model_outside_ocr`: OCR có đáp án, câu trả lời nằm ngoài OCR | 8,5 | other 143 · extended 123 |

Đọc bảng này cần thận trọng:
- **81% là cận trên của lỗi do OCR.** `answer_missing` gồm cả câu cần suy luận (đáp án không viết nguyên văn trong ảnh) và câu có gold nhiễu. Audit tay ở 5.6 sẽ tách các trường hợp này.
- Chỉ 33,1% câu có đáp án nằm nguyên văn trong OCR, trong khi EM của B3-full là 53,35%. Nghĩa là model **tự đọc ảnh đúng** ở nhiều câu mà OCR không giúp được.
- Khi OCR có đáp án mà model vẫn sai, lỗi chủ yếu là **cắt sai ranh giới**: 383/492 câu lấy thiếu hoặc thừa từ.
- **Sai dấu từ OCR là thách thức riêng của tiếng Việt.** 11,5% câu có OCR đọc đúng chữ nhưng sai dấu, và 393/450 lỗi `tone` của B3-full nằm ở nhóm `ocr_upstream`.

Theo loại câu hỏi, `ocr_upstream` cao nhất ở `contact` (90,9%) và `text_content` (87,6%), thấp nhất ở `number_time` (72,5%).

### 5.4 OCR sửa được gì và làm hỏng gì (B2 → B3-full, ghép cặp)

| Trạng thái OCR | n | sửa được | làm hỏng | cùng đúng | cùng sai |
|---|---|---|---|---|---|
| tất cả | 10.028 | 705 | 395 | 4.645 | 4.283 |
| `has_answer` | 3.321 | **326** | 116 | 2.105 | 774 |
| `wrong_diacritics` | 1.149 | 90 | 68 | 506 | 485 |
| `answer_missing` | 5.526 | 289 | 211 | 2.023 | 3.003 |
| `no_text` | 32 | 0 | 0 | 11 | 21 |

- Hiệu số 705 − 395 = +310 câu (+3,09 EM), khớp đúng mức chênh B2 → B3-full ở mục 2.1.
- Lợi ích tập trung ở câu OCR có đáp án (sửa/hỏng = 326 : 116). Ở câu OCR không có đáp án thì gần cân bằng (289 : 211).
- **OCR làm phân tâm** ở 138 câu (34,9% câu bị làm hỏng, khoảng 1,4% tập test): B2 trả lời đúng, nhưng B3-full trả lời bằng một cụm khác lấy từ OCR.

### 5.5 ViVQA (C3): nhầm gì?

- `number` (444 câu, sai 23,0%): **toàn bộ** lỗi là đếm lệch, chủ yếu ±1 (bốn → ba 11, ba → bốn 10, một → hai 10).
- `color` (625 câu, sai 15,7%): nhầm **màu gần nhau** (nâu → đen 5, đỏ → cam 5). Có cả "đỏ → tía" 4 câu, đúng lỗi dịch máy "tía" mà data audit W07 đã gặp.
- `object` (1.247 câu, sai 23,3%): nhiều cặp **đồng nghĩa hoặc khác mức chi tiết** (máy tính → laptop 4, laptop → máy vi tính 3, hải âu → con chim 3, tủ lạnh → tủ đá 6). Nhiều khả năng là đúng ý; audit 5.6 sẽ xác nhận.
- `location` (685 câu, sai 30,7%, khó nhất): lệch mức chi tiết (phòng → phòng ngủ 6, nhà vệ sinh → phòng tắm 5).

**Ảnh trùng train–test.** 49% ảnh test ViVQA có trong train/val (W07 đã đo 46% khi chỉ tính train). C3 đạt EM 77,66 trên ảnh đã thấy và 75,67 trên ảnh chưa thấy. Mức thổi phồng do trùng ảnh vì vậy nhỏ (khoảng 2 điểm). Trên tập "sạch" (ảnh chưa thấy **và** cặp câu hỏi–đáp án chưa có trong train, 1.118 câu), C3 đạt 71,74 so với 76,64 trên toàn tập. Các tập con khác nhau về tỉ lệ loại câu hỏi, nên đây chỉ là số mô tả.

### 5.6 Audit tay 150 lỗi (100 B3-full + 50 C3)

**Thiết kế.** 100 lỗi EM của B3-full (ViTextVQA full test) và 50 lỗi EM của C3 (ViVQA test), chọn ngẫu nhiên (seed 42) trong các câu sai EM. Danh sách 150 câu được khoá ở commit `e3e6168` trước khi gán nhãn.
- Người gán: một người (SV4), gán **mù**. Phiếu chỉ có ảnh, câu hỏi, gold và câu trả lời của mô hình được audit. Phiếu không có kiểu lỗi hay nguồn lỗi tự động, không có dự đoán của run khác, không có văn bản OCR.
- Bộ nhãn giống audit W07.
- Nguồn số: `experiments/W12_error_audit/audit_summary.md`. Nhãn: `docs/W12/audit/labels/W12_audit_150_Truong.xlsx`.

| | B3-full (ViTextVQA, 100 lỗi) | C3 (ViVQA, 50 lỗi) | Tham chiếu B2 · W07 (ViTextVQA / ViVQA) |
|---|---|---|---|
| % lỗi EM mà thật ra **đúng ý** [CI95 Wilson] | **55,0** [45,2; 64,4] | **64,0** [50,1; 75,9] | 46 / 62 |
| % lỗi là hallucination | 5,0 [2,2; 11,2] | 16,0 [8,3; 28,5] | 9 / 5 |
| EM → **EM theo người đọc** [CI95] | 53,35 → **79,01** [74,44; 83,39] | 76,64 → **91,59** [88,34; 94,37] | |
| Hallucination trên mọi câu trả lời | 2,33% | 3,74% | |

**Kiểu lỗi (nhãn tay, % số lỗi đã duyệt):**

| Kiểu lỗi | B3-full | C3 |
|---|---|---|
| Định dạng / độ dài (ý đúng) | **40,0** [30,9; 49,8] | — |
| Đồng nghĩa / khác từ (ý đúng) | 4,0 | **36,0** [24,1; 49,9] |
| Đọc chữ sai (OCR) | 29,0 [21,0; 38,5] | — |
| Sai dấu thanh | 7,0 | — |
| Sai vùng / không có trong ảnh | 15,0 | 28,0 |
| Gold sai / nhiễu | 5,0 | **28,0** [17,5; 41,7] |
| Đếm sai | — | 6,0 |
| Khác | — | 2,0 |

**Nhãn tay đối chiếu nguồn lỗi tự động (B3-full, 100 câu; nguồn lỗi theo mục 5.3):**
- `ocr_upstream` (78 câu):
  - lỗi **đọc chữ thật** (đọc sai 27 + sai dấu 7) = 34 / 78 = **43,6%** [33,1; 54,6];
  - **đúng ý** (định dạng 26 + đồng nghĩa 2) = 28 / 78 = 35,9% [26,1; 47,0];
  - sai vùng 11; gold nhiễu 5.
- `model_wrong_span` (14 câu): 11 là định dạng / độ dài (ý đúng). Đây chính là lỗi cắt sai ranh giới, phần lớn người đọc chấp nhận.
- `model_outside_ocr` (8 câu): định dạng 3, đọc chữ sai 2, sai vùng 2, đồng nghĩa 1.

**Kết luận (chỉ mô tả):**
1. **EM đánh giá thấp mô hình rất nhiều**, nhất là trên ViTextVQA. Hơn một nửa số lỗi EM của B3-full là đúng ý, nên **EM theo người đọc ≈ 79%** (EM 53,35%). Kiểu lỗi lớn nhất là **định dạng / độ dài (40%)**: câu trả lời và gold đúng nội dung nhưng khác cách viết (ví dụ "combo 3" so với gold "3"; khoảng trắng quanh dấu chấm trong email). Đây là giới hạn của EM trên ViTextVQA và cần ghi trong Discussion.
2. **Con số 81% `ocr_upstream` ở mục 5.3 không có nghĩa 81% lỗi do OCR.** Trong nhóm này chỉ 43,6% là lỗi đọc chữ thật; 35,9% là đúng ý. Tính trên cả mẫu, lỗi đọc chữ thật (đọc sai + sai dấu) chiếm **36%** [27,3; 45,8] số lỗi của B3-full, tức khoảng **17 điểm EM**. Đây là ước lượng thô phần mà một OCR tiếng Việt tốt hơn có thể sửa được.
3. **ViVQA (C3):** 28% lỗi là gold sai / nhiễu, khớp với các câu hỏi dịch máy khó hiểu ở mục 7. Thêm 36% là đồng nghĩa (đúng ý), nên **EM theo người đọc ≈ 92%** (EM 76,64%).
4. **Hallucination** (nói thứ không có trong ảnh) chiếm 5% lỗi của B3-full và 16% lỗi của C3, tức khoảng **2,3% và 3,7% mọi câu trả lời**. So với W07 (B2: 9% / 5%) chỉ để mô tả: khác người gán, khác mô hình, và CI chồng nhau.
5. **Giới hạn:** một người gán nên W12 không có κ (κ của bộ nhãn này đo ở W07, Huân–Trường 0,62). 50 câu ViVQA cho CI rộng. "Đúng ý" là đánh giá chủ quan của người gán.

Tham chiếu W07 (audit 200 lỗi EM của **B2**): 54% "lỗi" thực ra đúng ý. Trên ViTextVQA, nhóm lỗi đọc chữ chiếm 59% (đọc chữ sai 28%, sai vùng 23%, sai dấu thanh 8%); gold sai/nhiễu 12,5% (ViVQA 19%).

## 6. Hallucination analysis (H4)

**Thiết kế:** POPE (Li et al., EMNLP 2023) làm bằng tiếng Việt, gọi là POPE-vi.
- Ảnh: 500 ảnh ViVQA test **chưa từng có trong train/val**, nhãn vật thể lấy từ annotation COCO 2014.
- Mỗi ảnh có 3 câu "có", cùng 3 câu "không" cho mỗi cách chọn vật vắng mặt: random, popular, adversarial.
- Mẫu câu: `Trong ảnh có {vật} không? Trả lời "có" hoặc "không".`
- Bỏ trùng còn 4.985 câu duy nhất.
- Luật, từ điển 80 lớp và file câu hỏi được khoá trên git **trước** khi chạy model (commit `9209170`, `b09aff4`). Nguồn: `experiments/W12_pope_vi_L4_seed42/pope_vi_summary.md`.

| Cấu hình | F1 random | F1 popular | F1 adversarial | yes-ratio | Không đọc được |
|---|---|---|---|---|---|
| B1 (zero-shot, prompt thường) | 84,60 | 83,17 | 81,19 | 40–43% | 0,00% |
| B1-ocr (zero-shot, prompt OCR) | 83,39 | 81,99 | 80,59 | 38–41% | 0,00% |
| C3 | — | — | — | — | **91,43%** ⚠ |
| **B3-full** | **88,89** | **85,65** | **83,06** | 48–55% | 0,00% |
| SH (adapter prototype) | 77,70 | 72,83 | 71,17 | **77–89%** | 0,16% |

**Phán quyết theo luật ghi trước** (ΔF1 so với zero-shot cùng prompt, CI95 bootstrap theo ảnh, biên −2 điểm):
- B3-full: +5,50 / +3,65 / +2,47 → **KHÔNG TĂNG HALLUCINATION**.
- C3: 91,43% câu trả lời không phải "có/không" (> 5%) → không hợp lệ → **CHƯA KẾT LUẬN**.
- SH: −5,69 / −9,16 / −9,42, cận trên CI95 < −2 ở cả 3 cách → **TĂNG HALLUCINATION** (kiểm tra an toàn cho prototype, không tính vào H4).
- **H4 theo luật: MỘT PHẦN.**

**Phân tích sau (không thuộc luật, chỉ mô tả):**
1. **F1 che mất việc B3-full báo sai nhiều hơn.** Tỉ lệ báo sai (câu "không" bị trả lời "có") của B1-ocr là 2,7 / 5,7 / 8,9%, của **B3-full là 8,7 / 16,1 / 22,4%** (random / popular / adversarial). F1 tăng vì recall tăng (73,5 → 86,9%), bù cho precision giảm. Nếu đo hallucination vật thể theo nghĩa hẹp ("nói có vật khi vật không có") thì B3-full bịa nhiều hơn zero-shot, khoảng 2,5 lần ở cách adversarial. Một phần là vì zero-shot vốn quá dè dặt (yes-ratio khoảng 40%, dưới mức 50% của thiết kế POPE). **Giới hạn phương pháp:** F1, metric chọn từ W02 theo cách báo cáo của LLaVA-1.5, không đủ để kết luận về hallucination. Cần báo kèm tỉ lệ báo sai và yes-ratio.
2. **C3 quên cách trả lời có/không.** ViVQA chỉ có 4 loại câu hỏi (vật, số, màu, vị trí), không có câu có/không. Sau QLoRA 2 epoch, C3 trả lời câu có/không bằng danh từ ("bức ảnh" 525 lần, "cái ghế" 434 lần); chỉ 372/4.985 câu là "không". Đây là một dạng quên kỹ năng tuân theo chỉ dẫn khi fine-tune trên tập đáp án hẹp.
3. **SH thiên lệch "có" mạnh:** trả lời "có" 3.773/4.985 câu; tỉ lệ báo sai 55–79%. Đếm trong dữ liệu train cho thấy cả ViTextVQA train (35.159 câu) lẫn ViVQA train (10.799 câu) **không có** đáp án "có", nên giả thuyết "học lệch từ dữ liệu có/không" **bị bác**. Nguyên nhân chưa rõ; sẽ chẩn đoán và xử lý ở W13 trước khi dùng SH cho prototype.
4. **Code-switching:** B3-full trả lời bằng tiếng Anh "no" ở 881/4.985 câu (17,7%), và gần như luôn đúng (802 câu ở câu "không").
5. Vật bị báo sai nhiều nhất ở mọi cấu hình là "cốc" (cup), rồi ghế, bát, chai. Có thể một phần do annotation COCO thiếu nhãn (giới hạn đã ghi trước).
6. Dòng OCR làm F1 zero-shot giảm nhẹ (−0,6 đến −1,2 điểm), chủ yếu qua recall.

Đối chiếu audit tay (mục 5.6): hallucination chiếm 5,0% [2,2; 11,2] lỗi EM của B3-full và 16,0% [8,3; 28,5] lỗi của C3, tức khoảng 2,3% và 3,7% mọi câu trả lời. Tham chiếu W07: 7,0% [4,2; 11,4] trong 200 lỗi EM của B2.

## 7. Qualitative examples

Chọn từ các ứng viên mà `error_analysis.py` rút tất định (seed 42). Ảnh sẽ chèn khi viết bản tiếng Anh; nhận định về **nội dung ảnh** cần xem ảnh xác nhận (đánh dấu †).

| # | Nhóm | question_id | Câu hỏi | Gold | B2 (không OCR) | B3-full (OCR) | Nhận xét |
|---|---|---|---|---|---|---|---|
| 1 | OCR sửa đúng (tên riêng) | test_42304 | trạm xăng này có tên là gì ? | mipec | mipeo | **mipec** | Dòng OCR giúp chép đúng từng ký tự |
| 2 | OCR sửa đúng (số) | test_13350 | bức hình được chụp vào năm nào ? | 2016 | 2017 | **2016** | Số dễ đọc nhầm khi chỉ nhìn ảnh |
| 3 | OCR sửa đúng (tên trường) | test_30959 | trường thpt này mang tên gọi là gì ? | đào duy từ | thành phố đào duy trì | **đào duy từ** | Sửa cả sai chữ lẫn thừa chữ |
| 4 | OCR làm phân tâm (sai dấu từ OCR) | test_26011 | … liên hệ với người nào ? | cô tuyết | **cô tuyết** | có tuyết | OCR đọc sai dấu ("có"), model chép theo OCR |
| 5 | Cắt sai ranh giới | test_43847 | món cơm gà hải nam làm theo nước nào ? | kiểu singapore | **kiểu singapore** | singapore | Đúng ý nhưng thiếu chữ → EM sai |
| 6 | Sai dấu thanh | test_46777 | di tích lịch sử này tên gì ? | nhà tù hòa bình | nhà từ hòa bình | nhà tu hòa bình | Cả hai model sai dấu "tù" |
| 7 | Định dạng (đúng ý) | test_26064 | máy mài góc … là sản phẩm thuộc combo nào ? | 3 | combo 3 | combo 3 | Gold chỉ ghi "3"; câu trả lời đúng ý |
| 8 | Chuỗi dài, đọc sai ký tự | test_34842 | địa chỉ email của công ty là gì ? | tnhhanphatvn@gmail . com | tnhanhphatvn@gmail . com | tnhphanhatvn@gmail . com | OCR không đọc đúng chuỗi dài; model vẫn lệch vài ký tự |
| 9 | ViVQA · câu hỏi nhiễu do dịch máy | test_1885 | đồng hồ và tranh cũ dòng những gì | tường | (B1) đồng hồ và tranh cũ dòng những bức tranh | (C3) con chim | Câu hỏi tiếng Việt khó hiểu (dịch từ COCO-QA) † |
| 10 | ViVQA · nhầm màu / gold "tía" | test_384 | màu của nho là gì | màu đỏ | (B1) Đen | (C3) màu tía | Ranh giới đỏ / tía / đen của nho mơ hồ † |
| 11 | POPE-vi · lệch định dạng | — | Trong ảnh có {vật} không? Trả lời "có" hoặc "không". | có / không | — | C3: "bức ảnh", "cái ghế" | C3 trả lời bằng danh từ (mục 6, ý 2) |
| 12 | POPE-vi · thiên lệch "có" | — | (câu "không", adversarial) | không | — | SH: "có" (78,7% câu "không") | Rủi ro cho prototype (mục 6, ý 3) |

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
