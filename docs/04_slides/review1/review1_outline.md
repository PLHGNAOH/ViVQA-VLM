# Review 1 — Outline slide

> → render `.pptx` bằng Claude skill (navy/teal/amber), **KHÔNG render ở Cursor**.
> Ngày outline: 2026-09-17. Số liệu chỉ lấy từ file nguồn; ô chưa có số → ghi "W4–W5" / "TODO-human", **không bịa**.

Pack 9 mục deliverable (bắt buộc có mặt). Slide phụ (title, roles, Q&A) nằm ngoài 9 mục.

| Slide | Tiêu đề | Bullet nội dung | File nguồn |
|------:|---------|-----------------|------------|
| 0 | Title | ViVQA-VLM · Efficient Vietnamese VQA using Vision-Language Models · nhóm 4 SV · GVHD Assoc.Prof. Đặng Ngọc Minh Đức · **REVIEW 1** | `docs/master_instruction.md` (header) |
| **1** | **Bài toán + motivation** | Ảnh tiếng Việt + câu hỏi tiếng Việt → VLM → câu trả lời tiếng Việt. Khác classification/detection: cần hiểu ảnh **và** ngôn ngữ. Khó: English-centric pretraining; scene-text/OCR + dấu thanh; code-switching; hallucination. Ý nghĩa: thích nghi VLM dưới **ngân sách tham số** (không full FT) — đóng góp đo được, không phải demo app. | `docs/master_instruction.md` §1–2; `docs/W01/Research_Orientation.md` |
| **2** | **Research Gap G1–G4 + Gap×Paper** | G1 English-centric · G2 Vietnamese VQA yếu, scene-text · **G3 PEFT cho VLM tiếng Việt (trọng tâm)** · G4 Hallucination chưa đo tiếng Việt. Bảng Gap×Paper dùng nhãn **L1–L10 theo tên paper** (không dùng weekly #1–#10). Ô ⚠️REVIEW (L7, L8, L10) để trống trên slide hoặc đánh "chờ xác nhận" — không tự tick. | `docs/W02/Research_Gap.md`; `docs/02_references/Literature_Matrix.md` (ma trận đã remap) |
| **3** | **RQ1–RQ4** | RQ1 zero-shot 3 VLM trên ViVQA/ViTextVQA (EM, VQA-Acc, ANLS). **RQ2 (trọng tâm)** PEFT vs zero-shot/prompt, không full FT. RQ3 OCR-prompting × PEFT trên ViTextVQA (ANLS). RQ4 PEFT × hallucination (POPE-vi). Phụ: RQ2a–e, RQ3a, RQ4a (một dòng "ablation"). **Không** nói RQ0–RQ3 scaffold. | `docs/W02/Research_Questions.md`; `docs/W03/RQ_Hypothesis_Reconciliation.md` |
| **4** | **Hypotheses H1–H8** | H1 QLoRA ≥ +5 EM/VQA-Acc vs zero-shot (trọng tâm Review 1). H2 4-bit ≥97% 16-bit, VRAM −≥40%. H3 all-linear > attention-only. H4 rank bão hòa r≤16. H5 PEFT không tăng hallucination. H6 OCR +≥3 ANLS, cộng hưởng QLoRA. H7 format-prompt vi +≥3 EM. H8 adapter chung ≥ riêng. Nhắc: ngưỡng kế thừa weekly, chưa đo trên tiếng Việt. **Không** chiếu H9–H11 (out-of-scope). | `docs/W02/Hypotheses.md` |
| **5** | **Dataset Analysis (stats thật, seed 42)** | Run `python src/data/analyze_dataset.py`, 2026-09-14, seed **42**. **ViVQA** (HF `minhnguyent546/ViVQA`): 15,000 QA / 10,328 ảnh; train 11,999 / test 3,001; **không val trên mirror**. **ViTextVQA** (HF `nhonhoccode/ViTextVQA`): 45,187 QA / 15,086 ảnh; train 35,159 / test 10,028; **val trống trên mirror** — **không tự tách**. ViVQA answers 99.83% ≤3 từ → EM/VQA-Acc; ViTextVQA 29,202 unique answers → ANLS. Heuristic question-type: exploratory only (~30–41% Other). Licence ViVQA/ViTextVQA = TODO-human. | `docs/W02/Dataset_Analysis.md`; `experiments/W02_dataset_stats/{vivqa,vitextvqa}_stats.json` |
| **6** | **Candidate methods + baseline đề xuất** | Cải tiến ưu tiên: QLoRA NF4 all-linear (trọng tâm) · format-prompt vi · OCR-prompting · adapter chung; RAG optional; soft-prompt **loại**. **Baseline đề xuất:** Qwen2.5-VL-3B-Instruct — nhãn **"ĐỀ XUẤT, chờ thầy Đức xác nhận"**; backup Florence-2. Dòng CHỐT để trống. Không tuyên bố đã chốt. VRAM trọng số 4-bit 3B ≈ 2–2.5 GB (Candidate_Methods, chưa đo P100). | `docs/W02/Candidate_Methods.md`; `docs/W03/Baseline_Decision.md` |
| **7** | **Kế hoạch 15 tuần + milestones** | W1 orientation · W2 literature/data · **W3 Review 1** · W4–W5 baseline · W6 method · **W7 Review 2** · W8–W9 PEFT/OCR · W10–W11 experiments/eval · W12 ablation+hallucination · **W13 Faculty** · W14 prototype/thesis · **W15 Final**. Không đảo milestone. W4: env + zero-shot khởi động + pilot QLoRA 200 mẫu. | `docs/W01/Project_Plan_15_Weeks.md`; `docs/master_instruction.md` §9 |
| **8** | **Risk matrix** | ≥5 rủi ro (bảng plan): model quá lớn/OOM · GPU/Colab ngắt · QLoRA chậm · dataset/licence · baseline khó reproduce · OCR kém · train không hội tụ · demo khó deploy. Mitigation: QLoRA 4-bit, model 3B, đổi OCR, prompting backup. | `docs/W01/Project_Plan_15_Weeks.md` §6; `docs/master_instruction.md` §12 |
| **9** | **(optional) Zero-shot sanity** | Review 1 **không** bắt buộc số model. Nếu chiếu: "protocol W5 — cùng split/seed/hardware; chưa chạy". **Cấm** điền EM/VQA-Acc/ANLS bịa. Có thể 1 slide "sanity checklist" (NFC, greedy decode, log VRAM) không có số. | `docs/W02/Candidate_Methods.md` §3; `docs/templates/Experiment_Log_Template.md` |
| 10 | Team roles | SV1 VLM/Baseline · SV2 Dataset/OCR · SV3 PEFT/Training (leader) · SV4 Eval/Deploy. Individual responsibility. | `docs/master_instruction.md` §8; `docs/W03/W03_Team_Report.md` |
| 11 | Ràng buộc PEFT-only | CHỈ LoRA/QLoRA 4/8-bit. KHÔNG full FT. Cùng split, seed, hardware. Runtime: Colab / GPU trường / (đề xuất) Kaggle P100 16 GB — GPU **chưa chốt**. | `docs/master_instruction.md` §4; `docs/W03/Baseline_Decision.md` |
| 12 | Q&A / quyết định cần hội đồng | (1) Xác nhận baseline 3B? (2) GPU nào? (3) Tách val seed=42 khi mirror thiếu? (4) Ô Gap×Paper ⚠️REVIEW. | `docs/W03/Baseline_Decision.md`; `docs/W03/W03_Team_Report.md` |

## Ghi chú render

- Palette: navy / teal / amber (Claude skill). Một ý/slide; ưu tiên bảng hơn đoạn văn.
- Trích số **chỉ** từ cột "File nguồn". Dataset sizes = JSON seed 42. Paper metrics = để TODO hoặc LLaVA ⚠️ secondary (không giấu tag).
- Không dùng numbering scaffold (RQ0, scaffold H2=OCR).
