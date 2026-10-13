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
