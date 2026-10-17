# ViVQA-VLM — Context Pack cho AI trợ lý (cập nhật 15/10/2026, sau khi đóng W08)

> File kiến thức dùng chung cho Gemini (và bất kỳ AI nào). Khi có handoff mới hơn, handoff thắng.
> Chỉ chứa thông tin đã chốt; mục nào chưa chắc ghi "(cần xác minh)".

## 1. Người dùng & dự án
- Hoàng (Phạm Lê Huy Hoàng), SV ĐH FPT, trưởng nhóm 4 người; vai trò chính thức SV3 PEFT/Training nhưng tự làm toàn bộ pipeline rồi chuyển giao cho nhóm.
- Đề tài: "Efficient Vietnamese Visual Question Answering using Vision-Language Models" (ViVQA-VLM), KLTN AIP491, chuyên ngành NLP.
- GVHD: PGS. Đặng Ngọc Minh Đức (thầy Đức) và thầy Phạm Minh Trí.
- Vai trò nhóm (để hiểu bối cảnh, không để phân việc): VLM/Baseline · Dataset/OCR · PEFT/Training (Hoàng) · Evaluation/Deploy.

## 2. Ràng buộc đề tài (từ Master Instruction + đăng ký đề tài)
- Reproduce ĐÚNG 1 baseline trước khi cải tiến → **Qwen2.5-VL-3B-Instruct** (GVHD xác nhận 07/10/2026).
- ≥ 1 cải tiến kiểm chứng được so với baseline, cùng điều kiện → hướng chính đã chọn: **OCR-enhanced prompting kết hợp QLoRA** (quyết định W8, 12/10).
- CHỈ PEFT (LoRA/QLoRA). Không full fine-tuning.
- Metric chính EM, VQA-Acc, ANLS (token-F1 là phụ). Bắt buộc ablation, error analysis, hallucination analysis.
- Datasets: ViVQA (general) và ViTextVQA (scene-text, dùng repo HF CHÍNH THỨC có gating `minhquan6203/ViTextVQA`; đáp án test lấy từ `ViTextVQA_test_gt.json`).
- Tiền xử lý tiếng Việt: chuẩn hoá Unicode NFC.

## 3. Lịch thực
- Tuần thực 1 = T2 07/09/2026. Tuần ảo trong chat/repo chạy trước lịch.
- Review 1 (W3) xong. **Review 2 = tuần thực 7: 19–25/10/2026.** Hội đồng Khoa = 30/11–06/12. Final = 14–20/12.
- Review 1, 2 do 2 GV phản biện (không phải GVHD). Slide tiếng Anh, nói tiếng Việt. Sau mỗi review điền checklist AI_Capstone_Review123, gửi mail GVHD + phản biện, upload Drive.
- Quy ước: mỗi review chỉ chứa nội dung của mốc đó + sửa theo góp ý review đó; kết quả sau đưa vào review sau.

## 4. Cấu hình chốt
- Qwen2.5-VL-3B-Instruct · bf16 · 4-bit NF4 + double quant · max_pixels 401.408 (≈512 visual token) · greedy, max_new_tokens 32 · seed 42 · GPU L4 (Colab Pro).
- QLoRA: r=16, α=32, target q/k/v/o, 7.372.800 tham số train (0,196%), lr 2e-4, batch hiệu dụng 16, warmup 3%, 1 epoch, checkpoint mỗi 200 step.
- OCR đóng băng (chọn trên dev300): PP-OCRv6_medium_det + VietOCR vgg_transformer · template v1 ("Văn bản đọc được trong ảnh: …") · max_chars 600 · min_conf 0. Cache `data/vitextvqa_official/ocr/paddledet_vietocr.json` (4.422 ảnh).
- Tập đánh giá cố định ViTextVQA: test2000 (2.000 câu, seed 42); dev300 để chọn cấu hình; train subset 8.000 câu / 2.670 ảnh (seed 42).

## 5. Kết quả đã có (KHÔNG làm tròn lại, không suy diễn thêm)
| Run | Tập | EM | ANLS | Ghi chú |
|---|---|---|---|---|
| B1 zero-shot | ViVQA test 3.001 | 0,2209 | 0,2861 | |
| B2 QLoRA 1 ep | ViVQA test 3.001 | 0,7378 | 0,7566 | train 3,03 h, VRAM 6,62 GB |
| B1 / B2 | ViTextVQA full 10.028 | 29,12 / 50,26 | 47,60 / 72,02 | B2 vượt ViTextBLIP-2 (EM 25,48) |
| R0 / R0b | TextVQA val 5.000 | VQA-Acc 79,60 / 75,21 | | tái lập đạt (paper 79,3) |
| R1 Qwen2-VL-7B | ViTextVQA 2.000 | 16,7 | | kém 3B do trả lời dài |
| FS5 few-shot | ViVQA / ViTextVQA | 20,03 / 27,60 | | không giúp |
| B1 / B1+OCR | test2000 | 0,2880 / 0,2935 | 0,4753 / 0,4802 | p=0,547, không khác biệt |
| B2 / B2+OCR | test2000 | 0,4970 / 0,5070 | 0,7191 / 0,7214 | p=0,24 |
| B2-sub (8k, không OCR) | test2000 | 0,4555 | 0,6848 | train 2,76 h |
| **B3 (8k, OCR train+test)** | test2000 | **0,4925** | **0,7038** | vs B2-sub: McNemar p=1,7e-6, ΔANLS +0,019 [+0,007; +0,031] → CẢI THIỆN |

- Rò rỉ ViVQA đã kiểm: trên tập clean 1.219 câu, B1→B2 0,250→0,691 → mức tăng không do rò rỉ.
- Error audit 200 lỗi của B2: đúng ý 54% [47,1; 60,8], hallucination 7% [4,2; 11,4], gold nhiễu 12,5%; ViTextVQA: lỗi OCR 28%, nhầm vùng chữ 23%, dấu thanh 8%. 42 dòng AI-assisted (bỏ đi kết luận không đổi).
- OCR dev300: PaddleOCR làm rơi dấu tiếng Việt; VietOCR giữ dấu tốt hơn (McNemar p=0,0008) → bằng chứng gap G2.

## 6. Việc đang mở (đầu W09)
1. Thêm seed (vd 43) cho B2-sub và B3 (~6 h L4/seed).
2. B3 trên full test 10.028 (~2,8 h).
3. B3 bỏ OCR lúc test (~35 phút) — kiểm phụ thuộc OCR.
4. QLoRA ViVQA 2 epoch, chọn epoch trên val (~6,5 h).
5. Sau Review 2: điền checklist, gửi mail, upload Drive.
6. Viết phần Results (bản tiếng Việt) cho OCR khi số liệu ổn định.
- Colab: L4 ≈ 1,54 CU/giờ; còn ≈133 CU ngày 14/10.

## 7. Repo & quy ước
- Local: `D:\FPT_university\termno9\ViVQA-VLM` · GitHub public `github.com/PLHGNAOH/ViVQA-VLM` · Drive `G:\My Drive\ViVQA-VLM` = `/content/drive/MyDrive/ViVQA-VLM` trên Colab.
- `docs/W0X/` (README + W0X_Team_Report.md), `docs/02_references/` (Literature_Matrix bản sống duy nhất), `docs/03_report/` (R1..R7), `docs/04_slides/`, `src/{data,baseline,adaptation,eval,prototype,models,prompting}`, `experiments/W0X_*`, `adapters/`, `configs/`, `notebooks/NN_topic_model.ipynb`.
- Quy trình: AI soạn → Cursor kiểm/chép/selftest/commit/push, ghi `hand_off_to_claude.md` → AI kiểm lại trên GitHub → Colab (mở notebook từ GitHub, Run all) → Cursor chép kết quả từ Drive vào `experiments/`.
- Bài học: paddlepaddle-gpu ghi đè NCCL của torch → phải "Disconnect and delete runtime"; VietOCR cài `--no-deps`; PowerShell không phân biệt hoa/thường tên biến; git pathspec wildcard kết thúc `/*`.

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
