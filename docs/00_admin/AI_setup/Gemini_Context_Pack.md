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
