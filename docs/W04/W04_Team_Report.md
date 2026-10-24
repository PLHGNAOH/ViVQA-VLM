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
