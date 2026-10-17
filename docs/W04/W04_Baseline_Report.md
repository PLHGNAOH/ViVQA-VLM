# W4 — Baseline Report (ViVQA-VLM)

> **Mục tiêu tài liệu:** tổng kết những gì đã làm ở Week 4 để (1) ôn lại, (2) bàn giao cho nhóm học theo, (3) làm điểm khởi động Week 5.
> **Phạm vi W4:** hiểu kiến trúc VLM → dựng data pipeline → **baseline zero-shot đo được** trên ViTextVQA. *Chưa train* (QLoRA để sau Review 1).
> Baseline = Qwen2.5-VL-3B-Instruct. Thuật ngữ tra ở `docs/Glossary.md`.

---

## 1. Kết quả baseline (con số chính của W4)

| Metric | Giá trị |
|---|---|
| **Exact Match (EM)** | **38.3%** |
| **ANLS** | **55.9%** |
| EM↔ANLS gap | 17.6 điểm |
| Câu trả lời dài (nghi thừa chữ) | 87/300 (29%) |

**Baseline:** Qwen2.5-VL-3B-Instruct, **zero-shot** (chưa train gì thêm).
**Đọc số:** EM 38.3% *không* nghĩa là model chỉ đúng 38% — nhiều câu đúng-ý nhưng lệch format bị EM đánh trượt (xem Mục 4). Sự thật nằm giữa EM và ANLS. Còn ~61% chưa exact-match → **dư địa cải tiến rõ ràng**.

---

## 2. Evaluation Protocol (để tái lập — BẮT BUỘC ghi)

| Hạng mục | Giá trị |
|---|---|
| Dataset | ViTextVQA — **mirror `nhonhoccode/ViTextVQA`** (⚠️ chưa phải official, xem Mục 6) |
| Split | test (3.353 ảnh / 10.028 câu hỏi; tổng 16.762 ảnh trong `images.zip`) |
| Số mẫu (n) | 300 |
| Cách lấy mẫu | shuffle toàn bộ với **seed 42** rồi lấy 300 (KHÔNG lấy `[:300]` vì đầu file bị gom theo ảnh) |
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| dtype | float16 (T4 không hỗ trợ bfloat16) |
| Sinh câu trả lời | greedy (`do_sample=False`), `max_new_tokens=32` |
| max_pixels | `1024*28*28` (giới hạn visual token → tránh OOM) |
| Chuẩn hoá khi chấm | NFC, lowercase, bỏ dấu câu, gộp khoảng trắng (GIỮ dấu tiếng Việt + số) |
| EM | khớp tuyệt đối sau chuẩn hoá |
| ANLS | 1 − Levenshtein chuẩn hoá, ngưỡng τ=0.5 |
| VQA-Accuracy | ≈ EM (mỗi câu chỉ 1 đáp án chuẩn → công thức đa-annotator thu về EM) |
| Phần cứng | Google Colab, Tesla T4 15GB |
| Thời gian inference | ~1.3s/mẫu (~6.5 phút cho 300 mẫu) |
| Môi trường | transformers 5.17.0 · torch 2.11.0+cu128 · CUDA 12.8 · qwen-vl-utils 0.0.14 · pillow<12 (pin do torchvision) |

---

## 3. Bài học về mẫu (n nhỏ đánh lừa)

| n | EM | ANLS | Ghi chú |
|---|---|---|---|
| 20 (lấy `[:20]`) | 15.0% | 40.4% | **Lệch** — gom quanh vài ảnh khó |
| 300 (shuffle seed 42) | 38.3% | 55.9% | Đại diện, đáng tin hơn |

→ Kết luận: baseline phải **đủ lớn + lấy ngẫu nhiên**. n=20 cho bức tranh bi quan sai lệch.

---

## 4. Phát hiện lõi (nền cho methodology + error analysis)

**4.1. EM↔ANLS gap = model đúng-ý nhưng lệch format.** Ví dụ: đáp án `cool air`, model trả `Kẹo Cool Air` → EM=0 nhưng ANLS=0.67. 29% câu trả lời dài dòng góp phần vào gap; phần còn lại là đúng-một-phần. → Cần cả hai metric; chỉ dùng EM sẽ đánh giá thấp năng lực thật.

**4.2. Anchoring (đoán lì).** 3 câu "tiệm thuốc bán kẹo gì?" có đáp án khác nhau (cool air / doublemint / alpenliebe) nhưng model trả **"Cool Air" cả ba**. *Việc cần kiểm:* 3 câu này trỏ 3 ảnh khác nhau hay cùng 1 ảnh — nếu khác → ca anchoring điển hình.

**4.3. Đọc sai chữ số.** `32500 → 32.000`, `44500 → 44.000`. Lỗi OCR chữ số → chính là chỗ OCR-enhanced prompting có thể cứu.

**4.4. Hallucination địa danh/tên.** "nơi này là đâu?" → model chế tên nghe hợp lý mà sai (Đại sảnh Quốc hội / Thúy Cánh Farm). Cần đo hallucination riêng.

**4.5. ⚠️ ANLS bị "thổi" bởi câu dài.** Model trả cả câu trùng nhiều ký tự với đáp án → ANLS cao *không hẳn vì đúng*. → Phải siết prompt ngắn trước khi tin ANLS.

---

## 5. Failure cases định tính (3 ca — failure-case visualization)

| Ảnh | Loại | Kết quả | Gap |
|---|---|---|---|
| Xe cổ (demo) | General, ảnh sạch | ✅ Đúng, mô tả tốt | — (baseline làm được) |
| Infographic AI/ML | Scene-text, chữ in số hoá | ⚠️ Diacritics hoàn hảo, NHƯNG **hallucination**: đọc nhầm mô tả ô GenAI sang ô LLMs (lỗi spatial reading order) | G4 |
| Bìa sách (2 cuốn) | Scene-text, ảnh chụp + chữ cách điệu | ❌ **Degenerate** (sinh `!!!!` vô nghĩa); thêm anti-repeat vẫn loạn → **lỗi ở vision, không ở decode** | G2 |

**Kết luận đắt:** ca bìa sách chứng minh vấn đề nằm ở khâu *nhìn/đọc chữ*, không phải khâu sinh chữ. VRAM khi đó 14.2/15GB (gần OOM do ảnh phân giải cao). → Đây là lý do OCR-enhanced prompting đáng thử, và là bằng chứng cho gap G2.

*(TODO: lưu 3 ảnh + output vào `experiments/W04_.../failure_cases/` kèm nhãn.)*

---

## 6. Ba lever cải tiến (evidence-based) → cầu nối Week 5

Mỗi lever đều **xuất phát từ bằng chứng W4**, không phải phỏng đoán:

| # | Lever | Bằng chứng W4 | Sẽ ablate thế nào (W5+) |
|---|---|---|---|
| 1 | **Siết prompt ngắn** (prompt optimization) | 29% câu dài + gap 17.6 | prompt gốc vs "chỉ 1–3 từ" → so EM/ANLS |
| 2 | **OCR-enhanced prompting** (VietOCR/PaddleOCR) | sai số (4.3), degenerate chữ cách điệu (5) | có OCR vs không OCR |
| 3 | **Đo & giảm hallucination** | 4.2, 4.4 | định nghĩa metric hallucination; có/không ràng buộc "chỉ trả lời từ ảnh" |

Lever 1 rẻ nhất, nên thử đầu tiên (mini-ablation). Cải tiến dựa-trên-train (QLoRA) là hướng lớn của giai đoạn sau, vẫn dưới ràng buộc **PEFT-only, không full fine-tuning**.

---

## 7. Việc còn mở cho Week 5

- [ ] **Đổi mirror → official split.** `minhquan6203/ViTextVQA` là gated (cần đăng nhập HF + đồng ý điều khoản). Số hiện tại là *chỉ báo trên mirror* — số baseline **chính thức** phải trên official + verify train/val/test.
- [ ] **Mở rộng n** (→ vài trăm–full) cho official baseline.
- [ ] **Phân loại question_type** (Yes-No / counting / text-reading / reasoning) — hiện đang "unknown" — để error analysis phân rã theo loại câu hỏi.
- [ ] **Baseline ViVQA (general)** — cần ảnh COCO (nặng hơn ViTextVQA).
- [ ] **Xác nhận baseline (Qwen2.5-VL-3B) với thầy** trước Review 1.
- [ ] Thử **lever 1 (prompt ngắn)** như ablation đầu tiên.

---

## 8. Tái lập (reproducibility)

**File đã sinh** trong `experiments/W04_zeroshot_qwen25vl_seed42/`:
- `env.json` — version + GPU mỗi run
- `zeroshot_sample.json` — demo ảnh xe
- `zeroshot_vitextvqa_n20.json` — thử nhỏ (biết là lệch)
- `zeroshot_vitextvqa_n300_seed42.json` — **baseline chính**

**Chạy lại từ đầu (mỗi phiên Colab):**
1. Cell setup (mount Drive + version + GPU log + seed 42 + env.json)
2. Cell load model (Qwen2.5-VL-3B, float16, sdpa)
3. Cell B5 (zero-shot n=300, shuffle seed 42, có resume)
4. Cell B6 (chấm EM/ANLS)

**Data trên Drive:** `data/vitextvqa/` (test.json, test_flat.json, images/). `images.zip` (5.59GB) tải 1 lần từ mirror, giải nén → 16.762 ảnh.

---

*Baseline này là mốc tham chiếu. Mọi cải tiến ở Week 5+ phải so với chính con số EM 38.3% / ANLS 55.9% này, dưới điều kiện y hệt (cùng split, seed, prompt, phần cứng).*
