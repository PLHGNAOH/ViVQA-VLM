# Ôn tập W1–W4 & Chuẩn bị Bảo vệ (ViVQA-VLM)

> **Mục đích:** ôn lại toàn bộ tuần 1→4 và luyện trả lời hội đồng.
> **Cấu trúc mỗi tuần:** (1) Đã làm được gì · (2) Kiến thức học được · (3) Câu hỏi hội đồng có thể hỏi + hướng trả lời.
> Thuật ngữ tra ở `docs/Glossary.md`. Chi tiết baseline W4 ở `docs/W04/W04_Baseline_Report.md`.

**Bức tranh tổng:** đồ án theo triết lý **reproduce-first, PEFT-only, evidence-based, ablation-driven**. 4 tuần đầu đã đi trọn: định hướng → literature + dataset → Review 1 → **baseline đo được**. Chưa train (QLoRA để sau).

---

## WEEK 1 — Research Orientation + Planning

**① Đã làm:** Báo cáo định hướng (slide) gồm vai trò nhóm, literature matrix 10 paper, research gap, thống kê dataset, hypotheses, hướng PEFT, baseline đề xuất, kế hoạch 15 tuần, ma trận rủi ro. Dọn repo, chốt cấu trúc thư mục, sửa lỗi tên đề tài "ViVOA"→"ViVQA".

**② Kiến thức:**
- VQA là gì, **khác** image classification / object detection / image captioning ở đâu.
- VLM là gì; research gap là gì (G1–G4).
- Ràng buộc lõi: **PEFT-only, cấm full fine-tuning**; metric EM/VQA-Acc/ANLS; bắt buộc ablation + error + hallucination.
- Khung 15 tuần, 4 milestone: W3 (Review 1) · W7 (Review 2) · W13 (Hội đồng Khoa) · W15 (Bảo vệ cuối).

**③ Câu hỏi hội đồng:**
| Câu hỏi | Hướng trả lời |
|---|---|
| VQA khác image captioning thế nào? | Captioning *mô tả* ảnh; VQA phải *trả lời một câu hỏi cụ thể* dựa trên ảnh → cần hiểu + suy luận theo câu hỏi |
| Vì sao chọn đề tài này? Gap ở đâu? | VLM pretrain chủ yếu tiếng Anh (G1); yếu scene-text tiếng Việt/diacritics (G2); PEFT cho VLM tiếng Việt chưa được khai thác (G3); hallucination (G4) |
| Vì sao PEFT-only, không full fine-tuning? | Efficiency là ràng buộc thiết kế tường minh; chạy được trên GPU nhỏ (Colab); full FT tốn tài nguyên, mâu thuẫn với luận điểm "hiệu quả" |
| Rủi ro lớn nhất của kế hoạch? | GPU không đủ → QLoRA/quantization; baseline khó reproduce → model thay thế; OCR kém → đổi engine |

---

## WEEK 2 — Literature + Dataset

**① Đã làm:** Literature Matrix (10 paper + ma trận Gap×Paper), Dataset Analysis chạy thật seed 42 (**ViVQA 10.328 ảnh / 15.000 QA**; ViTextVQA mirror 15.086 ảnh / 45.187 QA), + Research_Gap, Research_Questions, Hypotheses, Candidate_Methods. **W2 PASS.**

**② Kiến thức:**
- Cách đọc paper có hệ thống (mỗi paper trả lời "học/dùng được gì cho đề tài").
- Benchmark dataset: ViVQA (general) vs ViTextVQA (scene-text).
- **NFC normalization** cho tiếng Việt; phân loại câu hỏi (Yes/No · counting · object · text-reading · reasoning).
- **Mirror vs official**; khái niệm split (train/val/test) + seed.

**③ Câu hỏi hội đồng:**
| Câu hỏi | Hướng trả lời |
|---|---|
| Dataset lấy từ đâu? Có official split không? | *(điểm yếu — chủ động thừa nhận)* Hiện dùng mirror HF; official gated (cần xin quyền); đang verify official split ở W5 |
| Vì sao dùng CẢ ViVQA và ViTextVQA? | ViVQA đo VQA chung; ViTextVQA đo đọc-chữ-trong-ảnh (scene-text) — đúng điểm yếu tiếng Việt |
| Contribution mới so với 4 paper PEFT lõi? | Chưa paper nào (LoRA/QLoRA/VL-Adapter/LLaVA-1.5) validate PEFT cho VLM trên ngôn ngữ có thanh điệu / scene-text phi tiếng Anh |
| Số liệu dataset có nguồn không? | Mọi con số phải có citation; đã chạy thật seed 42 và log lại |

---

## WEEK 3 — Review 1

**① Đã làm:** Dọn nợ kỹ thuật W2 (thống nhất đánh số RQ/Hypothesis, remap ma trận Gap×Paper, sửa link chết); scaffold `docs/W03/`; chuẩn bị bộ Review 1: **deck 15 slide tiếng Anh** + checklist trường + kịch bản thuyết trình (~22' trình + ~20' Q&A); trình trước **2 GV Reviewer** (không phải thầy hướng dẫn). **Phát hiện quan trọng:** ViVQA/ViTextVQA chỉ có mirror, thiếu val split → cảnh báo W4 sẽ vướng data loading (và đúng là đã vướng thật ở W4).

**② Kiến thức:**
- Quy tắc review: slide **tiếng Anh**, trình bày **nói tiếng Việt**; Review 1&2 do reviewer ngoài (không phải supervisor).
- Chuỗi logic nghiên cứu: **research gap → RQ → hypothesis → contribution**.
- Thuật ngữ VLM sâu (Vision Encoder, Vision-Language Alignment, Multimodal Reasoning); pipeline Qwen2.5-VL + QLoRA + nhánh OCR cho ViTextVQA.

**③ Câu hỏi hội đồng:**
| Câu hỏi | Hướng trả lời |
|---|---|
| RQ và hypothesis của em là gì? | Nêu RQ chính + 1 hypothesis đo được (vd "thêm OCR sẽ tăng ANLS trên ViTextVQA") |
| Contribution mới, đo bằng cách nào? | ≥1 cải tiến PEFT/prompt/OCR vượt baseline, đo bằng EM/ANLS + ablation |
| Baseline nào, vì sao? | Qwen2.5-VL-3B (xem W4) |
| Làm sao chứng minh cải tiến đến từ đâu? | **Ablation**: bật/tắt từng thành phần, so số |

---

## WEEK 4 — Reproduce Baseline *(phần kỹ thuật sâu nhất — hội đồng hỏi nhiều)*

**① Đã làm:**
- Hiểu kiến trúc Qwen VL end-to-end; **chốt baseline Qwen2.5-VL-3B** (đổi từ Qwen3-VL-4B vì tooling chín + đúng danh sách duyệt + nhẹ + hợp T4; Qwen3-VL-4B giữ làm reference).
- Dựng môi trường Colab tái lập (mount Drive, pin version, log GPU, seed 42, `env.json`).
- Tải ViTextVQA từ HF mirror (16.762 ảnh), **join COCO-style** (ảnh ↔ annotation) + flatten 10.028 record.
- Zero-shot + eval harness EM/ANLS. **Baseline (n=300, seed 42): EM 38.3% / ANLS 55.9%.**
- 3 failure case + phát hiện: EM↔ANLS gap, anchoring, đọc sai số, hallucination, degenerate.

**② Kiến thức:**
- **Forward pass VLM**: Vision Encoder → VL Merger → LLM; DeepStack; dynamic resolution (ảnh → visual token).
- **float16 vs bfloat16** trên T4; tương thích version `transformers`; xung đột Pillow↔torchvision.
- **COCO-style annotation** + join theo `image_id`.
- **EM vs ANLS vs VQA-Acc**; normalize; Levenshtein; greedy (tái lập).
- `max_pixels` ↔ visual token ↔ VRAM; workflow Colab (mount, log GPU, resume); **mẫu đại diện** (n=20 lệch vs n=300).

**③ Câu hỏi hội đồng:**
| Câu hỏi | Hướng trả lời |
|---|---|
| Giải thích kiến trúc VLM em dùng? | Vision Encoder đọc ảnh → Merger nén + căn chiều → LLM ghép visual+text token → sinh câu trả lời |
| Vì sao đổi Qwen3-VL → Qwen2.5-VL? | Tooling chín, đúng danh sách baseline, nhẹ hơn (hợp T4 free), version yêu cầu thấp → chắc chắn/tái lập; giữ Qwen3-VL làm reference |
| Baseline bao nhiêu? EM≠ANLS vì sao? | EM 38.3 / ANLS 55.9; EM đòi khớp tuyệt đối, ANLS chấp nhận gần giống → gap = model đúng-ý-nhưng-lệch-format |
| EM 38% có phải model dốt? | Không — nhiều câu đúng ý bị EM đánh trượt do thừa chữ/khác format; sự thật nằm giữa EM và ANLS |
| Vì sao ANLS hợp scene-text? | Chữ trong ảnh hay lệch hoa/thường/khoảng trắng → ANLS đo gần giống công bằng hơn EM |
| Đảm bảo tái lập thế nào? | seed 42, log version + GPU + thời gian, protocol cố định (split/prompt/max_pixels) |
| Model sai kiểu gì? Ví dụ? | 3 ca: general OK · hallucination (đọc nhầm cột infographic) · degenerate (chữ cách điệu → sinh rác) |
| Vì sao n=300, không phải full/n=20? | n=20 lệch (gom quanh vài ảnh); n=300 shuffle đại diện; full tốn thời gian, để W5 |
| `max_pixels` để làm gì? | Giới hạn visual token → tránh OOM; là biến ablation (độ phân giải ↔ ANLS ↔ VRAM) |
| Hallucination là gì, đo sao? | Trả lời trôi chảy nhưng sai so với ảnh; đo bằng phân tích tỉ lệ/loại (hướng POPE) |

---

## BẢO VỆ CUỐI — Câu hỏi tích hợp (xuyên đề tài)

| Câu hỏi | Hướng trả lời |
|---|---|
| Đóng góp mới của cả đồ án là gì? | Validate PEFT (LoRA/QLoRA) hoặc OCR/prompt cho VQA tiếng Việt scene-text — điều 4 paper lõi chưa làm — có ablation + error/hallucination analysis |
| Chứng minh "efficiency" thế nào? | PEFT-only (không full FT), chạy trên Colab T4; báo cáo thời gian train/inference + VRAM |
| Nếu cải tiến KHÔNG vượt baseline thì sao? | Vẫn là kết quả khoa học hợp lệ nếu có ablation + error analysis giải thích *vì sao*; trung thực báo cáo, không bịa |
| Điểm yếu lớn nhất của đồ án? | *(chủ động)* dùng mirror data, chưa official split; đang xử lý |
| Đạo đức AI? | Privacy ảnh upload; bias English-centric; hallucination trong ngữ cảnh nhạy cảm; truyền đạt rõ confidence + giới hạn |
| Nếu làm lại, đổi gì? | (Trả lời thật theo trải nghiệm — vd chốt official split sớm hơn) |

---

## ⚠️ Điểm yếu phải CHỦ ĐỘNG thừa nhận (đừng để hội đồng bắt trước)

1. **Data mirror, chưa official split** — đang chuyển sang official ở W5.
2. **0/10 PDF paper đọc kỹ** — nợ từ W2, cần đọc để điền số benchmark.
3. **n=300 là chỉ báo**, chưa phải baseline full/official.
4. **Chưa có cải tiến kiểm chứng** — đó là công việc W5+ (baseline mới xong).

> Mẹo bảo vệ: nêu điểm yếu *kèm kế hoạch khắc phục* → hội đồng đánh giá cao sự trung thực hơn là giấu.

---

## Bạn có đủ điều kiện sang W5 chưa?

**Có.** Gate W4 = "có baseline đo được + tái lập được": đã đạt (EM 38.3/ANLS 55.9, n=300 seed 42, protocol đầy đủ, log lưu trên Drive). Việc còn lại (official split, mở rộng n, phân loại question_type) **thuộc W5**, không chặn việc chuyển tuần.

**W5 mở ra làm gì:** đổi mirror→official split · mở rộng n · phân loại question_type cho error analysis · thử **lever prompt ngắn** như ablation đầu tiên → bước đầu biến baseline thành cải tiến kiểm chứng được.
