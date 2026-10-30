# SV1 Paper Review — VLM / Baseline

> **Đề tài:** ViVQA-VLM — Efficient Vietnamese Visual Question Answering using Vision-Language Models
> **Vai trò:** SV1 — VLM / Baseline (BLIP-2, LLaVA, Florence-2, Qwen2.5-VL)
> **Tuần:** W01 · **Ngày:** 2026-09-14 · **Reviewer:** SV1
> **Yêu cầu:** ≥ 3 paper, ưu tiên: 1 paper kiến trúc VLM nền tảng (BLIP-2 / LLaVA), 1–2 paper baseline ứng viên (Florence-2, Qwen2.5-VL). Mỗi paper trả lời **đủ 10 câu hỏi** + mục **"Paper này liên quan gì đến đề tài ViVQA-VLM?"**. Cấm chỉ tóm tắt abstract.

## 0. Danh sách paper đã chọn & lý do

| # | Paper | Venue / Năm | Nhóm | Vì sao chọn |
|---|-------|-------------|------|-------------|
| P1 | BLIP-2 (Li et al.) | ICML 2023 | Kiến trúc VLM (Q-Former) | Paper nền tảng về cách nối frozen vision encoder ↔ frozen LLM; nhẹ, nhiều tài liệu, là ứng viên baseline dễ reproduce nhất trên Colab. |
| P2 | Florence-2 (Xiao et al.) | CVPR 2024 | Baseline ứng viên | VLM seq2seq siêu nhẹ (0.23B/0.77B) có OCR nội tại — quan trọng cho scene-text ViTextVQA; test khả năng chạy trên GPU yếu. |
| P3 | Qwen2.5-VL Technical Report (Bai et al.) | 2025 | Baseline ứng viên | VLM hiện đại, đa ngữ (có tiếng Việt), OCR mạnh, dynamic resolution — ứng viên baseline tiềm năng nhất cho Vietnamese VQA. |

> **Nguồn số liệu:** trích từ bảng chính của mỗi paper (bản ICML/CVPR/arXiv tương ứng) và model card HuggingFace. Số nhớ gần đúng đánh dấu "≈".

---

## P1. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

| Trường | Giá trị |
|--------|---------|
| Title | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models |
| Author(s) | Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi (Salesforce Research) |
| Year / Venue | 2023, ICML (arXiv:2301.12597) |
| Code / Weights | https://github.com/salesforce/LAVIS (tích hợp HF `transformers`: `Blip2ForConditionalGeneration`) |
| Reviewer | SV1 |

### 1. What problem does the paper solve?
Chi phí pre-training vision-language end-to-end quá lớn vì phải huấn luyện cả vision encoder lẫn LLM. BLIP-2 giảm chi phí bằng cách **tận dụng mô hình đơn thể (unimodal) đã pretrain và đóng băng chúng**, chỉ huấn luyện một module cầu nối nhẹ.

### 2. Why is the problem important?
VLM state-of-the-art (Flamingo) cần hàng tỉ tham số huấn luyện → ngoài tầm với của phần lớn nhóm nghiên cứu. Nếu tái sử dụng được image encoder và LLM đóng băng, ta có VLM mạnh với chi phí huấn luyện nhỏ — đúng tinh thần "efficiency" của ViVQA-VLM.

### 3. What is the research gap?
Các phương pháp trước hoặc huấn luyện end-to-end tốn kém, hoặc không tận dụng được LLM lớn đã pretrain. Khoảng trống: **làm sao bắc cầu giữa modality vision và một LLM đóng băng** mà không bị "modality gap" và không phải fine-tune LLM.

### 4. What is the proposed method?
Kiến trúc **Q-Former (Querying Transformer)** — một transformer nhẹ với **32 learnable query vectors**, đặt giữa frozen image encoder và frozen LLM. Pipeline: `Image → Frozen ViT → Q-Former (32 queries) → Frozen LLM → Answer`. Huấn luyện **2 giai đoạn**: (1) vision-language representation learning (ITC + ITM + image-grounded text generation) từ frozen ViT; (2) vision-to-language generative learning nối Q-Former vào frozen LLM (OPT hoặc FlanT5).

### 5. What is the key technical idea?
Query vectors học cách **trích xuất thông tin hình ảnh giàu ngữ nghĩa nhất và phù hợp với văn bản** (thông qua cross-attention với ViT features), rồi ép chúng thành một số cố định token đưa vào LLM. Q-Former đóng vai trò "bottleneck" giảm modality gap.

### 6. What dataset is used?
Pretrain trên ~129M ảnh: COCO, Visual Genome, CC3M, CC12M, SBU, và LAION-400M (dùng CapFilt lọc). Đánh giá trên VQAv2, OK-VQA, GQA, NoCaps, Flickr30k.

### 7. What metrics are used?
VQA accuracy (VQAv2, OK-VQA, GQA), CIDEr/SPICE (captioning).

### 8. What are the main results?
BLIP-2 (ViT-g + FlanT5-XXL) đạt **zero-shot VQAv2 ≈ 65.0**, vượt Flamingo80B (56.3) với **ít hơn ~54× tham số huấn luyện**. Bản nhẹ (ViT-g + OPT-2.7B) vẫn cạnh tranh và chạy được trên GPU phổ thông.

### 9. What are the limitations?
- **English-centric**: LLM (OPT/FlanT5) và dữ liệu pretrain chủ yếu tiếng Anh ⇒ yếu với tiếng Việt.
- Không cải thiện qua in-context learning như Flamingo.
- Kế thừa lỗi/hallucination của frozen LLM; **yếu scene-text** (không có OCR nội tại).
- Q-Former với 32 query có thể mất chi tiết ảnh nhiều chữ.

### 10. What can we learn/use for our project? **(BẮT BUỘC)**
BLIP-2 là **ứng viên baseline nhẹ, dễ reproduce**. Bản OPT-2.7B (~3.9B tổng) fit Colab ở 4-bit. **Q-Former và projector là điểm SV3 gắn LoRA/QLoRA** (LLM đóng băng, chỉ adapt phần ngôn ngữ + interface). Ta có thể đo zero-shot rồi so với QLoRA để trả lời RQ2.

> ### Paper này liên quan gì đến đề tài ViVQA-VLM?
> Là **baseline ứng viên #1 về độ dễ reproduce**. Chạy được trên Colab Pro/Premium (4-bit). Kiến trúc frozen-encoder + frozen-LLM khớp hoàn hảo ràng buộc PEFT-only: ta chỉ thêm LoRA vào Q-Former/LLM. Điểm yếu tiếng Việt và scene-text của BLIP-2 chính là **động lực cho cải tiến** (dữ liệu Việt + OCR-prompting) mà nhóm sẽ kiểm chứng.

---

## P2. Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks

| Trường | Giá trị |
|--------|---------|
| Title | Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks |
| Author(s) | Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, Lu Yuan (Microsoft) |
| Year / Venue | 2024, CVPR (arXiv:2311.06242) |
| Code / Weights | HF: `microsoft/Florence-2-base` (0.23B), `microsoft/Florence-2-large` (0.77B) |
| Reviewer | SV1 |

### 1. What problem does the paper solve?
Xây **một mô hình vision hợp nhất** giải nhiều tác vụ (caption, detection, grounding, OCR, segmentation) chỉ bằng **prompt văn bản**, thay vì nhiều mô hình chuyên biệt.

### 2. Why is the problem important?
Một mô hình đa nhiệm gọn nhẹ dễ triển khai, dễ tái sử dụng — đặc biệt hữu ích khi tài nguyên hạn chế. Với ViVQA-VLM, khả năng OCR + grounding nội tại có thể hỗ trợ scene-text mà không cần OCR ngoài.

### 3. What is the research gap?
Trước Florence-2, các mô hình vision hoặc chuyên biệt một tác vụ, hoặc thiếu dữ liệu annotation đa nhiệm quy mô lớn để học biểu diễn hợp nhất.

### 4. What is the proposed method?
Kiến trúc **seq2seq**: vision encoder **DaViT** → transformer encoder-decoder; mọi tác vụ được biểu diễn thành cặp (prompt, chuỗi đầu ra) — kể cả toạ độ box được token hoá. Huấn luyện đa nhiệm trên **FLD-5B** (5.4 tỉ annotation trên 126M ảnh) sinh bằng "data engine" tự động + tinh chỉnh.

### 5. What is the key technical idea?
**Unified prompt-based seq2seq**: dùng token đặc biệt cho từng tác vụ; multi-task learning trên annotation dày đặc giúp một mô hình nhỏ học biểu diễn tổng quát.

### 6. What dataset is used?
FLD-5B (nội bộ, tự sinh). Đánh giá: COCO caption/detection, RefCOCO (grounding), TextVQA, v.v.

### 7. What metrics are used?
CIDEr (caption), AP (detection), accuracy (VQA/OCR-related), theo từng tác vụ.

### 8. What are the main results?
Mô hình **rất nhỏ (0.23B/0.77B)** đạt hiệu năng zero-shot/fine-tune cạnh tranh với mô hình lớn hơn nhiều; mạnh ở detection/grounding/OCR.

### 9. What are the limitations?
- **Câu trả lời ngắn**, thiên tác vụ vision hơn là hội thoại/VQA mở.
- **Hỗ trợ tiếng Việt hạn chế** (tokenizer/dữ liệu English-centric).
- Không phải LLM lớn ⇒ reasoning ngôn ngữ yếu hơn Qwen2.5-VL/LLaVA.

### 10. What can we learn/use for our project? **(BẮT BUỘC)**
Florence-2 là **baseline "siêu nhẹ"** dùng để kiểm chứng giới hạn dưới về tài nguyên và để **so sánh OCR nội tại vs OCR ngoài (PaddleOCR/VietOCR)**. Có thể dùng module OCR/region của nó làm nguồn evidence cho OCR-enhanced prompting.

> ### Paper này liên quan gì đến đề tài ViVQA-VLM?
> Ứng viên baseline khi **tài nguyên cực hạn** (chạy được cả trên GPU 8–12 GB). OCR nội tại phù hợp ViTextVQA. Tuy nhiên **rào cản tiếng Việt** khiến nó nhiều khả năng là baseline phụ/đối chứng chứ không phải baseline chính; giúp trả lời RQ1 (zero-shot của các VLM khác nhau) và ablation backbone (A8).

---

## P3. Qwen2.5-VL Technical Report

| Trường | Giá trị |
|--------|---------|
| Title | Qwen2.5-VL Technical Report |
| Author(s) | Qwen Team, Alibaba (Bai et al.) |
| Year / Venue | 2025 (arXiv:2502.13923) |
| Code / Weights | HF: `Qwen/Qwen2.5-VL-3B-Instruct`, `-7B-Instruct`, `-72B-Instruct` |
| Reviewer | SV1 |

### 1. What problem does the paper solve?
Xây một họ VLM mã nguồn mở **mạnh, đa năng** (document/OCR, multilingual, video, grounding, agent) và **hiệu quả về token/thời gian** ở nhiều kích cỡ (3B/7B/72B).

### 2. Why is the problem important?
Vietnamese VQA cần một baseline **đa ngữ + OCR mạnh + xử lý ảnh độ phân giải cao** (nhiều chữ). Qwen2.5-VL đáp ứng cả ba, lại có bản 3B fit Colab — trực tiếp phục vụ đề tài.

### 3. What is the research gap?
Các VLM mở trước đó hoặc yếu OCR/đa ngữ, hoặc cắt ảnh về độ phân giải cố định (mất chi tiết chữ), hoặc quá lớn để tinh chỉnh trên 1 GPU.

### 4. What is the proposed method?
Vision encoder **ViT với native dynamic resolution** + **window attention** (giảm chi phí), **MLP merger** nén patch token trước khi đưa vào **Qwen2.5 LLM**; **absolute time encoding** cho video. Pipeline: `Image (dynamic res) → ViT + window attn → MLP merger → Qwen2.5 LLM → Answer`. Huấn luyện trên corpus đa phương thức quy mô lớn (document, OCR, grounding, multilingual).

### 5. What is the key technical idea?
**Dynamic resolution + token merger**: giữ chi tiết ảnh nhiều chữ mà vẫn kiểm soát số token; kết hợp LLM Qwen2.5 đa ngữ mạnh ⇒ OCR & multilingual tốt.

### 6. What dataset is used?
Corpus nội bộ khổng lồ (image-text, document, OCR, grounding, video). Đánh giá: DocVQA, ChartQA, OCRBench, InfoVQA, MMMU, MMBench, video benchmarks…

### 7. What metrics are used?
Accuracy theo benchmark (DocVQA ANLS, OCRBench, MMMU accuracy…).

### 8. What are the main results?
Qwen2.5-VL-72B cạnh tranh GPT-4o/Claude 3.5 trên nhiều benchmark document/OCR; bản **3B/7B đạt hiệu năng rất cao so với kích cỡ**, dẫn đầu nhóm mở cùng cỡ về OCR & document understanding.

### 9. What are the limitations?
- Bản 7B tốn VRAM khi full precision (cần 4-bit để fit Colab).
- Đa ngữ **chưa tối ưu riêng cho tiếng Việt** (dấu thanh, code-switching) — vẫn có thể sai/hallucinate.
- Report kỹ thuật, chi tiết pretraining không đầy đủ để tái lập tuyệt đối.

### 10. What can we learn/use for our project? **(BẮT BUỘC)**
Đây là **baseline chính được đề xuất**: bản **Qwen2.5-VL-3B-Instruct** fit Colab ở 4-bit, có OCR & tiếng Việt sẵn ⇒ nền tốt nhất để đo zero-shot rồi gắn QLoRA. SV3 gắn LoRA all-linear vào phần LLM, freeze ViT; dynamic resolution hỗ trợ trực tiếp ViTextVQA. ⚠️ Lưu ý licence: bản 3B chính thức dùng Qwen Research License (phi thương mại) — dùng cho capstone học thuật được nhưng phải ghi đúng tên giấy phép; nếu cần Apache-2.0 thì chọn bản 7B.

> ### Paper này liên quan gì đến đề tài ViVQA-VLM?
> **Ứng viên baseline #1 về chất lượng cho Vietnamese VQA.** Reproduce được trên Colab Pro/Premium (3B 4-bit ≈ 2–3 GB trọng số) hoặc GPU trường (7B). Kiến trúc ViT → merger → LLM là nơi SV3 gắn QLoRA; multilingual + OCR + dynamic resolution phục vụ cả ViVQA (general) lẫn ViTextVQA (scene-text). Nhiều khả năng là **baseline chốt ở W03–W04**.

---

## Tổng hợp cho nhóm

### So sánh 3 baseline ứng viên (để nhóm chọn ĐÚNG 1 baseline ở W03–W04)

| Tiêu chí | BLIP-2 | Florence-2 | Qwen2.5-VL |
|----------|--------|------------|------------|
| Kích cỡ (tham số) | ~3.9B (OPT-2.7B) / ~12B (FlanT5-XXL) | 0.23B (base) / 0.77B (large) | 3B / 7B / 72B |
| Kiến trúc interface | Q-Former (32 query) | seq2seq (DaViT + enc-dec) | ViT dynamic-res + MLP merger |
| Hỗ trợ tiếng Việt | Yếu (OPT/FlanT5 English) | Yếu (English-centric) | **Khá** (LLM Qwen2.5 đa ngữ) |
| Scene-text / OCR nội tại | Không | **Có** (OCR nội tại) | **Mạnh** (OCR/document) |
| VRAM inference (4-bit, ước lượng) | ~2–3 GB (OPT-2.7B) | < 1–2 GB | 3B ≈ 2–3 GB · 7B ≈ 4.5–5.5 GB |
| Có code + checkpoint HF | Có | Có | Có |
| Licence | BSD-3 (LAVIS) | MIT | 3B: Qwen Research License (phi thương mại) · 7B: Apache-2.0 · 72B: Qwen License |
| Đề xuất | Baseline phụ (dễ reproduce) | Đối chứng siêu nhẹ / OCR | **Baseline chính (khuyến nghị)** |

**Khuyến nghị SV1:** chốt **Qwen2.5-VL-3B-Instruct** làm baseline chính (đa ngữ + OCR + fit Colab); giữ **BLIP-2 (OPT-2.7B)** làm đối chứng dễ reproduce; **Florence-2** dùng cho ablation backbone (A8) và so sánh OCR nội tại.

### Hàng đóng góp vào `Literature_Matrix.md`
- [x] Đã chép 3 hàng (SV1): BLIP-2 (#1), Florence-2 (#2), Qwen2.5-VL (#3)

### Follow-up experiment idea (W04–W05)
Đo zero-shot EM/VQA-Acc (ViVQA) và ANLS (ViTextVQA) của cả 3 model trên cùng subset test để chọn baseline chính có căn cứ số liệu; ghi VRAM + thời gian infer/sample cho mỗi model (reproducibility).
