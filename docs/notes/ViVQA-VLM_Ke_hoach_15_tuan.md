# KẾ HOẠCH THỰC HIỆN CAPSTONE PROJECT — ViVQA-VLM

> **Efficient Vietnamese Visual Question Answering using Vision-Language Models**
> Nghiên cứu hệ thống hỏi đáp trực quan tiếng Việt hiệu quả sử dụng mô hình Vision-Language
> **Nhóm:** 4 SV (SV1 VLM/Baseline · SV2 Dataset/OCR · SV3 PEFT/Training · SV4 Evaluation/Hallucination/Deployment)
> **Milestone cố định:** W3 Review 1 · W7 Review 2 · W13 Hội đồng cấp Khoa · W15 Final Defense
> **Ràng buộc lõi:** PEFT-only (LoRA/QLoRA) — *không* full fine-tuning · reproduce đúng 1 baseline trước khi cải tiến · metric chính EM / VQA-Acc / ANLS · bắt buộc ablation + error + hallucination analysis · mọi thí nghiệm phải log để tái lập.

*Ghi chú kỹ thuật: bảng tổng quan 15 tuần là bảng rộng (6 cột) — khi export PDF nên đặt trang **landscape** cho mục 2 để không tràn cột.*

---

## 1. Nhận xét về Kế hoạch Tuần 1

**Kết luận tổng quát:** Kế hoạch Tuần 1 của nhóm **đạt chuẩn học thuật và khả thi**. Nó thể hiện đúng "research mindset" mà đề tài yêu cầu (đọc paper có hệ thống → xác định gap → quan sát thực nghiệm → đặt câu hỏi nghiên cứu), thay vì lao vào code sớm. Đặc biệt, việc đưa **pilot experiment** vào ngay Tuần 1 là quyết định tốt: nó biến "research gap" từ tuyên bố trên giấy thành quan sát có bằng chứng.

Tuy nhiên, để Tuần 1 thật sự vững và không tạo nợ kỹ thuật cho các tuần sau, cần **bổ sung/làm rõ 6 điểm** dưới đây.

### 1.1. Điểm mạnh (giữ nguyên)

| Hạng mục | Vì sao tốt |
|---|---|
| Đọc ≥ 3 paper/SV + Literature Matrix | Bao phủ ~12 paper, đủ để xác định gap; matrix ép mỗi paper trả lời *"What can we use for our project?"*. |
| Xác định Research Gap sớm | Bám đúng yêu cầu "reproduce trước, cải tiến sau" — gap là nền cho contribution. |
| Khảo sát ViVQA & ViTextVQA | Hiểu dữ liệu trước khi chọn mô hình là đúng thứ tự. |
| Pilot 10–20 mẫu | De-risk sớm, cung cấp *evidence* cho gap và cho question-type taxonomy. |
| RQ + Hypotheses | Đặt "vạch đỗ" nghiên cứu ngay từ đầu, tránh trôi thành demo. |

### 1.2. Cần bổ sung / điều chỉnh (6 điểm)

1. **Tách bạch *pilot* với *baseline reproduction*.** Pilot Tuần 1 chỉ là **quan sát định tính** (zero-shot, 10–20 mẫu, mục tiêu: nhìn thấy các kiểu lỗi — sai OCR, mất dấu, hallucination). Đây **không phải** baseline chính thức. Baseline reproduction (đo trên full test split, có số liệu đóng băng) diễn ra ở W4–W5. Nêu rõ điều này trong báo cáo để hội đồng không hiểu nhầm.
2. **Gap phải *evidence-based*.** Mỗi gap (G1–G4) cần **trích dẫn paper** chứng minh, không chép lại gap trong đề cương. Ví dụ: gap "scene-text tiếng Việt yếu" phải dẫn được ViTextVQA (Nguyen et al., 2024) + một VLM survey (Zhang et al., TPAMI 2024).
3. **Dựng hạ tầng tái lập ngay Tuần 1.** Tạo repo skeleton theo cấu trúc dự án (`src/`, `experiments/`, `configs/`, `adapters/`), **chốt convention logging** (seed, config, hardware, thời gian train/inference) và **quy ước Git** (branch, code review chéo). Nợ khoản này sẽ khiến reproducibility package ở W15 rất khổ.
4. **Kiểm tra khả thi dữ liệu & license (Ethics).** Xác nhận **tải được** ViVQA + ViTextVQA, kiểm tra **license** và vấn đề **privacy** (khuôn mặt/giấy tờ trong ảnh). Đây là rủi ro thật — verify ở W1 rẻ hơn nhiều so với phát hiện ở W4.
5. **Khởi tạo *question-type taxonomy* từ pilot.** Ngay khi xem mẫu, phân loại câu hỏi thành **Yes/No · Counting · Object recognition · Text reading · Reasoning**. Đây là *cơ sở bắt buộc* cho error analysis ở W12; làm sớm sẽ tiết kiệm rất nhiều.
6. **Lập tiêu chí chọn baseline (chưa cần chốt).** Chưa cần quyết định model ở W1, nhưng nên có **shortlist + tiêu chí** (GPU footprint, hỗ trợ HuggingFace, khả năng tiếng Việt/scene-text, license). Quyết định cuối cùng đặt ở W2–W3.

### 1.3. Checklist "Definition of Done" cho Tuần 1

- [ ] Literature Matrix ≥ 12 paper, mỗi dòng có cột *"Dùng được gì cho đề tài"*.
- [ ] `Research_Gap.md` — G1–G4 kèm **citation** cho từng gap.
- [ ] `Research_Questions.md` + `Hypotheses.md` (RQ đo được, hypothesis kiểm chứng được).
- [ ] `Dataset_Analysis.md` v0 — thống kê sơ bộ + xác nhận tải/license được.
- [ ] Pilot report: 10–20 mẫu zero-shot + bảng phân loại lỗi + taxonomy v0.
- [ ] Repo skeleton + convention logging + Git workflow đã push.
- [ ] Shortlist baseline + tiêu chí lựa chọn.

> **Nhận xét feasibility:** khối lượng trên hoàn toàn vừa cho 1 tuần với 4 người, **với điều kiện** phần pilot dùng model nhỏ (zero-shot) và không sa đà tune. Nếu chưa có GPU sẵn sàng, ưu tiên dựng repo/log + đọc paper trước, pilot có thể dùng bản inference nhẹ trên Colab.

---

## 2. Bảng kế hoạch tổng quan 15 tuần

| Tuần | Giai đoạn / Milestone | Nội dung trọng tâm | Công việc chi tiết | Phân công (SV1–SV4) | Deliverables |
|---|---|---|---|---|---|
| **W1** | TP1 · Orientation | Paper, gap, dataset, pilot, RQ | ≥3 paper/SV + matrix; khảo sát ViVQA/ViTextVQA; pilot zero-shot 10–20 mẫu; RQ+Hypotheses; dựng repo + log convention | SV1: shortlist baseline · SV2: dataset survey + license · SV3: đọc LoRA/QLoRA + setup env · SV4: khung eval harness + schema hallucination | Literature Matrix, Research\_Gap (evidence), RQ, Hypotheses, Dataset\_Analysis v0, repo skeleton |
| **W2** | TP2 · Foundation | Chốt gap/RQ, EDA sâu, cố định split, chọn baseline | Finalize matrix+gap; EDA (phân bố câu hỏi, vocab, taxonomy); cố định split train/val/test + seed; quyết định baseline chính thức | SV1: decision matrix chọn baseline + plan reproduce · SV2: EDA + split + data card · SV3: env PEFT (pin versions) + candidate methods · SV4: cài EM/VQA-Acc/ANLS + unit test | Split cố định, Dataset\_Analysis final, eval harness v1, baseline + lý do chọn |
| **W3** | ★ **REVIEW 1** | Chốt định hướng nghiên cứu | Tổng hợp W1–W2 thành báo cáo + slide; xác nhận gap có evidence; trình bày pilot + kế hoạch 15 tuần + risk register | Mỗi SV trình bày mảng của mình; SV1 điều phối slide | Slide Review 1, Report 1 (Intro), Report 3 (Existing Systems) draft |
| **W4** | TP2 · Baseline bring-up | Chạy baseline end-to-end + số zero-shot đầu | Load baseline, inference ảnh+câu hỏi→trả lời trên test subset; OCR v0 (Paddle/Viet) cho ViTextVQA; log đầy đủ | SV1: pipeline inference baseline · SV2: dataloader + OCR v0 · SV3: scaffold LoRA/QLoRA config · SV4: eval zero-shot → bảng metric đầu | Baseline inference pipeline, zero-shot metrics v0, OCR v0 |
| **W5** | TP2 · Baseline done | Reproduce hoàn tất + đóng băng số tham chiếu | Hoàn thiện + sanity-check reproduce; đo chất lượng OCR; dry-run training tí hon validate loop | SV1: chốt + document reproduce · SV2: OCR pipeline + báo cáo chất lượng · SV3: dry-run QLoRA (no OOM, loss giảm) · SV4: **freeze baseline reference** + bảng theo dõi | Reproduced baseline report, baseline reference (frozen), OCR quality report |
| **W6** | TP2→TP3 · Methodology | Report 4 + chốt cải tiến + freeze protocol | Viết Methodology; chốt improvement (vd QLoRA + OCR-prompting); freeze experimental protocol; **khởi động run cải tiến đầu tiên** | SV1: mô tả kiến trúc baseline · SV2: methodology data+OCR · SV3: thiết kế + start run PEFT đầu · SV4: chốt protocol eval/ablation/hallucination | Report 4 (Methodology), protocol đóng băng, kết quả cải tiến sơ bộ |
| **W7** | ★ **REVIEW 2** | Baseline + method + preliminary results | Trình bày baseline reproduced, phương pháp đề xuất, **kết quả sơ bộ** (baseline vs cải tiến đầu) | Mỗi SV trình bày; SV3+SV4 nhấn preliminary results | Slide Review 2, bảng preliminary results |
| **W8** | TP3 · Improvement v1 | Cài đặt đầy đủ cải tiến + train hoàn chỉnh | Train LoRA/QLoRA; tune rank/lr/epochs; (nếu RAG) build index; OCR-prompt injection; eval từng run + đo efficiency | SV3: train + tune (lead) · SV1: tích hợp adapter + inference · SV2: OCR/RAG refine · SV4: eval + efficiency log | Adapted model v1, results vs baseline |
| **W9** | TP3 · Improvement refine | Đạt cải thiện đo được, ổn định | Refine hyperparams (4-bit vs 8-bit, rank); đảm bảo so sánh cùng điều kiện; multi-seed nếu khả thi | SV3: refine (lead) · SV1: fair comparison · SV2: refine OCR/RAG · SV4: consolidate + robustness | Best config locked, cải thiện xác nhận vs baseline |
| **W10** | TP3→TP4 · Main experiments | Full grid main results (+ VLM tham chiếu) | Full runs cho ablation (zero-shot/prompt-opt/PEFT; ±OCR; ±RAG); chạy VLM thứ 2 (optional, reference) | SV1: run VLM tham chiếu · SV3: hoàn tất runs ablation · SV2: hỗ trợ OCR/RAG · SV4: main results table | Main results (Report 6 draft), toàn bộ log |
| **W11** | TP4 · Evaluation | Hoàn tất eval định lượng | EM/VQA-Acc/ANLS mọi run; bảng efficiency (%params, time, mem) | SV4: eval toàn bộ (lead) · SV3: cung cấp runs · SV1/SV2: hỗ trợ | Full evaluation tables + efficiency analysis |
| **W12** | TP4 · Analysis | Ablation + error + hallucination | Ablation attribution; error theo question-type; failure-case viz ảnh nhiều chữ; hallucination rate/type + case study (diacritics, code-switching, OCR) | SV4: hallucination + error (lead) · SV2: viz text-heavy + lỗi OCR · SV3: ablation attribution · SV1: hỗ trợ | Ablation + error + hallucination report (Report 6&7) |
| **W13** | ★ **FACULTY REVIEW** | Milestone chính — phần lớn research phải xong | Trình bày methodology, experiments, main results, ablation, error, hallucination | Cả nhóm; phân vai trình bày rõ | Slide Faculty Review, Results/Discussion draft |
| **W14** | TP5 · Prototype + Thesis | Web prototype + viết thesis | Prototype (Gradio/Streamlit): upload ảnh, hỏi tiếng Việt, answer, **confidence**, OCR/evidence; dùng adapted model (không train mới); viết thesis theo phần | SV4: prototype + deploy (lead) · SV1/SV2/SV3: viết phần chuyên môn | Prototype chạy được, thesis draft |
| **W15** | ★ **FINAL DEFENSE** | Hoàn thiện + bảo vệ | Hoàn thiện thesis; đóng gói reproducibility package; slide defense; bảo vệ | Cả nhóm; SV phụ trách từng phần Q&A | Final thesis, reproducibility package, slide defense |

**Bốn work-stream chồng lên timeline:** TP1 (W1) · TP2 (W2–W6) · TP3 (W7–W10) · TP4 (W11–W13) · TP5 (W14–W15). Các milestone W3/W7/W13/W15 là **checkpoint** phủ lên các work-stream, không phải tuần "làm việc" riêng.

---

## 3. Phân rã chi tiết Task Packages

### TP1 (W1) — Orientation *(nhóm đã có, xem Mục 1)*
Nền cho toàn bộ: literature matrix, gap có evidence, RQ/hypothesis, pilot định tính, hạ tầng log.

---

### TP2 (W2–W6) — Dataset · OCR · Baseline reproduction

**Mục tiêu:** có (a) pipeline dữ liệu chuẩn hoá + split cố định, (b) OCR tiếng Việt hoạt động, (c) **1 baseline reproduce được, số liệu đóng băng** làm mốc so sánh.

**3.1. Data preprocessing (SV2)**
- Chuẩn hoá ảnh: resize theo yêu cầu của backbone; giữ tỉ lệ; lưu ý ảnh scene-text độ phân giải cao.
- Chuẩn hoá câu hỏi/đáp án tiếng Việt: **Unicode NFC normalization**, quyết định rõ về lowercasing và xử lý dấu câu (ghi vào data card để tái lập).
- **Split cố định** `train/val/test` + `seed` — commit thành file, không random lại giữa các run.
- Gắn nhãn **question-type taxonomy** trên một sample (cơ sở cho error analysis W12).
- *Tools:* `datasets` (HuggingFace), `torchvision.transforms`, `pandas`.

**3.2. OCR integration (SV2)**
- Pipeline: **detect → recognize → post-process**. So sánh **VietOCR** (transformer, mạnh cho tiếng Việt) vs **PaddleOCR** (`lang='vi'`); giữ lại engine tốt hơn, engine kia làm backup.
- Đo chất lượng OCR trên mẫu (CER/WER nếu có GT, hoặc chấm tay); chú ý **dấu thanh** và text nghiêng/nhoè.
- Chuẩn hoá output OCR thành trường text để chèn vào prompt ở TP3.

**3.3. Baseline reproduction (SV1) — chọn ĐÚNG 1 model**

| Baseline | Kích thước tiêu biểu | Ưu điểm | Hạn chế cho tiếng Việt | PEFT/GPU feasibility |
|---|---|---|---|---|
| **BLIP-2** (OPT/FlanT5) | ~2.7B–T5 | Nhẹ, có sẵn HF, dễ chạy | Instruction-following yếu; tiếng Việt & scene-text kém | Cao (nhẹ nhất) |
| **Florence-2** | 0.23B / 0.77B | Rất nhỏ, hiệu quả; unified vision (có OCR task) | Không phải chat-VQA thuần; sinh tiếng Việt hạn chế; cần task prompt | Rất cao |
| **Qwen2.5-VL** | 3B / 7B | **Multilingual mạnh** (tiếng Việt tốt hơn hẳn); đọc scene-text/OCR nội tại tốt; hợp ViTextVQA | Bản 7B nặng | 3B: khả thi QLoRA trên Colab Pro; 7B: cân nhắc |

> **Khuyến nghị:** chọn **Qwen2.5-VL-3B-Instruct** làm *primary baseline* — cân bằng tốt nhất giữa năng lực tiếng Việt, khả năng scene-text (rất hợp ViTextVQA) và feasibility QLoRA. Giữ **BLIP-2** làm *fallback nhẹ* nếu GPU quá hạn chế. **Bắt buộc verify GPU** trước khi chốt.
>
> **Cảnh báo scope:** Ràng buộc là reproduce **đúng 1** baseline chính thức. Một VLM thứ hai (nếu có) **chỉ là reference** cho phần "different backbones" trong ablation — *đừng* biến nó thành baseline chính thức thứ hai, sẽ over-scope và loãng contribution.

- *Reproduce discipline:* dùng checkpoint chính thức trên HF; **pin version** (`transformers`, `accelerate`, `bitsandbytes`); sanity-check với repo/paper gốc ở mức khả thi; document mọi sai lệch và lý do. Đóng băng số zero-shot làm **baseline reference** (W5).
- *Frameworks:* `transformers`, `accelerate`, `bitsandbytes`, `peft`.

---

### TP3 (W7–W10) — PEFT adaptation & cải tiến *(contribution lõi)*

**Mục tiêu:** đề xuất + kiểm chứng **≥ 1 cải tiến đo được** so với baseline, **PEFT-only**, cùng điều kiện.

**3.4. LoRA / QLoRA (SV3)**
- **QLoRA** = quantize base model **4-bit (NF4)** + gắn **LoRA adapters** → nhét được VLM lớn vào GPU nhỏ. Dùng `bitsandbytes` 4-bit + `peft.LoraConfig` + `prepare_model_for_kbit_training`.
- Cấu hình chính: `target_modules` (thường `q_proj,k_proj,v_proj,o_proj`, có thể thêm `gate/up/down_proj`), `r` (8/16/32), `lora_alpha`, `lora_dropout`.
- **Giữ vision encoder frozen**, chỉ adapt phần language/interface là chiến lược phổ biến, tiết kiệm và ổn định.
- **Bằng chứng efficiency:** log `%trainable params` (thường **< 1–2%**) — đây là con số phải xuất hiện trong thesis.
- Training practicalities: `bf16` mixed precision, **gradient checkpointing**, **gradient accumulation** (effective batch nhỏ), **early stopping** theo val metric, log seed/config/thời gian/GPU mem.

**3.5. OCR-enhanced prompting (SV2 + SV3)**
- Chèn text OCR vào prompt cho câu hỏi scene-text, ví dụ: `Văn bản trong ảnh: "{ocr_text}". Câu hỏi: {question}`.
- Thiết kế để **ablate được** (có/không OCR) — bắt buộc cho W12.

**3.6. Các hướng cải tiến khác (tùy chọn, theo thời gian)**
- **Prompt optimization / multimodal prompting:** template tiếng Việt hệ thống, few-shot.
- **RAG (retrieval-augmented VQA):** build **FAISS** index truy hồi text/tri thức. *Cảnh báo:* RAG cho VQA nặng và dễ trượt tiến độ — **ưu tiên QLoRA + OCR-prompting làm contribution chính**, RAG chỉ thêm nếu còn thời gian.

**3.7. Fair comparison (SV1)**
- Adapted model vs baseline trên **cùng split, cùng decoding params, cùng phần cứng**. Mọi khác biệt phải giải thích được — nếu không, không được tuyên bố "cải thiện".

---

### TP4 (W11–W13) — Evaluation · Ablation · Error · Hallucination

**Mục tiêu:** định lượng đầy đủ + phân tích *tại sao* cải thiện (không chỉ *bao nhiêu*).

**3.8. Metric (SV4)** — cài chính xác, có unit test:
- **Exact Match (EM):** normalized (NFC, lowercase, strip, bỏ dấu câu) — cẩn thận với tiếng Việt.
- **VQA Accuracy:** soft accuracy `min(#người-đồng-ý / 3, 1)` nếu có nhiều đáp án; hoặc accuracy vs GT đơn nếu format ViVQA là 1 đáp án — **làm rõ theo đúng format dataset**.
- **ANLS:** Average Normalized Levenshtein Similarity, threshold **0.5** — metric then chốt cho **ViTextVQA / scene-text**.
- *(BLEU/CIDEr chỉ báo cáo như chỉ số phụ cho câu trả lời sinh mở.)*

**3.9. Ablation study (bắt buộc)** — grid tối thiểu:
- Zero-shot vs prompt-optimized vs LoRA/QLoRA-adapted.
- Có vs không OCR-enhanced prompting.
- Có vs không retrieval augmentation.
- Các VLM backbone khác nhau *(dùng VLM reference)*.

**3.10. Error analysis (bắt buộc):**
- Phân rã lỗi theo **question-type** (yes/no, counting, text-reading, reasoning).
- **Failure-case visualization** trên ảnh tiếng Việt nhiều chữ.
- Chỉ ra pattern lỗi (vd: sai chủ yếu ở scene-text nghiêng, hay ở counting).

**3.11. Hallucination analysis (bắt buộc):**
- **Object hallucination:** dùng **POPE-style** (Li et al., EMNLP 2023 — [9]): polling yes/no về sự tồn tại object.
- **Text hallucination:** đo mismatch giữa đáp án và OCR/nội dung ảnh.
- **Annotation schema** rõ ràng, ≥ 2 người chấm + báo cáo **inter-annotator agreement**.
- Case study đặc thù tiếng Việt: diacritics, code-switching, lỗi OCR.

**3.12. Efficiency reporting (bắt buộc):** `%trainable params`, thời gian train, latency inference, GPU memory, kích thước adapter — đây là chiều đánh giá cốt lõi của đề tài ("efficient").

*Tools:* `pandas` (bảng), `matplotlib/seaborn` (viz), metric tự cài + kiểm chứng.

---

### TP5 (W14–W15) — Prototype · Thesis · Defense

**3.13. Web prototype (SV4)**
- Stack: **Gradio** (nhanh, ưu tiên) hoặc **Streamlit**.
- Thành phần: upload ảnh → nhập câu hỏi tiếng Việt → sinh câu trả lời → **hiển thị confidence** → hiển thị **OCR text / evidence** (nếu RAG).
- Chạy trên **adapted (quantized) model** — không train mới; đây chính là minh chứng cho khía cạnh *efficient deployment*.
- **Confidence là proxy, phải trung thực:** với VLM sinh, "confidence" nên lấy từ **sequence log-prob / token probability** (hoặc self-consistency), và **nói rõ giới hạn** trên UI (gắn với Responsible AI — CLO7).
- Deploy: Colab + `gradio share`/ngrok, hoặc HF Spaces (lưu ý giới hạn kích thước model), hoặc GPU server trường. Backup: **demo video** nếu deploy live trục trặc.

**3.14. Thesis (cả nhóm)** — cấu trúc: Introduction · Related Work/Existing Systems · Methodology · Experiments & Results · Discussion (ablation/error/hallucination) · Conclusion & Future Work. Mỗi SV viết phần thuộc mảng của mình.

**3.15. Reproducibility package (bắt buộc nộp):** source code · **LoRA/QLoRA adapter weights** · configuration files · inference script · README (setup & run) · experiment log (metric, seed, prompt, settings từng run).

**3.16. Defense slides:** problem → gap (evidence) → method → results → ablation → error/hallucination → prototype → **limitations & future work**.

---

## 4. Ma trận Quản trị Rủi ro & Phương án Dự phòng

| # | Rủi ro | Xác suất | Ảnh hưởng | Dấu hiệu nhận biết | Phòng ngừa (Mitigation) | Dự phòng (Contingency) |
|---|---|---|---|---|---|---|
| R1 | **GPU không đủ / OOM** khi train VLM lớn | Cao | Cao | CUDA OOM; batch phải = 1; train quá chậm | QLoRA **4-bit NF4**, gradient checkpointing, grad accumulation, chọn bản **3B thay vì 7B**, giảm image resolution | Chuyển **BLIP-2/Florence-2** (nhẹ hơn); Colab Pro+; freeze thêm layer |
| R2 | **Baseline không reproduce khớp** kết quả kỳ vọng | Trung bình | Cao | Số liệu lệch xa paper/repo; output vô nghĩa | Dùng checkpoint chính thức; **pin version**; đối chiếu repo gốc; document từng bước | Đổi baseline khác trong danh sách; coi số reproduce là **reference nội bộ** và giải trình rõ |
| R3 | **OCR tiếng Việt kém** (diacritics, scene-text nghiêng) | Trung bình | Cao | CER cao; mất dấu; đọc sai chữ | Dùng **VietOCR** cho tiếng Việt; thử cả PaddleOCR; deskew/crop preprocessing | Đổi OCR engine (Paddle↔Viet); thu hẹp scope OCR-prompting về tập câu hỏi OCR sạch |
| R4 | **Cải tiến không vượt baseline** (no measurable gain) | Trung bình | **Rất cao** | Adapted ≈ baseline trên mọi metric | **Chuẩn bị ≥ 2 hướng** (QLoRA + OCR-prompting); bắt đầu sớm (W6); ablation để cô lập nguyên nhân | **Negative result vẫn hợp lệ** nếu có ablation + error analysis giải thích; pivot sang RAG/prompt-opt |
| R5 | **Training không hội tụ / overfitting** | Trung bình | Cao | Loss phân kỳ; val giảm khi train tăng | LR schedule + warmup; giảm rank; early stopping; tăng/augment data | Chuyển sang **prompting/OCR-prompting** (không cần train) làm contribution |
| R6 | **Dataset lỗi / license / khó tải** | Thấp–TB | Cao | Link hỏng; annotation lệch; ràng buộc license | **Verify sớm ở W1–W2**; kiểm license/privacy; viết data card | Dùng subset đã tải; liên hệ tác giả; thay bằng phần dữ liệu khả dụng |
| R7 | **Trượt tiến độ trước milestone** (W7/W13) | Trung bình | Cao | Chưa có preliminary result trước W7; research chưa xong trước W13 | Buffer 10–15%; **freeze scope sớm**; weekly standup; khởi động improvement từ W6 | **Cắt scope xuống 1 cải tiến**; ưu tiên baseline + 1 improvement + analysis đầy đủ |
| R8 | **Hallucination khó đo khách quan** | Trung bình | Trung bình | Bất đồng giữa người chấm; định nghĩa mơ hồ | Schema rõ; **POPE-style** cho object; ≥ 2 annotator + đo **agreement** | Giảm cỡ mẫu annotation nhưng giữ rigor; tập trung một loại hallucination |
| R9 | **Phối hợp nhóm** (SV không hoàn thành phần của mình) | Trung bình | Cao | Task trễ dây chuyền; trùng/bỏ sót việc | **Individual deliverables** rõ; code review chéo; Git workflow; log tiến độ | Tái phân công; pair-work; nâng đỡ mảng nghẽn |

---

## 5. Nguyên tắc điều hành (lưu ý của người hướng dẫn)

1. **Contribution là "vạch đỗ".** Prototype chạy đẹp *không* thay được cải tiến kiểm chứng so với baseline. Bắt đầu improvement **đủ sớm (W6)** để có preliminary result trình Review 2 (W7).
2. **Reproduce trước, cải tiến sau — cùng điều kiện.** Mọi tuyên bố "tốt hơn" phải so với baseline đã đóng băng, cùng split/seed/decoding/hardware.
3. **Log từ ngày đầu.** Không có log = không tái lập = mất điểm rigor và khổ ở W15.
4. **Reproduce đúng 1 baseline.** VLM thứ hai chỉ là reference cho ablation backbone — đừng over-scope.
5. **Efficiency vừa là ràng buộc, vừa là chiều đánh giá.** Luôn báo cáo `%params / time / memory`, không chỉ accuracy.
6. **Chuẩn bị ≥ 2 hướng cải tiến; chấp nhận negative result nếu có rigor.** R4 là rủi ro nguy hiểm nhất — de-risk bằng đa hướng và bắt đầu sớm.
7. **Không bịa số liệu.** Báo cáo trung thực; một kết quả khiêm tốn có phân tích tốt giá trị hơn một con số đẹp không tái lập được.

---

*Tài liệu kế hoạch tổng thể ViVQA-VLM. Mọi thay đổi kỹ thuật phải nhất quán với ràng buộc PEFT-only và giao thức đánh giá (EM/VQA-Acc/ANLS + ablation + error + hallucination). Chi tiết thao tác từng tuần triển khai trong `docs/Wxx/`.*
