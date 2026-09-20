# ViVQA-VLM — MASTER PROJECT INSTRUCTION

> **Efficient Vietnamese Visual Question Answering using Vision-Language Models**
> Nghiên cứu hệ thống hỏi đáp trực quan tiếng Việt hiệu quả sử dụng mô hình Vision-Language
> **Abbreviation:** ViVQA-VLM · **Specialty:** NLP · **Supervisor:** Assoc.Prof. Đặng Ngọc Minh Đức

---

## 0. Cách dùng tài liệu này

Đây là **bản hướng dẫn tổng (master)** — nguồn tham chiếu duy nhất cho toàn bộ Capstone.

* Tài liệu này trả lời: *Project làm gì? Ràng buộc nào bắt buộc? Đánh giá ra sao? Ai làm gì? 15 tuần chạy thế nào?*
* Hướng dẫn **chi tiết theo tuần** (nhiệm vụ, template, checklist) đặt riêng trong `docs/Wxx/`.
* Mọi quyết định kỹ thuật đều phải **nhất quán với ràng buộc ở Mục 4** và **giao thức đánh giá ở Mục 7**.

Nguyên tắc xuyên suốt: **không xây "một ứng dụng AI", mà thực hiện một nghiên cứu tái lập được và kiểm chứng một giả thuyết kỹ thuật.**

---

## 1. Bài toán & Bối cảnh

**Vấn đề.** Visual Question Answering (VQA) yêu cầu hệ thống trả lời một câu hỏi ngôn ngữ tự nhiên dựa trên nội dung một ảnh. Khác với image classification hay object detection, VQA đòi hỏi hiểu **đồng thời** nội dung ảnh, ngữ nghĩa ngôn ngữ và khả năng suy luận.

```text
Vietnamese Image
       +
Vietnamese Question
       ↓
 Vision-Language Model (VLM)
       ↓
Vietnamese Answer
```

**Vì sao khó với tiếng Việt.** Hầu hết VLM mã nguồn mở (BLIP-2, Florence-2, LLaVA, Qwen2.5-VL) được pretrain chủ yếu trên dữ liệu tiếng Anh. Khi áp dụng cho tiếng Việt, hệ thống gặp:

* Text-in-image / scene-text (đọc chữ trong ảnh) — cần OCR tốt cho tiếng Việt.
* Dấu thanh (diacritics) và hiện tượng code-switching Anh–Việt.
* Hallucination: câu trả lời trôi chảy nhưng **sai** so với ảnh.

**Vì sao đề tài có ý nghĩa.** Thích nghi VLM cho tiếng Việt **dưới ngân sách tham số nghiêm ngặt** (không full fine-tuning) là hướng thực tế và còn ít được nghiên cứu. Đây là đóng góp kỹ thuật đo lường được, không phải một demo ứng dụng.

---

## 2. Mục tiêu & Đóng góp nghiên cứu

**Mục tiêu học thuật (learning objectives):**

* Nắm nền tảng VQA và Vision-Language Models.
* Nắm các benchmark Vietnamese VQA: **ViVQA** (general), **ViTextVQA** (scene-text).
* Đánh giá vài VLM pretrained trên benchmark tiếng Việt dưới ngân sách tham số.
* Khảo sát kỹ thuật thích nghi nhẹ: PEFT (LoRA/QLoRA), prompting, OCR/retrieval augmentation.
* Phân tích hành vi hallucination trên Vietnamese VQA.
* Xây prototype web hoàn chỉnh.

**Đóng góp nghiên cứu (contribution — bắt buộc, là "vạch đỗ" của cả đề tài):**

> Đề xuất và **kiểm chứng bằng thực nghiệm** ít nhất **một cải tiến kỹ thuật** so với một VLM baseline đã reproduce (ví dụ: một công thức PEFT, OCR-enhanced prompting, hoặc retrieval-augmented VQA) — sao cho **cải thiện đo được** độ chính xác Vietnamese VQA **hoặc** giảm hallucination, có **ablation** và **error analysis** đi kèm, và **không dùng full fine-tuning**.

Nếu không có phần cải tiến được kiểm chứng so với baseline, đề tài **chưa đạt** yêu cầu nghiên cứu — dù prototype có chạy đẹp.

---

## 3. Research Gap

| # | Gap | Nội dung |
|---|-----|----------|
| G1 | English-centric pretraining | Alignment ảnh–ngôn ngữ tối ưu cho tiếng Anh, chưa tối ưu cho tiếng Việt. |
| G2 | Vietnamese VQA yếu, nhất là scene-text | Text-in-image (OCR), diacritics, code-switching còn hạn chế. |
| G3 | PEFT cho VLM tiếng Việt chưa được khai thác | LoRA/QLoRA để thích nghi VLM cho Vietnamese VQA còn thiếu nghiên cứu bài bản. |
| G4 | Hallucination | VLM sinh câu trả lời trôi chảy nhưng sai (object/text hallucination) — thách thức về độ tin cậy. |

**Yêu cầu:** nhóm phải **xác nhận gap bằng literature** (evidence từ paper), không sao chép nguyên gap trong đề cương, và chỉ ra **gap nào khả thi** trong phạm vi Capstone.

---

## 4. Phạm vi & Ràng buộc bắt buộc (ĐỌC KỸ)

Đây là "luật chơi" — vi phạm là lệch đề tài.

**4.1. Baseline (bắt buộc reproduce đúng 1 model).** Trước mọi cải tiến, nhóm phải tái lập **đúng một** baseline mã nguồn mở để hội đồng có mốc tham chiếu rõ ràng:

* BLIP-2, **hoặc**
* Florence-2, **hoặc**
* Qwen2.5-VL.

**4.2. Cải tiến (novelty — bắt buộc ≥ 1).** Sau khi có baseline, triển khai **ít nhất một** (hoặc đóng góp tương đương):

* LoRA / QLoRA fine-tuning
* OCR-enhanced prompting (PaddleOCR / VietOCR)
* Retrieval-Augmented VQA (RAG)
* Prompt optimization
* Multimodal prompt engineering
* Lightweight VLM adaptation

Cải tiến phải **xuất phát từ Research Gap** và được đánh giá **so với baseline dưới điều kiện y hệt**.

**4.3. Ràng buộc huấn luyện (bắt buộc).** **KHÔNG được full fine-tuning.** Chỉ được dùng parameter-efficient fine-tuning:

* PEFT (họ adapter / LoRA)
* LoRA
* QLoRA (quantize 4-bit hoặc 8-bit)

Mục đích: chạy được trên **Google Colab Pro/Premium hoặc GPU server của trường**, và biến **efficiency thành một ràng buộc thiết kế tường minh**.

---

## 5. Datasets

| Dataset | Loại | Vai trò |
|---------|------|---------|
| **ViVQA** | General VQA | Câu hỏi chung trên ảnh. |
| **ViTextVQA** | Scene-text VQA | Câu hỏi cần đọc chữ trong ảnh (OCR). |

**Chuẩn bị dữ liệu:** chuẩn hoá xử lý ảnh; normalize câu hỏi/câu trả lời; trích OCR cho câu hỏi text-in-image (**PaddleOCR / VietOCR**); cố định split **train/validation/test**.

Nhóm cần lấy sample thực tế để quan sát và **phân loại câu hỏi** thành: Yes/No · Counting · Object recognition · Text reading · Reasoning — đây là cơ sở cho error analysis về sau.

---

## 6. Phương pháp — Pipeline tổng thể

```text
        ┌──────────────────────────┐
        │  Data prep + OCR + split │   (ViVQA, ViTextVQA)
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Reproduce 1 VLM baseline│   (BLIP-2 / Florence-2 / Qwen2.5-VL)
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Cải tiến (PEFT only):   │   LoRA/QLoRA · OCR-prompting
        │  ≥ 1 kỹ thuật            │   · RAG · prompt-opt
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Evaluation + Ablation   │   EM · VQA-Acc · ANLS
        │  + Error/Hallucination   │   + reproducibility log
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Web prototype           │   upload ảnh + hỏi tiếng Việt
        └──────────────────────────┘
```

VLM cần hiểu ở mức: Vision Encoder → Visual Representation → Vision-Language Interface → Language Model → Answer.

---

## 7. Giao thức đánh giá (Evaluation Protocol)

**7.1. Metric chính:**

| Metric | Dùng cho |
|--------|----------|
| Exact Match (EM) | Câu trả lời khớp tuyệt đối. |
| VQA Accuracy | Độ chính xác VQA chuẩn. |
| ANLS | Đặc biệt cho ViTextVQA / scene-text. |

> BLEU/CIDEr **không** dùng làm metric chính; chỉ báo cáo như chỉ số phụ cho câu trả lời sinh mở.

**7.2. Phân tích thêm:** hallucination analysis (tỉ lệ và loại object/text bị hallucinate); human evaluation trên một mẫu (nếu khả thi) về tính đúng và faithfulness.

**7.3. Ablation study (bắt buộc):**

* Zero-shot vs. prompt-optimized vs. LoRA/QLoRA-adapted.
* Có vs. không OCR-enhanced prompting.
* Có vs. không retrieval augmentation.
* Các VLM backbone khác nhau.

**7.4. Error analysis (bắt buộc):**

* Phân rã lỗi theo loại câu hỏi (yes/no, counting, text-reading, reasoning).
* Trực quan hoá failure case trên ảnh tiếng Việt nhiều chữ.
* Case study hallucination + thách thức đặc thù tiếng Việt (diacritics, code-switching, lỗi OCR).

**7.5. Báo cáo để tái lập (mọi thí nghiệm):** dataset split, random seed, cấu hình phần cứng, thời gian inference **và** thời gian PEFT fine-tuning.

---

## 8. Phân công nhóm (4 sinh viên)

Nhóm 4 SV — phải thể hiện **rõ trách nhiệm cá nhân**, không chấp nhận kiểu "cả nhóm làm dataset". Phân công khởi điểm (nhóm có thể tự đề xuất khác nếu hợp lý):

| SV | Mảng | Trách nhiệm kỹ thuật chính |
|----|------|----------------------------|
| SV1 | **VLM / Baseline** | Hiểu & reproduce baseline; VQA/VLM; BLIP-2, LLaVA, Florence-2, Qwen2.5-VL. |
| SV2 | **Dataset / OCR** | ViVQA, ViTextVQA; scene-text; PaddleOCR/VietOCR; thách thức tiếng Việt. |
| SV3 | **PEFT / Training** | LoRA, QLoRA, prompt engineering, multimodal prompting, RAG. |
| SV4 | **Evaluation / Deployment** | EM/VQA-Acc/ANLS; hallucination; efficient inference; prototype/deploy. |

Mỗi SV phải có: **individual tasks · individual deliverables · technical responsibility.** Song song vẫn có **team tasks · team deliverables**.

---

## 9. Lộ trình 15 tuần & Milestones

Bảng dưới là **khung tổng**; nhóm tự chi tiết hoá tasks từng tuần nhưng **không được đổi các milestone bắt buộc**.

| Week | Objective | Deliverable / Milestone |
|------|-----------|-------------------------|
| W1 | Research orientation & planning | W1 Report, kế hoạch 15 tuần |
| W2 | Literature & dataset | Literature matrix, dataset analysis |
| **W3** | **Review 1** | **REVIEW 1** |
| W4 | Baseline | Môi trường + baseline khởi động |
| W5 | Baseline | Baseline reproduce xong |
| W6 | Baseline → Method | Methodology (Report 4) |
| **W7** | **Review 2** | **REVIEW 2** — baseline + proposed method + preliminary results |
| W8 | Improvement | PEFT/OCR/RAG bản đầu |
| W9 | Improvement | Cải tiến hoàn thiện |
| W10 | Experiments | Kết quả chính |
| W11 | Evaluation | Metric đầy đủ |
| W12 | Analysis | Ablation + error + hallucination |
| **W13** | **Hội đồng cấp Khoa** | **FACULTY REVIEW** — phần lớn nghiên cứu phải xong |
| W14 | Prototype / Thesis | Prototype + viết thesis |
| W15 | Finalization | **FINAL DEFENSE** + reproducibility package |

**Diễn giải milestone:**

* **W3 — Review 1:** xác định rõ trình bày gì, deliverable trước review, ai phụ trách, kết quả tối thiểu.
* **W7 — Review 2:** baseline đã reproduce? proposed method là gì? có preliminary results? trình bày experiment nào?
* **W13 — Hội đồng cấp Khoa:** coi là **milestone chính**, không phải tuần thường. Trước W13 phải xong phần lớn: methodology, experiments, main results, ablation, error analysis, hallucination analysis.
* **W14–W15:** hoàn thiện prototype, thesis, documentation và final defense.

---

## 10. Deliverables & Reproducibility Package

**Sản phẩm chính:**

* Hệ thống Vietnamese VQA (VLM đã adapt bằng PEFT).
* Prototype web demo.
* Báo cáo đánh giá so sánh các VLM, kèm ablation và hallucination analysis.
* Thesis Capstone hoàn chỉnh (khuyến khích hướng tới công bố nghiên cứu sinh viên).

**Reproducibility package (bắt buộc nộp):**

* Source code
* LoRA/QLoRA adapter weights
* Configuration files
* Inference script
* README (hướng dẫn setup & run)
* Experiment log (metric, seed, prompt, settings từng run)

---

## 11. Prototype (Web)

Yêu cầu prototype web:

* Upload một ảnh tiếng Việt.
* Nhập câu hỏi bằng tiếng Việt.
* Sinh câu trả lời ngôn ngữ tự nhiên.
* Trực quan hoá **confidence** của model, và (khi liên quan) **OCR text trích được / evidence được retrieve**.

Runtime: Colab Pro/Premium hoặc GPU server trường · Python + PyTorch + HuggingFace Transformers + PEFT.

---

## 12. Rủi ro & Phương án dự phòng

Kế hoạch phải có **≥ 5 rủi ro**. Khung tham khảo:

| Risk | Impact | Mitigation | Backup |
|------|--------|-----------|--------|
| Model quá lớn | High | Dùng model nhỏ hơn | — |
| GPU không đủ | High | QLoRA / quantization | Colab |
| Dataset lỗi | Medium | Preprocessing kỹ | — |
| Baseline khó reproduce | High | VLM thay thế | — |
| OCR không tốt | Medium | Đổi OCR (Paddle↔Viet) | — |
| Training không hội tụ | High | Chuyển sang prompting | — |
| Deployment khó | Medium | Cloud / server | — |

---

## 13. AI Ethics & Responsible AI

Theo CLO7 (Professionalism and Ethical Conduct in AI):

* **Privacy** ảnh upload (khuôn mặt, giấy tờ, dữ liệu cá nhân) và licensing dataset phù hợp.
* **Bias & fairness** — thiên lệch English-centric/văn hoá trong VLM có thể bất lợi cho nội dung tiếng Việt hoặc nhóm cụ thể.
* **Hallucination & reliability** — câu trả lời trôi chảy-nhưng-sai; giới hạn của VQA tự động trong ngữ cảnh nhạy cảm.
* **Responsible use** — truyền đạt rõ confidence và giới hạn hệ thống.

---

## 14. Nguyên tắc làm việc & Mindset

**Chuyển dịch tư duy bắt buộc:**

> Từ *"Build an AI application"* → sang *"Conduct a reproducible research project and validate a technical hypothesis."*

**Quy tắc vận hành:**

* **Reproduce trước, cải tiến sau** — mọi cải tiến phải so với baseline đã tái lập, cùng điều kiện.
* **Mọi thí nghiệm phải log** — split, seed, phần cứng, hyperparameters, thời gian (mục tiêu: reproducibility).
* **Ablation-driven** — mỗi tuyên bố "cải thiện" phải có ablation chứng minh phần nào tạo ra cải thiện đó.
* **Evidence-based** — research gap và kết luận đều phải có evidence từ literature/thí nghiệm.
* **Cách đọc paper:** mỗi paper trả lời tối thiểu 10 câu hỏi, trong đó **bắt buộc** câu *"What can we learn/use for our project?"*.

---

## 15. Cấu trúc thư mục dự án

```text
ViVQA-VLM/
│
├── docs/
│   ├── W01/ … W15/            # hướng dẫn + báo cáo theo tuần
│   │   ├── SVx_Paper_Review.md
│   │   ├── Literature_Matrix.md
│   │   ├── Dataset_Analysis.md
│   │   ├── Research_Gap.md
│   │   ├── Research_Questions.md
│   │   ├── Hypotheses.md
│   │   ├── Candidate_Methods.md
│   │   ├── Project_Plan_15_Weeks.md
│   │   └── Wxx_Team_Report.md
│   └── Master_Instruction.md  # tài liệu này
│
├── data/                      # dataset đã chuẩn hoá + split cố định
├── src/
│   ├── data/                  # preprocessing, OCR
│   ├── baseline/              # reproduce VLM baseline
│   ├── adaptation/            # LoRA/QLoRA, prompting, RAG
│   ├── eval/                  # EM, VQA-Acc, ANLS, hallucination
│   └── prototype/             # web app
├── experiments/               # log theo run (seed, config, metrics)
│   └── Wxx_*/
├── adapters/                  # LoRA/QLoRA weights
├── configs/                   # file cấu hình
└── README.md
```

---

## 16. Định nghĩa "Hoàn thành" (Definition of Done)

Project được xem là đạt khi chứng minh đủ 4 trụ:

* **Knowledge** — hiểu VQA, VLM, Vietnamese VQA, ViVQA/ViTextVQA, các hướng adaptation.
* **Research** — có research gap (evidence), RQ, hypothesis, candidate methods; reproduce baseline; đề xuất + kiểm chứng ≥ 1 cải tiến (PEFT only) so với baseline.
* **Rigor** — ablation + error analysis + hallucination analysis; mọi thí nghiệm tái lập được (split/seed/hardware/time).
* **Delivery** — prototype web chạy được; thesis + reproducibility package đầy đủ; bảo vệ thành công.

---

## Phụ lục — Tài liệu tham khảo chính

**Foundations:** BLIP-2 (Li et al., ICML 2023) · LLaVA — Visual Instruction Tuning (Liu et al., NeurIPS 2023).
**Benchmark / Dataset:** ViVQA (Tran et al., PACLIC 2021) · ViTextVQA (Nguyen et al., arXiv:2404.10652, 2024; bản mở rộng trên *Expert Systems with Applications*, 2025).
**Baseline Models:** Florence-2 (Xiao et al., CVPR 2024) · Qwen2.5-VL Technical Report (Bai et al., 2025).
**Recent methods:** Vietnamese VQA with Transformer & Convolutional Integration (Nguyen et al., *Computers & Electrical Engineering*, vol. 119, 2024) · Florence-VL (Chen et al., CVPR 2025).
**Hallucination:** POPE — Evaluating Object Hallucination in LVLMs (Li et al., EMNLP 2023).
**Survey:** Vision-Language Models for Vision Tasks: A Survey (Zhang et al., IEEE TPAMI 2024).

---

*Tài liệu định hướng tổng thể. Chi tiết thao tác theo tuần xem trong `docs/Wxx/`. Mọi thay đổi kỹ thuật phải nhất quán với Mục 4 (ràng buộc) và Mục 7 (giao thức đánh giá).*
