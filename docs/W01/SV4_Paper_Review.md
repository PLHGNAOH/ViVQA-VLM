# SV4 Paper Review — Evaluation / Deployment

> **Đề tài:** ViVQA-VLM — Efficient Vietnamese Visual Question Answering using Vision-Language Models
> **Vai trò:** SV4 — Evaluation / Deployment (EM, VQA-Acc, ANLS, hallucination, efficient inference, prototype web)
> **Tuần:** W01 · **Ngày:** 2026-09-14 · **Reviewer:** SV4
> **Yêu cầu:** ≥ 3 paper, ưu tiên: 1 paper hallucination (POPE — Li et al., EMNLP 2023), 1 paper về metric VQA/ANLS (VQAv2 — Goyal et al., 2017; ST-VQA — Biten et al., 2019), 1 paper về efficient inference / evaluation VLM (survey Zhang et al., TPAMI 2024). Mỗi paper trả lời **đủ 10 câu hỏi** + mục **"Paper này liên quan gì đến đề tài ViVQA-VLM?"**. Cấm chỉ tóm tắt abstract.

## 0. Danh sách paper đã chọn & lý do

| # | Paper | Venue / Năm | Nhóm | Vì sao chọn |
|---|-------|-------------|------|-------------|
| P1 | POPE — Evaluating Object Hallucination in LVLMs (Li et al.) | EMNLP 2023 | Hallucination | Chuẩn đo hallucination ổn định, dễ tái lập; là gốc cho POPE-vi tiếng Việt. |
| P2 | Making the V in VQA Matter — VQAv2 (Goyal et al.) | CVPR 2017 | Metric VQA | Định nghĩa chuẩn **VQA Accuracy** + cảnh báo language prior — nền tảng cách chấm điểm của nhóm. |
| P3 | Vision-Language Models for Vision Tasks: A Survey (Zhang et al.) | IEEE TPAMI 2024 | Evaluation & adaptation | Bản đồ tổng quan VLM + PEFT + protocol đánh giá — định vị phương pháp & cách so sánh fair. |

> Bổ sung định nghĩa **ANLS** (ST-VQA, Biten et al., ICCV 2019) trong phần metric của P2 và bảng tổng hợp, vì ANLS là metric chính cho ViTextVQA.

---

## P1. Evaluating Object Hallucination in Large Vision-Language Models (POPE)

| Trường | Giá trị |
|--------|---------|
| Title | Evaluating Object Hallucination in Large Vision-Language Models |
| Author(s) | Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, Ji-Rong Wen |
| Year / Venue | 2023, EMNLP |
| Code | https://github.com/RUCAIBox/POPE |
| Reviewer | SV4 |

### 1. What problem does the paper solve?
Đo **object hallucination** (mô hình khẳng định có vật thể không tồn tại trong ảnh) của LVLM một cách **ổn định, khách quan**, thay cho CHAIR (dựa trên caption) vốn nhạy với độ dài và cách sinh.

### 2. Why is the problem important?
Hallucination phá độ tin cậy của VQA; với tiếng Việt (dấu thanh, code-switching, OCR) rủi ro càng cao. Cần thước đo tin cậy để kiểm chứng cải tiến PEFT **không làm tăng** hallucination.

### 3. What is the research gap?
Các phép đo cũ không ổn định, phụ thuộc prompt/độ dài caption; thiếu cách đo trực tiếp, có thể so sánh giữa các model.

### 4. What is the proposed method?
**POPE (Polling-based Object Probing Evaluation):** hỏi loạt câu **yes/no** dạng *"Is there a &lt;object&gt; in the image?"*, cân bằng số câu có/không, với **3 chiến lược chọn object âm**: random, popular (vật thể phổ biến), adversarial (vật thể hay đồng xuất hiện). Quy về **bài toán phân loại nhị phân**.

### 5. What is the key technical idea?
Biến đánh giá hallucination thành **probing nhị phân** ⇒ đo bằng Accuracy/Precision/Recall/F1 + **yes-ratio** (đo thiên lệch trả lời "yes"), ổn định và tái lập.

### 6. What dataset is used?
Ảnh + annotation object từ **MSCOCO** (mở rộng A-OKVQA, GQA); object âm lấy theo 3 chiến lược ở trên.

### 7. What metrics are used?
Accuracy, Precision, Recall, **F1**, và **yes-ratio**. F1 là chỉ số tổng hợp chính.

### 8. What are the main results?
LVLM hallucinate đáng kể và **thiên về trả lời "yes"**; setting **adversarial khó nhất**. Bộ đo phơi bày khác biệt giữa các model mà caption-based bỏ sót.

### 9. What are the limitations?
- Chỉ đo **object existence**, chưa đo hallucination thuộc tính/quan hệ/text.
- Tiếng Anh + cần annotation object.
- Không đo trực tiếp scene-text hallucination.

### 10. What can we learn/use for our project? **(BẮT BUỘC)**
Xây **POPE-vi**: dịch/chuẩn hoá câu probing sang tiếng Việt trên một subset có annotation object, đo **F1 + yes-ratio** trước/sau PEFT (kiểm chứng H5). Cài trong `src/eval/` như một module độc lập.

> ### Paper này liên quan gì đến đề tài ViVQA-VLM?
> Cung cấp **rubric hallucination** để trả lời RQ4 và bảo vệ tuyên bố "cải tiến không đánh đổi độ tin cậy". POPE-vi là phần **analysis bắt buộc** (Master §7.2) ở W12; ảnh hưởng prototype (hiển thị confidence, cảnh báo khi model có thể hallucinate).

---

## P2. Making the V in VQA Matter (VQAv2) + chuẩn metric VQA/ANLS

| Trường | Giá trị |
|--------|---------|
| Title | Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering |
| Author(s) | Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, Devi Parikh |
| Year / Venue | 2017, CVPR (arXiv:1612.00837) |
| Code | VQAv2 dataset + eval (https://visualqa.org) |
| Reviewer | SV4 |

### 1. What problem does the paper solve?
VQA v1 có **language prior** mạnh: model đoán đúng nhờ thống kê câu hỏi mà **không cần nhìn ảnh**. Paper làm cân bằng dataset để buộc model dùng ảnh.

### 2. Why is the problem important?
Nếu không kiểm soát prior, điểm số VQA gây hiểu nhầm. Với tiếng Việt (dữ liệu dịch, phân bố lệch) rủi ro prior còn lớn hơn — ta phải đánh giá công bằng.

### 3. What is the research gap?
Thiếu benchmark cân bằng và thiếu nhận thức về mức độ model "ăn may" nhờ prior ngôn ngữ.

### 4. What is the proposed method?
**Balanced VQA**: với mỗi câu hỏi, tìm **cặp ảnh bổ sung** cho **câu trả lời khác nhau** ⇒ model buộc phải phân biệt ảnh.

### 5. What is the key technical idea?
Cân bằng theo cặp (complementary pairs) triệt tiêu phần lớn language prior; đồng thời chuẩn hoá **công thức VQA Accuracy**.

### 6. What dataset is used?
VQAv2: **≈ 1.1M câu hỏi** trên **≈ 200K ảnh** COCO, mỗi câu 10 câu trả lời của người.

### 7. What metrics are used?
**VQA Accuracy** (soft):
`acc(ans) = min(#người-trả-lời-trùng / 3, 1)`, lấy trung bình trên các subset 10-choose-9 người.
Kèm chuẩn hoá câu trả lời (lowercase, bỏ dấu câu, số ↔ chữ, mạo từ). **ANLS** (ST-VQA, Biten 2019) cho scene-text: `1 − NL(pred, gt)` nếu NLD ≤ **0.5**, ngược lại 0 — dùng cho ViTextVQA.

### 8. What are the main results?
Sau cân bằng, điểm nhiều model **giảm mạnh** rồi cải thiện khi thực sự dùng ảnh ⇒ chứng minh prior tồn tại và cần dataset cân bằng.

### 9. What are the limitations?
- Ảnh tự nhiên tiếng Anh; không scene-text.
- Metric soft cần **≥ nhiều annotator**; ViVQA thường chỉ 1 đáp án ⇒ EM/accuracy đơn giản hoá.

### 10. What can we learn/use for our project? **(BẮT BUỘC)**
Cài **VQA Accuracy** đúng công thức + **chuẩn hoá câu trả lời tiếng Việt (NFC)**; **ANLS** cho ViTextVQA; luôn báo cáo cạnh EM. Cảnh giác language prior: so sánh với baseline "blind" (không ảnh) như sanity check.

> ### Paper này liên quan gì đến đề tài ViVQA-VLM?
> Định nghĩa **metric chính** (EM, VQA-Acc, ANLS) và **giao thức chuẩn hoá** trong `src/eval/`. Bài học language-prior định hình cách nhóm báo cáo fair (cùng split/seed/prompt) và tránh thổi phồng cải tiến.

---

## P3. Vision-Language Models for Vision Tasks: A Survey

| Trường | Giá trị |
|--------|---------|
| Title | Vision-Language Models for Vision Tasks: A Survey |
| Author(s) | Jingyi Zhang, Jiaxing Huang, Sheng Jin, Shijian Lu |
| Year / Venue | 2024, IEEE TPAMI (arXiv:2304.00685) |
| Code | Repo tổng hợp paper (awesome-list) đi kèm |
| Reviewer | SV4 |

### 1. What problem does the paper solve?
Hệ thống hoá **toàn cảnh VLM**: kiến trúc, mục tiêu pre-training, dataset, và **phương pháp transfer/adaptation** (bao gồm PEFT, prompt tuning, adapter).

### 2. Why is the problem important?
Giúp nhóm định vị lựa chọn baseline, phương pháp cải tiến và protocol đánh giá trong bức tranh chung — tránh "phát minh lại bánh xe".

### 3. What is the research gap?
Tài liệu VLM tản mát; thiếu một taxonomy thống nhất về pretraining objectives và transfer methods.

### 4. What is the proposed method?
Đề xuất **taxonomy**: (a) network/kiến trúc; (b) objectives: contrastive / generative / alignment; (c) datasets; (d) **transfer learning**: prompt tuning, visual prompt, **adapter/LoRA**; (e) knowledge distillation; kèm bảng benchmark.

### 5. What is the key technical idea?
Khung phân loại giúp so sánh phương pháp theo trục dữ liệu, mục tiêu học và cách thích nghi hiệu quả tham số.

### 6. What dataset is used?
Không đề xuất dataset mới; tổng hợp benchmark (image classification, retrieval, VQA, detection…).

### 7. What metrics are used?
Tổng hợp metric theo tác vụ (accuracy, mAP, recall@k, VQA acc…).

### 8. What are the main results?
Chỉ ra xu hướng: pretraining tương phản + generative mạnh; **PEFT/prompt tuning là hướng transfer hiệu quả** khi tài nguyên hạn chế.

### 9. What are the limitations?
Là survey ⇒ không có đóng góp thực nghiệm mới; cập nhật đến thời điểm xuất bản (một số model mới hơn chưa có).

### 10. What can we learn/use for our project? **(BẮT BUỘC)**
Dùng làm **khung tham chiếu** để chọn PEFT phù hợp, thiết kế bảng so sánh phương pháp (Candidate_Methods) và biện luận vì sao PEFT-only là hợp lý cho ngân sách của nhóm.

> ### Paper này liên quan gì đến đề tài ViVQA-VLM?
> Cung cấp **bản đồ phương pháp & đánh giá** để định vị đóng góp của nhóm (PEFT cho VLM tiếng Việt) và chuẩn hoá cách so sánh fair; là nguồn trích dẫn cho phần related work ở Review 1/2.

---

## Tổng hợp cho nhóm

### Giao thức đánh giá đề xuất (chuyển sang `Candidate_Methods.md` / `Project_Plan_15_Weeks.md`)

| Metric | Công thức / chuẩn hóa | Dataset áp dụng | Thư viện / script |
|--------|-----------------------|-----------------|-------------------|
| Exact Match (EM) | Khớp tuyệt đối sau chuẩn hoá NFC + lowercase + bỏ dấu câu cuối | ViVQA, ViTextVQA | `src/eval/em.py` |
| VQA Accuracy | `min(#người trùng / 3, 1)`; ViVQA 1 đáp án ⇒ rút gọn về accuracy có chuẩn hoá | ViVQA | `src/eval/vqa_acc.py` |
| ANLS | `1 − NL(pred,gt)` nếu NLD ≤ 0.5, ngược lại 0; trung bình trên câu hỏi | ViTextVQA | `src/eval/anls.py` |
| Hallucination (POPE-vi) | Probing yes/no; báo cáo **F1 + yes-ratio** (random/popular/adversarial) | Subset có annotation object | `src/eval/pope_vi.py` |
| BLEU / CIDEr (phụ) | Chỉ báo cáo cho câu trả lời sinh mở | Câu trả lời mở | `src/eval/gen_metrics.py` |

### Chuẩn hóa câu trả lời tiếng Việt trước khi chấm
- [x] Chuẩn hoá Unicode **NFC** (thống nhất dấu thanh), lowercase, bỏ dấu câu cuối
- [x] Chuẩn hoá **số ↔ chữ số** ("hai" ↔ "2"), bỏ khoảng trắng thừa
- [x] Danh sách từ đồng nghĩa/biến thể (mapping thủ công nhỏ) để giảm EM sai do cách viết

### Hàng đóng góp vào `Literature_Matrix.md`
- [x] Đã chép 3 hàng (SV4): POPE (#8), VQAv2 (bổ sung), Survey Zhang (bổ sung)
