# Glossary — ViVQA-VLM

> **Sổ tay thuật ngữ** cho đồ án *Efficient Vietnamese Visual Question Answering using Vision-Language Models*.
> Mục đích: tra nhanh khi đọc câu trả lời / paper / thesis, và ôn trước defense (buổi bảo vệ).
> Quy ước: **giữ nguyên thuật ngữ tiếng Anh** (vì bạn sẽ gặp lại đúng từ đó trong paper và thesis) + giải thích tiếng Việt dễ hiểu.
>
> *Tài liệu sống — thêm từ mới mỗi tuần. Cập nhật lần cuối: W4.*

---

## 1. Bài toán & dữ liệu

| Thuật ngữ (EN) | Nghĩa dễ hiểu | Vai trò trong ViVQA-VLM |
|---|---|---|
| **VQA** (Visual Question Answering) | Hỏi–đáp dựa trên ảnh: đưa ảnh + câu hỏi → nhận câu trả lời | Chính là bài toán của đề tài |
| **VLM** (Vision-Language Model) | Model AI *vừa nhìn ảnh vừa hiểu chữ* rồi trả lời | Qwen3-VL là một VLM |
| **ViVQA** | Bộ dữ liệu VQA tiếng Việt, câu hỏi *chung* về ảnh | Dataset chính (general) |
| **ViTextVQA** | Bộ dữ liệu VQA tiếng Việt cần *đọc chữ trong ảnh* | Dataset chính (scene-text) |
| **Scene-text / text-in-image** | Chữ nằm *bên trong* ảnh (biển hiệu, nhãn, bảng) | Điểm yếu lớn của VLM với tiếng Việt |
| **OCR** (Optical Character Recognition) | Công nghệ *đọc chữ trong ảnh* thành text | PaddleOCR / VietOCR, dùng cho câu hỏi text-in-image |
| **Diacritics** (dấu thanh) | Dấu tiếng Việt: sắc, huyền, hỏi, ngã, nặng | Nguồn lỗi hay gặp khi model đọc/viết tiếng Việt |
| **Code-switching** | Trộn lẫn hai ngôn ngữ trong một câu (Việt–Anh) | Thách thức đặc thù tiếng Việt |

## 2. Kiến trúc model

| Thuật ngữ (EN) | Nghĩa dễ hiểu | Vai trò trong ViVQA-VLM |
|---|---|---|
| **Vision Encoder** | Bộ phận *đọc ảnh* → biến ảnh thành số (visual token) | Module 1 của Qwen3-VL (dùng SigLIP-2); **đóng băng** khi PEFT |
| **VL Merger** (vision–language merger) | Bộ phận *nối* thông tin ảnh với phần ngôn ngữ, và nén cho gọn | Module 2 của Qwen3-VL; **đóng băng** khi PEFT |
| **LLM decoder** (Large Language Model) | Bộ phận *hiểu chữ và sinh câu trả lời* | Module 3 của Qwen3-VL; **nơi gắn LoRA** |
| **Visual token** | Ảnh được cắt thành nhiều mẩu số cho model đọc | Ảnh càng to → càng nhiều token → càng tốn bộ nhớ (VRAM) |
| **Dynamic resolution** | Xử lý ảnh theo *tỉ lệ gốc*, số token co giãn theo kích thước | Nút thắt VRAM chính; là một biến ablation tự nhiên |
| **DeepStack** | Kỹ thuật đưa thông tin ảnh từ *nhiều tầng* vào phần ngôn ngữ | Giúp Qwen3-VL đọc chữ trong ảnh tốt hơn (hợp scene-text) |
| **SigLIP-2** | Một loại vision encoder có sẵn, chất lượng cao | Bản 4B dùng SigLIP2-Large (300M) |

## 3. Huấn luyện & thích nghi (adaptation)

| Thuật ngữ (EN) | Nghĩa dễ hiểu | Vai trò trong ViVQA-VLM |
|---|---|---|
| **Fine-tuning** | *Dạy thêm* cho model bằng dữ liệu của mình | Dạy Qwen3-VL quen tiếng Việt hơn |
| **Full fine-tuning** | Dạy lại *toàn bộ* model — rất tốn tài nguyên | ❌ **BỊ CẤM** trong đề tài |
| **PEFT** (Parameter-Efficient Fine-Tuning) | Chỉ chỉnh *một phần nhỏ* model → nhẹ, chạy được GPU yếu | ✅ Ràng buộc bắt buộc của đề tài |
| **LoRA** (Low-Rank Adaptation) | Một kỹ thuật PEFT: gắn thêm "miếng vá" nhỏ để học | Gắn vào LLM decoder của Qwen3-VL |
| **QLoRA** | LoRA + nén model (quantization) cho đỡ tốn bộ nhớ | Phương án chính để train trên Colab |
| **Quantization** (4-bit / 8-bit) | Nén con số trong model xuống ít bit hơn → nhẹ hơn | Giúp model lớn vừa VRAM nhỏ |
| **Adapter weights** | "Miếng vá" LoRA đã học xong (file nhỏ ~vài trăm MB) | Sản phẩm nộp trong reproducibility package |
| **Prompt engineering** | *Cách đặt câu lệnh/câu hỏi* cho model để nó trả lời tốt hơn | Một hướng cải tiến hợp lệ |
| **RAG** (Retrieval-Augmented Generation) | Cho model *tra thêm tài liệu ngoài* trước khi trả lời | Một hướng cải tiến hợp lệ (tùy chọn) |

## 4. Chạy & suy luận

| Thuật ngữ (EN) | Nghĩa dễ hiểu | Vai trò trong ViVQA-VLM |
|---|---|---|
| **Zero-shot** | Cho model làm bài *luôn*, không train thêm gì | Cách thử baseline ở W4 |
| **Inference** | Lúc model *chạy để trả lời* (khác lúc train) | Bấm nút → chờ → ra câu trả lời |
| **Wall-clock (time)** | Thời gian *đồng hồ thật* phải chờ | AI viết code không rút ngắn được phần này |
| **Checkpoint** | Bản lưu model tại một thời điểm train | Lưu adapter về Drive để dùng lại |
| **Hallucination** | Model trả lời *trôi chảy nhưng sai*, "chém" thứ không có trong ảnh | Một trong bốn research gap; phải phân tích |

## 5. Đánh giá & tái lập (cốt lõi — hội đồng hay hỏi)

| Thuật ngữ (EN) | Nghĩa dễ hiểu | Vai trò trong ViVQA-VLM |
|---|---|---|
| **Baseline** | *Mốc tham chiếu* ban đầu, chưa cải tiến gì | Mọi cải tiến phải so với nó |
| **EM** (Exact Match) | Câu trả lời khớp *tuyệt đối* với đáp án chuẩn → tính đúng | Metric chính |
| **VQA Accuracy** | Thước đo độ chính xác chuẩn của VQA | Metric chính |
| **ANLS** (Average Normalized Levenshtein Similarity) | Đo độ *gần giống* chuỗi, tha thứ sai lệch nhỏ | Metric chính, hợp scene-text (ViTextVQA) |
| **Ablation** (ablation study) | Thí nghiệm *tắt/bật từng thành phần* để biết phần nào tạo cải thiện | ✅ Bắt buộc — bằng chứng cải tiến đến từ đâu |
| **Error analysis** | Phân tích *model sai ở đâu, sai kiểu gì* | ✅ Bắt buộc — theo loại câu hỏi |
| **Split** (train/val/test) | Chia dữ liệu thành phần học / phần dò / phần thi | Cố định để công bằng; seed 42 |
| **Seed** (random seed) | Con số cố định để *chạy lại ra kết quả y hệt* | Seed 42, ghi vào mọi log |
| **Reproducibility** | Khả năng *người khác chạy lại ra kết quả giống mình* | Nguyên tắc xuyên suốt đề tài |
| **BLEU / CIDEr** | Metric cho câu sinh dài | ⚠️ Chỉ là chỉ số phụ, KHÔNG dùng làm chính |

## 6. Sản phẩm & công cụ

| Thuật ngữ (EN) | Nghĩa dễ hiểu | Vai trò trong ViVQA-VLM |
|---|---|---|
| **Prototype** | Bản demo *chạy được* để trình diễn, chưa cần hoàn hảo | Web: tải ảnh → hỏi tiếng Việt → nhận câu trả lời (W14) |
| **Colab** (Google Colab) | Dịch vụ *mượn GPU* trên trình duyệt | Nơi chạy phần cần GPU |
| **GPU / VRAM** | Card đồ họa / bộ nhớ của card | Tài nguyên khan hiếm; quyết định model chạy được hay không |
| **`transformers`** (thư viện) | Thư viện Python để *nạp và chạy* model | Qwen3-VL cần bản `>= 4.57.0` |
| **Gradio** | Thư viện dựng *giao diện web demo* nhanh | Dùng cho prototype |

## 7. Học thuật & quy trình Capstone

| Thuật ngữ (EN) | Nghĩa dễ hiểu | Vai trò trong ViVQA-VLM |
|---|---|---|
| **Thesis** | *Luận văn* — tài liệu viết dài trình bày toàn bộ nghiên cứu (vấn đề, phương pháp, kết quả, kết luận) | Sản phẩm viết chính, bằng tiếng Anh |
| **Capstone (project)** | *Đồ án tốt nghiệp* tổng hợp mọi kiến thức đã học | Chính là môn/đề tài này (AIP491) |
| **Defense** (thesis defense) | *Buổi bảo vệ*: trình bày trước hội đồng và trả lời câu hỏi | W15 — chặng cuối |
| **Supervisor** | *Giảng viên hướng dẫn* | Assoc.Prof. Đặng Ngọc Minh Đức |
| **Review** (Review 1 / Review 2) | *Buổi rà soát tiến độ* giữa kỳ, kiểm tra nhóm đi đúng hướng | W3 (Review 1), W7 (Review 2) |
| **Faculty review** | *Hội đồng cấp Khoa* — mốc kiểm tra lớn, phần lớn nghiên cứu phải xong | W13 — mốc chính |
| **Milestone** | *Cột mốc bắt buộc* trong lịch, không được dời | W3 · W7 · W13 · W15 |
| **Deliverable** | *Sản phẩm phải nộp* ở mỗi giai đoạn (báo cáo, code, kết quả) | Có cả deliverable cá nhân và nhóm |
| **Report (R1–R7)** | Các *báo cáo* theo khuôn mẫu, ghép dần thành thesis | R1 Intro … R4 Methodology … R6–7 Results |
| **Literature review** | *Tổng quan tài liệu*: đọc & tổng hợp các paper đã có trước | Việc chính của W1–W2 |
| **Research gap** | *Khoảng trống nghiên cứu* — chỗ chưa ai làm tốt | 4 gap của đề tài (G1–G4); phải có evidence |
| **Research question (RQ)** | *Câu hỏi nghiên cứu* mà đồ án đặt ra để trả lời | Định hướng thí nghiệm |
| **Hypothesis** | *Giả thuyết* — điều mình dự đoán, cần thí nghiệm kiểm chứng | Ví dụ: "thêm OCR sẽ tăng ANLS" |
| **Contribution / Novelty** | *Đóng góp mới* — điểm khác biệt so với cái đã có | Bắt buộc ≥ 1 cải tiến kiểm chứng được |
| **Methodology** | *Phương pháp* — chương mô tả *cách làm* (data, model, cách đánh giá) | Report 4 |
| **Evaluation protocol** | *Giao thức đánh giá* — quy tắc chấm điểm thống nhất | Ghi rõ split, seed, phần cứng, thời gian |
| **Preliminary results** | *Kết quả sơ bộ* ban đầu, chưa đầy đủ | Cần trình ở Review 2 (W7) |
| **Human evaluation** | *Đánh giá bởi người* thay vì chỉ bằng máy | Phân tích thêm (nếu khả thi) |
| **Reproducibility package** | *Gói tái lập*: code + adapter + config + log để người khác chạy lại | Bắt buộc nộp cuối kỳ |
| **CLO** (Course Learning Outcome) | *Chuẩn đầu ra môn học* — kỹ năng phải đạt | Đồ án map vào CLO2–CLO10 |

---

### Cách dùng file này
- Mỗi khi gặp thuật ngữ mới trong lúc làm việc, thêm một dòng vào đúng nhóm.
- Trước defense: đọc lướt cả bảng — nhớ kỹ **baseline** và **ablation** (hai từ "linh hồn" của đồ án).
- Khi viết thesis tiếng Anh: có thể tạo bản *English-only* (bỏ cột tiếng Việt) để đưa vào phần *List of Terms / Abbreviations*.
