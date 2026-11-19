# SV2 Paper Review — Dataset / OCR

> **Đề tài:** ViVQA-VLM — Efficient Vietnamese Visual Question Answering using Vision-Language Models
> **Vai trò:** SV2 — Dataset / OCR (ViVQA, ViTextVQA, scene-text, PaddleOCR / VietOCR, thách thức tiếng Việt)
> **Tuần:** W01 · **Ngày:** 2026-09-14 · **Reviewer:** SV2
> **Yêu cầu:** ≥ 3 paper, ưu tiên: paper ViVQA (Tran et al., PACLIC 2021), paper ViTextVQA (Nguyen et al., 2024/2025), 1 paper về OCR tiếng Việt hoặc scene-text VQA (TextVQA / ST-VQA / OCR-VQA). Mỗi paper trả lời **đủ 10 câu hỏi** + mục **"Paper này liên quan gì đến đề tài ViVQA-VLM?"**. Cấm chỉ tóm tắt abstract.

## 0. Danh sách paper đã chọn & lý do

| # | Paper | Venue / Năm | Nhóm | Vì sao chọn |
|---|-------|-------------|------|-------------|
| P1 | ViVQA: Vietnamese Visual Question Answering (Tran et al.) | PACLIC 2021 | Benchmark general VQA tiếng Việt | Dataset general VQA tiếng Việt đầu tiên & phổ biến nhất; nguồn split để cố định cho nhóm. |
| P2 | ViTextVQA (Nguyen et al.) | arXiv 2024 / ESWA 2025 | Benchmark scene-text VQA tiếng Việt | Dataset scene-text tiếng Việt quy mô lớn; dùng ANLS; động lực cho OCR-enhanced prompting. |
| P3 | Towards VQA Models That Can Read — TextVQA + LoRRA (Singh et al.) | CVPR 2019 | Scene-text VQA (nền tảng OCR) | Paper nền tảng về VQA cần đọc chữ; giới thiệu cơ chế OCR + copy — bản thiết kế cho OCR-prompting tiếng Việt. |

> **Nguồn số liệu:** trích từ paper gốc và trang dataset chính thức. Con số quy mô có thể chênh nhẹ giữa các bản (arXiv vs journal) — đánh dấu "≈"; số chính xác sẽ chốt khi tải dataset ở W02.

---

## P1. ViVQA: Vietnamese Visual Question Answering

| Trường | Giá trị |
|--------|---------|
| Title | ViVQA: Vietnamese Visual Question Answering |
| Author(s) | Khanh Quoc Tran, An Trong Nguyen, An Tran-Hoai Le, Kiet Van Nguyen |
| Year / Venue | 2021, PACLIC 35 |
| Data / Code | Bộ dữ liệu ViVQA công bố kèm paper (dựa trên ảnh COCO) |
| Reviewer | SV2 |

### 1. What problem does the paper solve?
Thiếu **dataset VQA cho tiếng Việt**. Paper xây bộ ViVQA và baseline để mở đường cho nghiên cứu VQA tiếng Việt.

### 2. Why is the problem important?
VQA gần như chỉ tồn tại cho tiếng Anh; tiếng Việt là ngôn ngữ ít tài nguyên. Một benchmark chuẩn là điều kiện tiên quyết để đo và cải tiến — chính là dataset general mà ViVQA-VLM dùng cho RQ1/RQ2.

### 3. What is the research gap?
Chưa có benchmark VQA tiếng Việt; chưa rõ mô hình vision-language hoạt động ra sao với đặc thù ngôn ngữ tiếng Việt.

### 4. What is the proposed method?
Xây dataset bằng **dịch (bán tự động) tập câu hỏi–trả lời từ dữ liệu VQA tiếng Anh trên ảnh COCO** sang tiếng Việt kèm hậu kiểm; đề xuất mô hình baseline kết hợp **CNN (đặc trưng ảnh) + LSTM/attention (câu hỏi)** với word embedding tiếng Việt (PhoW2V/fastText).

### 5. What is the key technical idea?
Kết hợp embedding tiếng Việt chất lượng + cơ chế co-attention giữa ảnh và câu hỏi để dự đoán câu trả lời (phân loại trên tập answer).

### 6. What dataset is used?
ViVQA: **15.000 cặp câu hỏi–trả lời** trên **10.328 ảnh** (nguồn COCO; lưu ý paper gốc ghi 5.000 ảnh — số ảnh không nhất quán giữa các nguồn), chia train/test tỉ lệ **8:2**. (Split 10.328/3.441 hay gặp ở các nghiên cứu downstream nhưng KHÔNG khớp tổng 15.000 — **số chính xác sẽ chốt khi tải dataset ở W02**.) Loại câu hỏi chủ yếu: **object, number, color, location** — là **general VQA, không phải scene-text**.

### 7. What metrics are used?
**Accuracy** (khớp câu trả lời) — cơ sở cho EM / VQA-Accuracy trong đề tài.

### 8. What are the main results?
Mô hình có attention + embedding tiếng Việt tốt vượt baseline đơn giản; accuracy ở mức trung bình, cho thấy dư địa cải thiện lớn (đặc biệt với VLM hiện đại).

### 9. What are the limitations?
- **Nhiễu do dịch máy** (câu hỏi/câu trả lời dịch chưa tự nhiên).
- Chỉ **general VQA**, không có text-in-image ⇒ không đo được scene-text.
- Quy mô vừa; câu trả lời ngắn/đóng.

### 10. What can we learn/use for our project? **(BẮT BUỘC)**
ViVQA là **dataset general chính** cho RQ1/RQ2. Cần **cố định split + seed (42)**, chuẩn hoá câu trả lời (NFC, lowercase) trước khi chấm EM/VQA-Acc. Lưu ý nhiễu dịch để không quy toàn bộ lỗi cho model.

> ### Paper này liên quan gì đến đề tài ViVQA-VLM?
> Đây là **benchmark general VQA tiếng Việt** để đo baseline và cải tiến PEFT. Split ViVQA sẽ được nhóm cố định; metric EM/VQA-Acc áp cho dataset này; thách thức tiếng Việt (dấu thanh, từ đồng nghĩa) xuất hiện ngay ở khâu chuẩn hoá câu trả lời.

---

## P2. ViTextVQA: A Large-Scale Vietnamese Text-based Visual Question Answering Dataset

| Trường | Giá trị |
|--------|---------|
| Title | ViTextVQA: A Large-Scale Vietnamese Text-based Visual Question Answering Dataset (bản mở rộng trên Expert Systems with Applications) |
| Author(s) | Nguyen et al. (UIT NLP@UIT) |
| Year / Venue | arXiv 2024 (arXiv:2404.10652) · ESWA 2025 |
| Data / Code | Dataset ViTextVQA công bố kèm paper |
| Reviewer | SV2 |

### 1. What problem does the paper solve?
Thiếu benchmark **scene-text VQA** cho tiếng Việt — câu hỏi cần **đọc chữ trong ảnh** (biển hiệu, bao bì, poster) bằng tiếng Việt.

### 2. Why is the problem important?
Scene-text tiếng Việt có **dấu thanh, font đa dạng, layout phức tạp**; đây là gap (G2) rõ nhất của Vietnamese VQA và là nơi OCR + VLM phải phối hợp.

### 3. What is the research gap?
Các dataset scene-text VQA (TextVQA/ST-VQA) đều tiếng Anh; tiếng Việt chưa có benchmark quy mô lớn có text-in-image + đánh giá ANLS.

### 4. What is the proposed method?
Thu thập ảnh chứa chữ tiếng Việt, gán câu hỏi–trả lời yêu cầu đọc chữ; kiểm định chất lượng annotation; benchmark nhiều phương pháp (kể cả OCR-based và VLM), phân tích ảnh hưởng của **thứ tự token OCR** và chất lượng OCR.

### 5. What is the key technical idea?
Chỉ ra rằng **chất lượng và thứ tự OCR quyết định lớn** đến hiệu năng scene-text VQA tiếng Việt; cung cấp bộ dữ liệu + protocol đánh giá (ANLS).

### 6. What dataset is used?
ViTextVQA: quy mô lớn — **≈ 16.000+ ảnh** và **≈ 50.000 cặp QA** (con số chính xác chốt khi tải). Câu hỏi đòi hỏi text reading là chủ đạo.

### 7. What metrics are used?
**ANLS** (Average Normalized Levenshtein Similarity, ngưỡng 0.5) là metric chính cho scene-text; kèm EM/Accuracy.

### 8. What are the main results?
Các mô hình gặp khó; hiệu năng phụ thuộc mạnh vào OCR; khoảng cách lớn so với human ⇒ nhiều dư địa cho OCR-enhanced prompting.

### 9. What are the limitations?
- Phụ thuộc **nhiễu OCR**; dấu thanh & font khó.
- Layout phức tạp (nhiều vùng chữ) khó cho model độ phân giải thấp.

### 10. What can we learn/use for our project? **(BẮT BUỘC)**
ViTextVQA là **dataset scene-text chính** cho RQ3; dùng **ANLS**. Kết luận "thứ tự & chất lượng OCR quan trọng" ⇒ thiết kế **OCR-enhanced prompting** (chèn text OCR có sắp xếp theo layout vào prompt) và chọn OCR engine kỹ (PaddleOCR vs VietOCR).

> ### Paper này liên quan gì đến đề tài ViVQA-VLM?
> Là **benchmark scene-text tiếng Việt** để kiểm chứng cải tiến OCR-prompting (H6) và đo ANLS. Thách thức dấu thanh/scene-text chính là gap G2; dataset này quyết định lựa chọn OCR engine và định dạng lưu OCR (`data/ocr/`).

---

## P3. Towards VQA Models That Can Read (TextVQA + LoRRA)

| Trường | Giá trị |
|--------|---------|
| Title | Towards VQA Models That Can Read |
| Author(s) | Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, Marcus Rohrbach (Facebook AI) |
| Year / Venue | 2019, CVPR (arXiv:1904.08920) |
| Data / Code | TextVQA dataset; MMF framework |
| Reviewer | SV2 |

### 1. What problem does the paper solve?
Các mô hình VQA **không đọc được chữ trong ảnh** — trong khi nhiều câu hỏi thực tế cần đọc text (biển hiệu, nhãn hàng).

### 2. Why is the problem important?
Text-in-image là loại câu hỏi phổ biến và khó; giải được nó mở rộng VQA sang ứng dụng thực (đúng bối cảnh ViTextVQA tiếng Việt).

### 3. What is the research gap?
Trước đó không có dataset VQA tập trung text-in-image và không có cơ chế đưa **OCR token** vào mô hình VQA.

### 4. What is the proposed method?
Dataset **TextVQA** + mô hình **LoRRA (Look, Read, Reason & Answer)**: thêm **module OCR** và **cơ chế copy** cho phép mô hình chọn câu trả lời từ **OCR token trong ảnh** hoặc từ answer vocabulary cố định.

### 5. What is the key technical idea?
**Copy mechanism / dynamic pointer** vào tập OCR token: câu trả lời có thể là một chuỗi text đọc được trong ảnh, không nằm trong từ vựng cố định.

### 6. What dataset is used?
TextVQA: **28.408 ảnh** (OpenImages) với **45.336 câu hỏi** cần đọc chữ.

### 7. What metrics are used?
VQA accuracy (soft, giống VQAv2). Dòng nghiên cứu này (ST-VQA) sau đó đưa ra **ANLS**.

### 8. What are the main results?
LoRRA cải thiện rõ so với mô hình VQA không đọc được chữ, nhưng vẫn kém xa con người ⇒ OCR + reasoning còn khó.

### 9. What are the limitations?
- **Tiếng Anh**; phụ thuộc chất lượng OCR.
- Copy theo token, khó với chuỗi dài/nhiều vùng chữ.

### 10. What can we learn/use for our project? **(BẮT BUỘC)**
Là **bản thiết kế gốc cho OCR-enhanced VQA**: đưa OCR token vào input. Ta hiện thực hoá bằng **OCR-prompting** (chèn text từ PaddleOCR/VietOCR vào prompt của VLM) thay vì copy-mechanism — phù hợp kiến trúc VLM hiện đại và ràng buộc PEFT-only.

> ### Paper này liên quan gì đến đề tài ViVQA-VLM?
> Cung cấp **nguyên lý OCR-enhanced VQA** và di sản metric ANLS cho scene-text. Ta chuyển ý tưởng sang tiếng Việt: OCR (PaddleOCR/VietOCR) → chèn text vào prompt → đo ANLS trên ViTextVQA (RQ3, ablation A6).

---

## Tổng hợp cho nhóm

### Bảng thống kê nhanh dataset (chuyển sang `Dataset_Analysis.md`)

| Dataset | #Ảnh | #QA | Split (train/test) | Loại câu hỏi | Metric chính | Licence | Link |
|---------|------|-----|--------------------|--------------|--------------|---------|------|
| ViVQA | 10.328 (paper gốc ghi 5.000 — không nhất quán) | 15.000 | 8:2 (chốt ở W02) | object, number, color, location (general) | EM / VQA-Acc | Nghiên cứu (theo paper) | Công bố kèm PACLIC 2021 |
| ViTextVQA | ≈ 16.000+ | ≈ 50.000 | chốt khi tải | text reading (scene-text) | ANLS | Nghiên cứu (theo paper) | arXiv:2404.10652 |

> Con số "≈" theo paper; **số chính xác + phân bố loại câu hỏi sẽ đo và chốt ở W02** khi tải dataset (đây là deliverable W02, không thuộc W1).

### Lựa chọn OCR

| Công cụ | Hỗ trợ dấu tiếng Việt | Tốc độ | Độ chính xác (định tính) | Ghi chú |
|---------|----------------------|--------|--------------------------|---------|
| PaddleOCR | Có (model đa ngữ + detection) | Nhanh (GPU) | Tốt cho scene-text/layout; xuất bbox + confidence | Có cả detection + recognition; dễ lấy thứ tự vùng chữ theo layout |
| VietOCR | **Rất tốt** (transformer/attention chuyên tiếng Việt) | Trung bình | Mạnh với chữ in/viết tay tiếng Việt có dấu | Chủ yếu recognition (cần detector rời, vd của PaddleOCR/CRAFT) |

**Đề xuất SV2:** dùng **PaddleOCR làm detector + recognizer scene-text (lấy bbox/thứ tự)**, đối chứng **VietOCR ở khâu recognition dòng chữ tiếng Việt có dấu**; **CER/WER trên 50 ảnh mẫu sẽ đo ở W02** (pilot) để chốt engine — không điền số bịa ở W1.

### Hàng đóng góp vào `Literature_Matrix.md`
- [x] Đã chép 3 hàng (SV2): ViVQA (#4), ViTextVQA (#5), TextVQA/LoRRA (bổ sung)
