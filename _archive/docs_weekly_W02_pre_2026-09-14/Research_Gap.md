# Research Gap — ViVQA-VLM

> Owner: cả nhóm (mỗi SV cung cấp evidence từ paper mình review) · Cập nhật: _YYYY-MM-DD_
> Nguyên tắc: **gap phải có evidence từ literature**, không sao chép nguyên gap trong đề cương; chỉ rõ gap nào **khả thi** trong 15 tuần với ràng buộc PEFT-only.

## 1. Bảng gap

| # | Gap | Evidence từ paper (trích dẫn cụ thể) | SV cung cấp | Khả thi trong Capstone? | Cải tiến tương ứng |
|---|-----|----------------------------------------|-------------|-------------------------|--------------------|
| G1 | **English-centric pretraining** — alignment ảnh–ngôn ngữ tối ưu cho tiếng Anh | LLaVA-1.5 (Liu et al., 2024): đa ngôn ngữ chỉ là hiện tượng *nổi lên* từ ShareGPT, chưa được đo, còn lỗi ở ngôn ngữ khác; LoRA/QLoRA/VL-Adapter đều chỉ đánh giá tiếng Anh. _SV1 bổ sung evidence từ BLIP-2 / Qwen2.5-VL._ | SV1, SV3 | Một phần (đo + adapt bằng PEFT) | LoRA/QLoRA + dữ liệu Việt |
| G2 | **Vietnamese VQA yếu, nhất là scene-text** (OCR, dấu thanh, code-switching) | VL-Adapter: 224 px, không scene-text; LLaVA-1.5: TextVQA tiếng Anh, độ phân giải là đòn bẩy. _SV2 bổ sung evidence từ ViVQA / ViTextVQA._ | SV2, SV3 | Có | OCR-enhanced prompting, độ phân giải |
| G3 | **PEFT cho VLM tiếng Việt chưa được khai thác** | LoRA (Hu 2022) & QLoRA (Dettmers 2023): chỉ LLM văn bản tiếng Anh; VL-Adapter (Sung 2022): PEFT trên VQA nhưng mô hình nhỏ, tiếng Anh; LLaVA-1.5: LoRA ≈ full FT nhưng chưa ai kiểm chứng cho ngôn ngữ ít tài nguyên có dấu thanh. | SV3 | **Có — gap trọng tâm** | QLoRA NF4 all-linear trên Qwen2.5-VL / BLIP-2 / Florence-2 |
| G4 | **Hallucination** trong Vietnamese VQA | LLaVA-1.5 §5.2: hallucination giảm khi tăng độ phân giải; LoRA có POPE cao hơn full FT. _SV4 bổ sung evidence từ POPE._ | SV4, SV3 | Một phần (đo + phân tích) | POPE-vi, phân tích lỗi |

## 2. Gap nào nhóm CHỌN làm trọng tâm?

- **Trọng tâm:** G3 (PEFT cho VLM tiếng Việt) — vì: có evidence rõ từ 4 paper SV3; nằm trọn trong ràng buộc PEFT-only; đo được bằng EM/VQA-Acc/ANLS so với baseline.
- **Phụ:** G2 (scene-text + OCR-prompting) và G4 (hallucination) — làm trục phân tích/ablation.
- **Không làm:** _…_ (ghi rõ gap loại bỏ và lý do)

## 3. Câu phát biểu gap (1 đoạn, dùng trong Report/Review 1)

> _…_ (Ví dụ khung: "Mặc dù LoRA/QLoRA đã chứng minh ngang full fine-tuning cho LLM và VLM tiếng Anh [Hu 2022; Dettmers 2023; Liu 2024], và PEFT đã được benchmark trên VQA tiếng Anh [Sung 2022], **chưa có nghiên cứu nào kiểm chứng PEFT-only adaptation của VLM hiện đại cho Vietnamese VQA — đặc biệt scene-text có dấu thanh — dưới ngân sách GPU đơn.**")

## 4. Checklist
- [ ] Mỗi gap có ≥ 2 trích dẫn paper
- [ ] Đã đánh dấu khả thi / không khả thi
- [ ] Gap trọng tâm được cả nhóm đồng thuận (ghi ngày)
