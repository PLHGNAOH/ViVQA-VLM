# Nội dung chuyển từ báo cáo Review 1 sang Review 2 (W07, 08/10/2026)

## Vì sao
Báo cáo Review 1 (`docs/W03/index_TV.html`, `index_TA.html`) chỉ nên chứa nội dung tuần 3 cộng phần sửa theo góp ý của **reviewer Review 1**. Hai lần sửa sau đó đưa vào kết quả W4–W6 và góp ý của **GVHD (07/10)**, tức là nội dung của mốc Review 2 (tuần 7). Để giữ đúng quy trình:

- Cả hai file được đưa về bản **74fa721** (06/10, "cập nhật theo góp ý reviewer"). Bản EN được dịch lại cho khớp từng dòng với bản VN.
- Tag `review1-report` đánh dấu bản báo cáo Review 1 chính thức.
- Không xoá lịch sử: toàn bộ nội dung bị gỡ vẫn lấy lại được từ git (bảng dưới).

## Nội dung đã gỡ khỏi Review 1 → dùng cho Review 2

| Nội dung | Lấy lại từ | Ghi chú khi đưa vào Review 2 |
|---|---|---|
| Số liệu W4–W6: baseline W5 (ViVQA EM 0,225 / ANLS 0,288; ViTextVQA EM 0,285 / ANLS 0,468 / F1 0,530), chẩn đoán fp16 (23 → 0 câu hỏng), ablation 512 vs 1.280 token, smoke QLoRA, bằng chứng lỗi W5 (dấu, code-switching, hallucination), ViTextVQA chính thức (35.159 / 5.155 / 10.028), rủi ro T4/bf16 | `git show 480aa60:docs/W03/index_TV.html` · bản EN: `git show 148c09c:docs/W03/index_TA.html` | Số baseline W5 (fp16, T4) chỉ là **B0, tham khảo**. Mốc chính thức là **B1** (bf16, 512 token, 4-bit NF4, L4, notebook 02). |
| Hình 6.A — quy trình xây dựng ViVQA + điểm nhiễu N1–N4 | `git show e9acd3c:docs/W03/index_TV.html` (Phần 06) | Slide phải bằng tiếng Anh → dịch nhãn. |
| Hình 7.A — kiến trúc Qwen2.5-VL-3B-Instruct + vị trí LoRA (vẽ lại từ Figure 1, Bai et al., 2025; thông số từ `config.json`) | như trên (Phần 07) | Sửa chú thích "adapter ≈ 39 MB": trọng số LoRA = 7.372.800 × 4 byte (fp32) ≈ **29,5 MB**; 39 MB nhiều khả năng là cả thư mục (kèm tokenizer/processor) → kiểm tra `adapter_model.safetensors` trên Drive rồi ghi đúng. |
| Hình 7.B — giao thức đánh giá | như trên (Phần 07) | Dịch nhãn. |
| Bảng góp ý GVHD 07/10 và cách nhóm xử lý | như trên (Phần 14) | Đưa vào phần "Response to feedback" của Review 2. |

## Việc còn lại cho Review 2
1. Xuất 3 sơ đồ (bản EN) vào `docs/04_slides/review2/figures/`.
2. Bảng kết quả: B1 (chính thức) + B0 (tham khảo, giải thích lý do chạy lại) + B2 QLoRA sơ bộ.
3. Checklist Review 1 (AI_Capstone_Review123) ghi góp ý của reviewer R1 và cách đã sửa trong bản `review1-report`.
