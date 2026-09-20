# Hypotheses — ViVQA-VLM

> **Canonical (chốt W03, 2026-09-17).** Hệ đánh số duy nhất: **H1–H8**.
> Scaffold cũ **H1–H4** *khác nghĩa* (vd. scaffold H2 = OCR ≠ weekly/canonical H2 = 4-bit vs 16-bit) **không còn dùng số**.
> Bảng map: `docs/W03/RQ_Hypothesis_Reconciliation.md`.
> Mỗi giả thuyết phải **falsifiable**: điều kiện bác bỏ rõ, metric, thí nghiệm kiểm chứng.
> Owner: cả nhóm (SV3 soạn các giả thuyết PEFT) · Cập nhật: **2026-09-17**

## 1. Giả thuyết chính (phạm vi Capstone)

| # | Hypothesis | RQ | Cơ sở từ literature | Cách kiểm chứng | Điều kiện bác bỏ | Owner |
|---|------------|----|---------------------|-----------------|------------------|-------|
| H1 | QLoRA (NF4, 4-bit) trên VLM baseline cải thiện EM/VQA-Acc trên ViVQA **≥ +5 điểm** so với zero-shot cùng prompt. | RQ2 | LoRA ≈ full FT (Hu 2022); LLaVA-1.5 LoRA ≈ full FT; QLoRA NF4 ≈ 16-bit (Dettmers 2023) | W08–W10: train QLoRA trên train split, eval test split, 3 seed | Δ < +2 điểm hoặc không ổn định qua seed | SV3 |
| H2 | QLoRA 4-bit đạt **≥ 97%** hiệu năng của LoRA 16-bit (cùng r, cùng targets) trong khi giảm VRAM đỉnh ≥ 40%. | RQ2c | QLoRA Table (NF4+DQ ≈ BF16 trong ±1 điểm) | W09: chạy cặp 4-bit / 16-bit trên A100, log VRAM + thời gian | Chênh > 3% hoặc VRAM không giảm | SV3 |
| H3 | LoRA trên **tất cả linear layer** của LLM tốt hơn attention-only ở cùng ngân sách tham số. | RQ2a | QLoRA §4 (Figure: all-linear cần thiết để khớp full FT) | W08 ablation A2 | Attention-only ≥ all-linear | SV3 |
| H4 | Hiệu năng bão hòa ở **r ≤ 16**; tăng r lên 64 không cải thiện > 1 điểm. | RQ2b | LoRA §7.2 (r=1 đủ cho Wq+Wv) | W09 ablation A3 | r=64 hơn r=16 > 1 điểm nhất quán | SV3 |
| H5 | PEFT adaptation **không làm tăng** hallucination (POPE-vi F1 không giảm) so với baseline zero-shot. | RQ4 | LLaVA-1.5 Model Zoo: LoRA POPE 86.4–86.7 > full FT 85.9 | W12 với SV4 | POPE-vi giảm > 2 điểm F1 | SV3, SV4 |
| H6 | OCR-enhanced prompting tăng ANLS trên ViTextVQA ≥ +3 điểm, và cộng hưởng (không triệt tiêu) với QLoRA. | RQ3 | LLaVA-1.5 (OCR-VQA/TextCaps data giúp TextVQA); ViTextVQA paper | W08–W11 ablation A7 | Δ ANLS < +1 hoặc âm khi kết hợp | SV2, SV3 |
| H7 | Response-format prompt tiếng Việt ("Trả lời bằng một từ hoặc cụm từ ngắn.") tăng EM zero-shot ≥ +3 điểm mà không cần huấn luyện. | RQ2e | LLaVA-1.5 Table 2 (format prompt: MME 1197 → 1323.8) | W05 khi reproduce baseline | Δ EM < +1 | SV3, SV4 |
| H8 | Adapter chung cho ViVQA + ViTextVQA ≥ adapter riêng trên tập nhỏ hơn (ViTextVQA). | RQ2d | VL-Adapter: Single Adapter NLVR2 74.2 vs Multiple 69.8 | W09 ablation A5 | Adapter riêng hơn > 1 điểm | SV3 |

**Giả thuyết trọng tâm cho Review 1:** H1 (gắn RQ2). Checklist nhóm chốt ngày vẫn mở — xem mục 3.

> Ngưỡng số (+5, 97%, +3, …) **kế thừa từ weekly W02**, chưa đo trên Vietnamese VQA. Không đổi ngưỡng ở W03. Sau zero-shot W4–W5, nhóm có thể hiệu chỉnh — ghi changelog.

## 2. Future / out-of-scope hypotheses (không kiểm chứng trong Capstone)

Các mục dưới đây từng là **stub H9–H11** trong weekly W02. Chúng **không** thuộc hệ H1–H8, **không** được kiểm chứng trong phạm vi Capstone 15 tuần, và **không** giữ số H9–H11 trên slide Review 1.

| Ý (cũ) | Nội dung stub (nguyên weekly) | Vì sao out-of-scope | Ghi chú reconciliation |
|--------|-------------------------------|---------------------|------------------------|
| (cũ H9) | _…_ (baseline: ví dụ Qwen2.5-VL zero-shot > BLIP-2 zero-shot trên ViVQA do tokenizer đa ngữ) | Stub; chưa có điều kiện bác bỏ / metric / thí nghiệm. So sánh zero-shot 3 backbone đã nằm trong **RQ1** (đo, không cần giả thuyết đánh số riêng). | Không nâng thành H9 trong-scope. |
| (cũ H10) | _…_ (dataset/OCR) | Stub rỗng; OCR đã có **H6** (falsifiable). | Không giữ số. |
| (cũ H11) | _…_ (evaluation) | Stub rỗng; hallucination đã có **H5** + **RQ4a**. | Không giữ số. |

Nếu sau Faculty Review nhóm muốn thêm giả thuyết mới: đặt **H12+** (không tái sử dụng H9–H11) và viết changelog.

## 3. Checklist

- [x] Mỗi H (H1–H8) có điều kiện bác bỏ định lượng
- [x] Mỗi H gắn ≥ 1 ablation trong `Candidate_Methods.md` (A1–A9)
- [ ] Nhóm chốt H1 là giả thuyết trọng tâm cho Review 1 (ngày: ____)
- [x] Stub H9–H11 không còn trạng thái treo lửng (chuyển mục 2, out-of-scope)

## Changelog

| Ngày | Thay đổi |
|------|----------|
| 2026-09-17 (W03) | Chốt canonical **H1–H8**. Scaffold H1–H4 khác nghĩa không còn dùng số. Stub H9–H11 → mục "Future / out-of-scope"; **không** kiểm chứng trong Capstone. |
| 2026-09-16 (W02 closeout) | Hợp nhất weekly H1–H8 vào file này; stub H9–H11 còn treo. |
