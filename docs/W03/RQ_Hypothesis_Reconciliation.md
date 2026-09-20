# RQ / Hypothesis Reconciliation — W03

> Canonical = **weekly**: RQ1–RQ4 (+ RQ2a–e, RQ3a, RQ4a); H1–H8.
> Scaffold cũ (`_archive/W02_scaffold_pre_merge_2026-09-16/`): RQ0–RQ3; H1–H4 **khác nghĩa**.
> Ngày: 2026-09-17. Không bịa; ô không chắc → ⚠️REVIEW.

Nguồn scaffold: `_archive/W02_scaffold_pre_merge_2026-09-16/{Research_Questions,Hypotheses}.md`
Nguồn weekly (đã là nội dung `docs/W02/` sau W02 closeout): bản H1–H8 / RQ1–RQ4 hiện hành.

## 1. Map Research Questions

| Ý scaffold (cũ) | Nội dung scaffold | → Canonical | Ghi chú |
|-----------------|-------------------|-------------|---------|
| RQ0 (draft, primary) | PEFT-only (LoRA/QLoRA) cải thiện Vietnamese VQA bao nhiêu so với frozen zero-shot, đo EM / VQA-Acc (ViVQA) và ANLS (ViTextVQA), cùng split/seed/hardware | **RQ2** | Scaffold gọi đây là RQ *chính*; weekly đặt zero-shot 3-backbone thành RQ1 và PEFT thành RQ2. Cùng ý PEFT-vs-zero-shot. |
| RQ1 (draft, secondary) | OCR-enhanced prompting cải thiện ANLS trên ViTextVQA vs no-OCR? | **RQ3** | **Không** map vào canonical RQ1 (RQ1 weekly = zero-shot 3 VLM). |
| RQ2 (draft, secondary) | LoRA target modules / rank / quantization (4-bit vs 8-bit) trade-off accuracy vs efficiency | **RQ2a + RQ2b + RQ2c** | Scaffold gộp 3 ablation thành 1 RQ. Weekly tách. 8-bit nằm RQ2c; scaffold không nêu 16-bit — weekly RQ2c có 16-bit (không mâu thuẫn, chỉ giàu hơn). |
| RQ3 (draft, secondary) | PEFT có đổi hallucination (POPE-vi) so với zero-shot? | **RQ4** | Khớp H5. |

### Ý weekly không có chỗ trong scaffold

| Ý weekly (canonical) | Có trong scaffold? | Xử lý |
|----------------------|--------------------|--------|
| RQ1 — zero-shot 3 VLM (BLIP-2 / Florence-2 / Qwen2.5-VL) trên ViVQA + ViTextVQA | Không | **Giữ** — đây là RQ đo G1/G2; scaffold chỉ có TODO "fix baseline model". |
| RQ2d — adapter chung vs riêng | Không (gần VL-Adapter, không thành RQ) | **Giữ** (weekly). |
| RQ2e — response-format prompt tiếng Việt | Không | **Giữ**. |
| RQ3a — độ phân giải × OCR-prompting | Không | **Giữ**. |
| RQ4a — hallucination theo loại câu hỏi | Không | **Giữ**. |

## 2. Map Hypotheses

| Ý scaffold (cũ) | Nội dung scaffold | → Canonical | Cảnh báo numbering |
|-----------------|-------------------|-------------|---------------------|
| H1 | QLoRA cải thiện VQA-Acc trên ViVQA ≥ X pts vs zero-shot | **H1** | Cùng nghĩa. Weekly chốt ngưỡng **+5** (scaffold còn `TODO: set X`). Giữ +5; không đổi ở W03. |
| H2 | OCR-enhanced prompting cải thiện ANLS trên ViTextVQA vs no-OCR | **H6** | **KHÔNG phải H2.** Canonical H2 = QLoRA 4-bit vs LoRA 16-bit. Trộn số = sai khoa học. |
| H3 | All-linear LoRA targets > attention-only (cùng param budget) | **H3** | Cùng nghĩa (weekly gắn RQ2a / ablation A2). |
| H4 | PEFT does not increase (may reduce) hallucination (POPE-vi F1) | **H5** | **KHÔNG phải H4.** Canonical H4 = bão hòa rank r≤16. |

### Ý weekly không có chỗ trong scaffold H1–H4

| Canonical | Nội dung | Xử lý |
|-----------|----------|--------|
| H2 | QLoRA 4-bit ≥ 97% LoRA 16-bit, VRAM −≥40% | **Giữ** — không có tương đương scaffold. |
| H4 | Rank bão hòa r≤16 | **Giữ**. |
| H7 | Format-prompt tiếng Việt +≥3 EM zero-shot | **Giữ**. |
| H8 | Adapter chung ≥ adapter riêng trên ViTextVQA | **Giữ**. |

## 3. Stub H9–H11

Chuyển **toàn bộ** sang "Future / out-of-scope" trong `docs/W02/Hypotheses.md` mục 2. **Không** kiểm chứng trong Capstone. **Không** dùng số H9–H11 trên slide.

| Stub | Vì sao không nâng thành giả thuyết in-scope |
|------|---------------------------------------------|
| H9 (Qwen2.5-VL zero-shot > BLIP-2 nhờ tokenizer đa ngữ) | Chỉ là ví dụ trong ngoặc; thiếu ngưỡng, thiếu điều kiện bác bỏ, thiếu protocol. Ý so sánh backbone đã cover bởi **RQ1** (câu hỏi đo lường, không phải H). |
| H10 (dataset/OCR) | Rỗng. OCR falsifiable = **H6**. |
| H11 (evaluation) | Rỗng. Eval/hallucination = **H5** + **RQ4a**. |

## 4. Reconciliation notes (ý không có chỗ / xung đột)

1. Scaffold **không** có RQ tương đương canonical RQ1 (zero-shot 3 backbone). Không phải mất mát — weekly bổ sung. Không ⚠️REVIEW.
2. Scaffold H1 ngưỡng `X` vs weekly `+5`: đã chọn weekly. Nguồn +5 = draft weekly, **không** phải số đo trên ViVQA. Giữ nguyên; hiệu chỉnh sau W5 nếu nhóm muốn (ghi changelog). Không tự đổi.
3. Scaffold ghi "recommended Qwen2.5-VL-3B" trong RQ0 TODO — **không** map thành giả thuyết. Baseline = `docs/W03/Baseline_Decision.md` (đề xuất, chưa CHỐT).
4. Không phát hiện xung đột nghĩa *còn lại* mà không map được. Ô ⚠️REVIEW: **không có** ở bảng RQ/H.

## 5. Hệ đánh số sau W03 (duy nhất)

| Dùng | Không dùng |
|------|------------|
| RQ1, RQ2, RQ2a–e, RQ3, RQ3a, RQ4, RQ4a | RQ0; scaffold RQ1–RQ3 (nghĩa cũ) |
| H1–H8 | Scaffold H1–H4 (nghĩa cũ); H9–H11 |

File sống: `docs/W02/Research_Questions.md`, `docs/W02/Hypotheses.md`.
