# W02 DIFF REPORT — `docs/W02/` (canonical, MỚI) vs `docs/weekly/W02/` (CŨ)

Ngày: 2026-09-16. Quy tắc: không sửa `docs/W02/`. Unified diff: `git diff --no-index` với **weekly = old (−)** và **W02 = new (+)**.

Không file nào [IDENTICAL]. SHA256 16 hex:

| File | W02 bytes / sha | weekly bytes / sha |
|------|-----------------|---------------------|
| Literature_Matrix.md | 11320 / B8BF2CE249506BE3 | 3713 / CFEE2F6D631AD38B |
| Dataset_Analysis.md | 9343 / ECD4AD06DF12AC82 | 3407 / 697E01EED60A5F83 |
| Research_Gap.md | 6159 / 07EF343D38D2504A | 3129 / 57CEB9F6A850435B |
| Research_Questions.md | 1173 / AA81B0558D674007 | 2538 / D6B141435E460167 |
| Hypotheses.md | 1041 / 0FAB7C75D523E0BF | 3262 / 9FC51371CCD13902 |
| Candidate_Methods.md | 1309 / CD19DE835AF3B7BF | 4170 / 03842A9D81807057 |

Tóm tắt phân loại:

| File | Phân loại | Lý do ngắn |
|------|-----------|------------|
| Literature_Matrix.md | **[WEEKLY CÓ NỘI DUNG RIÊNG]** | W02 giàu 10 ref chính thức + số đo dataset; weekly có hàng LoRA/QLoRA/VL-Adapter (số liệu thật) + ma trận Gap×Paper, không có trong W02. |
| Dataset_Analysis.md | **[WEEKLY CÓ NỘI DUNG RIÊNG]** | W02 giàu số liệu thật; weekly gần như template rỗng nhưng có schema JSON fine-tune, OCR path, bảng rủi ro, hàng stats/challenge không có bên W02. |
| Research_Gap.md | **[WEEKLY CÓ NỘI DUNG RIÊNG]** | W02 giàu evidence L-id + đo JSON; weekly có cột SV / cải tiến, mục “Không làm”, khung gap statement trích Hu/Dettmers/Sung. |
| Research_Questions.md | **[WEEKLY CÓ NỘI DUNG RIÊNG]** | Weekly dài hơn: RQ1–RQ4 + sub-RQ2a–e, owner, metric. W02 là W3-PREP scaffold. |
| Hypotheses.md | **[WEEKLY CÓ NỘI DUNG RIÊNG]** | Weekly H1–H8 có ngưỡng số + điều kiện bác bỏ; W02 4 draft + `TODO: set X`. |
| Candidate_Methods.md | **[WEEKLY CÓ NỘI DUNG RIÊNG]** | Weekly có recipe PEFT, ablation A1–A9, VRAM, loại soft-prompt; W02 bảng rút gọn + TODO. |

---

## A1. Literature_Matrix.md — [WEEKLY CÓ NỘI DUNG RIÊNG]

W02 = 10 hàng L1–L10 theo master references, verification tags, sizes đo từ JSON, cố ý **loại** LoRA/QLoRA/VL-Adapter. Weekly = bảng tiếng Việt, một phần `_…_`, nhưng hàng #6 #7 #9 #10 **đã điền số**.

### Unified diff

```diff
diff --git a/docs/weekly/W02/Literature_Matrix.md b/docs/W02/Literature_Matrix.md
--- a/docs/weekly/W02/Literature_Matrix.md
+++ b/docs/W02/Literature_Matrix.md
@@ -1,34 +1,58 @@
-# Literature Matrix — ViVQA-VLM
-
-> Mỗi hàng = 1 paper. Cột **"What can we use for our project?"** là bắt buộc. Mỗi SV điền ≥ 3 hàng từ file `SVx_Paper_Review.md` của mình. Cập nhật: _YYYY-MM-DD_
-
-## Bảng tổng
-
-| # | Author / Year / Venue | Problem | Method | Dataset / Metric | Kết quả chính | Limitation | **What can we use for our project?** | SV |
-|---|------------------------|---------|--------|------------------|---------------|------------|--------------------------------------|----|
-| 1 | Li et al. / 2023 / ICML | Vision-language alignment | BLIP-2 (Q-Former) | _…_ | _…_ | _…_ | Ứng viên baseline | SV1 |
-| 2 | Xiao et al. / 2024 / CVPR | Unified vision tasks | Florence-2 | _…_ | _…_ | _…_ | Ứng viên baseline | SV1 |
-| 3 | Bai et al. / 2025 / TR | Open VLM | Qwen2.5-VL | _…_ | _…_ | _…_ | Ứng viên baseline | SV1 |
-| 4 | Tran et al. / 2021 / PACLIC | Vietnamese VQA | ViVQA benchmark | ViVQA / _…_ | _…_ | _…_ | Dataset general | SV2 |
-| 5 | Nguyen et al. / 2024 (ESWA 2025) | Scene-text VQA tiếng Việt | ViTextVQA | ViTextVQA / ANLS | _…_ | _…_ | Dataset scene-text + ANLS | SV2 |
-| 6 | Hu et al. / 2022 / ICLR | Full FT LLM quá tốn bộ nhớ/lưu trữ | LoRA: ΔW = BA hạng thấp, gộp được | GLUE, E2E NLG, WikiSQL, MNLI, SAMSum / Acc, BLEU, ROUGE | GPT-3 175B: LoRA 4.7M tham số ≈/＞ full FT; 10.000× ít tham số, 0 độ trễ | Chỉ LLM tiếng Anh; chọn module heuristic; vẫn cần W₀ FP16 | **Kỹ thuật bắt buộc**; r nhỏ trên nhiều module; gộp trọng số cho demo; rank sweep | SV3 |
-| 7 | Dettmers et al. / 2023 / NeurIPS | Fine-tune 33–65B trên 1 GPU | QLoRA: NF4 + Double Quant + Paged Optimizer, LoRA all-linear | MMLU, Vicuna (GPT-4/human Elo), GLUE, SuperNI | Guanaco-65B 99.3% ChatGPT, 24h/1 GPU 48GB; NF4 ≈ 16-bit | Chưa so full-FT 65B; chậm hơn 16-bit; tiếng Anh, thuần văn bản | **Fit Colab**; cấu hình NF4+DQ; all-linear targets; data quality > size | SV3 |
-| 8 | Li et al. / 2023 / EMNLP | Object hallucination | POPE | _…_ | _…_ | _…_ | Rubric hallucination | SV4 |
-| 9 | Sung et al. / 2022 / CVPR | PEFT cho V&L (VQA, caption) | VL-Adapter: adapter chia sẻ đa task, Compacter, Hyperformer, LoRA, prompt | VQAv2, GQA, NLVR2, COCO / Acc, CIDEr, % params | Single Adapter 4.18% ≈ full FT (77.4 vs 77.6); prompt-tuning yếu (59.0) | Mô hình nhỏ, 224 px, tiếng Anh, không OCR | Freeze vision encoder; adapter chung ViVQA+ViTextVQA; bỏ soft-prompt | SV3 |
-| 10 | Liu et al. / 2024 / CVPR | Recipe VLM mạnh, rẻ, tái lập | LLaVA-1.5 (MLP connector, 336 px, VQA data + format prompt); LoRA r=128 | 12 benchmark incl. VQAv2, TextVQA, POPE, MME | SOTA 11/12; LoRA ≈ full FT ở 7B/13B (POPE cao hơn) | Tiếng Anh; 8×A100; hallucination còn; LoRA chỉ ở Model Zoo | Format prompt tiếng Việt; 2 LR (LoRA/projector); độ phân giải; POPE-vi | SV3 |
-| 11 | _…_ | | | | | | | SV1 |
-| 12 | _…_ | | | | | | | SV2 |
-| 13 | _…_ | | | | | | | SV4 |
-
-## Ma trận Gap × Paper (đánh ✓ nếu paper cung cấp evidence cho gap)
-
-| Gap | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 |
-|-----|----|----|----|----|----|----|----|----|----|-----|
-| G1 English-centric pretraining | | | | | | ✓ | ✓ | | ✓ | ✓ |
-| G2 Vietnamese VQA yếu, scene-text | | | | | | | | | ✓ | ✓ |
-| G3 PEFT cho VLM tiếng Việt chưa có | | | | | | ✓ | ✓ | | ✓ | ✓ |
-| G4 Hallucination | | | | | | | | | | ✓ |
-
-## Ghi chú
-- Reading note chi tiết: `docs/W01/SVx_Paper_Review.md` (10 câu hỏi + "liên quan gì đến ViVQA-VLM").
-- Hàng #6, #7, #9, #10 do SV3 điền từ `SV3_Paper_Review.md`.
+# Literature Matrix — ViVQA-VLM (Week 2)
+… (toàn bộ 10 hàng L1–L10 + rationale + TODO PDF — xem docs/W02/Literature_Matrix.md; không lặp lại ở đây vì canonical giữ nguyên)
```

Phần `+` đầy đủ = toàn bộ file `docs/W02/Literature_Matrix.md` (58 dòng). Diff tool đã thay thế 1:1; không truncate phía weekly (−).

### Trích nguyên văn — chỉ có ở weekly (cân nhắc port)

Hàng PEFT đã điền (W02 cố ý loại khỏi 10 ref; số liệu nằm ở `docs/literature/SV3_Paper_Review.md`):

```
| 6 | Hu et al. / 2022 / ICLR | Full FT LLM quá tốn bộ nhớ/lưu trữ | LoRA: ΔW = BA hạng thấp, gộp được | GLUE, E2E NLG, WikiSQL, MNLI, SAMSum / Acc, BLEU, ROUGE | GPT-3 175B: LoRA 4.7M tham số ≈/＞ full FT; 10.000× ít tham số, 0 độ trễ | Chỉ LLM tiếng Anh; chọn module heuristic; vẫn cần W₀ FP16 | **Kỹ thuật bắt buộc**; r nhỏ trên nhiều module; gộp trọng số cho demo; rank sweep | SV3 |
| 7 | Dettmers et al. / 2023 / NeurIPS | Fine-tune 33–65B trên 1 GPU | QLoRA: NF4 + Double Quant + Paged Optimizer, LoRA all-linear | MMLU, Vicuna (GPT-4/human Elo), GLUE, SuperNI | Guanaco-65B 99.3% ChatGPT, 24h/1 GPU 48GB; NF4 ≈ 16-bit | Chưa so full-FT 65B; chậm hơn 16-bit; tiếng Anh, thuần văn bản | **Fit Colab**; cấu hình NF4+DQ; all-linear targets; data quality > size | SV3 |
| 9 | Sung et al. / 2022 / CVPR | PEFT cho V&L (VQA, caption) | VL-Adapter: adapter chia sẻ đa task, Compacter, Hyperformer, LoRA, prompt | VQAv2, GQA, NLVR2, COCO / Acc, CIDEr, % params | Single Adapter 4.18% ≈ full FT (77.4 vs 77.6); prompt-tuning yếu (59.0) | Mô hình nhỏ, 224 px, tiếng Anh, không OCR | Freeze vision encoder; adapter chung ViVQA+ViTextVQA; bỏ soft-prompt | SV3 |
```

Cột **Limitation** + cột **SV** (không có trên W02). Hàng trống 11–13.

Ma trận Gap × Paper:

```
## Ma trận Gap × Paper (đánh ✓ nếu paper cung cấp evidence cho gap)

| Gap | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 |
|-----|----|----|----|----|----|----|----|----|----|-----|
| G1 English-centric pretraining | | | | | | ✓ | ✓ | | ✓ | ✓ |
| G2 Vietnamese VQA yếu, scene-text | | | | | | | | | ✓ | ✓ |
| G3 PEFT cho VLM tiếng Việt chưa có | | | | | | ✓ | ✓ | | ✓ | ✓ |
| G4 Hallucination | | | | | | | | | | ✓ |
```

```
## Ghi chú
- Reading note chi tiết: `docs/W01/SVx_Paper_Review.md` (10 câu hỏi + "liên quan gì đến ViVQA-VLM").
- Hàng #6, #7, #9, #10 do SV3 điền từ `SV3_Paper_Review.md`.
```

---

## A2. Dataset_Analysis.md — [WEEKLY CÓ NỘI DUNG RIÊNG]

W02 = số liệu thật (seed 42, provenance JSON). Weekly = template SV2, ô trống, nhưng có heading/schema không xuất hiện bên W02.

### Unified diff (toàn bộ file; weekly − / W02 +)

```diff
diff --git a/docs/weekly/W02/Dataset_Analysis.md b/docs/W02/Dataset_Analysis.md
--- a/docs/weekly/W02/Dataset_Analysis.md
+++ b/docs/W02/Dataset_Analysis.md
@@ -1,88 +1,107 @@
-# Dataset Analysis — ViVQA & ViTextVQA
-
-> Owner: **SV2** (Dataset / OCR) · Hỗ trợ: SV4 (metric), SV3 (định dạng dữ liệu fine-tune) · Cập nhật: _YYYY-MM-DD_
-
-## 1. Tổng quan dataset
-
-| Dataset | Nguồn / Paper | Loại | #Ảnh | #QA (train / val / test) | Ngôn ngữ | Licence | Link tải |
-|---------|---------------|------|------|--------------------------|----------|---------|----------|
-| ViVQA | Tran et al., PACLIC 2021 | General VQA | | | vi | | |
-| ViTextVQA | Nguyen et al., 2024/2025 | Scene-text VQA | | | vi | | |
-
-## 2. Thống kê mô tả (điền sau khi tải)
-
-| Chỉ số | ViVQA | ViTextVQA |
-|--------|-------|-----------|
-| Độ dài câu hỏi trung bình (token) | | |
-| Độ dài câu trả lời trung bình (token) | | |
-| % câu trả lời 1 từ / ≤ 3 từ | | |
-| Số câu trả lời duy nhất | | |
-| Kích cỡ ảnh phổ biến (px) | | |
-| % ảnh có chữ (ước lượng bằng OCR) | | |
-
-## 3. Phân loại câu hỏi (mẫu ≥ 200 câu / dataset — cơ sở cho error analysis W12)
-
-| Loại | Định nghĩa | Ví dụ | % ViVQA | % ViTextVQA |
-|------|------------|-------|---------|-------------|
-| Yes/No | | | | |
-| Counting | | | | |
-| Object recognition | | | | |
-| Text reading | | | | |
-| Reasoning | | | | |
-| Khác | | | | |
-
-## 4. Thách thức đặc thù tiếng Việt (kèm ví dụ thực từ dataset)
-
-| Thách thức | Ví dụ | Ảnh hưởng đến metric | Xử lý đề xuất |
-|------------|-------|----------------------|---------------|
-| Dấu thanh / Unicode (NFC vs NFD) | | EM sai dù đúng nghĩa | Chuẩn hóa NFC |
-| Code-switching Anh–Việt | | | |
-| Từ đồng nghĩa / cách viết số | | | |
-| Lỗi dịch máy trong dataset (nếu có) | | | |
-| Chữ trong ảnh nghiêng / mờ / có dấu | | ANLS | OCR đa engine |
-
-## 5. OCR cho ViTextVQA
-
-| Engine | Phiên bản | Hỗ trợ dấu | Tốc độ (ảnh/s, GPU) | CER/WER trên 50 ảnh mẫu | Quyết định |
-|--------|-----------|------------|---------------------|-------------------------|-----------|
-| PaddleOCR | | | | | |
-| VietOCR | | | | | |
-
-Định dạng lưu OCR: `data/ocr/<dataset>/<image_id>.json` gồm `text`, `bbox`, `confidence`.
-
-## 6. Split cố định & seed
-
-| Dataset | Train | Val | Test | Seed | Ghi chú |
-|---------|-------|-----|------|------|---------|
-| ViVQA | | | | 42 | Dùng split gốc nếu có |
-| ViTextVQA | | | | 42 | |
-
-## 7. Định dạng dữ liệu cho fine-tune (thống nhất với SV3)
-
-```json
-{
-  "id": "vivqa_000001",
-  "image": "data/processed/vivqa/000001.jpg",
-  "question": "Có bao nhiêu người trong ảnh?",
-  "answer": "hai",
-  "answers": ["hai", "2"],
-  "question_type": "counting",
-  "ocr_text": "",
-  "source": "ViVQA",
-  "split": "train"
-}
-```
-
-## 8. Rủi ro dữ liệu & mitigation
-
-| Rủi ro | Impact | Mitigation | Owner |
-|--------|--------|-----------|-------|
-| Link tải hỏng / licence hạn chế | High | Liên hệ tác giả; dùng subset công bố | SV2 |
-| Câu trả lời nhiễu | Medium | Bộ lọc + kiểm tra thủ công mẫu | SV2 |
-| OCR kém trên ảnh tiếng Việt | Medium | Đổi engine, ensemble | SV2 |
-
-## 9. Checklist
-- [ ] Tải và kiểm tra checksum 2 dataset
-- [ ] Thống kê mục 2–3
-- [ ] Chọn OCR engine, chạy toàn bộ ViTextVQA
-- [ ] Cố định split + seed, commit `data/splits/`
+# Dataset Analysis — ViVQA & ViTextVQA (Week 2)
+… (toàn bộ nội dung thật §1–§4 = docs/W02/Dataset_Analysis.md)
```

### Trích nguyên văn — chỉ có ở weekly

Hàng thống kê không có bên W02:

```
| Kích cỡ ảnh phổ biến (px) | | |
| % ảnh có chữ (ước lượng bằng OCR) | | |
```

Hàng thách thức không có bên W02:

```
| Từ đồng nghĩa / cách viết số | | | |
| Lỗi dịch máy trong dataset (nếu có) | | | |
```

```
## 5. OCR cho ViTextVQA

| Engine | Phiên bản | Hỗ trợ dấu | Tốc độ (ảnh/s, GPU) | CER/WER trên 50 ảnh mẫu | Quyết định |
|--------|-----------|------------|---------------------|-------------------------|-----------|
| PaddleOCR | | | | | |
| VietOCR | | | | | |

Định dạng lưu OCR: `data/ocr/<dataset>/<image_id>.json` gồm `text`, `bbox`, `confidence`.
```

```
## 7. Định dạng dữ liệu cho fine-tune (thống nhất với SV3)

```json
{
  "id": "vivqa_000001",
  "image": "data/processed/vivqa/000001.jpg",
  "question": "Có bao nhiêu người trong ảnh?",
  "answer": "hai",
  "answers": ["hai", "2"],
  "question_type": "counting",
  "ocr_text": "",
  "source": "ViVQA",
  "split": "train"
}
```
```

```
## 8. Rủi ro dữ liệu & mitigation

| Rủi ro | Impact | Mitigation | Owner |
|--------|--------|-----------|-------|
| Link tải hỏng / licence hạn chế | High | Liên hệ tác giả; dùng subset công bố | SV2 |
| Câu trả lời nhiễu | Medium | Bộ lọc + kiểm tra thủ công mẫu | SV2 |
| OCR kém trên ảnh tiếng Việt | Medium | Đổi engine, ensemble | SV2 |

## 9. Checklist
- [ ] Tải và kiểm tra checksum 2 dataset
- [ ] Thống kê mục 2–3
- [ ] Chọn OCR engine, chạy toàn bộ ViTextVQA
- [ ] Cố định split + seed, commit `data/splits/`
```

(§6 split seed=42 weekly trống; W02 đã ghi blocker val-split trong §4.4.)

---

## A3. Research_Gap.md — [WEEKLY CÓ NỘI DUNG RIÊNG]

W02 viết lại G1–G4 gắn L-id + số JSON + đoạn Review 1 hoàn chỉnh. Weekly bảng ngắn hơn, có owner SV và cột cải tiến.

### Unified diff

```diff
diff --git a/docs/weekly/W02/Research_Gap.md b/docs/W02/Research_Gap.md
--- a/docs/weekly/W02/Research_Gap.md
+++ b/docs/W02/Research_Gap.md
@@ -1,28 +1,59 @@
-# Research Gap — ViVQA-VLM
-
-> Owner: cả nhóm (mỗi SV cung cấp evidence từ paper mình review) · Cập nhật: _YYYY-MM-DD_
-> Nguyên tắc: **gap phải có evidence từ literature**, không sao chép nguyên gap trong đề cương; chỉ rõ gap nào **khả thi** trong 15 tuần với ràng buộc PEFT-only.
-
-## 1. Bảng gap
-
-| # | Gap | Evidence từ paper (trích dẫn cụ thể) | SV cung cấp | Khả thi trong Capstone? | Cải tiến tương ứng |
-|---|-----|----------------------------------------|-------------|-------------------------|--------------------|
-| G1 | **English-centric pretraining** — alignment ảnh–ngôn ngữ tối ưu cho tiếng Anh | LLaVA-1.5 (Liu et al., 2024): đa ngôn ngữ chỉ là hiện tượng *nổi lên* từ ShareGPT, chưa được đo, còn lỗi ở ngôn ngữ khác; LoRA/QLoRA/VL-Adapter đều chỉ đánh giá tiếng Anh. _SV1 bổ sung evidence từ BLIP-2 / Qwen2.5-VL._ | SV1, SV3 | Một phần (đo + adapt bằng PEFT) | LoRA/QLoRA + dữ liệu Việt |
-| G2 | **Vietnamese VQA yếu, nhất là scene-text** (OCR, dấu thanh, code-switching) | VL-Adapter: 224 px, không scene-text; LLaVA-1.5: TextVQA tiếng Anh, độ phân giải là đòn bẩy. _SV2 bổ sung evidence từ ViVQA / ViTextVQA._ | SV2, SV3 | Có | OCR-enhanced prompting, độ phân giải |
-| G3 | **PEFT cho VLM tiếng Việt chưa được khai thác** | LoRA (Hu 2022) & QLoRA (Dettmers 2023): chỉ LLM văn bản tiếng Anh; VL-Adapter (Sung 2022): PEFT trên VQA nhưng mô hình nhỏ, tiếng Anh; LLaVA-1.5: LoRA ≈ full FT nhưng chưa ai kiểm chứng cho ngôn ngữ ít tài nguyên có dấu thanh. | SV3 | **Có — gap trọng tâm** | QLoRA NF4 all-linear trên Qwen2.5-VL / BLIP-2 / Florence-2 |
-| G4 | **Hallucination** trong Vietnamese VQA | LLaVA-1.5 §5.2: hallucination giảm khi tăng độ phân giải; LoRA có POPE cao hơn full FT. _SV4 bổ sung evidence từ POPE._ | SV4, SV3 | Một phần (đo + phân tích) | POPE-vi, phân tích lỗi |
-
-## 2. Gap nào nhóm CHỌN làm trọng tâm?
-
-- **Trọng tâm:** G3 (PEFT cho VLM tiếng Việt) — vì: có evidence rõ từ 4 paper SV3; nằm trọn trong ràng buộc PEFT-only; đo được bằng EM/VQA-Acc/ANLS so với baseline.
-- **Phụ:** G2 (scene-text + OCR-prompting) và G4 (hallucination) — làm trục phân tích/ablation.
-- **Không làm:** _…_ (ghi rõ gap loại bỏ và lý do)
-
-## 3. Câu phát biểu gap (1 đoạn, dùng trong Report/Review 1)
-
-> _…_ (Ví dụ khung: "Mặc dù LoRA/QLoRA đã chứng minh ngang full fine-tuning cho LLM và VLM tiếng Anh [Hu 2022; Dettmers 2023; Liu 2024], và PEFT đã được benchmark trên VQA tiếng Anh [Sung 2022], **chưa có nghiên cứu nào kiểm chứng PEFT-only adaptation của VLM hiện đại cho Vietnamese VQA — đặc biệt scene-text có dấu thanh — dưới ngân sách GPU đơn.**")
-
-## 4. Checklist
-- [ ] Mỗi gap có ≥ 2 trích dẫn paper
-- [ ] Đã đánh dấu khả thi / không khả thi
-- [ ] Gap trọng tâm được cả nhóm đồng thuận (ghi ngày)
+# Research Gap — ViVQA-VLM (Week 2, evidence-linked)
+… (= toàn bộ docs/W02/Research_Gap.md)
```

### Trích nguyên văn — chỉ có ở weekly

Cột **SV cung cấp** và **Cải tiến tương ứng** (W02 không gán owner / không cột method):

```
| G1 | … | SV1, SV3 | Một phần (đo + adapt bằng PEFT) | LoRA/QLoRA + dữ liệu Việt |
| G2 | … | SV2, SV3 | Có | OCR-enhanced prompting, độ phân giải |
| G3 | … | SV3 | **Có — gap trọng tâm** | QLoRA NF4 all-linear trên Qwen2.5-VL / BLIP-2 / Florence-2 |
| G4 | … | SV4, SV3 | Một phần (đo + phân tích) | POPE-vi, phân tích lỗi |
```

```
- **Không làm:** _…_ (ghi rõ gap loại bỏ và lý do)
```

Khung gap statement (placeholder `_…_` + ví dụ; W02 đã viết đoạn hoàn chỉnh khác, **không** nêu Hu 2022 / Dettmers 2023 / Sung 2022 trong câu chính):

```
> _…_ (Ví dụ khung: "Mặc dù LoRA/QLoRA đã chứng minh ngang full fine-tuning cho LLM và VLM tiếng Anh [Hu 2022; Dettmers 2023; Liu 2024], và PEFT đã được benchmark trên VQA tiếng Anh [Sung 2022], **chưa có nghiên cứu nào kiểm chứng PEFT-only adaptation của VLM hiện đại cho Vietnamese VQA — đặc biệt scene-text có dấu thanh — dưới ngân sách GPU đơn.**")
```

```
- [ ] Mỗi gap có ≥ 2 trích dẫn paper
- [ ] Gap trọng tâm được cả nhóm đồng thuận (ghi ngày)
```

---

## A4. Research_Questions.md — [WEEKLY CÓ NỘI DUNG RIÊNG]

W02 tự gắn nhãn **W3-PREP SCAFFOLD**. Weekly có bảng RQ đầy đủ hơn (owner, gap, metric, sub-RQ).

### Unified diff

```diff
diff --git a/docs/weekly/W02/Research_Questions.md b/docs/W02/Research_Questions.md
--- a/docs/weekly/W02/Research_Questions.md
+++ b/docs/W02/Research_Questions.md
@@ -1,35 +1,17 @@
 # Research Questions — ViVQA-VLM
 
-> Owner: cả nhóm · Cập nhật: _YYYY-MM-DD_
-> Mỗi RQ phải: gắn với 1 gap trong `Research_Gap.md`, có metric đo được, có SV chịu trách nhiệm.
+> **[W3-PREP SCAFFOLD — does NOT block Week-2 100%]** Template + TODOs. To be finalized for Review 1 (W3).
+> Constraint of record: **PEFT-only** (LoRA/QLoRA, no full fine-tuning). Metrics: EM / VQA-Acc / ANLS.
 
-## 1. Research question chính
+## Primary RQ (measurable, PEFT-only)
+- **RQ0 (draft):** To what extent can **PEFT-only** adaptation (LoRA/QLoRA) of a modern open VLM improve Vietnamese VQA over a frozen zero-shot baseline, measured by EM / VQA-Acc (ViVQA) and ANLS (ViTextVQA) under identical split/seed/hardware?
+  - [ ] TODO: fix baseline model (recommended Qwen2.5-VL-3B) and datasets/splits.
 
-| # | Research Question | Gap | Metric | Dataset | Owner |
-|---|-------------------|-----|--------|---------|-------|
-| RQ1 | Một VLM mã nguồn mở (BLIP-2 / Florence-2 / Qwen2.5-VL) đạt hiệu năng zero-shot thế nào trên Vietnamese VQA (ViVQA, ViTextVQA)? | G1, G2 | EM, VQA-Acc, ANLS | ViVQA, ViTextVQA | SV1, SV4 |
-| RQ2 | PEFT (LoRA / QLoRA 4-bit) trên VLM đó cải thiện Vietnamese VQA bao nhiêu so với zero-shot và prompt-optimized, **không full fine-tuning**? | G3 | ΔEM, ΔVQA-Acc, ΔANLS; % tham số; VRAM; thời gian train | ViVQA, ViTextVQA | SV3 |
-| RQ3 | OCR-enhanced prompting (PaddleOCR / VietOCR) có cải thiện scene-text VQA tiếng Việt không, và tương tác thế nào với PEFT? | G2 | ANLS, EM theo loại "text reading" | ViTextVQA | SV2, SV3 |
-| RQ4 | PEFT adaptation ảnh hưởng thế nào đến hallucination (object / text) trên ảnh tiếng Việt? | G4 | POPE-vi F1, tỉ lệ hallucination theo loại | Subset thủ công | SV4, SV3 |
+## Secondary RQs
+- **RQ1 (draft):** Does **OCR-enhanced prompting** improve ANLS on ViTextVQA scene-text questions vs. no OCR? `❌ TODO: refine`
+- **RQ2 (draft):** How do **LoRA target modules / rank / quantization (4-bit vs 8-bit)** trade off accuracy vs. efficiency (%trainable params, VRAM, time)? `❌ TODO`
+- **RQ3 (draft):** Does PEFT change **hallucination** (POPE-vi) relative to zero-shot? `❌ TODO`
 
-## 2. Research question phụ (ablation)
-
-| # | Sub-RQ | Xuất phát từ | Owner |
-|---|--------|--------------|-------|
-| RQ2a | LoRA target modules: attention-only vs all-linear? | QLoRA (Dettmers 2023), LoRA (Hu 2022) | SV3 |
-| RQ2b | Rank r ∈ {8,16,32,64} — hiệu năng bão hòa ở đâu? | LoRA §7 | SV3 |
-| RQ2c | 4-bit NF4 vs 8-bit vs 16-bit LoRA: trade-off chất lượng / VRAM / thời gian? | QLoRA | SV3 |
-| RQ2d | Adapter chung cho ViVQA + ViTextVQA vs adapter riêng? | VL-Adapter (Sung 2022) | SV3 |
-| RQ2e | Response-format prompt tiếng Việt cải thiện EM bao nhiêu (trước khi train)? | LLaVA-1.5 (Liu 2024) | SV3, SV4 |
-| RQ3a | Độ phân giải ảnh × OCR-prompting trên ViTextVQA? | LLaVA-1.5 §5.2 | SV2, SV3 |
-| RQ4a | Hallucination phân rã theo loại câu hỏi (yes/no, counting, text, reasoning)? | Master Instruction 7.4 | SV4 |
-
-## 3. Ràng buộc chung cho mọi RQ
-- Cùng baseline đã reproduce, cùng split, cùng seed, cùng hardware protocol.
-- Không full fine-tuning.
-- Báo cáo: dataset split, seed, hardware, thời gian inference **và** thời gian PEFT fine-tuning.
-
-## 4. Checklist
-- [ ] Mỗi RQ có gap, metric, owner
-- [ ] RQ2 (PEFT) là RQ trọng tâm — nhóm xác nhận
-- [ ] Đối chiếu với `Hypotheses.md`
+## Links
+- Evidence gaps: `docs/W02/Research_Gap.md` (G1–G4).
+- Methods: `docs/W02/Candidate_Methods.md`. Hypotheses: `docs/W02/Hypotheses.md`.
```

### Trích nguyên văn — chỉ có ở weekly (toàn bộ bảng RQ)

```
## 1. Research question chính

| # | Research Question | Gap | Metric | Dataset | Owner |
|---|-------------------|-----|--------|---------|-------|
| RQ1 | Một VLM mã nguồn mở (BLIP-2 / Florence-2 / Qwen2.5-VL) đạt hiệu năng zero-shot thế nào trên Vietnamese VQA (ViVQA, ViTextVQA)? | G1, G2 | EM, VQA-Acc, ANLS | ViVQA, ViTextVQA | SV1, SV4 |
| RQ2 | PEFT (LoRA / QLoRA 4-bit) trên VLM đó cải thiện Vietnamese VQA bao nhiêu so với zero-shot và prompt-optimized, **không full fine-tuning**? | G3 | ΔEM, ΔVQA-Acc, ΔANLS; % tham số; VRAM; thời gian train | ViVQA, ViTextVQA | SV3 |
| RQ3 | OCR-enhanced prompting (PaddleOCR / VietOCR) có cải thiện scene-text VQA tiếng Việt không, và tương tác thế nào với PEFT? | G2 | ANLS, EM theo loại "text reading" | ViTextVQA | SV2, SV3 |
| RQ4 | PEFT adaptation ảnh hưởng thế nào đến hallucination (object / text) trên ảnh tiếng Việt? | G4 | POPE-vi F1, tỉ lệ hallucination theo loại | Subset thủ công | SV4, SV3 |

## 2. Research question phụ (ablation)

| # | Sub-RQ | Xuất phát từ | Owner |
|---|--------|--------------|-------|
| RQ2a | LoRA target modules: attention-only vs all-linear? | QLoRA (Dettmers 2023), LoRA (Hu 2022) | SV3 |
| RQ2b | Rank r ∈ {8,16,32,64} — hiệu năng bão hòa ở đâu? | LoRA §7 | SV3 |
| RQ2c | 4-bit NF4 vs 8-bit vs 16-bit LoRA: trade-off chất lượng / VRAM / thời gian? | QLoRA | SV3 |
| RQ2d | Adapter chung cho ViVQA + ViTextVQA vs adapter riêng? | VL-Adapter (Sung 2022) | SV3 |
| RQ2e | Response-format prompt tiếng Việt cải thiện EM bao nhiêu (trước khi train)? | LLaVA-1.5 (Liu 2024) | SV3, SV4 |
| RQ3a | Độ phân giải ảnh × OCR-prompting trên ViTextVQA? | LLaVA-1.5 §5.2 | SV2, SV3 |
| RQ4a | Hallucination phân rã theo loại câu hỏi (yes/no, counting, text, reasoning)? | Master Instruction 7.4 | SV4 |

## 3. Ràng buộc chung cho mọi RQ
- Cùng baseline đã reproduce, cùng split, cùng seed, cùng hardware protocol.
- Không full fine-tuning.
- Báo cáo: dataset split, seed, hardware, thời gian inference **và** thời gian PEFT fine-tuning.

## 4. Checklist
- [ ] Mỗi RQ có gap, metric, owner
- [ ] RQ2 (PEFT) là RQ trọng tâm — nhóm xác nhận
- [ ] Đối chiếu với `Hypotheses.md`
```

**Port gợi ý (W3):** RQ1 zero-shot so sánh 3 backbone không có trong W02 (W02 gộp thành RQ0 PEFT-vs-zeroshot). Sub-RQ2a–e, RQ3a, RQ4a không có bên W02.

---

## A5. Hypotheses.md — [WEEKLY CÓ NỘI DUNG RIÊNG]

### Unified diff

```diff
diff --git a/docs/weekly/W02/Hypotheses.md b/docs/W02/Hypotheses.md
--- a/docs/weekly/W02/Hypotheses.md
+++ b/docs/W02/Hypotheses.md
@@ -1,30 +1,13 @@
 # Hypotheses — ViVQA-VLM
 
-> Owner: cả nhóm (SV3 soạn các giả thuyết PEFT) · Cập nhật: _YYYY-MM-DD_
-> Mỗi giả thuyết phải **falsifiable**: có điều kiện bác bỏ rõ, có metric, có thí nghiệm kiểm chứng.
+> **[W3-PREP SCAFFOLD — does NOT block Week-2 100%]** Template + TODOs. Finalize for Review 1 (W3).
+> Every hypothesis must state the **metric** and the **baseline** it is compared against.
 
-## 1. Giả thuyết chính
+| ID | Hypothesis (draft) | Metric | Baseline / condition | Status |
+|----|--------------------|--------|----------------------|--------|
+| H1 | QLoRA adaptation improves VQA-Accuracy on ViVQA by ≥ X pts vs zero-shot | VQA-Acc / EM | frozen zero-shot, same test split & seed | `❌ TODO: set X` |
+| H2 | OCR-enhanced prompting improves ANLS on ViTextVQA vs no-OCR | ANLS (thr 0.5) | same PEFT model, ±OCR | `❌ TODO` |
+| H3 | All-linear LoRA targets > attention-only at equal param budget | EM/ANLS + %params | attention-only LoRA | `❌ TODO` (evidence: QLoRA review, in-repo) |
+| H4 | PEFT does not increase (may reduce) hallucination | POPE-vi F1 | zero-shot | `❌ TODO` (evidence: L2 LLaVA) |
 
-| # | Hypothesis | RQ | Cơ sở từ literature | Cách kiểm chứng | Điều kiện bác bỏ | Owner |
-|---|------------|----|---------------------|-----------------|------------------|-------|
-| H1 | QLoRA (NF4, 4-bit) trên VLM baseline cải thiện EM/VQA-Acc trên ViVQA **≥ +5 điểm** so với zero-shot cùng prompt. | RQ2 | LoRA ≈ full FT (Hu 2022); LLaVA-1.5 LoRA ≈ full FT; QLoRA NF4 ≈ 16-bit (Dettmers 2023) | W08–W10: train QLoRA trên train split, eval test split, 3 seed | Δ < +2 điểm hoặc không ổn định qua seed | SV3 |
-| H2 | QLoRA 4-bit đạt **≥ 97%** hiệu năng của LoRA 16-bit (cùng r, cùng targets) trong khi giảm VRAM đỉnh ≥ 40%. | RQ2c | QLoRA Table (NF4+DQ ≈ BF16 trong ±1 điểm) | W09: chạy cặp 4-bit / 16-bit trên A100, log VRAM + thời gian | Chênh > 3% hoặc VRAM không giảm | SV3 |
-| H3 | LoRA trên **tất cả linear layer** của LLM tốt hơn attention-only ở cùng ngân sách tham số. | RQ2a | QLoRA §4 (Figure: all-linear cần thiết để khớp full FT) | W08 ablation A2 | Attention-only ≥ all-linear | SV3 |
-| H4 | Hiệu năng bão hòa ở **r ≤ 16**; tăng r lên 64 không cải thiện > 1 điểm. | RQ2b | LoRA §7.2 (r=1 đủ cho Wq+Wv) | W09 ablation A3 | r=64 hơn r=16 > 1 điểm nhất quán | SV3 |
-| H5 | PEFT adaptation **không làm tăng** hallucination (POPE-vi F1 không giảm) so với baseline zero-shot. | RQ4 | LLaVA-1.5 Model Zoo: LoRA POPE 86.4–86.7 > full FT 85.9 | W12 với SV4 | POPE-vi giảm > 2 điểm F1 | SV3, SV4 |
-| H6 | OCR-enhanced prompting tăng ANLS trên ViTextVQA ≥ +3 điểm, và cộng hưởng (không triệt tiêu) với QLoRA. | RQ3 | LLaVA-1.5 (OCR-VQA/TextCaps data giúp TextVQA); ViTextVQA paper | W08–W11 ablation A7 | Δ ANLS < +1 hoặc âm khi kết hợp | SV2, SV3 |
-| H7 | Response-format prompt tiếng Việt ("Trả lời bằng một từ hoặc cụm từ ngắn.") tăng EM zero-shot ≥ +3 điểm mà không cần huấn luyện. | RQ2e | LLaVA-1.5 Table 2 (format prompt: MME 1197 → 1323.8) | W05 khi reproduce baseline | Δ EM < +1 | SV3, SV4 |
-| H8 | Adapter chung cho ViVQA + ViTextVQA ≥ adapter riêng trên tập nhỏ hơn (ViTextVQA). | RQ2d | VL-Adapter: Single Adapter NLVR2 74.2 vs Multiple 69.8 | W09 ablation A5 | Adapter riêng hơn > 1 điểm | SV3 |
-
-## 2. Giả thuyết do SV1 / SV2 / SV4 bổ sung
-
-| # | Hypothesis | RQ | Cơ sở | Kiểm chứng | Bác bỏ | Owner |
-|---|------------|----|-------|------------|--------|-------|
-| H9 | _…_ (baseline: ví dụ Qwen2.5-VL zero-shot > BLIP-2 zero-shot trên ViVQA do tokenizer đa ngữ) | RQ1 | | | | SV1 |
-| H10 | _…_ (dataset/OCR) | RQ3 | | | | SV2 |
-| H11 | _…_ (evaluation) | RQ4 | | | | SV4 |
-
-## 3. Checklist
-- [ ] Mỗi H có điều kiện bác bỏ định lượng
-- [ ] Mỗi H gắn ≥ 1 ablation trong `Candidate_Methods.md`
-- [ ] Nhóm chốt H1 là giả thuyết trọng tâm cho Review 1 (W03)
+- [ ] TODO: turn each into a falsifiable statement with a threshold once baseline zero-shot numbers exist (W4–W5).
```

### Trích nguyên văn — chỉ có ở weekly

```
## 1. Giả thuyết chính

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

## 2. Giả thuyết do SV1 / SV2 / SV4 bổ sung

| # | Hypothesis | RQ | Cơ sở | Kiểm chứng | Bác bỏ | Owner |
|---|------------|----|-------|------------|--------|-------|
| H9 | _…_ (baseline: ví dụ Qwen2.5-VL zero-shot > BLIP-2 zero-shot trên ViVQA do tokenizer đa ngữ) | RQ1 | | | | SV1 |
| H10 | _…_ (dataset/OCR) | RQ3 | | | | SV2 |
| H11 | _…_ (evaluation) | RQ4 | | | | SV4 |

## 3. Checklist
- [ ] Mỗi H có điều kiện bác bỏ định lượng
- [ ] Mỗi H gắn ≥ 1 ablation trong `Candidate_Methods.md`
- [ ] Nhóm chốt H1 là giả thuyết trọng tâm cho Review 1 (W03)
```

W02 không có H2 (4-bit vs 16-bit), H4 (rank saturation), H6 ngưỡng +3 ANLS, H7 format-prompt, H8 adapter chung — đây là phần port quan trọng nhất sang W3.

---

## A6. Candidate_Methods.md — [WEEKLY CÓ NỘI DUNG RIÊNG]

### Unified diff

```diff
diff --git a/docs/weekly/W02/Candidate_Methods.md b/docs/W02/Candidate_Methods.md
--- a/docs/weekly/W02/Candidate_Methods.md
+++ b/docs/W02/Candidate_Methods.md
@@ -1,66 +1,23 @@
 # Candidate Methods — ViVQA-VLM
 
-> Owner: SV3 (PEFT) · SV1 (baseline) · SV2 (OCR) · SV4 (eval/prototype) · Cập nhật: _YYYY-MM-DD_
-> Ràng buộc: **PEFT-only** (LoRA / QLoRA 4-bit hoặc 8-bit). Không full fine-tuning. Mọi cải tiến so với **đúng 1** baseline đã reproduce.
-
-## 1. Baseline ứng viên (SV1 chọn ĐÚNG 1 ở W03–W04)
-
-| Model | Kích cỡ | Interface | Ưu | Nhược | VRAM 4-bit (ước lượng) | Trạng thái |
-|-------|---------|-----------|----|-------|------------------------|-----------|
-| BLIP-2 | ~3.9B (OPT-2.7B) / ~12B (FlanT5-XXL) | Q-Former | Nhẹ, nhiều tài liệu | LLM tiếng Anh, yếu scene-text | | _…_ |
-| Florence-2 | 0.23B / 0.77B | seq2seq, task prompt | Rất nhẹ, OCR nội tại | Tiếng Việt hạn chế, câu trả lời ngắn | | _…_ |
-| Qwen2.5-VL | 3B / 7B | ViT + merger + Qwen2.5 LLM | Đa ngữ, OCR mạnh, dynamic resolution | VRAM (7B) | 3B ≈ 2–2.5 GB, 7B ≈ 4.5–5.5 GB trọng số | _…_ |
-
-## 2. Danh mục cải tiến (theo Master Instruction 4.2) — xếp ưu tiên
-
-| Ưu tiên | Phương pháp | Mô tả ngắn | Xuất phát từ paper | Cần train? | Owner | Tuần |
-|---------|-------------|------------|--------------------|-----------|-------|------|
-| 1 | **QLoRA fine-tuning** (NF4 + DQ, LoRA all-linear trên LLM, freeze ViT) | Cải tiến trọng tâm | Hu 2022; Dettmers 2023; Liu 2024 | Có | SV3 | W08–W09 |
-| 2 | **Response-format prompt tiếng Việt** | Prompt optimization, không train | Liu 2024 | Không | SV3, SV4 | W05 |
-| 3 | **OCR-enhanced prompting** (PaddleOCR / VietOCR → chèn text vào prompt) | Multimodal prompt engineering | ViTextVQA; Liu 2024 | Không | SV2, SV3 | W08 |
-| 4 | **Adapter chung đa dataset** (ViVQA + ViTextVQA) | Lightweight VLM adaptation | Sung 2022 | Có | SV3 | W09 |
-| 5 | Retrieval-Augmented VQA (RAG) — few-shot ví dụ tương tự qua CLIP embedding | Tùy chọn nếu còn thời gian | — | Không | SV3 | W09+ |
-| — | Soft prompt-tuning | **Loại** — yếu trên V&L (Sung 2022: 59.0 vs 77.6) | Sung 2022 | — | — | — |
-
-## 3. Recipe PEFT xuất phát (chi tiết trong `SV3_Paper_Review.md` §5.2)
-
-| Tham số | Giá trị xuất phát | Ablation |
-|---------|-------------------|----------|
-| Quantization | NF4 + double quant, compute BF16 (FP16 trên T4) | 4-bit / 8-bit / 16-bit |
-| LoRA r / α / dropout | 16 / 32 / 0.05 | r ∈ {8,16,32,64} |
-| Target modules | q,k,v,o,gate,up,down (all-linear LLM) | attention-only |
-| Vision tower | Freeze | + LoRA trên ViT (ViTextVQA) |
-| Projector / merger | Freeze (ablation: mở, LR 2e-5) | mở / đóng |
-| LR / optimizer | 2e-4, paged AdamW 32-bit, max_grad_norm 0.3 | 1e-4 |
-| Epoch / batch | 1–3 / 2 × grad-accum 8 | — |
-| Seed | 42 (+ 2 seed phụ cho kết quả chính) | — |
-
-## 4. Ma trận ablation (bắt buộc — Master Instruction 7.3)
-
-| # | Ablation | H | Owner | Tuần |
-|---|----------|---|-------|------|
-| A1 | Zero-shot vs prompt-optimized vs LoRA/QLoRA | H1, H7 | SV3 | W08–W10 |
-| A2 | Target modules attention-only vs all-linear | H3 | SV3 | W08 |
-| A3 | Rank sweep | H4 | SV3 | W09 |
-| A4 | 4-bit vs 8-bit vs 16-bit LoRA (VRAM, thời gian) | H2 | SV3 | W09 |
-| A5 | Adapter chung vs riêng | H8 | SV3 | W09 |
-| A6 | Có vs không OCR-enhanced prompting | H6 | SV2, SV3 | W08–W11 |
-| A7 | Có vs không retrieval augmentation (nếu làm) | — | SV3 | W09+ |
-| A8 | Backbone khác (nếu tài nguyên cho phép) | — | SV1 | W10 |
-| A9 | Hallucination trước / sau PEFT | H5 | SV4, SV3 | W12 |
-
-## 5. Công cụ / thư viện
-
-| Thành phần | Thư viện | Phiên bản (pin ở W04) |
-|------------|----------|------------------------|
-| Model | `transformers`, `accelerate` | |
-| PEFT | `peft` (LoRA), `bitsandbytes` (NF4/8-bit) | |
-| Train | `trl` SFTTrainer hoặc HF `Trainer` | |
-| OCR | `paddleocr`, `vietocr` | |
-| Eval | `src/eval/` (EM, VQA-Acc, ANLS, POPE-vi) | |
-| Prototype | Gradio / Streamlit | |
-
-## 6. Checklist
-- [ ] Baseline chốt (SV1) — ngày: ____
-- [ ] Recipe PEFT chốt (SV3) — ngày: ____
-- [ ] Ma trận ablation gắn với `Hypotheses.md`
+> **[W3-PREP SCAFFOLD — does NOT block Week-2 100%]** …
+## Baseline candidates … Qwen2.5-VL-3B recommended …
+## Improvement directions (4 bullets + RAG optional)
+## Ablation grid (draft) — one-line, no A1–A9 IDs
```

### Trích nguyên văn — chỉ có ở weekly

VRAM ước lượng (W02 không có số GB):

```
| Qwen2.5-VL | 3B / 7B | ViT + merger + Qwen2.5 LLM | Đa ngữ, OCR mạnh, dynamic resolution | VRAM (7B) | 3B ≈ 2–2.5 GB, 7B ≈ 4.5–5.5 GB trọng số | _…_ |
```

Loại soft-prompt + adapter chung + RAG ưu tiên:

```
| 4 | **Adapter chung đa dataset** (ViVQA + ViTextVQA) | Lightweight VLM adaptation | Sung 2022 | Có | SV3 | W09 |
| 5 | Retrieval-Augmented VQA (RAG) — few-shot ví dụ tương tự qua CLIP embedding | Tùy chọn nếu còn thời gian | — | Không | SV3 | W09+ |
| — | Soft prompt-tuning | **Loại** — yếu trên V&L (Sung 2022: 59.0 vs 77.6) | Sung 2022 | — | — | — |
```

Recipe PEFT:

```
## 3. Recipe PEFT xuất phát (chi tiết trong `SV3_Paper_Review.md` §5.2)

| Tham số | Giá trị xuất phát | Ablation |
|---------|-------------------|----------|
| Quantization | NF4 + double quant, compute BF16 (FP16 trên T4) | 4-bit / 8-bit / 16-bit |
| LoRA r / α / dropout | 16 / 32 / 0.05 | r ∈ {8,16,32,64} |
| Target modules | q,k,v,o,gate,up,down (all-linear LLM) | attention-only |
| Vision tower | Freeze | + LoRA trên ViT (ViTextVQA) |
| Projector / merger | Freeze (ablation: mở, LR 2e-5) | mở / đóng |
| LR / optimizer | 2e-4, paged AdamW 32-bit, max_grad_norm 0.3 | 1e-4 |
| Epoch / batch | 1–3 / 2 × grad-accum 8 | — |
| Seed | 42 (+ 2 seed phụ cho kết quả chính) | — |
```

Ablation A1–A9 (W02 chỉ 1 dòng gộp):

```
| A1 | Zero-shot vs prompt-optimized vs LoRA/QLoRA | H1, H7 | SV3 | W08–W10 |
| A2 | Target modules attention-only vs all-linear | H3 | SV3 | W08 |
| A3 | Rank sweep | H4 | SV3 | W09 |
| A4 | 4-bit vs 8-bit vs 16-bit LoRA (VRAM, thời gian) | H2 | SV3 | W09 |
| A5 | Adapter chung vs riêng | H8 | SV3 | W09 |
| A6 | Có vs không OCR-enhanced prompting | H6 | SV2, SV3 | W08–W11 |
| A7 | Có vs không retrieval augmentation (nếu làm) | — | SV3 | W09+ |
| A8 | Backbone khác (nếu tài nguyên cho phép) | — | SV1 | W10 |
| A9 | Hallucination trước / sau PEFT | H5 | SV4, SV3 | W12 |
```

```
## 5. Công cụ / thư viện
| Model | `transformers`, `accelerate` |
| PEFT | `peft` (LoRA), `bitsandbytes` (NF4/8-bit) |
| Train | `trl` SFTTrainer hoặc HF `Trainer` |
| OCR | `paddleocr`, `vietocr` |
| Eval | `src/eval/` (EM, VQA-Acc, ANLS, POPE-vi) |
| Prototype | Gradio / Streamlit |
```

---

## B. FILE CHỈ CÓ Ở `docs/weekly/W02/` (không có trong `docs/W02/`)

| File | STUB / thật | Tóm tắt 2 dòng |
|------|-------------|----------------|
| `SV1_Paper_Review.md` | **STUB** | Template 10 câu hỏi, mọi ô `_…_`. Reviewer = SV1 — VLM/Baseline. |
| `SV2_Paper_Review.md` | **STUB** | Cùng template; Reviewer = SV2 — Dataset/OCR. Không số liệu/paper. |
| `SV3_Paper_Review.md` | **STUB** | Cùng template; Reviewer = SV3 — PEFT. Bản review thật nằm `docs/literature/SV3_Paper_Review.md` + `docs/weekly/W01/`. |
| `SV4_Paper_Review.md` | **STUB** | Cùng template; Reviewer = SV4 — Evaluation. |
| `tracker.md` | **STUB** | Tiêu đề “Week 2 — Literature & dataset”; task SV1–4 trống; “Milestone: none”. |
| `W02_Team_Report.md` | **STUB** | Khung Weekly_Report_Template: mục tiêu copy README, per-SV `_…_`, metric trống, ngày nộp `_YYYY-MM-DD_`. Không phải báo cáo đã điền. |
| `README.md` | **Thật (playbook tuần)** | Objective, deliverable list, task theo SV, DoD, ràng buộc PEFT. Checklist chưa tick. Không phải kết quả thí nghiệm. |
| `WEEK_02_Literature_and_Dataset.txt` | **Thật (brief/spec tuần 2)** | Mục tiêu đóng gói literature + đụng data thật; yêu cầu ≥12 paper, thống kê, ≥50 mẫu/nhãn, chốt 1 baseline; deliverable tên `.txt`. Không chứa số liệu chạy. |

Không còn file weekly/W02 nào khác ngoài 6 file trùng tên + 8 file trên (14 file).

---

## C. W2 COMPLETENESS

Milestone đề tài W2: **Literature matrix + Dataset analysis**.

### C1. `docs/W02/` — 6 deliverable

| File | Heading H1/H2 | Nội dung thật? |
|------|---------------|----------------|
| Literature_Matrix.md | `# Literature Matrix`; `## Matrix`; `## What-we-can-use rationale`; `## TODO — papers still unread` | **Thật.** 10 hàng L1–L10, cột đủ, sizes ViVQA/ViTextVQA đo JSON, metric paper gần như `❌ TODO` (chỉ L2 `⚠️ secondary`). Không stub. |
| Dataset_Analysis.md | `# Dataset Analysis`; `## 1. Overview`; `## 2. Question-type distribution`; `## 3. Vietnamese-specific challenges` (3.1–3.4); `## 4. Preprocessing plan` | **Thật.** Số từ JSON seed 42; sample path thật; disclaimer heuristic. Không stub. |
| Research_Gap.md | `# Research Gap`; `## Gaps` G1–G4; `## Focus decision`; `## One-paragraph gap statement`; `## Checklist` | **Thật.** Evidence L-id + JSON; feasibility Yes/Partial; focus G3. PDF cite còn TODO. |
| Research_Questions.md | `# Research Questions`; `## Primary RQ`; `## Secondary RQs`; `## Links` | **Scaffold W3-PREP** (tự đánh dấu). RQ0–RQ3 draft, nhiều `❌ TODO`. Không chặn DoD W2. |
| Hypotheses.md | `# Hypotheses` + 1 bảng H1–H4 | **Scaffold W3-PREP.** H1 chưa có ngưỡng X. |
| Candidate_Methods.md | `# Candidate Methods`; `## Baseline candidates`; `## Improvement directions`; `## Ablation grid` | **Scaffold W3-PREP.** Qwen2.5-VL-3B recommended; config `❌ TODO`. |

### C2. Báo cáo tổng kết tuần 2

| Ứng viên | Có? |
|----------|-----|
| `docs/W02/W02_Team_Report.md` | **Không** |
| `WEEK2_STATUS.md` (root) | **Có — nội dung thật.** Self-audit DoD D1–D6, 90%, reproduce `python src/data/analyze_dataset.py`, blocker PDF + val split. Đây là báo cáo tuần 2 đang dùng. |
| `docs/weekly/W02/W02_Team_Report.md` | Stub template (sẽ archive cùng thư mục weekly). |

### C3. `experiments/W02_dataset_stats/`

**Có.** 3 stats JSON, cả ba `"status": "OK"`, `"seed": 42`:

- `vivqa_stats.json` — 15,000 QA / 10,328 images
- `vitextvqa_stats.json`
- `vivqa_x_stats.json` (related, không thay ViVQA)

`samples/`: **18 file** = 3 dataset × 6 loại (Yes-No, Counting, Object-recognition, Text-reading, Reasoning, Other). Đủ 6 loại câu hỏi.

### C4. `src/data/analyze_dataset.py`

**Có — đúng script sinh stats.** Docstring: output `experiments/W02_dataset_stats/{vivqa,vitextvqa,vivqa_x}_stats.json` + `samples/<dataset>__<qtype>.json`; `DEFAULT_OUT_DIR = experiments/W02_dataset_stats`; `SEED = 42`; `SCRIPT_VERSION = w02-analyze-1.0`. JSON `generated_at_utc` 2026-09-14 khớp run-of-record trong WEEK2_STATUS / Dataset_Analysis provenance.

### Kết luận sơ bộ: W2 **ĐỦ** (milestone Literature matrix + Dataset analysis)

Đủ để chấm pass phần lõi tuần 2: matrix 10 hàng không bịa số + dataset analysis dựa trên run thật hai dataset chính.

**THIẾU / nợ (không phủ nhận pass lõi):**

1. 0/10 PDF paper trong repo → ô metric literature phần lớn `❌ TODO`.
2. Licence + official split chưa verify; mirror không có val.
3. `docs/W02/` không có Team Report (status đang ở **root** `WEEK2_STATUS.md`).
4. RQ / Hypotheses / Candidate_Methods bên W02 là scaffold; bản weekly **dày hơn** (H1–H8, A1–A9, recipe) — nên port ở W3, không đụng `docs/W02/` trong bước archive này.
5. Ảnh chưa tải (W4). Question-type heuristic, ~30–41% Other.
6. SVx_Paper_Review trong weekly/W02 là stub; review thật W1/`docs/literature/`.

Tự đánh giá nhóm: **90%** (WEEK2_STATUS).

---

## D. ARCHIVE PLAN (thực hiện sau khi file này đã ghi)

```
Move-Item .\docs\weekly\W02  .\_archive\docs_weekly_W02_pre_2026-09-14
```

Đếm `docs/W02/` trước archive: **n=6**. Không sửa 6 file đó.
