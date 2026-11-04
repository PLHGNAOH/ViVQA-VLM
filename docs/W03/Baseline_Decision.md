# Baseline Decision — W03

> **Không tự CHỐT.** Nhãn khuyến nghị: **ĐỀ XUẤT, chờ thầy Đức xác nhận.**
> So sánh bám nguồn in-repo. Ô chưa đo / chưa đọc PDF / mâu thuẫn nguồn → ⚠️REVIEW.
> Ngày: 2026-09-17.

**CHỐT bởi nhóm: ____**  *(để trống — điền sau khi thầy Đức xác nhận)*

## 1. Ứng viên (Master Instruction §4.1 — chọn đúng 1)

Ba ứng viên được phép: BLIP-2 **hoặc** Florence-2 **hoặc** Qwen2.5-VL. Bảng dưới tập trung **Qwen2.5-VL-3B-Instruct vs 7B vs Florence-2** (đúng phạm vi task W03). BLIP-2 không phải đối tượng chốt lần này; SV1 từng xếp BLIP-2 là đối chứng dễ reproduce (`docs/W01/SV1_Paper_Review.md`).

## 2. Bảng so sánh

Nguồn cột:

- Kích cỡ / interface: `docs/W02/Candidate_Methods.md` §1; `docs/02_references/Literature_Matrix.md` L5, L6; `docs/W01/SV1_Paper_Review.md` bảng so sánh.
- VRAM 4-bit *trọng số* (ước lượng, **chưa đo trên P100**): Candidate_Methods (3B ≈ 2–2.5 GB, 7B ≈ 4.5–5.5 GB); SV1 (3B ≈ 2–3 GB, 7B ≈ 4.5–5.5 GB; Florence-2 < 1–2 GB inference 4-bit).
- Licence (SV1, chưa mở file licence gốc): 3B Qwen Research License (phi thương mại); 7B Apache-2.0; Florence-2 MIT. ⚠️REVIEW xác nhận trên HuggingFace card.
- Kaggle P100 16 GB free-tier: **không có số đo in-repo.** So sánh định tính với T4 16 GB (cùng dung lượng VRAM) được nêu trong `docs/02_references/Reading_Notes/SV3_Paper_Review.md` §P2 (QLoRA / Colab T4). P100 ≠ T4 (kiến trúc cũ hơn, không BF16). ⚠️REVIEW với thầy Đức.

| Tiêu chí | Qwen2.5-VL-3B-Instruct | Qwen2.5-VL-7B-Instruct | Florence-2 (base 0.23B / large 0.77B) |
|----------|------------------------|------------------------|----------------------------------------|
| Tham số | 3B | 7B | 0.23B / 0.77B |
| Interface | ViT + merger + Qwen2.5 LLM; dynamic resolution | Cùng họ | seq2seq, task prompt (DaViT + encoder-decoder) |
| VRAM 4-bit *trọng số* (ước lượng in-repo) | ≈ 2–2.5 GB (Candidate_Methods) / ≈ 2–3 GB (SV1) — ⚠️REVIEW hai khoảng | ≈ 4.5–5.5 GB | SV1: < 1–2 GB inference 4-bit (ước lượng). Candidate_Methods **để trống**. ⚠️REVIEW |
| VRAM *train* QLoRA / peak | **Chưa đo.** SV3: 3B NF4 ≈ 2–2.5 GB trọng số; activation + ảnh làm tăng peak. ⚠️REVIEW đo W04 | SV3: "7B khả thi trên T4 16 GB (chậm)" — ước lượng, chưa đo. ⚠️REVIEW | Không có số train in-repo. ⚠️REVIEW |
| Fit Kaggle P100 16 GB free-tier? | **Khả năng cao hơn 7B** (cùng class 16 GB với T4 được SV3 nhắm cho 3B QLoRA). **Chưa chạy P100.** ⚠️REVIEW | Chật hơn 3B; SV3 cho là khả thi trên T4 16 GB nhưng chậm. P100 chưa đo. ⚠️REVIEW | Rất nhẹ — SV1: GPU 8–12 GB. Fit 16 GB về dung lượng; **rào cản là tiếng Việt**, không phải VRAM. |
| Phù hợp framing **efficiency**? | Có — model nhỏ, PEFT-only, đóng góp đo được trên GPU đơn | Mạnh hơn về dung lượng; efficiency kém tường minh hơn 3B trên free-tier | Cực efficiency về tham số; lệch "VLM hiện đại đa ngữ" |
| Đa ngữ / tiếng Việt | SV1: đa ngữ, có tiếng Việt (nhận định review; PDF ⚠️REVIEW) | Cùng họ | SV1: rào cản tiếng Việt — "nhiều khả năng baseline phụ" |
| OCR / scene-text | SV1 + L6: OCR mạnh, dynamic resolution (nhận định; số DocVQA/TextVQA = TODO-human) | Cùng họ, bản lớn hơn | OCR nội tại (L5); hữu ích ViTextVQA như **backup / ablation** |
| Rủi ro | Licence 3B (Qwen Research, phi thương mại) — SV1; capstone học thuật thường được nhưng ⚠️REVIEW ghi đúng tên giấy phép | VRAM train trên 16 GB; chậm hơn | Yếu tiếng Việt; câu trả lời ngắn; có thể không đại diện VLM instruction-tuned hiện đại |
| Trạng thái chốt | **ĐỀ XUẤT, chờ thầy Đức xác nhận** | Không đề xuất primary trên P100 16 GB | **Backup** nếu 3B OOM / licence / reproduce thất bại |

## 3. Khuyến nghị (không phải quyết định nhóm)

- **Primary đề xuất:** `Qwen/Qwen2.5-VL-3B-Instruct`
  - Khớp ràng buộc PEFT-only + GPU đơn 16 GB (lập luận in-repo: Candidate_Methods, SV1, SV3).
  - Khớp framing efficiency (Capstone đo cải tiến trên baseline *nhỏ đủ chạy*, không phải SOTA 72B).
  - Cùng họ với phương án nâng 7B nếu sau này có A100 (ablation A8) — không đổi recipe LoRA.
- **Backup:** Florence-2 (base hoặc large — ⚠️REVIEW chọn size khi kích hoạt backup).
  - Lý do backup: VRAM cực thấp; OCR nội tại cho ViTextVQA.
  - Không đề xuất làm primary vì SV1 đã ghi rào cản tiếng Việt.
- **Không đề xuất 7B làm primary trên Kaggle P100 16 GB free-tier** cho đến khi có số peak VRAM thật (W04).

## 4. Việc người phải làm trước khi ghi "CHỐT"

- [ ] Thầy Đức xác nhận GPU thật (Kaggle P100 16 GB? Colab? GPU trường?) — ⚠️REVIEW
- [ ] Xác nhận licence 3B vs 7B trên model card
- [ ] Đo VRAM zero-shot + 1 bước QLoRA (W04), ghi `experiments/`
- [ ] Điền dòng **CHỐT bởi nhóm** ở đầu file + ngày

## 5. Liên kết

- Recipe PEFT: `docs/W02/Candidate_Methods.md`
- Run naming: `docs/00_admin/CONVENTIONS.md`
- Literature: L5 Florence-2, L6 Qwen2.5-VL trong `docs/02_references/Literature_Matrix.md`
