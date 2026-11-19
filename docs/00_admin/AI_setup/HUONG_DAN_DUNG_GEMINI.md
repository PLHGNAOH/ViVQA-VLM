# Hướng dẫn dùng Gemini cho KLTN ViVQA-VLM

> Cập nhật 17/10/2026. Lưu tại `docs/00_admin/AI_setup/HUONG_DAN_DUNG_GEMINI.md`.
> Quên thì mở file này. Phần A làm MỖI LẦN; phần D chỉ làm khi thiết lập lại.

---

## A. MỖI LẦN MỞ PHIÊN LÀM VIỆC VỚI GEMINI (4 bước, ~2 phút)

**Bước 1 — Mở chat trong Gem**
Mở Gemini → bấm Gem **ViVQA-VLM · Trợ lý KLTN** → bắt đầu chat mới.

**Bước 2 — Tải file gộp mới nhất (NGUỒN CHÍNH, bắt buộc)**
1. Mở trình duyệt: https://github.com/PLHGNAOH/ViVQA-VLM/blob/ai-context/00_ALL_REPORTS.md
2. Bấm nút **Download raw file** (biểu tượng mũi tên tải xuống, góc phải phía trên nội dung file).
3. Trong Gemini: **+ → Tải tệp lên** → chọn `00_ALL_REPORTS.md` vừa tải.
- File này tự cập nhật sau MỖI lần Cursor push lên GitHub (khoảng 1 phút), nên luôn tải bản mới ở đầu chat.
- Dùng xong có thể xoá file trong Downloads (giữ Downloads sạch).

**Bước 3 — (Tuỳ chọn) Nhập mã repo — chỉ khi cần hỏi về code / cấu trúc thư mục**
**+ → Nhập mã** → dán `https://github.com/PLHGNAOH/ViVQA-VLM/tree/ai-context` → **Nhập**.
- Liên kết GitHub đã làm 1 lần, không cần làm lại.
- Lưu ý: Nhập mã thấy được tên file nhưng KHÔNG đọc tin cậy nội dung tài liệu → số liệu luôn lấy từ `00_ALL_REPORTS.md`.

**Bước 4 — Dán Mẫu A (mục B), sửa phần `<…>`, rồi hỏi bất kỳ câu nào.**
- Nếu đang GIỮA TUẦN và vừa chuyển từ Claude sang: ở bước 2 tải thêm file handoff Claude đưa (`Wxx_session_handoff.md`).
- Đặt tên chat: bấm **⋮** góc phải trên → đổi tên `W10 · <việc> (Gemini)`.
- Cùng một chat thì không cần làm lại bước 2–3, trừ khi muốn Gemini thấy commit mới vừa push.

---

## B. CÁC MẪU CÂU (copy–dán)

### Mẫu A — Mở phiên (dùng MỌI lần)
```
Mở phiên: <W10 · Review 2 + B3 full data> (Gemini).
Tôi đã tải lên 00_ALL_REPORTS.md (file gộp Context Pack + README/Team Report mọi tuần + R1–R7)<, và repo nhánh ai-context qua Nhập mã><, và handoff Wxx_session_handoff.md>.
Trước khi làm việc, trả lời ngắn, mỗi ý ghi đường dẫn nguồn (dòng "# FILE: …"):
1. Dòng "Built from main @ …" của 00_ALL_REPORTS.md ghi commit nào?
2. Tuần thực hiện tại là tuần mấy, mốc gần nhất là gì?
3. Tuần ảo mới nhất có Team Report là tuần nào?
4. 3 việc đang mở ưu tiên nhất.
Sau đó: <việc hôm nay, vd "lên kế hoạch tập nói slide Review 2">
```
→ Gemini trả lời sai tuần hoặc không trích được "# FILE: …" ⇒ tải lại file ở bước 2.

### Mẫu B — Đóng phiên Gemini (khi chuyển về Claude)
```
Đóng phiên. Viết Wxx_session_handoff.md theo khuôn mục 8 của Context Pack, ghi "viết bởi Gemini", liệt kê mọi số liệu/code bạn đã đưa ra trong phiên này ở mục Nguồn gốc.
```
→ Tải file handoff về, mở chat Claude, đính kèm file đó.

### Câu kiểm tra nhanh (khi nghi Gemini đọc sai)
```
Theo 00_ALL_REPORTS.md, EM và ANLS zero-shot ở W04 là bao nhiêu, và W06 tìm ra nguyên nhân gì của output suy biến? Ghi đường dẫn nguồn.
```
Đáp án đúng: **EM 38,3% / ANLS 55,9%** (`docs/W04/...`) và **fp16** (`docs/W06/W06_Team_Report.md`).

---

## C. CHIA VIỆC CLAUDE ↔ GEMINI

| Việc | Dùng |
|---|---|
| Viết code, notebook, prompt Cursor; kiểm commit; tính lại số | **Claude** (Gemini khi Claude hết hạn mức) |
| Tìm paper / SOTA | Gemini **Deep Research** (bật trong chat) → đọc bản gốc trước khi đưa vào báo cáo |
| Viết nháp báo cáo R1–R7 (bản tiếng Việt trước) | Gemini **Canvas** hoặc Claude |
| Ôn khái niệm chuẩn bị bảo vệ | Gemini **Học có hướng dẫn** |
| Kiểm chéo kết luận quan trọng | Hỏi AI thứ hai trước khi đưa vào slide/thesis |

**Quy tắc vàng**
1. Mọi thay đổi repo đi qua **Cursor** (Gemini chỉ đọc, không push).
2. Số liệu Gemini đưa mà **không trích được "# FILE: …"** → chưa dùng; chạy lại từ code hoặc nhờ Claude kiểm.
3. Đóng góp đáng kể của AI → thêm 1 dòng vào `docs/00_admin/AI_setup/AI_Usage_Log.md` (qua Cursor).
4. **Không** tải lên Gemini: `ViTextVQA_test_gt.json`, dữ liệu thô dataset, file chứa thông tin cá nhân.
5. Sắp hết hạn mức Claude → nói với Claude "viết handoff giữa tuần" trước khi bị chặn.

---

## D. THIẾT LẬP MỘT LẦN (đã xong — chỉ xem khi cần làm lại)

| Hạng mục | Trạng thái |
|---|---|
| Gem `ViVQA-VLM · Trợ lý KLTN`: ô **Chỉ dẫn** = toàn bộ `docs/00_admin/AI_setup/Gemini_Gem_Instructions.md` | ✅ (cập nhật lại nếu file này đổi) |
| Gem — **Mô tả**: 1 câu nhận diện; **Công cụ mặc định**: Không có | ✅ |
| Gem — **Tri thức**: Context Pack, 2 handoff gần nhất, master_instruction, PDF đăng ký đề tài | ✅ (cập nhật ở các mốc lớn: sau Review 2, trước W13) |
| Liên kết GitHub (PLHGNAOH) với Gemini | ✅ |
| GitHub Action `.github/workflows/ai_context.yml` tự dựng nhánh `ai-context` + `00_ALL_REPORTS.md` sau mỗi push | ✅ |

**Mỗi lần đóng tuần (Claude lo, bạn không phải làm thêm):** gói đóng tuần có Context Pack mới → Cursor push → Action tự dựng lại `00_ALL_REPORTS.md`. Bạn chỉ cần tải file mới ở đầu chat Gemini tiếp theo.

**Nếu `00_ALL_REPORTS.md` không mở được / cũ:** vào https://github.com/PLHGNAOH/ViVQA-VLM/actions → xem lần chạy "Build ai-context branch" gần nhất có dấu ✅ không; dấu ❌ thì chụp màn hình gửi Claude.

---

## E. VÌ SAO LÀM VẬY (để trả lời hội đồng nếu được hỏi)
- Gem chỉ biết file được tải lên, không tự đọc chat Claude → cần một nguồn chung: repo GitHub.
- Repo đầy đủ ~50 MB (PDF, predictions) vượt khả năng đọc của Gemini; Nhập mã chỉ thấy tên file → dùng file gộp văn bản ~170 KB tự sinh từ repo.
- Mọi AI dùng cùng nguồn + thứ tự ưu tiên nguồn rõ ràng; mọi số liệu trích được về file trong repo → minh bạch, tái lập được (CLO7).
