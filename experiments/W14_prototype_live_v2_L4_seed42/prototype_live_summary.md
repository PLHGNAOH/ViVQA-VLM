# W14 bước 3 — prototype chạy thật (OCR trực tiếp + model), prototype-live-v2

OCR: tiến trình riêng, device `gpu` · engine trùng cache: ✅

## 1. R-OCR — văn bản OCR vào prompt qua giao diện = OCR đọc thẳng file gốc (cùng phiên)?

| Tập | Ảnh | Trùng hệt | Khác | Độ giống ≥ ngưỡng | Độ giống thấp nhất |
|---|---|---|---|---|---|
| vivqa | 200 | 100,0% | 0 | 100,0% | 1,000 |
| vitextvqa_official | 200 | 95,5% | 9 | 99,5% | 0,821 |

Bảng trên: **so với cache** (dựng trên T4) — chỉ mô tả. Bảng dưới: **luật R-OCR v2** (cùng tiến trình OCR, cùng L4).

| Tập | Ảnh | Giao diện = đọc thẳng | Khác |
|---|---|---|---|
| vivqa | 200 | 100,0% | 0 |
| vitextvqa_official | 200 | 100,0% | 0 |

➜ **R-OCR: ĐẠT** (trùng hệt ≥ 98% từng tập)

- vitextvqa_official `14173.jpg` (giống 0,821): trực tiếp ['PGS25', 'unsenser', 'GS25', 'Lifestyle Platform'] · cache ['PGS25', 'inversionalized', 'GS25', 'Lifestyle Platform']
- vitextvqa_official `16475.jpg` (giống 0,921): trực tiếp ['ĐƠN VỊ THI CÔNG PHÒNG NET CAO CẤP THẾ', 'ĐẠI LYINTERNET VÀ ĐIỂM CUNG CẤP DỊCH VỤ TRÒ CHƠI ĐIỆN TỪ CÔNG CỘNG', 'ORBIT', 'NHẬT QUÂN', 'ORBIT', 'TUNG DÙNG KHAI TRUỦNG NETGAME'] · cache ['ĐƠN VỊ THI CÔNG PHÒNG NET CAO CẤP THẾ', 'ĐẠI LYINTERNET VÀ ĐIỂM CUNG CẤP DỊCH VỤ TRÒ CHƠI ĐIỆN TỪ CÔNG CỘNG', 'ORBIT', 'NHẬT QUÂN', 'ORBIT', 'TUNG DÙNG KHAI TRUỦNG NETGAME']
- vitextvqa_official `6351.jpg` (giống 0,954): trực tiếp ['patibility', '020', 'CONTENTINUATION', 'HỢP TÁC XÁT TÍNH CAM U', 'HTX NÔNG NGHIỆP BỊ GIA', 'LONG BUNG'] · cache ['mationing', '020', 'CONTENTINUATION', 'HỢP TÁC XÁT TÍNH CAM U', 'HTX NÔNG NGHIỆP BỊ GIA', 'LONG BUNG']

## 2. R-LIVE — câu trả lời qua đường giao diện = run eval W11?

| Tập | Câu (bỏ có/không · bỏ ảnh xoay) | Trùng hệt | EM trực tiếp | EM W11 | Δ |
|---|---|---|---|---|---|
| vivqa | 211 (0 · 0) | 100,0% | 75,83 | 75,83 | 0,00 |
| vitextvqa_official | 259 (0 · 2) | 100,0% | 50,19 | 50,19 | 0,00 |

➜ **R-LIVE: ĐẠT** (trùng ≥ 97% và Δ EM ≥ −1,5 điểm, từng tập)


## 3. Chỉ mô tả — tốc độ, bộ nhớ

Khởi động: OCR 17,2 s · tổng (gồm nạp model) 60,8 s · GPU NVIDIA L4 · VRAM đỉnh tiến trình chính 2,61 GB · VRAM cuối (mọi tiến trình) 4784 MiB

| Tập | n | OCR p50 / p95 (s) | Model p50 / p95 (s) | Tổng p50 / p95 (s) |
|---|---|---|---|---|
| vivqa | 206 | 0,04 / 0,28 | 0,54 / 0,83 | 0,61 / 0,89 |
| vitextvqa_official | 261 | 0,72 / 2,51 | 0,83 / 1,80 | 1,72 / 3,96 |
| all | 467 | 0,32 / 2,10 | 0,65 / 1,62 | 1,00 / 3,58 |

Mức tin cậy trên mẫu: {'Cao': 144, 'Trung bình': 156, 'Thấp': 172} · câu có ảnh đi nguyên file: 470 · ảnh xoay EXIF: 2 câu · ảnh thu nhỏ: 0 câu
