# W13 bước 8 — prototype chạy thật (OCR trực tiếp + model)

OCR: tiến trình riêng, device `gpu` · engine trùng cache: ✅

## 1. R-OCR — văn bản OCR vào prompt: trực tiếp = cache?

| Tập | Ảnh | Trùng hệt | Khác |
|---|---|---|---|
| vivqa | 200 | 100,0% | 0 |
| vitextvqa_official | 200 | 87,5% | 25 |

➜ **R-OCR: KHÔNG ĐẠT** (ngưỡng ≥ 98% từng tập)

- vitextvqa_official `1849.jpg`: trực tiếp ['Trà Sữa', 'Reportite', 'Cho Banh Que Ma Tây', 'KIM HƯƠNG', 'DC.130B NGUYỄN TRỊ PHƯƠNG, P. 9.QUẬN 5', 'HOTLINE: 0906 60 60 93'] · cache ['Lin', 'WHIRS', 'Transly', 'Contracts', '0301000000120', 'messing']
- vitextvqa_official `1981.jpg`: trực tiếp ['Bên', 'Thau', 'Ironipor', 'Man', 'Cố'] · cache ['Bên', 'Thau', 'Ironpot', 'Man', 'Cố']
- vitextvqa_official `2012.jpg`: trực tiếp ['HÙNG THỊNH', 'CHỈ CỤC CHẤN NUÔI VÀ THỦ Y HÀ NỘI', 'NG ĐỒ THỜ CAO CẤP', 'TRAN CHÂN NƯỚI VÀ THÚY HÀ ĐÔNG', '12 P NGUYỄN TRÃI, QHÀ ĐÔNG, TP HÀ NỘI', 'ĐT: 0989 095 809-0915 220 744'] · cache ['K', 'HÙNG THỊNH', 'CHỈ CỤC CHẤN NUÔI VÀ THỦ Y HÀ NỘI', 'NG ĐỒ THỜ CAO CẤP', 'TRAN CHAN NUÔI VÀ THÚY HÀ ĐÔNG', '12 P NGUYỄN TRÃI, QHÀ ĐÔNG, TP HÀ NỘI']

## 2. R-LIVE — câu trả lời qua đường giao diện = run eval W11?

| Tập | Câu (bỏ có/không) | Trùng hệt | EM trực tiếp | EM W11 | Δ |
|---|---|---|---|---|---|
| vivqa | 211 (0) | 100,0% | 75,83 | 75,83 | 0,00 |
| vitextvqa_official | 261 (0) | 98,5% | 50,19 | 49,81 | 0,38 |

➜ **R-LIVE: ĐẠT** (trùng ≥ 97% và Δ EM ≥ −1,5 điểm, từng tập)

- vitextvqa_official test_5273: trực tiếp «quận 5» · W11 «quận tân bình»
- vitextvqa_official test_27242: trực tiếp «tnhh mtv hải đảo lý sơn quảng ngãi» · W11 «công ty tnhh mtv hải đảo lý sơn quảng ngãi»
- vitextvqa_official test_42763: trực tiếp «51d - 0421» · W11 «5a 421»

## 3. Chỉ mô tả — tốc độ, bộ nhớ

Khởi động: OCR 16,6 s · tổng (gồm nạp model) 59,8 s · GPU NVIDIA L4 · VRAM đỉnh tiến trình chính 2,61 GB · VRAM cuối (mọi tiến trình) 4814 MiB

| Tập | n | OCR p50 / p95 (s) | Model p50 / p95 (s) | Tổng p50 / p95 (s) |
|---|---|---|---|---|
| vivqa | 206 | 0,04 / 0,26 | 0,53 / 0,81 | 0,59 / 0,88 |
| vitextvqa_official | 261 | 0,69 / 2,40 | 0,81 / 1,75 | 1,63 / 3,65 |
| all | 467 | 0,31 / 1,92 | 0,63 / 1,58 | 0,98 / 3,30 |

Mức tin cậy trên mẫu: {'Cao': 145, 'Trung bình': 155, 'Thấp': 172} · ảnh đi nguyên file: 447
