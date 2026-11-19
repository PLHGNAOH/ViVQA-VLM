# W13 · Prototype backend — parity (R-PAR), độ tin cậy (R-CONF), detector có/không (R-YB-det)

## 1. R-PAR — backend trả lời trùng hệt run eval W11

| Tập | Câu | Trùng hệt | Lệch | EM backend | EM W11 |
|---|---|---|---|---|---|
| vivqa | 3001 | 100.0% | 0 | 73.51 | 73.51 |
| vitextvqa_official | 2000 | 100.0% | 0 | 49.50 | 49.50 |

➜ **R-PAR: ĐẠT** (ngưỡng ≥ 99.0% từng tập)


## 2. R-CONF — độ tin cậy câu thường (ngưỡng chọn trên nửa A theo ảnh: Thấp < 0.839 ≤ Trung bình < 0.958 ≤ Cao)

Nửa A 2496 câu (chọn ngưỡng) · nửa B 2505 câu (đánh giá).

| Nửa B | Cao: EM (n) | Trung bình: EM (n) | Thấp: EM (n) | ECE |
|---|---|---|---|---|
| gộp | 92.0% (790) | 69.6% (823) | 34.2% (892) | 0.198 |
| vivqa | 94.9% (610) | 76.6% (444) | 42.5% (454) | 0.125 |
| vitextvqa_official | 82.2% (180) | 61.5% (379) | 25.6% (438) | 0.308 |

➜ **R-CONF: ĐẠT** (EM giảm dần Cao > Trung bình > Thấp: gộp và từng tập) · giao diện chỉ hiện MỨC (ECE > 0,05), không hiện con số

## 3. R-YB-det — câu bị detector bắt đi đường có/không

| Tập | Câu bị bắt | EM greedy | EM nếu đi đường có/không | Giảm |
|---|---|---|---|---|
| vivqa | 1/3001 | 73.51 | 73.48 | +0.03 |
| vitextvqa_official | 0/2000 | 49.50 | 49.50 | +0.00 |

➜ **R-YB-det: ĐẠT** (giảm ≤ 1.5 điểm từng tập)

- vivqa: «điều này có một giấc ngủ ngắn trên một chiếc ghế dài chữ l không» gold ['con chó'] · greedy «con chó» · có/không «có»

## 4. Chỉ mô tả — độ tin cậy câu có/không (σ(|margin − τ*|), ngưỡng chọn trên calibration)

Ngưỡng: 0.975 / 0.999 · ECE POPE-vi 0.050

| POPE-vi | Cao | Trung bình | Thấp |
|---|---|---|---|
| đúng (n) | 98.2% (952) | 95.6% (1722) | 77.1% (2311) |

## 5. Chỉ mô tả — τ* với mẫu câu 3 (`Có X trong ảnh không? Trả lời "có" hoặc "không".`)

| Cấu hình | Cách chọn | F1 | yes-ratio | báo sai |
|---|---|---|---|---|
| SH_tau_star_t3 | random | 85.80 | 42.2% | 5.3% |
| SH_tau_star_t3 | popular | 84.60 | 43.5% | 7.9% |
| SH_tau_star_t3 | adversarial | 83.07 | 45.3% | 11.4% |
| SH_tau0_t3 | random | 81.16 | 71.0% | 43.8% |
| SH_tau0_t3 | popular | 77.75 | 76.3% | 54.4% |
| SH_tau0_t3 | adversarial | 74.68 | 81.5% | 64.8% |
| SH_tau_star_t1 | random | 85.14 | 41.5% | 5.1% |
| SH_tau_star_t1 | popular | 83.32 | 43.5% | 9.1% |
| SH_tau_star_t1 | adversarial | 81.69 | 45.4% | 12.9% |
| B1_ocr_greedy_t1 | random | 83.39 | 38.1% | 2.7% |
| B1_ocr_greedy_t1 | popular | 81.99 | 39.6% | 5.7% |
| B1_ocr_greedy_t1 | adversarial | 80.59 | 41.2% | 8.9% |

## 6. Chi phí

| Tập | s/câu | VRAM đỉnh (GB) | câu bị detector bắt |
|---|---|---|---|
| vivqa | 0.5447 | 2.6 | 1 |
| vitextvqa_official | 0.9542 | 2.61 | 0 |
