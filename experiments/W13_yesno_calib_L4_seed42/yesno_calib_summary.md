# W13 · Thiên lệch "có" của SH — chẩn đoán logit + ngưỡng lúc suy luận (R-YB)

## 1. Kiểm hợp lệ: quyết định logit (τ = 0) trùng greedy W12

| Cấu hình | Trùng | Câu đọc được |
|---|---|---|
| SH | 87.4% | 4977/4985 |
| B1_ocr | 94.4% | 4985/4985 |
| B3full | 83.1% | 4985/4985 |

Ngưỡng hợp lệ (SH, B1-ocr): ≥ 98.0% → **KHÔNG HỢP LỆ**

## 2. Chẩn đoán (chỉ mô tả): AUC của margin = P(câu có-thật xếp trên câu không-có)

| Cấu hình | calib | random | popular | adversarial | t2 random | t2 popular | t2 adversarial |
|---|---|---|---|---|---|---|---|
| SH | 0.967 | 0.945 | 0.917 | 0.898 | 0.957 | 0.944 | 0.924 |
| B1_ocr | 0.968 | 0.954 | 0.933 | 0.909 | 0.959 | 0.942 | 0.914 |
| B3full | 0.972 | 0.957 | 0.929 | 0.912 | 0.956 | 0.943 | 0.918 |

## 3. Ngưỡng τ chọn trên calibration (balanced accuracy)

| Cấu hình | τ* | BA(τ*) | BA(τ = 0) |
|---|---|---|---|
| SH | +5.75 | 91.1% | 63.4% |
| B1_ocr | -1.20 | 90.0% | 89.1% |
| B3full | +2.15 | 91.4% | 85.6% |

## 4. Luật R-YB — SH với τ* = +5.75 so với B1-ocr greedy (POPE-vi, đánh giá 1 lần)

| Cách chọn | F1 SH-sửa | F1 B1-ocr | ΔF1 [CI95] | yes-ratio | báo sai SH-sửa | báo sai B1-ocr |
|---|---|---|---|---|---|---|
| random | 85.10 | 83.39 | +1.71 [+0.60; +2.85] | 41.5% | 5.1% | 2.7% |
| popular | 83.25 | 81.99 | +1.26 [+0.13; +2.45] | 43.5% | 9.2% | 5.7% |
| adversarial | 81.59 | 80.59 | +1.01 [-0.18; +2.21] | 45.4% | 13.0% | 8.9% |

## ➜ R-YB: **KHÔNG HỢP LỆ**

## 5. Chỉ mô tả (không thuộc luật)

| Cấu hình | Cách chọn | F1 | yes-ratio | báo sai |
|---|---|---|---|---|
| SH_tau0_popevi | random | 71.73 | 88.7% | 77.9% |
| SH_tau0_popevi | popular | 69.19 | 93.8% | 88.1% |
| SH_tau0_popevi | adversarial | 68.77 | 94.6% | 89.8% |
| SH_greedy_popevi | random | 77.70 | 77.1% | 55.4% |
| SH_greedy_popevi | popular | 72.83 | 85.4% | 72.4% |
| SH_greedy_popevi | adversarial | 71.17 | 88.5% | 78.7% |
| B1_ocr_own_tau_popevi | random | 87.63 | 51.0% | 13.5% |
| B1_ocr_own_tau_popevi | popular | 84.88 | 54.3% | 20.1% |
| B1_ocr_own_tau_popevi | adversarial | 80.93 | 59.4% | 30.3% |
| B3full_own_tau_popevi | random | 86.27 | 42.0% | 4.7% |
| B3full_own_tau_popevi | popular | 83.84 | 44.7% | 10.0% |
| B3full_own_tau_popevi | adversarial | 82.59 | 46.1% | 12.9% |
| SH_fixed_t2 | random | 75.62 | 31.2% | 1.0% |
| SH_fixed_t2 | popular | 75.40 | 31.4% | 1.5% |
| SH_fixed_t2 | adversarial | 74.97 | 31.9% | 2.4% |
| SH_tau0_t2 | random | 87.15 | 57.4% | 21.2% |
| SH_tau0_t2 | popular | 85.85 | 59.0% | 24.5% |
| SH_tau0_t2 | adversarial | 81.75 | 64.5% | 35.4% |
| B1_ocr_tau0_t2 | random | 81.05 | 35.3% | 1.5% |
| B1_ocr_tau0_t2 | popular | 80.36 | 36.0% | 2.9% |
| B1_ocr_tau0_t2 | adversarial | 79.31 | 37.2% | 5.2% |
| B3full_tau0_t2 | random | 88.35 | 47.3% | 8.6% |
| B3full_tau0_t2 | popular | 86.89 | 48.9% | 11.9% |
| B3full_tau0_t2 | adversarial | 84.03 | 52.3% | 18.6% |

## 6. Chi phí

| Run | s/câu | GPU | VRAM đỉnh (GB) | top-1 là token có/không |
|---|---|---|---|---|
| SH/calib | 0.2424 | NVIDIA L4 | 2.6 | 63.9% |
| SH/popevi | 0.241 | NVIDIA L4 | 2.58 | 76.8% |
| SH/popevi_t2 | 0.2427 | NVIDIA L4 | 2.58 | 63.6% |
| B1_ocr/calib | 0.2116 | NVIDIA L4 | 2.57 | 52.1% |
| B1_ocr/popevi | 0.2154 | NVIDIA L4 | 2.55 | 56.7% |
| B1_ocr/popevi_t2 | 0.2121 | NVIDIA L4 | 2.55 | 73.0% |
| B3full/calib | 0.2425 | NVIDIA L4 | 2.6 | 36.5% |
| B3full/popevi | 0.243 | NVIDIA L4 | 2.58 | 42.9% |
| B3full/popevi_t2 | 0.2424 | NVIDIA L4 | 2.58 | 89.7% |
