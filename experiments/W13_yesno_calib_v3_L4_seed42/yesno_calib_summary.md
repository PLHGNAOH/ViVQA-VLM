# W13 · Thiên lệch "có" của SH — R-YB v3 (token đầu, max mỗi phía, repetition penalty như generate)

repetition_penalty (đọc từ generation_config, như generate): **1.05**

## 1. Kiểm hợp lệ: quyết định τ = 0 trùng greedy W12

| Cấu hình | Trùng (v3) | Câu đọc được | Lệch greedy→logit | Trùng nếu bỏ penalty (mô tả) |
|---|---|---|---|---|
| SH | 100.0% | 4977/4985 | — | 99.2% |
| B1_ocr | 100.0% | 4985/4985 | — | 95.8% |
| B3full | 99.9% | 4985/4985 | yes->no 3 | 97.8% |

Ngưỡng hợp lệ (SH, B1-ocr): ≥ 98.0% → **HỢP LỆ**

## 2. Chẩn đoán (chỉ mô tả): AUC của margin = P(câu có-thật xếp trên câu không-có)

| Cấu hình | Tập | random | popular | adversarial |
|---|---|---|---|---|
| SH | calib (gộp) | 0.967 | | |
| SH | popevi | 0.944 | 0.917 | 0.897 |
| SH | popevi_raw | 0.945 | 0.917 | 0.898 |
| SH | popevi_t2 | 0.956 | 0.943 | 0.923 |
| B1_ocr | calib (gộp) | 0.969 | | |
| B1_ocr | popevi | 0.955 | 0.934 | 0.910 |
| B1_ocr | popevi_raw | 0.957 | 0.938 | 0.914 |
| B1_ocr | popevi_t2 | 0.958 | 0.944 | 0.917 |
| B3full | calib (gộp) | 0.971 | | |
| B3full | popevi | 0.956 | 0.931 | 0.911 |
| B3full | popevi_raw | 0.954 | 0.931 | 0.909 |
| B3full | popevi_t2 | 0.957 | 0.941 | 0.916 |

## 3. Ngưỡng τ chọn trên calibration (balanced accuracy)

| Cấu hình | τ* | BA(τ*) | BA(τ = 0) |
|---|---|---|---|
| SH | +4.85 | 91.1% | 74.6% |
| B1_ocr | -2.05 | 90.1% | 88.4% |
| B3full | +0.45 | 91.3% | 90.9% |

## 4. Luật R-YB — SH với τ* = +4.85 so với B1-ocr greedy (POPE-vi)

| Cách chọn | F1 SH-sửa | F1 B1-ocr | ΔF1 [CI95] | yes-ratio | báo sai SH-sửa | báo sai B1-ocr |
|---|---|---|---|---|---|---|
| random | 85.14 | 83.39 | +1.75 [+0.68; +2.89] | 41.5% | 5.1% | 2.7% |
| popular | 83.32 | 81.99 | +1.33 [+0.21; +2.50] | 43.5% | 9.1% | 5.7% |
| adversarial | 81.69 | 80.59 | +1.11 [-0.05; +2.30] | 45.4% | 12.9% | 8.9% |

## ➜ R-YB v3: **ĐẠT**

## 5. Chỉ mô tả (không thuộc luật)

| Cấu hình | Cách chọn | F1 | yes-ratio | báo sai |
|---|---|---|---|---|
| SH_tau0_popevi | random | 77.70 | 77.1% | 55.4% |
| SH_tau0_popevi | popular | 72.83 | 85.6% | 72.4% |
| SH_tau0_popevi | adversarial | 71.17 | 88.7% | 78.7% |
| SH_greedy_popevi | random | 77.70 | 77.1% | 55.4% |
| SH_greedy_popevi | popular | 72.83 | 85.4% | 72.4% |
| SH_greedy_popevi | adversarial | 71.17 | 88.5% | 78.7% |
| B1_ocr_own_tau_popevi | random | 87.88 | 50.7% | 12.9% |
| B1_ocr_own_tau_popevi | popular | 85.23 | 53.8% | 19.1% |
| B1_ocr_own_tau_popevi | adversarial | 81.31 | 58.8% | 29.1% |
| B3full_own_tau_popevi | random | 87.55 | 44.5% | 6.3% |
| B3full_own_tau_popevi | popular | 84.80 | 47.6% | 12.4% |
| B3full_own_tau_popevi | adversarial | 82.82 | 49.9% | 17.1% |
| SH_fixed_t2 | random | 78.76 | 33.8% | 1.6% |
| SH_fixed_t2 | popular | 78.35 | 34.2% | 2.5% |
| SH_fixed_t2 | adversarial | 77.74 | 34.9% | 3.8% |
| SH_tau0_t2 | random | 87.65 | 54.4% | 17.3% |
| SH_tau0_t2 | popular | 86.33 | 56.0% | 20.5% |
| SH_tau0_t2 | adversarial | 82.66 | 60.7% | 29.9% |
| B1_ocr_tau0_t2 | random | 82.64 | 37.0% | 2.1% |
| B1_ocr_tau0_t2 | popular | 81.57 | 38.1% | 4.3% |
| B1_ocr_tau0_t2 | adversarial | 80.42 | 39.4% | 6.9% |
| B3full_tau0_t2 | random | 86.98 | 42.2% | 4.2% |
| B3full_tau0_t2 | popular | 84.93 | 44.4% | 8.7% |
| B3full_tau0_t2 | adversarial | 82.97 | 46.7% | 13.1% |

## 6. Chi phí

| Run | s/câu | GPU | VRAM đỉnh (GB) | top-1 là token đầu của từ ứng viên | top-1 khác |
|---|---|---|---|---|---|
| SH/calib | 0.2402 | NVIDIA L4 | 2.6 | 99.9% | c 4 |
| SH/popevi | 0.2419 | NVIDIA L4 | 2.58 | 99.7% | c 14 |
| B1_ocr/popevi | 0.213 | NVIDIA L4 | 2.55 | 100.0% | — |
| SH/popevi_t2 | 0.2412 | NVIDIA L4 | 2.58 | 69.0% | c 1114, b 300, m 56, hai 23, con 15 |
| B1_ocr/calib | 0.2114 | NVIDIA L4 | 2.57 | 100.0% | — |
| B1_ocr/popevi_t2 | 0.2123 | NVIDIA L4 | 2.55 | 100.0% | — |
| B3full/calib | 0.2457 | NVIDIA L4 | 2.6 | 99.9% | v 1, s 1 |
| B3full/popevi | 0.2427 | NVIDIA L4 | 2.58 | 99.9% | c 6 |
| B3full/popevi_t2 | 0.2446 | NVIDIA L4 | 2.58 | 99.3% | c 6, s 5, bur 2, 2 2, o 2 |
