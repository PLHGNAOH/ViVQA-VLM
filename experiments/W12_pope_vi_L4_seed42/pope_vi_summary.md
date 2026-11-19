# POPE-vi (H4) — 500 ảnh · 4985 câu duy nhất · sha256 53d04eb23ad4…

| Cấu hình | Cách chọn | Acc | Prec | Rec | **F1** | yes-ratio | không đọc được |
|---|---|---|---|---|---|---|---|
| B1_plain | random | 86.20 | 95.71 | 75.80 | **84.60** | 39.6% | 0.00% |
| B1_plain | popular | 84.67 | 92.14 | 75.80 | **83.17** | 41.1% | 0.00% |
| B1_plain | adversarial | 82.43 | 87.39 | 75.80 | **81.19** | 43.4% | 0.00% |
| B1_ocr | random | 85.37 | 96.41 | 73.47 | **83.39** | 38.1% | 0.00% |
| B1_ocr | popular | 83.87 | 92.76 | 73.47 | **81.99** | 39.6% | 0.00% |
| B1_ocr | adversarial | 82.30 | 89.23 | 73.47 | **80.59** | 41.2% | 0.00% |
| C3 ⚠ | random | 7.87 | 3.94 | 3.60 | **3.76** | 1.8% | 91.43% |
| C3 ⚠ | popular | 7.40 | 3.90 | 3.60 | **3.74** | 1.8% | 91.43% |
| C3 ⚠ | adversarial | 5.83 | 3.77 | 3.60 | **3.68** | 1.8% | 91.43% |
| B3full | random | 89.13 | 90.93 | 86.93 | **88.89** | 47.8% | 0.00% |
| B3full | popular | 85.43 | 84.40 | 86.93 | **85.65** | 51.5% | 0.00% |
| B3full | adversarial | 82.27 | 79.51 | 86.93 | **83.06** | 54.7% | 0.00% |
| SH | random | 71.67 | 64.06 | 98.73 | **77.70** | 77.1% | 0.16% |
| SH | popular | 63.17 | 57.69 | 98.73 | **72.83** | 85.4% | 0.16% |
| SH | adversarial | 60.00 | 55.63 | 98.73 | **71.17** | 88.5% | 0.16% |

## So sánh theo luật (ΔF1 điểm, CI95 bootstrap theo ảnh)

| So sánh | random | popular | adversarial | Phán quyết |
|---|---|---|---|---|
| C3_vs_B1_plain | -80.84 [-82.52, -79.16] | -79.43 [-81.08, -77.78] | -77.50 [-79.20, -75.81] | **CHƯA KẾT LUẬN** |
| B3full_vs_B1_ocr | +5.50 [+4.22, +6.80] | +3.65 [+2.29, +5.02] | +2.47 [+1.08, +3.88] | **KHÔNG TĂNG HALLUCINATION** |
| SH_vs_B1_ocr | -5.69 [-7.30, -4.06] | -9.16 [-10.73, -7.57] | -9.42 [-10.98, -7.82] | **TĂNG HALLUCINATION** |

Chỉ mô tả — B1_ocr vs B1_plain: random -1.21 [-1.81, -0.63] · popular -1.18 [-1.79, -0.58] · adversarial -0.60 [-1.20, -0.01]

## ➜ H4 THEO LUẬT: **MỘT PHẦN** (C3: CHƯA KẾT LUẬN · B3-full: KHÔNG TĂNG HALLUCINATION) · SH (prototype): TĂNG HALLUCINATION
