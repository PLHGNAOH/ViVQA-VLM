# Chuẩn hoá + tách từ khi chấm điểm (metric-tokenization-v1)

## vivqa_test — 3,001 câu, 3,001 đáp án gold

| Bậc | Kỹ thuật | % gold bị đổi | EM B1 | EM C3 |
|---|---|---|---|---|
| S0 | chuỗi thô (strip) | 0.0 | 1.7 | 76.64 |
| S1 | + Unicode NFC | 0.0 | 1.7 | 76.64 |
| S2 | + chữ thường (str.lower) | 0.0 | 2.4 | 76.64 |
| S3 | + regex bỏ dấu câu [^\w\s] (giữ dấu thanh) | 0.27 | 2.4 | 76.64 |
| S4 | + gộp khoảng trắng \s+ | 0.0 | 2.67 | 76.64 |
| S5 | + bỏ tiền tố 'màu ' ở đầu | 20.86 | 11.86 | 76.64 |
| S6 | + số đứng một mình: chữ → chữ số (= normalize_vi) | 14.83 | 22.09 | 76.64 |

Gold chưa ở dạng NFC: 0 · TB 1.57 âm tiết / 1.36 từ mỗi đáp án · 20.89% đáp án có từ ghép (pyvi)

| Run | EM | EM sau tách từ | token-F1 âm tiết | token-F1 từ (pyvi) |
|---|---|---|---|---|
| B1 | 22.09 | 22.09 | 38.05 | 35.14 |
| C3 | 76.64 | 76.64 | 78.84 | 77.66 |

| Cặp | ΔF1 âm tiết [CI95] | ΔF1 từ [CI95] |
|---|---|---|
| B1 → C3 | +40.79 [+39.09; +42.5] | +42.52 [+40.79; +44.28] |

Thứ hạng theo F1 âm tiết: B1 < C3 · theo F1 từ: B1 < C3 → KHÔNG đổi

## vitextvqa_testfull — 10,028 câu, 10,028 đáp án gold

| Bậc | Kỹ thuật | % gold bị đổi | EM B1 | EM B2 | EM B3-full |
|---|---|---|---|---|---|
| S0 | chuỗi thô (strip) | 0.0 | 4.26 | 48.73 | 51.8 |
| S1 | + Unicode NFC | 0.0 | 4.26 | 48.73 | 51.8 |
| S2 | + chữ thường (str.lower) | 0.0 | 19.69 | 48.74 | 51.8 |
| S3 | + regex bỏ dấu câu [^\w\s] (giữ dấu thanh) | 25.75 | 19.76 | 48.85 | 51.89 |
| S4 | + gộp khoảng trắng \s+ | 25.73 | 29.12 | 50.26 | 53.35 |
| S5 | + bỏ tiền tố 'màu ' ở đầu | 0.01 | 29.12 | 50.26 | 53.35 |
| S6 | + số đứng một mình: chữ → chữ số (= normalize_vi) | 0.02 | 29.12 | 50.26 | 53.35 |

Gold chưa ở dạng NFC: 0 · TB 3.67 âm tiết / 2.97 từ mỗi đáp án · 41.81% đáp án có từ ghép (pyvi)

| Run | EM | EM sau tách từ | token-F1 âm tiết | token-F1 từ (pyvi) |
|---|---|---|---|---|
| B1 | 29.12 | 29.12 | 53.89 | 51.83 |
| B2 | 50.26 | 50.26 | 71.16 | 69.07 |
| B3-full | 53.35 | 53.35 | 73.73 | 71.67 |

| Cặp | ΔF1 âm tiết [CI95] | ΔF1 từ [CI95] |
|---|---|---|
| B1 → B2 | +17.27 [+16.58; +17.97] | +17.24 [+16.51; +17.96] |
| B2 → B3-full | +2.57 [+2.11; +3.02] | +2.6 [+2.12; +3.06] |
| B1 → B3-full | +19.84 [+19.11; +20.58] | +19.84 [+19.1; +20.62] |

Thứ hạng theo F1 âm tiết: B1 < B2 < B3-full · theo F1 từ: B1 < B2 < B3-full → KHÔNG đổi
