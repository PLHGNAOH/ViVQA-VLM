# W13 · Chẩn đoán vì sao R-YB v1 KHÔNG HỢP LỆ (27/10/2026, không tốn GPU)

Nguồn: `meta.json` + `logits.json` của 9 run `W13_YN_*` (Drive) và `predictions.json` của 3 run greedy W12 (repo). Chỉ mô tả, không phải luật.

## 1. Từ "không" viết thường là 2 token → bị v1 loại

| Từ | Token id (Qwen2.5-VL) | 1 token? | v1 dùng? |
|---|---|---|---|
| có | [129133] | có | có |
| Có | [34, 1794] | không | **không** |
| CÓ | [34, 39062] | không | **không** |
| yes | [9693] | có | có |
| Yes | [9454] | có | có |
| không | [30664, 24015] | không | **không** |
| Không | [142899] | có | có |
| KHÔNG | [75729, 131371, 6140] | không | **không** |
| no | [2152] | có | có |
| No | [2753] | có | có |

v1 chỉ lấy từ 1 token nên `s_no` thiếu đúng câu trả lời "không" (kh + ông) — câu trả lời "không" phổ biến nhất của SH và B3-full. Token top-1 không thuộc tập v1 gần như toàn là `kh` (đầu của "không") hoặc `C` (đầu của "Có"):

| Cấu hình | POPE-vi: top-1 ngoài tập v1 | Chữ đầu greedy W12 (nguyên văn) |
|---|---|---|
| SH | `kh` 1159 | có 3773, không 1085, no 119, cây 4, cái 3 |
| B1_ocr | `kh` 1827, `C` 331 | Không 2809, có 963, không 838, Có 323, Không. 52 |
| B3full | `kh` 2847 | không 2299, có 1804, no 881, yes 1 |

## 2. Greedy W12 bị phạt chữ "có": đổi sang "Không"/"no" khi chênh lệch nhỏ

Xét câu POPE-vi có token top-1 (logit, không phạt) là "có". Cột = khoảng margin v1 (log P(có, yes, Yes) − log P(Không, no, No)); ô = số câu greedy W12 lại trả lời "Không"/"no" / tổng:

| Cấu hình | [0; 0.5) | [0.5; 1) | [1; 1.5) | [1.5; 2) | [2; 3) | [3; ∞) |
|---|---|---|---|---|---|---|
| SH | 9/9 | 33/56 | 0/160 | 0/294 | 0/614 | 0/2682 |
| B1_ocr | 125/125 | 81/132 | 0/111 | 0/110 | 0/155 | 0/536 |
| B3full | 28/28 | 81/82 | 1/119 | 0/160 | 0/340 | 0/1183 |

Chuyển trạng thái sắc nét quanh 0,5–1 đơn vị log: dưới 0,5 greedy **luôn** đổi, trên 1 **không bao giờ** đổi. Nhiễu số học không tạo ra ngưỡng như vậy; một khoản phạt cố định lên logit của "có" thì có. Giải thích phù hợp nhất: `model.generate()` dùng `generation_config` mặc định của Qwen2.5-VL-3B-Instruct, trong đó có `repetition_penalty` > 1 (giá trị được ghi vào `meta.json` của run v2). Penalty phạt mọi token **đã có trong input**. Prompt POPE-vi chứa `Trả lời "có" hoặc "không"`, nên nhiều khả năng token `có` và `kh`/`ông` bị phạt, còn `Không` (viết hoa) và `no` (tiếng Anh) thì không. Đây vẫn là giả thuyết; v2 dùng đúng `input_ids` thật nên không phụ thuộc giả thuyết này, và smoke test v2 (trùng greedy ≥ 98%) là phép kiểm.

## 3. Hệ quả

1. **v1 đo sai hai chỗ**: thiếu từ nhiều token, và so điểm *không phạt* với greedy *có phạt* → kiểm hợp lệ (≥ 98%) không thể đạt. Đúng như luật: KHÔNG HỢP LỆ, không kết luận từ số v1.
2. **Sửa cách đo (v2)**: chấm chính xác từ nhiều token bằng teacher forcing + áp đúng repetition penalty như `generate()`. Ngưỡng luật giữ nguyên. Ghi trước ở `docs/W13/README.md` (R-YB v2).
3. **Giả thuyết hậu kiểm (chưa kiểm định)**: penalty đẩy model khỏi "có"/"không" (có trong prompt) sang "Không"/"no" → có thể là nguyên nhân của code-switching "no" ở B3-full (17,7% câu POPE-vi, W12). Mọi run eval W05–W12 dùng **cùng** cách sinh này nên các so sánh giữa run vẫn công bằng; cần ghi rõ giá trị penalty trong phần Phương pháp.

