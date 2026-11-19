# Real photos — local SH (demo path)

- Created: 2026-11-16T03:07:28.291867+00:00
- OMP/MKL threads: 12/12
- N total: 20
- EM overall: 70.0%

## EM by set

| Set | N | EM | EM open | EM yesno |
|---|---:|---:|---:|---:|
| w13 | 20 | 70.0% | 60.0% | 100.0% |

## EM by mode (all sets)

- open: n=15 EM=60.0%
- yesno: n=5 EM=100.0%

## Confidence levels

Thấp: 8, Trung bình: 11, Cao: 1

## Latency

- t_total p50=8.645s  p95=13.025s
- t_ocr p50=7.125s  t_model p50=1.305s

## Bộ demo

Đường dẫn: `D:\FPT_university\termno9\ViVQA-VLM\demo_images\` (phẳng: `*.jpg` + `cau_hoi.txt`; gitignored). Chi tiết v2: `demo_set_v2_summary.md`.

Kiểm sạch: `image_in_train = (image ∈ data/vivqa/train.json)` — đúng split train của SH
(`W11_SH_…/run_signature.json` → `data/vivqa/train.json`). Tham chiếu W07:
`experiments/W07_data_checks/vivqa_split_overlap.json` (train∩test images = 1274);
clean ≈ 1219 câu (không trùng ảnh + không trùng cặp hỏi–đáp) trong `docs/W07/W07_Team_Report.md`.

EM ViVQA test (`experiments/W11_full_eval_table/full_eval_table.json (+ .csv same table)`): **SH (demo) = 73,51%** · C3 (so sánh) = 76,64%.

### CORE (5 ảnh)

| # | File | Nguồn | image_in_train | Câu hỏi | Local | Mức |
|---:|---|---|---|---|---|---|
| 0 | `00_vivqa_test_qua_bong.jpg` | ViVQA test (test_23) | **false** | màu của quả bóng là gì | màu cam | Cao |
| 1 | `01_cua_hang_dac_san.jpg` | Ảnh thật nhóm (W13) | n/a | Cửa hàng bán đặc sản của địa phương nào? | hà nội | Trung bình |
| 2 | `02_chai_xit_con_trung.jpg` | Ảnh thật nhóm (W13) | n/a | Bình xịt côn trùng có hương gì? | cam chanh | Trung bình |
| 3 | `03_bang_noi_quy_thang_may.jpg` | Ảnh thật nhóm (W13) | n/a | Thang máy có tải trọng tối đa bao nhiêu kg? | 1150kg | Trung bình |
| 4 | `04_duong_pho_chieu_ta.jpg` | Ảnh thật nhóm (W13) | n/a | Trong ảnh có con chó không? | không | Trung bình |

Lời nói câu 00: «Câu hỏi chung về ảnh từ tập test ViVQA (ảnh không có trong dữ liệu huấn luyện). Adapter dùng chung (SH) trong demo đạt EM 73,51% trên ViVQA test; adapter riêng ViVQA (C3) đạt 76,64%.»

### THÊM

| # | File | Local | Mức |
|---:|---|---|---|
| 5 | `05_bang_thong_tin_duoi_uoi.jpg` | tèo | Cao |
| 6 | `06_ga_metro.jpg` | có | Trung bình |
| 7 | `07_canh_xe_dem.jpg` | 1 | Thấp |

### du_phong ViVQA

| File | id | image_in_train | Câu hỏi | Local |
|---|---|---|---|---|
| `vivqa_test_xe_buyt.jpg` | test_134 | **false** | màu của xe buýt là gì | màu đỏ |

### 8 ứng viên (`vivqa_demo_candidates.json`)

| id | image_in_train | Câu hỏi |
|---|---|---|
| test_2397 | False | những gì chứa đầy những bông hoa màu đỏ và trắng trên đầu bàn |
| test_142 | True | màu sắc của trang phục là gì |
| test_23 | False | màu của quả bóng là gì |
| test_337 | False | màu của đồng cỏ là gì |
| test_299 | False | màu của dấu chấm là gì |
| test_279 | False | màu của áo là gì |
| test_180 | False | màu của con cừu là gì |
| test_134 | False | màu của xe buýt là gì |

### gioi_han

Giữ như trước; `anh_meo_trong_khung.jpg`: câu mở → chó (Thấp) do OCR XUÂN.

## Skipped

- w15: thiếu thư mục hoặc cau_hoi.csv (D:\vivqa_models\real_photos\w15)

## W13 local vs Colab W14 (answer string match)

- Match: 20/20

| Image | Local | Colab | Match | Local EM | Colab EM |
|---|---|---|---|---:|---:|
| xe_buyt_nha_khoa.jpg | orion | orion | yes | 0.0 | 0.0 |
| bang_huong_dan_y_te.jpg | công dân số tphcm | công dân số tphcm | yes | 1.0 | 1.0 |
| tui_cua_hang.jpg | eleven | eleven | yes | 0.0 | 0.0 |
| cua_hang_dac_san.jpg | hà nội | hà nội | yes | 1.0 | 1.0 |
| bang_noi_quy_thang_may.jpg | 1150kg | 1150kg | yes | 1.0 | 1.0 |
| binh_chua_chay.jpg | lắp dụng bình pccc | lắp dụng bình pccc | yes | 0.0 | 0.0 |
| may_anh_canon.jpg | có | có | yes | 1.0 | 1.0 |
| chai_xit_con_trung.jpg | cam chanh | cam chanh | yes | 1.0 | 1.0 |
| bia_sach.jpg | khúc hát đêm thâu | khúc hát đêm thâu | yes | 1.0 | 1.0 |
| bang_go_trang_tri.jpg | làm đi | làm đi | yes | 1.0 | 1.0 |
| bang_thong_tin_duoi_uoi.jpg | tèo | tèo | yes | 1.0 | 1.0 |
| ga_metro.jpg | có | có | yes | 1.0 | 1.0 |
| goi_khan_uot.jpg | lamy | lamy | yes | 1.0 | 1.0 |
| hop_dien_thoai.jpg | không | không | yes | 1.0 | 1.0 |
| anh_meo_trong_khung.jpg | không | không | yes | 1.0 | 1.0 |
| moc_ao.jpg | dây | dây | yes | 0.0 | 0.0 |
| bo_ho_nuoc_ban_dem.jpg | sân cát | sân cát | yes | 0.0 | 0.0 |
| dan_meo_con.jpg | sáu | sáu | yes | 0.0 | 0.0 |
| duong_pho_chieu_ta.jpg | không | không | yes | 1.0 | 1.0 |
| canh_xe_dem.jpg | 1 | 1 | yes | 1.0 | 1.0 |

## Per-question

| Set | Image | Q | Gold | Pred | EM | Mode | Level | t_total |
|---|---|---|---|---|---:|---|---|---:|
| w13 | xe_buyt_nha_khoa.jpg | Trên xe buýt quảng cáo nha khoa nào? | nha khoa orion | orion | 0 | open | Thấp | 8.91 |
| w13 | bang_huong_dan_y_te.jpg | Ứng dụng được hướng dẫn cài đặt tên là gì? | công dân số tphcm | công dân số tphcm | 1 | open | Trung bình | 15.10 |
| w13 | tui_cua_hang.jpg | Túi trong ảnh mang thương hiệu nào? | 7-eleven \| 7 eleven | eleven | 0 | open | Trung bình | 8.25 |
| w13 | cua_hang_dac_san.jpg | Cửa hàng bán đặc sản của địa phương nào? | hà nội | hà nội | 1 | open | Trung bình | 9.29 |
| w13 | bang_noi_quy_thang_may.jpg | Thang máy có tải trọng tối đa bao nhiêu kg? | 1150kg \| 1150 kg | 1150kg | 1 | open | Trung bình | 12.92 |
| w13 | binh_chua_chay.jpg | Trên thùng đựng bình có chữ gì? | tủ đựng bình pccc \| tủ đựng bình pccc 114 | lắp dụng bình pccc | 0 | open | Thấp | 8.71 |
| w13 | may_anh_canon.jpg | Máy ảnh trong hình có chữ Canon không? | có | có | 1 | yesno | Trung bình | 8.25 |
| w13 | chai_xit_con_trung.jpg | Bình xịt côn trùng có hương gì? | cam chanh \| hương cam chanh | cam chanh | 1 | open | Trung bình | 9.07 |
| w13 | bia_sach.jpg | Tên cuốn sách trong hình là gì? | khúc hát đêm thâu | khúc hát đêm thâu | 1 | open | Thấp | 9.26 |
| w13 | bang_go_trang_tri.jpg | Trên bảng gỗ có dòng chữ lớn màu xanh nào? | làm đi | làm đi | 1 | open | Thấp | 8.58 |
| w13 | bang_thong_tin_duoi_uoi.jpg | Con đười ươi trên biển thông tin tên gì? | tèo | tèo | 1 | open | Cao | 9.49 |
| w13 | ga_metro.jpg | Trong ảnh có chữ HCMC METRO không? | có | có | 1 | yesno | Trung bình | 8.12 |
| w13 | goi_khan_uot.jpg | Khăn ướt trong ảnh có thương hiệu nào? | lamy | lamy | 1 | open | Thấp | 8.80 |
| w13 | hop_dien_thoai.jpg | Trong ảnh có con chó thật không? | không | không | 1 | yesno | Trung bình | 9.86 |
| w13 | anh_meo_trong_khung.jpg | Trong ảnh có con chó không? | không | không | 1 | yesno | Trung bình | 7.91 |
| w13 | moc_ao.jpg | Vật kim loại treo trên cửa sổ là gì? | móc áo \| móc treo quần áo | dây | 0 | open | Thấp | 8.28 |
| w13 | bo_ho_nuoc_ban_dem.jpg | Phía trước hồ nước là bề mặt gì? | cát \| bãi cát | sân cát | 0 | open | Thấp | 8.43 |
| w13 | dan_meo_con.jpg | Trong ảnh có bao nhiêu con mèo? | 5 \| 5 con | sáu | 0 | open | Trung bình | 8.45 |
| w13 | duong_pho_chieu_ta.jpg | Trong ảnh có con chó không? | không | không | 1 | yesno | Trung bình | 7.95 |
| w13 | canh_xe_dem.jpg | Trong ảnh có bao nhiêu chiếc ô tô? | 1 \| một | 1 | 1 | open | Thấp | 8.12 |
