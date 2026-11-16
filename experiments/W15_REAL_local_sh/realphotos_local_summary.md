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
