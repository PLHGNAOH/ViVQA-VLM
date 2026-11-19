# Demo set v2 — flatten + ảnh mới

Thư mục: `D:\\FPT_university\\termno9\\ViVQA-VLM\\demo_images\\` (chỉ `*.jpg` + `cau_hoi.txt`, gitignored).

- Phần 1 (CORE): 5
- Phần 2 (đúng): 30 (trong đó ảnh mới accept: 20)
- Phần 3 (sai): 8
- Local EM=1 trên ứng viên đã hỏi: **23/24** (95.8%)
- W15: skip: G:\My Drive\ViVQA-VLM\data\real_photos_w15\cau_hoi.csv không có

## Đổi tên (cũ → mới)

| Cũ | Mới |
|---|---|
| `main/00_vivqa_test_qua_bong.jpg` | `01_vivqa_test_qua_bong.jpg` |
| `main/01_cua_hang_dac_san.jpg` | `02_cua_hang_dac_san.jpg` |
| `main/02_chai_xit_con_trung.jpg` | `03_chai_xit_con_trung.jpg` |
| `main/03_bang_noi_quy_thang_may.jpg` | `04_bang_noi_quy_thang_may.jpg` |
| `main/04_duong_pho_chieu_ta.jpg` | `05_duong_pho_chieu_ta.jpg` |
| `main/05_bang_thong_tin_duoi_uoi.jpg` | `06_bang_thong_tin_duoi_uoi.jpg` |
| `main/06_ga_metro.jpg` | `07_ga_metro.jpg` |
| `main/07_canh_xe_dem.jpg` | `08_canh_xe_dem.jpg` |
| `du_phong/vivqa_test_xe_buyt.jpg` | `09_vivqa_test_xe_buyt.jpg` |
| `du_phong/bang_huong_dan_y_te.jpg` | `10_bang_huong_dan_y_te.jpg` |
| `du_phong/may_anh_canon.jpg` | `11_may_anh_canon.jpg` |
| `du_phong/hop_dien_thoai.jpg` | `12_hop_dien_thoai.jpg` |
| `du_phong/bang_go_trang_tri.jpg` | `13_bang_go_trang_tri.jpg` |
| `du_phong/bia_sach.jpg` | `14_bia_sach.jpg` |
| `du_phong/goi_khan_uot.jpg` | `15_goi_khan_uot.jpg` |
| `gioi_han/xe_buyt_nha_khoa.jpg` | `91_sai_xe_buyt_nha_khoa.jpg` |
| `gioi_han/tui_cua_hang.jpg` | `92_sai_tui_cua_hang.jpg` |
| `gioi_han/binh_chua_chay.jpg` | `93_sai_binh_chua_chay.jpg` |
| `gioi_han/moc_ao.jpg` | `94_sai_moc_ao.jpg` |
| `gioi_han/bo_ho_nuoc_ban_dem.jpg` | `95_sai_bo_ho_nuoc_ban_dem.jpg` |
| `gioi_han/dan_meo_con.jpg` | `96_sai_dan_meo_con.jpg` |
| `gioi_han/anh_meo_trong_khung.jpg` | `97_sai_anh_meo_trong_khung.jpg` |

## Ảnh mới được chọn

| File | Nguồn | id | Hỏi | Local | Mức |
|---|---|---|---|---|---|
| `16_test_46409_noi_dung_tren_bang.jpg` | vitextvqa_test | test_46409 | nội dung trên bảng màu xanh là gì ? | các lớp sau | Cao |
| `17_test_2410_khau_hien_tren_toa.jpg` | vitextvqa_test | test_2410 | khẩu hiện trên tòa nhà nói về việc làm gì để đàng hoàng hơn to đẹp hơn ? | xây dựng lại đất nước ta | Cao |
| `18_test_21749_bai_tho_tuc_canh.jpg` | vitextvqa_test | test_21749 | bài thơ tức cảnh pác bó do ai viết ? | hồ chí minh | Trung bình |
| `19_test_37779_gia_cua_mot_phan.jpg` | vitextvqa_test | test_37779 | giá của một phần mì vịt tiềm là bao nhiêu ? | 40k | Cao |
| `20_test_18809_chiec_xe_may_mau.jpg` | vitextvqa_test | test_18809 | chiếc xe máy màu xe góc phải bức hình là của thương hiệu nào ? | honda | Cao |
| `21_test_43969_ten_kinh_doanh_cua.jpg` | vitextvqa_test | test_43969 | tên kinh doanh của quán ốc trên ? | xấu quắc | Trung bình |
| `22_test_21624_buc_anh_chup_ha.jpg` | vitextvqa_test | test_21624 | bức ảnh chụp hà nội vào năm nào ? | 2020 | Cao |
| `23_test_20622_com_ga_o_au.jpg` | vitextvqa_test | test_20622 | cơm gà ở đâu ? | chợ 200 | Cao |
| `24_test_30954_nhung_mon_ngon_am.jpg` | vitextvqa_test | test_30954 | những món ngon ẩm thực đến từ nước nào ? | thái lan | Trung bình |
| `25_test_22947_san_pham_cua_ai.jpg` | vitextvqa_test | test_22947 | sản phẩm của ai ? | vinamilk | Trung bình |
| `26_test_33970_phan_qua_tri_gia.jpg` | vitextvqa_test | test_33970 | phần quà trị giá bao nhiêu ? | 200k | Cao |
| `27_test_29137_cua_hang_nay_ban.jpg` | vitextvqa_test | test_29137 | cửa hàng này bán gì ? | trái cây & rau quả | Cao |
| `28_test_38444_hanh_tay_o_ay.jpg` | vitextvqa_test | test_38444 | hành tây ở đây có giá bao nhiêu ? | 33 . 900 đ / kg | Trung bình |
| `29_test_1036_con_meo_en_trang.jpg` | vivqa_test | test_1036 | con mèo đen trắng ở đâu | cái rổ | Trung bình |
| `30_test_556_mau_cua_chiec_binh.jpg` | vivqa_test | test_556 | màu của chiếc bình là gì | màu đỏ | Cao |
| `31_test_159_mau_sac_cua_i.jpg` | vivqa_test | test_159 | màu sắc của đi bộ là gì | màu đen | Trung bình |
| `32_test_2741_co_bao_nhieu_xe.jpg` | vivqa_test | test_2741 | có bao nhiêu xe ngựa ở phía trước của một lâu đài | ba | Trung bình |
| `33_test_2454_nhung_gi_trong_phong.jpg` | vivqa_test | test_2454 | những gì trong phòng khách nằm trên đi văng | con chó | Cao |
| `34_test_1043_wc_o_au.jpg` | vivqa_test | test_1043 | wc ở đâu | phòng tắm | Trung bình |
| `35_test_547_mau_cua_man_hinh.jpg` | vivqa_test | test_547 | màu của màn hình là gì | màu cam | Trung bình |

## Ứng viên đã thử (rút gọn)

Xem JSON: `demo_set_v2_summary.json` (`candidates_tried`, n=46).

## Bộ demo v3 (19/11/2026)

Theo yêu cầu Hoàng: giữ **13 ảnh**, không đánh số lại; chuyển vào Recycle Bin các số **05, 07, 08, 09, 12, 16, 17, 18, 20, 22, 23, 24** và **toàn bộ từ 26 trở đi** (26–35 và 91–98). Bỏ Phần 3 trong `cau_hoi.txt`. Phần 1 = DEMO CHÍNH (01–04).

| File | Câu hỏi | Đáp án | Nguồn |
|---|---|---|---|
| `01_vivqa_test_qua_bong.jpg` | màu của quả bóng là gì | màu cam | vivqa_test |
| `02_cua_hang_dac_san.jpg` | Cửa hàng bán đặc sản của địa phương nào? | hà nội | w13_real |
| `03_chai_xit_con_trung.jpg` | Bình xịt côn trùng có hương gì? | cam chanh \| hương cam chanh | w13_real |
| `04_bang_noi_quy_thang_may.jpg` | Thang máy có tải trọng tối đa bao nhiêu kg? | 1150kg \| 1150 kg | w13_real |
| `06_bang_thong_tin_duoi_uoi.jpg` | Con đười ươi trên biển thông tin tên gì? | tèo | w13_real |
| `10_bang_huong_dan_y_te.jpg` | Ứng dụng được hướng dẫn cài đặt tên là gì? | công dân số tphcm | w13_real |
| `11_may_anh_canon.jpg` | Máy ảnh trong hình có chữ Canon không? | có | w13_real |
| `13_bang_go_trang_tri.jpg` | Trên bảng gỗ có dòng chữ lớn màu xanh nào? | làm đi | w13_real |
| `14_bia_sach.jpg` | Tên cuốn sách trong hình là gì? | khúc hát đêm thâu | w13_real |
| `15_goi_khan_uot.jpg` | Khăn ướt trong ảnh có thương hiệu nào? | lamy | w13_real |
| `19_test_37779_gia_cua_mot_phan.jpg` | giá của một phần mì vịt tiềm là bao nhiêu ? | 40k | vitextvqa_test |
| `21_test_43969_ten_kinh_doanh_cua.jpg` | tên kinh doanh của quán ốc trên ? | xấu quắc | vitextvqa_test |
| `25_test_22947_san_pham_cua_ai.jpg` | sản phẩm của ai ? | vinamilk | vitextvqa_test |

