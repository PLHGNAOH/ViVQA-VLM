# Hand-off → Claude

- **Tác vụ đã hoàn thành:**
  - Thêm `.cursor/rules/handoff_protocol.mdc` (`alwaysApply: true`) — luật phối hợp Claude–Cursor + ràng buộc an toàn repo.
  - Ghi đè `hand_off_to_claude.md` theo cấu trúc luật mới (file này **chưa** nằm trong commit vừa push).

- **Logic cốt lõi đã thay đổi:**
  Handoff Claude↔Cursor giờ là Project Rule luôn bật: Cursor không git tự động trừ khi chỉ thị nói rõ; không xóa `data/`/`adapters/`/JSON experiments; không tuyên bố GPU/Colab đã chạy; sau mỗi việc ghi `hand_off_to_claude.md`.

- **Trạng thái verify:**
  Kiểm đọc file: header YAML `alwaysApply: true`, 27 dòng, nội dung khớp chỉ thị. Không liên quan GPU/Colab.

- **Commit + push:**
  `77fbfb4ce4d55240459257d064d73ae31a2c4f2f` — đã push `main` (`dc794e6..77fbfb4`). Chỉ gồm `.cursor/rules/handoff_protocol.mdc`. `hand_off_to_claude.md` vẫn untracked, chưa git.

- **Vấn đề tồn đọng / Lỗi phát sinh:**
  Luật `alwaysApply` có thể chưa có hiệu lực ở phiên Cursor hiện tại cho đến khi user reload/restart Cursor.

- **Cần quyết định:**
  Có commit + push `hand_off_to_claude.md` lên GitHub không (hiện chỉ local)?
