# Công việc ứng dụng desktop

## MVP desktop

- [x] Tách ứng dụng thành cửa sổ PySide6 chạy cục bộ.
- [x] Mở webcam bằng OpenCV và tách vòng capture khỏi vòng AI.
- [x] Dùng queue giới hạn/latest frame để không tích lũy frame inference cũ.
- [x] Đo riêng capture/preview/AI FPS, latency và dropped frames.
- [x] Tải model từ manifest và xác minh SHA-256.
- [x] Kết nối YOLO với face, pose và hand analyzer cục bộ.
- [x] Vẽ detection bbox, mặt, pose landmarks và hand landmarks trên khung xem.
- [x] Kết nối temporal rules, event aggregation, risk và metadata JSONL cục bộ.
- [x] Hiển thị tình trạng từng analyzer và giữ thông điệp human review.
- [x] Hiển thị preview trực tiếp từ frame BGR, bỏ bước mã hóa rồi giải mã JPEG trên UI.
- [x] Mở webcam và chạy preview trong lúc model được tải, không đợi warmup xong mới bật camera.
- [x] Giới hạn YOLO theo lớp trong manifest ở cả lúc suy luận và lúc vẽ khung.
- [x] Chia sẻ một RGB frame giữa face/pose/hands và chặn gửi thêm inference khi lượt trước vẫn chạy.
- [x] Ẩn overlay quá 250 ms, báo khung webcam cũ sau 1 giây và hiển thị device YOLO đang dùng.
- [ ] Xác nhận cài đặt và camera thật trên máy Windows mục tiêu.
- [ ] Đo/ghi benchmark preview FPS, AI FPS, p95 latency, drop rate cùng thông số máy; không mặc định cam kết AI 15 FPS.
- [ ] Đánh giá recall/false-positive bằng bộ video được phép sử dụng, riêng cho điện thoại, hướng đầu, pose và tay.
- [ ] Đóng gói installer `.exe` và kiểm tra khởi chạy trên máy sạch.

## Sau MVP

- [ ] Lịch sử phiên và màn hình review event đầy đủ.
- [ ] Chính sách xóa/retention và export dữ liệu do giám thị chủ động yêu cầu.
- [ ] Đánh giá tùy chọn lưu evidence sau khi thống nhất yêu cầu quyền riêng tư.
- [ ] Hỗ trợ nhiều thí sinh/phòng thi nếu phần cứng và quy trình vận hành cho phép.
- [ ] Thiết lập cập nhật ứng dụng/model có xác minh chữ ký/checksum.

## Phạm vi kiến trúc hiện tại

Repository chỉ cần một ứng dụng desktop Python. Camera, suy luận, domain rules, giao diện và metadata phiên được xử lý cục bộ; kiểm thử tập trung vào core và pipeline AI.
