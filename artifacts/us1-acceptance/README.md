# Tình trạng MVP desktop

Ứng dụng hiện có khung desktop PySide6, webcam qua OpenCV, pipeline YOLO/MediaPipe chạy tại chỗ, overlay và event/risk cục bộ. Tài liệu này không khẳng định đã đạt benchmark phần cứng hay độ chính xác nghiệp vụ.

Trước khi coi là checkpoint phát hành, cần chạy app trên máy Windows với webcam thật, xác minh khung mặt/vật thể và ghi lại preview FPS, AI FPS, p95 latency, dropped frames, cấu hình model và thông tin GPU/CPU. Bộ video kiểm thử phải có quyền sử dụng phù hợp.

Để khởi chạy, xem [quickstart](../../specs/001-smart-exam-proctoring/quickstart.md). Cửa sổ và pipeline AI được mở cùng ứng dụng desktop.
