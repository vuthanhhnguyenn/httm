# Product

## Platform

Native desktop application for Windows, with local camera capture and local AI inference.

## Users

Giám thị vận hành một phiên thi trên máy tính, xem webcam trực tiếp và rà soát các tín hiệu cần chú ý.

## Product purpose

Smart Exam Proctoring phân tích hình ảnh webcam theo thời gian, hiển thị khung người/mặt/vật thể/tư thế/bàn tay, và tổng hợp các tín hiệu kéo dài thành sự kiện có thể xem lại. AI không kết luận thí sinh gian lận; giám thị đưa ra đánh giá cuối cùng.

## Operating context

Ứng dụng mở camera trực tiếp qua OpenCV. Preview được vẽ trên cửa sổ desktop độc lập với worker inference; camera, AI và giao diện cùng chạy cục bộ. Model được tải một lần vào thư mục máy và xác minh SHA-256 trước khi sử dụng.

## Capabilities and constraints

- Phase 1 tập trung vào một webcam và một thí sinh.
- Đo riêng preview FPS, AI FPS, độ trễ và frame bỏ qua. Giao diện phải trình bày số đo thực, không biến tốc độ mục tiêu thành kết quả.
- YOLO nhận diện các lớp COCO như person, cell phone và book; vật thể nhỏ hoặc bị che có thể bị bỏ sót.
- MediaPipe Tasks cung cấp face landmarks, ước lượng hướng đầu, pose landmarks và hand landmarks. Đây không phải nhận dạng danh tính hay phép đo ánh mắt đã hiệu chuẩn.
- Hành vi chỉ tạo sự kiện sau khi thỏa ngưỡng confidence và thời lượng đã cấu hình.
- Sự kiện và risk hỗ trợ ưu tiên rà soát, không tự áp dụng hình phạt.
- Không ghi video toàn phiên mặc định. Preview được giữ trong RAM; metadata sự kiện được lưu cục bộ.

## Brand commitments

- Tên: Smart Exam Proctoring.
- UI tiếng Việt; giữ các tên kỹ thuật phổ biến như YOLO, MediaPipe và FPS.
- Dashboard giám sát tối màu với preview webcam là vùng chính.
- Giám thị giữ quyền quyết định cuối cùng.
