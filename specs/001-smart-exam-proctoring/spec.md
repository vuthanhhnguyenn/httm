# Đặc tả: Smart Exam Proctoring Desktop

**Ngày cập nhật:** 2026-09-23
**Trạng thái:** MVP desktop đang được hoàn thiện
**Nguồn yêu cầu:** README và các yêu cầu đã xác nhận: chạy cục bộ, webcam preview trực tiếp, hiển thị khung nhận diện và giữ giám thị là người quyết định cuối cùng.

## Mục tiêu

Giám thị mở một ứng dụng trên máy Windows, chọn webcam và theo dõi một thí sinh trong một phiên. Ảnh webcam được xử lý ngay trên máy, không chuyển qua máy chủ. Cửa sổ phải hiển thị preview, khung người/vật thể/mặt/tư thế/bàn tay, FPS đo thực tế, tình trạng từng bộ phân tích và các tín hiệu cần rà soát.

Ứng dụng hỗ trợ rà soát, không tự kết luận thí sinh gian lận, nhận dạng danh tính hay áp dụng hình phạt.

## Người dùng và phạm vi

- Người dùng chính: giám thị vận hành phiên thi trên laptop hoặc desktop.
- MVP: một camera cục bộ và một thí sinh dự kiến.
- Lưu metadata sự kiện cục bộ dạng JSONL; preview ở RAM. Ảnh bằng chứng cho
  event đang bật theo cấu hình và có thời hạn lưu, không ghi video toàn phiên.
- Không yêu cầu tài khoản từ xa hay đồng bộ dữ liệu qua mạng.
- Mở rộng nhiều thí sinh/phòng thi và gói cài đặt Windows là các hạng mục sau MVP.

## Luồng chính

1. Mở cửa sổ desktop và chọn thiết bị camera.
2. Bắt đầu phiên; ứng dụng khởi tạo camera và model, hiển thị rõ bộ phân tích nào sẵn sàng hoặc lỗi.
3. Preview tiếp tục được cập nhật độc lập với tốc độ suy luận. Các khung nhận diện chỉ hiển thị khi còn mới.
4. Ứng dụng tổng hợp điều kiện kéo dài thành sự kiện có thời điểm, mức độ, độ tin cậy và giải thích.
5. Giám thị xem tín hiệu, tự đánh giá, rồi dừng phiên để giải phóng camera.

## Yêu cầu chức năng

- FR-01: Mở camera bằng thiết bị OpenCV cục bộ; báo lỗi dễ hiểu nếu không thể mở hoặc đọc frame.
- FR-02: Hiển thị preview trực tiếp và đo riêng Preview FPS, AI FPS, độ trễ và frame bị bỏ qua.
- FR-03: Dùng latest-frame queue giới hạn để AI không xử lý một hàng frame cũ kéo dài.
- FR-04: Vẽ bbox vật thể/người, mặt, điểm pose và bàn tay trên preview, gắn kết quả với frame đã phân tích.
- FR-05: Hiển thị trạng thái và lý do lỗi riêng cho YOLO, face, pose và hand analyzer.
- FR-06: Áp dụng rule thời gian cho quay/cúi đầu, thiếu người/mặt, điện thoại và tín hiệu hoạt động tay khi analyzer tương ứng sẵn sàng.
- FR-07: Chỉ tạo event sau khi điều kiện thỏa threshold và duration; event có giải thích để giám thị rà soát.
- FR-08: Tổng hợp risk như mức ưu tiên rà soát; không thể hiện như kết luận vi phạm.
- FR-09: Ghi metadata event vào `storage/sessions/<session-id>/events.jsonl`;
  ảnh bằng chứng theo cấu hình có thời hạn lưu, không ghi video phiên mặc định.
- FR-10: Dừng phiên an toàn, giải phóng camera/model/runtime và xóa buffer ảnh khỏi RAM.
- FR-11: Xác minh checksum model theo manifest trước khi dùng; báo model thiếu hoặc sai hash trên giao diện.

## Tiêu chí chấp nhận

- Khi camera hoạt động, preview thay đổi liên tục và FPS phản ánh số frame thực sự được vẽ, không lấy từ target cấu hình.
- AI FPS đo số frame suy luận hoàn tất; màn hình phân biệt rõ tốc độ preview và tốc độ AI.
- Nếu inference chậm hơn camera, frame cũ bị bỏ thay vì tạo độ trễ tích lũy.
- Nếu analyzer lỗi hoặc model chưa cài, giao diện báo degraded/unavailable và không giả vờ đã nhận diện.
- Khi điện thoại hoặc hướng đầu thỏa rule đã cấu hình đủ thời gian, event xuất hiện một lần với giải thích; tín hiệu ngắn không tạo event kéo dài.
- Nút dừng kết thúc phiên và cho phép mở lại camera mà không phải khởi động lại ứng dụng.
- Ứng dụng hoạt động không cần dịch vụ nền hoặc kết nối mạng sau khi model đã có trên máy.

## Giới hạn đã biết

- FPS suy luận phụ thuộc GPU/CPU và model; mục tiêu preview 20–30 FPS không đồng nghĩa AI cũng chạy ở cùng tốc độ.
- COCO YOLO có thể bỏ sót điện thoại nhỏ/xa/bị che; không có lớp tổng quát đủ tin cậy cho mọi tài liệu.
- Hướng đầu dựa trên landmark không phải eye gaze đã hiệu chuẩn.
- Tín hiệu khuôn mặt không nhận dạng danh tính.
- Bản hiện tại chạy từ mã nguồn bằng Python; chưa phải bộ cài `.exe` độc lập.

## Đề xuất mở rộng core (đang lập kế hoạch, chưa triển khai)

Nguồn đầu vào là tài liệu thiết kế core do người dùng cung cấp và ca báo nhầm bộ bài
thành điện thoại. Phần này là yêu cầu cho giai đoạn tiếp theo, không mô tả tính năng
đã hoạt động.

- FR-12: Mỗi cảnh báo vật thể phải truy được về đúng frame đã phân tích, bbox,
  confidence, model/version, object track và thời điểm; giám thị có thể đánh dấu
  báo nhầm. Không diễn giải confidence YOLO như xác suất gian lận.
- FR-13: Một vật thể xuất hiện liên tục chỉ mở một episode sự kiện; mất detection
  ngắn hoặc đổi person track không được tự tạo cảnh báo điện thoại thứ hai.
- FR-14: Đánh giá và giảm nhầm điện thoại với bộ bài, ví, điều khiển, sách và vật
  hình chữ nhật bằng tập video có nhãn. Mọi điều chỉnh model/ngưỡng phải báo cả
  recall điện thoại và báo nhầm, không chỉ tốc độ.
- FR-15: Hiệu chỉnh hướng đầu có thể tự đề xuất khi quan sát ổn định, nhưng phải
  hiển thị trạng thái, từ chối baseline không chắc chắn và giữ thao tác hiệu chỉnh
  thủ công. Không dùng một góc nhìn lệch ban đầu làm mốc trung tính im lặng.
- FR-16: Gaze và chuyển động môi, nếu bổ sung, chỉ là tín hiệu thử nghiệm để rà
  soát; phải có chất lượng quan sát, hiệu chuẩn, thời hạn dữ liệu và đánh giá riêng.
  Không mặc định suy ra nhìn tài liệu, nói chuyện hay gian lận từ một chỉ số.
- FR-17: Cải thiện rời chỗ/camera bị che bằng nhiều dấu hiệu có theo dõi thời gian,
  tách lỗi camera/thiếu dữ liệu khỏi hành vi; ngưỡng lấy từ dữ liệu mục tiêu.
- FR-18: Correlation chỉ ghép event đã xác nhận, cùng phiên/người, còn mới và có
  analyzer khỏe; chống cộng trùng, có cap và liên kết event nguồn.
- FR-19: Risk hiện tại tiếp tục decay khi hành vi dừng; số event, điểm tích lũy và
  điểm đỉnh chỉ là lịch sử phiên, không biến thành floor điểm hiện tại.

Điều kiện chấp nhận trước khi bật mặc định: bộ video độc lập có cả trường hợp âm
tính khó (đặc biệt bộ bài), báo cáo precision/recall và cảnh báo nhầm/phút, độ trễ
event, FPS/p95 latency trên cấu hình máy đã ghi rõ, kiểm tra riêng từng nhóm điều
kiện và xác nhận quyền lưu/xóa mẫu thử. Các ngưỡng và mục tiêu định lượng cụ thể
được chốt sau khi có baseline; không lấy các số trong tài liệu đầu vào làm chuẩn.
