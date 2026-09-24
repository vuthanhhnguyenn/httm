# Dàn ý slide: Core AI của Smart Exam Proctoring

## Thông tin trình bày

- **Chủ đề:** Core AI hiện tại của ứng dụng giám sát thi trên desktop.
- **Mục đích:** Giải thích core biến đầu ra của model thành tín hiệu hành vi, xác nhận sự kiện đáng rà soát, tính risk và xử lý các giới hạn nhận diện hiện tại.
- **Người nghe:** Chủ dự án và người có hiểu biết cơ bản về ứng dụng desktop, webcam và AI.
- **Thời lượng:** Khoảng 5 phút. Người dùng yêu cầu khoảng 9 slide, vì vậy dàn ý có 9 slide tính cả trang bìa.
- **Ngôn ngữ:** Tiếng Việt. Giữ các tên phổ biến như YOLO, MediaPipe, GPU, FPS, CPU.
- **Định dạng:** PowerPoint 16:9, nội dung và sơ đồ có thể chỉnh sửa.
- **Giọng điệu:** Thẳng, gọn, mô tả đúng trạng thái code và phép đo. Phân biệt rõ tính năng hiện có, giới hạn và kế hoạch.

## Nội dung nguồn

- `README.md`: cách chạy app desktop, model, GPU, màn hình và giới hạn.
- `docs/core-project-summary.md`: kiến trúc, bộ phân tích, rule, event, risk, bằng chứng và roadmap.
- `docs/ai-stability-performance-plan.md`: thay đổi xử lý, số đo GPU trên ảnh tổng hợp và kế hoạch hiệu năng.
- `docs/event-detection-tuning.md`: cách xác nhận điện thoại và quay đầu.
- Cấu hình `configs/default.yaml`, manifest `models/manifest.yaml` và mã nguồn trong `apps/desktop`, `packages/proctoring_ai`, `packages/proctoring_core`.

## Ba hướng hình ảnh

1. **Technical editorial, được chọn:** nền sáng ngà, chữ than đậm, màu teal cho luồng xử lý, màu hổ phách cho cảnh báo. Bố cục thoáng như một tài liệu kỹ thuật có biên tập. Dùng sơ đồ native có thể chỉnh sửa.
2. **Phòng điều khiển tối:** nền navy, chữ trắng, teal và cam cho trạng thái. Gợi giao diện ứng dụng nhưng có thể khiến deck giống dashboard.
3. **Sổ tay nghiên cứu:** nền trắng, chữ than, đỏ gạch làm điểm nhấn. Tập trung vào cơ chế, phép đo và giới hạn mô hình.

## Dàn ý 9 slide

1. **Trang bìa, title.** Core AI giám sát thi. Smart Exam Proctoring, trạng thái hiện tại.
2. **Đường đi từ frame đến quyết định, diagram.** Webcam/OpenCV, hàng đợi frame mới nhất, analyzer, observation, bộ luật, EventAggregator, CorrelationEngine và RiskEngine. Nhấn mạnh model tạo quan sát, còn luật theo thời gian quyết định lúc nào mở event.
3. **Quan sát mà model cung cấp, comparison.** YOLO phát hiện người/điện thoại/sách; MediaPipe cung cấp landmark mặt, hướng đầu, pose và bàn tay. Ghi rõ YOLO có thể dùng CUDA, MediaPipe chạy CPU trong cấu hình Windows hiện tại.
4. **Điện thoại: từ box đến event, diagram.** Giải thích object track, yêu cầu xác nhận cùng vật qua nhiều lượt, ngưỡng bình thường và đường confidence cao, khoảng hụt cho phép, cùng cơ chế tránh hai event cho một lần xuất hiện.
5. **Quay đầu: baseline và thời gian duy trì, diagram.** Mốc trung tính sau hiệu chỉnh, yaw/pitch/roll, lọc góc, ngưỡng bắt đầu/thoát và bộ đếm lượt quay đã hoàn tất. Phân biệt hướng đầu với hướng mắt.
6. **Tín hiệu hỗ trợ và correlation, comparison.** Tay gần mặt/hoạt động tay, nhiều người, cúi đầu và safety. Nêu ba luật ghép hiện có, thời hạn dữ liệu, cùng track khi cần và điểm tối đa. Tổ hợp là tín hiệu rà soát.
7. **Event, risk và ảnh bằng chứng, diagram.** Event được mở/cập nhật/đóng; risk hiện tại chỉ giữ sàn khi event còn mới; điểm lịch sử lưu riêng; ảnh bằng chứng lấy từ đúng frame phân tích và ghi nền.
8. **Chất lượng và hiệu năng hiện tại, metrics.** Đưa số benchmark ảnh tổng hợp M1200, phân biệt với FPS webcam trong ảnh chụp trước bản sửa. Nêu nhầm điện thoại, bộ bài, giới hạn lớp COCO và các heuristic còn yếu.
9. **Thứ tự cải thiện core, timeline.** P0 dữ liệu và benchmark; P1 giảm nhầm/bỏ sót điện thoại; P2 cải thiện baseline mặt, ghế và camera; P3 gaze/MAR thử nghiệm; P4 đo hiệu quả correlation và báo cáo.

## Quy tắc nội dung

- Không gọi benchmark ảnh tổng hợp là FPS webcam.
- Không mô tả hướng đầu như hướng mắt.
- Không nói hệ thống tự kết luận gian lận.
- Ghi rõ `POSSIBLE_TALKING` chưa có analyzer runtime cấp dữ liệu.
- Số liệu và ngưỡng luôn gắn với cấu hình hoặc phép đo tương ứng.
- Nguồn nội bộ được đặt trong speaker notes để slide vẫn gọn.
