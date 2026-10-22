# Phương pháp đánh giá và giới hạn của số liệu

Tài liệu này giải thích **số liệu đánh giá chứng minh được điều gì và không chứng minh được điều gì**.
Đọc kèm [kết quả đánh giá](evaluation-results.md) và
[spec 002](../specs/002-phase1-hardening/spec.md).

## 1. Nguyên tắc: không tuyên bố độ chính xác khi chưa có holdout

Một con số độ phủ/báo nhầm chỉ được dùng để **ra quyết định phát hành** khi thoả cả bốn điều kiện:

1. **Nhãn độc lập với hệ thống.** Người gán nhãn không xem output của hệ thống và không phải người viết
   luật phát hiện.
2. **Tách holdout theo người và thiết bị.** Tập huấn luyện/tinh chỉnh ngưỡng và tập báo cáo không dùng
   chung thí sinh, cùng không dùng chung webcam/ánh sáng/bố cục.
3. **Đúng bối cảnh triển khai.** Video webcam ngồi bàn, đúng khoảng cách, đúng độ phân giải và đúng
   điều kiện ánh sáng của phòng thi thật.
4. **Đủ mẫu cho từng tín hiệu bật mặc định.** Với tín hiệu hiếm, cần đủ số khoảng nhãn để khoảng tin cậy
   của độ phủ không rộng hơn ±10 điểm.

Nếu thiếu một điều kiện, số liệu chỉ là **chỉ báo kỹ thuật nội bộ**, không dùng trong tài liệu bán hàng,
hợp đồng hay cam kết với cơ sở giáo dục.

## 2. OEP chứng minh được gì

Bộ OEP (Atoum et al., IEEE TMM 2017) hữu ích cho:

- Kiểm tra pipeline phát lại chạy đầu-cuối trên video thật, ổn định qua nhiều thí sinh.
- So sánh **trước/sau** một thay đổi luật trên cùng dữ liệu (cùng ngưỡng, cùng nhịp phân tích).
- Phát hiện hồi quy rõ ràng: độ phủ tụt mạnh, báo nhầm tăng mạnh, hoặc FPS giảm.
- Kiểm tra tính tất định: cùng video + cùng cấu hình cho cùng danh sách sự kiện.

## 3. OEP không chứng minh được gì

| Giới hạn | Hệ quả |
|---|---|
| Camera đeo trên kính (wearcam) của thí sinh | YOLO nhận thiết bị là điện thoại → báo nhầm điện thoại không đại diện môi trường thi thật |
| Nhãn OEP gộp nhiều hành vi vào một mã | Một khoảng "xem sách/ghi chú" có thể khớp cả quay đầu lẫn hướng mắt; độ phủ theo từng loại chỉ là **cận dưới** |
| Không có nhãn cho "nói chuyện" và "hướng mắt" theo đúng định nghĩa sản phẩm | Không thể kết luận các tín hiệu thử nghiệm đạt yêu cầu |
| Bố cục phòng thi khác (một người, bàn học, ánh sáng) | Ngưỡng tối ưu trên OEP có thể sai trên webcam mục tiêu |
| Không có ca thi giấy, bộ bài/ví, che camera | SC-005 và các ca hard-negative chưa được đo |

Vì vậy các dòng `GAZE_*`, `POSSIBLE_TALKING`, `DOCUMENT_DETECTED` trong báo cáo OEP **không** được dùng
để bật mặc định các tính năng đó.

## 4. Chấm điểm: một dự đoán chỉ khớp một nhãn

Từ thay đổi này, `_score()` trong
[`matching.py`](../packages/proctoring_core/src/proctoring_core/evaluation/matching.py) gán mỗi sự kiện dự
đoán cho **tối đa một khoảng nhãn** cùng loại:

- Processing theo thứ tự thời gian, chọn ứng viên gần nhất (ưu tiên sự kiện bắt đầu trong khoảng).
- Một sự kiện không thể đồng thời tính TP cho hai khoảng nhãn chồng lấn.
- Với nhãn nhiều loại, mỗi loại chấm riêng; `ANY` khớp một lần cho mỗi khoảng.
- "Trùng lặp" chỉ tính khi **cùng một tín hiệu** mở nhiều sự kiện trong một khoảng nhãn. Nhiều tín hiệu
  khác nhau cùng phủ một khoảng không bị coi là trùng lặp.
- Sự kiện thừa nằm trong khoảng nhãn là trùng lặp, không phải báo nhầm. Báo nhầm chỉ tính sự kiện không
  khớp nhãn nào và không nằm trong vùng bỏ qua.

Chấm lại (`--rescore`) dùng đúng dung sai của tệp ánh xạ nhãn, không dùng mặc định 2000 ms.

## 5. Cổng trước khi bật mặc định một tín hiệu

Một tín hiệu chỉ được bật trong `configs/default.yaml` khi có báo cáo holdout đạt tất cả:

| Cổng | Ngưỡng tối thiểu |
|---|---|
| Độ phủ trên holdout | ≥ 70% số khoảng nhãn của tín hiệu đó |
| Báo nhầm trên đoạn làm bài bình thường | ≤ 0.5 lần/phút/phiên |
| Trễ phát hiện trung vị | ≤ 3 giây so với nhãn |
| Số sự kiện mỗi khoảng | ≤ 1.2 (không phân mảnh) |
| Tất định | Hai lần chạy cùng video + cấu hình cho cùng danh sách sự kiện |
| Suy giảm FPS | ≤ 15% so với mốc trước khi thêm tín hiệu |

Không đạt cổng nào → giữ tắt mặc định, ghi rõ "chưa đánh giá" trong UI và tài liệu.

## 6. Thay đổi ngưỡng phải kèm số đo

Mỗi lần sửa ngưỡng trong `configs/default.yaml`:

1. Chạy lại trên tập con cố định (cùng danh sách ca) để so sánh công bằng.
2. Ghi trước/sau vào [event-detection-tuning.md](event-detection-tuning.md) và
   [evaluation-results.md](evaluation-results.md).
3. Ghi rõ cấu hình nào đã tạo ra số liệu (mã băm cấu hình trong `report.json`).
4. Không trộn số liệu của hai cấu hình khác nhau trong cùng một bảng so sánh.
