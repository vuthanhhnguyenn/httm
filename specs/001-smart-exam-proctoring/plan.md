# Kế hoạch kỹ thuật: ứng dụng desktop cục bộ

## Quyết định kiến trúc

Ứng dụng dùng một tiến trình Python với cửa sổ PySide6. OpenCV lấy frame trực tiếp từ webcam; YOLO và MediaPipe chạy cục bộ; domain core tạo event và risk. Camera, phân tích và hiển thị thuộc cùng ứng dụng trên máy.

```text
Webcam -> OpenCV capture -> latest-frame queue -> ObservationPipeline
   |                                               |
   +-> raw BGR preview buffer (RAM) -> Qt paint    +-> YOLO manifest classes
                                                   +-> MediaPipe face/pose/hands
                                                   +-> Core rules -> event/risk
                                                                      |
                                                                      +-> JSONL metadata
```

Capture và inference có nhịp riêng. Queue giới hạn (cấu hình hiện là hai frame)
giữ frame mới, tránh backlog. Giao diện đọc preview qua buffer có khóa; signal Qt
chỉ mang metadata, không đẩy frame ảnh qua event queue.

## Cấu trúc module

```text
apps/desktop/                 Cửa sổ Qt và điều phối vòng đời phiên
packages/proctoring_ai/       OpenCV, YOLO, MediaPipe, pipeline và metrics
packages/proctoring_core/     Domain session, behavior rules, event và risk
configs/                      Cấu hình camera, ngưỡng và chính sách thi
models/manifest.yaml          URL, license, phiên bản và checksum model
storage/sessions/             Metadata cục bộ dạng JSONL
```

## Luồng dữ liệu và vòng đời

1. Desktop mở camera và bắt đầu preview trước khi warmup model; trạng thái analyzer hiện sau khi khởi tạo model.
2. Camera capture tạo frame ID và timestamp monotonic. Buffer preview giữ frame BGR mới nhất trong RAM, không mã hóa rồi giải mã JPEG.
3. Latest-frame queue bỏ frame cũ khi inference không theo kịp capture.
4. Pipeline trả observation cho cùng frame; UI vẽ bbox/keypoint chỉ khi kết quả chưa quá hạn.
5. Core temporal rules và event aggregator tạo event; RiskEngine cập nhật điểm ưu tiên rà soát.
6. Event metadata được append vào JSONL. Ảnh bằng chứng đúng frame phân tích
   được ghi nền khi cấu hình cho phép, theo thời hạn lưu; không ghi video toàn phiên.
7. Stop/hỏng worker hủy task, đóng camera và adapter model, xóa preview buffer.

## Cấu hình và dependency

- Python 3.12, `uv`, PySide6, OpenCV, PyTorch/Ultralytics, MediaPipe, Pydantic và PyYAML.
- `configs/default.yaml` cấu hình thiết bị, resolution, target FPS, YOLO confidence và kích thước ảnh.
- `configs/exam_policy.yaml` cấu hình vật dụng được phép.
- Mỗi model phải có license và SHA-256 trong `models/manifest.yaml`; tải model là bước riêng, không tải ngầm lúc mở camera.
- `opencv-python` dành cho máy có giao diện desktop; CI có thể dùng headless.

## An toàn, quyền riêng tư và giới hạn

- Không gửi frame, landmark hoặc event qua mạng.
- Không lưu video toàn phiên; frame preview sống trong RAM.
- Event/risk chỉ gợi ý cho người vận hành; không có quyết định tự động về gian lận.
- Analyzer chậm/thiếu phải hiện degraded/unavailable. FPS được đo riêng, không được báo bằng target cấu hình.
- Mỗi analyzer chỉ có tối đa một lượt suy luận native đang chạy. Khi lượt đó quá timeout, frame mới được bỏ qua cho tới khi worker cũ kết thúc thay vì xếp thêm hàng.
- YOLO chỉ suy luận các lớp ghi trong manifest; lớp ngoài danh sách được lọc thêm trước khi vẽ overlay.
- Giám thị có thể xem device YOLO thực tế; chọn CUDA cần PyTorch và GPU tương thích.
- Mục tiêu preview 20–30 FPS cần xác nhận trên máy thực tế; AI FPS thay đổi theo phần cứng.

## Kiểm chứng trước phát hành

- Xác nhận lệnh cài và mở cửa sổ trên Windows/Python 3.12.
- Thử mở/dừng camera nhiều lần, mất camera, model thiếu và checksum không hợp lệ.
- Đo preview FPS, AI FPS, p95 latency và drop rate trên hardware target.
- Kiểm tra overlay bám đúng frame và không tồn tại bbox quá hạn.
- Chưa xem target 15 FPS AI là đạt cho đến khi có benchmark có cấu hình phần cứng ghi kèm.

## Kế hoạch mở rộng core: bằng chứng, độ chính xác và tín hiệu thử nghiệm

### Technical Context và phạm vi

Giữ ứng dụng desktop một tiến trình, OpenCV/YOLO11n/MediaPipe/PySide6 và
`ObservationBundle` gắn với một frame. Không đổi sang web/API. Baseline hiện tại:
YOLO COCO chỉ nạp person/cell phone/book; face adapter có landmark nhưng chỉ xuất
bbox/góc đầu; `POSSIBLE_TALKING` chưa có dữ liệu; `CorrelationEngine` đã có ba
luật; `RiskEngine` đã tách điểm hiện tại với lịch sử phiên; evidence JPEG đã ghi
nền và có retention. `image_quality` đã có brightness/entropy/blur nhưng
`occlusion_estimate` cố định bằng 0. Ghi nhận một ca bộ bài bị báo điện thoại hai
lần; chưa có bbox/confidence gốc để kết luận vị trí lỗi cụ thể.

Đầu ra mới chỉ là **tín hiệu cần rà soát**, không phải kết luận gian lận. Chưa thêm
microphone, face recognition, lưu video toàn phiên hoặc adaptive risk decay.

### Constitution Check — trước và sau thiết kế

| Nguyên tắc | Quyết định thiết kế / cổng kiểm tra |
| --- | --- |
| Human review | Đổi cách diễn đạt thành nghi vấn; evidence ghi bbox/confidence/model và cho phép gắn nhãn báo nhầm. Không có verdict tự động. |
| Temporal & multi-signal | Không mở event từ một frame; gaze/mouth chỉ thử nghiệm; correlation chỉ dùng event đã xác nhận. |
| Config ngoài code | Ngưỡng, cửa sổ thời gian, cooldown và feature flags vào schema/config; giá trị chỉ chốt sau benchmark. |
| AI tách business | Face/YOLO xuất observation; rules/event/risk ở core; model không quyết định “gian lận”. |
| Privacy | Chỉ lưu frame quanh event khi bật; dữ liệu thử có đồng ý, được ẩn danh, có thời hạn lưu/xóa. Không lưu landmarks thô mặc định. |
| Measured quality | Không bật mặc định tín hiệu mới trước replay có nhãn và so sánh báo nhầm, bỏ sót, độ trễ, FPS trên phần cứng mục tiêu. |

Không có ngoại lệ hiến pháp. Thiết kế sau Phase 1 giữ nguyên cả sáu cổng; đề xuất
“session risk floor” và “phone bất kể confidence + tay = CRITICAL” trong tài liệu
đầu vào **không qua cổng** human review/risk bounded/false-positive nên bị loại.

### Thứ tự triển khai đề xuất

| Giai đoạn | Nội dung | Điều kiện chuyển tiếp |
| --- | --- | --- |
| P0 — Dữ liệu & đo gốc | Chốt kịch bản có đồng ý sử dụng; thu các clip điện thoại thật, bộ bài và vật dễ nhầm ở nhiều ánh sáng/góc; gán bbox + thời gian bắt đầu/kết thúc; đo confusion, báo nhầm/phút, recall, p50/p95 alert delay và FPS. Sửa replay để thực sự chạy pipeline, không chỉ in manifest. | Có baseline tái lập và tập holdout không dùng để chỉnh ngưỡng. |
| P1 — Sửa cảnh báo điện thoại | Lưu snapshot bbox/confidence/track/model đúng frame trong metadata evidence và vẽ trên ảnh review; có nhãn giám thị “báo nhầm”; định danh episode theo object track thay vì person track; chống đóng/mở lại khi detection chập chờn. Thử threshold theo precision–recall; nếu chưa đủ, bổ sung hard negatives/fine-tune model hoặc verifier nhẹ có benchmark chi phí. | Bộ bài trong holdout không tạo event điện thoại; recall và p95 delay điện thoại thật không giảm so với baseline đã thống nhất; một episode chỉ một event. |
| P2 — Nền tảng mặt & an toàn | Trích đặc trưng tối thiểu trên cùng frame, kiểm tra quality/validity; thử auto-baseline ổn định có trạng thái rõ và nút manual fallback; thay `LEAVING_SEAT` bằng dịch chuyển track/tỉ lệ bbox so với baseline; thay camera blocked bằng tổ hợp brightness/entropy/blur có thời gian. | Test thiếu mặt, ánh sáng xấu, mất camera, ngồi thấp/đứng dậy; không suy đoán hành vi khi analyzer lỗi. |
| P3 — Gaze/MAR thử nghiệm | Tính chỉ số mắt/môi khi landmark hợp lệ; hiệu chuẩn cá nhân và chất lượng; chỉ hiện “cần rà soát”, tắt mặc định, không tự cộng rủi ro cao. Đo báo nhầm khi đọc đề, nhìn bàn phím, ngáp, nói chuyện bình thường. | Chỉ bật có giới hạn sau đánh giá mù trên holdout và chấp thuận ngưỡng theo chính sách; nếu không đạt thì giữ disabled. |
| P4 — Correlation & báo cáo | Mở rộng engine hiện có có kiểm soát, dựa trên event nguồn cùng người/còn mới; chống lặp/cap. Giữ current score decay + history riêng; thêm báo cáo độ tin cậy, tỷ lệ giám thị bác bỏ. | Không cộng trùng và không nâng một dự đoán vật thể sai lên mức CRITICAL bằng tín hiệu tay; regression end-to-end đạt. |

Tất cả giai đoạn có công tắc tắt và kiểm thử hồi quy riêng. P1 là ưu tiên trước
vì ca báo nhầm hiện tại; P3 có thể dừng vĩnh viễn nếu dữ liệu không đủ chất lượng.

### Cổng chất lượng và hiệu năng

- Dùng holdout độc lập, phân nhóm theo người/camera/ánh sáng/loại vật. Bắt buộc
  có bộ bài, hộp bài, ví và điện thoại trong/ngoài ốp. Báo confusion per class và
  cảnh báo nhầm trên mỗi phút video âm tính; không suy ra độ đúng từ một demo.
  Bộ pilot tối thiểu 30 episode điện thoại, 30 episode bộ/hộp bài và 30 episode
  vật chữ nhật khác; tăng cỡ mẫu trước khi công bố chất lượng sản phẩm.
- Ghi cùng một máy, độ phân giải, model hash, precision, device, target FPS. P1–P4
  không được làm giảm preview FPS/AI FPS quá 10% hoặc tăng p95 latency quá 10%
  so với baseline mà không có chấp thuận trade-off rõ ràng.
- Một vật liên tục chỉ một event; mất detection ngắn trong `maximum_gap_ms` không
  tạo event mới. Nếu mất dài, event mới phải có ID/nguồn và giải thích rõ.
- Bài thử “bộ bài” trong holdout phải có 0 cảnh báo điện thoại; đây là regression
  gate theo bộ mẫu, **không** phải cam kết 0 báo nhầm ngoài đời. Recall điện thoại
  và độ trễ không được kém baseline khi sửa báo nhầm.
- Các chỉ số gaze/MAR, rời chỗ và camera bị che cần confusion riêng trên tình
  huống bình thường và đối chứng. Không đạt thì không bật mặc định.

Chi tiết nghiên cứu, dữ liệu, giao diện và cách thử ở `research.md`,
`data-model.md`, `contracts/desktop-review.md`, `quickstart.md`. Chưa tạo tasks
hoặc sửa runtime trong bước lập kế hoạch này.
