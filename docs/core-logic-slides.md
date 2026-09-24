+# Core logic của Smart Exam Proctoring

*Tài liệu nền cho bài trình bày 9 slide. Nội dung đối chiếu với code và cấu hình đang có trong workspace; các ngưỡng có thể đổi qua YAML. Tên sự kiện trong ngoặc vuông là mã dùng trong chương trình.*

## Slide 1 — Hệ thống đang giải quyết việc gì?

Smart Exam Proctoring là ứng dụng desktop theo dõi một thí sinh qua webcam. Mỗi khung hình được phân tích để lấy thông tin về người, vật thể, hướng mặt, tư thế và bàn tay. Bộ luật chuyển các thông tin đó thành sự kiện cần giám thị xem lại.

Luồng chính:

~~~text
Webcam → quan sát AI → luật hành vi → sự kiện → ghép dấu hiệu → điểm risk → giao diện và bằng chứng
~~~

Các tín hiệu hiện có gồm: không thấy người, có nhiều người, điện thoại, sách/tài liệu, quay trái/phải, cúi mặt, quay đầu lặp lại, mặt không thấy, camera bị che, rời khỏi vùng hình và hoạt động tay đáng chú ý. Hệ thống còn có mã luật nói chuyện, nhưng pipeline hiện chưa đưa dữ liệu chuyển động miệng vào luật đó.

Điểm cần nói rõ khi báo cáo: mỗi tín hiệu cho biết camera quan sát được một tình huống đáng chú ý. Điểm risk giúp giám thị ưu tiên rà soát; nó không phải kết luận thí sinh gian lận.

*Có thể trình bày: “Camera cung cấp dữ liệu; các model tạo quan sát; rule xác nhận quan sát theo thời gian; sau đó hệ thống mở sự kiện để giám thị kiểm tra.”*

Mã nguồn tham khảo: [runtime.py](../apps/desktop/src/proctoring_desktop/runtime.py), [types.py](../packages/proctoring_core/src/proctoring_core/types.py).

## Slide 2 — Kiến trúc và công nghệ

Ứng dụng chia thành ba phần để tách nhận diện hình ảnh khỏi quyết định nghiệp vụ:

| Phần | Trách nhiệm | Công nghệ chính |
| --- | --- | --- |
| Desktop | Cửa sổ, webcam preview, overlay, nút phiên và danh sách sự kiện | Python 3.12, PySide6, OpenCV |
| AI adapter | Chạy model và chuẩn hóa kết quả thành quan sát | Ultralytics YOLO11n, PyTorch, MediaPipe Tasks, NumPy |
| Core | Rule theo thời gian, event, correlation, risk và cấu hình | Python, Pydantic 2, PyYAML |

YOLO hiện được cấu hình cho ba lớp COCO: `person`, `cell phone`, `book`. MediaPipe có Face Landmarker, Pose Landmarker Lite và Hand Landmarker. Model được nạp từ file cục bộ theo `models/manifest.yaml`; manifest lưu phiên bản, giấy phép và SHA-256 để kiểm tra file trước khi nạp.

Khi bắt đầu phiên, runtime đọc `configs/default.yaml` và `configs/exam_policy.yaml`, khởi tạo camera, model, tracker, bộ luật và nơi lưu dữ liệu phiên. Schema Pydantic từ chối khóa cấu hình lạ, giúp phát hiện lỗi gõ nhầm thay vì âm thầm bỏ qua.

Mã nguồn tham khảo: [main.py](../apps/desktop/main.py), [analyzer_factory.py](../packages/proctoring_ai/src/proctoring_ai/analyzer_factory.py), [schema.py](../packages/proctoring_core/src/proctoring_core/config/schema.py), [manifest.yaml](../models/manifest.yaml).

## Slide 3 — Một khung hình đi qua pipeline ra sao?

OpenCV đọc ảnh cùng `frame_id`, giờ UTC và đồng hồ monotonic. Đồng hồ monotonic dùng để tính khoảng thời gian vì không bị ảnh hưởng khi giờ hệ thống thay đổi. Preview UI được cập nhật riêng với luồng AI.

Trong runtime hiện tại, hàng đợi dành cho AI có sức chứa **1 khung hình**. Nếu AI xử lý chưa kịp và hàng đã đầy, frame cũ nhất bị bỏ để frame tiếp theo không phải chờ một chuỗi ảnh lỗi thời. Vì vậy preview có thể trơn hơn hoặc nhanh hơn nhịp inference; FPS camera và FPS AI là hai số khác nhau.

Với cùng một frame, `ObservationPipeline` chạy song song YOLO, các analyzer khuôn mặt/tư thế/bàn tay và bộ đo chất lượng ảnh. Pose được xếp lịch tối thiểu mỗi 250 ms, bàn tay mỗi 200 ms; khuôn mặt không đặt khoảng nghỉ trong cấu hình này. Một bộ quan sát gắn kết quả với đúng frame và timestamp, không ghép landmark cũ vào ảnh mới.

Mốc đánh giá hiệu năng trong code: sau 3 giây khởi động, trạng thái realtime yêu cầu cả capture FPS lẫn processed FPS đạt ít nhất 20, đồng thời độ trễ p95 không quá 200 ms. Cấu hình camera đặt mục tiêu 30 FPS; đó là mục tiêu đọc từ camera, không phải cam kết model xử lý được 30 FPS.

Hai phép đo trong cửa sổ trượt 2 giây được tính như sau:

~~~text
FPS = (số_mẫu − 1) / (thời_điểm_mẫu_cuối − thời_điểm_mẫu_đầu)
latency_ms = thời_điểm_xử_lý_xong − thời_điểm_capture
~~~

Mã nguồn tham khảo: [runtime.py](../apps/desktop/src/proctoring_desktop/runtime.py), [frame_queue.py](../packages/proctoring_ai/src/proctoring_ai/pipeline/frame_queue.py), [observation_pipeline.py](../packages/proctoring_ai/src/proctoring_ai/pipeline/observation_pipeline.py), [metrics.py](../packages/proctoring_ai/src/proctoring_ai/pipeline/metrics.py).

## Slide 4 — AI nhìn thấy những gì?

YOLO trả về nhãn lớp, độ tin cậy và hộp giới hạn (bounding box). Tọa độ hộp được chuẩn hóa theo kích thước ảnh:

~~~text
x = left / image_width       y = top / image_height
width = box_width / image_width   height = box_height / image_height
~~~

Nhờ tọa độ nằm trong khoảng 0–1, rule có thể so sánh vị trí tương đối giữa các khung hình có kích thước khác nhau. Người chỉ được giữ lại khi confidence từ 0,60; ngưỡng YOLO mặc định là 0,25. Điện thoại và sách có ngưỡng riêng cao hơn ở bước xác nhận.

Tracker gán ID cho vật thể cùng lớp bằng IoU, tức tỷ lệ diện tích giao trên diện tích hợp của hai hộp. IoU dưới 0,15 tạo track mới. Điện thoại cần ít nhất hai detection đủ confidence trên cùng track để được xác nhận. Tracker bỏ track đã quá 600 ms và không biến hộp cũ thành detection mới.

~~~text
IoU(A, B) = diện_tích(A ∩ B) / diện_tích(A ∪ B)
~~~

MediaPipe tạo landmark khuôn mặt, ma trận hướng đầu, landmark tư thế cơ thể và landmark bàn tay. Bộ đo chất lượng chuyển ảnh BGR sang grayscale rồi tính:

~~~text
brightness = mean(pixel_gray) / 255
entropy    = −Σ pᵢ × log₂(pᵢ)          với pᵢ là tần suất mức xám i
blur       = Var(Laplacian(gray))
~~~

`occlusion_estimate` hiện được đặt cố định bằng 0, nên chưa phải bộ phát hiện che camera thực sự.

Mã nguồn tham khảo: [yolo_detector.py](../packages/proctoring_ai/src/proctoring_ai/detectors/yolo_detector.py), [object_tracks.py](../packages/proctoring_ai/src/proctoring_ai/tracking/object_tracks.py), [landmarks.py](../packages/proctoring_ai/src/proctoring_ai/face/landmarks.py), [image_quality.py](../packages/proctoring_ai/src/proctoring_ai/features/image_quality.py).

## Slide 5 — Từ ma trận khuôn mặt ra góc đầu

Face Landmarker cung cấp ma trận biến đổi khuôn mặt. Code dùng phân rã SVD để loại scale và shear, lấy ma trận quay `R = U × Vᵀ`, rồi suy ra ba góc yaw, pitch, roll. Trong hệ tọa độ đang dùng, pitch dương nghĩa là cúi xuống; yaw dương hướng về bên phải ảnh chưa lật gương.

Với `sy = √(R₀₀² + R₁₀²)`, công thức chính là:

~~~text
yaw   = atan2(-R₂₀, sy)
pitch = atan2( R₂₁, R₂₂)     khi sy ≥ 10⁻⁶
roll  = atan2( R₁₀, R₀₀)
~~~

Các góc được đổi từ radian sang độ. Khi gần singularity (`sy < 10⁻⁶`), code dùng công thức pitch thay thế để tránh chia tách không ổn định.

Khi giám thị bấm hiệu chỉnh, app thu mốc trung tính. Mỗi trục lấy median làm offset; code cần ít nhất 8 mẫu ổn định trải trên 1,5 giây. Nếu biên độ một trục vượt 8°, chuỗi mẫu ổn định bị làm lại; khoảng mất mẫu trên 600 ms cũng làm rỗng chuỗi đang thu.

Trước khi vào rule, góc dùng median của tối đa 3 mẫu gần nhất rồi làm mượt EMA theo thời gian:

~~~text
α = 1 − e^(−Δt / 100 ms)
góc_mới = góc_trước + α × (góc_đã_hiệu_chỉnh − góc_trước)
~~~

Rule yêu cầu pose hợp lệ, chất lượng mặt ít nhất 0,35 và trạng thái hiệu chỉnh `calibrated`. Bộ lọc giúp bỏ spike nhưng cộng thêm độ trễ phụ thuộc nhịp frame và mức chuyển động.

Mã nguồn tham khảo: [head_pose.py](../packages/proctoring_ai/src/proctoring_ai/face/head_pose.py), [landmarks.py](../packages/proctoring_ai/src/proctoring_ai/face/landmarks.py), [head.py](../packages/proctoring_core/src/proctoring_core/behavior/head.py).

## Slide 6 — Rule hành vi và các ngưỡng đang bật

Một điều kiện chỉ thành `ACTIVE` khi tín hiệu hợp lệ kéo dài đủ lâu. Với thời điểm bắt đầu `t₀` và ngưỡng thời gian `T`, điều kiện đạt khi `t − t₀ ≥ T`. Nếu khoảng cách giữa các mẫu vượt 600 ms, rule đầu/cúi sẽ xóa chuỗi xác nhận để không tính xuyên qua lúc camera hoặc analyzer bị ngắt.

| Sự kiện | Điều kiện mặc định hiện tại | Ý nghĩa |
| --- | --- | --- |
| `HEAD_TURN_LEFT/RIGHT` | `|yaw| ≥ 22° trong 0,30 giây; đường nhanh từ 32° trong 0,20 giây` | Quay đầu trái/phải theo ảnh chưa lật |
| `LOOK_DOWN` | `pitch ≥ 15° trong 0,40 giây; đường nhanh từ 28° trong 0,20 giây` | Cúi mặt nhìn xuống |
| `ABNORMAL_HEAD_MOVEMENT` | Từ 4 lượt quay được xác nhận trong 30 giây | Đếm lượt mới sau khi đầu về vùng giữa ±12° đủ 0,30 giây |
| `PHONE_DETECTED` | Confidence thường 0,55; giữ 0,60 giây. Nhanh: confidence 0,85, cùng track, tối thiểu 2 hit và 0,20 giây | Chính sách `allow_phone` có thể tắt cảnh báo |
| `DOCUMENT_DETECTED` | Lớp sách confidence 0,70, giữ 1 giây | Quyền dùng sách/giấy phụ thuộc chính sách thi |
| `PERSON_MISSING` / `MULTIPLE_PERSON_DETECTED` | Hiệu lực runtime: cùng ngưỡng 3 giây | Bộ đếm dùng ngưỡng `person_missing`; giá trị riêng 2 giây cho `multiple_person` trong YAML hiện chưa được chuyển vào evaluator |
| `FACE_NOT_VISIBLE` / `CAMERA_BLOCKED` / `LEAVING_SEAT` | Hiệu lực runtime: cùng ngưỡng 1 giây | Safety dùng ngưỡng của `face_not_visible`; mức 2 giây riêng cho `camera_blocked` và `leaving_seat` trong YAML chưa được chuyển vào evaluator. Rời chỗ dựa trên `bbox.y > 0,9` hoặc `allowed_zone` nếu caller truyền vùng này |
| `CAMERA_BLOCKED` | Dấu hiệu ảnh: blur < 20 hoặc occlusion ≥ 0,8 | `occlusion_estimate` hiện luôn 0; vì vậy hiện chỉ nhánh blur có thể kích hoạt rule này |
| `SUSPICIOUS_HAND_ACTIVITY` | 2 giây | Tay gần mặt hoặc tâm tay di chuyển ít nhất 0,025 chiều rộng/chiều cao ảnh giữa hai lượt đo |

Độ dịch chuyển tay là khoảng cách Euclid giữa tâm hộp tay hiện tại và lần đo trước; hoạt động được bật khi khoảng cách đạt 0,025. Nếu mặt còn thấy, hộp mặt được nới rộng để kiểm tra tâm tay có nằm gần mặt hay không.

Khi event đầu/cúi đã mở, ngưỡng duy trì hạ 5° để giảm hiện tượng đóng/mở liên tục quanh ranh giới. Hai đường của `LOOK_DOWN` cùng tạo một loại sự kiện. Do đó cúi sâu hoặc vừa cúi vừa quay mặt đều có thể phát `LOOK_DOWN` khi góc mặt còn đọc được.

Mã nguồn tham khảo: [default.yaml](../configs/default.yaml), [schema.py](../packages/proctoring_core/src/proctoring_core/config/schema.py), [head.py](../packages/proctoring_core/src/proctoring_core/behavior/head.py), [presence.py](../packages/proctoring_core/src/proctoring_core/behavior/presence.py), [objects.py](../packages/proctoring_core/src/proctoring_core/behavior/objects.py).

## Slide 7 — Rule trở thành event, rồi được ghép với dấu hiệu khác

Rule phát tín hiệu theo từng frame. `EventAggregator` gom các tín hiệu cùng phiên, người, loại sự kiện và rule key thành một event. Frame đầu tiên đạt ngưỡng mở event; khi còn active, event được cập nhật; khi điều kiện chấm dứt, event đóng. Cách gom này giúp một hành vi liên tục được theo dõi như một episode.

`CorrelationEngine` chỉ ghép event đã được rule xác nhận, còn mới và có analyzer tương ứng hoạt động. Các cặp hiện tại:

| Tổ hợp | Điều kiện ghép | Điểm cộng |
| --- | --- | ---: |
| `PHONE_MULTIPLE_PERSON` | Có điện thoại và nhiều người | 10 |
| `HEAD_AND_HAND` | Quay trái/phải và hoạt động tay của cùng track | 4 |
| `LOOK_DOWN_AND_HAND` | Cúi đầu và hoạt động tay của cùng track | 4 |

Hai event phải bắt đầu trong cửa sổ 5 giây, còn mới trong 2 giây và tồn tại đồng thời đủ 0,25 giây. Mỗi cặp nguồn chỉ được cộng một lần; cùng một loại tổ hợp được cộng tối đa một lần trong mỗi cửa sổ 5 giây; tổng điểm correlation bị giới hạn ở 20 điểm cho cả phiên. Event nguồn vẫn được lưu ID để giám thị lần ngược tổ hợp đã tạo điểm.

Khi event mở, `EvidenceRecorder` sao chép đúng frame AI vừa phân tích vào hàng đợi nền rồi ghi JPEG chất lượng 85. `frame_id` và timestamp phải trùng với bundle. Ảnh và event index được lưu dưới `storage/sessions/<session-id>`; cấu hình mặc định giữ ảnh tối đa 30 ngày và tắt ghi clip toàn phiên.

Mã nguồn tham khảo: [aggregator.py](../packages/proctoring_core/src/proctoring_core/events/aggregator.py), [correlation.py](../packages/proctoring_core/src/proctoring_core/events/correlation.py), [evidence.py](../apps/desktop/src/proctoring_desktop/evidence.py).

## Slide 8 — Điểm risk được tính thế nào?

Trọng số mặc định theo loại event: `PHONE_DETECTED` 28; `MULTIPLE_PERSON_DETECTED` 32; `PERSON_MISSING` 22; `HEAD_TURN_LEFT` và `HEAD_TURN_RIGHT` mỗi loại 10; `LOOK_DOWN` 9; `ABNORMAL_HEAD_MOVEMENT` 16; `DOCUMENT_DETECTED` 12; `LEAVING_SEAT` 18; `CAMERA_BLOCKED` 26; `FACE_NOT_VISIBLE` 16; `SUSPICIOUS_HAND_ACTIVITY` 5; `POSSIBLE_TALKING` 4.

Điểm thô cho một event được tính theo:

~~~text
duration_factor   = 1 + min(1, duration_ms / 10.000)
repetition_factor = 1 + min(0,5, (số_event_cùng_loại − 1) × 0,1)
raw               = trọng_số × max(0,1, confidence)
                    × duration_factor × repetition_factor
đóng_góp          = min(raw, risk_cap)
~~~

Trong công thức, hệ số thời lượng tối đa gấp đôi sau 10 giây; hệ số lặp tăng 0,1 cho mỗi event mới cùng loại và dừng ở 1,5. `risk_cap` mặc định là 100 nếu signal không truyền giới hạn riêng.

Có một chi tiết triển khai ảnh hưởng đến cách đọc công thức: `EventAggregator` chỉ gán `duration_ms` khi đóng event, còn `RiskEngine` kết thúc xử lý event đóng trước khi tính đóng góp. Vì vậy event trong runtime được chấm khi còn mở với `duration_ms = None`, khiến `duration_factor = 1`. Công thức có hệ số thời lượng, nhưng thời lượng hiện chưa làm điểm của event tăng lên. Ví dụ bên dưới phản ánh hành vi thực tế đó.

Score hiện tại giảm 1,5 điểm mỗi giây khi không có cập nhật mới, không xuống dưới 0. Event active còn mới giữ một “sàn” score bằng mức đóng góp cao nhất của event đó trong TTL 2 giây. Tổng score giới hạn ở 100.

Mức phân loại: `NORMAL` = 0; `LOW` > 0; `MEDIUM` từ 30; `HIGH` từ 60; `CRITICAL` từ 80. Correlation cộng trực tiếp điểm đã xác nhận. Cumulative score, peak score và số event được lưu riêng làm lịch sử, không giảm theo decay.

Ví dụ: event cúi đầu đầu tiên có confidence 0,8 được tính `9 × 0,8 × 1 × 1 = 7,2` điểm trong runtime hiện tại. Đây là điểm risk theo weight mặc định, không phải xác suất gian lận 7,2%.

Mã nguồn tham khảo: [engine.py](../packages/proctoring_core/src/proctoring_core/risk/engine.py), [default.yaml](../configs/default.yaml), [correlation.py](../packages/proctoring_core/src/proctoring_core/events/correlation.py).

## Slide 9 — Cách đọc kết quả và giới hạn hiện tại

Ở giao diện, giám thị có thể xem preview kèm overlay, trạng thái từng analyzer, FPS, độ trễ p95, số frame bị bỏ, event và score phiên. Event mới có thể đi kèm ảnh JPEG đúng frame đã kích hoạt. Khi analyzer lỗi, timeout hoặc thiếu model, trạng thái sức khỏe được đưa lên UI để biết vì sao một loại quan sát không có dữ liệu.

Khi giải thích kết quả, cần tách ba lớp: model nhận dạng mẫu hình trong ảnh; rule kiểm tra góc/độ tian; event là tín hiệu cho người xem lại. Hệ thống hiện không ước lượng hướng nhìn bằng mắt, không nhận diện vị trí “dưới ngăn bàn”, và không phân loại hành vi chép bài. Cúi mặt xuống là chỉ dấu tư thế. Nếu góc cúi sâu làm mất mặt, rule góc không thể xác nhận từ khunn cậy/thời gig đó; sự kiện mất mặt có thể thay thế khi giữ đủ thời gian.

Pose toàn thân hiện tạo landmark để hiển thị nhưng chưa được dùng để suy ra “đang viết dưới bàn”. Tay là tín hiệu hỗ trợ thô: khoảng cách giữa tâm bàn tay qua các lần đo; correlation `LOOK_DOWN_AND_HAND` làm tăng mức cần rà soát nhưng cũng chưa chứng minh hành vi. YOLO COCO chỉ có lớp `book`, nên phát hiện giấy hoặc vật nhỏ dưới bàn còn hạn chế. `POSSIBLE_TALKING` chưa có dữ liệu miệng nối vào runtime.

Vì vậy nội dung báo cáo nên gọi sản phẩm là hệ thống hỗ trợ giám sát realtime bằng computer vision và rule theo thời gian. Để đánh giá độ chính xác, dự án cần video có gán nhãn và thống kê false positive, false negative, p50/p95 độ trễ trên nhiều điều kiện webcam.

Mã nguồn tham khảo: [runtime.py](../apps/desktop/src/proctoring_desktop/runtime.py), [pose_estimator.py](../packages/proctoring_ai/src/proctoring_ai/pose/pose_estimator.py), [hand_estimator.py](../packages/proctoring_ai/src/proctoring_ai/hands/hand_estimator.py), [event-detection-tuning.md](event-detection-tuning.md).
