# Nghiên cứu kỹ thuật

Tài liệu này ghi lại các quyết định cần cho kế hoạch triển khai ngày 2026-09-19. Phiên bản phụ
thuộc sẽ được khóa trong `uv.lock` và `package-lock.json` sau khi chạy smoke test trên Windows và
Linux. Không lấy bản preview làm baseline.

## 1. Phạm vi của kế hoạch

**Quyết định**: Xây Phase 1 trước, nhưng giữ các interface cần cho Phase 2 ngay từ đầu. Phase 1 chỉ
cho phép một session camera hoạt động. Phase 2 bổ sung multi-tracking và UI phòng thi trên cùng event
model.

**Lý do**: README có hai phase và yêu cầu Phase 1 là MVP. Việc triển khai đồng thời toàn bộ Phase 2
sẽ kéo dài vòng đo false positive đầu tiên. Các điểm mở rộng cần có sớm gồm `Tracker`, state theo
`track_id`, object association và risk theo subject.

**Phương án đã cân nhắc**: Tách Phase 2 thành một codebase khác. Phương án này làm lặp event engine,
policy và evidence nên không được chọn.

## 2. Mô hình tiến trình và xử lý frame

**Quyết định**: API và session orchestration chạy trong một tiến trình. Capture cùng inference chạy
trong worker riêng, không chạy trên event loop. Mỗi nguồn hình có hàng đợi latest-frame kích thước
nhỏ; khi worker chậm, hệ thống bỏ frame cũ và tăng metric `dropped_frames`.

**Lý do**: Mục tiêu realtime phụ thuộc độ trễ cuối đường ống hơn số frame xử lý tuyệt đối. Hàng đợi
không giới hạn sẽ khiến cảnh báo xuất hiện muộn dù FPS trung bình trông ổn. Một tiến trình dễ chạy và
debug cho MVP; interface của runner vẫn cho phép tách worker ở giai đoạn triển khai nhiều camera.

**Phương án đã cân nhắc**:

- Chạy suy luận trực tiếp trong route hoặc coroutine. Tác vụ CPU/GPU sẽ chặn API và heartbeat.
- Dùng message broker ngay ở Phase 1. Cách này tăng vận hành nhưng chưa đem lại lợi ích cho một camera.
- Xử lý mọi frame. Khi tải tăng, độ trễ sẽ tăng liên tục thay vì giảm chất lượng có kiểm soát.

## 3. Camera và preview

**Quyết định**: Backend mở webcam hoặc video bằng OpenCV. Preview được encode JPEG và cung cấp qua
endpoint MJPEG. Event, risk, health và metric đi qua WebSocket dạng JSON. Debug overlay được vẽ ở
backend từ cùng observation bundle với frame preview.

**Lý do**: README giao nhiệm vụ capture cho OpenCV. Cách này dùng được với webcam cục bộ, file
regression và camera stream mà không thay domain. Tách preview khỏi WebSocket giữ message sự kiện
nhỏ và dễ kiểm thử.

Gói `opencv-python` có wheel cho Python 3.12 và Windows. Bản headless chỉ dùng cho CI hoặc container
không cần cửa sổ native. Nguồn: [opencv-python trên PyPI](https://pypi.org/project/opencv-python/).

**Phương án đã cân nhắc**:

- Browser dùng `getUserMedia()` rồi gửi frame lên backend. Phương án này thêm bước encode, truyền và
  đồng bộ ở máy cục bộ, trong khi pipeline vẫn chạy bằng Python.
- Gửi frame nhị phân chung WebSocket sự kiện. Việc backpressure và reconnect khó tách khỏi event bus.
- WebRTC. Đây là lựa chọn hợp lý cho camera từ xa, nhưng nằm ngoài MVP cục bộ.

## 4. Detector và ranh giới giấy phép

**Quyết định**: Dùng Ultralytics YOLO làm baseline cho person, phone và book qua adapter
`ObjectDetector`. Chỉ dùng model release ổn định đã khóa phiên bản và hash. Class `document` hoặc
`note_paper` chỉ được bật sau khi có model đã fine-tune và bộ đánh giá riêng.

**Lý do**: Ultralytics có API predict, track, export và benchmark, phù hợp với roadmap PyTorch sang
ONNX. Adapter ngăn type của thư viện đi vào domain. Domain chỉ nhận bbox, class, confidence,
timestamp và model metadata.

Ultralytics cung cấp AGPL-3.0 và Enterprise. Dự án phải chốt cách tuân thủ trước khi phát hành. Gate
pháp lý này phải hoàn tất trước khi phân phối sản phẩm. Nguồn:
[tài liệu Ultralytics](https://docs.ultralytics.com/) và
[hướng dẫn tuân thủ AGPL](https://docs.ultralytics.com/help/contributing/).

**Phương án đã cân nhắc**:

- Gọi trực tiếp `ultralytics.YOLO` trong behavior rule. Cách này phá ranh giới AI và nghiệp vụ.
- Huấn luyện detector từ đầu. README yêu cầu bắt đầu bằng pretrained model và chỉ fine-tune khi có
  dữ liệu false positive/false negative.
- Coi mọi giấy là tài liệu cấm. Exam policy có thể cho phép giấy nháp nên detector không được quyết định.

## 5. Face, pose và hand landmark

**Quyết định**: Dùng MediaPipe Tasks ở chế độ video hoặc live stream. Face landmarks cung cấp điểm
đầu vào cho head pose; pose và hand landmarks chỉ tạo feature. Mỗi analyzer có timeout, health state
và có thể bị tắt bằng config.

**Lý do**: API live stream xử lý bất đồng bộ và yêu cầu timestamp tăng dần, phù hợp với timestamp
monotonic của frame pipeline. `FaceLandmarkerOptions` cho phép đặt confidence và số khuôn mặt; các
giá trị này được ánh xạ từ config. Nguồn:
[FaceLandmarkerOptions](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerOptions)
và [FaceLandmarker](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarker).

MediaPipe hiện công bố hỗ trợ Python 3.12 trên PyPI. Nguồn:
[mediapipe trên PyPI](https://pypi.org/project/mediapipe/).

**Phương án đã cân nhắc**:

- Eye gaze chính xác cao. Webcam phổ thông, kính, ánh sáng và độ phân giải làm baseline này thiếu ổn định.
- Dùng blendshape để kết luận đang nói. `POSSIBLE_TALKING` chỉ là tín hiệu thử nghiệm nên không cần
  thêm phụ thuộc audio hoặc mô hình phát ngôn ở Phase 1.

## 6. Head pose

**Quyết định**: Lấy tập landmark ổn định ở mũi, mắt, cằm và miệng; giải bài toán pose với camera
matrix được ước lượng từ kích thước frame. Sau calibration, lưu yaw, pitch, roll và độ tin cậy vào
observation. Rule chỉ đọc các giá trị này cùng config.

**Lý do**: Head pose đáp ứng `HEAD_TURN_LEFT`, `HEAD_TURN_RIGHT` và `LOOK_DOWN` mà không cần nhận dạng
danh tính hay eye gaze. Calibration theo người và camera giúp giảm sai lệch do vị trí ngồi ban đầu.

**Phương án đã cân nhắc**: Dùng landmark x-coordinate đơn lẻ. Cách đó nhạy với dịch chuyển toàn bộ
khuôn mặt và không cho góc có ý nghĩa để cấu hình.

## 7. Temporal engine và event lifecycle

**Quyết định**: Mỗi track có ring buffer chứa observation đã chuẩn hóa trong cửa sổ tối đa cần cho
rule. Rule trả về trạng thái `inactive`, `candidate` hoặc `active`. Event aggregator mở một event khi
rule đủ điều kiện, cập nhật trong lúc còn active và đóng khi condition kết thúc hoặc timeout.

**Lý do**: Một event kéo dài phải có một identity và một khoảng thời gian. State machine tránh tạo
event mới ở mỗi frame. Monotonic time dùng để tính duration; UTC chỉ dùng cho dữ liệu trao đổi và lưu trữ.

**Phương án đã cân nhắc**:

- Một chuỗi `if` theo frame. Phương án này vi phạm constitution và làm tăng false alert.
- Lưu toàn bộ observation vào database. Dữ liệu lớn, nhạy cảm và không cần cho nghiệp vụ mặc định.

## 8. Risk score

**Quyết định**: Risk engine là hàm xác định từ event transition, config snapshot và thời gian. Điểm
được giới hạn trong 0 đến 100, có decay theo giây, hệ số duration/repetition và correlation rule có
tên. Mọi lần thay đổi ghi `reason_codes` và danh sách event đóng góp.

**Lý do**: Cùng input phải tạo cùng risk để test và giải thích. Correlation chỉ chạy trên cửa sổ đã
cấu hình; mỗi rule phải có mức cộng tối đa để tránh đếm lặp vô hạn.

**Phương án đã cân nhắc**: Một mô hình học máy dự đoán risk trực tiếp. Hiện chưa có dataset, calibration
hoặc cách giải thích đủ để dùng cho quyết định ưu tiên review.

## 9. Multi-object tracking

**Quyết định**: Phase 1 dùng `SingleSubjectTracker` đơn giản. Phase 2 triển khai interface `Tracker`
với ByteTrack làm baseline, sau đó mới đánh giá BoT-SORT bằng cùng tập video. Track ID chỉ có ý nghĩa
trong một session.

**Lý do**: ByteTrack phù hợp mốc ban đầu trong README. Ultralytics hỗ trợ tracker qua file cấu hình;
adapter cho phép thay tracker mà không đổi event contract. Nguồn:
[Ultralytics Track mode](https://docs.ultralytics.com/modes/track).

**Phương án đã cân nhắc**:

- Dùng BoT-SORT ngay. ReID tăng chi phí và không loại bỏ nhu cầu đo ID switch.
- Xem track ID là danh tính. Tracker có thể mất hoặc đổi ID, nên dữ liệu chỉ dùng anonymous track.

## 10. Object-to-person association

**Quyết định**: Association trả về `assigned(track_id, score)` hoặc `ambiguous`. Score kết hợp object
nằm trong person bbox, khoảng cách đến wrist/torso và tính liên tục theo thời gian. Không gán khi hai
candidate có điểm gần nhau hơn ngưỡng chênh lệch cấu hình.

**Lý do**: Spec cấm gán im lặng cho sai người. Trạng thái ambiguous cho phép UI hiển thị vật thể ở
cấp phòng mà không làm tăng risk của một track cụ thể.

**Phương án đã cân nhắc**: Chọn person gần nhất cho mọi object. Cách này tạo lỗi có hệ thống khi hai
người ngồi sát hoặc bị che.

## 11. REST và WebSocket

**Quyết định**: REST quản lý session, lịch sử, config, policy, review và evidence. WebSocket
`/ws/sessions/{session_id}` phát update của session đang chạy. Mỗi message có `schema_version`,
`sequence`, `occurred_at`, `session_id`, `type` và `data`.

**Lý do**: FastAPI hỗ trợ WebSocket cùng dependency cho xác thực. TestClient cũng có thể kiểm thử
WebSocket theo contract. Nguồn: [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
và [Testing WebSockets](https://fastapi.tiangolo.com/advanced/testing-websockets/).

**Phương án đã cân nhắc**:

- Polling cho timeline. Trễ và tạo request đều đặn khi không có thay đổi.
- Dùng WebSocket cho CRUD. REST dễ idempotent, audit và sinh OpenAPI hơn.
- Cố replay mọi message sau reconnect. Frontend lấy snapshot qua REST rồi dùng sequence mới, đơn giản
  và đủ cho ứng dụng cục bộ.

## 12. Persistence

**Quyết định**: Dùng SQLAlchemy 2 và Alembic. Phase 1 dùng SQLite cục bộ qua một writer queue. WAL chỉ
bật khi SQLite runtime là 3.51.3 trở lên hoặc một bản backport đã sửa lỗi; nếu không, ứng dụng dùng
rollback journal và giữ một writer. SQLite database không đặt trên network filesystem.

**Lý do**: Một writer phù hợp tần suất event và tránh tranh chấp. WAL cho phép reader hoạt động cùng
writer, nhưng SQLite chỉ có một writer và yêu cầu shared memory trên cùng máy. Tài liệu SQLite còn
ghi nhận lỗi WAL-reset đã sửa ở 3.51.3 và một số bản backport. Nguồn:
[SQLite WAL](https://www.sqlite.org/wal.html).

Phase 2 thay URL database sang PostgreSQL và giữ repository contract. Migration sẽ bổ sung index theo
`session_id`, `track_id`, `started_at`, `event_type` và `severity`.

**Phương án đã cân nhắc**:

- PostgreSQL ngay ở MVP. Việc cài và vận hành không cần thiết cho một máy, một session.
- Ghi database trực tiếp trong inference worker. I/O có thể làm chậm pipeline và trộn transaction với AI.

## 13. Evidence và quyền riêng tư

**Quyết định**: Frame ring buffer giữ ảnh đã encode trong RAM theo cửa sổ evidence. Event frame là
bắt buộc khi storage hoạt động; clip ngắn là tùy policy. Metadata nằm trong database, file nằm dưới
`storage/sessions/{session_id}/`. Tên file dùng UUID, không có tên người.

Evidence service thực hiện ghi nguyên tử qua file tạm, kiểm tra đường dẫn nằm trong storage root,
ghi audit khi xem hoặc tải và xóa file khi hết hạn. Nếu ghi thất bại, event vẫn tồn tại với trạng thái
evidence `FAILED`; UI phải hiển thị lỗi này.

**Lý do**: Tách metadata và file giúp Phase 2 chuyển file sang object storage mà không đổi event.
Ring buffer tạo được pre-event evidence nhưng không lưu toàn bộ video.

**Phương án đã cân nhắc**: Ghi video liên tục rồi cắt sau. Phương án này đi ngược nguyên tắc giảm dữ liệu.

## 14. Xác thực và phân quyền

**Quyết định**: Contract dùng bearer token và ba role: `operator`, `reviewer`, `admin`. Dev mode có
identity cục bộ được bật rõ bằng cấu hình và bị từ chối khi environment là production. Evidence yêu
cầu `reviewer` hoặc `admin`; config và policy chỉ `admin` được sửa.

**Lý do**: Spec yêu cầu evidence có kiểm soát và audit trong production. Tách auth provider sau một
interface để có thể dùng OIDC sau này mà không đổi route.

**Phương án đã cân nhắc**: Không xác thực vì ứng dụng chạy local. Cách này không đáp ứng yêu cầu khi
đưa vào production và dễ bị giữ lại ngoài ý muốn.

## 15. Frontend

**Quyết định**: React với TypeScript và Vite. REST client chịu trách nhiệm snapshot và history;
session reducer áp dụng WebSocket message theo `sequence`. Màn hình chính gồm camera, status, risk,
timeline và control. Phase 2 bổ sung room grid và candidate detail trên cùng type contract.

Vite có template `react-ts` và plugin React chính thức. Nguồn:
[Vite Getting Started](https://vite.dev/guide/).

**Phương án đã cân nhắc**: Thêm state framework ngay từ đầu. Phần lớn state là snapshot server và
stream của một session, nên reducer và cache REST đã đủ cho MVP.

## 16. Chiến lược test

**Quyết định**:

- Unit test cho rule, temporal buffer, event aggregator, risk, policy, association và retention.
- Contract test xác nhận API response và WebSocket envelope khớp tài liệu.
- Integration test dùng fake camera, fake detector và SQLite tạm để chạy đến database/WebSocket.
- Video regression dùng manifest có expected event window và tolerance.
- Benchmark ghi hardware, model hash, resolution, precision, FPS, p50/p95 latency và dropped frames.
- Frontend test component với fixture contract; Playwright chạy luồng session và review bằng fake backend.

**Lý do**: Detector accuracy không cho biết event có hữu ích hay không. Bộ test phải đo thêm false
alerts mỗi phút, missed event, event F1 và detection delay như README yêu cầu.

**Phương án đã cân nhắc**: Chỉ test bằng webcam trực tiếp. Kết quả không lặp lại và không dùng được
để so sánh regression.

## 17. Tối ưu model

**Quyết định**: Đầu tiên đo PyTorch baseline. Chỉ export ONNX khi profiler cho thấy detector là nút
thắt và regression xác nhận output tương đương trong tolerance. TensorRT là bước sau cho GPU NVIDIA.

PyTorch có exporter ONNX dựa trên `torch.export`, còn model ONNX có thể chạy bằng runtime khác.
Nguồn: [PyTorch ONNX tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html).

**Lý do**: Tối ưu trước khi có baseline làm khó việc xác định cải thiện và có thể đổi accuracy.

**Phương án đã cân nhắc**: Bắt đầu bằng TensorRT hoặc INT8. Cả hai tăng công sức build, calibration
và kiểm tra sai lệch trong khi chưa biết pipeline chậm ở đâu.

## 18. Quản lý phiên bản phụ thuộc

**Quyết định**: `pyproject.toml` khai báo range tương thích; `uv.lock` khóa bản dùng để phát triển và
CI. Frontend commit `package-lock.json`. Model file có manifest gồm nguồn, license, checksum, class
map và metric baseline. Upgrade phụ thuộc AI phải chạy lại video regression và benchmark.

**Lý do**: Các gói CV thay đổi model, class map, binary wheel và runtime nhanh hơn application code.
Lock file cùng model manifest giúp tái lập kết quả.

**Phương án đã cân nhắc**: Luôn lấy bản mới nhất khi cài. Một lần cài lại có thể đổi output mà không
có thay đổi source code.

## Kết luận nghiên cứu

Nghiên cứu không còn câu hỏi mở. Trước khi phát hành, đội dự án phải kiểm tra giấy phép Ultralytics,
phiên bản SQLite khi bật WAL, policy retention của đơn vị vận hành và benchmark trên phần cứng mục tiêu.
