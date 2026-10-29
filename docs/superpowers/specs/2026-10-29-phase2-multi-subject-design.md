# Phase 2: Phân tích nhiều thí sinh trên một video — Thiết kế

- Ngày: 2026-10-29
- Trạng thái: chờ duyệt
- Người yêu cầu: Nguyên

## 1. Mục tiêu và phạm vi

### Đã thống nhất

- Phase 2 chuyển từ một thí sinh sang **nhiều thí sinh trong cùng một khung hình**.
- Bước đầu là **một webcam thấy nhóm nhỏ 2–6 thí sinh** ngồi gần nhau (phương án B). Camera
  toàn phòng 20–60 người (phương án A) là mục tiêu sau; thiết kế giữ đường nâng cấp nhưng
  không làm trong spec này.
- Tín hiệu theo **phương án C**: mọi thí sinh luôn có mặt + hướng đầu + điện thoại/sách;
  tay và tư thế chỉ chạy thêm cho tối đa 2 thí sinh đang có tín hiệu nghi.
- Kết quả đầu tiên là **script offline chạy trên file video**, chưa sửa app desktop.
- Phương pháp bắt mặt: **người trước, mặt sau** (YOLO tìm người → tracker → cắt vùng đầu →
  FaceLandmarker trên từng ảnh cắt).
- Mục tiêu tốc độ: **≥ 8 FPS AI với 6 thí sinh** trên Quadro M1200 (mong muốn 8–10 FPS).
- AI không kết luận gian lận; sự kiện chỉ để giám thị ưu tiên rà soát (giữ nguyên PRODUCT.md).

### Giả định (chưa xác nhận, sửa nếu sai)

- Video thử lấy từ SAV (https://github.com/Ritatanz/SAV) hoặc video stock miễn phí, chọn một
  đoạn có 4–6 người thấy rõ mặt, ≥ 720p, dài 1–3 phút. Video không đưa vào Git.
- Máy đo là laptop Windows của người dùng, `.venv-gpu` (PyTorch CUDA 11.8), YOLO trên
  `cuda:0`, MediaPipe trên CPU.

### Ngoài phạm vi

- Nhận dạng danh tính, gắn thí sinh với số báo danh.
- Camera toàn phòng (phương án A), nhiều camera.
- Sửa UI desktop, chế độ webcam trực tiếp nhiều người (làm sau khi số liệu ổn).
- Model mặt/hướng đầu chạy GPU (bước 2b, chỉ làm nếu mục tiêu tốc độ không đạt).

## 2. Kiến trúc

Code mới nằm trong `packages/proctoring_ai/src/proctoring_ai/multi/`. Pipeline một người
của phase 1 và app desktop giữ nguyên.

```
VideoFileFrameSource
  → YOLO (GPU, person/cell phone/book)          [có thể cách 1 frame]
  → MultiSubjectTracker (ghép IoU + nối lại theo ghế)
  → SubjectCropper (vùng đầu/vai từng track)
  → PerSubjectFaceAnalyzer (song song theo track, có bỏ qua khi đứng yên)
  → FocusScheduler → tay/tư thế cho ≤ 2 track đang nghi
  → luật sự kiện theo track (aggregator sẵn có, khóa theo track_id)
  → writer: video có vẽ khung, events.jsonl, evidence/, timing.json
```

### 2.1 `MultiSubjectTracker`

- Vào: danh sách box `person` của YOLO (đã lọc `person_confidence_threshold`).
- Gán ID bằng ghép IoU tham lam giữa box mới và các ghế đang theo dõi (tự viết, không thêm
  phụ thuộc). Đã cân nhắc ByteTrack của ultralytics: bản khóa 8.4.160 cần gói `lap` không có
  trong `uv.lock` và sẽ tự `pip install` lúc chạy; với 2–6 người ngồi yên, ghép IoU + nối lại
  theo ghế là đủ. Interface `IdAssigner` cho phép thay ByteTrack sau.
- Lớp nối lại theo ghế: mỗi `runtime_track_id` gắn với một "ghế" (tâm box trung bình). Khi
  box mới không ghép được với ghế nào mà tâm box nằm trong bán kính 0,5 × chiều rộng box của một ghế đã
  mất < 10 s, gán lại ID cũ. Mục đích: một thí sinh không bị tách thành nhiều track.
- Ra: cùng dạng dict với `SingleSubjectTracker` (`runtime_track_id`, `track_id` UUID,
  `bbox`, `confidence`) để luật và aggregator dùng lại được.
- Giới hạn `max_subjects = 6`: nếu nhiều hơn, chỉ phân tích 6 track có box lớn nhất; số còn
  lại vẫn được vẽ và ghi log `SUBJECT_LIMIT_EXCEEDED`.

### 2.2 `SubjectCropper`

- Lấy 45% phía trên box người, nới lề 15% mỗi phía, kẹp trong khung hình.
- Phóng cạnh ngắn lên `crop_min_side` (mặc định 256, thử 192 khi đo).
- Trả về ảnh cắt và phép biến đổi (offset, scale) để đổi landmark về tọa độ khung gốc.

### 2.3 `PerSubjectFaceAnalyzer`

- Mỗi ghế có một FaceLandmarker (chế độ VIDEO, `num_faces=1`) và một `HeadPoseEstimator`
  riêng, tối đa 6 instance. Ghế hết hạn thì đóng instance. Mặt có tâm nằm ngoài box người
  của ghế (bắt nhầm mặt người phía sau) bị loại, ghi `status="mismatch"`.
- Dùng lại `face/features.py` và `face/head_pose.py` của phase 1.
- Các track chạy song song trên `ThreadPoolExecutor` (số luồng = min(số track, số nhân CPU)).
- **Bỏ qua khi đứng yên:** tính chênh lệch trung bình tuyệt đối giữa ảnh cắt xám thu nhỏ
  64×64 với lần chạy trước. Nếu dưới `motion_skip_threshold` và kết quả cũ chưa quá 500 ms,
  dùng lại kết quả cũ (đánh dấu `reused=true`; sự kiện mở trên kết quả dùng lại ghi
  `face_reused=true` trong `events.jsonl`).
- **Tự hiệu chỉnh:** trung vị yaw/pitch trong 3 s đầu mặt ổn định của track là mốc "nhìn
  thẳng" (dùng cơ chế gợi ý mốc sẵn có của `HeadPoseEstimator`, `suggest_after_ms=3000`, tự
  chấp nhận gợi ý). Track được nối lại theo ghế thì giữ mốc của ghế. Trước khi có mốc, không sinh sự
  kiện quay/cúi đầu cho track đó.

### 2.4 `FocusScheduler`

- Track có tín hiệu đang chờ xác nhận (candidate) ở luật quay đầu/cúi đầu/điện thoại/nhìn
  sang bên cạnh được đưa vào danh sách "focus" trong 5 s.
- Mỗi lượt chạy tay + tư thế trên ảnh cắt toàn thân của tối đa 2 track focus, theo interval
  của phase 1 (`pose_interval_ms`, `hands_interval_ms`).
- Ngân sách lượt: nếu tổng thời gian lượt trước vượt `frame_budget_ms` (mặc định 110 ms),
  lượt sau bỏ tay/tư thế và xoay vòng chạy mặt cho track ổn định; track mới hoặc đang focus
  luôn được ưu tiên. Bảo đảm mỗi track được phân tích mặt ≥ 4 lần/giây.

### 2.5 Luật sự kiện theo track

- Dùng lại aggregator (khóa đã có `track_id`) và các luật quay trái/phải, cúi đầu,
  `FACE_NOT_VISIBLE` (có người nhưng không thấy mặt), điện thoại, sách.
- Điện thoại/sách gán cho track có box người chồng lấn (IoU vùng mở rộng) lớn nhất.
- Tắt `MULTIPLE_PERSON_DETECTED` ở chế độ nhiều người.
- Tín hiệu mới `LOOK_AT_NEIGHBOR` (thêm vào `EventType`): yaw đã hiệu chỉnh vượt ngưỡng quay
  đầu và hướng quay chỉ về phía tâm box của track kế bên gần nhất (cùng hàng, khoảng cách
  ngang < 2,5 × chiều rộng box), kéo dài ≥ 1,5 s. Mức thấp, có trần điểm như tín hiệu ánh mắt.

### 2.6 Script `scripts/analyze_video_multi.py`

```
python scripts/analyze_video_multi.py VIDEO --out storage/multi/<tên>
    [--mode full|realtime] [--fps 10] [--yolo-size 416|640] [--yolo-every 1|2]
    [--crop-size 256] [--max-subjects 6] [--no-motion-skip] [--no-focus]
```

- `full`: xử lý mọi frame được lấy mẫu theo `--fps`, thời gian theo video, kết quả lặp lại
  được (dùng để kiểm tra luật).
- `realtime`: phát video theo đồng hồ thật, frame đến khi AI đang bận thì bỏ, giống camera
  (dùng để đo tốc độ thật).
- Đầu ra trong `--out`:
  - `annotated.mp4`: khung người + ID, khung mặt, hướng đầu, nhãn sự kiện.
  - `events.jsonl`: định dạng như phase 1, thêm `runtime_track_id`.
  - `evidence/`: JPEG khung đã phân tích khi có sự kiện mới.
  - `timing.json`: p50/p95 ms của YOLO, tracker, mặt (mỗi lần và mỗi lượt), tay, tư thế;
    AI FPS; độ trễ p95; số lần phân tích mặt/giây của từng track; tỉ lệ dùng lại kết quả;
    số lần đổi ID; số frame bị bỏ (realtime).

## 3. Tốc độ

Ước lượng ban đầu (chưa đo): mặt 10–20 ms/người trên CPU, YOLO11n trên M1200 chưa có số đo
GPU. Các cách tăng tốc nằm trong thiết kế, mỗi cách bật/tắt được bằng cờ để đo riêng:

1. Chạy mặt song song theo track.
2. Gối đầu GPU/CPU: YOLO frame t+1 chạy trong khi CPU làm mặt frame t.
3. Bỏ qua track đứng yên (2.3).
4. YOLO cách 1 frame (`--yolo-every 2`), tracker dự đoán frame giữa.
5. Ảnh cắt 192 thay vì 256 nếu hướng đầu không lệch quá 3° trung bình.
6. Chọn `--yolo-size` 416 hoặc 640 theo số đo phát hiện người.

**Tiêu chí đạt (đo ở chế độ realtime, 6 người):** AI ≥ 8 FPS, mỗi track ≥ 4 lần phân tích
mặt/giây, độ trễ p95 < 400 ms. Nếu bật hết 6 cách mà vẫn < 8 FPS thì mở bước 2b (model mặt +
hướng đầu chạy batch trên GPU, sau khi kiểm tra M1200/sm_50 chạy được và license đưa vào
manifest).

## 4. Xử lý lỗi

- Model nào không tải được: track ghi `not_run` cho phần đó, script vẫn chạy và ghi lý do
  vào `timing.json`.
- FaceLandmarker lỗi/timeout trên một track không làm hỏng các track khác.
- Video không mở được hoặc hết video: thoát với mã lỗi khác 0 / kết thúc bình thường và
  đóng các track.
- Vượt `max_subjects`: xem 2.1.

## 5. Kiểm thử

Unit test (không cần model):

- Tracker: nối lại ID theo ghế, không nối nhầm khi hai người đổi chỗ gần nhau, đóng track
  sau khi mất quá lâu, giới hạn 6 track.
- Cropper: đổi tọa độ landmark từ ảnh cắt về khung gốc, kẹp biên.
- Bỏ qua khi đứng yên: dùng lại kết quả dưới ngưỡng, chạy lại khi quá 500 ms.
- Scheduler: tối đa 2 track focus, ngân sách lượt, bảo đảm ≥ 4 lần/giây.
- `LOOK_AT_NEIGHBOR`: đúng hướng trái/phải, không báo khi không có người kế bên, cần đủ 1,5 s.
- Tự hiệu chỉnh: không sinh sự kiện quay đầu trước khi có mốc.

Kiểm tra trên video thật:

- Chạy `full` và `realtime` trên video thử, ghi kết quả `timing.json` vào
  `docs/evaluation-results.md`.
- Gắn nhãn tay một danh sách nhỏ (10–20 khoảng hành vi) theo định dạng của
  `proctoring_core.evaluation.labels`, so với `events.jsonl` bằng `scripts/evaluate.py`.
- Đếm số lần đổi ID trên toàn video; mục tiêu 0 với người ngồi yên.

## 6. Thứ tự làm

1. Chọn và tải video thử; đo thời gian YOLO + mặt hiện tại trên ảnh cắt (số liệu nền).
2. Tracker + cropper + face analyzer song song; script `full` xuất video và `timing.json`.
3. Luật theo track + `LOOK_AT_NEIGHBOR` + tự hiệu chỉnh; `events.jsonl`, evidence.
4. FocusScheduler (tay/tư thế), chế độ `realtime`.
5. Các cách tăng tốc 3–6, đo từng cách, chốt cấu hình mặc định.
6. Quyết định có cần bước 2b hay không dựa trên số đo.
