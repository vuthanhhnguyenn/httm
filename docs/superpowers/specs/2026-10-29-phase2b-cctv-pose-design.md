# Phase 2b: Chế độ CCTV cả phòng bằng YOLO-pose — Thiết kế

- Ngày: 2026-10-29
- Trạng thái: đã duyệt 2026-10-29
- Người yêu cầu: Nguyên
- Nối tiếp: `2026-10-29-phase2-multi-subject-design.md` (chế độ webcam 2–6 người, giữ nguyên)

## 1. Mục tiêu và phạm vi

### Vì sao cần

Lần đo trên Quadro M1200 với clip CCTV phòng thi thật (YouTube `EPf9O7UXRoM`, khoảng 25 thí
sinh, camera đặt cao) đạt 12.6–14.6 FPS nhưng kết quả sai:

- YOLO 416 px, ngưỡng 0.6 chỉ thấy tối đa 5 người gần camera; người khả nghi không được theo dõi.
- FaceLandmarker hiếm khi tìm được mặt người đang cúi viết nhìn từ trên cao, nên không ghế nào
  hiệu chỉnh được, không bắt được quay đầu, lại báo sai `FACE_NOT_VISIBLE`.
- Giám thị đi ngang bị tạo thành ghế rồi bị báo `LEAVING_SEAT`.

Thử YOLO11n-pose 960 px, ngưỡng 0.25 trên 3 ảnh chụp của clip: thấy 20–25 người, mỗi người có
điểm mũi/mắt/tai/vai/cổ tay dùng được, kể cả khi đang cúi.

### Đã thống nhất

- Thêm **chế độ CCTV** (`--scene cctv`) cho script `scripts/analyze_video_multi.py`. Chế độ
  webcam hiện tại (`--scene webcam`, mặc định) không đổi.
- Chế độ CCTV dùng **một lượt YOLO-pose trên GPU cho cả phòng**, không cắt mặt từng người,
  không chạy MediaPipe.
- Vẫn là script offline trên file video, chưa sửa app desktop.
- AI không kết luận gian lận; sự kiện chỉ để giám thị ưu tiên rà soát.

### Giả định (sửa nếu sai)

- Camera cố định, góc cao, nhìn về phía mặt thí sinh; 10–40 người; video ≥ 720p.
- Một camera mỗi phòng. Không cần biết số báo danh.

### Tiêu chí đạt (trên clip `EPf9O7UXRoM`, 2 phút đầu, máy M1200, `--mode realtime`)

1. AI ≥ 8 FPS.
2. `max_visible` ≥ 15 người theo dõi cùng lúc.
3. Giám thị đi lại không sinh sự kiện nào.
4. Người quay ra sau / quay sang bên khoảng 0:44–0:47 (ảnh chụp số 3) có sự kiện.
5. Không có `FACE_NOT_VISIBLE`, `LOOK_DOWN`, `GAZE_*` (các luật dựa trên mặt tắt ở chế độ CCTV).
6. Nhãn tay `phase2/nhan-tay-EPf9O7UXRoM.md` (0:02 và 0:51 chụm đầu trao đổi, 0:43 giám thị,
   0:46 quay ra sau, 0:51 quay đầu sang bên); Nguyên ghi tay danh sách mốc thời gian hành vi thật trong clip; báo cáo số sự kiện đúng/sai/sót
   theo danh sách đó.

### Ngoài phạm vi

- Huấn luyện model mới, nhận dạng danh tính, nhiều camera, UI desktop, camera trực tiếp.
- "Thao tác tay lạ" chung chung (ảnh 1): chưa có định nghĩa đo được; cần Nguyên mô tả cụ thể
  hơn hoặc thêm ví dụ trước khi làm luật riêng.

## 2. Kiến trúc

Code mới trong `packages/proctoring_ai/src/proctoring_ai/multi/cctv/`. Dùng lại tracker,
`assign_objects`, `NeighborLookEvaluator`, `TimingRecorder`, writer và luồng `run_multi_replay`.

```
frame ─► PoseDetectionStage (YOLO11n-pose, 960 px, GPU)  ─┐
      └► ObjectDetector (điện thoại, 960 px, mỗi 3 frame) ─┤
                                                          ▼
         MultiSubjectTracker (max 60) ─► SeatRoleClassifier ─► KeypointHeadPose ─► CctvSeatRules ─► events
```

### 2.1 `PoseDetectionStage`

- Model `yolo11n-pose.pt` (tải từ GitHub release của Ultralytics như model YOLO hiện có).
- `imgsz=960`, `conf=0.25`, chỉ lớp người, trên `cuda:0` nếu có (dùng lại `acceleration.py`).
- Đầu ra mỗi người: box + 17 điểm COCO (x, y, độ tin cậy), toạ độ chuẩn hoá.

### 2.2 Tracker

- Dùng `MultiSubjectTracker` sẵn có, `max_subjects=60`; ghế giữ thêm điểm khung xương mới nhất.

### 2.3 `SeatRoleClassifier` — thí sinh hay người đi lại

- Một track thành **thí sinh** khi điểm neo (giữa cạnh dưới box) dịch < 0.25 chiều cao box
  trong 3 giây liên tục.
- Track dịch > 0.5 chiều cao box trong 2 giây, hoặc 5 giây vẫn chưa ổn định, là **người đi lại**
  (giám thị, người qua): vẽ màu xám, không chạy luật.
- Thí sinh đứng dậy rời chỗ (điểm neo ra khỏi chỗ cũ > 5 giây) mới sinh `LEAVING_SEAT`.

### 2.4 `KeypointHeadPose` — hướng đầu từ khung xương

- Chỉ số quay ngang: `(mũi.x − giữa hai tai.x) / độ rộng vai` (thiếu một tai thì dùng mắt).
- **Mốc trung tính theo từng ghế** = trung vị chỉ số trong 5 giây hợp lệ đầu, cập nhật bằng trung vị
  trượt 30 giây. Không đòi đầu đứng yên như cách hiệu chỉnh cũ.
- **Quay ra sau**: mũi và hai mắt có độ tin cậy < 0.3 trong khi hai vai ≥ 0.5, ở ghế trước đó đã
  thấy mặt.

### 2.5 `CctvSeatRules` — luật sự kiện

| Sự kiện | Điều kiện (giữ ≥ 1.5 s trừ khi ghi khác) |
|---|---|
| `LOOK_AT_NEIGHBOR` | lệch khỏi mốc > ngưỡng và hướng về một ghế kề (dùng `NeighborLookEvaluator`) |
| `HEAD_TURN_LEFT/RIGHT` | lệch khỏi mốc > ngưỡng, không có ghế kề phía đó |
| `TURN_AROUND` (mới, "Quay ra phía sau") | điều kiện quay ra sau ở 2.4 |
| `HEADS_TOGETHER` (mới, "Chụm đầu với thí sinh bên cạnh") | khoảng cách đầu hai ghế kề < 0.6 lần khoảng cách bình thường của cặp đó (trung vị trượt 30 s) |
| `SUSPICIOUS_HAND_ACTIVITY` ("trao đổi đồ") | cổ tay của ghế A nằm trong box ghế B ≥ 1 s |
| `PHONE_DETECTED` | điện thoại gán cho ghế có cổ tay gần nhất (`assign_objects`) |
| `LEAVING_SEAT` | như 2.3 |

- `CctvSeatRules` tự mở/đóng sự kiện (cùng định dạng `events.jsonl`: `opened`/`closed`, `id`,
  `type`, `seat_id`, `runtime_track_id`, `video_offset_ms`), không đi qua `SessionRuleProcessor`, nên
  các luật dựa trên mặt không thể chạy ở chế độ CCTV.
- Ngưỡng ban đầu: lệch 0.35 (đơn vị độ rộng vai), chỉnh sau khi có nhãn tay của Nguyên.

## 3. Tốc độ (ước tính trên M1200)

- Pose 960 px khoảng 35–60 ms, phát hiện điện thoại mỗi 3 frame khoảng 12 ms trung bình, luật
  và tracker < 5 ms cho 40 người: khoảng 10–15 FPS.
- Nếu < 8 FPS: giảm `--pose-size 800`, rồi chạy pose mỗi 2 frame và nội suy điểm.

## 4. Xử lý lỗi

- Thiếu model pose hoặc không có GPU: vẫn chạy CPU, ghi lý do vào `timing.json` như hiện nay.
- Điểm khung xương thiếu: ghế ở trạng thái "không đủ dữ liệu", không sinh sự kiện.

## 5. Kiểm thử

- Unit test với khung xương tổng hợp: chỉ số quay đầu, mốc trung tính, quay ra sau, phân loại
  thí sinh/người đi lại, cổ tay sang ghế bên, gán điện thoại.
- Test pipeline với pose detector giả, kiểm tra không có sự kiện dựa trên mặt ở chế độ CCTV.
- Smoke test model thật trên cloud với 3 ảnh chụp: ≥ 15 người.
- Đo thật trên M1200 bằng clip `EPf9O7UXRoM` theo tiêu chí ở mục 1.

## 6. Thứ tự làm

1. `PoseDetectionStage` + tải model + smoke test số người.
2. `SeatRoleClassifier`.
3. `KeypointHeadPose`.
4. `CctvSeatRules` + `TURN_AROUND` + `HEADS_TOGETHER`.
5. Nối vào `analyze_video_multi.py --scene cctv`, vẽ màu theo vai trò, `timing.json`.
6. Đo trên M1200, chỉnh ngưỡng theo nhãn tay, ghi `docs/evaluation-results.md`.
