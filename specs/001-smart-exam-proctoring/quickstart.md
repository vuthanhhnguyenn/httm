# Hướng dẫn chạy và xác nhận thiết kế

Tài liệu này mô tả cách kiểm tra hệ thống sau khi implementation hoàn tất. Các lệnh là contract vận
hành mà `tasks.md` phải hiện thực. Nếu tên lệnh thay đổi trong lúc xây dựng, cập nhật tài liệu này cùng
thay đổi code.

## 1. Điều kiện chuẩn bị

- Python 3.12.x và `uv`.
- Node.js bản LTS đang được hỗ trợ và npm.
- Webcam mà OpenCV đọc được, hoặc bộ video trong `tests/video_samples/`.
- GPU NVIDIA cùng driver phù hợp nếu kiểm tra mục tiêu 15 FPS trở lên ở 720p.
- Model manifest và model file có checksum hợp lệ trong `models/pretrained/`.
- Quyết định giấy phép Ultralytics đã được ghi nhận trước khi phát hành hoặc triển khai production.

Kiểm tra công cụ:

```powershell
python --version
uv --version
node --version
npm --version
```

Kết quả mong đợi: Python báo 3.12.x; các công cụ còn lại chạy mà không lỗi.

## 2. Cài phụ thuộc

Tại repository root:

```powershell
uv sync --extra opencv-python --dev
npm install --prefix apps/web --no-audit --no-fund
```

Không cài đồng thời `opencv-python` và `opencv-python-headless` trong cùng environment. Máy phát triển
có preview native dùng `uv sync --extra opencv-python --dev`; CI không cần GUI dùng
`uv sync --extra opencv-python-headless --dev`.

## 3. Tạo cấu hình cục bộ

```powershell
Copy-Item .env.example .env
Copy-Item configs/default.example.yaml configs/default.yaml
Copy-Item configs/exam_policy.example.yaml configs/exam_policy.yaml
```

Giá trị tối thiểu cần kiểm tra trong `.env`:

```text
APP_ENV=development
DATABASE_URL=sqlite+aiosqlite:///./storage/proctoring.db
STORAGE_ROOT=./storage/sessions
AUTH_MODE=development
```

`AUTH_MODE=development` chỉ hợp lệ khi `APP_ENV=development`. Ứng dụng phải từ chối khởi động nếu
cấu hình này xuất hiện trong production.

## 4. Kiểm tra config và model

```powershell
uv run python scripts/validate_config.py --config configs/default.yaml --policy configs/exam_policy.yaml
uv run python scripts/verify_models.py --manifest models/manifest.yaml
```

Kết quả mong đợi:

- Config hợp lệ, không còn threshold ngoài schema.
- Model file tồn tại và checksum khớp manifest.
- Manifest ghi model source, version, class map và license.
- Nếu SQLite runtime không có bản vá WAL phù hợp, validator thông báo dùng rollback journal và single writer.

## 5. Khởi tạo database

```powershell
uv run alembic upgrade head
```

Sau migration, kiểm tra schema:

```powershell
uv run python scripts/check_database.py
```

Kết quả mong đợi: database có đủ bảng trong [data-model.md](./data-model.md), config version đầu tiên
được active và policy mặc định đã được import.

## 6. Chạy ứng dụng

Terminal 1, chạy API:

```powershell
uv run uvicorn proctoring_api.main:app --app-dir apps/api/src --host 127.0.0.1 --port 8000 --reload
```

Terminal 2, chạy frontend:

```powershell
npm run dev --prefix apps/web
```

Mở URL do Vite in ra. Trước khi start session, màn hình phải cho biết camera chưa chạy và risk bằng 0.

Kiểm tra health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Kết quả mong đợi: `status` là `healthy` hoặc `degraded`. Nếu degraded, response phải nêu component
không sẵn sàng; UI không được hiển thị trạng thái monitoring đầy đủ.

## 7. Kịch bản smoke test Phase 1

### 7.1 Tạo và start session

Dùng giao diện tạo session `SINGLE_PERSON`, chọn webcam và policy mặc định. Nhấn Start.

Xác nhận:

- State đi qua `CREATED`, `STARTING`, rồi `RUNNING`.
- Camera preview xuất hiện trong 2 giây ở môi trường tham chiếu.
- Camera status, processing status, FPS và risk đều có giá trị.
- Timeline chưa có event khi thí sinh ngồi bình thường.
- WebSocket message có sequence tăng dần.

### 7.2 Head turn ngắn

Quay đầu qua ngưỡng rồi trở lại trước `min_duration_ms`.

Xác nhận: không tạo `HEAD_TURN_LEFT` hoặc `HEAD_TURN_RIGHT`. Metric có thể thay đổi nhưng timeline
không có event mới.

### 7.3 Head turn đủ thời gian

Quay đầu qua ngưỡng lâu hơn `min_duration_ms`.

Xác nhận:

- Timeline có đúng một event head turn ở trạng thái mở rồi đóng.
- Event có confidence, start/end, duration, reason code, explanation và risk effect.
- Event frame có trạng thái `AVAILABLE`, trừ khi storage được cố ý cấu hình lỗi.

### 7.4 Phone và exam policy

Chạy cùng video phone hai lần:

1. Policy có `allow_phone: false`.
2. Policy có `allow_phone: true`.

Lần đầu phải tạo `PHONE_DETECTED` khi đạt confidence và duration. Lần sau không tạo event và không
tăng risk vì phone. Detector result có thể tồn tại trong runtime, nhưng policy quyết định event.

### 7.5 Person missing và multiple person

Dùng video cố định để tránh khác biệt khi thao tác trực tiếp:

```powershell
uv run python scripts/replay.py tests/video_samples/person_missing.mp4 --policy configs/exam_policy.yaml
uv run python scripts/replay.py tests/video_samples/second_person.mp4 --policy configs/exam_policy.yaml
```

Xác nhận mỗi video tạo đúng loại event sau minimum duration. Một người đi ngang ngắn hơn ngưỡng không
được tạo `MULTIPLE_PERSON_DETECTED`.

### 7.6 Stop session

Nhấn Stop hai lần liên tiếp.

Xác nhận:

- Lệnh có tính idempotent.
- State kết thúc ở `STOPPED`.
- Camera được giải phóng, temporal buffer bị xóa và WebSocket đóng bằng code `1000`.
- Mở lại session history vẫn thấy event và evidence đã lưu.

## 8. Test tự động

### 8.1 Python

```powershell
uv run pytest packages/proctoring_core/tests -q
uv run pytest packages/proctoring_ai/tests -q
uv run pytest apps/api/tests tests/contract tests/integration -q
```

Các test bắt buộc gồm:

- Boundary duration, frequency, confidence và sliding window.
- Event open/update/close không tạo duplicate theo frame.
- Risk cap, decay, correlation cap và explanation.
- Exam policy cho phép hoặc chặn material đúng cách.
- Camera/analyzer/storage failure chuyển hệ thống sang degraded hoặc failed có thông báo.
- Authorization cho evidence, config và policy.
- Retention xóa file an toàn và chạy lại được.
- WebSocket envelope, sequence gap và reconnect snapshot.

### 8.2 Frontend

```powershell
npm run test --prefix apps/web
npm run test:e2e --prefix apps/web
```

Xác nhận component dùng type sinh từ OpenAPI hoặc schema contract, timeline xử lý event update đúng ID,
và UI hiển thị trạng thái stale khi WebSocket mất kết nối.

## 9. Video regression

Manifest mẫu:

```text
tests/video_samples/manifest.yaml
```

Chạy toàn bộ suite:

```powershell
uv run python scripts/evaluate.py --manifest tests/video_samples/manifest.yaml --output artifacts/evaluation
```

Suite tối thiểu gồm:

```text
normal.mp4
head_turn.mp4
phone_usage.mp4
look_down.mp4
second_person.mp4
leave_seat.mp4
camera_blocked.mp4
mixed_behavior.mp4
```

Kết quả mong đợi trong `artifacts/evaluation/`:

- Event precision, recall và F1 theo loại.
- False alerts mỗi phút, gồm xác nhận không có false `HIGH` hoặc `CRITICAL` trong tập normal.
- Missed events mỗi session và detection delay.
- Detector precision/recall; tracking metric khi tracker được bật.
- Config hash, model hash, commit ID và thông tin phần cứng.

Lệnh trả exit code khác 0 nếu vi phạm acceptance gate đã cấu hình.

## 10. Benchmark

Chạy webcam trong 60 giây:

```powershell
uv run python scripts/benchmark.py --source 0 --width 1280 --height 720 --duration 60 --output artifacts/benchmark
```

Chạy bằng video để so sánh giữa các commit:

```powershell
uv run python scripts/benchmark.py --source tests/video_samples/mixed_behavior.mp4 --realtime --output artifacts/benchmark
```

Report phải có CPU, GPU, RAM, OS, model, precision, input resolution, capture FPS, processed FPS,
dropped frames, latency p50/p95 và latency từng analyzer.

Gate Phase 1 trên GPU tham chiếu:

- Processed FPS tối thiểu 15 ở 720p.
- 95% event/status update xuất hiện trong 300 ms sau thời điểm đủ điều kiện.
- Không có queue tăng không giới hạn; dropped frame được đếm.

## 11. Kiểm tra degraded mode

Chạy với hand analyzer bị tắt:

```powershell
uv run python scripts/replay.py tests/video_samples/mixed_behavior.mp4 --set analyzers.hand.enabled=false
```

Xác nhận:

- Session vẫn chạy nếu các analyzer bắt buộc sẵn sàng.
- Health và UI ghi rõ `SUSPICIOUS_HAND_ACTIVITY` không khả dụng.
- Không có event hand giả được tạo để bù phần thiếu.

Lặp lại với storage root chỉ đọc. Event vẫn phải được persist với evidence `FAILED`; UI hiển thị lỗi và
không tạo đường dẫn file giả.

## 12. Kiểm tra retention và audit

Tạo một policy test có retention ngắn trong môi trường test, sau đó chạy:

```powershell
uv run python scripts/retention.py --dry-run
uv run python scripts/retention.py --execute
```

Xác nhận dry-run không thay đổi dữ liệu. Lần execute xóa file đã hết hạn, chuyển metadata sang
`DELETED` và tạo audit record. Truy cập lại content trả `410 Gone`.

## 13. Gate trước Phase 2

Chỉ bắt đầu Phase 2 khi Phase 1 đạt acceptance criteria và có baseline được lưu. Sau đó chạy benchmark
theo 1, 5, 10 và, nếu phần cứng mục tiêu cho phép, 20 người:

```powershell
uv run python scripts/benchmark.py --manifest tests/video_samples/room_manifest.yaml --people 1,5,10,20 --output artifacts/room-benchmark
```

Report Phase 2 bổ sung ID switches, track loss, fragmentation, occlusion recovery và accuracy của
object-to-person association. Object ambiguous phải được đếm riêng, không tính như gán đúng.

## 14. Đối chiếu contract

- REST contract: [contracts/openapi.yaml](./contracts/openapi.yaml)
- WebSocket contract: [contracts/websocket-events.md](./contracts/websocket-events.md)
- Data model và state transition: [data-model.md](./data-model.md)
- Quyết định kỹ thuật và nguồn: [research.md](./research.md)

Một bản build chỉ đạt Definition of Done khi các lệnh test, regression và benchmark liên quan chạy
thành công, đồng thời output ghi đủ config hash, model hash và môi trường thực thi.
