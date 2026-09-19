# Contract WebSocket của session

## Endpoint

```text
GET /ws/sessions/{session_id}
```

Client gửi bearer token trong cơ chế được runtime hỗ trợ. Ở browser, triển khai ưu tiên secure
cookie hoặc WebSocket subprotocol; không ghi token vào URL hay log. Server kiểm tra quyền xem session
trước khi `accept()` kết nối.

## Message envelope

Mọi message server gửi đều có cùng envelope:

```json
{
  "schema_version": "1.0",
  "sequence": 42,
  "type": "event.created",
  "session_id": "9ec6621b-e69c-4f8f-ac44-ad75602bf38c",
  "occurred_at": "2026-09-19T10:10:12.318Z",
  "data": {}
}
```

| Trường | Quy tắc |
|---|---|
| `schema_version` | Major đổi khi contract không tương thích ngược; minor chỉ thêm trường tùy chọn |
| `sequence` | Số nguyên tăng dần trong một lần chạy session, bắt đầu từ 1 |
| `type` | Một trong các loại message được định nghĩa bên dưới |
| `session_id` | UUID của session trong URL |
| `occurred_at` | UTC ISO 8601, có mili giây |
| `data` | Payload theo từng loại message |

Client phải bỏ qua trường chưa biết trong cùng major version. Nếu nhận major version không hỗ trợ,
client dừng áp dụng message và hiển thị lỗi cần tải lại hoặc nâng cấp.

## Luồng kết nối

1. Server xác thực user và quyền truy cập session.
2. Server gửi `connection.ready` chứa state hiện tại và `latest_sequence`.
3. Client gọi `GET /api/sessions/{session_id}` nếu snapshot cục bộ thiếu hoặc sequence bị đứt.
4. Server gửi update mới theo sequence.
5. Client gửi `ping` định kỳ; server trả `pong`.

WebSocket không phải nguồn lịch sử. Sau reconnect, client lấy snapshot và danh sách event qua REST,
sau đó áp dụng message mới. Cách này tránh phụ thuộc vào buffer replay trong bộ nhớ.

## Message từ server

### `connection.ready`

```json
{
  "schema_version": "1.0",
  "sequence": 40,
  "type": "connection.ready",
  "session_id": "9ec6621b-e69c-4f8f-ac44-ad75602bf38c",
  "occurred_at": "2026-09-19T10:10:10.000Z",
  "data": {
    "session_status": "RUNNING",
    "latest_sequence": 40,
    "server_time": "2026-09-19T10:10:10.000Z"
  }
}
```

### `session.status_changed`

```json
{
  "schema_version": "1.0",
  "sequence": 41,
  "type": "session.status_changed",
  "session_id": "9ec6621b-e69c-4f8f-ac44-ad75602bf38c",
  "occurred_at": "2026-09-19T10:10:11.000Z",
  "data": {
    "previous_status": "STARTING",
    "status": "RUNNING",
    "failure_code": null
  }
}
```

### `camera.status_changed`

```json
{
  "schema_version": "1.0",
  "sequence": 43,
  "type": "camera.status_changed",
  "session_id": "9ec6621b-e69c-4f8f-ac44-ad75602bf38c",
  "occurred_at": "2026-09-19T10:10:13.000Z",
  "data": {
    "status": "DEGRADED",
    "reason_code": "LOW_BRIGHTNESS",
    "message": "Hình ảnh quá tối trong 2 giây gần nhất."
  }
}
```

### `track.updated`

Payload dùng cho overlay và Phase 2 grid. Server có thể gộp nhiều track trong một message.

```json
{
  "schema_version": "1.0",
  "sequence": 44,
  "type": "track.updated",
  "session_id": "9ec6621b-e69c-4f8f-ac44-ad75602bf38c",
  "occurred_at": "2026-09-19T10:10:13.050Z",
  "data": {
    "frame_id": 581,
    "tracks": [
      {
        "id": "a5b944b6-c675-4c3a-882f-80945d5d12be",
        "runtime_track_id": 1,
        "status": "ACTIVE",
        "bbox": {"x": 0.28, "y": 0.09, "width": 0.41, "height": 0.82},
        "visibility": 0.96,
        "risk_score": 38,
        "risk_level": "LOW"
      }
    ]
  }
}
```

`track.updated` có thể được throttle riêng, ví dụ 5 đến 10 lần mỗi giây. Việc giảm tần suất update UI
không được làm chậm temporal engine.

### `event.created`

```json
{
  "schema_version": "1.0",
  "sequence": 45,
  "type": "event.created",
  "session_id": "9ec6621b-e69c-4f8f-ac44-ad75602bf38c",
  "occurred_at": "2026-09-19T10:10:14.000Z",
  "data": {
    "id": "04bca43f-b57b-4c65-8c22-975b46e5ba82",
    "track_id": "a5b944b6-c675-4c3a-882f-80945d5d12be",
    "event_type": "PHONE_DETECTED",
    "status": "OPEN",
    "severity": "HIGH",
    "confidence": 0.91,
    "risk_delta": 30,
    "risk_score_after": 68,
    "started_at": "2026-09-19T10:10:13.100Z",
    "ended_at": null,
    "duration_ms": null,
    "reason_codes": ["PHONE_CONFIDENCE", "PHONE_MIN_DURATION", "POLICY_PHONE_DENIED"],
    "explanation": "Điện thoại xuất hiện 900 ms với confidence 0.91; policy không cho phép điện thoại.",
    "evidence_status": "PENDING"
  }
}
```

### `event.updated`

Dùng khi event được đóng, confidence thay đổi có ý nghĩa hoặc evidence đổi trạng thái.

```json
{
  "schema_version": "1.0",
  "sequence": 46,
  "type": "event.updated",
  "session_id": "9ec6621b-e69c-4f8f-ac44-ad75602bf38c",
  "occurred_at": "2026-09-19T10:10:16.300Z",
  "data": {
    "id": "04bca43f-b57b-4c65-8c22-975b46e5ba82",
    "status": "CLOSED",
    "confidence": 0.93,
    "ended_at": "2026-09-19T10:10:16.200Z",
    "duration_ms": 3100,
    "risk_score_after": 72,
    "evidence_status": "AVAILABLE"
  }
}
```

### `risk.updated`

```json
{
  "schema_version": "1.0",
  "sequence": 47,
  "type": "risk.updated",
  "session_id": "9ec6621b-e69c-4f8f-ac44-ad75602bf38c",
  "occurred_at": "2026-09-19T10:10:16.310Z",
  "data": {
    "track_id": "a5b944b6-c675-4c3a-882f-80945d5d12be",
    "score_before": 68,
    "score_after": 72,
    "level_after": "HIGH",
    "reason": "EVENT_UPDATE",
    "reason_codes": ["PHONE_DURATION_FACTOR"],
    "contributing_event_ids": ["04bca43f-b57b-4c65-8c22-975b46e5ba82"]
  }
}
```

### `metrics.updated`

```json
{
  "schema_version": "1.0",
  "sequence": 48,
  "type": "metrics.updated",
  "session_id": "9ec6621b-e69c-4f8f-ac44-ad75602bf38c",
  "occurred_at": "2026-09-19T10:10:17.000Z",
  "data": {
    "capture_fps": 29.8,
    "processed_fps": 21.4,
    "dropped_frames": 84,
    "latency_ms_p50": 96.2,
    "latency_ms_p95": 181.7,
    "analyzer_latency_ms": {
      "object_detector": 31.4,
      "face_landmarker": 9.8,
      "pose_landmarker": 12.3
    }
  }
}
```

### `system.degraded`

```json
{
  "schema_version": "1.0",
  "sequence": 49,
  "type": "system.degraded",
  "session_id": "9ec6621b-e69c-4f8f-ac44-ad75602bf38c",
  "occurred_at": "2026-09-19T10:10:18.000Z",
  "data": {
    "component": "hand_landmarker",
    "reason_code": "ANALYZER_TIMEOUT",
    "message": "Phân tích tay tạm dừng sau 5 lần timeout liên tiếp.",
    "capabilities_unavailable": ["SUSPICIOUS_HAND_ACTIVITY"],
    "recoverable": true
  }
}
```

### `error`

Lỗi có thể phục hồi ở cấp kết nối hoặc request từ client. Lỗi session nghiêm trọng vẫn đi kèm
`session.status_changed` sang `FAILED`.

```json
{
  "schema_version": "1.0",
  "sequence": 50,
  "type": "error",
  "session_id": "9ec6621b-e69c-4f8f-ac44-ad75602bf38c",
  "occurred_at": "2026-09-19T10:10:19.000Z",
  "data": {
    "code": "INVALID_CLIENT_MESSAGE",
    "message": "Loại message từ client không được hỗ trợ.",
    "request_id": "req_01J..."
  }
}
```

## Message từ client

Client chỉ gửi message điều khiển kết nối. Start và stop session vẫn dùng REST cùng
`Idempotency-Key`.

### `ping`

```json
{
  "type": "ping",
  "client_time": "2026-09-19T10:10:20.000Z"
}
```

Server trả:

```json
{
  "type": "pong",
  "client_time": "2026-09-19T10:10:20.000Z",
  "server_time": "2026-09-19T10:10:20.004Z"
}
```

## Ordering và mất message

- Server cấp `sequence` sau khi state change đã commit hoặc được event bus chấp nhận.
- Client chỉ áp dụng message có sequence lớn hơn sequence cuối đã xử lý.
- Nếu sequence nhảy quá một đơn vị, client đánh dấu state là stale và tải lại snapshot REST.
- Message duplicate được bỏ qua theo sequence.
- Không suy ra duration từ thời điểm client nhận message; dùng timestamp trong event.

## Đóng kết nối

| Close code | Ý nghĩa | Hành vi client |
|---:|---|---|
| `1000` | Session dừng bình thường | Không tự reconnect trừ khi user yêu cầu |
| `1008` | Sai policy hoặc không đủ quyền | Dừng reconnect, yêu cầu đăng nhập hoặc đổi quyền |
| `1011` | Lỗi server tạm thời | Backoff và reconnect, sau đó tải snapshot |
| `4001` | Token hết hạn | Refresh token rồi kết nối lại một lần |
| `4004` | Không tìm thấy session | Dừng reconnect và quay lại danh sách session |
| `4009` | Major schema không tương thích | Dừng kết nối và yêu cầu tải lại ứng dụng |

Reconnect dùng exponential backoff có jitter, giới hạn 30 giây. UI luôn hiển thị trạng thái mất kết
nối; dữ liệu cũ không được trình bày như dữ liệu đang cập nhật.
