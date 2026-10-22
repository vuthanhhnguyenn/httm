# Hợp đồng: dữ liệu phiên (JSONL v2), cấu hình và giao diện

Mở rộng [hợp đồng spec 001](../../001-smart-exam-proctoring/contracts/desktop-review.md). Reader phải
đọc được dòng v1 (thiếu trường mới hiển thị "không có dữ liệu").

## `storage/sessions/<id>/session.json` (mới)

```json
{"schema_version": "2.0", "session_id": "…", "exam_mode": "paper|computer",
 "started_at": "…", "config_sha256": "…", "policy_sha256": "…",
 "models": {"…": "…"}, "calibration": {"head": "calibrated", "gaze": "uncalibrated", "seat": "calibrated"}}
```

## `events.jsonl` — dòng sự kiện

Giữ mọi trường v1 (`id, action, type, severity, confidence, explanation, metrics, occurred_at,
risk_score, risk_level, cumulative_score, peak_score, event_count, risk_reason_codes, evidence_status,
label_vi, severity_vi, message_vi`) và thêm:

| Trường | Bắt buộc | Mô tả |
|---|---|---|
| `schema_version` | có | `"2.0"` |
| `episode_id` | có | ổn định qua `opened/updated/closed` của cùng episode |
| `exam_mode` | có | |
| `supporting_signals` | có (có thể rỗng) | loại tín hiệu hệ quả bị gộp vào sự kiện này |
| `object_trace` | khi là sự kiện vật thể | `{frame_id, bbox, confidence, model_name, model_version, object_track_id}` |
| `experimental` | có | `true` cho `GAZE_*`, `POSSIBLE_TALKING` |
| `risk_cap` | khi có | trần điểm của tín hiệu hỗ trợ |

## `events.jsonl` — dòng rà soát (append-only)

```json
{"schema_version": "2.0", "action": "review", "event_id": "…", "episode_id": "…",
 "verdict": "unreviewed|valid_signal|false_positive|inconclusive", "note": "",
 "reviewed_at": "…", "reviewer": "local-operator"}
```

- Không sửa dòng cũ; trạng thái hiện tại là dòng `review` cuối cùng cho `event_id`.
- Không đổi `risk_score`/`peak_score` đã ghi.

## Cấu hình — khóa mới (tóm tắt; schema Pydantic là nguồn chính)

```yaml
behavior:
  quality: { face_min_quality: 0.35, max_sample_gap_ms: 600 }
  head_turn: { … giữ nguyên … }
  look_down: { enabled: true, … }
  gaze:
    enabled: true
    min_eye_width_px: 18
    min_openness: 0.18
    max_head_yaw_degrees: 35
    off_screen_margin: 0.12
    off_screen_min_duration_seconds: 1.5
    sideways_ratio: 0.32
    sideways_min_duration_seconds: 2.0
    risk_cap: 15
  talking:
    enabled: true
    window_seconds: 2.0
    min_oscillations: 3
    min_amplitude: 0.05
    min_std: 0.02
    yawn_mar: 0.55
    min_duration_seconds: 2.5
    risk_cap: 15
  hands:
    near_face_radius: 0.55        # theo độ rộng mặt
    near_face_min_duration_seconds: 2.0
    rapid_speed: 1.8              # độ rộng mặt / giây
    rapid_min_duration_seconds: 2.0
    risk_cap: 25
  phone: { … giữ …, episode_gap_ms: 1200, geometry_filter: false }
  document: { classes: [book], … }
  leaving_seat: { minimum_duration_seconds: 2.0, zone_margin: 0.15, min_area_ratio: 0.35 }
  camera_blocked: { minimum_duration_seconds: 2.0, max_brightness: 25, min_contrast: 8, min_edge_density: 0.01, blur_threshold: 20 }
  multiple_person: { minimum_duration_seconds: 2.0 }
exam_modes:
  paper:    { look_down: { enabled: false }, gaze: { off_screen_enabled: false } }
  computer: { look_down: { enabled: true, minimum_duration_seconds: 3.0 } }
hierarchy:
  PERSON_MISSING: [FACE_NOT_VISIBLE, LEAVING_SEAT, HEAD_TURN_LEFT, HEAD_TURN_RIGHT, LOOK_DOWN, ABNORMAL_HEAD_MOVEMENT, GAZE_OFF_SCREEN, GAZE_SIDEWAYS, POSSIBLE_TALKING, SUSPICIOUS_HAND_ACTIVITY]
  CAMERA_BLOCKED: [ … ]
  LEAVING_SEAT: [FACE_NOT_VISIBLE, HEAD_TURN_LEFT, HEAD_TURN_RIGHT, LOOK_DOWN, GAZE_OFF_SCREEN, GAZE_SIDEWAYS, POSSIBLE_TALKING]
evaluation: { analysis_rate_hz: 8, tolerance_ms: 2000, report_dir: storage/evaluation }
```

Giá trị số ở đây là mặc định khởi đầu; giá trị cuối cùng chốt theo báo cáo đánh giá và ghi lại trong
`docs/event-detection-tuning.md`.

## Giao diện (PySide6)

- Trước khi bắt đầu: chọn camera, chọn **Kiểu bài thi** (Thi giấy / Thi máy, nhớ lựa chọn trước).
  Khóa lựa chọn khi phiên chạy.
- Bảng trạng thái thêm: "Hướng mắt: chưa hiệu chỉnh / đã hiệu chỉnh / chất lượng thấp", "Hiệu chỉnh đầu:
  đề xuất — nhấn để chấp nhận".
- Nút **Hiệu chỉnh** mở luồng: giữ đầu yên nhìn tâm (đầu) → nhìn lần lượt các điểm hiển thị trên màn
  hình (thi máy: tâm + 4 góc) hoặc tâm + "nhìn xuống bài" (thi giấy). Mỗi điểm tối đa 4 giây; thất bại
  hiển thị lý do.
- Danh sách sự kiện: một hàng cho mỗi episode; `updated` cập nhật hàng, không thêm hàng. Tín hiệu thử
  nghiệm có nhãn "Thử nghiệm".
- Hộp chi tiết sự kiện: ảnh đúng frame + khung bao, độ tin cậy, mô hình/phiên bản, kiểu bài thi, lý do,
  tín hiệu phụ, và ba nút "Hợp lệ", "Báo nhầm", "Không kết luận".
