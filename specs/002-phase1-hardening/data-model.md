# Data Model: Hoàn thiện Phase 1

**Feature**: [spec.md](./spec.md) | **Research**: [research.md](./research.md)

> Cập nhật sau implement: giữ `ObservationBundle` (dict) ở ranh giới AI, `Observation` có kiểu ở lớp
> nghiệp vụ; tái dùng `ReviewType` sẵn có thay vì kiểu mới; `SignalState` có ba giá trị.

Các kiểu dưới đây là dataclass bất biến (`frozen=True, slots=True`) trong `proctoring_core`, trừ khi
ghi khác. Tọa độ chuẩn hóa `[0, 1]` theo khung hình; thời gian đơn điệu tính bằng ms.

## 1. Kiểu bài thi

```text
ExamMode(StrEnum): PAPER = "paper" | COMPUTER = "computer"
```

- Chọn trước phiên; lưu trong `ExamSession.policy_snapshot["exam_mode"]`; bất biến trong phiên.

## 2. Quan sát (`proctoring_core.observations`)

| Kiểu | Trường | Ràng buộc |
|---|---|---|
| `BBox` | `x, y, width, height: float` | mọi giá trị trong `[0,1]`, `x+width ≤ 1`, `y+height ≤ 1` (tái dùng `types.BoundingBox`) |
| `PersonObs` | `bbox, confidence, track_id: str \| None` | confidence `[0,1]` |
| `ObjectObs` | `label, confidence, bbox, object_track_id: int \| None, track_confirmed: bool, model_name, model_version` | label thuộc class map của mô hình |
| `EyeObs` | `h_ratio, v_ratio: float, openness: float, width_px: float, visible: bool` | ratio thô trước hiệu chỉnh |
| `GazeObs` | `left: EyeObs, right: EyeObs, h: float, v: float, quality: float, status: GazeStatus` | `status ∈ {uncalibrated, calibrated, low_quality}` |
| `MouthObs` | `mar: float, width_ratio: float, occluded: bool` | mar ≥ 0 |
| `FaceObs` | `bbox, yaw, pitch, roll, pose_valid, quality, calibration_status, landmarks_ref: FaceKeypoints, gaze: GazeObs \| None, mouth: MouthObs \| None` | quality `[0,1]` |
| `FaceKeypoints` | các điểm mốc tên hóa dùng cho luật: `left_ear, right_ear, chin, mouth_center, nose_tip` | chỉ giữ điểm cần thiết, không lưu toàn bộ mesh |
| `HandObs` | `bbox, wrist, index_tip, handedness, confidence, hand_id: int` | `hand_id` ổn định giữa các lượt nhờ ghép tay |
| `HandMotion` | `hand_id, speed: float` | tốc độ chuẩn hóa theo độ rộng mặt / giây |
| `PoseObs` | `bbox, left_shoulder, right_shoulder, quality` | |
| `ImageQualityObs` | `brightness, contrast, edge_density, blur, frozen: bool` | |
| `AnalyzerStatus` | `name, status, reason, latency_ms` | status ∈ `ready, no_detection, skipped, busy, timeout, error, not_run, unavailable` |
| `Observation` | `frame_id, captured_at_utc, monotonic_ms, frame_size, people, objects, faces, hands, hand_motion, poses, image_quality, analyzers` | thay cho các list dict trong `ObservationBundle`; `ObservationBundle` giữ làm adapter trong giai đoạn chuyển đổi |

## 3. Tín hiệu luật (`proctoring_core.behavior.signals`)

```text
RuleSignal
  event_type: EventType
  state: SignalState            # ACTIVE | CANDIDATE | INACTIVE (INACTIVE = vừa kết thúc hoặc bị thứ bậc giữ)
  session_id: UUID
  track_id: str | None
  confidence: float             # [0,1]
  severity: Severity
  metrics: Mapping[str, float | int | str | bool]
  reason_codes: tuple[str, ...]
  risk_cap: float | None        # tín hiệu hỗ trợ: gaze, talking, hands
  root_of: tuple[EventType,...] # do SignalArbiter điền
  supporting: tuple[EventType,...]
```

- `active` của dict cũ = `state is ACTIVE`; CANDIDATE chỉ do luật gốc (presence, safety) báo để
  `SignalArbiter` giữ tín hiệu hệ quả sớm. `trace` (object_trace) đi kèm tín hiệu vật thể.
- `EventAggregator.process` nhận `RuleSignal` (giữ hàm chuyển đổi từ dict cho kiểm thử cũ tới khi gỡ).

## 4. Loại sự kiện mới

```text
EventType += GAZE_OFF_SCREEN, GAZE_SIDEWAYS
```

- `POSSIBLE_TALKING` đã có, nay có luật thật.
- `SUSPICIOUS_HAND_ACTIVITY` giữ tên; reason code phân biệt `HAND_NEAR_FACE`, `HAND_NEAR_EAR`,
  `HAND_RAPID_MOVEMENT`.

## 5. Thứ bậc tín hiệu

```text
SignalHierarchy (config.hierarchy)
  roots: {PERSON_MISSING: [FACE_NOT_VISIBLE, LEAVING_SEAT, HEAD_*, LOOK_DOWN, GAZE_*, POSSIBLE_TALKING,
                           SUSPICIOUS_HAND_ACTIVITY],
          CAMERA_BLOCKED: [mọi loại trừ MULTIPLE_PERSON_DETECTED],
          LEAVING_SEAT:   [FACE_NOT_VISIBLE, HEAD_*, GAZE_*, POSSIBLE_TALKING]}
```

- Một tín hiệu bị thứ bậc chặn không mở sự kiện; tên của nó thêm vào `supporting_signals` của sự kiện gốc.
- Không có chu trình (kiểm tra khi nạp cấu hình).

## 6. Hồ sơ hiệu chỉnh (`proctoring_core.calibration`)

```text
CalibrationStatus(StrEnum): UNCALIBRATED, SUGGESTED, COLLECTING, CALIBRATED, FAILED

HeadBaseline: yaw_offset, pitch_offset, roll_offset, captured_at_ms
GazeTarget: name ("center","top_left","top_right","bottom_left","bottom_right","paper"), h, v, head_yaw, head_pitch
GazeCalibration: targets: tuple[GazeTarget,...], h_bounds: (min,max), v_bounds: (min,max), status
SeatZone: bbox: BBox, person_area: float
CalibrationProfile: head: HeadBaseline | None, gaze: GazeCalibration | None, seat: SeatZone | None,
                    head_status, gaze_status, updated_at_utc
```

Chuyển trạng thái (cho cả head và gaze):

```text
UNCALIBRATED ──(ổn định 2s)──▶ SUGGESTED ──(giám thị chấp nhận)──▶ COLLECTING
UNCALIBRATED ──(giám thị bấm hiệu chỉnh)──▶ COLLECTING
COLLECTING ──(đủ mẫu ổn định)──▶ CALIBRATED
COLLECTING ──(hết thời hạn / mất mặt)──▶ FAILED ──(thử lại)──▶ COLLECTING
CALIBRATED ──(giám thị hiệu chỉnh lại)──▶ COLLECTING
```

## 7. Sự kiện (mở rộng `SuspiciousEvent`)

Trường thêm:

| Trường | Kiểu | Ghi chú |
|---|---|---|
| `exam_mode` | `ExamMode` | từ phiên |
| `supporting_signals` | `tuple[EventType, ...]` | do thứ bậc |
| `object_trace` | `ObjectTrace \| None` | `frame_id, bbox, confidence, model_name, model_version, object_track_id` (FR-015) |
| `review` | `ReviewLabel \| None` | |
| `video_offset_ms` | `int \| None` | chỉ có khi phát lại |

```text
ReviewType (đã có trong events/review_models.py): VALID_SIGNAL, FALSE_POSITIVE, INCONCLUSIVE
Dòng review JSONL: verdict (chữ thường), note (≤ 500 ký tự), reviewed_at, reviewer ("local-operator")
Không có dòng review = chưa rà soát.
```

- Bốn trạng thái theo hợp đồng spec 001 (`chưa_rà_soát`, `tín_hiệu_hợp_lệ`, `báo_nhầm`,
  `không_kết_luận`). Giao diện hiển thị nút "Báo nhầm", "Hợp lệ", "Không kết luận".

- Gắn nhãn rà soát không đổi `risk_score_after` đã ghi; điểm lịch sử giữ nguyên (FR-028).

## 8. Đánh giá (`proctoring_core.evaluation`)

```text
LabelInterval: event_type: EventType, start_ms, end_ms, source_code: str, ignore: bool
VideoCase: case_id, video_path, labels: tuple[LabelInterval,...], exam_mode, fps, duration_ms, source
EvaluationSet: name, cases, mapping_version, license_note
PredictedEvent: event_type, start_ms, end_ms, severity
TypeMetrics: event_type, tp, fn, fp, duplicates, coverage, false_alerts_per_min, median_delay_ms,
             labelled_ms, unlabelled_ms
CaseResult: case_id, per_type: Mapping[EventType, TypeMetrics], frames_analyzed, ai_fps
EvaluationReport: set_name, created_at_utc, config_sha256, model_versions, hardware, analysis_rate_hz,
                  cases, totals, baseline_delta: Mapping[EventType, Mapping[str, float]] | None
```

Quy tắc kiểm tra:
- `0 ≤ start_ms < end_ms ≤ duration_ms` (nhãn vượt thời lượng → lỗi có tên tệp và số dòng).
- Mã nhãn gốc không có trong ánh xạ → lỗi, trừ khi ánh xạ ghi `ignore`.

## 9. Cấu hình (mở rộng `AppConfig`)

```text
AppConfig
  camera, detection, risk, evidence, runtime   (giữ)
  behavior: BehaviorConfig                      (mở rộng: thresholds, gaze, talking, hands, leaving_seat, camera_blocked)
  exam_modes: {paper: BehaviorOverrides, computer: BehaviorOverrides}
  hierarchy: HierarchyConfig
  evaluation: EvaluationConfig                  (analysis_rate_hz, tolerance_ms, report_dir)
ExamPolicyConfig += exam_mode_default: ExamMode
```

- `resolve_behavior(config, mode) -> BehaviorConfig`: hợp nhất sâu `exam_modes[mode]` vào `behavior`,
  rồi kiểm tra lại toàn bộ.
- Mặc định của mọi khóa mới = giá trị đang viết cứng trong mã hiện tại (bước tái cấu trúc không đổi hành vi).
