# Phase 2 Multi-Subject Video Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Một script offline phân tích 2–6 thí sinh trong một file video: ID ổn định theo ghế, mặt + hướng đầu cho mọi người, tay/tư thế cho ≤ 2 người đang nghi, sự kiện theo từng người, video có vẽ khung và bảng thời gian từng model.

**Architecture:** Code mới trong `proctoring_ai/multi/` (tracker, cropper, motion gate, face pool, focus, bundle builder, pipeline, timing) và `proctoring_runtime/multi_replay.py` (vòng chạy + ghi đầu ra). Luật sự kiện phase 1 được dùng lại nguyên vẹn bằng cách chạy **một `SessionRuleProcessor` cho mỗi ghế**, mỗi processor nhận một `ObservationBundle` chỉ chứa người, mặt, tay/tư thế và vật thể của ghế đó. Tín hiệu mới `LOOK_AT_NEIGHBOR` cần hình học của mọi ghế nên chạy ở cấp nhiều người và được đưa vào processor của ghế qua `extra_signals`.

**Tech Stack:** Python 3.12, numpy, OpenCV, ultralytics YOLO11n (GPU), MediaPipe Tasks (CPU), pytest + pytest-asyncio (`asyncio_mode = "auto"`).

**Spec:** `docs/superpowers/specs/2026-10-29-phase2-multi-subject-design.md`

## Global Constraints

- Python 3.12; ruff `line-length = 100`; mypy `strict = true` cho mã nguồn (test không strict).
- Không thêm phụ thuộc mới (không `lap`, không ByteTrack); không chạy `uv sync`/`uv run` trong `.venv-gpu`.
- Model chỉ nạp qua `models/manifest.yaml` có SHA-256 (dùng lại `AnalyzerFactory._resolve_model`).
- Video thử và đầu ra (`storage/multi/...`) không đưa vào Git.
- AI không kết luận gian lận; nhãn hiển thị tiếng Việt, mã kỹ thuật giữ tiếng Anh trong JSONL.
- `max_subjects = 6`; ảnh cắt đầu: 45% trên của box người, lề 15%, cạnh ngắn ≥ 256 px (`--crop-size`).
- Bỏ qua khi đứng yên: ảnh xám 64×64, dùng lại kết quả tối đa 500 ms.
- Tự hiệu chỉnh: `suggest_after_ms = 3000`, tự chấp nhận gợi ý mốc.
- Focus: tối đa 2 ghế, giữ 5 s; `frame_budget_ms = 110`; mỗi ghế ≥ 4 lần phân tích mặt/giây (khoảng cách ≤ 250 ms).
- Nối lại theo ghế: bán kính 0,5 × chiều rộng box, ghế mất < 10 s.
- `LOOK_AT_NEIGHBOR`: ngưỡng yaw = `behavior.head_turn.yaw_degrees`, người kế bên cùng hàng, khoảng cách ngang < 2,5 × chiều rộng box, kéo dài ≥ 1,5 s, mức LOW, có `risk_cap`.
- Tiêu chí tốc độ (realtime, 6 người, M1200): AI ≥ 8 FPS, mặt ≥ 4 lần/giây/ghế, độ trễ p95 < 400 ms.
- Chạy test: `python -m pytest <path> -v` từ thư mục gốc repo (trên Windows: `.\.venv-gpu\Scripts\python.exe -m pytest ...`).

## Review Focus

1. Hai người chồng lấn (người ngồi sau lọt vào ảnh cắt đầu của người trước): FaceLandmarker có thể trả mặt của người sau. Mong đợi: mặt có tâm nằm ngoài box người của ghế bị loại với `status="mismatch"`. Test ở Task 6.
2. Người sát mép khung (box chạm biên ảnh): ảnh cắt phải bị kẹp trong ảnh, không lỗi, và landmark đổi về đúng tọa độ khung. Test ở Task 4.
3. Video ngắn hơn 3 s hoặc người luôn cử động nên không bao giờ hiệu chỉnh xong: script chạy hết, không sinh sự kiện quay đầu, `timing.json` ghi `calibration_status` từng ghế. Test ở Task 11.
4. Video không có người nào: đầu ra vẫn được ghi đủ, `timing.json` không chia cho 0. Test ở Task 10.
5. Ghế hết hạn (thí sinh rời đi > 10 s) khi đang có sự kiện mở: sự kiện của ghế đó phải được đóng, không treo tới cuối video. Test ở Task 11.

---

## File Structure

| File | Trách nhiệm |
|---|---|
| `packages/proctoring_core/src/proctoring_core/types.py` | thêm `EventType.LOOK_AT_NEIGHBOR` |
| `packages/proctoring_core/src/proctoring_core/behavior/neighbor.py` | luật nhìn sang người bên cạnh |
| `packages/proctoring_core/src/proctoring_core/risk/engine.py`, `config/schema.py`, `proctoring_runtime/messages.py` | trọng số, nhóm tín hiệu mặt, nhãn tiếng Việt |
| `packages/proctoring_runtime/src/proctoring_runtime/session_engine.py` | `extra_signals`, `last_signals` |
| `packages/proctoring_ai/src/proctoring_ai/multi/tracker.py` | ghế, ghép IoU, nối lại, giới hạn 6 |
| `.../multi/crop.py` | ảnh cắt đầu/thân + đổi tọa độ |
| `.../multi/motion.py` | bỏ qua khi đứng yên |
| `.../multi/face_pool.py` | FaceLandmarker theo ghế, chạy song song |
| `.../multi/focus.py` | chọn ghế và lịch chạy mặt/tay/tư thế |
| `.../multi/focus_analyzers.py` | tay/tư thế cho các ghế focus |
| `.../multi/bundles.py` | gán vật thể, dựng `ObservationBundle` theo ghế |
| `.../multi/pipeline.py` | ghép các bước cho một frame |
| `.../multi/timing.py` | số liệu `timing.json` |
| `packages/proctoring_ai/src/proctoring_ai/face/landmarks.py` | `suggest_after_ms`, `auto_calibrate()` |
| `packages/proctoring_ai/src/proctoring_ai/analyzer_factory.py` | `build_multi_pipeline()` |
| `packages/proctoring_runtime/src/proctoring_runtime/multi_replay.py` | vòng chạy full/realtime, processor theo ghế, ghi đầu ra |
| `scripts/analyze_video_multi.py` | CLI |
| Tests | `packages/proctoring_core/tests/behavior/test_neighbor.py`, `packages/proctoring_ai/tests/multi/test_*.py`, `tests/test_multi_replay.py` |

---

### Task 1: Event type `LOOK_AT_NEIGHBOR` và luật `NeighborLookEvaluator`

**Files:**
- Modify: `packages/proctoring_core/src/proctoring_core/types.py:75-90` (thêm `LOOK_AT_NEIGHBOR = "LOOK_AT_NEIGHBOR"`)
- Modify: `packages/proctoring_core/src/proctoring_core/risk/engine.py:16-32` (`EventType.LOOK_AT_NEIGHBOR: 5.0`)
- Modify: `packages/proctoring_core/src/proctoring_core/config/schema.py:201-204` (thêm vào `_FACE_SIGNALS`)
- Modify: `packages/proctoring_runtime/src/proctoring_runtime/messages.py:~23` (`"LOOK_AT_NEIGHBOR": "Quay đầu về phía thí sinh bên cạnh"`)
- Create: `packages/proctoring_core/src/proctoring_core/behavior/neighbor.py`
- Test: `packages/proctoring_core/tests/behavior/test_neighbor.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True, slots=True) class SubjectHead: track_id: str; box: Box; yaw: float | None; quality: float; calibrated: bool` (`Box` từ `proctoring_core.observations`)
  - `class NeighborLookEvaluator(*, yaw_degrees: float, min_duration_ms: int = 1500, max_gap_widths: float = 2.5, min_quality: float = 0.35, max_sample_gap_ms: int = 600, risk_cap: float = 25.0)`
  - `NeighborLookEvaluator.evaluate(*, session_id: Any, subjects: Sequence[SubjectHead], monotonic_ms: int) -> list[RuleSignal]` — một tín hiệu cho mỗi `track_id` có trạng thái khác INACTIVE, cộng một tín hiệu INACTIVE khi điều kiện vừa đóng.
  - `NeighborLookEvaluator.forget(track_id: str) -> list[RuleSignal]` — trả tín hiệu INACTIVE nếu ghế đang mở.

Quy tắc (không suy ra được từ chữ ký):
- Kế bên = ghế khác có `abs(cy_other - cy) < 0.5 * box.height` và `abs(cx_other - cx) < max_gap_widths * box.width`; nếu nhiều thì lấy gần nhất theo x ở phía đầu đang quay.
- `yaw > yaw_degrees` ⇒ hướng sang phải ảnh (cx tăng); `yaw < -yaw_degrees` ⇒ sang trái. Đúng điều kiện chỉ khi có người kế bên ở phía đó.
- Mỗi ghế một `TemporalCondition(min_duration_ms, max_sample_gap_ms)`; ghế chưa `calibrated`, `yaw is None` hoặc `quality < min_quality` cho mẫu `False`.
- `RuleSignal`: `severity=Severity.LOW`, `reason_codes=("HEAD_TOWARDS_NEIGHBOR",)`, `metrics` gồm `duration_ms`, `yaw`, `neighbor_track_id`, `risk_cap`; `rule_key="LOOK_AT_NEIGHBOR"`.

- [ ] **Step 1: Write the failing tests**

```python
from uuid import uuid4
from proctoring_core.behavior.neighbor import NeighborLookEvaluator, SubjectHead
from proctoring_core.behavior.signals import SignalState
from proctoring_core.observations import Box
from proctoring_core.types import EventType, Severity

LEFT = Box(0.10, 0.30, 0.15, 0.40)   # cx = 0.175
RIGHT = Box(0.30, 0.32, 0.15, 0.40)  # cx = 0.375, same row
FAR = Box(0.80, 0.30, 0.15, 0.40)    # 4.2 widths away

def run(evaluator, yaw_left, ms_values, others=(("b", RIGHT, 0.0),)):
    session = uuid4(); out = []
    for ms in ms_values:
        subjects = [SubjectHead("a", LEFT, yaw_left, 0.9, True)]
        subjects += [SubjectHead(t, b, y, 0.9, True) for t, b, y in others]
        out = evaluator.evaluate(session_id=session, subjects=subjects, monotonic_ms=ms)
    return [s for s in out if s.track_id == "a"]

def test_turn_towards_right_neighbor_activates_after_1500_ms():
    signals = run(NeighborLookEvaluator(yaw_degrees=22), 30.0, range(0, 1600, 100))
    assert signals[-1].state is SignalState.ACTIVE
    assert signals[-1].event_type is EventType.LOOK_AT_NEIGHBOR
    assert signals[-1].severity is Severity.LOW
    assert signals[-1].metrics["neighbor_track_id"] == "b"

def test_shorter_than_1500_ms_stays_candidate():
    signals = run(NeighborLookEvaluator(yaw_degrees=22), 30.0, range(0, 1400, 100))
    assert signals[-1].state is SignalState.CANDIDATE

def test_turn_away_from_neighbor_is_ignored():
    assert run(NeighborLookEvaluator(yaw_degrees=22), -30.0, range(0, 2000, 100)) == []

def test_no_neighbor_within_2_5_widths_is_ignored():
    assert run(NeighborLookEvaluator(yaw_degrees=22), 30.0, range(0, 2000, 100),
               others=(("b", FAR, 0.0),)) == []

def test_uncalibrated_subject_never_fires():
    evaluator = NeighborLookEvaluator(yaw_degrees=22); session = uuid4(); last = []
    for ms in range(0, 2000, 100):
        last = evaluator.evaluate(session_id=session, monotonic_ms=ms, subjects=[
            SubjectHead("a", LEFT, 30.0, 0.9, False), SubjectHead("b", RIGHT, 0.0, 0.9, True)])
    assert [s for s in last if s.track_id == "a"] == []

def test_forget_closes_open_condition():
    evaluator = NeighborLookEvaluator(yaw_degrees=22)
    run(evaluator, 30.0, range(0, 1600, 100))
    closed = evaluator.forget("a")
    assert [s.state for s in closed] == [SignalState.INACTIVE]
```

Also add to the existing risk test file a one-line check `assert RiskEngine.DEFAULT_WEIGHTS[EventType.LOOK_AT_NEIGHBOR] == 5.0`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest packages/proctoring_core/tests/behavior/test_neighbor.py -v`
Expected: FAIL with `ModuleNotFoundError: proctoring_core.behavior.neighbor`

- [ ] **Step 3: Add the enum value, weight, face-signal group entry and Vietnamese label; implement `NeighborLookEvaluator` in `behavior/neighbor.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest packages/proctoring_core/tests -v`
Expected: PASS (whole core suite, so the new enum member breaks nothing)

- [ ] **Step 5: Commit**

```bash
git add packages/proctoring_core packages/proctoring_runtime/src/proctoring_runtime/messages.py
git commit -m "feat(core): add LOOK_AT_NEIGHBOR rule for multi-subject sessions"
```

---

### Task 2: `SessionRuleProcessor` nhận tín hiệu ngoài và giữ tín hiệu gần nhất

**Files:**
- Modify: `packages/proctoring_runtime/src/proctoring_runtime/session_engine.py:170-200`
- Test: `tests/test_multi_replay.py` (tạo file; các task sau thêm test vào đây)

**Interfaces:**
- Consumes: `RuleSignal` (Task 1).
- Produces:
  - `SessionRuleProcessor.process(bundle, frame=None, *, extra_signals: Sequence[RuleSignal] = ()) -> list[dict[str, Any]]` — `extra_signals` được xử lý sau `rules.evaluate(...)`, qua cùng aggregator/risk.
  - `SessionRuleProcessor.last_signals: tuple[RuleSignal, ...]` — mọi tín hiệu (kể cả CANDIDATE) của lần `process` gần nhất, gồm cả `extra_signals`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_extra_signal_opens_event_through_aggregator():
    processor = create_replay_processor(session_id, policy, raw_config)
    signal = RuleSignal(EventType.LOOK_AT_NEIGHBOR, SignalState.ACTIVE, session_id, str(seat_id),
                        0.8, Severity.LOW, ("HEAD_TOWARDS_NEIGHBOR",), {"risk_cap": 25.0})
    items = await processor.process(empty_bundle(session_id, frame_id=1, ms=100), extra_signals=[signal])
    assert [(i["type"], i["action"]) for i in items] == [("LOOK_AT_NEIGHBOR", "opened")]
    assert items[0]["label_vi"] == "Quay đầu về phía thí sinh bên cạnh"

async def test_last_signals_include_candidates_and_extra():
    ...
    await processor.process(bundle, extra_signals=[candidate_signal])
    assert candidate_signal in processor.last_signals
```

`empty_bundle(...)` là helper trong file test: `ObservationBundle` 640×480, không người, `analyzer_health={"object_detector": ObservationHealth("ready")}`. `policy`/`raw_config` đọc từ `configs/exam_policy.yaml` và `configs/default.yaml` như `scripts/evaluate.py` làm.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_multi_replay.py -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'extra_signals'`

- [ ] **Step 3: Implement** — build `signals = [*self.rules.evaluate(...), *extra_signals]`, store `self.last_signals = tuple(signals)`, loop unchanged. Initialise `last_signals = ()` in `__init__`.

- [ ] **Step 4: Run** `python -m pytest tests/test_multi_replay.py tests/test_event_responsiveness.py -v` — Expected: PASS

- [ ] **Step 5: Commit** `git commit -m "feat(runtime): accept extra rule signals per processed frame"`

---

### Task 3: `MultiSubjectTracker` — ghế, ghép IoU, nối lại, giới hạn 6

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/__init__.py` (rỗng, có docstring)
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/tracker.py`
- Test: `packages/proctoring_ai/tests/multi/test_tracker.py`

**Interfaces:**
- Produces:
  - `@dataclass(slots=True) class Seat: seat_id: UUID; runtime_track_id: int; bbox: Box; anchor: tuple[float, float]; confidence: float; first_seen_ms: int; last_seen_ms: int; visible: bool`
  - `@dataclass(frozen=True, slots=True) class TrackerUpdate: analyzed: list[Seat]; over_limit: list[Seat]; missing: list[Seat]; expired: list[Seat]; created: int; relinked: int`
  - `class IdAssigner(Protocol): def assign(self, seats: Sequence[Seat], boxes: Sequence[Box]) -> dict[int, UUID]` (box index → seat id)
  - `class IouAssigner(min_iou: float = 0.3)` — ghép tham lam theo IoU giảm dần.
  - `class MultiSubjectTracker(*, max_subjects: int = 6, relink_radius: float = 0.5, relink_window_ms: int = 10_000, anchor_alpha: float = 0.1, assigner: IdAssigner | None = None)`
  - `MultiSubjectTracker.update(people: Sequence[Mapping[str, Any]], monotonic_ms: int) -> TrackerUpdate` — `people` là detection YOLO lớp `person` (dict có `bbox`, `confidence`).
  - `MultiSubjectTracker.seats: dict[UUID, Seat]` (ghế còn hạn).

Quy tắc: ghép IoU chạy trên mọi ghế còn hạn (đang thấy và đang mất). `relinked` đếm số ghế đang mất được thấy lại trong lần `update` này (qua IoU hoặc anchor). Box không ghép được → nối với ghế đang mất (chưa hết hạn) có `anchor` cách tâm box < `relink_radius * seat.bbox.width` (gần nhất), ngược lại tạo ghế mới với `runtime_track_id` tăng từ 1. `anchor` = EMA của tâm box. Ghế mất ≥ `relink_window_ms` vào `expired` và bị xóa. Nếu số ghế thấy được > `max_subjects`, `analyzed` = `max_subjects` ghế có diện tích box lớn nhất, còn lại vào `over_limit`.

- [ ] **Step 1: Write the failing tests**

```python
def person(x, y, w=0.15, h=0.4, c=0.9): return {"class": "person", "confidence": c,
                                                 "bbox": {"x": x, "y": y, "width": w, "height": h}}

def test_ids_stable_while_people_shift_slightly():
    t = MultiSubjectTracker()
    first = t.update([person(0.1, 0.3), person(0.4, 0.3)], 0)
    ids = [s.runtime_track_id for s in first.analyzed]
    later = t.update([person(0.11, 0.31), person(0.41, 0.29)], 100)
    assert sorted(s.runtime_track_id for s in later.analyzed) == sorted(ids) == [1, 2]

def test_neighbours_swapping_jitter_do_not_swap_ids():
    t = MultiSubjectTracker(); t.update([person(0.10, 0.3), person(0.27, 0.3)], 0)
    by_x = lambda u: [s.runtime_track_id for s in sorted(u.analyzed, key=lambda s: s.bbox.x)]
    assert by_x(t.update([person(0.12, 0.3), person(0.26, 0.3)], 100)) == [1, 2]

def test_returning_person_relinks_to_same_seat_within_10_s():
    t = MultiSubjectTracker(); t.update([person(0.1, 0.3, w=0.3)], 0)   # centre (0.25, 0.5)
    assert [s.runtime_track_id for s in t.update([], 500).missing] == [1]
    back = t.update([person(0.1, 0.55, w=0.3, h=0.1)], 9_000)  # IoU 0.25 < 0.3, centre 0.1 < 0.15 away
    assert [s.runtime_track_id for s in back.analyzed] == [1] and back.relinked == 1

def test_seat_expires_after_10_s_and_new_person_gets_new_id():
    t = MultiSubjectTracker(); t.update([person(0.1, 0.3)], 0)
    gone = t.update([], 10_000)
    assert [s.runtime_track_id for s in gone.expired] == [1]
    assert [s.runtime_track_id for s in t.update([person(0.1, 0.3)], 10_100).analyzed] == [2]

def test_more_than_six_people_keeps_six_largest():
    t = MultiSubjectTracker()
    people = [person(0.02 + i * 0.13, 0.3, w=0.10 + 0.002 * i) for i in range(7)]
    update = t.update(people, 0)
    assert len(update.analyzed) == 6 and len(update.over_limit) == 1
    assert update.over_limit[0].bbox.width == pytest.approx(0.10)
```

- [ ] **Step 2: Run** `python -m pytest packages/proctoring_ai/tests/multi/test_tracker.py -v` — Expected: FAIL (module missing)
- [ ] **Step 3: Implement `tracker.py`**
- [ ] **Step 4: Run again** — Expected: PASS
- [ ] **Step 5: Commit** `git commit -m "feat(ai): add seat-based multi-subject tracker"`

---

### Task 4: `SubjectCropper` và đổi tọa độ

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/crop.py`
- Test: `packages/proctoring_ai/tests/multi/test_crop.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True, slots=True) class CropTransform: x0: int; y0: int; width: int; height: int; frame_width: int; frame_height: int` (pixel trong khung gốc)
    - `map_point(x: float, y: float) -> tuple[float, float]` — chuẩn hóa theo ảnh cắt → chuẩn hóa theo khung
    - `map_box(box: Mapping[str, float]) -> dict[str, float]`
    - `map_face(face: Mapping[str, Any]) -> dict[str, Any]` — đổi `bbox`, mọi điểm trong `keypoints`, nhân `face_width` với `width / frame_width`, đặt `frame_width/frame_height` của khung
    - `map_hand(hand) -> dict` — `bbox`, `landmarks[*]`, `wrist`, `index_tip`
    - `map_pose(pose) -> dict` — `bbox`, `landmarks[*]`
  - `@dataclass(frozen=True, slots=True) class Crop: image: Any; transform: CropTransform` (`image` BGR, liền bộ nhớ, đã phóng)
  - `class SubjectCropper(*, head_fraction: float = 0.45, margin: float = 0.15, min_side: int = 256, body_margin: float = 0.10)`
    - `crop_head(image: Any, box: Box) -> Crop | None`
    - `crop_body(image: Any, box: Box) -> Crop | None`
    - Trả `None` khi vùng sau khi kẹp < 4 px một chiều. Phóng bằng `cv2.resize(..., INTER_LINEAR)` sao cho cạnh ngắn ≥ `min_side` (không thu nhỏ).

- [ ] **Step 1: Write the failing tests**

```python
np = pytest.importorskip("numpy"); pytest.importorskip("cv2")

def test_head_crop_is_upper_45_percent_with_margin_and_upscaled():
    image = np.zeros((480, 640, 3), np.uint8)
    crop = SubjectCropper().crop_head(image, Box(0.25, 0.25, 0.25, 0.5))  # 160×240 px person
    t = crop.transform
    assert (t.x0, t.width) == (160 - 24, 160 + 48)          # 15% of width each side
    assert t.y0 == 120 - 16 and t.height == round(240 * 0.45) + 2 * 16  # 15% of head height
    assert min(crop.image.shape[:2]) >= 256

def test_landmark_maps_back_to_frame_coordinates():
    t = CropTransform(x0=100, y0=50, width=200, height=100, frame_width=640, frame_height=480)
    assert t.map_point(0.5, 0.5) == pytest.approx((200 / 640, 100 / 480))
    face = t.map_face({"bbox": {"x": 0, "y": 0, "width": 1, "height": 1}, "face_width": 0.5,
                       "keypoints": {"nose": (0.5, 0.5)}})
    assert face["bbox"] == pytest.approx({"x": 100/640, "y": 50/480, "width": 200/640, "height": 100/480})
    assert face["face_width"] == pytest.approx(0.5 * 200 / 640)
    assert face["keypoints"]["nose"] == pytest.approx((200 / 640, 100 / 480))
    assert (face["frame_width"], face["frame_height"]) == (640, 480)

def test_box_touching_frame_edge_is_clamped():
    crop = SubjectCropper().crop_head(np.zeros((480, 640, 3), np.uint8), Box(0.9, 0.0, 0.2, 0.5))
    t = crop.transform
    assert t.x0 + t.width <= 640 and t.y0 == 0

def test_degenerate_box_returns_none():
    assert SubjectCropper().crop_head(np.zeros((480, 640, 3), np.uint8), Box(0.5, 0.5, 0.001, 0.001)) is None
```

Add a `map_hand` test asserting `wrist` and `landmarks[0]` are mapped with the same formula.

- [ ] **Step 2: Run** `python -m pytest packages/proctoring_ai/tests/multi/test_crop.py -v` — Expected: FAIL
- [ ] **Step 3: Implement `crop.py`**
- [ ] **Step 4: Run again** — Expected: PASS
- [ ] **Step 5: Commit** `git commit -m "feat(ai): add subject crops with frame coordinate mapping"`

---

### Task 5: `MotionGate` — bỏ qua người đứng yên

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/motion.py`
- Test: `packages/proctoring_ai/tests/multi/test_motion.py`

**Interfaces:**
- Produces: `class MotionGate(*, threshold: float = 4.0, max_reuse_ms: int = 500, size: int = 64)`
  - `should_run(seat_id: UUID, image: Any, monotonic_ms: int) -> bool` — `True` khi chưa có mẫu, khi đã quá `max_reuse_ms` kể từ lần chạy gần nhất, hoặc khi trung bình |chênh lệch| ảnh xám `size×size` (thang 0–255) ≥ `threshold`.
  - `record(seat_id: UUID, image: Any, monotonic_ms: int) -> None` — lưu mẫu sau một lần chạy thật.
  - `forget(seat_id: UUID) -> None`
- `threshold = 4.0` là giá trị khởi đầu; Task 12 chỉnh lại theo số đo.

- [ ] **Step 1: Write the failing tests**

```python
def test_static_crop_reused_until_500_ms():
    gate = MotionGate(); seat = uuid4(); img = np.full((120, 100, 3), 80, np.uint8)
    assert gate.should_run(seat, img, 0)  # first sample always runs
    gate.record(seat, img, 0)
    assert not gate.should_run(seat, img, 499)
    assert gate.should_run(seat, img, 500)

def test_moving_crop_runs_again():
    ...; moved = img.copy(); moved[:60] = 200
    assert gate.should_run(seat, moved, 100)

def test_different_crop_sizes_are_compared_after_resize():
    ...; assert not gate.should_run(seat, np.full((240, 200, 3), 80, np.uint8), 100)
```

- [ ] **Step 2–4:** run `python -m pytest packages/proctoring_ai/tests/multi/test_motion.py -v` (FAIL → implement → PASS)
- [ ] **Step 5: Commit** `git commit -m "feat(ai): add motion gate to reuse face results for still subjects"`

---

### Task 6: `PerSubjectFaceAnalyzer` — FaceLandmarker theo ghế, song song, tự hiệu chỉnh

**Files:**
- Modify: `packages/proctoring_ai/src/proctoring_ai/face/landmarks.py:23-36` (thêm tham số và phương thức)
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/face_pool.py`
- Test: `packages/proctoring_ai/tests/multi/test_face_pool.py`, thêm 1 test vào `packages/proctoring_ai/tests/face/test_head_suggestion.py`

**Interfaces:**
- Consumes: `Crop`, `CropTransform.map_face` (Task 4), `MotionGate` (Task 5), `Box`.
- Produces:
  - `MediaPipeFaceLandmarker(..., suggest_after_ms: int = 2000)` — truyền vào `HeadPoseEstimator(suggest_after_ms=...)`.
  - `MediaPipeFaceLandmarker.auto_calibrate() -> str` — nếu `calibration_status == "suggested"` thì gọi `begin_calibration()`; trả trạng thái hiện tại.
  - `class FaceAnalyzerLike(Protocol)`: `async def initialize() -> dict[str, Any]`, `async def analyze(frame: dict[str, Any]) -> dict[str, Any]`, `def auto_calibrate() -> str`, `def close() -> None`
  - `@dataclass(frozen=True, slots=True) class FaceResult: status: str; face: dict[str, Any] | None; reused: bool; latency_ms: float; calibration_status: str`
  - `class PerSubjectFaceAnalyzer(factory: Callable[[], FaceAnalyzerLike], *, max_instances: int = 6, motion_gate: MotionGate | None = None)`
    - `async analyze(crops: Mapping[UUID, Crop], seat_boxes: Mapping[UUID, Box], *, frame_id: int, monotonic_ms: int, force: Set[UUID] = frozenset()) -> dict[UUID, FaceResult]`
    - `release(seat_id: UUID) -> None` (đóng instance, quên motion gate), `close() -> None`

Hành vi:
- Instance tạo lười theo ghế (`await instance.initialize()` lần đầu); không bao giờ quá `max_instances` (ghế thừa: `FaceResult("not_run", None, False, 0, "uncalibrated")`).
- Mọi ghế cần chạy được `asyncio.gather` cùng lúc; mỗi lời gọi nhận frame `{"image": crop.image, "frame_id": frame_id, "captured_at_monotonic_ms": monotonic_ms}`.
- Ghế không trong `force` và `motion_gate.should_run(...)` là `False` → trả kết quả trước đó với `reused=True` (không gọi analyzer).
- Kết quả `faces[0]` được `map_face`; nếu tâm `bbox` đã đổi nằm ngoài `seat_boxes[seat]` thì `status="mismatch"`, `face=None`.
- Sau mỗi lần chạy thật gọi `instance.auto_calibrate()` và ghi vào `calibration_status`.

- [ ] **Step 1: Write the failing tests** (dùng `FakeFace` trả kết quả định sẵn, đếm số lần gọi, ghi thời điểm bắt đầu/kết thúc)

```python
async def test_each_seat_gets_its_own_instance_and_mapped_face(): ...
    assert len(created) == 2
    assert results[a].face["bbox"]["x"] == pytest.approx(crop_a.transform.x0 / 640)

async def test_seats_run_concurrently():
    # FakeFace.analyze sleeps 50 ms via asyncio.sleep; three seats finish in < 120 ms
    assert elapsed < 0.12

async def test_still_seat_reuses_previous_result_unless_forced():
    first = await pool.analyze(...); second = await pool.analyze(..., monotonic_ms=100)
    assert second[a].reused and fake.calls == 1
    third = await pool.analyze(..., monotonic_ms=200, force={a})
    assert not third[a].reused and fake.calls == 2

async def test_face_centred_outside_seat_box_is_mismatch():
    assert results[a].status == "mismatch" and results[a].face is None

async def test_seventh_seat_is_not_run(): ...

async def test_one_seat_raising_does_not_affect_others():
    # FakeFace for seat a raises RuntimeError → results[a].status == "error"; results[b].status == "ready"

async def test_release_closes_instance(): ...
```

In `test_head_suggestion.py`: `test_auto_calibrate_accepts_suggestion_after_3_s` — `MediaPipeFaceLandmarker(None, suggest_after_ms=3000)`, feed its `_pose_estimator` 3.1 s of steady samples, `auto_calibrate()` returns `"calibrated"`; at 2.0 s it returns `"uncalibrated"`.

- [ ] **Step 2: Run** `python -m pytest packages/proctoring_ai/tests/multi/test_face_pool.py packages/proctoring_ai/tests/face -v` — Expected: FAIL
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run again** — Expected: PASS
- [ ] **Step 5: Commit** `git commit -m "feat(ai): run one face landmarker per seat in parallel"`

---

### Task 7: `FocusScheduler` và `FocusAnalyzers`

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/focus.py`
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/focus_analyzers.py`
- Test: `packages/proctoring_ai/tests/multi/test_focus.py`

**Interfaces:**
- Consumes: `Crop`, `CropTransform.map_hand/map_pose` (Task 4).
- Produces:
  - `FOCUS_EVENT_TYPES: frozenset[EventType]` = `{HEAD_TURN_LEFT, HEAD_TURN_RIGHT, LOOK_DOWN, PHONE_DETECTED, LOOK_AT_NEIGHBOR}`
  - `@dataclass(frozen=True, slots=True) class FramePlan: face_seats: frozenset[UUID]; force_face: frozenset[UUID]; focus_seats: tuple[UUID, ...]; run_focus: bool`
  - `class FocusScheduler(*, max_focus: int = 2, focus_hold_ms: int = 5000, frame_budget_ms: float = 110.0, min_face_interval_ms: int = 250)`
    - `mark(seat_id: UUID, monotonic_ms: int) -> None`
    - `plan(seat_ids: Sequence[UUID], monotonic_ms: int, *, last_frame_ms: float | None) -> FramePlan`
    - `record_face(seat_id: UUID, monotonic_ms: int) -> None`; `forget(seat_id: UUID) -> None`
  - `@dataclass(frozen=True, slots=True) class FocusResult: hands: list[dict[str, Any]]; poses: list[dict[str, Any]]; hands_status: str; pose_status: str; latency_ms: float`
  - `class FocusAnalyzers(hand_factory: Callable[[], Any] | None, pose_factory: Callable[[], Any] | None, *, slots: int = 2, hands_interval_ms: int = 200, pose_interval_ms: int = 250)`
    - `async analyze(crops: Mapping[UUID, Crop], *, frame_id: int, monotonic_ms: int) -> dict[UUID, FocusResult]`
    - `close() -> None`

Quy tắc `plan`:
- Focus = ghế được `mark` trong `focus_hold_ms` gần nhất, mới nhất trước, tối đa `max_focus`.
- `last_frame_ms is None or last_frame_ms <= frame_budget_ms` ⇒ `face_seats` = mọi ghế, `run_focus = True`.
- Quá ngân sách ⇒ `run_focus = False`; `face_seats` = ghế chưa từng chạy mặt + ghế focus + ghế có `now - last_face >= min_face_interval_ms`.
- `force_face` = ghế chưa từng chạy mặt ∪ ghế focus (bỏ qua motion gate).

Quy tắc `FocusAnalyzers`: mỗi slot gắn với một ghế; ghế mới chiếm slot của ghế không còn trong `crops` → đóng instance cũ, tạo mới qua factory (VIDEO mode cần lịch sử riêng). Tay/tư thế chỉ chạy khi đã qua interval của slot; ngược lại trạng thái `"skipped"`. Kết quả được `map_hand`/`map_pose`. Factory `None` ⇒ trạng thái `"not_run"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_marked_seats_are_focus_for_5_s_max_two_newest_first():
    s = FocusScheduler(); a, b, c = uuid4(), uuid4(), uuid4()
    s.mark(a, 0); s.mark(b, 100); s.mark(c, 200)
    assert s.plan([a, b, c], 300, last_frame_ms=50).focus_seats == (c, b)
    assert s.plan([a, b, c], 5_300, last_frame_ms=50).focus_seats == ()

def test_over_budget_skips_focus_and_rotates_stable_seats_at_4_hz():
    ...; s.record_face(a, 0); s.record_face(b, 100)
    plan = s.plan([a, b], 260, last_frame_ms=140)
    assert not plan.run_focus and plan.face_seats == {a}

def test_new_and_focus_seats_are_forced(): ...

async def test_focus_slot_rebinds_and_closes_previous_instance(): ...
async def test_hands_respect_interval(): ...   # second call at +100 ms → hands_status "skipped"
```

- [ ] **Step 2–4:** `python -m pytest packages/proctoring_ai/tests/multi/test_focus.py -v` (FAIL → implement → PASS)
- [ ] **Step 5: Commit** `git commit -m "feat(ai): schedule face rotation and hands/pose focus per seat"`

---

### Task 8: Dựng `ObservationBundle` theo ghế

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/bundles.py`
- Test: `packages/proctoring_ai/tests/multi/test_bundles.py`

**Interfaces:**
- Consumes: `Seat` (Task 3), `FaceResult` (Task 6), `FocusResult` (Task 7), `ObservationBundle`, `ObservationHealth`.
- Produces:
  - `assign_objects(objects: Sequence[Mapping[str, Any]], seats: Sequence[Seat], *, expand: float = 0.2) -> dict[UUID, list[dict[str, Any]]]` — chỉ lớp khác `person`; gán cho ghế có diện tích giao (box ghế nới 20% mỗi phía) / diện tích vật lớn nhất; không giao ghế nào thì bỏ.
  - `build_subject_bundle(*, frame: Mapping[str, Any], session_id: UUID, seat: Seat, visible: bool, face: FaceResult | None, focus: FocusResult | None, objects: list[dict[str, Any]], image_quality: dict[str, Any], detector_health: ObservationHealth) -> ObservationBundle`

Quy tắc `build_subject_bundle`:
- `people = [{"class": "person", "bbox": seat.bbox.as_dict(), "confidence": seat.confidence, "track_id": str(seat.seat_id), "runtime_track_id": seat.runtime_track_id}]` nếu `visible`, ngược lại `[]`.
- `faces = [face.face]` khi `face.status == "ready"` và có mặt; health `face`: `ready`/`no_detection` theo `FaceResult.status` (`mismatch` → `no_detection`), `not_run` khi ghế không thấy được (`reason="SUBJECT_NOT_VISIBLE"`).
- `hands`/`poses` và health `hands`/`pose` từ `FocusResult`; không có → `ObservationHealth("not_run", reason="NOT_IN_FOCUS")`.

- [ ] **Step 1: Write the failing tests**

```python
def test_phone_goes_to_seat_with_largest_overlap(): ...
def test_object_far_from_every_seat_is_dropped(): ...
def test_visible_seat_bundle_contains_only_its_person_and_face():
    bundle = build_subject_bundle(...)
    assert [p["track_id"] for p in bundle.people] == [str(seat.seat_id)]
    assert bundle.analyzer_health["face"].status == "ready"
    assert Observation.from_bundle(bundle).people[0].track_id == str(seat.seat_id)
def test_missing_seat_bundle_has_no_person_and_face_not_run(): ...
def test_mismatch_face_is_reported_as_no_detection(): ...
```

- [ ] **Step 2–4:** `python -m pytest packages/proctoring_ai/tests/multi/test_bundles.py -v` (FAIL → implement → PASS)
- [ ] **Step 5: Commit** `git commit -m "feat(ai): split frame observations into per-seat bundles"`

---

### Task 9: `MultiSubjectPipeline` và `AnalyzerFactory.build_multi_pipeline`

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/pipeline.py`
- Modify: `packages/proctoring_ai/src/proctoring_ai/analyzer_factory.py` (lưu đường dẫn model đã kiểm SHA trong `_initialize_landmarker`; thêm `build_multi_pipeline`)
- Test: `packages/proctoring_ai/tests/multi/test_pipeline.py`

**Interfaces:**
- Consumes: Task 3–8.
- Produces:
  - `@dataclass(frozen=True, slots=True) class FrameTiming: detect_ms: float; track_ms: float; face_ms: float; face_each_ms: dict[UUID, float]; focus_ms: float; total_ms: float; fresh_faces: int; reused_faces: int; detector_reused: bool`
  - `@dataclass(frozen=True, slots=True) class DetectionStage: frame: dict[str, Any]; detections: list[dict[str, Any]]; tracker: TrackerUpdate; detector_health: ObservationHealth; detect_ms: float; track_ms: float; detector_reused: bool`
  - `@dataclass(frozen=True, slots=True) class MultiFrameResult: frame_id: int; monotonic_ms: int; tracker: TrackerUpdate; bundles: dict[UUID, ObservationBundle]; faces: dict[UUID, FaceResult]; focus: dict[UUID, FocusResult]; objects: list[dict[str, Any]]; timing: FrameTiming`
  - `class MultiSubjectPipeline(*, detector: Any, tracker: MultiSubjectTracker, cropper: SubjectCropper, faces: PerSubjectFaceAnalyzer, focus_analyzers: FocusAnalyzers, scheduler: FocusScheduler, image_quality: Any | None = None, person_confidence_threshold: float = 0.6, yolo_every: int = 1, timeout_ms: int = 1200)`
    - `async detect(frame: dict[str, Any]) -> DetectionStage`
    - `async analyze_subjects(stage: DetectionStage, *, session_id: UUID) -> MultiFrameResult`
    - `async analyze(frame: dict[str, Any], *, session_id: UUID) -> MultiFrameResult` (= hai bước trên)
    - `mark_focus(seat_id: UUID, monotonic_ms: int) -> None` (chuyển cho scheduler)
    - `close() -> None`
  - `AnalyzerFactory.build_multi_pipeline(config: dict[str, Any] | None = None, *, max_subjects: int = 6, crop_size: int = 256, yolo_every: int = 1, motion_skip: bool = True, focus: bool = True) -> MultiSubjectPipeline` — face factory `lambda: MediaPipeFaceLandmarker(path, suggest_after_ms=3000, **options["face"])`; hand/pose factory tương tự (hoặc `None` khi `focus=False` hay model không sẵn sàng).

Hành vi `detect`: `yolo_every = 2` ⇒ frame có chỉ số lẻ (đếm từ 0 trong pipeline) dùng lại detection gần nhất, `detector_reused=True`, health `ObservationHealth("ready", 0.0, "REUSED_DETECTIONS")`. Lọc `person` theo `person_confidence_threshold` trước khi đưa vào tracker. Lỗi/timeout detector ⇒ detections rỗng, health tương ứng (giống `ObservationPipeline.detect`).

Hành vi `analyze_subjects`: plan từ scheduler với `last_frame_ms` = `total_ms` của frame trước; cắt đầu cho `face_seats`, chạy face pool với `force=plan.force_face`; nếu `run_focus` thì cắt thân và chạy focus cho `focus_seats`; `image_quality` chạy một lần cho cả khung; bundle cho mọi ghế trong `analyzed` (visible) và `missing` (không visible). Ghế trong `analyzed` nhưng không nằm trong `face_seats` thì **không có bundle** frame này (luật không nhận mẫu thiếu mặt giả). Ghế `expired` ⇒ `faces.release`, `scheduler.forget`.

- [ ] **Step 1: Write the failing tests** (fake detector trả danh sách định sẵn, fake face factory)

```python
async def test_two_people_get_two_bundles_with_their_own_faces(): ...
async def test_yolo_every_two_reuses_detections_on_odd_frames():
    assert [r.timing.detector_reused for r in results] == [False, True, False, True]
    assert detector.calls == 2
async def test_seat_skipped_by_rotation_has_no_bundle(): ...
async def test_detector_error_yields_missing_seats_not_exception(): ...
async def test_expired_seat_releases_face_instance(): ...
```

Factory test (skip when models absent): `test_build_multi_pipeline_without_models_reports_unavailable` — factory with an empty manifest builds a pipeline whose detector raises the configured reason, exactly like `build_pipeline`.

- [ ] **Step 2–4:** `python -m pytest packages/proctoring_ai/tests/multi -v` (FAIL → implement → PASS)
- [ ] **Step 5: Commit** `git commit -m "feat(ai): compose multi-subject pipeline"`

---

### Task 10: `TimingRecorder`

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/timing.py`
- Test: `packages/proctoring_ai/tests/multi/test_timing.py`

**Interfaces:**
- Consumes: `MultiFrameResult`, `FrameTiming` (Task 9).
- Produces: `class TimingRecorder`
  - `add(result: MultiFrameResult, *, latency_ms: float) -> None`
  - `add_dropped() -> None`
  - `set_calibration(runtime_track_id: int, status: str) -> None`
  - `summary(*, wall_s: float) -> dict[str, Any]` với khóa: `frames`, `dropped_frames`, `ai_fps` (`frames / wall_s`), `latency_ms: {p50, p95}`, `models: {detect, track, face_frame, face_each, focus, total}` mỗi khóa `{p50, p95}`, `detector_reuse_ratio`, `face_reuse_ratio`, `seats: {runtime_track_id: {face_hz, fresh_face_hz, calibration_status}}`, `seats_created`, `id_switches` (= `seats_created - max số ghế thấy được cùng lúc`, không âm), `over_limit_frames`.
- Percentile: nearest-rank trên danh sách đã sắp; danh sách rỗng ⇒ `0.0`.

- [ ] **Step 1: Write the failing tests**

```python
def test_percentiles_and_fps(): ...  # 10 frames total_ms 10..100 → p50 50, p95 100; wall 2 s → ai_fps 5.0
def test_face_hz_counts_fresh_and_reused_separately(): ...
def test_id_switches_counts_seats_beyond_concurrent_max(): ...
def test_empty_video_summary_has_zeros_not_errors():
    s = TimingRecorder().summary(wall_s=0.0)
    assert s["frames"] == 0 and s["ai_fps"] == 0.0 and s["latency_ms"] == {"p50": 0.0, "p95": 0.0}
```

- [ ] **Step 2–4:** `python -m pytest packages/proctoring_ai/tests/multi/test_timing.py -v` (FAIL → implement → PASS)
- [ ] **Step 5: Commit** `git commit -m "feat(ai): record multi-subject timing summary"`

---

### Task 11: Vòng chạy `run_multi_replay`, đầu ra và CLI

**Files:**
- Create: `packages/proctoring_runtime/src/proctoring_runtime/multi_replay.py`
- Create: `scripts/analyze_video_multi.py`
- Test: `tests/test_multi_replay.py` (đã tạo ở Task 2)

**Interfaces:**
- Consumes: `create_replay_processor`, `relax_timeouts`-style timeouts, `SessionRuleProcessor.process(..., extra_signals=)` + `last_signals` (Task 2), `NeighborLookEvaluator`/`SubjectHead` (Task 1), `MultiSubjectPipeline` (Task 9), `TimingRecorder` (Task 10), `FOCUS_EVENT_TYPES` (Task 7), `VideoFileFrameSource`.
- Produces:
  - `@dataclass(slots=True) class MultiReplayResult: events: list[dict[str, Any]]; timing: dict[str, Any]; frames_analyzed: int`
  - `async run_multi_replay(source: FrameSource, pipeline: MultiSubjectPipeline, *, policy: Mapping[str, Any], raw_config: Mapping[str, Any], out_dir: Path, mode: Literal["full", "realtime"] = "full", overlap: bool = True, write_video: bool = True, session_id: UUID | None = None, clock: Callable[[], float] = time.perf_counter) -> MultiReplayResult`
  - `class AnnotatedVideoWriter(path: Path, fps: float)` — `write(image, result: MultiFrameResult, open_labels: Mapping[UUID, list[str]]) -> None`, `close()`; vẽ bằng cv2: box người + `#runtime_track_id`, box mặt, `yaw/pitch`, nhãn sự kiện đang mở (`label_vi`), ghế over-limit màu xám.
  - CLI `scripts/analyze_video_multi.py VIDEO --out DIR [--mode full|realtime] [--fps 10] [--yolo-size 416|640] [--yolo-every 1|2] [--crop-size 256] [--max-subjects 6] [--no-motion-skip] [--no-focus] [--no-overlap] [--no-video] [--overwrite] [--config configs/default.yaml] [--policy configs/exam_policy.yaml] [--manifest models/manifest.yaml]`. Thoát mã 2 nếu `--out` đã có file mà không có `--overwrite`; mã 1 nếu không mở được video.

Hành vi:
- Một `SessionRuleProcessor` cho mỗi ghế (tạo khi ghế xuất hiện, cùng `session_id`), `NeighborLookEvaluator(yaw_degrees=config behavior.head_turn.yaw_degrees)` dùng chung.
- Mỗi frame: `SubjectHead` cho mọi ghế có bundle và có mặt (`calibrated = calibration_status == "calibrated"`) → tín hiệu neighbor chia theo `track_id` → `processor.process(bundle, frame, extra_signals=...)` cho từng ghế có bundle. Mọi item thêm `runtime_track_id`, `video_offset_ms`, `face_reused`. Ghế có tín hiệu trong `last_signals` với `event_type in FOCUS_EVENT_TYPES` và state CANDIDATE/ACTIVE ⇒ `pipeline.mark_focus`.
- Ghế `expired` ⇒ `neighbor.forget(...)` đưa qua `extra_signals` của một bundle "không người", rồi `processor.finish(...)` và bỏ processor. Cuối video: `finish` cho mọi processor còn lại.
- Item `opened` ⇒ ghi JPEG khung hiện tại vào `out_dir/evidence/<event id>.jpg` (cv2.imwrite, đồng bộ là đủ cho offline), đặt `evidence_status` = `"saved"`/`"failed"`.
- `events.jsonl` (UTF-8, mỗi dòng một item), `timing.json` (`TimingRecorder.summary` + `config` gồm các cờ CLI + `hardware` như `scripts/evaluate.py::_hardware`), `annotated.mp4` (fourcc `mp4v`, fps = tốc độ phân tích).
- `mode="full"`: xử lý mọi frame; `overlap=True` ⇒ bắt đầu `pipeline.detect(frame t+1)` trước khi chờ `analyze_subjects(frame t)`.
- `mode="realtime"`: source mở với `analysis_rate_hz = fps video`; frame có `video_offset_ms` nhỏ hơn thời gian đồng hồ đã trôi (từ `clock`) trừ một chu kỳ frame bị bỏ (`timing.add_dropped()`); `latency_ms` = thời gian xử lý frame + độ trễ so với đồng hồ.

- [ ] **Step 1: Write the failing tests** (fake pipeline trả `MultiFrameResult` định sẵn; source giả trả frame numpy; `clock` giả)

```python
async def test_events_carry_runtime_track_id_and_are_written(tmp_path): ...
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert {json.loads(l)["runtime_track_id"] for l in lines} == {1, 2}

async def test_neighbor_look_event_opens_for_turning_seat(tmp_path): ...  # seat 1 yaw 30, calibrated, seat 2 to the right, 2 s
async def test_expired_seat_closes_its_open_events(tmp_path):
    # seat 1 visible 1 s, then absent: PERSON_MISSING opens after behavior.person_missing duration,
    # seat expires at +10 s → a "closed" PERSON_MISSING item for runtime_track_id 1 with
    # video_offset_ms < last frame offset (video continues 3 s with seat 2 only)
async def test_short_video_never_calibrates_and_reports_status(tmp_path):
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert timing["seats"]["1"]["calibration_status"] != "calibrated"
    assert not any(e["type"].startswith("HEAD_TURN") for e in result.events)
async def test_realtime_mode_drops_frames_when_slow(tmp_path): ...  # fake clock advances 300 ms per frame at 30 fps → dropped_frames > 0
async def test_candidate_signal_marks_focus(tmp_path): ...
def test_cli_refuses_existing_out_dir_without_overwrite(tmp_path): ...  # subprocess or main(argv) → 2
```

- [ ] **Step 2: Run** `python -m pytest tests/test_multi_replay.py -v` — Expected: FAIL
- [ ] **Step 3: Implement `multi_replay.py` and the CLI** (CLI only parses args, builds `AnalyzerFactory(..., detector_image_size=args.yolo_size, detector_device=env PROCTORING_AI_DEVICE like run_gpu.ps1)`, awaits `initialize()`, calls `build_multi_pipeline`, `run_multi_replay`, prints `ai_fps`, `latency p95`, `seats`.)
- [ ] **Step 4: Run** `python -m pytest tests/test_multi_replay.py packages/proctoring_ai/tests packages/proctoring_core/tests -v` — Expected: PASS
- [ ] **Step 5: Lint/type check** `ruff check packages scripts tests && mypy packages/proctoring_ai/src/proctoring_ai/multi packages/proctoring_runtime/src/proctoring_runtime/multi_replay.py` — Expected: no errors
- [ ] **Step 6: Commit** `git commit -m "feat: offline multi-subject video analysis script"`

---

### Task 12: Đo trên video thật và chốt cấu hình (máy Windows của người dùng)

Task này chạy trên máy có GPU (Remote Control hoặc người dùng tự chạy), không chạy được trong cloud.

**Files:**
- Modify: `docs/evaluation-results.md` (thêm mục "Phase 2 — nhiều thí sinh")
- Create: `tests/video_samples/multi/labels.example.csv` (nhãn tay mẫu, không chứa video)
- Possibly modify defaults: `MotionGate.threshold`, `SubjectCropper.min_side`, CLI `--yolo-size` default

- [ ] **Step 1: Chọn video** — một đoạn 1–3 phút, 4–6 người thấy rõ mặt, ≥ 720p, từ SAV (link gốc trong repo SAV) hoặc video stock miễn phí. Lưu ngoài Git (`storage/videos/`), ghi nguồn + license vào `docs/evaluation-results.md`.
- [ ] **Step 2: Số liệu nền** — `--mode realtime --no-motion-skip --yolo-every 1 --no-overlap --crop-size 256 --yolo-size 416`. Ghi `ai_fps`, p95, `face_each` p50/p95.
- [ ] **Step 3: Bật từng cách tăng tốc** — overlap; motion skip; `--yolo-every 2`; `--crop-size 192` (so yaw trung bình với 256 trên cùng video bằng `events`/log, chấp nhận nếu lệch ≤ 3°); `--yolo-size 640` (so số người phát hiện). Mỗi lần ghi một dòng vào bảng.
- [ ] **Step 4: Độ đúng** — chạy `--mode full`, đếm `id_switches` (mục tiêu 0 với người ngồi yên), gắn 10–20 khoảng hành vi vào `labels.example.csv` theo định dạng `proctoring_core.evaluation.labels`, so sánh với `events.jsonl` (đọc span bằng `ReplayResult.predicted_events`-style pairing theo `id`).
- [ ] **Step 5: Chốt mặc định** — đặt mặc định CLI/hằng số theo cấu hình tốt nhất; chạy lại `python -m pytest packages/proctoring_ai/tests/multi tests/test_multi_replay.py -v`.
- [ ] **Step 6: Quyết định 2b** — nếu cấu hình tốt nhất vẫn < 8 FPS (realtime, 6 người) hoặc mặt < 4 lần/giây/ghế, ghi rõ trong `docs/evaluation-results.md` và mở spec bước 2b; nếu đạt, ghi "không cần 2b".
- [ ] **Step 7: Commit** `git commit -m "docs: record phase 2 multi-subject measurements"`
