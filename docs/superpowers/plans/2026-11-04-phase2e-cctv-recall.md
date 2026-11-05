# Phase 2e: Bắt đủ sự kiện CCTV (hai tầng) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tầng nhanh khoanh đoạn nghi vấn không bỏ sót; tầng kỹ dùng model train sẵn lớn hơn trên ảnh cắt để xác nhận (a) điện thoại, (b) chuyền đồ, (c) quay sang, (d) tài liệu giấu.

**Architecture:** `analyze_video_multi.py --scene cctv --suspects` chạy như cũ và thêm `suspects.jsonl` từ `SuspectTracker`. `scripts/review_suspects.py` đọc lại video gốc trong từng đoạn, cắt và phóng to vùng ghế, chạy `ReviewDetectors` (pose + nhiều detector vật thể train sẵn, cắm thêm qua YAML), rồi `review_rules` sinh sự kiện cuối.

**Tech Stack:** Python 3.12, Ultralytics 8.4.160 (YOLO11, YOLOE), OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-phase2e-cctv-recall-design.md`

## Global Constraints

- Không train model; chỉ dùng weights train sẵn tải theo `models/manifest.yaml` có SHA-256.
- Tầng nhanh trên M1200 ở `--fps 5` không chậm hơn hiện tại quá 10 % (~10 FPS); điều kiện mới chỉ dùng keypoint đã có, không thêm model.
- Tầng kỹ: 10 khung/giây trong đoạn, vùng cắt nới 30 %, cạnh dài phóng lên 960 px.
- Đoạn nghi vấn nới 2 s mỗi phía; điều kiện tầng nhanh giữ ≥ 0,5 s; đoạn cùng ghế chồng nhau thì gộp.
- Tọa độ trong mọi file JSONL là chuẩn hóa theo khung hình gốc (0..1), như hiện tại.
- Chữ hiển thị cho người dùng (nhãn sự kiện, thông báo CLI) bằng tiếng Việt.
- Máy Windows không được phải tải gì lúc chạy: embedding chữ của YOLOE tính sẵn và lưu trong file model.
- Tham số mới nằm trong dataclass cấu hình (`CctvSuspectConfig`, `ReviewRuleConfig`), không viết cứng trong luật.

## Review Focus

- Đoạn nghi vấn ở sát đầu/cuối video (nới 2 s ra ngoài khoảng [0, độ dài]): phải kẹp lại, không lỗi khi đọc video. Test trong Task 3 và Task 6.
- Ghế ở mép khung hình: vùng cắt nới 30 % vượt ra ngoài ảnh phải kẹp, và đổi tọa độ về khung gốc vẫn đúng. Test trong Task 4.
- Model phụ khai báo trong `configs/review_models.yaml` mà chưa tải hoặc sai SHA: bỏ qua model đó, ghi lý do vào `timing.json`, không dừng cả lượt. Test trong Task 4.
- `suspects.jsonl` rỗng (clip không có gì): `review_suspects.py` vẫn ghi `events.jsonl` rỗng và `timing.json`, trả mã 0. Test trong Task 6.
- Nhãn cũ không có `behavior` (clip 1 trước phase 2e): `eval_cctv.py` coi là `look`, không lỗi. Test trong Task 2.

---

### Task 1: Tải model train sẵn và thử YOLOE trên ảnh cắt CCTV

**Files:**
- Modify: `models/manifest.yaml`
- Create: `scripts/export_yoloe_prompts.py`
- Create: `docs/superpowers/notes/2026-11-04-yoloe-spike.md`

**Interfaces:**
- Produces: manifest entries `yolo11x-pose` (task `pose_review`), `yolo11x` (task `detect_review`), `yoloe-11l-seg-exam` (task `open_vocab_review`, file `pretrained/yoloe-11l-seg-exam.pt`, lớp `["phone", "paper", "notebook", "book"]`).

- [ ] **Step 1:** Thêm `yolo11x-pose` và `yolo11x` vào manifest (URL `https://github.com/ultralytics/assets/releases/download/v8.3.0/<tên>.pt`, license AGPL-3.0-only, `class_map` như các mục YOLO hiện có, SHA-256 tính từ file đã tải).
- [ ] **Step 2:** Viết `scripts/export_yoloe_prompts.py`: tải `yoloe-11l-seg.pt`, `model.set_classes(names, model.get_text_pe(names))` với `names = ["phone", "paper", "notebook", "book"]`, `model.save("models/pretrained/yoloe-11l-seg-exam.pt")`. Chạy, rồi thêm mục manifest với SHA của file đã lưu (không có `download_url`; `download_model.py` in hướng dẫn chạy script này khi thiếu).
- [ ] **Step 3:** Cắt 10 ảnh từ `/mnt/project-files/EPf9O7UXRoM.mp4` (0:53 chị giơ giấy ở vùng `[0.64, 0.15, 0.78, 0.33]`, 0:10, 0:30 các bàn hàng đầu), phóng lên 960 px, chạy `yolo11x` (conf 0.15) và YOLOE (conf 0.1). Ghi số lần thấy `paper`/`book`/`phone` và ảnh vẽ khung vào ghi chú spike.
- [ ] **Step 4:** Ghi quyết định vào ghi chú: YOLOE dùng hay bỏ (tiêu chí: thấy tờ giấy giơ lên ở 0:53 trong ≥ 1 ảnh và ≤ 2 khung sai trên 10 ảnh). Nếu bỏ, Task 4 tắt mục YOLOE trong `configs/review_models.yaml`.
- [ ] **Step 5:** `uv run --no-sync python scripts/download_model.py` chạy hết, SHA khớp. Commit.

### Task 2: Nhãn theo hành vi và độ phủ tầng nhanh trong `eval_cctv.py`

**Files:**
- Modify: `scripts/eval_cctv.py`
- Modify: `data/labels/EPf9O7UXRoM.yaml` (thêm `behavior: look` cho mọi nhãn `exchange`)
- Test: `tests/test_eval_cctv.py`

**Interfaces:**
- Produces: `score(events, labels) -> dict` thêm khóa `"by_behavior": {behavior: {"found": int, "total": int}}`; `coverage(windows: list[dict], labels: list[dict], duration_s: float) -> dict` trả `{"covered": [id], "uncovered": [id], "suspect_ratio": float}`; CLI `--suspects <suspects.jsonl>` và `--duration <giây>`.

- [ ] **Step 1: Test**

```python
def test_by_behavior_counts_and_default_look():
    labels = [{"id": "a", "kind": "exchange", "start_s": 0, "end_s": 5, "region": [0, 0, 1, 1]},
              {"id": "p", "kind": "exchange", "behavior": "phone", "start_s": 20, "end_s": 25, "region": [0, 0, 1, 1]}]
    events = [_opened(2.0, head=[0.5, 0.5])]
    result = score(events, labels)
    assert result["by_behavior"] == {"look": {"found": 1, "total": 1}, "phone": {"found": 0, "total": 1}}

def test_coverage_needs_overlap_in_time_and_region():
    labels = [{"id": "a", "kind": "exchange", "start_s": 10, "end_s": 12, "region": [0.1, 0.1, 0.3, 0.3]},
              {"id": "b", "kind": "exchange", "start_s": 40, "end_s": 42, "region": [0.6, 0.6, 0.8, 0.8]}]
    windows = [{"start_ms": 8000, "end_ms": 13000, "bbox": {"x": 0.15, "y": 0.15, "width": 0.1, "height": 0.1}},
               {"start_ms": 40000, "end_ms": 41000, "bbox": {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1}}]
    result = coverage(windows, labels, duration_s=60.0)
    assert result["covered"] == ["a"] and result["uncovered"] == ["b"]
    assert result["suspect_ratio"] == pytest.approx(6.0 / 60.0)
```

Vùng ghế khớp nhãn khi tâm `bbox` nằm trong `region` (dùng lại `_inside`, margin 0,02). `suspect_ratio` = tổng thời lượng hợp các đoạn / `duration_s` (đoạn chồng nhau chỉ tính một lần).

- [ ] **Step 2:** `uv run --no-sync pytest tests/test_eval_cctv.py -v` → FAIL.
- [ ] **Step 3:** Cài đặt; CLI in thêm một dòng mỗi hành vi `phone: 0/1` và, khi có `--suspects`, dòng `Tầng nhanh khoanh trúng 1/2 nhãn, đoạn nghi vấn chiếm 10 % clip`.
- [ ] **Step 4:** Test PASS, cả `tests/test_eval_cctv.py` cũ. Commit.

### Task 3: `SuspectTracker` trong tầng nhanh

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/suspects.py`
- Modify: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/rules.py` (gọi tracker trong `step`, trả qua thuộc tính `suspects`)
- Test: `packages/proctoring_ai/tests/multi/test_cctv_suspects.py` (dùng `cctv_kps.person` và các helper như `test_cctv_rules.py`)

**Interfaces:**
- Consumes: `head_yaw_index`, `head_lift`, `head_point`, `point`, `RollingBaseline` (`head.py`); `Seat`, `MultiFrameResult`.
- Produces:
  - `@dataclass(frozen=True) CctvSuspectConfig(yaw_factor=0.6, lean_widths=0.25, hand_out_widths=0.15, hands_meet_widths=0.5, object_min_conf=0.15, hold_ms=500, pad_ms=2000, desk_default=0.6, baseline_min_span_ms=1000)`
  - `@dataclass(frozen=True) SuspectWindow(seat_id: UUID, runtime_track_id: int | None, neighbor_ids: tuple[UUID, ...], start_ms: int, end_ms: int, reasons: tuple[str, ...], bbox: Box)`; `to_json() -> dict` với `start_ms`/`end_ms` là offset video.
  - `class DeskLine`: `add(wrist_y: float, ms: int)`, `value(box: Box, ms: int) -> float` (trung vị cổ tay khi đang viết, mặc định `box.y + desk_default * box.height`).
  - `class SuspectTracker(config: CctvSuspectConfig | None = None)`: `observe(candidates: Mapping[UUID, Seat], poses: Mapping[UUID, Keypoints], objects: Sequence[Mapping], offset_ms: int, ms: int, frame_aspect: float, rule_event_seats: Iterable[UUID]) -> None`; `finish(duration_ms: int | None) -> list[SuspectWindow]`.
  - `CctvSeatRules(..., suspects: SuspectTracker | None = None)`; `step` gọi `observe` với ghế có sự kiện vừa mở (`RULE_EVENT`).

Mã lý do đúng như spec: `HEAD_AWAY`, `LEAN`, `HAND_OUT`, `HANDS_MEET`, `HEAD_DOWN_HANDS_LOW`, `OBJECT_HINT`, `RULE_EVENT`. `HEAD_AWAY` dùng ngưỡng `yaw_factor * max(CctvRuleConfig.yaw_threshold, yaw_spread_factor * spread)` và không đòi ngẩng đầu. `HANDS_MEET` xét mọi cặp ghế mà đầu cách nhau ≤ 2,5 bề rộng khung (cùng hàng hoặc trước/sau). `RULE_EVENT` không cần giữ 0,5 s.

- [ ] **Step 1: Tests** (mỗi test phát 3 s keypoint tổng hợp ở 10 Hz, gọi `finish`):
  - `test_head_away_without_lift_opens_window`: đầu cúi (lift < 0,1) quay lệch 0,3 → một đoạn lý do `HEAD_AWAY` (luật cũ không mở sự kiện).
  - `test_condition_shorter_than_hold_is_ignored`: điều kiện đúng 0,3 s → không có đoạn.
  - `test_hands_meet_across_rows`: cổ tay ghế hàng trước và hàng sau cách 0,03 → đoạn cho cả hai ghế, `neighbor_ids` chứa nhau.
  - `test_head_down_hands_low_uses_desk_line`: sau 5 s viết (cổ tay ở y 0,45), cổ tay xuống 0,55 và cúi đầu 1 s → `HEAD_DOWN_HANDS_LOW`.
  - `test_object_hint_low_confidence_phone_in_seat`: `cell phone` conf 0,2 tâm trong khung → `OBJECT_HINT`.
  - `test_windows_padded_clamped_and_merged`: hai đợt cách nhau 3 s ở cùng ghế lúc 0,5 s và 4 s → một đoạn, `start_ms == 0` (kẹp), `end_ms` ≤ `duration_ms`.
  - `test_rules_without_tracker_unchanged`: `CctvSeatRules()` không truyền `suspects` cho đúng sự kiện như trước trên kịch bản quay đầu có sẵn trong `test_cctv_rules.py`.
- [ ] **Step 2:** Chạy `uv run --no-sync pytest packages/proctoring_ai/tests/multi/test_cctv_suspects.py -v` → FAIL.
- [ ] **Step 3:** Cài đặt. Gộp đoạn: cùng ghế và `[start - pad, end + pad]` chồng nhau; `reasons` và `neighbor_ids` hợp lại, `bbox` lấy hộp bao các khung ghế.
- [ ] **Step 4:** Test mới và toàn bộ `packages/proctoring_ai/tests/multi` PASS. Commit.

### Task 4: Ảnh cắt và bộ detector tầng kỹ cắm thêm được

**Files:**
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/crops.py`
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_detectors.py`
- Create: `configs/review_models.yaml`
- Test: `packages/proctoring_ai/tests/multi/test_cctv_crops.py`, `packages/proctoring_ai/tests/multi/test_cctv_review_detectors.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) CropTransform(x0: int, y0: int, scale: float, frame_size: tuple[int, int])`; `to_frame(x_px: float, y_px: float) -> tuple[float, float]` (chuẩn hóa theo khung gốc).
  - `crop_region(boxes: Sequence[Box], margin: float = 0.3) -> Box` (kẹp trong [0,1]).
  - `crop_image(image: ndarray, region: Box, long_side: int = 960) -> tuple[ndarray, CropTransform]`.
  - `@dataclass(frozen=True) ReviewObject(label: str, confidence: float, bbox: Box, source: str)`; `label` ∈ `{"phone", "paper", "book"}`.
  - `@dataclass(frozen=True) ModelSpec(name: str, kind: str, path: Path, class_map: dict[str, str], min_conf: float, sha256: str | None)`; `kind` ∈ `{"pose", "objects"}`.
  - `load_specs(config_path: Path, models_root: Path) -> list[ModelSpec]`.
  - `class ReviewDetectors(specs: list[ModelSpec], device: str | None = None)`: `pose(crop, transform) -> list[dict]` (cùng dạng `parse_pose_result`, tọa độ khung gốc); `objects(crop, transform) -> list[ReviewObject]`; `status: dict[str, str]` (`"ok"` hoặc lý do bỏ qua).

`configs/review_models.yaml` khai báo: `yolo11x-pose` (pose), `yolo11x` (objects, `cell phone: phone`, `book: book`, conf 0.35), `yoloe-11l-seg-exam` (objects, `phone: phone`, `paper: paper`, `notebook: paper`, `book: book`, conf 0.2), và một mục mẫu bị chú thích (comment) cho detector gian lận train sẵn tải từ GitHub/HuggingFace, chỉ cần điền `path`, `sha256`, `class_map`. Lớp không có trong `class_map` bị bỏ. Hai model cùng thấy một vật (IoU > 0,5, cùng `label`) thì giữ cái tin cậy cao hơn.

- [ ] **Step 1: Tests** (model giả, không cần weights):
  - `test_crop_region_clamped_at_frame_edge`: hộp ở `x=0.95` → vùng cắt nằm trong [0,1].
  - `test_crop_roundtrip_to_frame`: ảnh 1280×720, vùng `Box(0.5,0.5,0.1,0.1)` → ảnh cắt cạnh dài 960; điểm tâm ảnh cắt về lại ≈ `(0.55, 0.55)` (sai số 1e-3, tính cả margin).
  - `test_missing_or_bad_sha_model_skipped_with_reason`: một spec có file không tồn tại, một spec sai SHA → `status` ghi lý do tiếng Việt cho cả hai, model còn lại vẫn chạy.
  - `test_class_map_and_cross_model_dedup`: hai model giả trả `cell phone` và `phone` trùng chỗ → một `ReviewObject("phone", ...)` với confidence lớn hơn; lớp `cup` bị bỏ.
- [ ] **Step 2:** Test FAIL.
- [ ] **Step 3:** Cài đặt; nạp model bằng `ultralytics.YOLO` (YOLOE nạp được qua `YOLO` khi file đã có lớp chữ), kiểm tra SHA như `download_model.py`.
- [ ] **Step 4:** Test PASS. Commit.

### Task 5: Luật tầng kỹ và hai loại sự kiện mới

**Files:**
- Modify: `packages/proctoring_core/src/proctoring_core/types.py` (thêm `PASSING_OBJECT`, `LOOK_AT_LAP`, `UNCLASSIFIED_SUSPECT`)
- Modify: `packages/proctoring_runtime/src/proctoring_runtime/messages.py` (nhãn tiếng Việt: "Chuyền đồ vật với thí sinh bên cạnh", "Cúi nhìn xuống đùi lâu", "Đoạn nghi vấn chưa phân loại")
- Modify: `packages/proctoring_core/src/proctoring_core/risk/engine.py` (điểm: `PASSING_OBJECT` 10.0, `LOOK_AT_LAP` 6.0, `UNCLASSIFIED_SUSPECT` 0.0)
- Create: `packages/proctoring_ai/src/proctoring_ai/multi/cctv/review_rules.py`
- Test: `packages/proctoring_ai/tests/multi/test_cctv_review_rules.py`

**Interfaces:**
- Consumes: `SuspectWindow` (Task 3), `ReviewObject` (Task 4), `CctvSeatRules` (luật quay đầu hiện có), `head.py`, `DeskLine`.
- Produces:
  - `@dataclass(frozen=True) ReviewRuleConfig(phone_min_conf=0.35, phone_wrist_widths=0.3, phone_frames=3, passing_wrist_widths=0.3, passing_hold_ms=300, passing_transfer_ms=2000, document_hold_ms=1000, lap_hold_ms=3000)`.
  - `@dataclass(frozen=True) ReviewFrame(offset_ms: int, seats: Mapping[UUID, Box], poses: Mapping[UUID, Keypoints], objects: Sequence[ReviewObject])` (ghế lấy từ `SuspectWindow`, pose gán vào ghế theo tâm đầu gần tâm trên của khung ghế nhất).
  - `review_window(window: SuspectWindow, frames: Sequence[ReviewFrame], config: ReviewRuleConfig | None = None) -> list[dict]`: trả bản ghi `opened`/`closed` cùng dạng `CctvSeatRules._record` (có `type`, `seat_id`, `video_offset_ms`, `head`, `scene: "cctv"`, `stage: "review"`, `reasons` của đoạn), hoặc đúng một bản ghi `{"action": "dismissed", "seat_id", "start_ms", "end_ms", "reasons"}` khi không luật nào đạt.

Luật đúng như bảng mục 3.2 của spec. Quay đầu: phát lại `frames` qua một `CctvSeatRules` mới (seat map tạm từ `window`) với keypoint `yolo11x-pose`; mốc hướng đầu lấy từ chính đoạn (đã nới 2 s). `UNCLASSIFIED_SUSPECT` khi không luật nào đạt nhưng `len(set(window.reasons) - {"RULE_EVENT"}) >= 2`.

- [ ] **Step 1: Tests:**
  - `test_phone_near_wrist_three_frames_opens`; `test_phone_two_frames_only_dismissed`.
  - `test_passing_by_wrists_reports_both_seats`: cổ tay hai ghế cách 0,02 trong 0,4 s → hai `PASSING_OBJECT`, mỗi cái `neighbor_seat_id` là ghế kia.
  - `test_passing_by_object_transfer`: `paper` trong khung ghế A ở 1,0 s, trong khung B ở 2,5 s → `PASSING_OBJECT`.
  - `test_document_below_desk_line_opens`; `test_exam_paper_on_desk_in_front_not_document`: giấy ngay trước mặt, trên mép bàn → không có `DOCUMENT_DETECTED`.
  - `test_look_at_lap_needs_three_seconds`: 2,5 s → không; 3,2 s → có.
  - `test_head_turn_reuses_existing_rules`: kịch bản quay sang bên trong đoạn → `LOOK_AT_NEIGHBOR`.
  - `test_unclassified_needs_two_reasons`; `test_nothing_matched_gives_dismissed`.
- [ ] **Step 2:** Test FAIL. **Step 3:** Cài đặt. **Step 4:** Test mới + `uv run --no-sync pytest packages -q` PASS. Commit.

### Task 6: Nối hai tầng vào script

**Files:**
- Modify: `scripts/analyze_video_multi.py` (cờ `--suspects`: tạo `SuspectTracker`, ghi `suspects.jsonl` sau khi chạy, in "Đoạn nghi vấn: N, chiếm X % clip")
- Create: `scripts/review_suspects.py`
- Test: `tests/test_review_suspects.py`

**Interfaces:**
- Consumes: Task 3–5.
- Produces: `review_suspects.py <video> <suspects.jsonl> --out <dir> [--models configs/review_models.yaml] [--fps 10] [--device 0]`; ghi `events.jsonl` (chỉ sự kiện tầng kỹ + `dismissed`), `evidence/<event id>.jpg` (khung giữa sự kiện, vẽ khung ghế, keypoint, vật thể), `timing.json` (`windows`, `seconds_per_window` p50/max, `models` = `ReviewDetectors.status`). Hàm `review(video: Path, windows: list[SuspectWindow], detectors, out_dir: Path, fps: float = 10.0) -> dict` để test gọi trực tiếp.

Đọc khung bằng `cv2.VideoCapture` + `CAP_PROP_POS_MSEC` từ `max(0, start_ms)` đến `min(end_ms, độ dài)`; mỗi khung: `crop_region` các khung ghế trong đoạn, `crop_image`, `detectors.pose`/`objects`, dựng `ReviewFrame`.

- [ ] **Step 1: Tests** (video tổng hợp 3 s, 640×360 viết bằng `cv2.VideoWriter`; detector giả):
  - `test_empty_suspects_writes_empty_outputs`: `events.jsonl` rỗng, `timing.json` có `windows: 0`.
  - `test_window_past_video_end_is_clamped`: đoạn 2–6 s trên video 3 s → không lỗi, chỉ đọc đến 3 s.
  - `test_phone_window_produces_event_and_evidence`: detector giả trả điện thoại gần cổ tay mọi khung → một `PHONE_DETECTED` và file ảnh bằng chứng tồn tại.
- [ ] **Step 2:** FAIL. **Step 3:** Cài đặt. **Step 4:** PASS. Commit.

### Task 7: Chạy đầu-cuối trên clip 1, ghi kết quả

**Files:**
- Modify: `docs/evaluation-results.md` (mục "Phase 2e")
- Modify: `README.md` (hai lệnh chạy hai tầng, một đoạn ngắn)

- [ ] **Step 1:** `uv run --no-sync python scripts/analyze_video_multi.py /mnt/project-files/EPf9O7UXRoM.mp4 --scene cctv --mode full --fps 5 --suspects --out storage/multi/p2e-fast --overwrite`.
- [ ] **Step 2:** `uv run --no-sync python scripts/eval_cctv.py storage/multi/p2e-fast/events.jsonl data/labels/EPf9O7UXRoM.yaml --suspects storage/multi/p2e-fast/suspects.jsonl --duration 71.7` → kỳ vọng tầng nhanh vẫn 5/6 sự kiện như phase 2d, độ phủ ≥ 95 % nhãn có thể bắt (nhãn 0:02 được tính), đoạn nghi vấn ≤ 30 % clip. Nếu > 30 %: chỉnh `CctvSuspectConfig`, ghi lý do.
- [ ] **Step 3:** `uv run --no-sync python scripts/review_suspects.py /mnt/project-files/EPf9O7UXRoM.mp4 storage/multi/p2e-fast/suspects.jsonl --out storage/multi/p2e-review`, rồi chấm `eval_cctv.py storage/multi/p2e-review/events.jsonl ...` → không tệ hơn 5/6, ghi số báo sai/phút và thời gian mỗi đoạn.
- [ ] **Step 4:** Ghi bảng kết quả vào `docs/evaluation-results.md`; chép `suspects.jsonl`, `events.jsonl`, `timing.json` và 3–4 ảnh bằng chứng vào `/mnt/project-files/phase2e/clip1/`. Commit, push.
