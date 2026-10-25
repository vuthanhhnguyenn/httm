---
description: "Danh sách công việc: Hoàn thiện Phase 1"
---

# Tasks: Hoàn thiện Phase 1 — độ chính xác, hướng mắt, nói chuyện và chuẩn hóa cấu trúc

**Input**: `specs/002-phase1-hardening/` (plan.md, spec.md, research.md, data-model.md, contracts/)

**Tests**: Bắt buộc theo hiến pháp (nguyên tắc VI, quy trình mục 2): mỗi luật có kiểm thử tất định;
mỗi story có replay so với mốc.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: chạy song song được (khác tệp, không phụ thuộc task chưa xong)
- **[Story]**: US1..US7 theo spec.md

---

## Phase 1: Setup

- [x] T001 Dọn 25 lỗi ruff và 37 lỗi mypy hiện có để CI xanh trước khi đổi gì (`apps/`, `packages/`, `scripts/`)
- [x] T002 Tạo gói `packages/proctoring_runtime/src/proctoring_runtime/__init__.py` + `tests/`; thêm vào `mypy_path` và `[tool.ruff] src` trong `pyproject.toml`, `tests/conftest.py`, `apps/desktop/main.py`, lệnh mypy trong `.github/workflows/ci.yml`
- [x] T003 [P] Thêm `tests/video_samples/oep_set.example.yaml` và `tests/video_samples/oep_mapping.example.yaml` theo `contracts/evaluation.md`; bổ sung `.gitignore` cho `storage/evaluation/`

---

## Phase 2: Foundational — đo được và có mốc (chặn mọi story)

**Mục tiêu**: phát lại tất định + chấm điểm + golden trên mã HIỆN TẠI, trước khi tái cấu trúc.

- [x] T004 [P] `VideoFileFrameSource` (timestamp = frame_index·1000/fps, UTC = mốc + offset, chọn frame theo `analysis_rate_hz`) trong `packages/proctoring_ai/src/proctoring_ai/capture/video_file_source.py` + test `packages/proctoring_ai/tests/capture/test_video_file_source.py`
- [x] T005 [P] Đọc nhãn `oep`/`csv` + ánh xạ, kiểm tra lỗi có tệp/dòng trong `packages/proctoring_core/src/proctoring_core/evaluation/labels.py` + test `packages/proctoring_core/tests/evaluation/test_labels.py`
- [x] T006 [P] Chấm điểm mức sự kiện (tp/fn/fp/duplicates/coverage/false_alerts_per_min/median_delay, nhóm ANY, ignore) trong `packages/proctoring_core/src/proctoring_core/evaluation/matching.py` + test `packages/proctoring_core/tests/evaluation/test_matching.py`
- [x] T007 Báo cáo `report.json`/`report.md` + `baseline_delta` trong `packages/proctoring_core/src/proctoring_core/evaluation/report.py` + test
- [x] T008 Bộ phát lại: chạy `VideoFileFrameSource` → pipeline → luật hiện có (gọi đúng đường `DesktopRuntime._process_rules` thông qua một lớp điều phối headless tạm), xuất `PredictedEvent` và golden JSONL trong `packages/proctoring_runtime/src/proctoring_runtime/replay.py`
- [x] T009 CLI `scripts/evaluate.py` theo `contracts/evaluation.md` (mã thoát, `--case`, `--mode`, `--rate`, `--baseline`, `--save-events`); xóa stub `scripts/replay.py`
- [x] T010 [P] Kịch bản golden tổng hợp (fake analyzer theo thời gian: điện thoại có gián đoạn, rời ghế, quay đầu, che camera, hai người) trong `tests/golden/scenarios.py` + ghi `tests/golden/*.jsonl` + test so khớp `tests/golden/test_golden_replay.py`
- [x] T011 Xác minh định dạng OEP và nghĩa mã nhãn (xem frame tại các khoảng `gt.txt` của vài subject) → cập nhật `oep_mapping.example.yaml`, `research.md` R1, `quickstart.md` §2
- [x] T012 Chạy mốc OEP trên mã hiện tại (tập con nhanh + toàn bộ khi tải xong), lưu `storage/evaluation/baseline/`, tóm tắt vào `docs/evaluation-results.md` — đã chạy đủ 24 ca (mã trước thay đổi, `storage/evaluation/baseline/20261016-054444`), chấm lại theo luật matching mới ở `baseline_rescored/`

**Checkpoint**: Có golden tổng hợp (CI) và mốc OEP. Từ đây mọi thay đổi đều đo được.

---

## Phase 3: Tái cấu trúc không đổi hành vi (FR-029..FR-031)

**Cổng**: sau MỖI task, `pytest` xanh và golden khớp 100%.

- [x] T013 [P] Quan sát có kiểu `packages/proctoring_core/src/proctoring_core/observations.py` + adapter từ/đến `ObservationBundle` (data-model §2) + test
- [x] T014 [P] `RuleSignal`/`SignalState` trong `packages/proctoring_core/src/proctoring_core/behavior/signals.py`; `EventAggregator.process` nhận `RuleSignal` (giữ chuyển đổi dict) + test
- [x] T015 Mở rộng `config/schema.py`: mọi ngưỡng đang viết cứng (quality 0.35, gap 600 ms, vùng gần mặt, ngưỡng tay 0.025, blur 20, rời chỗ 0.9, risk_cap 25, ngưỡng MediaPipe), giữ mặc định hiện tại; `load_config_files` dùng trong runtime thay `_read_yaml` + test `packages/proctoring_core/tests/config/`
- [x] T016 Evaluator nhận model cấu hình có kiểu, bỏ parse dict/khóa cũ trong `behavior/head.py`, `objects.py`, `presence.py`, `safety.py`, `supporting.py`; lớp nền `behavior/base.py` gom registry `TemporalCondition` và dựng `RuleSignal`
- [x] T017 Chuyển `_RuleSet`, `_inside_person`, `_hands_near_face` từ `apps/desktop/src/proctoring_desktop/runtime.py` thành `BehaviorEngine` (`behavior/engine.py`) và `behavior/geometry.py` + test đơn vị engine với fake Observation
- [x] T018 `SessionEngine` headless trong `packages/proctoring_runtime/src/proctoring_runtime/session_engine.py` (capture/process loop, risk, correlation, callback thay Qt Signal); `bootstrap.py` (cấu hình, thiết bị CUDA)
- [x] T019 [P] `persistence.py` (`JsonlEventRepository` implement port `EventRepository`, risk summary, ghi nguyên tử) và chuyển `evidence.py` sang `proctoring_runtime`; `payloads.py` cho UI/JSONL
- [x] T020 `apps/desktop/.../runtime.py` còn lớp `QThread` bọc `SessionEngine`; `replay.py` dùng `SessionEngine` thay lớp điều phối tạm của T008
- [x] T021 Tách `window.py` thành `widgets/preview.py`, `widgets/event_list.py`, `widgets/status_panel.py`; `window.py` chỉ lắp ráp + test khói khởi tạo cửa sổ (offscreen)
- [x] T022 Cập nhật/di chuyển kiểm thử hiện có theo đường dẫn mới; xác nhận golden + OEP tập con giống mốc

**Checkpoint**: Hành vi y hệt mốc, cấu trúc đúng tầng.

---

## Phase 4: User Story 2 — Kiểu bài thi (P1)

**Goal**: chọn Thi giấy / Thi máy; luật đổi qua cấu hình.
**Independent Test**: cùng đoạn cúi đầu → paper 0 cảnh báo, computer có cảnh báo khi đủ lâu.

- [x] T023 [P] [US2] `ExamMode` + `exam_modes` overrides + `resolve_behavior()` hợp nhất sâu và kiểm tra lại trong `config/schema.py`; `exam_mode_default` trong policy + test
- [x] T024 [US2] `BehaviorEngine`/`HeadRuleEvaluator`: `look_down.enabled`, thời lượng riêng thi máy; kiểu bài thi vào `policy_snapshot`, `session.json`, mỗi dòng sự kiện + test `packages/proctoring_core/tests/behavior/test_exam_modes.py`
- [x] T025 [US2] Giao diện: combobox "Kiểu bài thi" (nhớ bằng QSettings, khóa khi chạy) trong `widgets/status_panel.py`/`window.py`; `--mode` trong `scripts/evaluate.py`
- [ ] T026 [US2] Replay: clip paper/computer (khi có) + OEP ở chế độ computer; ghi kết quả SC-005

---

## Phase 5: User Story 3 — Tín hiệu chính chính xác, không trùng (P1)

**Goal**: sửa nối cấu hình, episode vật thể, thứ bậc, rời chỗ, che camera, hiệu chỉnh tự đề xuất.
**Independent Test**: replay OEP + golden: báo nhầm/phút giảm ≥50%, độ phủ không giảm >5 điểm.

- [x] T027 [P] [US3] Sửa nối thời lượng: `multiple_person`, `leaving_seat`, `camera_blocked` dùng khóa riêng; xóa/dùng `hand_activity.minimum_count`; schema từ chối khóa không dùng + test
- [x] T028 [P] [US3] `SignalArbiter` (`behavior/arbiter.py`) + `hierarchy` trong cấu hình (kiểm tra không chu trình); aggregator ghi `supporting_signals` vào sự kiện gốc + test `test_arbiter.py`
- [x] T029 [US3] Episode vật thể theo lớp (giữ qua mất ngắn và đổi track), `episode_id`, `object_trace` (frame, bbox, confidence, model, version, track) trong `behavior/objects.py`, `events/models.py` + test
- [x] T030 [P] [US3] Lớp tài liệu chỉ `book`; ghi giới hạn giấy rời ở `messages.py`/README + test
- [x] T031 [P] [US3] `ImageQualityExtractor`: brightness, contrast, edge_density, blur, frozen; luật che camera đa đặc trưng; `CAMERA_FAULT` là trạng thái thiết bị trên UI, không phải sự kiện (`features/image_quality.py`, `behavior/safety.py`) + test
- [x] T032 [US3] `SeatZone` từ trung vị bbox người lúc hiệu chỉnh/3 giây đầu; luật rời chỗ kết hợp tâm, diện tích, vai (`calibration/`, `behavior/safety.py`) thay `bbox.y > 0.9` + test
- [x] T033 [US3] Hiệu chỉnh đầu tự đề xuất (`SUGGESTED`, ổn định 2 s, ±25°), chấp nhận bởi giám thị, từ chối mốc không chắc (`face/head_pose.py`, `calibration/`, UI) + test
- [ ] T034 [US3] Thử `image_size` 640 và lọc hình học điện thoại (tắt mặc định); chỉ bật khi replay chứng minh; ghi FPS + số đo
- [ ] T035 [US3] Chỉnh ngưỡng hướng đầu/điện thoại/vắng người theo replay OEP; ghi trước/sau vào `docs/evaluation-results.md`, cập nhật `configs/default.yaml` và `docs/event-detection-tuning.md`

---

## Phase 6: User Story 4 — Hướng mắt (P2)

**Goal**: ước lượng hướng mắt + hiệu chỉnh + `GAZE_OFF_SCREEN` (thi máy), `GAZE_SIDEWAYS` (cả hai).
**Independent Test**: liếc ra ngoài 5 s không quay đầu → 1 cảnh báo thấp; nhìn góc màn hình → 0.

- [x] T036 [P] [US4] Trích đặc trưng mắt từ 478 điểm mốc (tỉ lệ ngang/dọc, độ mở, độ rộng px) trong `packages/proctoring_ai/src/proctoring_ai/face/eyes.py`; `landmarks.py` trả `GazeObs` + `FaceKeypoints` + test với điểm mốc tổng hợp
- [x] T037 [P] [US4] `GazeCalibration` (thu mẫu theo điểm, khớp tuyến tính đầu+mắt → tọa độ màn hình chuẩn hóa, biên, thời hạn, lỗi) trong `packages/proctoring_core/src/proctoring_core/calibration/gaze.py` + test
- [x] T038 [US4] Luật `GazeRuleEvaluator` (`behavior/gaze.py`): chất lượng, off-screen + sideways, trần điểm, `experimental`, theo kiểu bài thi; thêm `EventType.GAZE_OFF_SCREEN/GAZE_SIDEWAYS`, nhãn tiếng Việt + test
- [x] T039 [US4] Hộp thoại hiệu chỉnh (`widgets/calibration_dialog.py`): điểm nhìn theo kiểu bài thi, tiến trình, thất bại có lý do; trạng thái "Hướng mắt" trên bảng trạng thái; đo tổng thời gian hiệu chỉnh < 20 s (SC-007)
- [ ] T040 [US4] Replay: clip `head-gaze` + OEP; đo độ phủ/báo nhầm và tác động FPS; chỉnh mặc định

---

## Phase 7: User Story 5 — Nói chuyện (P2)

**Goal**: dao động độ mở miệng có nhịp → `POSSIBLE_TALKING`; loại ngáp.
**Independent Test**: nói 5 s → 1 cảnh báo thấp; ngáp 3 s → 0.

- [x] T041 [P] [US5] Trích `MouthObs` (MAR theo điểm 13/14/61/291, che miệng) trong `face/mouth.py` + test
- [x] T042 [US5] `TalkingRuleEvaluator` (`behavior/talking.py`): cửa sổ trượt, đếm dao động, biên độ, độ lệch chuẩn, loại ngáp, chất lượng, trần điểm; thay nhánh chết trong `supporting.py` + test chuỗi MAR tổng hợp (nói, ngáp, nhai ngắn)
- [ ] T043 [US5] Replay: clip `talking` + nhãn nói chuyện OEP; ghi giới hạn tần số theo AI FPS

---

## Phase 8: User Story 6 — Hoạt động tay (P2)

**Goal**: ghép tay đúng, tốc độ chuẩn hóa, gần mặt/tai theo điểm mốc, loại chống cằm.
**Independent Test**: chống cằm 30 s / viết 60 s → 0; giữ tai 5 s → 1 cảnh báo thấp.

- [x] T044 [P] [US6] Ghép tay theo handedness + gần nhất, `hand_id` ổn định, tốc độ / độ rộng mặt / Δt trong `hands/hand_estimator.py` + test hoán đổi thứ tự tay
- [x] T045 [US6] `HandRuleEvaluator` (`behavior/hands.py`): `HAND_NEAR_EAR`, `HAND_NEAR_FACE` (loại chống cằm dưới cằm), `HAND_RAPID_MOVEMENT`; xóa `supporting.py` + test
- [ ] T046 [US6] Replay: clip `hands` + OEP; ghi số đo

---

## Phase 9: User Story 7 — Rà soát và báo nhầm (P3)

**Goal**: chi tiết sự kiện + 4 trạng thái rà soát lưu append-only.
**Independent Test**: mở cảnh báo, bấm "Báo nhầm" → dòng `review` trong `events.jsonl`, điểm lịch sử không đổi.

- [x] T047 [P] [US7] `ReviewVerdict`/`ReviewLabel` + ghi dòng `review` append-only, đọc trạng thái mới nhất (`events/review_models.py`, `proctoring_runtime/persistence.py`) + test hợp đồng
- [x] T048 [US7] `widgets/event_detail.py`: ảnh đúng frame + khung bao từ `object_trace`, độ tin cậy, mô hình, kiểu bài thi, lý do, tín hiệu phụ, 3 nút; danh sách một hàng/episode, nhãn "Thử nghiệm"
- [x] T049 [US7] Reader JSONL đọc được dòng v1 (thiếu trường hiển thị "không có dữ liệu") + test

---

## Phase 10: Polish & hoàn tất

- [x] T050 [P] Rủi ro: xác nhận FR-026/FR-027 (decay, floor chỉ khi sự kiện còn cập nhật, lịch sử không thành sàn); tín hiệu thử nghiệm không một mình vượt "trung bình" (SC-006) + test
- [x] T051 [P] Cập nhật README, `PRODUCT.md`, `docs/event-detection-tuning.md`, `docs/evaluation-results.md`; điền `docs/core-project-summary.md` (đang rỗng); liên kết spec 001 → 002
- [ ] T052 Replay toàn bộ OEP + clip, so với mốc, kiểm SC-001..SC-008; ghi kết quả
- [x] T053 `ruff`, `mypy`, `pytest` 0 lỗi (SC-009); thử thật theo `quickstart.md` §5 — lint/type/test sạch;
      bước thử camera thủ công do người vận hành thực hiện

---

## Dependencies & Execution Order

- Setup (T001–T003) → Foundational (T004–T012) → Tái cấu trúc (T013–T022) → các story.
- T012 cần dữ liệu OEP (đang tải); có thể làm T013+ với golden tổng hợp trong lúc chờ, nhưng mốc OEP phải
  được ghi bằng mã trước tái cấu trúc (giữ commit/worktree mốc).
- US2 (T023–T026) trước US3–US6 vì luật mới đọc cấu hình đã resolve theo kiểu bài thi.
- US4 và US5 dùng chung thay đổi `landmarks.py` (T036 trước T041).
- US7 cần `episode_id`/`object_trace` từ T029.
- Polish sau tất cả.

## Parallel Opportunities

- T004, T005, T006, T010 song song; T013, T014 song song; T027, T028, T030, T031 song song;
  T036, T037 song song; T041, T044, T047 song song.

## Implementation Strategy

MVP = Phase 1–5 (đo được, cấu trúc đúng, kiểu bài thi, tín hiệu chính đúng). Sau đó US4–US6 (tín hiệu
hỗ trợ), US7, Polish. Mỗi phase kết thúc bằng checkpoint và số đo replay.
