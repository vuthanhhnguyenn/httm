---
description: "Danh sách công việc triển khai Smart Exam Proctoring System"
---

# Công việc: Smart Exam Proctoring System

**Đầu vào**: Tài liệu thiết kế trong `specs/001-smart-exam-proctoring/`

**Tài liệu bắt buộc**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tổ chức**: Công việc được nhóm theo user story. Mỗi story có test riêng và tạo được một increment
có thể xác nhận độc lập.

## Định dạng task

`[ID] [P?] [Story?] Mô tả kèm đường dẫn file`

- `[P]`: Có thể làm song song sau khi dependency trực tiếp đã hoàn tất.
- `[US1]`, `[US2]`, `[US3]`: Task thuộc user story tương ứng.
- Task ở Setup, Foundation và Hoàn thiện không có nhãn story.

## Phase 1: Khởi tạo dự án

**Mục tiêu**: Tạo monorepo, dependency manifest và bộ công cụ phát triển dùng chung.

- [X] T001 Tạo cấu trúc monorepo và file giữ thư mục tại `apps/api/.gitkeep`, `apps/web/.gitkeep`, `packages/proctoring_core/.gitkeep`, `packages/proctoring_ai/.gitkeep`, `configs/.gitkeep`, `models/pretrained/.gitkeep`, `storage/sessions/.gitkeep`, `tests/contract/.gitkeep`, `tests/integration/.gitkeep`, `tests/video_samples/.gitkeep`, `tests/performance/.gitkeep`, `scripts/.gitkeep`, `docs/.gitkeep` và `docker/.gitkeep`
- [X] T002 Khởi tạo Python workspace 3.12 với các nhóm dependency API, AI, persistence, test và dev trong `pyproject.toml`; tách extra `opencv-python` cho máy phát triển và `opencv-python-headless` cho CI để hai gói không cùng tồn tại trong một environment
- [X] T003 [P] Khởi tạo React 19, TypeScript, Vite và Tailwind trong `apps/web/package.json`, `apps/web/tsconfig.json`, `apps/web/vite.config.ts` và `apps/web/src/main.tsx`
- [X] T004 [P] Cấu hình Ruff, mypy, pytest, coverage, ESLint, Prettier và Vitest trong `pyproject.toml`, `apps/web/eslint.config.js`, `apps/web/.prettierrc.json` và `apps/web/vite.config.ts`
- [X] T005 [P] Tạo cấu hình mẫu và biến môi trường tại `.env.example`, `configs/default.example.yaml`, `configs/exam_policy.example.yaml`; mặc định `allow_book=false`, `allow_scratch_paper=true`, `allow_phone=false`, `evidence_frame_enabled=true` và `evidence_clip_enabled=false`
- [X] T006 [P] Thiết lập quy tắc bỏ qua dữ liệu runtime, model lớn, evidence, database, report và secret trong `.gitignore`, đồng thời giữ model manifest và fixture metadata được version hóa
- [X] T007 [P] Tạo model manifest có `source`, `version`, `license`, `sha256`, `class_map` và metric baseline trong `models/manifest.yaml`, rồi ghi gate AGPL-3.0 hoặc Enterprise trong `docs/licensing.md`
- [X] T008 Thiết lập CI chạy lint, type check, unit test, contract test và frontend test trên Windows/Linux trong `.github/workflows/ci.yml`; chưa tải model hoặc video nhạy cảm từ nguồn không được kiểm soát

**Checkpoint**: Workspace cài được bằng `uv sync --extra opencv-python --dev` (máy phát triển),
CI dùng `uv sync --extra opencv-python-headless --dev`, và frontend cài được bằng
`npm install --prefix apps/web --no-audit --no-fund`.

## Phase 2: Nền tảng dùng chung

**Mục tiêu**: Hoàn tất các thành phần chặn mọi user story: domain boundary, config, persistence,
auth, API skeleton, realtime envelope và test fixture.

**Điều kiện**: Phải hoàn thành phase này trước khi bắt đầu implementation của user story.

- [X] T009 Đồng bộ state contract trong `specs/001-smart-exam-proctoring/data-model.md` và `specs/001-smart-exam-proctoring/contracts/openapi.yaml`: `STOPPED` là terminal; start lặp cùng `Idempotency-Key` trả kết quả cũ, còn start một session đã dừng trả `409`
- [X] T010 [P] Tạo enum và value object dùng chung cho session status, camera status, processing status, severity, risk level, event type, bbox chuẩn hóa và UTC timestamp trong `packages/proctoring_core/src/proctoring_core/types.py`; bbox và confidence phải nằm trong 0 đến 1, risk phải nằm trong 0 đến 100
- [X] T011 [P] Định nghĩa các port `Clock`, `FrameSource`, `ObjectDetector`, `LandmarkAnalyzer`, `Tracker`, `SessionRepository`, `EventRepository`, `EvidenceStore`, `AuditRepository` và `EventPublisher` trong `packages/proctoring_core/src/proctoring_core/ports/`
- [X] T012 [P] Tạo schema cấu hình có đủ nhóm `camera`, `detection`, `behavior.*`, `risk`, `evidence`, `runtime` và CLI validation trong `packages/proctoring_core/src/proctoring_core/config/schema.py` và `scripts/validate_config.py`; cấm threshold nằm ngoài schema
- [X] T013 Xây settings loader trong `apps/api/src/proctoring_api/settings.py`; từ chối `AUTH_MODE=development` khi `APP_ENV=production`, chuẩn hóa storage root và không ghi secret vào log
- [X] T014 [P] Tạo async database engine, unit of work và single-writer queue trong `apps/api/src/proctoring_api/persistence/database.py` và `writer.py`; chỉ bật WAL với SQLite 3.51.3 trở lên hoặc backport đã sửa, không cho SQLite nằm trên network filesystem
- [X] T015 Tạo Alembic environment, base metadata và script kiểm tra schema trong `apps/api/alembic.ini`, `apps/api/migrations/env.py`, `apps/api/src/proctoring_api/persistence/base.py` và `scripts/check_database.py`
- [X] T016 [P] Xây auth principal và RBAC cho `operator`, `reviewer`, `admin` trong `apps/api/src/proctoring_api/auth/principal.py`, `dependencies.py` và `permissions.py`; token hoặc credential không được xuất hiện trong URL hay log
- [X] T017 [P] Cài request ID, structured logging, exception mapping và bộ lọc dữ liệu nhạy cảm trong `apps/api/src/proctoring_api/observability/logging.py`, `middleware.py` và `errors.py`; log không chứa frame, landmark thô, bearer token hoặc đường dẫn evidence tuyệt đối
- [X] T018 Tạo FastAPI app, lifespan, router registry và health response cho database, storage, analyzer trong `apps/api/src/proctoring_api/main.py`, `routes/__init__.py` và `routes/health.py`
- [X] T019 [P] Xây session event bus, sequence tăng dần và WebSocket envelope `schema_version`, `sequence`, `type`, `session_id`, `occurred_at`, `data` trong `apps/api/src/proctoring_api/realtime/event_bus.py` và `schemas.py`
- [X] T020 [P] Sinh TypeScript API types/client từ `specs/001-smart-exam-proctoring/contracts/openapi.yaml` vào `apps/web/src/types/api.generated.ts` và `apps/web/src/services/apiClient.ts`; thêm script kiểm tra contract drift trong `apps/web/package.json`
- [X] T021 [P] Tạo app shell, router, layout, theme và trang lỗi trong `apps/web/src/App.tsx`, `apps/web/src/routes.tsx`, `apps/web/src/components/AppLayout.tsx` và `apps/web/src/styles.css`
- [X] T022 Tạo fixture clock, fake camera, fake detector, fake tracker, database tạm và WebSocket test client trong `tests/fixtures/`, `packages/proctoring_core/tests/conftest.py` và `apps/api/tests/conftest.py`

**Checkpoint**: Health API, auth dependency, database migration rỗng, event bus và frontend shell chạy
được; mọi story có thể dùng fake adapter để phát triển độc lập.

## Phase 3: User Story 1 - Theo dõi an toàn một thí sinh (P1, MVP)

**Mục tiêu**: Operator tạo, start, theo dõi và stop một session webcam; hệ thống phát event theo thời
gian, tính risk có giải thích, lưu event frame và hiển thị dashboard realtime mà không đưa ra kết luận
gian lận.

**Kiểm thử độc lập**: Chạy bộ video single-person bắt buộc và xác nhận session lifecycle, preview,
head turn, look down, phone, person missing, multiple person, risk decay, event timeline và evidence
frame. Chuyển động dưới ngưỡng hoặc detection một frame không được tạo event quan trọng.

### Test cho User Story 1

- [ ] T023 [P] [US1] Viết contract test cho `POST/GET /api/sessions`, `GET /api/sessions/{id}`, start/stop cùng `Idempotency-Key` và response `409` trong `tests/contract/test_sessions_api.py`
- [ ] T024 [P] [US1] Viết contract test cho session events, event detail, MJPEG preview và WebSocket envelope/sequence/reconnect trong `tests/contract/test_monitoring_contracts.py`
- [ ] T025 [P] [US1] Viết unit test state machine `CREATED -> STARTING -> RUNNING -> STOPPING -> STOPPED`, nhánh `FAILED`, giới hạn một active session và idempotency trong `packages/proctoring_core/tests/sessions/test_session_lifecycle.py`
- [ ] T026 [P] [US1] Viết unit test cho monotonic temporal buffer và event lifecycle `inactive/candidate/active`, bảo đảm một sustained condition chỉ tạo một event open/update/close trong `packages/proctoring_core/tests/events/test_temporal_event_lifecycle.py`
- [ ] T027 [P] [US1] Viết unit test risk cap 0 đến 100, decay, duration/repetition factor, correlation cap và `reason_codes` trong `packages/proctoring_core/tests/risk/test_risk_engine.py`
- [ ] T028 [P] [US1] Viết test boundary cho head turn, look down, abnormal movement, phone, document, person missing và multiple person trong `packages/proctoring_core/tests/behavior/test_primary_rules.py`; một frame và duration dưới ngưỡng phải tạo 0 event
- [ ] T029 [P] [US1] Viết test cho leaving seat, camera blocked, face not visible, hand activity và possible talking trong `packages/proctoring_core/tests/behavior/test_supporting_rules.py`; hand/talking độc lập không được đẩy risk lên `HIGH`
- [ ] T030 [P] [US1] Viết integration test từ fake camera qua observation, temporal rule, persistence và WebSocket trong `tests/integration/test_single_person_pipeline.py`, gồm analyzer timeout và storage failure hiển thị degraded/failed rõ ràng
- [ ] T031 [P] [US1] Viết component test cho camera/status/risk/timeline/session control và trạng thái WebSocket stale trong `apps/web/tests/monitoring-dashboard.test.tsx`
- [ ] T032 [P] [US1] Tạo manifest expected event window và runner fixture cho `normal`, `head_turn`, `phone_usage`, `look_down`, `second_person`, `leave_seat`, `camera_blocked`, `mixed_behavior` trong `tests/video_samples/manifest.yaml` và `scripts/replay.py`

### Domain và persistence cho User Story 1

- [ ] T033 [P] [US1] Tạo domain model `ExamSession` và `CandidateTrack` trong `packages/proctoring_core/src/proctoring_core/sessions/models.py`: `name` dài 1 đến 120 ký tự và không chứa dữ liệu sinh trắc học; mode chỉ `SINGLE_PERSON`/`MULTI_PERSON`; risk 0 đến 100; bbox và visibility 0 đến 1; unique `(session_id, runtime_track_id)`; config/policy snapshot bất biến sau start
- [ ] T034 [P] [US1] Tạo runtime/domain model `ObservationBundle`, `SuspiciousEvent` và `RiskSnapshot` trong `packages/proctoring_core/src/proctoring_core/events/models.py` và `risk/models.py`: frame ID và monotonic timestamp tăng dần; confidence 0 đến 1; `reason_codes` có ít nhất một phần tử; event `CLOSED` bắt buộc có `ended_at >= started_at` và `duration_ms >= 0`; risk snapshot chỉ chấp nhận score 0 đến 100
- [ ] T035 [US1] Tạo SQLAlchemy models và migration cho `exam_sessions`, `candidate_tracks`, `suspicious_events`, `risk_snapshots`, `exam_policies`, `system_config_versions` trong `apps/api/src/proctoring_api/persistence/models.py` và `apps/api/migrations/versions/0001_core_monitoring.py`, gồm index và unique constraint trong `data-model.md`
- [ ] T036 [US1] Cài repository cho session, track, event và risk snapshot trong `apps/api/src/proctoring_api/persistence/repositories.py`; không persist observation frame-level và chỉ ghi risk khi vượt epsilon, đổi level hoặc có event transition

### Pipeline AI cho User Story 1

- [ ] T037 [P] [US1] Cài `OpenCVFrameSource` cho webcam và file video, timestamp UTC/monotonic và giải phóng camera an toàn trong `packages/proctoring_ai/src/proctoring_ai/capture/opencv_source.py`
- [ ] T038 [P] [US1] Cài latest-frame queue có kích thước giới hạn, frame drop counter, latency histogram và in-memory preview buffer trong `packages/proctoring_ai/src/proctoring_ai/pipeline/frame_queue.py`, `metrics.py` và `preview_buffer.py`
- [ ] T039 [P] [US1] Cài YOLO object detector adapter chỉ trả class, confidence, bbox, model version/hash trong `packages/proctoring_ai/src/proctoring_ai/detectors/yolo_detector.py`; tạo `scripts/verify_models.py` để kiểm checksum và class map trước khi load
- [ ] T040 [P] [US1] Cài `SingleSubjectTracker` với anonymous runtime ID và trạng thái active/lost/closed trong `packages/proctoring_ai/src/proctoring_ai/tracking/single_tracker.py`; không suy diễn danh tính khi người rời rồi quay lại
- [ ] T041 [P] [US1] Cài MediaPipe face landmarker adapter ở video/live-stream mode với timestamp tăng dần, timeout và health result trong `packages/proctoring_ai/src/proctoring_ai/face/landmarks.py`
- [ ] T042 [US1] Cài head pose estimator và calibration đầu session để xuất yaw, pitch, roll cùng quality score trong `packages/proctoring_ai/src/proctoring_ai/face/head_pose.py`
- [ ] T043 [P] [US1] Cài MediaPipe pose và hand adapters chỉ xuất landmark/visibility/health trong `packages/proctoring_ai/src/proctoring_ai/pose/pose_estimator.py` và `hands/hand_estimator.py`
- [ ] T044 [P] [US1] Cài image quality extractor cho brightness, entropy, blur và occlusion estimate trong `packages/proctoring_ai/src/proctoring_ai/features/image_quality.py`
- [ ] T045 [US1] Ghép detector, tracker, face, pose, hand và image quality thành `ObservationBundle` trong `packages/proctoring_ai/src/proctoring_ai/pipeline/observation_pipeline.py`; analyzer thiếu kết quả phải ghi `not_run`, `timeout`, `no_detection` hoặc `error`

### Event và risk cho User Story 1

- [ ] T046 [P] [US1] Cài ring temporal buffer theo track, sliding window và monotonic duration trong `packages/proctoring_core/src/proctoring_core/behavior/temporal_buffer.py`
- [ ] T047 [P] [US1] Cài rule `HEAD_TURN_LEFT`, `HEAD_TURN_RIGHT`, `LOOK_DOWN`, `ABNORMAL_HEAD_MOVEMENT` dùng config snapshot trong `packages/proctoring_core/src/proctoring_core/behavior/head.py`
- [ ] T048 [P] [US1] Cài rule `PERSON_MISSING` và `MULTIPLE_PERSON_DETECTED` theo minimum duration trong `packages/proctoring_core/src/proctoring_core/behavior/presence.py`
- [ ] T049 [P] [US1] Cài feature/rule baseline `PHONE_DETECTED` và `DOCUMENT_DETECTED` theo confidence, duration và exam policy trong `packages/proctoring_core/src/proctoring_core/behavior/objects.py`
- [ ] T050 [P] [US1] Cài rule `LEAVING_SEAT`, `CAMERA_BLOCKED`, `FACE_NOT_VISIBLE` từ pose, allowed zone, image quality và visibility trong `packages/proctoring_core/src/proctoring_core/behavior/safety.py`
- [ ] T051 [P] [US1] Cài rule hỗ trợ `SUSPICIOUS_HAND_ACTIVITY` và experimental `POSSIBLE_TALKING` với severity/risk cap thấp trong `packages/proctoring_core/src/proctoring_core/behavior/supporting.py`
- [ ] T052 [US1] Cài event aggregator bảo đảm một event `OPEN` duy nhất cho `(session_id, track_id, event_type, rule_key)` và tạo explanation từ metric/threshold trong `packages/proctoring_core/src/proctoring_core/events/aggregator.py`
- [ ] T053 [US1] Cài deterministic risk engine có weight, decay, duration, repetition, named correlation, cap 100 và trace `reason_codes` trong `packages/proctoring_core/src/proctoring_core/risk/engine.py`
- [ ] T054 [US1] Cài evidence frame capture tối thiểu từ preview ring buffer, ghi nguyên tử bằng UUID dưới `storage/sessions/{session_id}/frames/` và trả trạng thái `AVAILABLE`/`FAILED` trong `apps/api/src/proctoring_api/evidence/frame_store.py`
- [ ] T055 [US1] Xây `SessionRunner` ghép capture worker, latest-frame queue, observation pipeline, temporal rules, event/risk persistence, evidence frame và event bus trong `apps/api/src/proctoring_api/services/session_runner.py`; Phase 1 chỉ cho một runner hoạt động và luôn giải phóng tài nguyên khi stop/fail

### API và giao diện cho User Story 1

- [ ] T056 [US1] Cài create/list/get/start/stop session routes đúng OpenAPI trong `apps/api/src/proctoring_api/routes/sessions.py`; start/stop phải idempotent theo key và phản hồi state trong 2 giây ở môi trường tham chiếu
- [ ] T057 [P] [US1] Cài list/filter session events và event detail trong `apps/api/src/proctoring_api/routes/events.py`, gồm cursor, track, type, severity, from/to và thứ tự mới nhất trước
- [ ] T058 [US1] Cài MJPEG preview và WebSocket session endpoint với auth, `connection.ready`, status, track, event, risk, metrics, degraded, ping/pong và close code trong `apps/api/src/proctoring_api/routes/preview.py` và `realtime/websocket.py`
- [ ] T059 [US1] Xây trang monitoring gồm `SessionControl`, `CameraView`, `DetectionOverlay`, `StatusPanel`, `RiskIndicator`, `EventTimeline` và reducer xử lý sequence gap/reconnect trong `apps/web/src/features/monitoring/` và `apps/web/src/pages/MonitoringPage.tsx`
- [ ] T060 [US1] Hoàn tất end-to-end checkpoint cho create/start/monitor/stop/history bằng fake backend và video fixtures trong `apps/web/tests/e2e/single-person.spec.ts` và `tests/integration/test_us1_acceptance.py`; ghi kết quả MVP vào `artifacts/us1-acceptance/README.md`

**Checkpoint US1**: MVP chạy độc lập. Normal video không tạo false `HIGH`/`CRITICAL`; event có timestamp,
confidence, explanation, risk effect và evidence status; risk giảm khi hành vi bất thường kết thúc.

## Phase 4: User Story 2 - Review evidence và áp dụng exam policy (P2)

**Mục tiêu**: Reviewer xem timeline/evidence và ghi disposition; admin quản lý policy/config có version;
evidence được phân quyền, audit và xóa theo retention.

**Kiểm thử độc lập**: Chạy cùng mixed-behavior video với policy cho phép giấy nháp nhưng cấm sách và
điện thoại; xác nhận suppression/event/risk, evidence access, audit, review disposition và retention.

### Test cho User Story 2

- [ ] T061 [P] [US2] Viết unit test versioning, canonical hash, một config `ACTIVE`, policy snapshot và constraint retention trong `packages/proctoring_core/tests/config/test_versioned_config_policy.py`
- [ ] T062 [P] [US2] Viết test evidence frame/clip, path confinement, atomic write, SHA-256, trạng thái `PENDING/AVAILABLE/FAILED/DELETED` và retention idempotent trong `apps/api/tests/test_evidence_retention.py`
- [ ] T063 [P] [US2] Viết test RBAC và audit cho evidence/config/policy/review trong `apps/api/tests/test_authorization_audit.py`; denied access cũng phải có audit outcome `DENIED`
- [ ] T064 [P] [US2] Viết contract test cho config, exam policy, event review và evidence content gồm `403`, `404`, `410` trong `tests/contract/test_review_policy_evidence_api.py`
- [ ] T065 [P] [US2] Viết frontend test cho history filter, event detail, evidence failure/deleted state, review form và settings validation trong `apps/web/tests/event-review-settings.test.tsx`
- [ ] T066 [P] [US2] Viết integration test mixed-behavior với `allow_scratch_paper=true`, `allow_book=false`, `allow_phone=false` và policy cho phép phone trong `tests/integration/test_policy_evidence_review_flow.py`

### Domain và persistence cho User Story 2

- [ ] T067 [P] [US2] Tạo domain model `ExamPolicy`, `SystemConfigVersion`, `EvidenceArtifact`, `ReviewDisposition`, `AuditLog` trong `packages/proctoring_core/src/proctoring_core/config/models.py` và `events/review_models.py`: policy name 1 đến 120 ký tự; pre/post event 0 đến 30 giây; retention tối thiểu 1 ngày; config SHA-256 là 64 ký tự hex; review note tối đa 2000 ký tự; disposition chỉ `VALID_SIGNAL`, `FALSE_POSITIVE`, `INCONCLUSIVE`
- [ ] T068 [US2] Tạo SQLAlchemy models và migration cho evidence, review, audit cùng các index retention/resource/actor trong `apps/api/src/proctoring_api/persistence/review_models.py` và `apps/api/migrations/versions/0002_evidence_review_audit.py`; audit là append-only ở tầng ứng dụng
- [ ] T069 [US2] Cài repository cho policy/config version, evidence metadata, review history và audit trong `apps/api/src/proctoring_api/persistence/review_repositories.py`; không cho xóa policy/config đã được session tham chiếu

### Service và API cho User Story 2

- [ ] T070 [P] [US2] Cài versioned config service với schema validation, optimistic `expected_active_version`, canonical JSON hash và đúng một bản `ACTIVE` trong `apps/api/src/proctoring_api/services/config_service.py`
- [ ] T071 [P] [US2] Cài exam policy service tạo version mới và chụp snapshot bất biến khi session start trong `apps/api/src/proctoring_api/services/policy_service.py`
- [ ] T072 [US2] Mở rộng evidence service thành pre/post frame ring buffer và clip tùy policy, giới hạn trong session boundary, ghi nguyên tử và không lưu full-session video trong `apps/api/src/proctoring_api/evidence/service.py`
- [ ] T073 [US2] Cài evidence authorization/streaming và audit cho `ALLOWED`, `DENIED`, `FAILED` trong `apps/api/src/proctoring_api/services/evidence_access.py`; API không được trả `relative_path` cho client
- [ ] T074 [P] [US2] Cài retention job dry-run/execute: xóa file trước, chuyển metadata sang `DELETED`, chạy lại an toàn và trả `410` khi truy cập file hết hạn trong `scripts/retention.py` và `apps/api/src/proctoring_api/services/retention.py`
- [ ] T075 [P] [US2] Cài review service ghi lịch sử disposition mới mà không ghi phán quyết gian lận trong `apps/api/src/proctoring_api/services/review_service.py`
- [ ] T076 [US2] Tích hợp policy snapshot vào object rules và evidence/risk calculation trong `packages/proctoring_core/src/proctoring_core/behavior/objects.py` và `apps/api/src/proctoring_api/services/session_runner.py`; material được phép phải tạo 0 event và 0 risk delta
- [ ] T077 [P] [US2] Cài `GET/PUT /api/config` và CRUD versioned exam policy đúng role/admin contract trong `apps/api/src/proctoring_api/routes/config.py` và `routes/policies.py`
- [ ] T078 [US2] Cài evidence content và event review routes đúng RBAC, audit, idempotency và status code trong `apps/api/src/proctoring_api/routes/evidence.py` và `routes/reviews.py`
- [ ] T079 [US2] Xây session history, filterable event timeline, event detail/replay, review form và admin settings trong `apps/web/src/features/review/`, `apps/web/src/features/settings/`, `apps/web/src/pages/EventReviewPage.tsx` và `SettingsPage.tsx`; risk luôn được ghi là mức ưu tiên review
- [ ] T080 [US2] Chạy end-to-end acceptance cho policy, evidence, audit, review và retention trong `apps/web/tests/e2e/review-policy.spec.ts` và `tests/integration/test_us2_acceptance.py`; lưu usability trial script 60 giây tại `tests/usability/review-event.md`

**Checkpoint US2**: Policy quyết định event đúng theo vật thể được phép/cấm; reviewer xem được lý do và
evidence theo quyền; mọi access được audit; retention xóa file nhưng giữ metadata theo policy.

## Phase 5: User Story 3 - Theo dõi phòng thi nhiều thí sinh (P3)

**Mục tiêu**: Theo dõi nhiều anonymous track, duy trì state/risk riêng, gán object có thể giải thích,
hỗ trợ seat mapping và cung cấp room dashboard.

**Kiểm thử độc lập**: Replay room recordings 1, 5, 10 và tối đa 20 người; đo track continuity,
ID switch, track loss, fragmentation, occlusion recovery, per-track risk isolation, association và UI.

### Test cho User Story 3

- [ ] T081 [P] [US3] Viết unit test tracker lifecycle `ACTIVE/OCCLUDED/LOST/CLOSED`, timeout recovery và tạo track mới khi recovery không đủ tin cậy trong `packages/proctoring_ai/tests/tracking/test_multi_tracker.py`
- [ ] T082 [P] [US3] Viết unit test object association dùng bbox, wrist/torso distance, temporal consistency và trả `ambiguous` khi chênh lệch score dưới ngưỡng trong `packages/proctoring_ai/tests/tracking/test_object_association.py`
- [ ] T083 [P] [US3] Viết unit test seat polygon tối thiểu 3 điểm, không tự cắt, nằm trong frame và hysteresis chống đổi seat một frame trong `packages/proctoring_core/tests/sessions/test_seat_mapping.py`
- [ ] T084 [P] [US3] Viết test per-track temporal buffer, event history và risk isolation khi track overlap/lost/recovered trong `packages/proctoring_core/tests/behavior/test_per_track_state.py`
- [ ] T085 [P] [US3] Viết contract test track list, `track.updated`, room metrics và object ambiguous payload trong `tests/contract/test_multi_person_contracts.py`
- [ ] T086 [P] [US3] Viết integration test room pipeline với fake multi-detector/tracker cho 1, 5, 10, 20 track trong `tests/integration/test_multi_person_pipeline.py`
- [ ] T087 [P] [US3] Viết frontend test cho room grid, risk priority disclaimer, candidate detail, event replay và stale track trong `apps/web/tests/room-dashboard.test.tsx`

### Domain và tracking cho User Story 3

- [ ] T088 [P] [US3] Tạo `SeatRegion` domain/SQLAlchemy model và migration trong `packages/proctoring_core/src/proctoring_core/sessions/seat.py`, `apps/api/src/proctoring_api/persistence/seat_models.py` và `apps/api/migrations/versions/0003_seat_regions.py`: unique `(room_key, seat_key)`, polygon tối thiểu 3 điểm, tọa độ 0 đến 1, không tự cắt
- [ ] T089 [US3] Mở rộng track repository cho unique `(session_id, runtime_track_id)`, last seen/lost/closed timestamp và state recovery trong `apps/api/src/proctoring_api/persistence/track_repository.py`
- [ ] T090 [P] [US3] Cài ByteTrack adapter sau interface `Tracker` và khóa tracker config trong `packages/proctoring_ai/src/proctoring_ai/tracking/bytetrack_adapter.py` và `configs/trackers/bytetrack.yaml`
- [ ] T091 [US3] Cài multi-track lifecycle coordinator, metric ID switch/loss/fragmentation/recovery và anonymous persistent UUID trong `packages/proctoring_ai/src/proctoring_ai/tracking/multi_tracker.py`
- [ ] T092 [US3] Cài object-to-person association trả `assigned(track_id, score)` hoặc `ambiguous`, không gán im lặng cho track sai trong `packages/proctoring_ai/src/proctoring_ai/tracking/object_association.py`
- [ ] T093 [P] [US3] Cài seat mapper và hysteresis theo region cố định trong `packages/proctoring_core/src/proctoring_core/sessions/seat_mapper.py`
- [ ] T094 [US3] Mở rộng temporal/event/risk state thành map độc lập theo persistent track UUID trong `packages/proctoring_core/src/proctoring_core/behavior/track_state.py` và `risk/per_track.py`
- [ ] T095 [US3] Mở rộng `SessionRunner` cho `MULTI_PERSON`, association, seat mapping, per-track event/risk và cleanup track lifecycle trong `apps/api/src/proctoring_api/services/session_runner.py`

### API, UI và đánh giá cho User Story 3

- [ ] T096 [US3] Cài `GET /api/sessions/{session_id}/tracks`, WebSocket `track.updated` và room-level metrics trong `apps/api/src/proctoring_api/routes/tracks.py` và `realtime/websocket.py`
- [ ] T097 [P] [US3] Xây `RoomView`, `StudentGrid`, `RiskPriorityList` và room event stream trong `apps/web/src/features/room/` và `apps/web/src/pages/RoomDashboardPage.tsx`; giao diện phải ghi rõ ranking chỉ dùng ưu tiên review
- [ ] T098 [P] [US3] Xây candidate detail, per-track history và evidence replay trong `apps/web/src/features/candidates/` và `apps/web/src/pages/CandidateDetailPage.tsx`
- [ ] T099 [US3] Tạo evaluator và benchmark room cho 1, 5, 10, 20 người, gồm ID switch, loss, fragmentation, recovery và association accuracy trong `scripts/evaluate_tracking.py`, `tests/video_samples/room_manifest.yaml` và `tests/performance/test_room_scale.py`
- [ ] T100 [US3] Chạy end-to-end acceptance cho multi-person room trong `apps/web/tests/e2e/room-monitoring.spec.ts` và `tests/integration/test_us3_acceptance.py`; lưu report theo hardware/profile tại `artifacts/us3-acceptance/README.md`

**Checkpoint US3**: Mỗi track có state, risk và history riêng; object sai hoặc ambiguous không làm tăng
risk của người khác; room dashboard hiển thị đúng dữ liệu và metric tracking được lưu trong report.

## Phase 6: Hoàn thiện và quality gate liên story

**Mục tiêu**: Hoàn tất regression, benchmark, bảo mật, tài liệu và Definition of Done cho toàn feature.

- [ ] T101 [P] Cài evaluator tổng hợp detector precision/recall/mAP, event precision/recall/F1, false alerts/phút, missed events/session và detection delay trong `scripts/evaluate.py` và `tests/evaluation/test_metrics.py`
- [ ] T102 [P] Cài benchmark ghi CPU, GPU, RAM, OS, resolution, model hash, precision, capture/processed FPS, dropped frames, p50/p95 và analyzer latency trong `scripts/benchmark.py` và `tests/performance/test_benchmark_report.py`
- [ ] T103 Thu thập hoặc tạo bộ video regression có consent/nguồn hợp lệ, gán nhãn segment và lưu metadata thay vì identity trong `tests/video_samples/README.md`, `tests/video_samples/manifest.yaml` và `datasets/annotations/regression_events.json`
- [ ] T104 [P] Viết threat model và security tests cho auth bypass, path traversal, malicious camera URI, oversized payload, evidence access và secret leakage trong `docs/security.md` và `tests/security/`
- [ ] T105 [P] Kiểm thử log scrubbing và data minimization, xác nhận production log không chứa ảnh, biometric identity, landmark thô, token hoặc absolute evidence path trong `tests/integration/test_privacy_logging.py`
- [ ] T106 Chốt dependency lock, model hash, SBOM và bằng chứng tuân thủ giấy phép; chặn release nếu `LicenseRef-Unspecified` còn trong `specs/001-smart-exam-proctoring/contracts/openapi.yaml` bằng `uv.lock`, `apps/web/package-lock.json`, `artifacts/sbom/` và `scripts/check_release_gates.py`
- [ ] T107 Chạy profiler trên PyTorch baseline và chỉ ghi task ONNX/TensorRT tiếp theo khi detector là bottleneck và regression giữ accuracy trong tolerance tại `artifacts/performance/baseline.md` và `docs/model-optimization.md`
- [ ] T108 [P] Bổ sung test degraded/recovery cho camera disconnect, analyzer timeout, database busy, storage full, WebSocket disconnect và clock/frame disorder trong `tests/integration/test_degraded_recovery.py`
- [ ] T109 [P] Tạo container và local orchestration cho CPU/GPU profile trong `docker/api.Dockerfile`, `docker/web.Dockerfile` và `docker-compose.yml`; không bake model, secret hoặc evidence vào image
- [ ] T110 [P] Hoàn thiện tài liệu kiến trúc, dataset, behavior rule, privacy/retention và vận hành trong `docs/architecture.md`, `docs/dataset.md`, `docs/behavior-rules.md`, `docs/privacy.md` và `docs/operations.md`
- [ ] T111 Chạy toàn bộ kịch bản trong `specs/001-smart-exam-proctoring/quickstart.md` trên Windows và Linux, sửa mọi lệnh hoặc expected outcome sai rồi lưu biên bản tại `artifacts/quickstart-validation.md`
- [ ] T112 Chạy full test, video regression, benchmark và constitution review; cập nhật trạng thái acceptance criteria/Definition of Done trong `artifacts/release-readiness.md` và chỉ đánh dấu feature hoàn tất khi mọi failure có owner hoặc đã được xử lý

## Dependency và thứ tự thực hiện

### Dependency theo phase

```text
Phase 1 Setup
    -> Phase 2 Foundation
        -> Phase 3 US1 MVP
            -> Phase 4 US2
                -> Phase 5 US3
                    -> Phase 6 Quality gates
```

- Phase 1 không có dependency.
- Phase 2 phụ thuộc Phase 1 và chặn mọi user story.
- US1 bắt đầu sau Phase 2 và là MVP đầu tiên.
- US2 có thể viết test/service bằng fake event sau Phase 2, nhưng acceptance cần event/evidence frame của US1.
- US3 có thể phát triển tracker/association/room UI bằng fake adapter sau Phase 2, nhưng acceptance cần
  pipeline/event/risk của US1 và review/evidence của US2.
- Phase 6 bắt đầu từng phần sau US1; release gate cuối phụ thuộc các story nằm trong scope phát hành.

### Dependency trong mỗi user story

1. Viết test và xác nhận test fail vì capability chưa tồn tại.
2. Tạo domain model và persistence model.
3. Cài service hoặc pipeline.
4. Cài endpoint và giao diện.
5. Chạy integration, E2E, regression và checkpoint.

## Cơ hội làm song song

### Setup và Foundation

- Sau T001, T003 đến T007 có thể chia cho frontend, tooling, config và model governance.
- Sau T010/T011, T012, T014, T016, T017, T019, T020, T021 có thể chạy theo các nhánh file độc lập.

### Ví dụ song song cho US1

```text
Nhóm test: T023, T024, T025, T026, T027, T028, T029, T030, T031, T032
Nhóm adapter AI sau khi port ổn định: T037, T038, T039, T040, T041, T043, T044
Nhóm behavior rule sau temporal buffer: T047, T048, T049, T050, T051
Frontend và event list API sau khi contract ổn định: T057, T059
```

### Ví dụ song song cho US2

```text
Nhóm test: T061, T062, T063, T064, T065, T066
Service độc lập sau model/repository: T070, T071, T074, T075
Routes và frontend sau service contract: T077, T079
```

### Ví dụ song song cho US3

```text
Nhóm test: T081, T082, T083, T084, T085, T086, T087
Adapter/trợ giúp độc lập: T090, T093
Hai nhánh UI sau contract: T097, T098
```

## Chiến lược triển khai

### MVP trước

1. Hoàn tất Phase 1 và Phase 2.
2. Hoàn tất US1 từ T023 đến T060.
3. Dừng để chạy checkpoint, video regression và benchmark Phase 1.
4. Chỉ demo MVP khi event không được trình bày như kết luận gian lận và normal video không có false
   `HIGH`/`CRITICAL`.

### Giao hàng tăng dần

1. US1 cung cấp monitoring một thí sinh.
2. US2 thêm review, policy, evidence, retention và audit.
3. US3 thêm multi-person room mà không thay contract human review.
4. Mỗi increment giữ test của increment trước và cập nhật baseline metric.

### Chiến lược nhiều người thực hiện

- Một nhóm hoàn tất Setup và Foundation trước.
- Sau Foundation, nhóm AI có thể làm adapter/tracker, nhóm domain làm event/risk/policy, nhóm API làm
  persistence/routes/realtime, nhóm web làm UI từ fake contract, nhóm QA chuẩn bị fixture/evaluator.
- Chỉ merge khi contract test, test riêng của story và constitution gate đều đạt.

## Ghi chú thực thi

- Task `[P]` chỉ được chạy song song khi dependency trực tiếp đã hoàn tất và không cùng sửa một file.
- Model hoặc thư viện AI thay đổi phải chạy lại video regression và benchmark.
- Không đưa video, model hoặc evidence nhạy cảm vào Git nếu chưa có policy và nguồn hợp lệ.
- Commit theo từng task hoặc nhóm logic nhỏ; ghi task ID trong commit message để truy vết.
- Khi task phát hiện spec/plan/contract không khớp, sửa artifact nguồn trước rồi mới tiếp tục code.
