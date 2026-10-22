# Implementation Plan: Hoàn thiện Phase 1

**Branch**: `002-phase1-hardening` | **Date**: 2026-10-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-phase1-hardening/spec.md`

## Summary

Đưa phase 1 từ "phát hiện cơ bản, chưa chính xác" thành bản hoàn chỉnh đo được: (1) dựng bộ phát lại và
chấm điểm mức sự kiện trên OEP dataset + clip tự quay để có mốc; (2) tái cấu trúc không đổi hành vi —
quan sát và tín hiệu có kiểu, cấu hình Pydantic là nguồn duy nhất, luật và điều phối luật về lớp nghiệp
vụ, gói runtime không phụ thuộc Qt dùng chung cho giao diện và công cụ đánh giá; (3) sửa độ chính xác
từng tín hiệu có số đo trước/sau; (4) thêm kiểu bài thi (giấy/máy), hướng mắt và nói chuyện từ điểm mốc
mặt sẵn có; (5) rà soát và đánh dấu báo nhầm.

## Technical Context

**Language/Version**: Python 3.12 (`>=3.12,<3.13`)

**Primary Dependencies**: PySide6 6.8+, OpenCV 4.10+, Ultralytics YOLO11n (AGPL), MediaPipe Tasks 0.10
(face/pose/hand landmarker), NumPy 2, Pydantic 2, PyYAML. Không thêm dependency mới.

**Storage**: Tệp cục bộ: `storage/sessions/<id>/{session.json, events.jsonl, risk-summary.json,
evidence/*.jpg}`, `storage/evaluation/<run>/report.{json,md}`; dữ liệu đánh giá ngoài repo
(`E:\datasets\oep`).

**Testing**: pytest + pytest-asyncio (đơn vị luật, hợp đồng, tích hợp với fake analyzer), golden replay
tổng hợp trên CI, replay OEP cục bộ; ruff; mypy strict.

**Target Platform**: Windows 10/11 desktop; CI chạy Ubuntu + Windows không có mô hình.

**Project Type**: Ứng dụng desktop + các gói thư viện Python nội bộ (monorepo `uv`, `package = false`).

**Performance Goals**: Không giảm AI FPS quá 15% so với mốc khi bật hướng mắt + nói chuyện (SC-008);
preview độc lập AI như hiện tại. Máy tham chiếu: Quadro M1200, YOLO ≈ 5,5 FPS (CUDA 11.8).

**Constraints**: Chạy offline sau khi có mô hình; không lưu video phiên; không nhận dạng danh tính/cảm
xúc; tín hiệu miệng/tay/mắt chỉ là hỗ trợ có trần điểm; giao diện tiếng Việt.

**Scale/Scope**: 1 camera, 1 thí sinh; 24 video OEP (~20–30 phút mỗi video) + ~8 clip tự quay.

## Constitution Check

*GATE: kiểm tra trước Phase 0 và sau Phase 1 thiết kế.*

| Nguyên tắc | Đánh giá | Ghi chú |
|---|---|---|
| I. Con người quyết định | PASS | Thêm rà soát 4 trạng thái; không nhãn "gian lận"; hiệu chỉnh tự đề xuất cần giám thị chấp nhận. |
| II. Thời gian, đa tín hiệu | PASS | Mọi luật mới có thời lượng/cửa sổ; thứ bậc tín hiệu chống cộng trùng. |
| III. Cấu hình | PASS | Chuyển mọi ngưỡng viết cứng vào schema; kiểu bài thi là phần ghi đè cấu hình. |
| IV. Tách AI và nghiệp vụ | PASS (sửa vi phạm hiện có) | `_RuleSet` và helper không gian chuyển từ `apps/desktop` về `proctoring_core`. Bộ trích đặc trưng mắt/miệng ở `proctoring_ai` chỉ ra số đo; luật ở core. |
| V. Quyền riêng tư | PASS | Không lưu landmark thô/video; OEP chỉ đánh giá nội bộ, ngoài Git; clip tự quay cần đồng ý. Không dùng âm thanh. |
| VI. Đo trước khi hoàn tất | PASS | US1 là điều kiện tiên quyết; báo cáo báo nhầm/phút, bỏ sót, độ trễ, FPS theo hồ sơ phần cứng. |
| Ràng buộc: tín hiệu dễ báo nhầm là hỗ trợ | PASS | `risk_cap` cho gaze/talking/hands; cờ `experimental`. |
| Quy trình: hồi quy video sau mỗi thay đổi AI | PASS | Golden tổng hợp trên CI + replay OEP so với mốc. |

**Sau thiết kế Phase 1**: không phát sinh vi phạm. Thay đổi so với đề xuất hợp đồng spec 001: tín hiệu
mắt/miệng **bật mặc định** (spec 001 đề xuất tắt) theo yêu cầu người dùng, bù bằng trần điểm, nhãn
"Thử nghiệm" và cờ tắt trong cấu hình.

## Project Structure

### Documentation (this feature)

```text
specs/002-phase1-hardening/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── evaluation.md
│   └── session-data.md
├── checklists/requirements.md
└── tasks.md
```

### Source Code (repository root)

```text
packages/proctoring_core/src/proctoring_core/
├── observations.py              # MỚI: Observation, FaceObs, GazeObs, MouthObs, HandObs, … (typed)
├── calibration/                 # MỚI: CalibrationProfile, trạng thái, SeatZone, GazeCalibration
├── behavior/
│   ├── signals.py               # MỚI: RuleSignal, SignalState
│   ├── base.py                  # MỚI: lớp nền evaluator (registry TemporalCondition theo key)
│   ├── geometry.py              # MỚI: inside_person, hand_near_face, … (từ runtime.py)
│   ├── engine.py                # MỚI: BehaviorEngine (thay _RuleSet), chọn theo kiểu bài thi
│   ├── arbiter.py               # MỚI: SignalArbiter (thứ bậc)
│   ├── head.py, objects.py, presence.py, safety.py, temporal_buffer.py   # tái cấu trúc
│   ├── gaze.py, talking.py, hands.py   # MỚI (hands tách từ supporting.py)
├── config/schema.py             # mở rộng: thresholds, gaze, talking, hands, exam_modes, hierarchy, evaluation
├── evaluation/                  # MỚI: labels.py, matching.py, report.py (thuần, không I/O mô hình)
├── events/, risk/, sessions/    # mở rộng trường (episode_id, supporting_signals, review)

packages/proctoring_ai/src/proctoring_ai/
├── capture/video_file_source.py # MỚI: nguồn frame từ tệp, timestamp theo video
├── face/eyes.py, face/mouth.py  # MỚI: trích đặc trưng mống mắt / độ mở miệng
├── face/landmarks.py            # trả FaceObs có gaze + mouth + keypoints
├── hands/hand_estimator.py      # ghép tay ổn định, tốc độ chuẩn hóa
├── features/image_quality.py    # brightness/contrast/edge/frozen

packages/proctoring_runtime/src/proctoring_runtime/   # MỚI (không Qt)
├── bootstrap.py                 # nạp + resolve cấu hình theo kiểu bài thi, chọn thiết bị
├── session_engine.py            # điều phối capture → pipeline → BehaviorEngine → aggregator → risk
├── persistence.py               # JsonlEventRepository, SessionManifestWriter, risk summary
├── evidence.py                  # chuyển từ apps/desktop (EvidenceRecorder)
├── payloads.py                  # map Observation/Event → dict cho UI và JSONL
└── replay.py                    # chạy video có nhãn → PredictedEvent, golden

apps/desktop/src/proctoring_desktop/
├── runtime.py                   # còn lớp QThread bọc SessionEngine, phát Signal
├── window.py                    # khung chính, lắp widget
├── widgets/                     # MỚI: preview.py, event_list.py, status_panel.py, calibration_dialog.py, event_detail.py
├── overlay.py, messages.py

scripts/evaluate.py              # MỚI: CLI theo contracts/evaluation.md (thay replay.py stub)
tests/
├── fixtures/                    # fake analyzers mở rộng cho gaze/mouth
├── golden/                      # MỚI: kịch bản tổng hợp + golden JSONL
└── …                            # kiểm thử theo story
```

**Structure Decision**: Giữ monorepo hiện có, thêm gói `proctoring_runtime` để công cụ đánh giá và giao
diện dùng chung điều phối (research R15). Cập nhật `mypy_path`, `tests/conftest.py`, `apps/desktop/main.py`,
CI cho đường dẫn mới.

## Thứ tự triển khai và cổng kiểm tra

1. **Nền** (Setup + Foundational): dọn lint/mypy về 0; bộ phát lại + chấm điểm (US1); ghi golden tổng
   hợp và mốc OEP **trên mã hiện tại**.
2. **Tái cấu trúc không đổi hành vi**: kiểu dữ liệu, cấu hình, BehaviorEngine, runtime package, tách UI.
   Cổng: golden khớp 100%, kiểm thử xanh.
3. **Thay đổi hành vi có chủ đích** theo story (US2→US7). Cổng mỗi story: kiểm thử đơn vị + replay OEP so
   với mốc, ghi kết quả vào `docs/evaluation-results.md`.
4. **Hoàn tất**: cập nhật spec 001 (liên kết), README, `event-detection-tuning.md`; lint/mypy/test 0 lỗi;
   thử thật trên ứng dụng theo quickstart §5.

## Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Tải OEP chậm (~0,8 MB/s) | Bắt đầu bằng golden tổng hợp + subject đã tải; replay OEP khi đủ dữ liệu. |
| Mã nhãn OEP chưa có tài liệu số hóa | Xác minh bằng cách xem frame tại các khoảng nhãn; ghi kết quả vào ánh xạ. |
| AI FPS thấp làm nói chuyện kém | Đo và ghi giới hạn; trần điểm; có thể tăng nhịp mặt riêng nếu FPS cho phép. |
| Phát lại toàn OEP lâu (CPU) | Chọn tập con cố định cho vòng lặp nhanh; chạy toàn bộ ở cổng story. |
| Tái cấu trúc lớn làm vỡ hành vi | Golden so khớp từng byte sau mỗi bước. |

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Thêm gói thứ tư `proctoring_runtime` | Công cụ đánh giá và giao diện phải dùng chung một đường điều phối (FR-001, FR-029) | Để trong `apps/desktop` kéo PySide6 vào CLI; để trong `proctoring_ai` trộn adapter mô hình với lưu trữ |
