from __future__ import annotations

from uuid import uuid4

import pytest
from proctoring_ai.multi.face_pool import FaceResult
from proctoring_ai.multi.pipeline import FrameTiming, MultiFrameResult
from proctoring_ai.multi.timing import TimingRecorder
from proctoring_ai.multi.tracker import Seat, TrackerUpdate
from proctoring_core.observations import Box


def seat(runtime_id: int) -> Seat:
    box = Box(0.1 * runtime_id, 0.3, 0.1, 0.4)
    return Seat(uuid4(), runtime_id, box, box.center, 0.9, 0, 0, True)


def result(ms: int, seats: list[Seat], *, total: float = 50.0, reused: bool = False, created: int = 0,
           detector_reused: bool = False, over_limit: int = 0) -> MultiFrameResult:
    faces = {s.seat_id: FaceResult("ready", {}, reused, 0.0 if reused else 10.0, "uncalibrated") for s in seats}
    timing = FrameTiming(20.0, 1.0, 12.0, {} if reused else {s.seat_id: 10.0 for s in seats}, 0.0, total,
                         0 if reused else len(seats), len(seats) if reused else 0, detector_reused)
    update = TrackerUpdate(list(seats), [seat(99)] * over_limit, [], [], created, 0)
    return MultiFrameResult(ms // 100, ms, update, {}, faces, {}, [], timing)


def test_percentiles_and_fps() -> None:
    recorder = TimingRecorder()
    for index in range(10):
        recorder.add(result(index * 100, [], total=10.0 * (index + 1)), latency_ms=10.0 * (index + 1))
    summary = recorder.summary(wall_s=2.0)
    assert summary["frames"] == 10 and summary["ai_fps"] == pytest.approx(5.0)
    assert summary["models"]["total"] == {"p50": 50.0, "p95": 100.0}
    assert summary["latency_ms"] == {"p50": 50.0, "p95": 100.0}
    assert summary["models"]["detect"] == {"p50": 20.0, "p95": 20.0}


def test_face_hz_counts_fresh_and_reused_separately() -> None:
    recorder = TimingRecorder()
    s = seat(1)
    for index in range(10):
        recorder.add(result(index * 100, [s], reused=index % 2 == 1), latency_ms=50.0)
    recorder.set_calibration(1, "calibrated")
    summary = recorder.summary(wall_s=1.0)
    assert summary["seats"][1]["face_hz"] == pytest.approx(10.0)
    assert summary["seats"][1]["fresh_face_hz"] == pytest.approx(5.0)
    assert summary["seats"][1]["calibration_status"] == "calibrated"
    assert summary["face_reuse_ratio"] == pytest.approx(0.5)
    assert summary["models"]["face_each"] == {"p50": 10.0, "p95": 10.0}


def test_id_switches_counts_seats_beyond_concurrent_max() -> None:
    recorder = TimingRecorder()
    a, b, c = seat(1), seat(2), seat(3)
    recorder.add(result(0, [a, b], created=2), latency_ms=1.0)
    recorder.add(result(100, [a, c], created=1), latency_ms=1.0)
    summary = recorder.summary(wall_s=1.0)
    assert summary["seats_created"] == 3 and summary["id_switches"] == 1


def test_dropped_reused_detector_and_over_limit_counts() -> None:
    recorder = TimingRecorder()
    recorder.add(result(0, [], detector_reused=True, over_limit=1), latency_ms=1.0)
    recorder.add(result(100, []), latency_ms=1.0)
    recorder.add_dropped()
    summary = recorder.summary(wall_s=1.0)
    assert summary["dropped_frames"] == 1
    assert summary["detector_reuse_ratio"] == pytest.approx(0.5)
    assert summary["over_limit_frames"] == 1


def test_empty_video_summary_has_zeros_not_errors() -> None:
    summary = TimingRecorder().summary(wall_s=0.0)
    assert summary["frames"] == 0 and summary["ai_fps"] == 0.0
    assert summary["latency_ms"] == {"p50": 0.0, "p95": 0.0}
    assert summary["seats"] == {} and summary["face_reuse_ratio"] == 0.0 and summary["id_switches"] == 0
