from __future__ import annotations

from math import cos, radians, sin
from uuid import uuid4

import numpy as np
import pytest
from proctoring_ai.face.head_pose import HeadPoseEstimator, pose_from_matrix
from proctoring_ai.tracking.object_tracks import ObjectEvidenceTracker
from proctoring_core.behavior.head import HeadRuleEvaluator
from proctoring_core.behavior.objects import ObjectRuleEvaluator
from proctoring_core.types import EventType
from proctoring_desktop.overlay import OverlayState


def rotation(yaw=0, pitch=0, roll=0):
    y, p, r = map(radians, (yaw, pitch, roll))
    ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    rx = np.array([[1, 0, 0], [0, cos(p), -sin(p)], [0, sin(p), cos(p)]])
    rz = np.array([[cos(r), -sin(r), 0], [sin(r), cos(r), 0], [0, 0, 1]])
    matrix = np.eye(4)
    matrix[:3, :3] = rz @ ry @ rx
    return matrix


@pytest.mark.parametrize("yaw,pitch,roll", [(35, 0, 0), (-35, 0, 0), (0, 30, 0), (0, 0, 40), (25, -15, 12)])
def test_head_rotation_axes_are_not_interchanged(yaw, pitch, roll):
    matrix = rotation(yaw, pitch, roll)
    matrix[:3, :3] *= 1.2
    assert pose_from_matrix(matrix) == pytest.approx({"yaw": yaw, "pitch": pitch, "roll": roll})


@pytest.mark.parametrize("matrix", [np.zeros((4, 4)), np.full((4, 4), np.nan), [1, 2, 3]])
def test_invalid_transform_does_not_mean_facing_forward(matrix):
    assert pose_from_matrix(matrix) is None


def test_calibration_removes_camera_offset_and_does_not_learn_a_turn_automatically():
    estimator = HeadPoseEstimator()
    face = {"yaw": 10, "pitch": -12, "roll": 2, "quality": 0.8}
    assert estimator.update(face, 0)["yaw"] == pytest.approx(10)
    estimator.begin_calibration()
    for timestamp in range(200, 2001, 200):
        result = estimator.update(face, timestamp)
    assert result["calibration_status"] == "calibrated"
    assert result["yaw"] == pytest.approx(0)
    assert result["pitch"] == pytest.approx(0)
    assert estimator.update({**face, "yaw": 50}, 2800)["yaw"] == pytest.approx(40)


def head_signals(evaluator, timestamp, **pose):
    return evaluator.evaluate(session_id=uuid4(), track_id=None,
                              sample={"monotonic_ms": timestamp, **pose})


def test_head_durations_are_independent_and_roll_never_means_turn():
    evaluator = HeadRuleEvaluator({"head_turn": {"minimum_duration_seconds": 1.0},
                                   "look_down": {"minimum_duration_seconds": 2.0, "fast_pitch_degrees": 40}})
    assert head_signals(evaluator, 0, roll=60) == []
    head_signals(evaluator, 200, yaw=45, pitch=30)
    head_signals(evaluator, 700, yaw=45, pitch=30)
    result = head_signals(evaluator, 1200, yaw=45, pitch=30)
    assert any(s["event_type"] is EventType.HEAD_TURN_RIGHT and s["active"] for s in result)
    assert not any(s["event_type"] is EventType.LOOK_DOWN and s["active"] for s in result)
    head_signals(evaluator, 1700, yaw=45, pitch=30)
    result = head_signals(evaluator, 2200, yaw=45, pitch=30)
    assert any(s["event_type"] is EventType.LOOK_DOWN and s["active"] for s in result)


def test_bad_face_or_long_gap_cannot_complete_head_candidate():
    evaluator = HeadRuleEvaluator({"min_duration_ms": 1000})
    head_signals(evaluator, 0, yaw=50)
    assert not any(s["active"] for s in head_signals(evaluator, 1000, yaw=50, quality=0))
    head_signals(evaluator, 1200, yaw=50)
    assert not any(s["active"] for s in head_signals(evaluator, 4000, yaw=50))


def test_abnormal_movement_requires_distinct_sustained_turns():
    evaluator = HeadRuleEvaluator({"min_duration_ms": 200,
                                   "abnormal_head_movement": {"minimum_turn_count": 2}})
    head_signals(evaluator, 0, yaw=65)
    result = head_signals(evaluator, 200, yaw=65)
    assert not any(s["event_type"] is EventType.ABNORMAL_HEAD_MOVEMENT for s in result)
    head_signals(evaluator, 400, yaw=0)
    head_signals(evaluator, 700, yaw=0)
    head_signals(evaluator, 800, yaw=-65)
    result = head_signals(evaluator, 1000, yaw=-65)
    assert any(s["event_type"] is EventType.ABNORMAL_HEAD_MOVEMENT and s["active"] for s in result)
    head_signals(evaluator, 1100, quality=0, calibration_status="collecting")
    assert not any(s["event_type"] is EventType.ABNORMAL_HEAD_MOVEMENT and s["active"]
                   for s in head_signals(evaluator, 1200, yaw=0))


def phone(confidence=0.6, x=0.4):
    return {"class": "cell phone", "confidence": confidence,
            "bbox": {"x": x, "y": 0.4, "width": 0.1, "height": 0.2}}


def test_phone_low_confidence_continues_only_a_confirmed_track():
    tracker = ObjectEvidenceTracker()
    assert tracker.update([phone(0.4)], 0)[0]["track_confirmed"] is False
    assert tracker.update([phone(0.6)], 200)[0]["track_confirmed"] is False
    assert tracker.update([phone(0.6)], 400)[0]["track_confirmed"] is True
    assert tracker.update([phone(0.4)], 600)[0]["track_confirmed"] is True
    assert tracker.update([], 800) == []
    assert tracker.update([phone(0.6, x=0.1)], 1000)[0]["track_confirmed"] is False
    assert tracker.update([phone(0.6)], 2000)[0]["track_confirmed"] is False


def test_phone_confidence_must_be_revalidated():
    tracker = ObjectEvidenceTracker()
    tracker.update([phone()], 0)
    tracker.update([phone()], 200)
    for timestamp in range(400, 2001, 200):
        result = tracker.update([phone(0.4)], timestamp)
    assert result[0]["track_confirmed"] is False


def test_phone_candidate_survives_one_miss_but_never_activates_from_missing_boxes():
    tracker = ObjectEvidenceTracker()
    evaluator = ObjectRuleEvaluator(phone_confidence_threshold=0.55,
                                     phone_release_confidence=0.35, max_gap_ms=600)
    session_id = uuid4()

    def step(timestamp, detections):
        return evaluator.evaluate(session_id=session_id, track_id=None,
                                  objects=tracker.update(detections, timestamp), monotonic_ms=timestamp)

    assert step(0, [phone()]) == []
    assert not step(200, [phone()])[0]["active"]
    step(400, [phone(0.4)])
    assert step(600, []) == []
    step(800, [phone(0.4)])
    step(1000, [phone()])
    assert step(1200, []) == []
    assert step(1400, [phone()])[0]["active"]
    assert step(2001, [])[0]["active"] is False


def test_overlay_bridges_slow_inference_but_expires_and_does_not_mutate_observations():
    overlay = OverlayState()
    original = {"captured_at_monotonic_ms": 1000, "objects": [phone()], "faces": []}
    overlay.update(original)
    assert overlay.snapshot(1515)["objects"]  # Previous 250 ms limit hid these boxes.
    assert "observed_at_ms" not in original["objects"][0]
    overlay.update({"captured_at_monotonic_ms": 1200, "objects": [], "faces": []})
    assert overlay.snapshot(1600)["objects"][0]["held"]
    overlay.update({"captured_at_monotonic_ms": 1400, "objects": [], "faces": []})
    assert overlay.snapshot(1600)["objects"] == []
    assert overlay.snapshot(2201) is None


def test_runtime_closes_head_signal_when_face_disappears():
    from datetime import UTC, datetime

    from proctoring_core.events.models import ObservationBundle, ObservationHealth
    from proctoring_desktop.runtime import _RuleSet

    session = uuid4()
    rules = _RuleSet(session, {}, {"behavior": {"head_turn": {"minimum_duration_seconds": 0}}})
    bundle = ObservationBundle(frame_id=1, session_id=session, captured_at_utc=datetime.now(UTC),
                               captured_at_monotonic_ms=100, frame_size=(640, 480),
                               faces=[{"yaw": 45, "quality": 0.8}],
                               analyzer_health={"face": ObservationHealth("ready")})
    assert any(s["active"] and s["event_type"] is EventType.HEAD_TURN_RIGHT for s in rules.evaluate(bundle))
    bundle.frame_id = 2
    bundle.captured_at_monotonic_ms = 300
    bundle.faces = []
    bundle.analyzer_health["face"] = ObservationHealth("no_detection")
    assert any(not s["active"] and s["event_type"] is EventType.HEAD_TURN_RIGHT for s in rules.evaluate(bundle))


def test_desktop_calibration_and_delayed_overlay_without_camera(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from proctoring_desktop.window import MainWindow, _paint_overlay
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path)
    window._timer.stop()
    try:
        window._on_state({"type": "analyzer_health", "analyzers": {"face": {"status": "ready"}}})
        window._on_state({"type": "running"})
        assert window.calibrate_button.isEnabled()
        window._on_analysis({"frame_id": 1, "captured_at_monotonic_ms": 1000,
                             "objects": [phone()], "faces": [], "analyzer_health": {}})
        result = window._overlay.snapshot(1515)
        assert result is not None
        pixmap = QPixmap(640, 480)
        _paint_overlay(pixmap, result, 515)
        window._on_state({"type": "stopped"})
        assert not window.calibrate_button.isEnabled()
        assert app is not None
    finally:
        window.close()


def test_gpu_launch_rejects_cpu_only_torch_before_opening_camera(monkeypatch):
    import asyncio
    from pathlib import Path

    import torch
    from proctoring_desktop.runtime import DesktopRuntime, LatestPreview

    monkeypatch.setenv("PROCTORING_AI_DEVICE", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = DesktopRuntime(camera_index=0, repository_root=Path(__file__).resolve().parents[1],
                             preview=LatestPreview())
    with pytest.raises(RuntimeError, match="CUDA chưa khả dụng"):
        asyncio.run(runtime._run_async())
