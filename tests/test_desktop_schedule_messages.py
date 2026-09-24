from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from proctoring_ai.pipeline.observation_pipeline import ObservationPipeline
from proctoring_core.config.schema import DetectionConfig
from proctoring_core.events.models import ObservationBundle, ObservationHealth
from proctoring_core.types import EventType
from proctoring_desktop.messages import EVENT_NAMES, event_description, health_reason, level_name
from proctoring_desktop.overlay import OverlayState
from proctoring_desktop.runtime import _RuleSet


@pytest.mark.asyncio
async def test_scheduler_keeps_face_and_objects_fresh_without_replaying_hand_evidence():
    detector = type("Detector", (), {"detect": AsyncMock(return_value=[])})()
    tracker = type("Tracker", (), {"update": AsyncMock(return_value=[])})()
    face = type("Face", (), {"analyze": AsyncMock(return_value={"status": "ready", "faces": [{"yaw": 35}]})})()
    pose = type("Pose", (), {"analyze": AsyncMock(return_value={"status": "no_detection"})})()
    hands = type("Hands", (), {"analyze": AsyncMock(return_value={"status": "ready", "hands": [{"activity": True}]})})()
    pipeline = ObservationPipeline(detector=detector, tracker=tracker, face_analyzer=face,
                                   pose_analyzer=pose, hand_analyzer=hands)
    session = uuid4()
    results = []
    for index, timestamp in enumerate((0, 50, 100, 200, 250)):
        results.append(await pipeline.analyze({"frame_id": index, "captured_at_monotonic_ms": timestamp}, session_id=session))
    assert detector.detect.await_count == face.analyze.await_count == 5
    assert pose.analyze.await_count == hands.analyze.await_count == 2
    assert results[1].analyzer_health["hands"].status == "skipped"
    assert not results[1].hands and results[1].faces
    assert results[3].hands
    assert pipeline.performance()["hands"]["observed_at_ms"] == 200
    # A failed analyzer must be retried, not masked by a scheduled skip.
    hands.analyze.side_effect = RuntimeError("failure")
    result = await pipeline.analyze({"frame_id": 5, "captured_at_monotonic_ms": 400}, session_id=session)
    assert result.analyzer_health["hands"].status == "error"
    result = await pipeline.analyze({"frame_id": 6, "captured_at_monotonic_ms": 450}, session_id=session)
    assert result.analyzer_health["hands"].status == "error"
    assert hands.analyze.await_count == 4


def test_scheduled_hand_skips_cannot_activate_a_rule():
    session = uuid4()
    rules = _RuleSet(session, {}, {"behavior": {"hand_activity": {"minimum_duration_seconds": 1}}})
    bundle = ObservationBundle(frame_id=0, session_id=session, captured_at_utc=datetime.now(UTC),
                               captured_at_monotonic_ms=0, frame_size=(640, 480),
                               hands=[{"activity": True}], analyzer_health={"hands": ObservationHealth("ready")})
    assert not any(s["active"] for s in rules.evaluate(bundle))
    bundle.captured_at_monotonic_ms = 1000
    bundle.hands = []
    bundle.analyzer_health["hands"] = ObservationHealth("skipped")
    assert not any(s["event_type"] is EventType.SUSPICIOUS_HAND_ACTIVITY for s in rules.evaluate(bundle))
    bundle.captured_at_monotonic_ms = 1200
    bundle.hands = [{"activity": True}]
    bundle.analyzer_health["hands"] = ObservationHealth("ready")
    assert any(s["active"] for s in rules.evaluate(bundle))


@pytest.mark.asyncio
async def test_measured_fps_counts_only_inference_and_expires(monkeypatch):
    import proctoring_ai.pipeline.observation_pipeline as module

    now = [1.0]
    monkeypatch.setattr(module, "monotonic", lambda: now[0])
    detector = type("Detector", (), {"detect": AsyncMock(return_value=[])})()
    tracker = type("Tracker", (), {"update": AsyncMock(return_value=[])})()
    hands = type("Hands", (), {"analyze": AsyncMock(return_value={"status": "no_detection"})})()
    pipeline = ObservationPipeline(detector=detector, tracker=tracker, hand_analyzer=hands)
    session = uuid4()
    for index, timestamp in enumerate((0, 100, 200)):
        now[0] = 1 + timestamp / 1000
        await pipeline.analyze({"frame_id": index, "captured_at_monotonic_ms": timestamp}, session_id=session)
    measured = pipeline.performance()
    assert measured["object_detector"]["fps"] == pytest.approx(10)
    assert measured["hands"]["fps"] == pytest.approx(5)
    assert measured["face"]["status"] == "not_run"
    now[0] = 4
    assert pipeline.performance()["hands"]["fps"] == 0


def test_scheduled_overlay_expires_from_original_observation_not_latest_frame():
    overlay = OverlayState()
    original = {"captured_at_monotonic_ms": 1000, "hands": [{"bbox": {}}], "poses": [{"landmarks": []}]}
    overlay.update(original)
    for timestamp in (1200, 1300, 1400):
        overlay.update({"captured_at_monotonic_ms": timestamp, "analyzer_health": {"hands": "skipped", "pose": "skipped"}})
        result = overlay.snapshot(timestamp)
        assert bool(result["hands"]) == (timestamp <= 1350)
        assert bool(result["poses"]) == (timestamp <= 1350)
    assert "held" not in original["hands"][0]
    overlay.update(original)
    overlay.update({"captured_at_monotonic_ms": 1100, "analyzer_health": {"hands": "no_detection", "pose": "error"}})
    assert not overlay.snapshot(1100)["hands"]
    assert not overlay.snapshot(1100)["poses"]


def test_all_event_codes_have_vietnamese_labels_and_schedule_is_bounded():
    assert set(EVENT_NAMES) == {item.value for item in EventType}
    assert level_name("NORMAL") == "Bình thường"
    assert "mã kiểm tra" in health_reason("ValueError:model SHA-256 does not match")
    assert "điện thoại" in event_description({"type": "PHONE_DETECTED", "confidence": 0.8})
    with pytest.raises(ValueError):
        DetectionConfig(hands_interval_ms=10000)


def test_window_displays_vietnamese_events_and_measured_analyzer_fps(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from proctoring_desktop.window import MainWindow
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path)
    window._timer.stop()
    try:
        window._on_event({"action": "opened", "type": "PHONE_DETECTED", "severity": "HIGH",
                          "confidence": 0.85, "risk_level": "MEDIUM", "risk_score": 35,
                          "occurred_at": "2026-09-24T10:20:30+00:00", "explanation": "English internal reason"})
        assert "Phát hiện điện thoại" in window.events.item(0).text()
        assert "Cao" in window.events.item(0).text()
        assert "85%" in window.events.item(0).toolTip()
        assert "English" not in window.events.item(0).toolTip()
        assert "Trung bình" in window.risk_value.text()
        window._on_analysis({"captured_at_monotonic_ms": 100, "analyzer_health": {"hands": "skipped"},
                             "analyzer_performance": {"hands": {"status": "ready", "fps": 4.2, "latency_ms": 30}}})
        assert window.health_fps_labels["hands"].text() == "4.2"
        assert "CPU" in window.health_labels["hands"].text()
        window._on_error("Không thể tiếp tục giám sát")
        window._on_state({"type": "stopped"})
        assert window.app_status.text() == "Cần kiểm tra"
        assert app is not None
    finally:
        window.close()


def test_fps_cards_fit_minimum_window_width(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from proctoring_desktop.window import MainWindow
    from PySide6.QtWidgets import QApplication, QScrollArea

    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path)
    window._timer.stop()
    try:
        window.resize(1100, 700)
        window._metrics = {"processed_fps": 12.7, "latency_ms_p95": 86,
                           "dropped_frames": 1728, "realtime_status": "DEGRADED"}
        window._preview_fps = 27.2
        window._last_analyzers = {key: {"status": "ready", "device": "cpu"}
                                  for key in ("object_detector", "face", "pose", "hands")}
        window._analyzer_performance = {key: {"status": "ready", "fps": 12.7, "latency_ms": 30}
                                        for key in ("object_detector", "face", "pose", "hands")}
        window._refresh_metrics()
        window._refresh_health()
        window.show()
        app.processEvents()
        rail = window.findChild(QScrollArea, "railScroll")
        assert rail is not None
        assert rail.widget().width() <= rail.viewport().width()
        assert window.health_fps_labels["object_detector"].text() == "12.7"
        assert window.health_fps_labels["object_detector"].width() >= 48
    finally:
        window.close()


def test_runtime_error_keeps_technical_details_out_of_user_message(tmp_path, monkeypatch):
    from proctoring_desktop.runtime import DesktopRuntime, LatestPreview

    runtime = DesktopRuntime(camera_index=0, repository_root=tmp_path, preview=LatestPreview())
    monkeypatch.setattr(runtime, "_run_async", AsyncMock(side_effect=ValueError("internal English failure")))
    messages = []
    runtime.runtime_error.connect(messages.append)
    runtime.run()
    assert "Không thể tiếp tục giám sát" in messages[0]
    assert "internal English failure" not in messages[0]
    assert "internal English failure" in (tmp_path / "storage/runtime-errors.log").read_text(encoding="utf-8")
