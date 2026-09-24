from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest
from proctoring_core.events.aggregator import EventAggregator
from proctoring_core.events.models import ObservationBundle, ObservationHealth
from proctoring_core.risk.engine import RiskEngine
from proctoring_desktop.evidence import EvidenceRecorder
from proctoring_desktop.runtime import DesktopRuntime, LatestPreview, _RuleSet


def sample(session_id, *, image=None, frame_id=1, timestamp=100):
    pixels = np.full((64, 80, 3), (20, 80, 230), dtype=np.uint8) if image is None else image
    frame = {"frame_id": frame_id, "captured_at_monotonic_ms": timestamp, "image": pixels}
    bundle = ObservationBundle(frame_id=frame_id, session_id=session_id,
                               captured_at_utc=datetime.now(UTC), captured_at_monotonic_ms=timestamp,
                               frame_size=(80, 64))
    return frame, bundle


@pytest.mark.asyncio
async def test_evidence_records_exact_analyzed_frame_and_expires_only_indexed_jpeg(tmp_path):
    session_id, event_id = uuid4(), uuid4()
    notifications = []
    recorder = EvidenceRecorder(tmp_path / "storage", session_id, enabled=True, retention_days=2,
                                on_result=notifications.append)
    frame, bundle = sample(session_id)
    original = frame["image"].copy()
    recorder.start()
    assert recorder.enqueue(event_id, frame, bundle) == "pending"
    frame["image"][:] = 0  # Camera buffer changes after the event.
    await recorder.close()
    result = notifications[0]
    assert result["status"] == "available"
    path = tmp_path / "storage" / result["relative_path"]
    encoded = path.read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == result["sha256"]
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == original.shape
    assert np.mean(np.abs(decoded.astype(float) - original.astype(float))) < 3
    assert np.mean(decoded) > 50  # The mutated black frame was not stored.
    assert result["captured_at"] == bundle.captured_at_utc.isoformat()
    keep = path.parent / "keep.jpg"
    keep.write_bytes(b"not indexed")
    assert EvidenceRecorder.purge_expired(tmp_path / "storage", now=bundle.captured_at_utc + timedelta(days=3)) == 1
    assert not path.exists() and keep.exists()
    assert EvidenceRecorder.purge_expired(tmp_path / "storage", now=bundle.captured_at_utc + timedelta(days=4)) == 0


@pytest.mark.asyncio
async def test_disabled_mismatch_and_full_queue_never_save_wrong_frame(tmp_path):
    session = uuid4()
    frame, bundle = sample(session)
    disabled = EvidenceRecorder(tmp_path / "storage", session, enabled=False, retention_days=1,
                                on_result=lambda _result: None)
    assert disabled.enqueue(uuid4(), frame, bundle) == "disabled"
    await disabled.close()
    recorder = EvidenceRecorder(tmp_path / "storage", session, enabled=True, retention_days=1,
                                on_result=lambda _result: None, queue_size=1)
    mismatch = {**frame, "frame_id": 2}
    assert recorder.enqueue(uuid4(), mismatch, bundle) == "frame_mismatch"
    assert recorder.enqueue(uuid4(), frame, bundle) == "pending"
    assert recorder.enqueue(uuid4(), frame, bundle) == "queue_full"
    recorder.start()
    await recorder.close()
    assert len(list((tmp_path / "storage" / "sessions" / str(session) / "evidence").glob("*.jpg"))) == 1


@pytest.mark.asyncio
async def test_runtime_event_and_background_evidence_share_id_and_frame(tmp_path):
    runtime = DesktopRuntime(camera_index=0, repository_root=tmp_path, preview=LatestPreview())
    runtime._event_path = tmp_path / "storage" / "sessions" / str(runtime.session_id) / "events.jsonl"
    runtime._event_path.parent.mkdir(parents=True)
    session = SimpleNamespace(id=runtime.session_id, current_risk_score=0,
                              update_risk=lambda score, level: None)
    rules = _RuleSet(runtime.session_id, {"allow_phone": False},
                     {"behavior": {"phone": {"minimum_duration_seconds": 0}}})
    frame, bundle = sample(runtime.session_id)
    bundle.objects = [{"class": "cell phone", "confidence": 0.9,
                       "track_confirmed": True, "object_track_id": 1}]
    bundle.analyzer_health = {"object_detector": ObservationHealth("ready")}
    recorder = EvidenceRecorder(tmp_path / "storage", runtime.session_id, enabled=True,
                                retention_days=3, on_result=runtime._on_evidence_result)
    recorder.start()
    await runtime._process_rules(bundle, rules, EventAggregator(), RiskEngine(), session, frame, recorder)
    await recorder.close()
    records = [json.loads(line) for line in runtime._event_path.read_text(encoding="utf-8").splitlines()]
    opened = next(item for item in records if item.get("action") == "opened")
    evidence = next(item for item in records if item.get("action") == "evidence")
    assert opened["id"] == evidence["event_id"]
    assert opened["evidence_status"] == "pending"
    assert evidence["status"] == "available"
    assert opened["cumulative_score"] > 0


def test_evidence_event_becomes_openable_in_vietnamese_ui(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from proctoring_desktop.window import MainWindow
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path)
    window._timer.stop()
    event_id = str(uuid4())
    try:
        window._on_event({"id": event_id, "action": "opened", "type": "PHONE_DETECTED",
                          "severity": "HIGH", "occurred_at": datetime.now(UTC).isoformat(),
                          "risk_score": 28, "risk_level": "LOW", "evidence_status": "pending"})
        assert "Đang lưu ảnh" in window.events.item(0).toolTip()
        relative = f"sessions/{uuid4()}/evidence/{event_id}.jpg"
        path = tmp_path / "storage" / relative
        path.parent.mkdir(parents=True)
        cv2.imwrite(str(path), np.full((40, 40, 3), 180, dtype=np.uint8))
        window._on_event({"action": "evidence", "event_id": event_id, "status": "available",
                          "relative_path": relative})
        assert window.events.item(0).data(Qt.ItemDataRole.UserRole) == relative
        window._open_evidence(window.events.item(0))
        assert len(window._evidence_dialogs) == 1
        assert app is not None
    finally:
        window.close()


def test_risk_summary_persists_live_and_session_history_separately(tmp_path):
    runtime = DesktopRuntime(camera_index=0, repository_root=tmp_path, preview=LatestPreview())
    runtime._event_path = tmp_path / "storage" / "sessions" / str(runtime.session_id) / "events.jsonl"
    runtime._event_path.parent.mkdir(parents=True)
    now = datetime.now(UTC)
    engine = RiskEngine(decay_per_second=1.5)
    from proctoring_core.events.models import SuspiciousEvent
    from proctoring_core.types import EventType, Severity

    event = SuspiciousEvent(session_id=runtime.session_id, event_type=EventType.PHONE_DETECTED,
                            severity=Severity.HIGH, confidence=1, started_at=now, reason_codes=["TEST"])
    engine.apply_event(event, occurred_at=now)
    engine.decay(runtime.session_id, occurred_at=now + timedelta(seconds=20))
    runtime._write_risk_summary(engine, runtime.session_id)
    summary = json.loads((runtime._event_path.parent / "risk-summary.json").read_text(encoding="utf-8"))
    assert summary["current_score"] == 0
    assert summary["cumulative_score"] == 28
    assert summary["peak_score"] == 28
    assert summary["event_count"] == 1


def test_retention_ignores_index_record_pointing_outside_evidence_directory(tmp_path):
    root = tmp_path / "storage"
    session, event_id = uuid4(), uuid4()
    evidence_dir = root / "sessions" / str(session) / "evidence"
    evidence_dir.mkdir(parents=True)
    outside = root / f"{event_id}.jpg"
    outside.write_bytes(b"must survive")
    (evidence_dir / "index.jsonl").write_text(
        json.dumps({"event_id": str(event_id), "status": "available",
                    "relative_path": outside.relative_to(root).as_posix(),
                    "retention_until": (datetime.now(UTC) - timedelta(days=1)).isoformat()}) + "\n",
        encoding="utf-8",
    )
    assert EvidenceRecorder.purge_expired(root) == 0
    assert outside.read_bytes() == b"must survive"
