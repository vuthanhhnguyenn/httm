from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest


def test_operator_marks_false_positive_from_event_detail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from proctoring_core.events.review_models import ReviewType
    from proctoring_desktop.window import MainWindow
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path)
    window._timer.stop()
    session_id, event_id = uuid4(), str(uuid4())
    try:
        window._on_state({"type": "running", "session_id": str(session_id)})
        window._on_event({"id": event_id, "episode_id": event_id, "action": "opened", "type": "PHONE_DETECTED",
                          "severity": "HIGH", "confidence": 0.71, "exam_mode": "computer",
                          "occurred_at": datetime.now(UTC).isoformat(), "risk_score": 21, "risk_level": "LOW",
                          "object_trace": {"frame_id": 42, "label": "cell phone", "confidence": 0.71,
                                           "bbox": {"x": 0.5, "y": 0.6, "width": 0.1, "height": 0.1},
                                           "model_name": "yolo11n", "model_version": "8.3.0"}})
        window._open_evidence(window.events.item(0))
        dialog = window._evidence_dialogs[0]
        texts = [label.text() for label in dialog.findChildren(type(window.app_status))]
        assert "Thi máy" in texts and "yolo11n 8.3.0" in texts and "42" in texts

        dialog.review_buttons[ReviewType.FALSE_POSITIVE].click()  # type: ignore[attr-defined]

        events = tmp_path / "storage" / "sessions" / str(session_id) / "events.jsonl"
        review = json.loads(events.read_text(encoding="utf-8").splitlines()[-1])
        assert review["action"] == "review" and review["verdict"] == "false_positive"
        assert review["event_id"] == event_id
        assert "đã rà soát: Báo nhầm" in window.events.item(0).text()
        assert dialog.status.text() == "Đã rà soát: Báo nhầm"  # type: ignore[attr-defined]
    finally:
        window.close()
    assert app is not None
