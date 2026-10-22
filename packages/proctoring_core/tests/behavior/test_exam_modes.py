from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from proctoring_core.behavior.head import HeadRuleEvaluator
from proctoring_core.config.schema import AppConfig, ValidationError
from proctoring_core.types import EventType, ExamMode

MODES = {
    "paper": {"look_down": {"enabled": False}},
    "computer": {"look_down": {"minimum_duration_seconds": 2.0, "fast_enabled": False}},
}


def _config(**extra: Any) -> AppConfig:
    return AppConfig.model_validate({"exam_modes": MODES, **extra})


def _look_down_events(mode: ExamMode, seconds: float, *, pitch: float = 35.0) -> list[dict[str, Any]]:
    behavior = _config().behavior_for(mode).model_dump()
    behavior["head_turn"]["require_calibration"] = False
    rules = HeadRuleEvaluator(behavior)
    session = uuid4()
    events: list[dict[str, Any]] = []
    for step in range(int(seconds * 8) + 1):
        events += rules.evaluate(session_id=session, track_id="t", sample={
            "monotonic_ms": 1 + step * 125, "yaw": 0.0, "pitch": pitch, "quality": 0.9, "pose_valid": True})
    return [item for item in events if item["event_type"] is EventType.LOOK_DOWN and item["active"]]


def test_paper_mode_never_flags_looking_down() -> None:
    assert _look_down_events(ExamMode.PAPER, 30) == []


def test_computer_mode_ignores_short_glance_and_flags_sustained_look_down() -> None:
    assert _look_down_events(ExamMode.COMPUTER, 1.5) == []
    sustained = _look_down_events(ExamMode.COMPUTER, 3.0)
    assert sustained
    assert sustained[0]["metrics"]["duration_ms"] >= 2000


def test_override_is_merged_and_revalidated() -> None:
    config = _config()
    assert config.behavior_for("computer").look_down.pitch_degrees == config.behavior.look_down.pitch_degrees
    assert config.behavior_for(ExamMode.PAPER).look_down.enabled is False
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"exam_modes": {"paper": {"look_down": {"pitch_degree": 3}}}})


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"exam_modes": {"oral": {}}})
