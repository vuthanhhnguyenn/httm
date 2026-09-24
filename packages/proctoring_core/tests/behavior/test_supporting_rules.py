from __future__ import annotations

from uuid import uuid4

from proctoring_core.behavior.safety import SafetyRuleEvaluator
from proctoring_core.behavior.supporting import SupportingRuleEvaluator
from proctoring_core.types import EventType, Severity


def test_supporting_rules_are_low_severity() -> None:
    evaluator = SupportingRuleEvaluator(min_duration_ms=100)
    session_id = uuid4()
    result = evaluator.evaluate(session_id=session_id, track_id=None, sample={"hand_activity": True, "monotonic_ms": 100})
    assert result[0]["event_type"] is EventType.SUSPICIOUS_HAND_ACTIVITY
    assert result[0]["severity"] is Severity.LOW
    assert result[0]["risk_cap"] == 25


def test_camera_blocked_is_sustained() -> None:
    evaluator = SafetyRuleEvaluator(min_duration_ms=100)
    session_id = uuid4()
    first = evaluator.evaluate(session_id=session_id, track_id=None, sample={"monotonic_ms": 0, "image_quality": {"blur": 0}})
    assert any(item["active"] is False for item in first)
    second = evaluator.evaluate(session_id=session_id, track_id=None, sample={"monotonic_ms": 100, "image_quality": {"blur": 0}})
    assert any(item["event_type"] is EventType.CAMERA_BLOCKED and item["active"] for item in second)
