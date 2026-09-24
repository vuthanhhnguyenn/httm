from __future__ import annotations

from uuid import uuid4

from proctoring_core.behavior.head import HeadRuleEvaluator
from proctoring_core.behavior.objects import ObjectRuleEvaluator
from proctoring_core.behavior.presence import PresenceRuleEvaluator
from proctoring_core.types import EventType


def test_head_turn_needs_duration() -> None:
    evaluator = HeadRuleEvaluator({"min_duration_ms": 1000, "head_turn": {"fast_yaw_degrees": 60}})
    session_id = uuid4()
    assert evaluator.evaluate(session_id=session_id, track_id=None, sample={"yaw": 50, "monotonic_ms": 0})[0]["active"] is False
    assert not any(item["active"] for item in evaluator.evaluate(
        session_id=session_id, track_id=None, sample={"yaw": 50, "monotonic_ms": 500}))
    active = evaluator.evaluate(session_id=session_id, track_id=None, sample={"yaw": 50, "monotonic_ms": 1000})
    assert any(item["event_type"] is EventType.HEAD_TURN_RIGHT and item["active"] for item in active)


def test_phone_and_presence_boundaries() -> None:
    session_id = uuid4()
    objects = ObjectRuleEvaluator({"allow_phone": False}, min_duration_ms=1000)
    assert objects.evaluate(session_id=session_id, track_id=None, objects=[{"class": "cell phone", "confidence": 0.9}], monotonic_ms=0)[0]["active"] is False
    assert objects.evaluate(session_id=session_id, track_id=None, objects=[{"class": "cell phone", "confidence": 0.9}], monotonic_ms=1000)[0]["event_type"] is EventType.PHONE_DETECTED
    presence = PresenceRuleEvaluator(min_duration_ms=1000)
    assert presence.evaluate(session_id=session_id, people=[], monotonic_ms=0)[0]["active"] is False
    assert presence.evaluate(session_id=session_id, people=[], monotonic_ms=1000)[0]["event_type"] is EventType.PERSON_MISSING
