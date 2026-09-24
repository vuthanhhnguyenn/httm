from __future__ import annotations

from uuid import uuid4

from proctoring_core.behavior.objects import ObjectRuleEvaluator


def test_mixed_behavior_policy_suppresses_allowed_material_and_phone_policy_controls_signal() -> None:
    objects = [{"class": "paper", "confidence": 0.95}, {"class": "cell phone", "confidence": 0.95}]
    strict = ObjectRuleEvaluator({"allow_scratch_paper": True, "allow_book": False, "allow_phone": False}, min_duration_ms=0)
    strict_signals = strict.evaluate(session_id=uuid4(), track_id=None, objects=objects, monotonic_ms=0)
    assert [signal["event_type"].value for signal in strict_signals] == ["PHONE_DETECTED"]
    permitted = ObjectRuleEvaluator({"allow_scratch_paper": True, "allow_book": False, "allow_phone": True}, min_duration_ms=0)
    assert permitted.evaluate(session_id=uuid4(), track_id=None, objects=objects, monotonic_ms=0) == []
