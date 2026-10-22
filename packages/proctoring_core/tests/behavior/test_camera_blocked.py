from __future__ import annotations

from typing import Any
from uuid import uuid4

from proctoring_core.behavior.safety import SafetyRuleEvaluator
from proctoring_core.types import EventType

NORMAL = {"blur": 150.0, "brightness": 0.5, "contrast": 45.0, "edge_density": 0.08}


def blocked_states(quality: dict[str, Any], *, person_present: bool) -> list[bool]:
    rules = SafetyRuleEvaluator(min_duration_ms=500, max_brightness=0.06, min_contrast=6.0,
                                min_edge_density=0.002, require_no_person=True)
    session = uuid4()
    states = []
    for timestamp in range(0, 1500, 125):
        for item in rules.evaluate(session_id=session, track_id=None, sample={
                "monotonic_ms": timestamp, "image_quality": quality, "face_visible": True,
                "person_present": person_present}):
            if item["event_type"] is EventType.CAMERA_BLOCKED:
                states.append(item["active"])
    return states


def test_each_low_information_feature_flags_a_covered_lens() -> None:
    for change in ({"brightness": 0.02}, {"contrast": 3.0}, {"edge_density": 0.0005}, {"blur": 4.0}):
        assert any(blocked_states({**NORMAL, **change}, person_present=False)), change


def test_normal_picture_and_visible_person_are_not_blocked() -> None:
    assert not any(blocked_states(NORMAL, person_present=False))
    # A dark room where the detector still sees the candidate is low light, not a covered lens.
    assert not any(blocked_states({**NORMAL, "brightness": 0.03}, person_present=True))


def test_missing_features_are_unknown_not_blocked() -> None:
    assert not any(blocked_states({"blur": 150.0}, person_present=False))
