from __future__ import annotations

from typing import Any

from ..types import EventType, Severity
from .temporal_buffer import ConditionState, TemporalCondition, TemporalSample


class SupportingRuleEvaluator:
    """Low-impact supporting signals; neither signal can independently become HIGH risk."""

    def __init__(self, *, min_duration_ms: int = 1200) -> None:
        self._conditions = {event: TemporalCondition(min_duration_ms=min_duration_ms) for event in (EventType.SUSPICIOUS_HAND_ACTIVITY, EventType.POSSIBLE_TALKING)}

    def evaluate(self, *, session_id: Any, track_id: Any, sample: dict[str, Any]) -> list[dict[str, Any]]:
        timestamp = int(sample.get("monotonic_ms", 0))
        checks = ((EventType.SUSPICIOUS_HAND_ACTIVITY, bool(sample.get("hands_near_face") or sample.get("hand_activity"))), (EventType.POSSIBLE_TALKING, bool(sample.get("mouth_motion") or sample.get("talking"))))
        results: list[dict[str, Any]] = []
        for event_type, matched in checks:
            condition = self._conditions[event_type]
            state = condition.update(TemporalSample(timestamp, matched, 0.6))
            if state is not ConditionState.INACTIVE or condition.last_closed:
                results.append({"session_id": session_id, "track_id": track_id, "event_type": event_type, "severity": Severity.LOW, "confidence": 0.6, "active": state is ConditionState.ACTIVE, "risk_cap": 25.0, "metrics": condition.current_metrics(), "reason_codes": [f"{event_type.value}_SUPPORTING"]})
        return results
