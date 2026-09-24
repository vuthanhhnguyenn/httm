from __future__ import annotations

from typing import Any

from ..types import EventType, Severity
from .temporal_buffer import ConditionState, TemporalCondition, TemporalSample


class PresenceRuleEvaluator:
    def __init__(self, *, min_duration_ms: int = 1000) -> None:
        self._conditions = {
            EventType.PERSON_MISSING: TemporalCondition(min_duration_ms=min_duration_ms),
            EventType.MULTIPLE_PERSON_DETECTED: TemporalCondition(min_duration_ms=min_duration_ms),
        }

    def evaluate(self, *, session_id: Any, people: list[dict[str, Any]], monotonic_ms: int) -> list[dict[str, Any]]:
        checks = ((EventType.PERSON_MISSING, len(people) == 0), (EventType.MULTIPLE_PERSON_DETECTED, len(people) > 1))
        results: list[dict[str, Any]] = []
        for event_type, matched in checks:
            condition = self._conditions[event_type]
            state = condition.update(TemporalSample(monotonic_ms, matched, 1.0))
            if state is not ConditionState.INACTIVE or condition.last_closed:
                results.append({"session_id": session_id, "track_id": None, "event_type": event_type, "severity": Severity.HIGH if state is ConditionState.ACTIVE else Severity.LOW, "confidence": 1.0, "active": state is ConditionState.ACTIVE, "metrics": {"people_count": len(people), **condition.current_metrics()}, "reason_codes": [f"{event_type.value}_COUNT"]})
        return results
