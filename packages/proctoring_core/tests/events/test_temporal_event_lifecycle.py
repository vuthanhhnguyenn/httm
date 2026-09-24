from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from proctoring_core.behavior.temporal_buffer import ConditionState, TemporalCondition, TemporalSample
from proctoring_core.events.aggregator import EventAggregator
from proctoring_core.types import EventType, Severity


def test_temporal_buffer_requires_sustained_monotonic_condition() -> None:
    condition = TemporalCondition(min_duration_ms=1000)
    assert condition.update(TemporalSample(0, True)) is ConditionState.CANDIDATE
    assert condition.update(TemporalSample(500, True)) is ConditionState.CANDIDATE
    assert condition.update(TemporalSample(1000, True)) is ConditionState.ACTIVE
    assert condition.update(TemporalSample(1100, False)) is ConditionState.INACTIVE


def test_aggregator_has_one_open_event_then_closes() -> None:
    aggregator = EventAggregator()
    session_id = uuid4()
    base = {"session_id": session_id, "track_id": None, "event_type": EventType.LOOK_DOWN, "severity": Severity.MEDIUM, "confidence": 0.8, "reason_codes": ["PITCH"], "active": True}
    start = datetime.now(UTC)
    opened = aggregator.process(base, occurred_at=start)
    updated = aggregator.process({**base, "metrics": {"pitch": 30}}, occurred_at=start + timedelta(seconds=1))
    closed = aggregator.process({**base, "active": False}, occurred_at=start + timedelta(seconds=2))
    assert opened is not None and opened.action == "opened"
    assert updated is not None and updated.action == "updated"
    assert closed is not None and closed.action == "closed"
    assert closed.event.duration_ms == 2000
