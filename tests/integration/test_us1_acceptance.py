from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from proctoring_core.events.aggregator import EventAggregator
from proctoring_core.risk.engine import RiskEngine
from proctoring_core.types import EventType, Severity


@pytest.mark.integration
def test_us1_acceptance_signal_has_explanation_and_decay() -> None:
    session_id = uuid4()
    aggregator = EventAggregator()
    now = datetime.now(UTC)
    transition = aggregator.process({"session_id": session_id, "event_type": EventType.HEAD_TURN_RIGHT, "severity": Severity.MEDIUM, "confidence": 0.8, "reason_codes": ["YAW_THRESHOLD"], "metrics": {"yaw": 40}, "active": True}, occurred_at=now)
    assert transition is not None
    assert transition.event.explanation
    engine = RiskEngine(decay_per_second=5)
    snapshot = engine.apply_event(transition.event, occurred_at=now)
    decayed = engine.decay(session_id, occurred_at=now + timedelta(seconds=10))
    assert 0 <= snapshot.score_after <= 100
    assert decayed.score_after < snapshot.score_after
