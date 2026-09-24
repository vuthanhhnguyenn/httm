from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from proctoring_core.events.models import SuspiciousEvent
from proctoring_core.risk.engine import RiskEngine
from proctoring_core.types import EventType, Severity


def event(session_id, event_type=EventType.PHONE_DETECTED, confidence=1.0, metrics=None):
    now = datetime.now(UTC)
    return SuspiciousEvent(session_id=session_id, event_type=event_type, severity=Severity.HIGH, confidence=confidence, started_at=now, reason_codes=["TEST"], metrics=metrics or {})


def test_risk_is_capped_and_decays() -> None:
    session_id = uuid4()
    engine = RiskEngine(decay_per_second=10)
    snapshot = engine.apply_event(event(session_id))
    assert 0 <= snapshot.score_after <= 100
    decayed = engine.decay(session_id, occurred_at=snapshot.occurred_at + timedelta(seconds=20))
    assert decayed.score_after < snapshot.score_after


def test_supporting_signal_has_low_cap_and_reason_codes() -> None:
    session_id = uuid4()
    engine = RiskEngine()
    snapshot = engine.apply_event(event(session_id, EventType.SUSPICIOUS_HAND_ACTIVITY, metrics={"risk_cap": 25}))
    assert snapshot.score_after <= 25
    assert "TEST" in snapshot.reason_codes
