from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from proctoring_core.events.models import SuspiciousEvent
from proctoring_core.risk.engine import RiskEngine
from proctoring_core.types import EventType, Severity


def event(session_id: UUID, event_type: EventType = EventType.PHONE_DETECTED,
          confidence: float = 1.0, metrics: dict[str, float] | None = None) -> SuspiciousEvent:
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


def test_supporting_signals_together_stay_below_high_band() -> None:
    from proctoring_core.types import RiskLevel

    engine = RiskEngine(supporting_group_cap=30, recovery_half_life_seconds=1.0)
    session, now = uuid4(), datetime.now(UTC)
    for event_type in (EventType.GAZE_OFF_SCREEN, EventType.GAZE_SIDEWAYS, EventType.POSSIBLE_TALKING,
                       EventType.SUSPICIOUS_HAND_ACTIVITY):
        event = SuspiciousEvent(session_id=session, event_type=event_type, severity=Severity.LOW, confidence=1.0,
                                started_at=now, reason_codes=["X"], metrics={"risk_cap": 25}, duration_ms=10_000)
        snapshot = engine.apply_event(event, occurred_at=now)
    assert snapshot.score_after <= 30
    assert engine.level_for(snapshot.score_after) in {RiskLevel.LOW, RiskLevel.MEDIUM}
    # A primary signal is not limited by the supporting group cap.
    phone = SuspiciousEvent(session_id=session, event_type=EventType.PHONE_DETECTED, severity=Severity.HIGH,
                            confidence=1.0, started_at=now, reason_codes=["X"], duration_ms=10_000)
    assert engine.apply_event(phone, occurred_at=now).score_after > 30
