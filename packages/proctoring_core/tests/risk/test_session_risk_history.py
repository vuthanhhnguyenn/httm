from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from proctoring_core.events.models import SuspiciousEvent
from proctoring_core.risk.engine import RiskEngine
from proctoring_core.types import EventType, Severity


def event(session, kind, *, track=None, confidence=1.0, at=None):
    return SuspiciousEvent(session_id=session, track_id=track, event_type=kind,
                           severity=Severity.MEDIUM, confidence=confidence,
                           started_at=at or datetime.now(UTC), reason_codes=["TEST"])


def test_same_event_updates_do_not_inflate_repetition_or_history():
    session = uuid4()
    now = datetime.now(UTC)
    engine = RiskEngine(decay_per_second=0)
    phone = event(session, EventType.PHONE_DETECTED, at=now)
    first = engine.apply_event(phone, occurred_at=now)
    for _ in range(20):
        updated = engine.apply_event(phone, occurred_at=now)
    assert updated.score_after == first.score_after == 28
    assert updated.cumulative_score == 28
    assert updated.event_count == 1
    assert engine.history(session)["events_by_type"] == {"PHONE_DETECTED": 1}


def test_session_scope_decay_and_non_decaying_history():
    session, track = uuid4(), uuid4()
    now = datetime.now(UTC)
    engine = RiskEngine(decay_per_second=1.5)
    opened = engine.apply_event(event(session, EventType.PHONE_DETECTED, track=track, at=now), occurred_at=now)
    after = engine.decay(session, occurred_at=now + timedelta(seconds=20))
    assert opened.score_after == 28
    assert after.score_after == 0
    assert engine.score(session, track) == engine.score(session) == 0
    assert after.cumulative_score == 28
    assert after.peak_score == 28
    assert after.event_count == 1


def test_separate_events_across_tracks_count_as_history_once_each():
    session = uuid4()
    now = datetime.now(UTC)
    engine = RiskEngine(decay_per_second=0)
    first = event(session, EventType.HEAD_TURN_LEFT, track=uuid4(), at=now)
    second = event(session, EventType.HEAD_TURN_LEFT, track=uuid4(), at=now)
    engine.apply_event(first, occurred_at=now)
    result = engine.apply_event(second, occurred_at=now)
    assert result.score_after == pytest.approx(21)
    assert result.event_count == 2
    assert engine.history(session)["events_by_type"] == {"HEAD_TURN_LEFT": 2}
    second.close(now)
    closed = engine.apply_event(second, occurred_at=now)
    assert closed.score_after == result.score_after
    assert closed.event_count == 2


def test_configurable_decay_rate_changes_live_score_not_history():
    session = uuid4()
    now = datetime.now(UTC)
    engine = RiskEngine(decay_per_second=0.3)
    engine.apply_event(event(session, EventType.PHONE_DETECTED, at=now), occurred_at=now)
    after = engine.decay(session, occurred_at=now + timedelta(seconds=20))
    assert after.score_after == pytest.approx(22)
    assert after.cumulative_score == 28


def test_repeated_active_event_holds_floor_without_recharging_history():
    session = uuid4()
    now = datetime.now(UTC)
    engine = RiskEngine(decay_per_second=1.5)
    phone = event(session, EventType.PHONE_DETECTED, at=now)
    engine.apply_event(phone, occurred_at=now)
    for seconds in range(1, 31):
        engine.apply_event(phone, occurred_at=now + timedelta(seconds=seconds))
    assert engine.score(session) == 28
    assert engine.history(session)["cumulative_score"] == 28


def test_active_floor_expires_after_missing_updates_and_closed_event_can_decay():
    session = uuid4()
    now = datetime.now(UTC)
    engine = RiskEngine(decay_per_second=1.5, active_event_ttl_seconds=2)
    phone = event(session, EventType.PHONE_DETECTED, at=now)
    engine.apply_event(phone, occurred_at=now)
    assert engine.decay(session, occurred_at=now + timedelta(seconds=1)).score_after == 28
    assert engine.decay(session, occurred_at=now + timedelta(seconds=4)).score_after == 23.5
    phone.close(now + timedelta(seconds=5))
    engine.apply_event(phone, occurred_at=now + timedelta(seconds=5))
    assert engine.decay(session, occurred_at=now + timedelta(seconds=7)).score_after == 19
    assert engine.history(session)["cumulative_score"] == 28


def test_out_of_order_timestamp_cannot_extend_next_decay_window():
    session = uuid4()
    now = datetime.now(UTC)
    engine = RiskEngine(decay_per_second=1)
    engine.apply_event(event(session, EventType.PHONE_DETECTED, at=now), occurred_at=now)
    engine.decay(session, occurred_at=now + timedelta(seconds=10))
    engine.decay(session, occurred_at=now + timedelta(seconds=5))
    later = engine.decay(session, occurred_at=now + timedelta(seconds=20))
    assert later.score_after == 8


def test_fast_recovery_halves_idle_score_and_preserves_session_history():
    session = uuid4()
    now = datetime.now(UTC)
    engine = RiskEngine(recovery_half_life_seconds=1)
    phone = event(session, EventType.PHONE_DETECTED, at=now)
    engine.apply_event(phone, occurred_at=now)
    phone.close(now)
    engine.apply_event(phone, occurred_at=now)
    for seconds, expected in ((0.25, 28 * 2 ** -0.25), (1, 14), (2, 7), (6, 0)):
        snapshot = engine.decay(session, occurred_at=now + timedelta(seconds=seconds))
        assert snapshot.score_after == pytest.approx(expected)
        assert snapshot.cumulative_score == snapshot.peak_score == 28
        assert snapshot.event_count == 1


@pytest.mark.parametrize("step_seconds", [0.05, 0.25, 1, 4])
def test_fast_recovery_waits_for_ttl_and_is_independent_of_ai_fps(step_seconds):
    session = uuid4()
    now = datetime.now(UTC)
    engine = RiskEngine(recovery_half_life_seconds=1, active_event_ttl_seconds=2)
    engine.apply_event(event(session, EventType.PHONE_DETECTED, at=now), occurred_at=now)
    for step in range(1, round(4 / step_seconds) + 1):
        seconds = step * step_seconds
        snapshot = engine.decay(session, occurred_at=now + timedelta(seconds=seconds))
        if seconds <= 2:
            assert snapshot.score_after == 28
    # Two seconds protected by TTL, then two seconds of half-life recovery.
    assert snapshot.score_after == pytest.approx(7)


def test_fast_recovery_starts_at_close_not_before_and_keeps_other_active_floor():
    session = uuid4()
    now = datetime.now(UTC)
    engine = RiskEngine(recovery_half_life_seconds=1)
    phone = event(session, EventType.PHONE_DETECTED, at=now)
    face = event(session, EventType.FACE_NOT_VISIBLE, at=now)
    engine.apply_event(phone, occurred_at=now)
    engine.apply_event(face, occurred_at=now)
    phone.close(now + timedelta(seconds=1))
    closed = engine.apply_event(phone, occurred_at=now + timedelta(seconds=1))
    assert closed.score_after == pytest.approx(42.5)
    for seconds in range(2, 31):
        engine.apply_event(face, occurred_at=now + timedelta(seconds=seconds))
    assert engine.score(session) == 16
    face.close(now + timedelta(seconds=30))
    engine.apply_event(face, occurred_at=now + timedelta(seconds=30))
    assert engine.decay(session, occurred_at=now + timedelta(seconds=31)).score_after == 8
    assert engine.history(session)["cumulative_score"] == 44


def test_fast_recovery_ignores_duplicate_and_older_decay_timestamps():
    session = uuid4()
    now = datetime.now(UTC)
    engine = RiskEngine(recovery_half_life_seconds=1)
    phone = event(session, EventType.PHONE_DETECTED, at=now)
    engine.apply_event(phone, occurred_at=now)
    phone.close(now)
    engine.apply_event(phone, occurred_at=now)
    engine.decay(session, occurred_at=now + timedelta(seconds=1))
    engine.decay(session, occurred_at=now + timedelta(seconds=1))
    engine.decay(session, occurred_at=now)
    assert engine.decay(session, occurred_at=now + timedelta(seconds=2)).score_after == 7
