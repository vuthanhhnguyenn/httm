from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from proctoring_core.events.aggregator import EventTransition
from proctoring_core.events.correlation import CorrelationEngine
from proctoring_core.events.models import ObservationBundle, ObservationHealth, SuspiciousEvent
from proctoring_core.risk.engine import RiskEngine
from proctoring_core.types import EventType, Severity


def event(session, kind, at, track=None):
    return SuspiciousEvent(session_id=session, event_type=kind, track_id=track,
                           severity=Severity.MEDIUM, confidence=1, started_at=at,
                           reason_codes=["TEST"])


def bundle(session, at, ms, *, detector="ready", face="ready", hands="ready"):
    return ObservationBundle(frame_id=ms + 1, session_id=session,
                             captured_at_utc=at + timedelta(milliseconds=ms),
                             captured_at_monotonic_ms=ms, frame_size=(80, 60),
                             analyzer_health={"object_detector": ObservationHealth(detector),
                                              "face": ObservationHealth(face),
                                              "hands": ObservationHealth(hands)})


def opened(*events):
    return [EventTransition("opened", item) for item in events]


def test_confirmed_pair_has_source_ids_and_scores_only_once():
    session, at, track = uuid4(), datetime.now(UTC), uuid4()
    hand = event(session, EventType.SUSPICIOUS_HAND_ACTIVITY, at, track)
    head = event(session, EventType.HEAD_TURN_LEFT, at, track)
    correlate, risk = CorrelationEngine(), RiskEngine(decay_per_second=0)
    risk.apply_event(hand, occurred_at=at)
    risk.apply_event(head, occurred_at=at)
    assert correlate.ingest(bundle(session, at, 0), opened(hand)) == []
    assert correlate.ingest(bundle(session, at, 100), opened(head)) == []
    assert correlate.ingest(bundle(session, at, 300, hands="skipped"), []) == []
    findings = correlate.ingest(bundle(session, at, 350, hands="skipped"), [])
    assert len(findings) == 1
    assert findings[0].source_event_ids == (head.id, hand.id)
    assert findings[0].track_id == track
    snapshot = risk.apply_correlation(findings[0])
    assert snapshot.score_after == 19
    assert snapshot.contributing_event_ids == (head.id, hand.id)
    assert risk.apply_correlation(findings[0]).score_after == 19
    assert risk.apply_correlation(replace(findings[0], id=uuid4())).score_after == 19
    assert correlate.ingest(bundle(session, at, 500), []) == []
    assert risk.history(session)["correlation_points"] == 4
    assert risk.history(session)["event_count"] == 2


def test_same_track_is_required_and_unhealthy_source_resets_confirmation():
    session, at = uuid4(), datetime.now(UTC)
    hand = event(session, EventType.SUSPICIOUS_HAND_ACTIVITY, at, uuid4())
    head = event(session, EventType.HEAD_TURN_RIGHT, at, uuid4())
    engine = CorrelationEngine(confirmation_seconds=0.25)
    assert engine.ingest(bundle(session, at, 0), opened(hand, head)) == []
    assert engine.ingest(bundle(session, at, 300), []) == []
    head.track_id = hand.track_id
    assert engine.ingest(bundle(session, at, 400), [EventTransition("updated", head)]) == []
    assert engine.ingest(bundle(session, at, 600, hands="timeout"), []) == []
    assert engine.ingest(bundle(session, at, 700), []) == []
    findings = engine.ingest(bundle(session, at, 950), [])
    assert len(findings) == 1


def test_stale_closed_and_out_of_window_episodes_do_not_correlate():
    session, at, track = uuid4(), datetime.now(UTC), uuid4()
    hand = event(session, EventType.SUSPICIOUS_HAND_ACTIVITY, at, track)
    head = event(session, EventType.HEAD_TURN_LEFT, at, track)
    stale = CorrelationEngine(window_seconds=5, freshness_seconds=1, confirmation_seconds=0)
    assert stale.ingest(bundle(session, at, 0), opened(hand)) == []
    assert stale.ingest(bundle(session, at, 1500), opened(head)) == []
    assert stale.ingest(bundle(session, at, 1800), []) == []
    fresh = CorrelationEngine(window_seconds=5, confirmation_seconds=0)
    other_hand = event(session, EventType.SUSPICIOUS_HAND_ACTIVITY, at, track)
    assert fresh.ingest(bundle(session, at, 0), opened(other_hand)) == []
    other_hand.close(at + timedelta(milliseconds=100))
    assert fresh.ingest(bundle(session, at, 100), [EventTransition("closed", other_hand)]) == []
    assert fresh.ingest(bundle(session, at, 200), opened(head)) == []
    delayed = CorrelationEngine(window_seconds=5, freshness_seconds=10, confirmation_seconds=0)
    another_hand = event(session, EventType.SUSPICIOUS_HAND_ACTIVITY, at, track)
    assert delayed.ingest(bundle(session, at, 0), opened(another_hand)) == []
    assert delayed.ingest(bundle(session, at, 6000), opened(head)) == []


def test_session_cap_and_cooldown_bound_independent_pairs():
    session, at = uuid4(), datetime.now(UTC)
    engine = CorrelationEngine(confirmation_seconds=0, session_cap=12,
                               window_seconds=5, freshness_seconds=2)
    phone = event(session, EventType.PHONE_DETECTED, at)
    people = event(session, EventType.MULTIPLE_PERSON_DETECTED, at)
    first = engine.ingest(bundle(session, at, 0), opened(phone, people))
    assert len(first) == 1 and first[0].points == 10
    track = uuid4()
    head = event(session, EventType.HEAD_TURN_LEFT, at, track)
    hand = event(session, EventType.SUSPICIOUS_HAND_ACTIVITY, at, track)
    second = engine.ingest(bundle(session, at, 100), opened(head, hand))
    assert len(second) == 1 and second[0].points == 2
    down = event(session, EventType.LOOK_DOWN, at, track)
    assert engine.ingest(bundle(session, at, 200), opened(down)) == []
    assert engine.ingest(bundle(session, at, 300), []) == []
    assert sum(item.points for item in first + second) == pytest.approx(12)


def test_phone_and_multiple_people_is_session_scoped_only():
    session, other, at = uuid4(), uuid4(), datetime.now(UTC)
    engine = CorrelationEngine(confirmation_seconds=0)
    phone = event(session, EventType.PHONE_DETECTED, at, uuid4())
    people = event(session, EventType.MULTIPLE_PERSON_DETECTED, at)
    alien = event(other, EventType.MULTIPLE_PERSON_DETECTED, at)
    assert engine.ingest(bundle(session, at, 0), opened(phone, alien)) == []
    findings = engine.ingest(bundle(session, at, 100), opened(people))
    assert len(findings) == 1
    assert findings[0].track_id is None
