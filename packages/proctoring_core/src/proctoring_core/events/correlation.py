"""Confirm bounded, traceable combinations of already-confirmed events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from ..types import EventType
from .aggregator import EventTransition
from .models import ObservationBundle, SuspiciousEvent


@dataclass(frozen=True, slots=True)
class CorrelationFinding:
    id: UUID
    session_id: UUID
    rule_code: str
    source_event_ids: tuple[UUID, UUID]
    points: float
    occurred_at: datetime
    track_id: UUID | None
    confidence: float


@dataclass(slots=True)
class _ActiveSource:
    event: SuspiciousEvent
    opened_ms: int
    last_seen_ms: int


class CorrelationEngine:
    """Use fresh event episodes; never infer a combination from raw frame detections."""

    RULES = (
        ("PHONE_MULTIPLE_PERSON", frozenset({EventType.PHONE_DETECTED}),
         frozenset({EventType.MULTIPLE_PERSON_DETECTED}), 10.0, False),
        ("HEAD_AND_HAND", frozenset({EventType.HEAD_TURN_LEFT, EventType.HEAD_TURN_RIGHT}),
         frozenset({EventType.SUSPICIOUS_HAND_ACTIVITY}), 4.0, True),
        ("LOOK_DOWN_AND_HAND", frozenset({EventType.LOOK_DOWN}),
         frozenset({EventType.SUSPICIOUS_HAND_ACTIVITY}), 4.0, True),
    )
    SOURCES = {
        EventType.PHONE_DETECTED: "object_detector",
        EventType.MULTIPLE_PERSON_DETECTED: "object_detector",
        EventType.HEAD_TURN_LEFT: "face",
        EventType.HEAD_TURN_RIGHT: "face",
        EventType.LOOK_DOWN: "face",
        EventType.SUSPICIOUS_HAND_ACTIVITY: "hands",
    }

    def __init__(self, *, window_seconds: float = 5.0, confirmation_seconds: float = 0.25,
                 freshness_seconds: float = 2.0, session_cap: float = 20.0) -> None:
        self.window_ms = max(1, round(window_seconds * 1000))
        self.confirmation_ms = max(0, round(confirmation_seconds * 1000))
        self.freshness_ms = max(1, round(freshness_seconds * 1000))
        self.session_cap = max(0.0, session_cap)
        self._active: dict[UUID, dict[UUID, _ActiveSource]] = {}
        self._pending: dict[tuple[UUID, str, UUID, UUID], int] = {}
        self._awarded: set[tuple[UUID, str, UUID, UUID]] = set()
        self._last_award_ms: dict[tuple[UUID, str], int] = {}
        self._awarded_points: dict[UUID, float] = {}

    def ingest(self, bundle: ObservationBundle,
               transitions: list[EventTransition]) -> list[CorrelationFinding]:
        session_id = bundle.session_id
        now_ms = bundle.captured_at_monotonic_ms
        active = self._active.setdefault(session_id, {})
        for transition in transitions:
            event = transition.event
            if event.session_id != session_id:
                continue
            if transition.action == "closed" or event.status == "CLOSED":
                active.pop(event.id, None)
                continue
            source = self.SOURCES.get(event.event_type)
            health = bundle.analyzer_health.get(source) if source else None
            if health is None or health.status != "ready":
                continue
            previous = active.get(event.id)
            opened_ms = previous.opened_ms if previous else now_ms
            active[event.id] = _ActiveSource(event, opened_ms, now_ms)

        for event_id, source in tuple(active.items()):
            if now_ms - source.last_seen_ms > self.freshness_ms:
                del active[event_id]

        findings: list[CorrelationFinding] = []
        valid_pairs: set[tuple[UUID, str, UUID, UUID]] = set()
        remaining = max(0.0, self.session_cap - self._awarded_points.get(session_id, 0.0))
        for code, left_types, right_types, points, same_track in self.RULES:
            left = (item for item in active.values() if item.event.event_type in left_types)
            right = [item for item in active.values() if item.event.event_type in right_types]
            for first in left:
                for second in right:
                    key = (session_id, code, first.event.id, second.event.id)
                    if not self._compatible(bundle, first, second, same_track):
                        continue
                    valid_pairs.add(key)
                    if key in self._awarded:
                        continue
                    started_ms = self._pending.setdefault(key, now_ms)
                    last = self._last_award_ms.get((session_id, code))
                    if (now_ms - started_ms < self.confirmation_ms or remaining <= 0
                            or (last is not None and now_ms - last < self.window_ms)):
                        continue
                    amount = min(points, remaining)
                    finding = CorrelationFinding(
                        id=uuid4(), session_id=session_id, rule_code=code,
                        source_event_ids=(first.event.id, second.event.id), points=amount,
                        occurred_at=bundle.captured_at_utc,
                        track_id=first.event.track_id if same_track else None,
                        confidence=min(first.event.confidence, second.event.confidence),
                    )
                    findings.append(finding)
                    self._awarded.add(key)
                    self._last_award_ms[(session_id, code)] = now_ms
                    self._awarded_points[session_id] = self._awarded_points.get(session_id, 0.0) + amount
                    remaining -= amount
        for key in tuple(self._pending):
            if key[0] == session_id and key not in valid_pairs:
                del self._pending[key]
        return findings

    def _compatible(self, bundle: ObservationBundle, first: _ActiveSource,
                    second: _ActiveSource, same_track: bool) -> bool:
        now_ms = bundle.captured_at_monotonic_ms
        if (abs(first.opened_ms - second.opened_ms) > self.window_ms
                or any(now_ms - source.last_seen_ms > self.freshness_ms
                       for source in (first, second))):
            return False
        for source in (first, second):
            name = self.SOURCES[source.event.event_type]
            health = bundle.analyzer_health.get(name)
            # A scheduled skip may reuse a recent confirmed event, but an error may not.
            if health is None or health.status not in {"ready", "skipped"}:
                return False
        if same_track:
            return (first.event.track_id is not None
                    and first.event.track_id == second.event.track_id)
        return True
