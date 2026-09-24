"""One risk score per session, with separate non-decaying history."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from uuid import UUID

from ..events.correlation import CorrelationFinding
from ..events.models import SuspiciousEvent
from ..types import EventType, RiskLevel, ensure_utc
from .models import RiskSnapshot


class RiskEngine:
    DEFAULT_WEIGHTS: dict[EventType, float] = {
        EventType.PHONE_DETECTED: 28.0,
        EventType.MULTIPLE_PERSON_DETECTED: 32.0,
        EventType.PERSON_MISSING: 22.0,
        EventType.HEAD_TURN_LEFT: 10.0,
        EventType.HEAD_TURN_RIGHT: 10.0,
        EventType.LOOK_DOWN: 9.0,
        EventType.ABNORMAL_HEAD_MOVEMENT: 16.0,
        EventType.DOCUMENT_DETECTED: 12.0,
        EventType.LEAVING_SEAT: 18.0,
        EventType.CAMERA_BLOCKED: 26.0,
        EventType.FACE_NOT_VISIBLE: 16.0,
        EventType.SUSPICIOUS_HAND_ACTIVITY: 5.0,
        EventType.POSSIBLE_TALKING: 4.0,
    }

    def __init__(self, *, weights: dict[EventType, float] | None = None, decay_per_second: float = 1.5,
                 active_event_ttl_seconds: float = 2.0,
                 recovery_half_life_seconds: float | None = None) -> None:
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.decay_per_second = max(0.0, decay_per_second)
        self.active_event_ttl_seconds = max(0.1, active_event_ttl_seconds)
        # None preserves linear decay for callers that explicitly use the old policy.
        self.recovery_half_life_seconds = (
            max(0.1, recovery_half_life_seconds)
            if recovery_half_life_seconds is not None else None
        )
        self._scores: dict[UUID, float] = defaultdict(float)
        self._last_at: dict[UUID, datetime] = {}
        self._counts: Counter[tuple[UUID, EventType]] = Counter()
        self._contributions: dict[UUID, float] = {}
        self._history_points: dict[UUID, float] = defaultdict(float)
        self._peak_scores: dict[UUID, float] = defaultdict(float)
        self._event_counts: Counter[UUID] = Counter()
        self._active_contributions: dict[UUID, dict[UUID, tuple[float, datetime]]] = defaultdict(dict)
        self._applied_combinations: set[tuple[UUID, str, frozenset[UUID]]] = set()
        self._correlation_points: dict[UUID, float] = defaultdict(float)

    @staticmethod
    def level_for(score: float) -> RiskLevel:
        if score >= 80:
            return RiskLevel.CRITICAL
        if score >= 60:
            return RiskLevel.HIGH
        if score >= 30:
            return RiskLevel.MEDIUM
        if score > 0:
            return RiskLevel.LOW
        return RiskLevel.NORMAL

    def _active_floor(self, session_id: UUID, now: datetime) -> float:
        active = self._active_contributions[session_id]
        for event_id, (_, seen_at) in tuple(active.items()):
            if (now - seen_at).total_seconds() > self.active_event_ttl_seconds:
                del active[event_id]
        return max((amount for amount, seen_at in active.values()
                    if 0 <= (now - seen_at).total_seconds() <= self.active_event_ttl_seconds),
                   default=0.0)

    def _decay(self, session_id: UUID, now: datetime) -> float:
        previous = self._last_at.get(session_id)
        score = self._scores.get(session_id, 0.0)
        if previous is not None:
            elapsed = max(0.0, (now - previous).total_seconds())
            if self.recovery_half_life_seconds is None:
                score = max(0.0, score - elapsed * self.decay_per_second)
            elif now > previous:
                score = self._recover(session_id, score, previous, now)
        score = max(score, self._active_floor(session_id, now))
        self._scores[session_id] = score
        self._last_at[session_id] = max(previous, now) if previous is not None else now
        return score

    def _recover(self, session_id: UUID, score: float, start: datetime, end: datetime) -> float:
        """Integrate active/idle intervals so recovery does not depend on AI FPS.

        Keep each live floor until it expires. Only the interval with no fresh
        active event uses exponential recovery; a delayed frame must not apply
        that faster decay retroactively to an active interval.
        """
        assert self.recovery_half_life_seconds is not None
        active = [
            (amount, seen_at, seen_at + timedelta(seconds=self.active_event_ttl_seconds))
            for amount, seen_at in self._active_contributions[session_id].values()
        ]
        boundaries = sorted({start, end} | {
            boundary for _, seen_at, expires_at in active
            for boundary in (seen_at, expires_at) if start < boundary < end
        })
        for left, right in zip(boundaries, boundaries[1:]):
            elapsed = (right - left).total_seconds()
            live = [amount for amount, seen_at, expires_at in active
                    if seen_at <= left < expires_at]
            if live:
                score = max(max(live), score - elapsed * self.decay_per_second)
            else:
                score *= 2 ** (-elapsed / self.recovery_half_life_seconds)
                if score < 0.5:
                    score = 0.0
        return score

    def _snapshot(self, session_id: UUID, before: float, after: float, reason: str,
                  now: datetime, *, track_id: UUID | None = None,
                  event_ids: tuple[UUID, ...] = (), reason_codes: tuple[str, ...] = ()) -> RiskSnapshot:
        self._peak_scores[session_id] = max(self._peak_scores[session_id], after)
        return RiskSnapshot(session_id, before, after, self.level_for(after), reason, event_ids,
                            reason_codes, now, track_id=track_id,
                            cumulative_score=self._history_points[session_id],
                            peak_score=self._peak_scores[session_id],
                            event_count=self._event_counts[session_id])

    def apply_event(self, event: SuspiciousEvent, *, occurred_at: datetime | None = None) -> RiskSnapshot:
        now = ensure_utc(occurred_at or event.updated_at or datetime.now(UTC))
        session_id = event.session_id
        if event.status == "CLOSED" and self.recovery_half_life_seconds is None:
            self._active_contributions[session_id].pop(event.id, None)
        before = self._decay(session_id, now)
        event_type = EventType(event.event_type)
        if event.status == "CLOSED":
            # Closing releases the floor from now onward, not before this frame.
            self._active_contributions[session_id].pop(event.id, None)
            return self._snapshot(session_id, before, before, "EVENT_UPDATE", now,
                                  track_id=event.track_id, event_ids=(event.id,), reason_codes=("EVENT_CLOSED",))
        new_event = event.id not in self._contributions
        if new_event:
            self._counts[(session_id, event_type)] += 1
            self._event_counts[session_id] += 1
        duration_factor = 1.0 + min(1.0, (event.duration_ms or 0) / 10_000)
        repetition_factor = 1.0 + min(0.5, (self._counts[(session_id, event_type)] - 1) * 0.1)
        raw = self.weights.get(event_type, 0.0) * max(0.1, event.confidence) * duration_factor * repetition_factor
        cap = float(event.metrics.get("risk_cap", 100.0)) if isinstance(event.metrics, dict) else 100.0
        contribution = min(raw, cap)
        previous_contribution = self._contributions.get(event.id, 0.0)
        delta = max(0.0, contribution - previous_contribution)
        self._contributions[event.id] = max(previous_contribution, contribution)
        self._history_points[session_id] += delta
        self._active_contributions[session_id][event.id] = (self._contributions[event.id], now)
        after = min(100.0, max(before + delta, self._active_floor(session_id, now)))
        self._scores[session_id] = after
        return self._snapshot(session_id, before, after, "EVENT_OPEN" if new_event else "EVENT_UPDATE",
                              now, track_id=event.track_id, event_ids=(event.id,),
                              reason_codes=tuple(event.reason_codes))

    def apply_correlation(self, finding: CorrelationFinding) -> RiskSnapshot:
        """Score one independently confirmed combination, linked to its source events."""
        now = ensure_utc(finding.occurred_at)
        session_id = finding.session_id
        before = self._decay(session_id, now)
        combination = (session_id, finding.rule_code, frozenset(finding.source_event_ids))
        if combination in self._applied_combinations:
            return self._snapshot(session_id, before, before, "CORRELATION_UPDATE", now,
                                  track_id=finding.track_id,
                                  event_ids=finding.source_event_ids)
        self._applied_combinations.add(combination)
        points = max(0.0, finding.points)
        self._correlation_points[session_id] += points
        self._history_points[session_id] += points
        after = min(100.0, before + points)
        self._scores[session_id] = after
        return self._snapshot(session_id, before, after, "CORRELATION", now,
                              track_id=finding.track_id, event_ids=finding.source_event_ids,
                              reason_codes=(f"CORRELATION_{finding.rule_code}",))

    def decay(self, session_id: UUID, *, track_id: UUID | None = None, occurred_at: datetime | None = None, elapsed_ms: int | None = None) -> RiskSnapshot:
        previous_at = self._last_at.get(session_id)
        if occurred_at is not None:
            now = ensure_utc(occurred_at)
        elif elapsed_ms is not None and previous_at is not None:
            now = previous_at + timedelta(milliseconds=max(0, elapsed_ms))
        else:
            now = datetime.now(UTC)
        before = self._scores.get(session_id, 0.0)
        after = self._decay(session_id, now)
        return self._snapshot(session_id, before, after, "DECAY", now,
                              track_id=track_id, reason_codes=("DECAY",))

    def score(self, session_id: UUID, track_id: UUID | None = None) -> float:
        return self._scores.get(session_id, 0.0)

    def history(self, session_id: UUID) -> dict[str, object]:
        """Non-decaying counters for review; never used as a live risk score."""
        return {"event_count": self._event_counts[session_id],
                "cumulative_score": self._history_points[session_id],
                "peak_score": self._peak_scores[session_id],
                "correlation_points": self._correlation_points[session_id],
                "events_by_type": {kind.value: count for (session, kind), count in self._counts.items()
                                   if session == session_id}}

    update = apply_event
