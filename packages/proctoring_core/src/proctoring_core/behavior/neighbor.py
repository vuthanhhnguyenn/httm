"""Multi-subject rule: a calibrated head turned towards the candidate seated next to it.

Unlike the single-subject evaluators this one needs every seat's geometry at once,
so the multi-subject runner calls it per frame and routes each signal to its seat.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..observations import Box
from ..types import EventType, Severity
from .signals import RuleSignal, SignalState
from .temporal_buffer import ConditionState, TemporalCondition, TemporalSample


@dataclass(frozen=True, slots=True)
class SubjectHead:
    track_id: str
    box: Box
    yaw: float | None
    quality: float
    calibrated: bool


class NeighborLookEvaluator:
    def __init__(self, *, yaw_degrees: float, min_duration_ms: int = 1500, max_gap_widths: float = 2.5,
                 min_quality: float = 0.35, max_sample_gap_ms: int = 600, risk_cap: float = 25.0) -> None:
        self.yaw_degrees = yaw_degrees
        self.min_duration_ms = min_duration_ms
        self.max_gap_widths = max_gap_widths
        self.min_quality = min_quality
        self.max_sample_gap_ms = max_sample_gap_ms
        self.risk_cap = risk_cap
        self._conditions: dict[str, TemporalCondition] = {}
        self._sessions: dict[str, Any] = {}

    def _neighbor(self, subject: SubjectHead, others: Sequence[SubjectHead], direction: int) -> SubjectHead | None:
        cx, cy = subject.box.center
        best: SubjectHead | None = None
        best_dx = 0.0
        for other in others:
            ox, oy = other.box.center
            dx = (ox - cx) * direction
            if (dx <= 0 or abs(oy - cy) >= 0.5 * subject.box.height
                    or dx >= self.max_gap_widths * subject.box.width):
                continue
            if best is None or dx < best_dx:
                best, best_dx = other, dx
        return best

    def _signal(self, session_id: Any, track_id: str, state: SignalState, metrics: dict[str, Any],
                confidence: float) -> RuleSignal:
        return RuleSignal(EventType.LOOK_AT_NEIGHBOR, state, session_id, track_id, confidence, Severity.LOW,
                          ("HEAD_TOWARDS_NEIGHBOR",), {**metrics, "risk_cap": self.risk_cap},
                          rule_key=EventType.LOOK_AT_NEIGHBOR.value)

    def evaluate(self, *, session_id: Any, subjects: Sequence[SubjectHead], monotonic_ms: int) -> list[RuleSignal]:
        signals: list[RuleSignal] = []
        for subject in subjects:
            others = [item for item in subjects if item.track_id != subject.track_id]
            neighbor: SubjectHead | None = None
            usable = (subject.calibrated and subject.yaw is not None and subject.quality >= self.min_quality)
            if usable and subject.yaw is not None and abs(subject.yaw) > self.yaw_degrees:
                neighbor = self._neighbor(subject, others, 1 if subject.yaw > 0 else -1)
            condition = self._conditions.setdefault(subject.track_id, TemporalCondition(
                min_duration_ms=self.min_duration_ms, max_sample_gap_ms=self.max_sample_gap_ms))
            self._sessions[subject.track_id] = session_id
            quality = max(0.0, min(1.0, subject.quality))
            state = condition.update(TemporalSample(monotonic_ms, neighbor is not None, quality))
            if state is ConditionState.INACTIVE and not condition.last_closed:
                continue
            metrics: dict[str, Any] = {"yaw": round(subject.yaw or 0.0, 2), **condition.current_metrics()}
            if neighbor is not None:
                metrics["neighbor_track_id"] = neighbor.track_id
            signals.append(self._signal(session_id, subject.track_id, SignalState(state.value), metrics, quality))
        return signals

    def forget(self, track_id: str) -> list[RuleSignal]:
        condition = self._conditions.pop(track_id, None)
        session_id = self._sessions.pop(track_id, None)
        if condition is None or condition.state is ConditionState.INACTIVE:
            return []
        return [self._signal(session_id, track_id, SignalState.INACTIVE, {"duration_ms": condition.duration_ms}, 0.0)]


__all__ = ["NeighborLookEvaluator", "SubjectHead"]
