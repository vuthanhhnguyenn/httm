"""Decide, per frame, which seats get face inference and which get hands/pose.

Every seat keeps at least ``1000 / min_face_interval_ms`` face analyses per second.
When the previous frame overran its budget, stable seats are rotated and the
expensive hands/pose focus is skipped; new and suspicious seats always go first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from proctoring_core.types import EventType

FOCUS_EVENT_TYPES: frozenset[EventType] = frozenset({
    EventType.HEAD_TURN_LEFT, EventType.HEAD_TURN_RIGHT, EventType.LOOK_DOWN,
    EventType.PHONE_DETECTED, EventType.LOOK_AT_NEIGHBOR,
})


@dataclass(frozen=True, slots=True)
class FramePlan:
    face_seats: frozenset[UUID]
    force_face: frozenset[UUID]
    focus_seats: tuple[UUID, ...]
    run_focus: bool


class FocusScheduler:
    def __init__(self, *, max_focus: int = 2, focus_hold_ms: int = 5000, frame_budget_ms: float = 110.0,
                 min_face_interval_ms: int = 250) -> None:
        self.max_focus = max_focus
        self.focus_hold_ms = focus_hold_ms
        self.frame_budget_ms = frame_budget_ms
        self.min_face_interval_ms = min_face_interval_ms
        self._marked: dict[UUID, int] = {}
        self._last_face: dict[UUID, int] = {}

    def mark(self, seat_id: UUID, monotonic_ms: int) -> None:
        self._marked[seat_id] = monotonic_ms

    def record_face(self, seat_id: UUID, monotonic_ms: int) -> None:
        self._last_face[seat_id] = monotonic_ms

    def forget(self, seat_id: UUID) -> None:
        self._marked.pop(seat_id, None)
        self._last_face.pop(seat_id, None)

    def plan(self, seat_ids: Sequence[UUID], monotonic_ms: int, *, last_frame_ms: float | None) -> FramePlan:
        present = set(seat_ids)
        recent = [(at, seat) for seat, at in self._marked.items()
                  if seat in present and 0 <= monotonic_ms - at < self.focus_hold_ms]
        focus = tuple(seat for _, seat in sorted(recent, key=lambda item: -item[0])[:self.max_focus])
        new = {seat for seat in present if seat not in self._last_face}
        force = frozenset(new | set(focus))
        if last_frame_ms is None or last_frame_ms <= self.frame_budget_ms:
            return FramePlan(frozenset(present), force, focus, True)
        due = {seat for seat in present
               if seat in self._last_face and monotonic_ms - self._last_face[seat] >= self.min_face_interval_ms}
        return FramePlan(frozenset(due | force), force, focus, False)


__all__ = ["FOCUS_EVENT_TYPES", "FocusScheduler", "FramePlan"]
