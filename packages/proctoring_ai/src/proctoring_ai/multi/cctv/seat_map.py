"""Fixed-camera seat map: candidates are bound to learned seats, everyone else is an outsider.

A seat is where somebody's head stayed still for a few seconds while sitting. Seats are learned
throughout the video and never move or expire, so a seat's number is stable however often its
candidate is missed by the detector. People not on a seat (invigilators, anyone walking) are
outsiders and get no rules. Leaving the seat is a seat that was occupied and then stays empty.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import hypot
from statistics import median
from typing import Any
from uuid import UUID, uuid4

from proctoring_core.observations import Box

from ..tracker import Seat, TrackerUpdate
from .head import Keypoints, distance, head_point
from .roles import CANDIDATE, PENDING, WALKER, is_standing


def looks_standing(box: Box, kps: Keypoints | None, *, max_aspect: float = 2.2) -> bool:
    """Standing bodies are tall and narrow; seated ones are cut by the desk."""

    return box.height / max(box.width, 1e-6) > max_aspect or is_standing(kps)


@dataclass(slots=True)
class _MapSeat:
    seat: Seat
    head: tuple[float, float]
    width: float
    occupancy: deque[tuple[int, bool]] = field(default_factory=deque)
    last_occupied_ms: int = 0
    last_nearby_ms: int = 0


@dataclass(slots=True)
class _Pending:
    seat: Seat
    head: tuple[float, float]
    history: deque[tuple[int, float, float, float]] = field(default_factory=deque)
    quiet_since: int = 0
    standing_since: int | None = None
    moving: bool = False


class SeatMapTracker:
    def __init__(self, *, settle_ms: int = 3_000, settle_drift: float = 0.35, walk_drift: float = 1.0,
                 walk_window_ms: int = 2_000, standing_ms: int = 1_000, seat_radius: float = 0.9, nearby_radius: float = 1.5,
                 head_alpha: float = 0.05,
                 pending_ttl_ms: int = 1_000, leave_ms: int = 5_000, occupied_ratio: float = 0.8,
                 occupancy_window_ms: int = 30_000, max_subjects: int = 80) -> None:
        self.settle_ms, self.settle_drift = settle_ms, settle_drift
        self.walk_drift, self.walk_window_ms, self.standing_ms = walk_drift, walk_window_ms, standing_ms
        self.seat_radius, self.pending_ttl_ms = seat_radius, pending_ttl_ms
        self.nearby_radius, self.head_alpha = nearby_radius, head_alpha
        self.leave_ms, self.occupied_ratio, self.occupancy_window_ms = leave_ms, occupied_ratio, occupancy_window_ms
        self.max_subjects = max_subjects
        self.seats: dict[UUID, _MapSeat] = {}
        self._pending: dict[UUID, _Pending] = {}
        self._next_seat_id = 1
        self._next_outsider_id = 1000
        self.roles: dict[UUID, str] = {}

    def _promote(self, pending: _Pending, ms: int) -> bool:
        if len(self.seats) >= self.max_subjects or pending.moving or pending.standing_since is not None:
            return False
        window = [item for item in pending.history if ms - item[0] <= self.settle_ms]
        if not window or ms - pending.quiet_since < self.settle_ms or ms - window[0][0] < 0.8 * self.settle_ms:
            return False
        hx, hy = pending.head
        width = median(w for *_, w in window)
        if any(hypot(hx - x, hy - y) > self.settle_drift * width for _, x, y, _ in window):
            return False
        head = (median(x for _, x, _, _ in window), median(y for _, _, y, _ in window))
        if any(distance(head, other.head) < self.seat_radius * max(width, other.width)
               for other in self.seats.values()):
            return False
        box = pending.seat.bbox
        seat = Seat(uuid4(), self._next_seat_id, box, head, pending.seat.confidence, ms, ms, True)
        self._next_seat_id += 1
        self.seats[seat.seat_id] = _MapSeat(seat, head, width, deque([(ms, True)]), ms)
        return True

    def update(self, people: Sequence[Mapping[str, Any]], monotonic_ms: int) -> TrackerUpdate:
        ms = monotonic_ms
        observed = [(box, item.get("keypoints"), float(item.get("confidence", 0.0)))
                    for item in people if (box := Box.parse(item.get("bbox"))) is not None]
        heads = [head_point(kps, box) for box, kps, _ in observed]
        # Greedy nearest assignment of people to seats within the seat radius.
        pairs = sorted((distance(heads[i], seat.head), i, seat_id) for i in range(len(observed))
                       for seat_id, seat in self.seats.items()
                       if distance(heads[i], seat.head) < self.seat_radius * seat.width)
        assigned: dict[int, UUID] = {}
        taken: set[UUID] = set()
        for _, index, seat_id in pairs:
            if index in assigned or seat_id in taken:
                continue
            box, kps, _ = observed[index]
            if looks_standing(box, kps):
                continue  # standing up at the seat: the seat counts as empty, the person as an outsider
            assigned[index] = seat_id
            taken.add(seat_id)
        for seat_id, entry in self.seats.items():
            occupied = seat_id in taken
            entry.occupancy.append((ms, occupied))
            while entry.occupancy and ms - entry.occupancy[0][0] > self.occupancy_window_ms + self.leave_ms:
                entry.occupancy.popleft()
            entry.seat.visible = occupied
            if not occupied and any(distance(head, entry.head) < self.nearby_radius * entry.width for head in heads):
                # Someone is right there (leaning, half hidden, standing up): not an empty seat yet.
                entry.last_nearby_ms = ms
        for index, seat_id in assigned.items():
            box, _, confidence = observed[index]
            entry = self.seats[seat_id]
            entry.seat.bbox, entry.seat.confidence, entry.seat.last_seen_ms = box, confidence, ms
            entry.last_occupied_ms = ms
            hx, hy = heads[index]
            alpha = self.head_alpha
            entry.head = (entry.head[0] + alpha * (hx - entry.head[0]), entry.head[1] + alpha * (hy - entry.head[1]))

        # Everyone else: short-lived outsider tracks, which become seats once they sit still.
        unmatched = [i for i in range(len(observed)) if i not in assigned]
        live = [p for p in self._pending.values() if ms - p.seat.last_seen_ms <= self.pending_ttl_ms]
        used: set[UUID] = set()
        created = 0
        for index in sorted(unmatched, key=lambda i: -observed[i][2]):
            box, kps, confidence = observed[index]
            head = heads[index]
            near = min((p for p in live if p.seat.seat_id not in used
                        and distance(head, p.head) < self.walk_drift * box.width),
                       key=lambda p: distance(head, p.head), default=None)
            if near is None:
                seat = Seat(uuid4(), self._next_outsider_id, box, head, confidence, ms, ms, True)
                self._next_outsider_id += 1
                near = _Pending(seat, head, quiet_since=ms)
                self._pending[seat.seat_id] = near
            used.add(near.seat.seat_id)
            near.seat.bbox, near.seat.confidence, near.seat.last_seen_ms = box, confidence, ms
            near.head = head
            near.history.append((ms, head[0], head[1], box.width))
            while near.history and ms - near.history[0][0] > max(self.settle_ms, self.walk_window_ms):
                near.history.popleft()
            near.standing_since = (near.standing_since or ms) if looks_standing(box, kps) else None
            moved = any(hypot(head[0] - x, head[1] - y) > self.walk_drift * box.width
                        for t, x, y, _ in near.history if ms - t <= self.walk_window_ms)
            if moved:
                near.quiet_since = ms
            near.moving = moved
            if self._promote(near, ms):
                created += 1
                self._pending.pop(near.seat.seat_id)
                used.discard(near.seat.seat_id)
        for seat_id in [s for s, p in self._pending.items() if ms - p.seat.last_seen_ms > self.pending_ttl_ms]:
            self._pending.pop(seat_id)

        self.roles = {seat_id: CANDIDATE for seat_id in self.seats}
        for seat_id, pending in self._pending.items():
            walking = pending.moving or (pending.standing_since is not None
                                         and ms - pending.standing_since >= self.standing_ms)
            self.roles[seat_id] = WALKER if walking else PENDING
        visible_seats = [entry.seat for entry in self.seats.values() if entry.seat.visible]
        outsiders = [p.seat for p in self._pending.values() if p.seat.last_seen_ms == ms]
        return TrackerUpdate(analyzed=visible_seats + outsiders, over_limit=[],
                             missing=[entry.seat for entry in self.seats.values() if not entry.seat.visible],
                             expired=[], created=created, relinked=0)

    def left_seat(self, seat_id: UUID, monotonic_ms: int) -> bool:
        """Empty for ``leave_ms`` after being occupied most of the preceding window."""

        entry = self.seats.get(seat_id)
        if entry is None or monotonic_ms - max(entry.last_occupied_ms, entry.last_nearby_ms) < self.leave_ms:
            return False
        before = [occupied for t, occupied in entry.occupancy
                  if entry.last_occupied_ms - self.occupancy_window_ms <= t <= entry.last_occupied_ms]
        span = entry.last_occupied_ms - entry.seat.first_seen_ms
        return span >= 10_000 and bool(before) and sum(before) / len(before) >= self.occupied_ratio


class SeatMapRoles:
    """Role source for ``CctvSeatRules`` backed by the seat map instead of per-track motion."""

    def __init__(self, tracker: SeatMapTracker) -> None:
        self.tracker = tracker

    def update(self, seat_id: UUID, box: Box, monotonic_ms: int, kps: Keypoints | None = None) -> str:
        return self.tracker.roles.get(seat_id, PENDING)

    def left_seat(self, seat_id: UUID, monotonic_ms: int) -> bool:
        return self.tracker.left_seat(seat_id, monotonic_ms)

    def forget(self, seat_id: UUID) -> None:
        return None

    def counts(self) -> dict[str, int]:
        roles = list(self.tracker.roles.values())
        return {"seats": len(self.tracker.seats), "pending": roles.count(PENDING), "walker": roles.count(WALKER)}


__all__ = ["SeatMapRoles", "SeatMapTracker", "looks_standing"]
