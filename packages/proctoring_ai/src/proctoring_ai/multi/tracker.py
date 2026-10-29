"""Seat-based tracking of several seated candidates.

A seat is the stable identity of one candidate: YOLO person boxes are matched to
seats by IoU, and a box that matches none is re-linked to a recently missing seat
whose anchor (smoothed box centre) is close, so a candidate who briefly leaves the
view or bends down keeps their id instead of being split into several tracks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot
from typing import Any, Protocol
from uuid import UUID, uuid4

from proctoring_core.observations import Box


@dataclass(slots=True)
class Seat:
    seat_id: UUID
    runtime_track_id: int
    bbox: Box
    anchor: tuple[float, float]
    confidence: float
    first_seen_ms: int
    last_seen_ms: int
    visible: bool


@dataclass(frozen=True, slots=True)
class TrackerUpdate:
    analyzed: list[Seat]
    over_limit: list[Seat]
    missing: list[Seat]
    expired: list[Seat]
    created: int
    relinked: int


class IdAssigner(Protocol):
    def assign(self, seats: Sequence[Seat], boxes: Sequence[Box]) -> dict[int, UUID]: ...


def iou(a: Box, b: Box) -> float:
    width = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
    height = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


class IouAssigner:
    """Greedy matching by decreasing IoU; enough for a few mostly still candidates."""

    def __init__(self, min_iou: float = 0.3) -> None:
        self.min_iou = min_iou

    def assign(self, seats: Sequence[Seat], boxes: Sequence[Box]) -> dict[int, UUID]:
        pairs = sorted(((iou(seat.bbox, box), index, seat.seat_id)
                        for index, box in enumerate(boxes) for seat in seats), key=lambda item: -item[0])
        matched: dict[int, UUID] = {}
        used: set[UUID] = set()
        for score, index, seat_id in pairs:
            if score < self.min_iou:
                break
            if index in matched or seat_id in used:
                continue
            matched[index] = seat_id
            used.add(seat_id)
        return matched


class MultiSubjectTracker:
    def __init__(self, *, max_subjects: int = 6, relink_radius: float = 0.5, relink_window_ms: int = 10_000,
                 anchor_alpha: float = 0.1, assigner: IdAssigner | None = None) -> None:
        self.max_subjects = max_subjects
        self.relink_radius = relink_radius
        self.relink_window_ms = relink_window_ms
        self.anchor_alpha = anchor_alpha
        self.assigner: IdAssigner = assigner or IouAssigner()
        self.seats: dict[UUID, Seat] = {}
        self._next_runtime_id = 1

    def _relink(self, box: Box, candidates: list[Seat]) -> Seat | None:
        cx, cy = box.center
        best: Seat | None = None
        best_distance = 0.0
        for seat in candidates:
            distance = hypot(seat.anchor[0] - cx, seat.anchor[1] - cy)
            if distance < self.relink_radius * seat.bbox.width and (best is None or distance < best_distance):
                best, best_distance = seat, distance
        return best

    def _observe(self, seat: Seat, box: Box, confidence: float, monotonic_ms: int) -> None:
        cx, cy = box.center
        alpha = self.anchor_alpha
        seat.anchor = (seat.anchor[0] + alpha * (cx - seat.anchor[0]), seat.anchor[1] + alpha * (cy - seat.anchor[1]))
        seat.bbox, seat.confidence = box, confidence
        seat.last_seen_ms, seat.visible = monotonic_ms, True

    def update(self, people: Sequence[Mapping[str, Any]], monotonic_ms: int,
               frame_size: tuple[int, int] | None = None) -> TrackerUpdate:
        detections = [(box, float(item.get("confidence", 0.0)))
                      for item in people if (box := Box.parse(item.get("bbox"))) is not None]
        boxes = [box for box, _ in detections]
        was_visible = {seat_id: seat.visible for seat_id, seat in self.seats.items()}
        matched = self.assigner.assign(list(self.seats.values()), boxes)
        used = set(matched.values())
        created = relinked = 0
        for index, (box, confidence) in enumerate(detections):
            seat_id = matched.get(index)
            if seat_id is None:
                candidates = [seat for seat in self.seats.values() if not seat.visible and seat.seat_id not in used]
                relink = self._relink(box, candidates)
                if relink is not None:
                    seat_id = relink.seat_id
                else:
                    seat = Seat(uuid4(), self._next_runtime_id, box, box.center, confidence,
                                monotonic_ms, monotonic_ms, True)
                    self._next_runtime_id += 1
                    self.seats[seat.seat_id] = seat
                    used.add(seat.seat_id)
                    created += 1
                    continue
            used.add(seat_id)
            seat = self.seats[seat_id]
            if not was_visible.get(seat_id, True):
                relinked += 1
            self._observe(seat, box, confidence, monotonic_ms)
        expired: list[Seat] = []
        for seat in list(self.seats.values()):
            if seat.seat_id in used:
                continue
            seat.visible = False
            if monotonic_ms - seat.last_seen_ms >= self.relink_window_ms:
                expired.append(self.seats.pop(seat.seat_id))
        visible = [seat for seat in self.seats.values() if seat.visible]
        largest = sorted(visible, key=lambda seat: -seat.bbox.width * seat.bbox.height)[:self.max_subjects]
        kept = {seat.seat_id for seat in largest}
        return TrackerUpdate(
            analyzed=[seat for seat in visible if seat.seat_id in kept],
            over_limit=[seat for seat in visible if seat.seat_id not in kept],
            missing=[seat for seat in self.seats.values() if not seat.visible],
            expired=expired,
            created=created,
            relinked=relinked,
        )


__all__ = ["IdAssigner", "IouAssigner", "MultiSubjectTracker", "Seat", "TrackerUpdate", "iou"]
