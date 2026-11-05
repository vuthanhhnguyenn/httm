"""Tell seated candidates from people walking around (invigilators, passers-by).

A track becomes a candidate only after its head has stayed put for a few seconds without standing;
a track whose head travels a box width within two seconds, or that stands, is a walker and gets no
rules until it sits still for longer. Tracks that never settle stay pending (no rules either).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from statistics import median
from uuid import UUID

from proctoring_core.observations import Box

from .head import Keypoints, head_point, point, shoulders

PENDING, CANDIDATE, WALKER = "pending", "candidate", "walker"
LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE = 11, 12, 13, 14


def is_standing(kps: Keypoints | None) -> bool:
    """Thighs hang below the hips (knee drop ≥ 0.6 torso) only when the person stands."""

    pair = shoulders(kps)
    hips = [p for index in (LEFT_HIP, RIGHT_HIP) if (p := point(kps, index, 0.5))]
    knees = [p for index in (LEFT_KNEE, RIGHT_KNEE) if (p := point(kps, index, 0.5))]
    if pair is None or not hips or not knees:
        return False
    shoulder_y = (pair[0][1] + pair[1][1]) / 2
    hip_y = sum(p[1] for p in hips) / len(hips)
    knee_y = sum(p[1] for p in knees) / len(knees)
    torso = hip_y - shoulder_y
    return torso > 0 and knee_y - hip_y >= 0.6 * torso


@dataclass(slots=True)
class _Track:
    first_ms: int
    history: deque[tuple[int, float, float, float]] = field(default_factory=deque)
    standing_since: int | None = None
    role: str = PENDING
    home: tuple[float, float] | None = None
    home_height: float = 0.0
    away_since: int | None = None
    walk_ms: int = 0


class SeatRoleClassifier:
    def __init__(self, *, settle_ms: int = 3_000, settle_drift: float = 0.35, walk_drift: float = 1.0,
                 walk_window_ms: int = 2_000, resettle_ms: int = 6_000, standing_ms: int = 2_000,
                 leave_widths: float = 1.5, leave_ms: int = 5_000) -> None:
        """Distances in box widths of the head movement; tracks that never settle stay ``pending``."""

        self.settle_ms, self.settle_drift = settle_ms, settle_drift
        self.walk_drift, self.walk_window_ms = walk_drift, walk_window_ms
        self.resettle_ms, self.standing_ms = resettle_ms, standing_ms
        self.leave_widths, self.leave_ms = leave_widths, leave_ms
        self._tracks: dict[UUID, _Track] = {}

    def role(self, seat_id: UUID) -> str:
        track = self._tracks.get(seat_id)
        return track.role if track is not None else PENDING

    def update(self, seat_id: UUID, box: Box, monotonic_ms: int, kps: Keypoints | None = None) -> str:
        """Movement is measured on the head in box widths: a seated box's bottom edge jumps whenever the
        desk or the candidate in front hides the legs, but the head stays put while someone writes."""

        track = self._tracks.setdefault(seat_id, _Track(monotonic_ms))
        anchor_x, anchor_y = head_point(kps, box)
        scale = max(box.width, 1e-6)
        if track.role == CANDIDATE and track.home is not None:
            away = hypot(anchor_x - track.home[0], anchor_y - track.home[1]) > self.leave_widths * track.home_height
            track.away_since = (track.away_since or monotonic_ms) if away else None
            return track.role
        track.history.append((monotonic_ms, anchor_x, anchor_y, scale))
        keep = max(self.settle_ms, self.walk_window_ms, self.resettle_ms)
        while track.history and monotonic_ms - track.history[0][0] > keep:
            track.history.popleft()
        standing = is_standing(kps)
        track.standing_since = (track.standing_since or monotonic_ms) if standing else None
        recent = [item for item in track.history if monotonic_ms - item[0] <= self.walk_window_ms]
        if (track.standing_since is not None and monotonic_ms - track.standing_since >= self.standing_ms) or any(
                hypot(anchor_x - x, anchor_y - y) > self.walk_drift * scale for _, x, y, _ in recent):
            track.role, track.walk_ms = WALKER, monotonic_ms
            return track.role
        # A walker who sits down for good becomes a candidate, but only after a longer quiet period.
        need = self.resettle_ms if track.role == WALKER else self.settle_ms
        window = [item for item in track.history if monotonic_ms - item[0] <= need]
        settled = all(hypot(anchor_x - x, anchor_y - y) <= self.settle_drift * scale for _, x, y, _ in window)
        quiet_since = track.walk_ms if track.role == WALKER else track.first_ms
        if (window and monotonic_ms - window[0][0] >= 0.8 * need and monotonic_ms - quiet_since >= need
                and settled and track.standing_since is None):
            track.role = CANDIDATE
            track.home = (median(x for _, x, _, _ in window), median(y for _, _, y, _ in window))
            track.home_height = median(w for *_, w in window)
        return track.role

    def left_seat(self, seat_id: UUID, monotonic_ms: int) -> bool:
        track = self._tracks.get(seat_id)
        return (track is not None and track.role == CANDIDATE and track.away_since is not None
                and monotonic_ms - track.away_since >= self.leave_ms)

    def forget(self, seat_id: UUID) -> None:
        self._tracks.pop(seat_id, None)

    def counts(self) -> dict[str, int]:
        result = {PENDING: 0, CANDIDATE: 0, WALKER: 0}
        for track in self._tracks.values():
            result[track.role] += 1
        return result


__all__ = ["CANDIDATE", "PENDING", "WALKER", "SeatRoleClassifier", "is_standing"]
