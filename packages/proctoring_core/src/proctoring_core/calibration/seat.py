"""Learn where the candidate sits, then judge "leaving the seat" against that zone.

A fixed screen position cannot work for every camera placement, so the zone is the
median person box over the first seconds of the session (or after re-calibration).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ..observations import Box


@dataclass(frozen=True, slots=True)
class SeatZone:
    box: Box

    @property
    def area(self) -> float:
        return max(1e-6, self.box.width * self.box.height)

    def contains(self, point: tuple[float, float], *, margin: float) -> bool:
        x, y = point
        dx, dy = self.box.width * margin, self.box.height * margin
        return (self.box.x - dx <= x <= self.box.x + self.box.width + dx
                and self.box.y - dy <= y <= self.box.y + self.box.height + dy)


class SeatZoneEstimator:
    """Collect person boxes for ``learning_ms`` and freeze their median as the seat zone."""

    def __init__(self, *, learning_ms: int, min_samples: int = 5) -> None:
        self.learning_ms = learning_ms
        self.min_samples = min_samples
        self.zone: SeatZone | None = None
        self._samples: list[Box] = []
        self._started_ms: int | None = None

    def reset(self) -> None:
        self.zone = None
        self._samples.clear()
        self._started_ms = None

    def update(self, person: Box | None, monotonic_ms: int) -> SeatZone | None:
        if self.zone is not None or person is None:
            return self.zone
        if self._started_ms is None:
            self._started_ms = monotonic_ms
        self._samples.append(person)
        if monotonic_ms - self._started_ms >= self.learning_ms and len(self._samples) >= self.min_samples:
            self.zone = SeatZone(Box(
                median(item.x for item in self._samples),
                median(item.y for item in self._samples),
                median(item.width for item in self._samples),
                median(item.height for item in self._samples),
            ))
        return self.zone

    def is_leaving(self, person: Box | None, shoulders: tuple[float, float] | None, *, margin: float,
                   min_area_ratio: float) -> bool:
        """Present but out of place: centre (or shoulder midpoint) outside the zone, or much smaller.

        An absent person is ``PERSON_MISSING``'s job, not this rule's.
        """

        if self.zone is None or person is None:
            return False
        anchor = shoulders or person.center
        if not self.zone.contains(anchor, margin=margin):
            return True
        return person.width * person.height < self.zone.area * min_area_ratio


__all__ = ["SeatZone", "SeatZoneEstimator"]
