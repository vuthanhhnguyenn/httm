"""Skip face inference for a candidate whose crop has not changed.

Candidates sit still most of an exam; reusing the last face result for a short
while saves most of the per-person landmark cost without hiding real movement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

cv2: Any
try:
    import cv2
except ImportError:  # pragma: no cover - exercised in environments without CV wheels
    cv2 = None


@dataclass(slots=True)
class _Sample:
    thumbnail: Any
    monotonic_ms: int


class MotionGate:
    def __init__(self, *, threshold: float = 4.0, max_reuse_ms: int = 500, size: int = 64) -> None:
        self.threshold = threshold
        self.max_reuse_ms = max_reuse_ms
        self.size = size
        self._samples: dict[UUID, _Sample] = {}

    def _thumbnail(self, image: Any) -> Any:
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32)

    def should_run(self, seat_id: UUID, image: Any, monotonic_ms: int) -> bool:
        sample = self._samples.get(seat_id)
        if sample is None or monotonic_ms - sample.monotonic_ms >= self.max_reuse_ms:
            return True
        difference = float(np.mean(np.abs(self._thumbnail(image) - sample.thumbnail)))
        return difference >= self.threshold

    def record(self, seat_id: UUID, image: Any, monotonic_ms: int) -> None:
        self._samples[seat_id] = _Sample(self._thumbnail(image), monotonic_ms)

    def forget(self, seat_id: UUID) -> None:
        self._samples.pop(seat_id, None)


__all__ = ["MotionGate"]
