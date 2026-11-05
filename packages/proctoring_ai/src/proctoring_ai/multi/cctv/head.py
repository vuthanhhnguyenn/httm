"""Head direction from body keypoints seen by a high, fixed room camera.

Candidates face the camera, so a turn shows up as the nose moving sideways relative to the
shoulders. Each seat is compared with its own rolling median, not with an absolute angle, so
perspective and a habitual lean cancel out without asking anybody to hold still.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from math import hypot
from statistics import median

from proctoring_core.observations import Box

from .pose_detector import (
    LEFT_EAR,
    LEFT_EYE,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_EAR,
    RIGHT_EYE,
    RIGHT_SHOULDER,
)

Keypoints = Sequence[Sequence[float]]
POINT_MIN_CONF = 0.3
SHOULDER_MIN_CONF = 0.5
FACE_MIN_CONF = 0.5


def point(kps: Keypoints | None, index: int, min_conf: float = POINT_MIN_CONF) -> tuple[float, float] | None:
    if kps is None or index >= len(kps) or len(kps[index]) < 3 or kps[index][2] < min_conf:
        return None
    return float(kps[index][0]), float(kps[index][1])


def shoulders(kps: Keypoints | None) -> tuple[tuple[float, float], tuple[float, float]] | None:
    left, right = point(kps, LEFT_SHOULDER, SHOULDER_MIN_CONF), point(kps, RIGHT_SHOULDER, SHOULDER_MIN_CONF)
    if left is None or right is None or abs(left[0] - right[0]) < 1e-3:
        return None
    return left, right


def head_yaw_index(kps: Keypoints | None) -> float | None:
    """Nose offset from the shoulder midpoint in shoulder widths; positive = nose towards image right."""

    nose, pair = point(kps, NOSE, FACE_MIN_CONF), shoulders(kps)
    if nose is None or pair is None:
        return None
    (lx, _), (rx, _) = pair
    return (nose[0] - (lx + rx) / 2) / abs(lx - rx)


def head_lift(kps: Keypoints | None, frame_aspect: float = 1.0) -> float | None:
    """How far the nose sits above the shoulder line, in shoulder widths (pixel units). Near zero or
    negative when the head is bent over the desk; ``frame_aspect`` is the frame's width / height."""

    nose, pair = point(kps, NOSE, FACE_MIN_CONF), shoulders(kps)
    if nose is None or pair is None:
        return None
    (lx, ly), (rx, ry) = pair
    return ((ly + ry) / 2 - nose[1]) / (abs(lx - rx) * frame_aspect)


def facing(kps: Keypoints | None) -> str:
    """``front`` when the face is seen, ``back`` when the face is hidden but the shoulders and an ear are,
    else ``unknown``. A head bowed low under a close camera hides the ears too, showing only its crown."""

    face = [point(kps, index, FACE_MIN_CONF) for index in (NOSE, LEFT_EYE, RIGHT_EYE)]
    if any(item is not None for item in face):
        return "front"
    hidden = all(point(kps, index, POINT_MIN_CONF) is None for index in (NOSE, LEFT_EYE, RIGHT_EYE))
    ear = any(point(kps, index) is not None for index in (LEFT_EAR, RIGHT_EAR))
    return "back" if hidden and ear and shoulders(kps) is not None else "unknown"


def head_point(kps: Keypoints | None, box: Box) -> tuple[float, float]:
    """Mean of the visible face points; top-centre of the box when none is visible."""

    visible = [p for index in (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR) if (p := point(kps, index))]
    if not visible:
        return box.x + box.width / 2, box.y + 0.12 * box.height
    return sum(p[0] for p in visible) / len(visible), sum(p[1] for p in visible) / len(visible)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


class RollingBaseline:
    """Median of the last ``window_ms``; ready once ``min_span_ms`` and ``min_samples`` are covered."""

    def __init__(self, *, window_ms: int = 30_000, min_span_ms: int = 5_000, min_samples: int = 8) -> None:
        self.window_ms, self.min_span_ms, self.min_samples = window_ms, min_span_ms, min_samples
        self._samples: deque[tuple[int, float]] = deque()
        self._first_ms: int | None = None

    def add(self, value: float, monotonic_ms: int) -> None:
        if self._first_ms is None:
            self._first_ms = monotonic_ms
        self._samples.append((monotonic_ms, value))
        while self._samples and monotonic_ms - self._samples[0][0] > self.window_ms:
            self._samples.popleft()

    def value(self, monotonic_ms: int) -> float | None:
        if (self._first_ms is None or monotonic_ms - self._first_ms < self.min_span_ms
                or len(self._samples) < self.min_samples):
            return None
        return float(median(value for _, value in self._samples))

    def spread(self, monotonic_ms: int) -> float:
        """Median absolute deviation around the baseline: how much this seat normally moves."""

        centre = self.value(monotonic_ms)
        return 0.0 if centre is None else float(median(abs(value - centre) for _, value in self._samples))


__all__ = ["RollingBaseline", "distance", "facing", "head_lift", "head_point", "head_yaw_index", "point", "shoulders"]
