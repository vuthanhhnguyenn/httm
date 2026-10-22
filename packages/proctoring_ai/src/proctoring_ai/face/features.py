"""Geometric eye, mouth and keypoint features from the 478-point face mesh.

Only scalar measurements and a handful of named keypoints leave this module; the
raw mesh is never stored. Indices follow MediaPipe's canonical face mesh, whose
refined model adds iris centres at 468 and 473.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import hypot
from typing import Any, Protocol


class _Point(Protocol):
    x: float
    y: float


# (outer corner, inner corner, upper lid, lower lid, iris centre) for each eye.
EYES = ((33, 133, 159, 145, 468), (263, 362, 386, 374, 473))
MOUTH = (13, 14, 61, 291)  # inner upper lip, inner lower lip, left corner, right corner
KEYPOINTS = {"nose_tip": 1, "chin": 152, "mouth_center": 13, "ear_left": 234, "ear_right": 454}
IRIS_LANDMARKS = 478


def _xy(points: Sequence[_Point], index: int) -> tuple[float, float]:
    point = points[index]
    return float(point.x), float(point.y)


def eye_features(points: Sequence[_Point], frame_width: int) -> dict[str, Any] | None:
    """Horizontal/vertical iris position (−0.5…0.5, 0 = centred) and eye openness.

    ``h`` grows toward the image right for both eyes, ``v`` grows downward. Returns
    ``None`` when the model did not produce iris landmarks.
    """

    if len(points) < IRIS_LANDMARKS:
        return None
    eyes = []
    for outer, inner, upper, lower, iris in EYES:
        (ox, oy), (ix, iy) = _xy(points, outer), _xy(points, inner)
        (_, uy), (_, ly) = _xy(points, upper), _xy(points, lower)
        cx, cy = _xy(points, iris)
        left, right = min(ox, ix), max(ox, ix)
        width = hypot(ox - ix, oy - iy)
        height = ly - uy
        if right - left <= 1e-6 or width <= 1e-6 or height <= 1e-6:
            continue
        eyes.append({
            "h": (cx - left) / (right - left) - 0.5,
            "v": (cy - uy) / height - 0.5,
            "openness": height / width,
            "width_px": width * frame_width,
        })
    if not eyes:
        return None
    total = sum(eye["width_px"] for eye in eyes)
    # The eye seen larger is less foreshortened and gets more weight.
    return {
        "h": sum(eye["h"] * eye["width_px"] for eye in eyes) / total,
        "v": sum(eye["v"] * eye["width_px"] for eye in eyes) / total,
        "openness": min(eye["openness"] for eye in eyes),
        "eye_width_px": min(eye["width_px"] for eye in eyes),
        "eyes_visible": len(eyes),
    }


def mouth_features(points: Sequence[_Point]) -> dict[str, float] | None:
    """Mouth aspect ratio: inner-lip gap over mouth width (scale free)."""

    if len(points) <= max(MOUTH):
        return None
    (ux, uy), (lx, ly) = _xy(points, MOUTH[0]), _xy(points, MOUTH[1])
    (ax, ay), (bx, by) = _xy(points, MOUTH[2]), _xy(points, MOUTH[3])
    width = hypot(ax - bx, ay - by)
    if width <= 1e-6:
        return None
    return {"mar": hypot(ux - lx, uy - ly) / width, "mouth_width": width}


def keypoints(points: Sequence[_Point]) -> dict[str, tuple[float, float]]:
    if len(points) <= max(KEYPOINTS.values()):
        return {}
    return {name: _xy(points, index) for name, index in KEYPOINTS.items()}


__all__ = ["eye_features", "keypoints", "mouth_features"]
