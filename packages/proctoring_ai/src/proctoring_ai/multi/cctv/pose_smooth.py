"""Steadier keypoints for the deep tier (bước 5).

``smooth_frames``: a One Euro filter (Casiraghi et al., 2012) per keypoint of every seat, over the frames'
``offset_ms``. A still point is smoothed hard (cutoff ``min_cutoff`` Hz), a moving one follows quickly (the
cutoff grows by ``beta`` per unit/s of speed). A point the model is unsure of (confidence < 0.4) passes
through as it is and does not move the filter; a seat that lost its skeleton starts its filters again.

``merge_flipped``: test-time flip for pose. The crop is run a second time mirrored; its people are mirrored
back around the crop's centre line, left and right keypoints swapped (COCO), and each is averaged with the
original person it overlaps (IoU >= 0.5), weighting each point by its confidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from uuid import UUID

from proctoring_core.observations import Box

from ..tracker import iou

if TYPE_CHECKING:
    from .review_rules import ReviewFrame

SMOOTH_MIN_CONF = 0.4
DERIVATIVE_CUTOFF = 1.0  # Hz, the filter on the speed itself
FLIP_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
FLIP_MIN_IOU = 0.5


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _OneEuro:
    __slots__ = ("beta", "dx", "min_cutoff", "t", "x")

    def __init__(self, min_cutoff: float, beta: float) -> None:
        self.min_cutoff, self.beta = min_cutoff, beta
        self.t: float | None = None
        self.x = self.dx = 0.0

    def __call__(self, t: float, x: float) -> float:
        if self.t is None:
            self.t, self.x, self.dx = t, x, 0.0
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        self.dx += _alpha(DERIVATIVE_CUTOFF, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += _alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


def smooth_frames(frames: Sequence[ReviewFrame], *, min_cutoff: float = 1.0, beta: float = 0.05) -> list[ReviewFrame]:
    """``frames`` in time order with each seat's keypoints smoothed (see the module doc). Seats, objects and
    confidences are kept; the input is not changed."""

    filters: dict[tuple[UUID, int, int], _OneEuro] = {}
    out: list[ReviewFrame] = []
    for frame in sorted(frames, key=lambda item: item.offset_ms):
        t = frame.offset_ms / 1000.0
        poses: dict[UUID, Any] = {}
        for seat_id in frame.seats:
            kps = frame.poses.get(seat_id)
            if kps is None:
                for key in [key for key in filters if key[0] == seat_id]:
                    del filters[key]  # lost: start again when the seat has a skeleton again
                continue
            smoothed: list[list[float]] = []
            for index, keypoint in enumerate(kps):
                row = [float(value) for value in keypoint]
                if len(row) >= 3 and row[2] >= SMOOTH_MIN_CONF:
                    for axis in (0, 1):
                        flt = filters.setdefault((seat_id, index, axis), _OneEuro(min_cutoff, beta))
                        row[axis] = flt(t, row[axis])
                smoothed.append(row)
            poses[seat_id] = smoothed
        # Poses of ids that are not seats of the frame (never in practice) go through unchanged.
        poses.update({key: value for key, value in frame.poses.items() if key not in frame.seats})
        out.append(replace(frame, poses=poses))
    return out


def _box(item: dict[str, Any]) -> Box | None:
    return Box.parse(item.get("bbox"))


def _mirrored(item: dict[str, Any], mirror_x: float) -> dict[str, Any]:
    """The person found in the flipped crop, back in the frame's own orientation."""

    out = dict(item)
    if (box := _box(item)) is not None:
        out["bbox"] = {"x": 2 * mirror_x - (box.x + box.width), "y": box.y, "width": box.width, "height": box.height}
    kps = [[2 * mirror_x - float(p[0]), *(float(v) for v in p[1:])] for p in item.get("keypoints") or []]
    for left, right in FLIP_PAIRS:
        if right < len(kps):
            kps[left], kps[right] = kps[right], kps[left]
    out["keypoints"] = kps
    return out


def _average(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for index in range(max(len(a), len(b))):
        if index >= len(b) or len(b[index]) < 3:
            out.append([float(v) for v in a[index]])
            continue
        if index >= len(a) or len(a[index]) < 3:
            out.append([float(v) for v in b[index]])
            continue
        p, q = a[index], b[index]
        weight = float(p[2]) + float(q[2])
        if weight <= 0:
            out.append([float(v) for v in p])
            continue
        out.append([(float(p[0]) * float(p[2]) + float(q[0]) * float(q[2])) / weight,
                    (float(p[1]) * float(p[2]) + float(q[1]) * float(q[2])) / weight,
                    weight / 2])
    return out


def merge_flipped(people: list[dict[str, Any]], flipped: list[dict[str, Any]], mirror_x: float) -> list[dict[str, Any]]:
    """``people`` from the crop and ``flipped`` from the same crop mirrored (full-frame coordinates around the
    crop's centre line ``mirror_x``) as one list: pairs overlapping by IoU >= 0.5 (best first, one to one) get
    confidence-weighted keypoints; a person seen on one side only is kept as found (mirrored back if it came
    from the flipped crop)."""

    back = [_mirrored(item, mirror_x) for item in flipped]
    pairs = sorted(((iou(a, b), i, j) for i, item in enumerate(people) if (a := _box(item)) is not None
                    for j, other in enumerate(back) if (b := _box(other)) is not None), reverse=True)
    match: dict[int, int] = {}
    used: set[int] = set()
    for value, i, j in pairs:
        if value < FLIP_MIN_IOU:
            break
        if i not in match and j not in used:
            match[i] = j
            used.add(j)
    out: list[dict[str, Any]] = []
    for i, item in enumerate(people):
        if i in match:
            out.append({**item, "keypoints": _average(item.get("keypoints") or [], back[match[i]]["keypoints"])})
        else:
            out.append(item)
    out += [item for j, item in enumerate(back) if j not in used]
    return out


__all__ = ["FLIP_PAIRS", "merge_flipped", "smooth_frames"]
