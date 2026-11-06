"""Which frames of a suspect window the live deep tier looks at: few enough to keep up with the room.

``per_s`` frames a second inside the window plus ``warmup_frames`` before it (baselines for the head
rules), at most ``max_frames`` in all, always including the moment the fast tier was most suspicious.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .suspects import SuspectWindow


class FrameCursor:
    """One window's frame grid over a video read in order: the first frame at or after each target, the next
    target one ``1000 / fps`` step after the frame taken, until ``end_ms``. Shared by the per-window seek
    (``scripts/review_suspects.py``) and the single pass of ``review_offline`` so both take the same frames."""

    def __init__(self, start_ms: float, end_ms: float, fps: float) -> None:
        self.step = 1000.0 / fps
        self.target = float(max(0.0, start_ms))
        self.end_ms = end_ms

    @property
    def done(self) -> bool:
        return self.target > self.end_ms

    def take(self, position_ms: float) -> bool:
        if self.done or position_ms + 1e-6 < self.target:
            return False
        self.target = position_ms + self.step
        return True


def _nearest(available: Sequence[int], target: float) -> int:
    return min(available, key=lambda t: abs(t - target))


def select_offsets(window: SuspectWindow, available: Sequence[int], *, per_s: float = 3.0, warmup_frames: int = 2,
                   warmup_ms: int = 6000, max_frames: int = 12) -> list[int]:
    if not available:
        return []
    available = sorted(available)
    inside = [t for t in available if window.start_ms <= t <= window.end_ms]
    before = [t for t in available if window.start_ms - warmup_ms <= t < window.start_ms]
    chosen: set[int] = set()
    if before and warmup_frames > 0:
        step = (len(before) - 1) / max(1, warmup_frames - 1) if warmup_frames > 1 else 0
        chosen |= {before[min(len(before) - 1, round(i * step))] for i in range(min(warmup_frames, len(before)))}
    budget = max(1, max_frames - len(chosen))
    if inside:
        span_s = max(0.0, (window.end_ms - window.start_ms) / 1000)
        want = min(budget, len(inside), max(1, math.ceil(span_s * per_s)))
        picks: set[int] = set()
        if window.peak_ms is not None:
            picks.add(_nearest(inside, window.peak_ms))
        rest = want - len(picks)
        if rest > 0:
            step = (window.end_ms - window.start_ms) / rest
            for i in range(rest):
                picks.add(_nearest(inside, window.start_ms + (i + 0.5) * step))
        # Nearest-neighbour picks can collide on short windows; fill up from the unused frames, evenly.
        spare = [t for t in inside if t not in picks]
        while len(picks) < want and spare:
            picks.add(spare.pop(len(spare) // 2))
        chosen |= picks
    return sorted(chosen)


__all__ = ["FrameCursor", "select_offsets"]
