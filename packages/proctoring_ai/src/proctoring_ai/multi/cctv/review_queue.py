"""Suspect windows waiting for the deep tier: merged per seat, most suspicious first, never dropped silently.

A window waiting longer than ``deadline_ms`` (wall clock) is taken out by ``expired`` and becomes a
``REVIEW_SKIPPED`` record ("Chưa kiểm tra kỹ") for the invigilator to look at by hand.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any
from uuid import uuid4

from proctoring_core.types import EventType

from .suspects import REASONS, SuspectWindow, _union


def merge_windows(a: SuspectWindow, b: SuspectWindow) -> SuspectWindow:
    seats = dict(a.seats)
    for key, box in b.seats.items():
        seats[key] = _union(seats[key], box) if key in seats else box
    reasons = set(a.reasons) | set(b.reasons)
    peaks = [p for p in (a.peak_ms, b.peak_ms) if p is not None]
    return replace(a, neighbor_ids=tuple(sorted(set(a.neighbor_ids) | set(b.neighbor_ids), key=str)),
                   start_ms=min(a.start_ms, b.start_ms), end_ms=max(a.end_ms, b.end_ms),
                   reasons=tuple(r for r in REASONS if r in reasons) + tuple(sorted(reasons - set(REASONS))),
                   bbox=_union(a.bbox, b.bbox), seats=seats, peak_ms=min(peaks) if peaks else None)


class ReviewQueue:
    def __init__(self, deadline_ms: int = 60000, max_frames: int = 12) -> None:
        self.deadline_ms = deadline_ms
        self.max_frames = max_frames
        self._waiting: list[tuple[int, int, SuspectWindow]] = []  # (pushed at, order, window)
        self._order = itertools.count()
        self.max_len = 0

    def __len__(self) -> int:
        return len(self._waiting)

    def push(self, window: SuspectWindow, now_wall_ms: int) -> None:
        for i, (pushed, order, other) in enumerate(self._waiting):
            if other.seat_id == window.seat_id and window.start_ms <= other.end_ms and other.start_ms <= window.end_ms:
                self._waiting[i] = (pushed, order, merge_windows(other, window))
                return
        self._waiting.append((now_wall_ms, next(self._order), window))
        self.max_len = max(self.max_len, len(self._waiting))

    @staticmethod
    def _rank(item: tuple[int, int, SuspectWindow]) -> tuple[int, int, int]:
        _pushed, order, window = item
        return (0 if "RULE_EVENT" in window.reasons else 1, -len(set(window.reasons)), order)

    def pop(self) -> SuspectWindow | None:
        item = self.pop_timed()
        return item[1] if item is not None else None

    def pop_timed(self) -> tuple[int, SuspectWindow] | None:
        """(wall ms when first pushed, window) of the most suspicious waiting window."""

        if not self._waiting:
            return None
        best = min(self._waiting, key=self._rank)
        self._waiting.remove(best)
        return best[0], best[2]

    def expired(self, now_wall_ms: int) -> list[SuspectWindow]:
        late = [item for item in self._waiting if now_wall_ms - item[0] > self.deadline_ms]
        for item in late:
            self._waiting.remove(item)
        return [item[2] for item in sorted(late, key=lambda item: item[1])]

    def drain_all(self) -> list[SuspectWindow]:
        out = [item[2] for item in sorted(self._waiting, key=lambda item: item[1])]
        self._waiting.clear()
        return out


def skipped_record(window: SuspectWindow, reason: str) -> dict[str, Any]:
    """``reason``: "deadline" | "frames_gone" | "shutdown"."""

    head = window.bbox.x + window.bbox.width / 2, window.bbox.y + 0.25 * window.bbox.height
    return {"action": "opened", "id": str(uuid4()), "type": EventType.REVIEW_SKIPPED.value,
            "seat_id": str(window.seat_id), "scene": "cctv", "stage": "review",
            "runtime_track_id": window.runtime_track_id,
            "video_offset_ms": window.peak_ms if window.peak_ms is not None else window.start_ms,
            "start_ms": window.start_ms, "end_ms": window.end_ms, "head": [round(head[0], 3), round(head[1], 3)],
            "reasons": list(window.reasons), "skip_reason": reason}


__all__ = ["ReviewQueue", "merge_windows", "skipped_record"]
