"""Phase 3 tier 1: every seat gets looked at for the whole video, not only where the fast tier suspected.

The fast tier still runs, for the seat map (``tracks.jsonl``), but it no longer gates the deep tier: each
candidate seat becomes one ``SuspectWindow`` from 0 to the end of the video, with its neighbours as extra
seats, and ``review_offline`` runs over those windows. Its cache (one ``save_window`` line per seat) is the
timeline the learned core reads.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import median
from typing import Any
from uuid import UUID

from proctoring_core.observations import Box

from .roles import CANDIDATE
from .suspects import SuspectWindow

REASON = "ALL_SEATS"


def load_tracks(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def all_seat_windows(tracks: Iterable[Mapping[str, Any]], duration_ms: int, *, min_candidate_share: float = 0.5,
                     neighbor_widths: float = 2.5, neighbor_rows_heights: float = 1.2) -> list[SuspectWindow]:
    """One window per seat that was a candidate in at least ``min_candidate_share`` of the rows it appears
    in; its box is the per-coordinate median, its neighbours the candidate seats whose centre lies within
    ``neighbor_widths`` seat widths sideways and ``neighbor_rows_heights`` seat heights up or down."""

    boxes: dict[UUID, list[list[float]]] = {}
    seen: dict[UUID, int] = {}
    candidate: dict[UUID, int] = {}
    for row in tracks:
        for key, value in (row.get("seats") or {}).items():
            seat = UUID(key)
            seen[seat] = seen.get(seat, 0) + 1
            if len(value) > 4 and value[4] == CANDIDATE:
                candidate[seat] = candidate.get(seat, 0) + 1
                boxes.setdefault(seat, []).append([float(v) for v in value[:4]])
    seats = {seat: Box(*(median(item[i] for item in boxes[seat]) for i in range(4)))
             for seat in boxes if candidate[seat] >= min_candidate_share * seen[seat]}

    def near(a: Box, b: Box) -> bool:
        (ax, ay), (bx, by) = a.center, b.center
        return abs(ax - bx) < neighbor_widths * a.width and abs(ay - by) < neighbor_rows_heights * a.height

    windows = []
    for seat in sorted(seats, key=str):
        box = seats[seat]
        neighbours = tuple(other for other in sorted(seats, key=str) if other != seat and near(box, seats[other]))
        windows.append(SuspectWindow(seat, None, neighbours, 0, int(duration_ms), (REASON,), box,
                                     {seat: box, **{other: seats[other] for other in neighbours}}))
    return windows


def load_timeline(folder: Path) -> tuple[list[SuspectWindow], dict[Any, list[Any]]]:
    """The seat windows and their frames of a ``build_timeline.py`` output folder (``tracks.jsonl`` +
    ``timeline.json.gz``); the video length is read off the cache keys."""

    from .review_cache import load_cache

    frames = load_cache(folder / "timeline.json.gz")
    duration = max((key[2] for key in frames), default=0)
    windows = all_seat_windows(load_tracks(folder / "tracks.jsonl"), duration)
    return windows, frames
