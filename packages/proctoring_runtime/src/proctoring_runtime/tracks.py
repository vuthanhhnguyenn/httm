"""Per-frame seat boxes written by the fast pass (``tracks.jsonl``) and read by the render pass.

One JSON line per analysed frame: ``{"t": offset_ms, "seats": {seat_id: [x, y, w, h, role]}}``.
Box values are normalised to 0..1 of the frame (same convention as the event and overlay boxes),
rounded to 4 digits; ``role`` is ``candidate``, ``walker`` or ``pending``.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

Tracks = list[tuple[int, dict[str, Any]]]


def track_line(offset_ms: int, seats: Mapping[str, tuple[float, float, float, float, str]]) -> str:
    """Serialise one frame; boxes are rounded to 4 digits."""

    rounded = {seat: [round(x, 4), round(y, 4), round(w, 4), round(h, 4), role]
               for seat, (x, y, w, h, role) in seats.items()}
    return json.dumps({"t": int(offset_ms), "seats": rounded}, ensure_ascii=False) + "\n"


def write_track(handle: TextIO, offset_ms: int, seats: Mapping[str, tuple[float, float, float, float, str]]) -> None:
    handle.write(track_line(offset_ms, seats))


def load_tracks(path: Path) -> Tracks:
    """Read ``tracks.jsonl`` sorted by time (the writer already emits it in order)."""

    rows: Tracks = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append((int(row["t"]), dict(row["seats"])))
    rows.sort(key=lambda row: row[0])
    return rows


def tracks_at(tracks: Tracks, t_ms: int, max_gap_ms: int = 200) -> dict[str, Any] | None:
    """Seats of the line nearest to ``t_ms``, or ``None`` when it is more than ``max_gap_ms`` away."""

    if not tracks:
        return None
    index = bisect_left(tracks, t_ms, key=lambda row: row[0])
    best: tuple[int, dict[str, Any]] | None = None
    for candidate in tracks[max(0, index - 1):index + 1]:
        if best is None or abs(candidate[0] - t_ms) < abs(best[0] - t_ms):
            best = candidate
    if best is None or abs(best[0] - t_ms) > max_gap_ms:
        return None
    return best[1]


__all__ = ["Tracks", "load_tracks", "track_line", "tracks_at", "write_track"]
