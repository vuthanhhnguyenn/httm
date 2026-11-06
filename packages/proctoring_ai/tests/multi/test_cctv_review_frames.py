from __future__ import annotations

from uuid import uuid4

from proctoring_ai.multi.cctv.review_frames import select_offsets
from proctoring_ai.multi.cctv.suspects import SuspectWindow
from proctoring_core.observations import Box

BOX = Box(0.1, 0.2, 0.1, 0.2)


def win(start, end, peak=None):
    seat = uuid4()
    return SuspectWindow(seat, 1, (), start, end, ("LEAN",), BOX, {seat: BOX}, peak)


def test_select_caps_and_keeps_peak():
    available = list(range(0, 50000, 100))
    chosen = select_offsets(win(10000, 40000, peak=23450), available)
    assert len(chosen) == 12 and chosen == sorted(chosen)
    assert 23400 in chosen or 23500 in chosen
    assert sum(1 for t in chosen if t < 10000) == 2
    assert all(t in available for t in chosen)


def test_select_short_window():
    chosen = select_offsets(win(10000, 11000), list(range(0, 20000, 100)))
    assert sum(1 for t in chosen if t < 10000) == 2 and sum(1 for t in chosen if t >= 10000) == 3


def test_select_nothing_available():
    assert select_offsets(win(10000, 11000), []) == []


def test_select_window_at_clip_start_has_no_warmup():
    chosen = select_offsets(win(0, 1000), list(range(0, 2000, 100)))
    assert len(chosen) == 3 and min(chosen) >= 0


def test_peak_ms_roundtrip():
    w = win(0, 1000, peak=500)
    assert SuspectWindow.from_json(w.to_json()).peak_ms == 500
    assert SuspectWindow.from_json({**w.to_json(), "peak_ms": None}).peak_ms is None
