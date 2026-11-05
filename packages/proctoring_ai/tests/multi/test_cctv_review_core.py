from __future__ import annotations

from uuid import uuid4

import numpy as np
from cctv_kps import person
from proctoring_ai.multi.cctv.review_core import analyse_window
from proctoring_ai.multi.cctv.review_rules import review_window
from proctoring_ai.multi.cctv.suspects import SuspectWindow
from proctoring_core.observations import Box


class _Detectors:
    def __init__(self, kps):
        self.kps, self.calls = kps, []

    def pose_many(self, crops, transforms):
        self.calls.append(("pose", len(crops)))
        return [[{"class": "person", "confidence": 0.9, "bbox": {"x": 0.45, "y": 0.3, "width": 0.1, "height": 0.2},
                  "keypoints": self.kps}] for _ in crops]

    def objects_many(self, crops, transforms):
        self.calls.append(("objects", len(crops)))
        return [[] for _ in crops]


def test_analyse_window_one_batch_and_same_records_as_rules():
    seat = uuid4()
    box = Box(0.45, 0.3, 0.1, 0.2)
    window = SuspectWindow(seat, 1, (), 1000, 3000, ("LEAN",), box, {seat: box})
    images = [(t, np.zeros((72, 128, 3), np.uint8)) for t in range(0, 3001, 250)]
    detectors = _Detectors(person(0.5, 0.3))
    records, frames = analyse_window(window, images, detectors)
    assert detectors.calls == [("pose", 13), ("objects", 13)]
    assert [f.offset_ms for f in frames] == [t for t, _ in images]
    assert frames[0].poses[seat] == person(0.5, 0.3)
    assert abs(frames[0].frame_aspect - 128 / 72) < 1e-6
    strip = lambda items: [{k: v for k, v in item.items() if k != "id"} for item in items]  # noqa: E731
    assert strip(records) == strip(review_window(window, frames))


def test_analyse_window_no_images():
    seat = uuid4()
    box = Box(0.45, 0.3, 0.1, 0.2)
    window = SuspectWindow(seat, 1, (), 1000, 3000, ("LEAN",), box, {seat: box})
    detectors = _Detectors(person())
    records, frames = analyse_window(window, [], detectors)
    assert frames == [] and detectors.calls == []
    assert [item["action"] for item in records] == ["dismissed"]


def _bent_frames(seat, box, start, end, step=200):
    from proctoring_ai.multi.cctv.review_rules import ReviewFrame

    kps = person(cx=0.5, top=0.3, bent=True)
    kps[9] = [0.5, 0.49, 0.9]  # both hands below the desk line, head down: the lap rule
    kps[10] = [0.52, 0.49, 0.9]
    return [ReviewFrame(t, {seat: box}, {seat: kps}, [], 16 / 9) for t in range(start, end, step)]


def test_history_carries_a_long_window_across_pieces():
    from proctoring_ai.multi.cctv.review_core import ReviewHistory

    seat = uuid4()
    box = Box(0.45, 0.3, 0.1, 0.2)
    first = SuspectWindow(seat, 1, (), 0, 10000, ("LEAN",), box, {seat: box})
    second = SuspectWindow(seat, 1, (), 10000, 12000, ("LEAN",), box, {seat: box})
    history = ReviewHistory(keep_ms=20000)
    history.add(first, _bent_frames(seat, box, 0, 10000))
    context = history.before(second)
    assert context and context[-1].offset_ms < second.start_ms and context[0].offset_ms >= 0
    # Only 2 s of new frames: too short for the 3 s lap hold alone, enough with what came before.
    new = _bent_frames(seat, box, 10000, 12000)
    assert "LOOK_AT_LAP" not in [r["type"] for r in review_window(second, new) if r["action"] == "opened"]
    assert "LOOK_AT_LAP" in [r["type"] for r in review_window(second, context + new) if r["action"] == "opened"]


def test_history_only_for_the_same_seats_and_recent():
    from proctoring_ai.multi.cctv.review_core import ReviewHistory

    seat, other = uuid4(), uuid4()
    box = Box(0.45, 0.3, 0.1, 0.2)
    history = ReviewHistory(keep_ms=5000)
    history.add(SuspectWindow(seat, 1, (), 0, 10000, ("LEAN",), box, {seat: box}), _bent_frames(seat, box, 0, 10000))
    later = SuspectWindow(seat, 1, (), 10000, 12000, ("LEAN",), box, {seat: box})
    assert min(f.offset_ms for f in history.before(later)) >= 5000
    with_neighbour = SuspectWindow(seat, 1, (other,), 10000, 12000, ("LEAN",), box, {seat: box, other: box})
    assert history.before(with_neighbour) == []


def test_analyse_window_rules_see_context_but_returns_new_frames_only():
    seat = uuid4()
    box = Box(0.45, 0.3, 0.1, 0.2)
    window = SuspectWindow(seat, 1, (), 10000, 12000, ("LEAN",), box, {seat: box})
    images = [(t, np.zeros((72, 128, 3), np.uint8)) for t in range(10000, 12000, 200)]
    kps = person(cx=0.5, top=0.3, bent=True)
    kps[9], kps[10] = [0.5, 0.49, 0.9], [0.52, 0.49, 0.9]
    records, frames = analyse_window(window, images, _Detectors(kps), context=_bent_frames(seat, box, 4000, 10000))
    assert [f.offset_ms for f in frames] == [t for t, _ in images]
    assert "LOOK_AT_LAP" in [r["type"] for r in records if r["action"] == "opened"]


def test_history_not_used_across_a_hole():
    # Seat reviewed at 0-10 s, next window at 25 s: old frames are no warm-up for it (review finding).
    from proctoring_ai.multi.cctv.review_core import ReviewHistory

    seat = uuid4()
    box = Box(0.45, 0.3, 0.1, 0.2)
    history = ReviewHistory(keep_ms=20000)
    history.add(SuspectWindow(seat, 1, (), 0, 10000, ("LEAN",), box, {seat: box}), _bent_frames(seat, box, 0, 10000))
    assert history.before(SuspectWindow(seat, 1, (), 25000, 35000, ("LEAN",), box, {seat: box})) == []
