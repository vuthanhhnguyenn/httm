from __future__ import annotations

from uuid import uuid4

from proctoring_ai.multi.cctv.review_cache import load_cache, save_window
from proctoring_ai.multi.cctv.review_detectors import ReviewObject
from proctoring_ai.multi.cctv.review_rules import ReviewFrame
from proctoring_ai.multi.cctv.review_score import Verdict, score_review
from proctoring_ai.multi.cctv.suspects import SuspectWindow
from proctoring_core.observations import Box


def _window(seat, start, end):
    box = Box(0.1, 0.2, 0.1, 0.2)
    return SuspectWindow(seat, 3, (), start, end, ("LEAN",), box, {seat: box})


def test_cache_roundtrip(tmp_path):
    seat = uuid4()
    box = Box(0.1, 0.2, 0.1, 0.2)
    frames = [ReviewFrame(t, {seat: box}, {seat: [[0.123456, 0.2, 0.9]] * 17},
                          [ReviewObject("phone", 0.51234, Box(0.11, 0.21, 0.02, 0.02), "yolo11x")], 16 / 9)
              for t in (1000, 1100)]
    path = tmp_path / "cache.json.gz"
    save_window(path, _window(seat, 1000, 2000), frames)
    save_window(path, _window(seat, 5000, 6000), frames[:1])
    cache = load_cache(path)
    assert set(cache) == {(str(seat), 1000, 2000), (str(seat), 5000, 6000)}
    got = cache[(str(seat), 1000, 2000)]
    assert [f.offset_ms for f in got] == [1000, 1100]
    assert got[0].poses[seat][0] == [0.1235, 0.2, 0.9]
    assert got[0].objects[0].label == "phone" and got[0].objects[0].confidence == 0.5123
    assert got[0].seats[seat] == box
    assert abs(got[0].frame_aspect - 16 / 9) < 1e-3


def _opened(kind, ms, head, action="opened"):
    return {"action": action, "type": kind, "video_offset_ms": ms, "head": list(head)}


def test_score_review():
    verdicts = [Verdict("LOOK_AT_LAP", 43.2, ((0.857, 0.669),), True),
                Verdict("PASSING_OBJECT", 17.2, ((0.336, 0.17), (0.288, 0.227)), False),
                Verdict("LOOK_AT_LAP", 51.2, ((0.776, 0.527),), True)]
    records = [_opened("LOOK_AT_LAP", 43500, (0.86, 0.67)),
               _opened("PASSING_OBJECT", 17500, (0.29, 0.23)),
               _opened("PASSING_OBJECT", 17200, (0.336, 0.17)),  # same tile, other seat: one false
               _opened("LOOK_AT_LAP", 20000, (0.5, 0.5)),  # not judged
               _opened("LOOK_AT_LAP", 45500, (0.86, 0.67)),  # 2.3 s off: not judged
               _opened("LOOK_AT_LAP", 51200, (0.776, 0.527), action="merged"),
               _opened("UNCLASSIFIED_SUSPECT", 51200, (0.776, 0.527))]
    score = score_review(records, verdicts)
    assert score["true_found"] == 1 and score["true_total"] == 2
    assert score["false_opened"] == 1 and score["false_events"] == 2
    assert score["unjudged"] == 2
    assert score["missed"] == ["LOOK_AT_LAP@51.2"]
    # A repeat the deduper folded into an alert already on screen still shows the invigilator that seat.
    assert score["true_covered"] == 2 and score["covered_only"] == ["LOOK_AT_LAP@51.2"]


def test_score_counts_alert_still_open_at_the_picture():
    verdicts = [Verdict("LOOK_AT_LAP", 59.0, ((0.776, 0.527),), True),
                Verdict("PASSING_OBJECT", 22.1, ((0.3, 0.2),), False)]
    lap = {**_opened("LOOK_AT_LAP", 51200, (0.776, 0.527)), "id": "a"}
    passing = {**_opened("PASSING_OBJECT", 17000, (0.3, 0.2)), "id": "b"}
    records = [lap, passing, {"action": "closed", "id": "a", "video_offset_ms": 66000},
               {"action": "closed", "id": "b", "video_offset_ms": 21000}]
    score = score_review(records, verdicts)
    assert score["true_found"] == 0 and score["false_opened"] == 0
    # Still on screen at 59.0 s; the passing alert ended 1.1 s before 22.1 s (within max_dt_s).
    assert score["true_covered"] == 1 and score["false_covered"] == 1
