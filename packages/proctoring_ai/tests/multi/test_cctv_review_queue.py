from __future__ import annotations

from uuid import uuid4

from proctoring_ai.multi.cctv.review_queue import ReviewQueue, skipped_record
from proctoring_ai.multi.cctv.suspects import SuspectWindow
from proctoring_core.observations import Box

BOX = Box(0.1, 0.2, 0.1, 0.2)


def win(seat=None, start=0, end=5000, reasons=("LEAN",), peak=None):
    seat = seat or uuid4()
    return SuspectWindow(seat, 1, (), start, end, tuple(reasons), BOX, {seat: BOX}, peak)


def test_rule_event_first_then_more_reasons_then_older():
    queue = ReviewQueue()
    a = win(reasons=("LEAN",))
    b = win(reasons=("LEAN", "HEAD_AWAY"))
    c = win(reasons=("RULE_EVENT",))
    d = win(reasons=("LEAN", "HEAD_AWAY"))
    for i, item in enumerate((a, b, c, d)):
        queue.push(item, now_wall_ms=i)
    assert [queue.pop(), queue.pop(), queue.pop(), queue.pop(), queue.pop()] == [c, b, d, a, None]


def test_overlapping_same_seat_merged():
    queue = ReviewQueue()
    seat = uuid4()
    queue.push(win(seat, 0, 5000, ("LEAN",)), now_wall_ms=0)
    queue.push(win(seat, 4000, 9000, ("HANDS_MEET",)), now_wall_ms=10)
    merged = queue.pop()
    assert (merged.start_ms, merged.end_ms) == (0, 9000)
    assert set(merged.reasons) == {"LEAN", "HANDS_MEET"}
    assert queue.pop() is None


def test_deadline_expires():
    queue = ReviewQueue(deadline_ms=60000)
    old = win()
    queue.push(old, now_wall_ms=0)
    fresh = win()
    queue.push(fresh, now_wall_ms=30000)
    assert queue.expired(now_wall_ms=60001) == [old]
    assert queue.pop() == fresh


def test_skipped_record():
    w = win(start=1000, end=6000, peak=3000)
    item = skipped_record(w, "deadline")
    assert item["type"] == "REVIEW_SKIPPED" and item["stage"] == "review" and item["action"] == "opened"
    assert item["video_offset_ms"] == 3000 and item["skip_reason"] == "deadline"
    assert skipped_record(win(start=1000), "shutdown")["video_offset_ms"] == 1000
