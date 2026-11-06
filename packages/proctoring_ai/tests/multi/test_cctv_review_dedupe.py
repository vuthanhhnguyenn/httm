from __future__ import annotations

from proctoring_ai.multi.cctv.review_dedupe import ReviewDeduper


def pair(event_id, seat, kind, ms, closed_ms=None):
    base = {"id": event_id, "seat_id": seat, "type": kind, "stage": "review"}
    return [{"action": "opened", "video_offset_ms": ms, "evidence": f"evidence/{event_id}.jpg", **base},
            {"action": "closed", "video_offset_ms": closed_ms or ms + 500, **base}]


def test_same_seat_type_within_10s_merged():
    dedupe = ReviewDeduper()
    first = dedupe.admit(pair("e1", "s1", "PASSING_OBJECT", 17200))
    second = dedupe.admit(pair("e2", "s1", "PASSING_OBJECT", 17500))
    assert [item["action"] for item in first] == ["opened", "closed"]
    assert len(second) == 1 and second[0]["action"] == "merged" and second[0]["merged_into"] == "e1"
    assert "evidence" not in second[0]


def test_after_10s_opens_again():
    dedupe = ReviewDeduper()
    dedupe.admit(pair("e1", "s1", "PASSING_OBJECT", 17200))
    again = dedupe.admit(pair("e2", "s1", "PASSING_OBJECT", 28000))
    assert [item["action"] for item in again] == ["opened", "closed"]


def test_other_seat_or_type_not_merged():
    dedupe = ReviewDeduper()
    dedupe.admit(pair("e1", "s1", "PASSING_OBJECT", 17200))
    assert dedupe.admit(pair("e2", "s2", "PASSING_OBJECT", 17300))[0]["action"] == "opened"
    assert dedupe.admit(pair("e3", "s1", "LOOK_AT_LAP", 17300))[0]["action"] == "opened"


def test_out_of_order_windows_still_merged():
    dedupe = ReviewDeduper()
    dedupe.admit(pair("e1", "s1", "PHONE_DETECTED", 60800))
    assert dedupe.admit(pair("e2", "s1", "PHONE_DETECTED", 56800))[0]["action"] == "merged"


def test_other_records_pass_through():
    dedupe = ReviewDeduper()
    dismissed = {"action": "dismissed", "seat_id": "s1", "start_ms": 0, "end_ms": 1}
    assert dedupe.admit([dismissed]) == [dismissed]
