from __future__ import annotations

from typing import Any

import pytest
from proctoring_ai.multi.tracker import MultiSubjectTracker, TrackerUpdate


def person(x: float, y: float, w: float = 0.15, h: float = 0.4, c: float = 0.9) -> dict[str, Any]:
    return {"class": "person", "confidence": c, "bbox": {"x": x, "y": y, "width": w, "height": h}}


def ids_by_x(update: TrackerUpdate) -> list[int]:
    return [s.runtime_track_id for s in sorted(update.analyzed, key=lambda s: s.bbox.x)]


def test_ids_stable_while_people_shift_slightly() -> None:
    t = MultiSubjectTracker()
    first = t.update([person(0.1, 0.3), person(0.4, 0.3)], 0)
    assert ids_by_x(first) == [1, 2] and first.created == 2
    later = t.update([person(0.41, 0.29), person(0.11, 0.31)], 100)
    assert ids_by_x(later) == [1, 2] and later.created == 0


def test_neighbours_jitter_do_not_swap_ids() -> None:
    t = MultiSubjectTracker()
    t.update([person(0.10, 0.3), person(0.27, 0.3)], 0)
    assert ids_by_x(t.update([person(0.26, 0.3), person(0.12, 0.3)], 100)) == [1, 2]


def test_returning_person_relinks_to_same_seat_within_10_s() -> None:
    t = MultiSubjectTracker()
    t.update([person(0.1, 0.3, w=0.3)], 0)  # centre (0.25, 0.5)
    assert [s.runtime_track_id for s in t.update([], 500).missing] == [1]
    back = t.update([person(0.1, 0.55, w=0.3, h=0.1)], 9_000)  # IoU 0.25 < 0.3, centre 0.1 < 0.15 away
    assert [s.runtime_track_id for s in back.analyzed] == [1] and back.relinked == 1


def test_distant_new_person_is_not_relinked() -> None:
    t = MultiSubjectTracker()
    t.update([person(0.1, 0.3)], 0)
    update = t.update([person(0.6, 0.3)], 500)
    assert [s.runtime_track_id for s in update.analyzed] == [2]
    assert [s.runtime_track_id for s in update.missing] == [1]


def test_seat_expires_after_10_s_and_new_person_gets_new_id() -> None:
    t = MultiSubjectTracker()
    t.update([person(0.1, 0.3)], 0)
    gone = t.update([], 10_000)
    assert [s.runtime_track_id for s in gone.expired] == [1] and gone.missing == []
    assert [s.runtime_track_id for s in t.update([person(0.1, 0.3)], 10_100).analyzed] == [2]


def test_more_than_six_people_keeps_six_largest() -> None:
    t = MultiSubjectTracker()
    people = [person(0.02 + i * 0.13, 0.3, w=0.10 + 0.002 * i) for i in range(7)]
    update = t.update(people, 0)
    assert len(update.analyzed) == 6 and len(update.over_limit) == 1
    assert update.over_limit[0].bbox.width == pytest.approx(0.10)


def test_seat_records_visibility_and_timestamps() -> None:
    t = MultiSubjectTracker()
    t.update([person(0.1, 0.3)], 0)
    seat = t.update([person(0.1, 0.3)], 250).analyzed[0]
    assert seat.visible and seat.first_seen_ms == 0 and seat.last_seen_ms == 250
    assert not t.update([], 400).missing[0].visible
