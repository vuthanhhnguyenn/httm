from __future__ import annotations

import asyncio

import pytest
from cctv_kps import person
from proctoring_ai.multi.cctv.seat_map import SeatMapRoles, SeatMapTracker
from proctoring_ai.multi.cctv.tiling import (
    TiledPoseDetector,
    map_from_tile,
    merge_people,
    tiles_for,
)

np = pytest.importorskip("numpy")


def body(cx: float, top: float = 0.3, *, conf: float = 0.9, height: float = 0.2, width: float = 0.1,
         **kwargs) -> dict:
    return {"class": "person", "confidence": conf, "keypoints": person(cx=cx, top=top, **kwargs),
            "bbox": {"x": cx - width / 2, "y": top, "width": width, "height": height}}


def feed(tracker: SeatMapTracker, frames, start: int = 0, step: int = 100):
    update = None
    for index, people in enumerate(frames):
        update = tracker.update(people, start + index * step)
    return update


def test_duplicate_boxes_of_one_head_merge_to_the_most_confident() -> None:
    big, small = body(0.5, conf=0.6, height=0.4), body(0.5, conf=0.8)
    assert merge_people([big, small, body(0.7, conf=0.5)]) == [small, body(0.7, conf=0.5)]


def test_tiles_cover_top_with_overlap_and_map_back() -> None:
    left, right = tiles_for(cols=2, top=0.6, overlap=0.2)
    assert left[0] == 0.0 and right[2] == 1.0 and left[2] > right[0] and left[3] == 0.6
    mapped = map_from_tile(body(0.5, top=0.5), (0.5, 0.0, 1.0, 0.6))
    assert mapped["bbox"]["x"] == pytest.approx(0.5 + 0.45 * 0.5)
    assert mapped["keypoints"][0][1] == pytest.approx(0.53 * 0.6)


def test_tiled_detector_adds_far_people_and_reuses_them_between_passes() -> None:
    class Fake:
        def __init__(self) -> None:
            self.sizes: list[tuple[int, int]] = []

        async def detect(self, frame: dict) -> list[dict]:
            self.sizes.append(frame["frame_size"])
            return [body(0.5)] if frame["frame_size"] != (100, 100) else [body(0.2, top=0.7)]

    fake = Fake()
    tiled = TiledPoseDetector(fake, every=2, cols=2, top=0.6)
    frame = {"image": np.zeros((100, 100, 3), np.uint8), "frame_size": (100, 100)}
    first = asyncio.run(tiled.detect(frame))
    second = asyncio.run(tiled.detect(frame))
    assert len(fake.sizes) == 4  # full + 2 tiles, then full only
    assert len(first) == 3 and len(second) == 3


def test_still_seated_person_becomes_a_seat_with_stable_number() -> None:
    tracker = SeatMapTracker()
    update = feed(tracker, [[body(0.3), body(0.6)]] * 35)
    assert len(tracker.seats) == 2 and set(tracker.roles.values()) == {"candidate"}
    numbers = sorted(seat.runtime_track_id for seat in update.analyzed)
    feed(tracker, [[body(0.6)]] * 10, start=3_500)  # first candidate missed for a second
    update = feed(tracker, [[body(0.3), body(0.6)]] * 2, start=4_500)
    assert sorted(seat.runtime_track_id for seat in update.analyzed) == numbers


def test_walking_person_never_becomes_a_seat_and_is_walker() -> None:
    tracker = SeatMapTracker()
    feed(tracker, [[body(0.05 + 0.02 * i, height=0.2)] for i in range(40)])
    assert tracker.seats == {} and list(tracker.roles.values()) == ["walker"]


def test_invigilator_standing_still_is_not_a_seat() -> None:
    tracker = SeatMapTracker()
    feed(tracker, [[body(0.5, top=0.05, height=0.3, width=0.08)]] * 60)  # tall, narrow
    assert tracker.seats == {} and list(tracker.roles.values()) == ["walker"]


def test_empty_seat_after_long_occupancy_is_left_and_rules_report_it() -> None:
    tracker = SeatMapTracker()
    roles = SeatMapRoles(tracker)
    feed(tracker, [[body(0.3), body(0.6)]] * 150)  # 15 s seated
    seat_id = next(s for s, e in tracker.seats.items() if e.head[0] < 0.4)
    feed(tracker, [[body(0.6)]] * 40, start=15_000)
    assert not roles.left_seat(seat_id, 18_000)
    assert roles.left_seat(seat_id, 18_900 + 1_200)


def test_brief_miss_in_far_row_is_not_leaving() -> None:
    tracker = SeatMapTracker()
    feed(tracker, [[body(0.3)] if i % 3 else [] for i in range(150)])  # detector flickers
    seat_id = next(iter(tracker.seats))
    feed(tracker, [[]] * 20, start=15_000)
    assert not tracker.left_seat(seat_id, 17_000)


def test_person_leaning_near_seat_is_not_leaving() -> None:
    tracker = SeatMapTracker()
    feed(tracker, [[body(0.3), body(0.6)]] * 150)
    seat_id = next(s for s, e in tracker.seats.items() if e.head[0] < 0.4)
    # Leans far enough to drop out of the seat radius but stays at the desk.
    feed(tracker, [[body(0.39, height=0.45, width=0.1), body(0.6)]] * 80, start=15_000)
    assert not tracker.left_seat(seat_id, 23_000)
