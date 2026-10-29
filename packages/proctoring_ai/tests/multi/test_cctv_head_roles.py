from __future__ import annotations

from uuid import uuid4

import pytest
from cctv_kps import person
from proctoring_ai.multi.cctv.head import RollingBaseline, facing, head_point, head_yaw_index
from proctoring_ai.multi.cctv.roles import (
    CANDIDATE,
    PENDING,
    WALKER,
    SeatRoleClassifier,
    is_standing,
)
from proctoring_core.observations import Box


def test_yaw_index_is_nose_offset_in_shoulder_widths() -> None:
    assert head_yaw_index(person(nose_dx=0.0)) == pytest.approx(0.0)
    assert head_yaw_index(person(nose_dx=0.4)) == pytest.approx(0.4)
    assert head_yaw_index(person(face=False)) is None


def test_facing_front_back_unknown() -> None:
    assert facing(person()) == "front"
    assert facing(person(face=False)) == "back"
    assert facing(None) == "unknown"


def test_head_point_falls_back_to_box_top() -> None:
    box = Box(0.4, 0.2, 0.2, 0.5)
    assert head_point([[0, 0, 0]] * 17, box) == pytest.approx((0.5, 0.26))
    x, y = head_point(person(cx=0.5, top=0.3), box)
    assert x == pytest.approx(0.5) and y == pytest.approx(0.33, abs=0.01)


def test_baseline_needs_span_and_samples_then_tracks_median() -> None:
    baseline = RollingBaseline(window_ms=30_000, min_span_ms=5_000, min_samples=8)
    for ms in range(0, 5_000, 500):
        baseline.add(0.1, ms)
        assert baseline.value(ms) is None
    baseline.add(0.9, 5_000)
    assert baseline.value(5_000) == pytest.approx(0.1)


def test_standing_needs_thighs_below_hips() -> None:
    assert is_standing(person(standing=True)) and not is_standing(person())


def run(classifier: SeatRoleClassifier, seat, boxes, start: int = 0, step: int = 100, kps=None) -> str:
    role = PENDING
    for index, box in enumerate(boxes):
        role = classifier.update(seat, box, start + index * step, kps)
    return role


def test_still_seated_track_becomes_candidate_after_settle_time() -> None:
    classifier, seat = SeatRoleClassifier(), uuid4()
    box = Box(0.4, 0.3, 0.1, 0.2)
    assert run(classifier, seat, [box] * 20) == PENDING
    assert run(classifier, seat, [box] * 12, start=2_000) == CANDIDATE


def test_moving_track_is_walker_and_stays_walker() -> None:
    classifier, seat = SeatRoleClassifier(), uuid4()
    boxes = [Box(0.1 + 0.02 * i, 0.3, 0.1, 0.3) for i in range(15)]
    assert run(classifier, seat, boxes) == WALKER
    assert run(classifier, seat, [Box(0.4, 0.3, 0.1, 0.3)] * 60, start=2_000) == WALKER


def test_standing_still_person_is_walker() -> None:
    classifier, seat = SeatRoleClassifier(), uuid4()
    assert run(classifier, seat, [Box(0.4, 0.1, 0.1, 0.4)] * 30, kps=person(standing=True)) == WALKER


def test_candidate_leaves_seat_after_moving_away_for_leave_time() -> None:
    classifier, seat = SeatRoleClassifier(), uuid4()
    home = Box(0.4, 0.3, 0.1, 0.2)
    run(classifier, seat, [home] * 35)
    assert classifier.role(seat) == CANDIDATE
    away = Box(0.7, 0.3, 0.1, 0.2)
    run(classifier, seat, [away] * 40, start=3_500)
    assert not classifier.left_seat(seat, 7_400)
    assert classifier.left_seat(seat, 8_600)
    classifier.update(seat, home, 8_700)
    assert not classifier.left_seat(seat, 8_700)
