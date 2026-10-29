from __future__ import annotations

import asyncio
from uuid import uuid4

from cctv_kps import person
from proctoring_ai.multi.cctv.rules import CctvSeatRules, keypoints_for
from proctoring_ai.multi.pipeline import FrameTiming, MultiFrameResult
from proctoring_ai.multi.tracker import Seat, TrackerUpdate
from proctoring_core.observations import Box

TIMING = FrameTiming(0.0, 0.0, 0.0, {}, 0.0, 0.0, 0, 0, False)


def seat(cx: float, runtime_id: int, *, top: float = 0.3) -> Seat:
    box = Box(cx - 0.05, top, 0.1, 0.2)
    return Seat(uuid4(), runtime_id, box, box.center, 0.9, 0, 0, True)


def frame(ms: int, seats, poses, objects=(), expired=()) -> MultiFrameResult:
    update = TrackerUpdate(list(seats), [], [], list(expired), 0, 0)
    return MultiFrameResult(ms // 100, ms, update, {}, {}, {}, list(objects), TIMING,
                            {s.seat_id: kps for s, kps in zip(seats, poses, strict=True) if kps is not None})


def play(rules: CctvSeatRules, seats, pose_at, start: int, end: int, step: int = 100, objects=()) -> list[dict]:
    events = []
    for ms in range(start, end, step):
        events += [item for _, item in asyncio.run(
            rules.step(frame(ms, seats, [pose_at(s, ms) for s in seats], objects), {}, ms))]
    return events


def opened(events, kind=None):
    return [e for e in events if e["action"] == "opened" and (kind is None or e["type"] == kind)]


def calm(s: Seat, _ms: int):
    return person(cx=s.bbox.x + 0.05, top=s.bbox.y)


def test_settling_and_quiet_candidates_raise_nothing() -> None:
    rules, a, b = CctvSeatRules(), seat(0.3, 1), seat(0.5, 2)
    assert opened(play(rules, [a, b], calm, 0, 12_000)) == []
    assert rules.seat_roles() == {a.seat_id: "candidate", b.seat_id: "candidate"}


def test_turn_towards_neighbour_is_look_at_neighbor_and_closes() -> None:
    rules, a, b = CctvSeatRules(), seat(0.3, 1), seat(0.45, 2)
    play(rules, [a, b], calm, 0, 10_000)

    def turned(s: Seat, ms: int):
        return person(cx=s.bbox.x + 0.05, top=s.bbox.y, nose_dx=0.6 if s is a else 0.0)

    events = play(rules, [a, b], turned, 10_000, 12_000)
    look = opened(events, "LOOK_AT_NEIGHBOR")
    assert [e["seat_id"] for e in look] == [str(a.seat_id)]
    assert look[0]["neighbor_seat_id"] == str(b.seat_id) and look[0]["video_offset_ms"] >= 11_500
    closed = [e for e in play(rules, [a, b], calm, 12_000, 14_000) if e["action"] == "closed"]
    assert [e["type"] for e in closed] == ["LOOK_AT_NEIGHBOR"]


def test_turn_with_no_neighbour_is_head_turn_from_candidate_view() -> None:
    rules, a = CctvSeatRules(), seat(0.3, 1)
    play(rules, [a], calm, 0, 10_000)
    events = play(rules, [a], lambda s, ms: person(cx=0.3, top=0.3, nose_dx=-0.6), 10_000, 12_000)
    assert [e["type"] for e in opened(events)] == ["HEAD_TURN_RIGHT"]


def test_short_glance_is_ignored() -> None:
    rules, a = CctvSeatRules(), seat(0.3, 1)
    play(rules, [a], calm, 0, 10_000)
    events = play(rules, [a], lambda s, ms: person(cx=0.3, top=0.3, nose_dx=0.6), 10_000, 11_000)
    assert opened(events) == []


def test_turn_around_after_face_was_seen() -> None:
    rules, a = CctvSeatRules(), seat(0.3, 1)
    play(rules, [a], calm, 0, 5_000)
    events = play(rules, [a], lambda s, ms: person(cx=0.3, top=0.3, face=False), 5_000, 7_000)
    assert [e["type"] for e in opened(events)] == ["TURN_AROUND"]


def test_heads_together_opens_for_both_seats() -> None:
    rules, a, b = CctvSeatRules(), seat(0.3, 1), seat(0.42, 2)
    play(rules, [a, b], calm, 0, 10_000)

    def lean(s: Seat, ms: int):
        cx = s.bbox.x + 0.05
        return person(cx=cx + (0.04 if s is a else -0.04), top=s.bbox.y)

    together = opened(play(rules, [a, b], lean, 10_000, 12_000), "HEADS_TOGETHER")
    assert sorted(e["seat_id"] for e in together) == sorted([str(a.seat_id), str(b.seat_id)])


def test_wrist_in_neighbour_box_is_hand_activity() -> None:
    rules, a, b = CctvSeatRules(), seat(0.3, 1), seat(0.5, 2)
    play(rules, [a, b], calm, 0, 5_000)

    def reach(s: Seat, ms: int):
        cx = s.bbox.x + 0.05
        return person(cx=cx, top=s.bbox.y, wrist=(0.5, 0.45) if s is a else None)

    hands = opened(play(rules, [a, b], reach, 5_000, 6_500), "SUSPICIOUS_HAND_ACTIVITY")
    assert [e["seat_id"] for e in hands] == [str(a.seat_id)]


def test_phone_goes_to_seat_containing_it() -> None:
    rules, a, b = CctvSeatRules(), seat(0.3, 1), seat(0.5, 2)
    play(rules, [a, b], calm, 0, 5_000)
    phone = {"class": "cell phone", "confidence": 0.5, "bbox": {"x": 0.49, "y": 0.4, "width": 0.02, "height": 0.02}}
    phones = opened(play(rules, [a, b], calm, 5_000, 6_500, objects=[phone]), "PHONE_DETECTED")
    assert [e["seat_id"] for e in phones] == [str(b.seat_id)]


def test_walker_never_raises_events() -> None:
    rules = CctvSeatRules()
    walker = Seat(uuid4(), 9, Box(0.05, 0.2, 0.1, 0.5), (0, 0), 0.9, 0, 0, True)
    events = []
    for index, ms in enumerate(range(0, 8_000, 100)):
        walker.bbox = Box(0.05 + 0.01 * (index % 40) * 2, 0.2, 0.1, 0.5)  # paces back and forth
        kps = person(cx=walker.bbox.x + 0.05, top=0.2, face=index % 20 < 10, nose_dx=0.8)
        events += [i for _, i in asyncio.run(rules.step(frame(ms, [walker], [kps]), {}, ms))]
    assert events == [] and rules.seat_roles()[walker.seat_id] == "walker"


def test_expired_seat_closes_open_events() -> None:
    rules, a = CctvSeatRules(), seat(0.3, 1)
    play(rules, [a], calm, 0, 5_000)
    play(rules, [a], lambda s, ms: person(cx=0.3, top=0.3, face=False), 5_000, 7_000)
    closed = [i for _, i in asyncio.run(rules.step(frame(7_000, [], [], expired=[a]), {}, 7_000))]
    assert [(e["type"], e["close_reason"]) for e in closed] == [("TURN_AROUND", "SEAT_EXPIRED")]


def test_keypoints_follow_best_matching_box() -> None:
    a = seat(0.3, 1)
    people = [{"bbox": a.bbox.as_dict(), "keypoints": [[1, 1, 1]]},
              {"bbox": {"x": 0.8, "y": 0.1, "width": 0.1, "height": 0.1}, "keypoints": [[2, 2, 2]]}]
    assert keypoints_for([a], people) == {a.seat_id: [[1, 1, 1]]}


def test_wrist_over_row_in_front_is_not_hand_activity() -> None:
    # Row behind overlaps the row in front in the image; a writing hand there is not a reach.
    rules, back, front = CctvSeatRules(), seat(0.4, 1, top=0.2), seat(0.42, 2, top=0.35)
    play(rules, [back, front], calm, 0, 5_000)

    def write(s: Seat, ms: int):
        return person(cx=s.bbox.x + 0.05, top=s.bbox.y, wrist=(0.42, 0.42) if s is back else None)

    assert opened(play(rules, [back, front], write, 5_000, 7_000)) == []


def test_turning_towards_row_behind_is_look_at_neighbor() -> None:
    rules, front, behind = CctvSeatRules(), seat(0.3, 1, top=0.4), seat(0.42, 2, top=0.2)
    play(rules, [front, behind], calm, 0, 10_000)

    def turned(s: Seat, ms: int):
        return person(cx=s.bbox.x + 0.05, top=s.bbox.y, nose_dx=0.6 if s is front else 0.0)

    look = opened(play(rules, [front, behind], turned, 10_000, 12_000), "LOOK_AT_NEIGHBOR")
    assert [e["neighbor_seat_id"] for e in look] == [str(behind.seat_id)]


def test_turn_is_caught_when_slow_hardware_analyses_two_frames_a_second() -> None:
    rules, a, b = CctvSeatRules(), seat(0.3, 1), seat(0.45, 2)
    play(rules, [a, b], calm, 0, 10_000, step=500)

    def turned(s: Seat, ms: int):
        return person(cx=s.bbox.x + 0.05, top=s.bbox.y, nose_dx=0.6 if s is a else 0.0)

    events = play(rules, [a, b], turned, 10_000, 13_000, step=500)
    assert [e["seat_id"] for e in opened(events, "LOOK_AT_NEIGHBOR")] == [str(a.seat_id)]


def _rows():
    return CctvSeatRules(), seat(0.3, 1, top=0.4), seat(0.32, 2, top=0.2)


def test_writer_bending_over_row_in_front_is_not_heads_together() -> None:
    rules, front, behind = _rows()
    play(rules, [front, behind], calm, 0, 10_000)

    def bend(s: Seat, ms: int):  # the one behind bends to write: the head drops towards the front row
        return person(cx=s.bbox.x + 0.05, top=s.bbox.y + (0.1 if s is behind else 0.0))

    assert opened(play(rules, [front, behind], bend, 10_000, 13_000), "HEADS_TOGETHER") == []


def test_front_row_turning_back_to_lean_in_is_heads_together() -> None:
    rules, front, behind = _rows()
    play(rules, [front, behind], calm, 0, 10_000)

    def lean_back(s: Seat, ms: int):
        if s is front:
            return person(cx=s.bbox.x + 0.05, top=s.bbox.y - 0.1, face=False)
        return person(cx=s.bbox.x + 0.05, top=s.bbox.y + 0.05)

    together = opened(play(rules, [front, behind], lean_back, 10_000, 13_000), "HEADS_TOGETHER")
    assert len(together) == 2
