from __future__ import annotations

from uuid import uuid4

from cctv_kps import person
from proctoring_ai.multi.cctv.review_detectors import ReviewObject
from proctoring_ai.multi.cctv.review_rules import (
    ReviewFrame,
    ReviewRuleConfig,
    assign_poses,
    review_window,
)
from proctoring_ai.multi.cctv.suspects import SuspectWindow
from proctoring_core.observations import Box


def box(cx: float, top: float = 0.3) -> Box:
    return Box(cx - 0.05, top, 0.1, 0.2)


def window(seats: dict, main, start: int, end: int, reasons=("OBJECT_HINT",)) -> SuspectWindow:
    others = tuple(key for key in seats if key != main)
    return SuspectWindow(main, 1, others, start, end, tuple(reasons), seats[main], seats)


def frames(seats: dict, pose_at, start: int, end: int, objects_at=lambda ms: (), step: int = 100):
    return [ReviewFrame(ms, seats, {key: pose_at(key, ms) for key in seats}, list(objects_at(ms)), 1.0)
            for ms in range(start, end, step)]


def kinds(records, action="opened"):
    return [item["type"] for item in records if item["action"] == action]


def wristed(kps, left=None, right=None):
    if left is not None:
        kps[9] = [left[0], left[1], 0.9]
    if right is not None:
        kps[10] = [right[0], right[1], 0.9]
    return kps


def phone(x: float, y: float, conf: float = 0.6) -> ReviewObject:
    return ReviewObject("phone", conf, Box(x - 0.01, y - 0.01, 0.02, 0.02), "yolo11x")


def test_phone_near_wrist_three_frames_opens() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    pose = lambda key, ms: wristed(person(cx=0.3, top=0.3), left=(0.3, 0.42))  # noqa: E731
    records = review_window(window(seats, a, 0, 1000),
                            frames(seats, pose, 0, 1000, lambda ms: [phone(0.3, 0.42)] if ms in (300, 500, 700) else []))
    assert kinds(records) == ["PHONE_DETECTED"]
    assert records[0]["stage"] == "review" and records[0]["seat_id"] == str(a)


def test_phone_two_frames_only_dismissed() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    pose = lambda key, ms: wristed(person(cx=0.3, top=0.3), left=(0.3, 0.42))  # noqa: E731
    records = review_window(window(seats, a, 0, 1000),
                            frames(seats, pose, 0, 1000, lambda ms: [phone(0.3, 0.42)] if ms in (300, 800) else []))
    assert [item["action"] for item in records] == ["dismissed"]


def test_passing_by_wrists_reports_only_the_window_seat_when_nobody_reaches_over() -> None:
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.38)}  # overlapping seat boxes, midline x = 0.34

    def pose(key, ms):  # each wrist is inside the other's box but short of the midline
        if key == a:
            return wristed(person(cx=0.3, top=0.3), left=(0.335, 0.42))
        return wristed(person(cx=0.38, top=0.3), left=(0.35, 0.42))

    records = review_window(window(seats, a, 0, 500, ("HANDS_MEET",)), frames(seats, pose, 0, 500))
    passing = [item for item in records if item["action"] == "opened" and item["type"] == "PASSING_OBJECT"]
    assert [(item["seat_id"], item["neighbor_seat_id"]) for item in passing] == [(str(a), str(b))]


def test_passing_by_object_transfer() -> None:
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.5)}
    paper = lambda x: ReviewObject("paper", 0.5, Box(x - 0.02, 0.4, 0.04, 0.03), "x")  # noqa: E731

    def objects(ms):
        if 800 <= ms <= 1000:
            return [paper(0.3)]
        if 2500 <= ms <= 2700:
            return [paper(0.5)]
        return []

    def pose(key, ms):
        cx = seats[key].center[0]
        holding = (key == a and 800 <= ms <= 1000) or (key == b and 2500 <= ms <= 2700)
        return wristed(person(cx=cx, top=0.3), left=(cx, 0.415) if holding else None)

    records = review_window(window(seats, a, 0, 3000), frames(seats, pose, 0, 3000, objects))
    assert kinds(records) == ["PASSING_OBJECT"]


def test_document_below_desk_line_opens() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    book = ReviewObject("book", 0.5, Box(0.28, 0.46, 0.04, 0.02), "yolo11x")
    holding = lambda key, ms: wristed(person(cx=0.3, top=0.3), left=(0.3, 0.47))  # noqa: E731
    records = review_window(window(seats, a, 0, 1500), frames(seats, holding, 0, 1500, lambda ms: [book]))
    assert kinds(records) == ["DOCUMENT_DETECTED"]


def test_exam_paper_on_desk_in_front_not_document() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    book = ReviewObject("book", 0.5, Box(0.28, 0.39, 0.04, 0.02), "yolo11x")
    records = review_window(window(seats, a, 0, 1500),
                            frames(seats, lambda key, ms: person(cx=0.3, top=0.3), 0, 1500, lambda ms: [book]))
    assert "DOCUMENT_DETECTED" not in kinds(records)


def test_look_at_lap_needs_three_seconds() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    lap = lambda key, ms: wristed(person(cx=0.3, top=0.3, bent=True), left=(0.28, 0.55), right=(0.32, 0.55))  # noqa: E731
    short = review_window(window(seats, a, 0, 2500, ("HEAD_DOWN_HANDS_LOW",)), frames(seats, lap, 0, 2500))
    assert "LOOK_AT_LAP" not in kinds(short)
    long = review_window(window(seats, a, 0, 3300, ("HEAD_DOWN_HANDS_LOW",)), frames(seats, lap, 0, 3300))
    assert kinds(long) == ["LOOK_AT_LAP"]


def test_head_turn_reuses_existing_rules() -> None:
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.45)}

    def pose(key, ms):
        return person(cx=seats[key].center[0], top=0.3, nose_dx=0.6 if key == a and ms >= 8000 else 0.0)

    records = review_window(window(seats, a, 7000, 10_000, ("HEAD_AWAY",)), frames(seats, pose, 0, 10_000))
    assert kinds(records) == ["LOOK_AT_NEIGHBOR"]
    assert records[0]["video_offset_ms"] >= 7000


def test_unclassified_needs_two_reasons() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    calm = lambda key, ms: person(cx=0.3, top=0.3)  # noqa: E731
    records = review_window(window(seats, a, 0, 1000, ("HEAD_AWAY", "LEAN")), frames(seats, calm, 0, 1000))
    assert kinds(records) == ["UNCLASSIFIED_SUSPECT"]


def test_nothing_matched_gives_dismissed() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    calm = lambda key, ms: person(cx=0.3, top=0.3)  # noqa: E731
    records = review_window(window(seats, a, 0, 1000, ("LEAN",)), frames(seats, calm, 0, 1000))
    assert records == [{"action": "dismissed", "seat_id": str(a), "start_ms": 0, "end_ms": 1000,
                        "reasons": ["LEAN"], "stage": "review"}]


def test_assign_poses_picks_head_inside_seat() -> None:
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.5)}
    people = [{"keypoints": person(cx=0.5, top=0.3)}, {"keypoints": person(cx=0.3, top=0.3)},
              {"keypoints": person(cx=0.9, top=0.3)}]
    assigned = assign_poses(seats, people)
    assert assigned[a] is people[1]["keypoints"] and assigned[b] is people[0]["keypoints"]


def test_faceless_neighbour_does_not_take_the_seat() -> None:
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.42)}
    own = person(cx=0.3, top=0.29)  # face visible, head a little high in the box
    neighbour = person(cx=0.42, top=0.3, face=False)
    people = [{"keypoints": neighbour, "bbox": {"x": 0.37, "y": 0.3, "width": 0.1, "height": 0.2}},
              {"keypoints": own, "bbox": {"x": 0.25, "y": 0.26, "width": 0.1, "height": 0.2}}]
    assigned = assign_poses(seats, people)
    assert assigned[a] is own and assigned[b] is neighbour


def test_paper_flicker_between_desks_is_not_passing() -> None:
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.42)}
    paper = lambda x: ReviewObject("book", 0.5, Box(x - 0.02, 0.44, 0.04, 0.03), "x")  # noqa: E731
    records = review_window(window(seats, a, 0, 3000),
                            frames(seats, lambda key, ms: person(cx=seats[key].center[0], top=0.3), 0, 3000,
                                   lambda ms: [paper(0.3 if (ms // 200) % 2 else 0.42)]))
    assert "PASSING_OBJECT" not in kinds(records)


def test_behaviour_that_began_in_warmup_is_reported_at_window_start() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    lap = lambda key, ms: wristed(person(cx=0.3, top=0.3, bent=True), left=(0.28, 0.55), right=(0.32, 0.55))  # noqa: E731
    records = review_window(window(seats, a, 4000, 6000, ("HEAD_DOWN_HANDS_LOW",)), frames(seats, lap, 0, 6000))
    lap_events = [item for item in records if item["action"] == "opened" and item["type"] == "LOOK_AT_LAP"]
    assert [item["video_offset_ms"] for item in lap_events] == [4000]


def test_turn_latched_in_warmup_is_reported_at_window_start() -> None:
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.45)}

    def pose(key, ms):
        return person(cx=seats[key].center[0], top=0.3, nose_dx=0.6 if key == a and ms >= 6000 else 0.0)

    records = review_window(window(seats, a, 8500, 10_000, ("HEAD_AWAY",)), frames(seats, pose, 0, 10_000))
    assert kinds(records) == ["LOOK_AT_NEIGHBOR"] and records[0]["video_offset_ms"] == 8500


def test_fast_tier_rule_event_counts_as_a_reason() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    calm = lambda key, ms: person(cx=0.3, top=0.3)  # noqa: E731
    records = review_window(window(seats, a, 0, 1000, ("LEAN", "RULE_EVENT")), frames(seats, calm, 0, 1000))
    assert kinds(records) == ["UNCLASSIFIED_SUSPECT"]


def test_event_head_is_taken_at_event_time() -> None:
    a = uuid4()
    seats = {a: box(0.3)}

    def pose(key, ms):
        kps = wristed(person(cx=0.3 if ms < 1500 else 0.33, top=0.3), left=(0.3, 0.42))
        return kps

    records = review_window(window(seats, a, 0, 3000),
                            frames(seats, pose, 0, 3000, lambda ms: [phone(0.3, 0.42)] if ms < 600 else []))
    opened = [item for item in records if item["action"] == "opened"]
    assert opened[0]["head"][0] < 0.31


def test_calculator_lying_on_desk_by_the_hand_is_not_phone() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    pose = lambda key, ms: wristed(person(cx=0.3, top=0.3), left=(0.3, 0.45))  # noqa: E731
    records = review_window(window(seats, a, 0, 1000),
                            frames(seats, pose, 0, 1000, lambda ms: [phone(0.31, 0.455)]))
    assert "PHONE_DETECTED" not in kinds(records)


def test_calculator_next_to_exam_paper_is_not_phone() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    paper = ReviewObject("book", 0.4, Box(0.24, 0.47, 0.06, 0.03), "yolo11x")
    pose = lambda key, ms: wristed(person(cx=0.3, top=0.3), left=(0.3, 0.48))  # noqa: E731
    records = review_window(window(seats, a, 0, 1000),
                            frames(seats, pose, 0, 1000, lambda ms: [phone(0.31, 0.49), paper]))
    assert "PHONE_DETECTED" not in kinds(records)


def test_object_above_the_head_on_the_desk_in_front_is_not_phone() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    pose = lambda key, ms: wristed(person(cx=0.3, top=0.3), left=(0.3, 0.33))  # noqa: E731
    records = review_window(window(seats, a, 0, 1000),
                            frames(seats, pose, 0, 1000, lambda ms: [phone(0.31, 0.31)]))
    assert "PHONE_DETECTED" not in kinds(records)


def test_sparse_frames_keep_lap_hold():
    """Frames 333 ms apart (live deep tier) must not cut a 4 s run into pieces shorter than the hold."""

    seats = {uuid4(): box(0.5)}
    main = next(iter(seats))

    def pose(_key, ms):
        kps = person(0.5, 0.3, bent=True)
        return wristed(kps, left=(0.47, 0.52), right=(0.53, 0.52)) if ms >= 6000 else wristed(kps, left=(0.47, 0.42))

    sparse = [f for f in frames(seats, pose, 0, 10000) if f.offset_ms % 333 < 100]
    records = review_window(window(seats, main, 6000, 10000, ("HEAD_DOWN_HANDS_LOW",)), sparse)
    assert "LOOK_AT_LAP" in kinds(records)


def test_passing_only_reacher() -> None:
    """Nguyên (phase 2d): only the candidate who reaches over is flagged, not the neighbour."""

    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.42)}  # midline x = 0.36

    def pose(key, ms):
        if key == a:
            return wristed(person(cx=0.3, top=0.3), left=(0.31, 0.42))
        return wristed(person(cx=0.42, top=0.3), left=(0.33, 0.42))  # b's hand is over a's desk

    records = review_window(window(seats, a, 0, 500, ("HANDS_MEET",)), frames(seats, pose, 0, 500))
    passing = [item for item in records if item["action"] == "opened" and item["type"] == "PASSING_OBJECT"]
    assert [(item["seat_id"], item["neighbor_seat_id"]) for item in passing] == [(str(b), str(a))]


def test_passing_both_reach_farther_wins() -> None:
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.42)}

    def pose(key, ms):
        if key == a:
            return wristed(person(cx=0.3, top=0.3), left=(0.37, 0.42))  # 0.01 over the midline
        return wristed(person(cx=0.42, top=0.3), left=(0.34, 0.42))  # 0.02 over

    records = review_window(window(seats, a, 0, 500, ("HANDS_MEET",)), frames(seats, pose, 0, 500))
    assert [item["seat_id"] for item in records if item["action"] == "opened"] == [str(b)]


def test_still_phone_far_from_wrist_ignored() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    pose = lambda key, ms: wristed(person(cx=0.3, top=0.3), left=(0.26, 0.42))  # noqa: E731
    still = phone(0.34, 0.42)  # 0.08 = 0.8 seat widths from the wrist, never moves
    records = review_window(window(seats, a, 0, 1200), frames(seats, pose, 0, 1200, lambda ms: [still]))
    assert "PHONE_DETECTED" not in kinds(records)


def test_held_phone_reported() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    pose = lambda key, ms: wristed(person(cx=0.3, top=0.3), left=(0.3, 0.42))  # noqa: E731
    records = review_window(window(seats, a, 0, 1000),
                            frames(seats, pose, 0, 1000, lambda ms: [phone(0.303, 0.42)] if ms in (300, 500, 700) else []))
    assert kinds(records) == ["PHONE_DETECTED"]


def test_single_frame_object_ignored() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    book = ReviewObject("book", 0.5, Box(0.28, 0.46, 0.04, 0.02), "yolo11x")
    holding = lambda key, ms: wristed(person(cx=0.3, top=0.3), left=(0.3, 0.47))  # noqa: E731
    records = review_window(window(seats, a, 0, 1500, ("LEAN",)),
                            frames(seats, holding, 0, 1500, lambda ms: [book] if ms == 700 else []))
    assert "DOCUMENT_DETECTED" not in kinds(records)


def test_moving_book_reported() -> None:
    a = uuid4()
    seats = {a: box(0.3)}

    def book(ms):  # slides 0.03 (0.75 of its width) under the desk, nobody's wrist near it
        return [ReviewObject("book", 0.5, Box(0.26 + 0.03 * ms / 1500, 0.46, 0.04, 0.02), "yolo11x")]

    records = review_window(window(seats, a, 0, 1500),
                            frames(seats, lambda key, ms: person(cx=0.3, top=0.3), 0, 1500, book))
    assert kinds(records) == ["DOCUMENT_DETECTED"]


def test_still_book_under_desk_without_hand_ignored() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    book = ReviewObject("book", 0.5, Box(0.28, 0.46, 0.04, 0.02), "yolo11x")
    records = review_window(window(seats, a, 0, 1500, ("LEAN",)),
                            frames(seats, lambda key, ms: person(cx=0.3, top=0.3), 0, 1500, lambda ms: [book]))
    assert "DOCUMENT_DETECTED" not in kinds(records)


def _pair(height: float):
    a, b = uuid4(), uuid4()
    return a, b, {a: Box(0.25, 0.3, 0.1, height), b: Box(0.37, 0.3, 0.1, height)}


def _meeting(a, wrist_a, wrist_b, conf=0.9):
    def pose(key, ms):
        kps = person(cx=0.3 if key == a else 0.42, top=0.3)
        kps[9] = [*(wrist_a if key == a else wrist_b), conf]
        return kps

    return pose


def test_far_row_hands_meet_without_object_ignored() -> None:
    a, b, seats = _pair(0.08)
    pose = _meeting(a, (0.355, 0.33), (0.36, 0.33))  # touching at the midline, nobody inside the other seat
    records = review_window(window(seats, a, 0, 1000, ("HANDS_MEET",)), frames(seats, pose, 0, 1000))
    assert "PASSING_OBJECT" not in kinds(records)


def test_far_row_wrist_inside_other_seat_reported() -> None:
    a, b, seats = _pair(0.08)
    pose = _meeting(a, (0.375, 0.33), (0.38, 0.33))  # a's hand is in b's seat box
    records = review_window(window(seats, a, 0, 1000, ("HANDS_MEET",)), frames(seats, pose, 0, 1000))
    assert "PASSING_OBJECT" in kinds(records)


def test_near_row_hands_meet_reported() -> None:
    a, b, seats = _pair(0.25)
    pose = _meeting(a, (0.37, 0.42), (0.38, 0.42))  # touching across the border: a's hand is in b's seat
    records = review_window(window(seats, a, 0, 1000, ("HANDS_MEET",)), frames(seats, pose, 0, 1000))
    assert "PASSING_OBJECT" in kinds(records)


def test_low_conf_wrist_ignored() -> None:
    a, b, seats = _pair(0.25)
    pose = _meeting(a, (0.35, 0.42), (0.36, 0.42), conf=0.45)
    records = review_window(window(seats, a, 0, 1000, ("HANDS_MEET",)), frames(seats, pose, 0, 1000))
    assert "PASSING_OBJECT" not in kinds(records)


def _lap(left, right):
    return lambda key, ms: wristed(person(cx=0.3, top=0.3, bent=True), left=left, right=right)


def test_writer_one_hand_on_desk_not_lap() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    pose = _lap((0.28, 0.42), (0.32, 0.55))  # writing hand on the desk, the other in the lap
    records = review_window(window(seats, a, 0, 4000, ("HEAD_DOWN_HANDS_LOW",)), frames(seats, pose, 0, 4000))
    assert "LOOK_AT_LAP" not in kinds(records)


def test_both_hands_below_desk_is_lap() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    records = review_window(window(seats, a, 0, 4000, ("HEAD_DOWN_HANDS_LOW",)),
                            frames(seats, _lap((0.28, 0.55), (0.32, 0.55)), 0, 4000))
    assert "LOOK_AT_LAP" in kinds(records)


def _elbowed(pose_at, y: float = 0.55, conf: float = 0.9):
    """Both elbows at height ``y`` (COCO 7 and 8) on top of ``pose_at``'s keypoints."""

    def pose(key, ms):
        kps = pose_at(key, ms)
        if kps is not None:
            kps[7] = [kps[5][0], y, conf]
            kps[8] = [kps[6][0], y, conf]
        return kps

    return pose


def test_hands_hidden_with_low_elbows_is_lap() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    records = review_window(window(seats, a, 0, 4000, ("HEAD_DOWN_HANDS_LOW",)),
                            frames(seats, _elbowed(_lap(None, None)), 0, 4000))
    assert "LOOK_AT_LAP" in kinds(records)


def test_hands_hidden_without_elbows_is_not_lap() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    records = review_window(window(seats, a, 0, 4000, ("HEAD_DOWN_HANDS_LOW",)),
                            frames(seats, _lap(None, None), 0, 4000))
    assert "LOOK_AT_LAP" not in kinds(records)


def test_wrist_on_exam_paper_below_line_is_not_lap() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    paper = ReviewObject("paper", 0.6, Box(0.25, 0.33, 0.1, 0.16), "yolo11x")  # lies on the desk, centre above the line
    records = review_window(window(seats, a, 0, 4000, ("HEAD_DOWN_HANDS_LOW",)),
                            frames(seats, _lap((0.28, 0.5), (0.32, 0.5)), 0, 4000, lambda ms: [paper]))
    assert "LOOK_AT_LAP" not in kinds(records)


def test_paper_in_lap_does_not_block_lap() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    paper = ReviewObject("paper", 0.6, Box(0.25, 0.42, 0.1, 0.09), "yolo11x")  # centre below the line
    records = review_window(window(seats, a, 0, 4000, ("HEAD_DOWN_HANDS_LOW",)),
                            frames(seats, _lap((0.28, 0.5), (0.32, 0.5)), 0, 4000, lambda ms: [paper]))
    assert "LOOK_AT_LAP" in kinds(records)


def test_wrist_moving_along_the_line_is_not_lap() -> None:
    a = uuid4()
    seats = {a: box(0.3)}

    def pose(_key, ms):
        return wristed(person(cx=0.3, top=0.3, bent=True), left=(0.26 + 0.08 * ms / 4000, 0.55), right=(0.32, 0.55))

    records = review_window(window(seats, a, 0, 4000, ("HEAD_DOWN_HANDS_LOW",)), frames(seats, pose, 0, 4000),
                            ReviewRuleConfig(lap_still_widths=0.15))
    assert "LOOK_AT_LAP" not in kinds(records)


def _armed(pose_at, elbow_y: float, wrist_y: float, other_wrist_y: float | None = None):
    """Elbows at ``elbow_y`` under the shoulders, wrists at ``wrist_y`` (right one at ``other_wrist_y``)."""

    def pose(key, ms):
        kps = pose_at(key, ms)
        kps[7], kps[8] = [kps[5][0], elbow_y, 0.9], [kps[6][0], elbow_y, 0.9]
        kps[9] = [kps[5][0] + 0.01, wrist_y, 0.9]
        kps[10] = [kps[6][0] - 0.01, wrist_y if other_wrist_y is None else other_wrist_y, 0.9]
        return kps

    return pose


# Clip 1 (Task 3b): the seat box ends at the desk edge, so a writer's hands lie below the default desk line
# (0.8 of the box) and the line never learns from them. The forearm tells: on the desk it is about level
# (wrist 0.05-0.23 shoulder widths below the elbow), in the lap it hangs (0.36-0.46).
def test_writer_forearms_level_below_default_line_is_not_lap() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    pose = _armed(_lap(None, None), elbow_y=0.5, wrist_y=0.52)
    records = review_window(window(seats, a, 0, 4000, ("HEAD_DOWN_HANDS_LOW",)), frames(seats, pose, 0, 4000))
    assert "LOOK_AT_LAP" not in kinds(records)


def test_forearms_hanging_into_the_lap_is_lap() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    pose = _armed(_lap(None, None), elbow_y=0.48, wrist_y=0.53)
    records = review_window(window(seats, a, 0, 4000, ("HEAD_DOWN_HANDS_LOW",)), frames(seats, pose, 0, 4000))
    assert "LOOK_AT_LAP" in kinds(records)


def test_one_level_forearm_blocks_lap() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    pose = _armed(_lap(None, None), elbow_y=0.48, wrist_y=0.53, other_wrist_y=0.49)
    records = review_window(window(seats, a, 0, 4000, ("HEAD_DOWN_HANDS_LOW",)), frames(seats, pose, 0, 4000))
    assert "LOOK_AT_LAP" not in kinds(records)


def test_forearm_test_switched_off_keeps_old_lap() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    pose = _armed(_lap(None, None), elbow_y=0.5, wrist_y=0.52)
    records = review_window(window(seats, a, 0, 4000, ("HEAD_DOWN_HANDS_LOW",)), frames(seats, pose, 0, 4000),
                            ReviewRuleConfig(lap_forearm_drop=None))
    assert "LOOK_AT_LAP" in kinds(records)


def test_empty_seat_gives_no_lap() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    records = review_window(window(seats, a, 0, 4000, ("HEAD_DOWN_HANDS_LOW",)),
                            frames(seats, lambda key, ms: None, 0, 4000))
    assert "LOOK_AT_LAP" not in kinds(records)


def test_passing_ignores_one_arm_given_to_both_people() -> None:
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.42)}

    def pose(key, ms):
        cx = 0.3 if key == a else 0.42
        return wristed(person(cx=cx, top=0.3), left=(0.36, 0.42))  # the very same spot

    records = review_window(window(seats, a, 0, 500, ("HANDS_MEET",)), frames(seats, pose, 0, 500))
    assert "PASSING_OBJECT" not in kinds(records)


def test_passing_ignores_long_side_by_side_hands_without_object() -> None:
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.42)}

    def pose(key, ms):
        if key == a:
            return wristed(person(cx=0.3, top=0.3), left=(0.36, 0.42))
        return wristed(person(cx=0.42, top=0.3), left=(0.375, 0.42))

    records = review_window(window(seats, a, 0, 5000, ("HANDS_MEET",)), frames(seats, pose, 0, 5000))
    assert "PASSING_OBJECT" not in kinds(records)


def test_document_below_desk_needs_a_hand_on_it() -> None:
    # Clip 1 at 10.0 s and 32.0 s: the paper lies under the desk line, the hand that writes is a third of
    # a seat height above it. Only a document a hand holds is cheating material.
    a = uuid4()
    seats = {a: box(0.3)}
    book = ReviewObject("book", 0.5, Box(0.28, 0.46, 0.04, 0.02), "yolo11x")

    def pose(key, ms):
        wrist = (0.3, 0.47) if ms == 0 else (0.3, 0.41)  # touched once, then written above it
        return wristed(person(cx=0.3, top=0.3), left=wrist)

    records = review_window(window(seats, a, 0, 2000), frames(seats, pose, 0, 2000, lambda ms: [book]))
    assert "DOCUMENT_DETECTED" not in kinds(records)


def test_near_row_hands_a_fifth_of_a_seat_apart_reported() -> None:
    # Clip 1 at 25.5 s (true): the handover wrists stayed 0.13-0.2 seat heights apart.
    a, b, seats = _pair(0.25)
    pose = _meeting(a, (0.375, 0.42), (0.42, 0.42))  # 0.045 = 0.18 seat heights, a's hand in b's seat
    records = review_window(window(seats, a, 0, 1000, ("HANDS_MEET",)), frames(seats, pose, 0, 1000))
    assert "PASSING_OBJECT" in kinds(records)


def _opened(records, kind):
    return [item for item in records if item["action"] == "opened" and item["type"] == kind]


def test_passing_hands_only_evidence() -> None:
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.42)}

    def pose(key, ms):
        if key == a:
            return wristed(person(cx=0.3, top=0.3), left=(0.33, 0.42))
        return wristed(person(cx=0.42, top=0.3), left=(0.345, 0.42))  # b's hand is in a's seat

    records = review_window(window(seats, a, 0, 500, ("HANDS_MEET",)), frames(seats, pose, 0, 500))
    assert [item["evidence_kind"] for item in _opened(records, "PASSING_OBJECT")] == ["hands_only"]


def test_passing_object_between_hands_evidence() -> None:
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.42)}

    def pose(key, ms):
        if key == a:
            return wristed(person(cx=0.3, top=0.3), left=(0.36, 0.42))
        return wristed(person(cx=0.42, top=0.3), left=(0.375, 0.42))

    objects = lambda ms: [ReviewObject("paper", 0.5, Box(0.36, 0.41, 0.02, 0.02), "x")]  # noqa: E731
    records = review_window(window(seats, a, 0, 500, ("HANDS_MEET",)), frames(seats, pose, 0, 500, objects))
    assert [item["evidence_kind"] for item in _opened(records, "PASSING_OBJECT")] == ["object_between"]


def test_passing_transfer_evidence() -> None:
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.5)}
    paper = lambda x: ReviewObject("paper", 0.5, Box(x - 0.02, 0.4, 0.04, 0.03), "x")  # noqa: E731

    def objects(ms):
        if 800 <= ms <= 1000:
            return [paper(0.3)]
        if 2500 <= ms <= 2700:
            return [paper(0.5)]
        return []

    def pose(key, ms):
        cx = seats[key].center[0]
        holding = (key == a and 800 <= ms <= 1000) or (key == b and 2500 <= ms <= 2700)
        return wristed(person(cx=cx, top=0.3), left=(cx, 0.415) if holding else None)

    records = review_window(window(seats, a, 0, 3000), frames(seats, pose, 0, 3000, objects))
    assert [item["evidence_kind"] for item in _opened(records, "PASSING_OBJECT")] == ["transfer"]


def _phone_run(y: float, confs=(0.5, 0.8, 0.6)):
    a = uuid4()
    seats = {a: box(0.3)}
    pose = lambda key, ms: wristed(person(cx=0.3, top=0.3), left=(0.3, y))  # noqa: E731
    by_ms = dict(zip((300, 500, 700), confs, strict=True))
    records = review_window(window(seats, a, 0, 1000),
                            frames(seats, pose, 0, 1000, lambda ms: [phone(0.3, y, by_ms[ms])] if ms in by_ms else []))
    return _opened(records, "PHONE_DETECTED")


def test_phone_raised_above_desk_edge_with_max_confidence() -> None:
    (item,) = _phone_run(0.40)
    assert item["raised"] is True
    assert item["confidence"] == 0.8


def test_phone_on_desk_is_not_raised() -> None:
    # Desk edge defaults to 0.46 here; 0.43 clears the event gate (0.44) but not the raised margin (0.42).
    (item,) = _phone_run(0.43)
    assert item["raised"] is False
    assert item["confidence"] == 0.8


def test_head_rules_use_one_event_loop_per_window(monkeypatch) -> None:
    """A loop per frame opened thousands of loops; on Windows each needs a loopback socketpair and CI hung."""

    import asyncio.events

    made = []
    real = asyncio.events.new_event_loop
    monkeypatch.setattr(asyncio.events, "new_event_loop", lambda: made.append(1) or real())
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.45)}

    def pose(key, ms):
        return person(cx=seats[key].center[0], top=0.3, nose_dx=0.6 if key == a and ms >= 8000 else 0.0)

    records = review_window(window(seats, a, 7000, 10_000, ("HEAD_AWAY",)), frames(seats, pose, 0, 10_000))
    assert kinds(records) == ["LOOK_AT_NEIGHBOR"]
    assert len(made) <= 1


def _phone_over_paper(on_paper_ms, cfg: ReviewRuleConfig | None = None, sightings=(100, 300, 500, 700, 900)):
    """A phone held at the wrist in every sighting; the seat's paper is under it only at ``on_paper_ms``.
    The paper is tall, so its grown top edge sits just above the phone and the desk-top gate still lets it through."""

    a = uuid4()
    seats = {a: box(0.3)}
    pose = lambda key, ms: wristed(person(cx=0.3, top=0.3), left=(0.3, 0.337))  # noqa: E731
    paper = ReviewObject("paper", 0.8, Box(0.25, 0.36, 0.1, 0.26), "yolo11x")

    def objects(ms):
        if ms not in sightings:
            return []
        return [phone(0.3, 0.337), *([paper] if ms in on_paper_ms else [])]

    records = review_window(window(seats, a, 0, 1000), frames(seats, pose, 0, 1000, objects), cfg)
    return _opened(records, "PHONE_DETECTED")


def test_phone_on_paper_most_of_the_time_is_calculator() -> None:
    assert _phone_over_paper({100, 300, 500, 700}) == []


def test_phone_lifted_off_paper_is_phone() -> None:
    (item,) = _phone_over_paper({100})
    assert item["aspect"] == 1.0


def test_single_sighting_on_paper_does_not_crash() -> None:
    assert _phone_over_paper({500}, sightings=(500,)) == []


def test_phone_paper_fraction_above_one_switches_the_rule_off() -> None:
    assert len(_phone_over_paper({100, 300, 500, 700}, ReviewRuleConfig(phone_paper_fraction=1.01))) == 1


def test_near_row_hands_meet_without_crossing_is_not_passing() -> None:
    # Two neighbours writing at the shared desk edge: wrists close, each in their own seat.
    a, b, seats = _pair(0.25)
    pose = _meeting(a, (0.345, 0.42), (0.365, 0.42))
    records = review_window(window(seats, a, 0, 1000, ("HANDS_MEET",)), frames(seats, pose, 0, 1000))
    assert "PASSING_OBJECT" not in kinds(records)


def test_near_row_wrist_inside_neighbour_seat_is_passing() -> None:
    a, b, seats = _pair(0.25)
    pose = _meeting(a, (0.345, 0.42), (0.365, 0.42))
    crossed = _meeting(a, (0.375, 0.42), (0.385, 0.42))

    def reach(key, ms):  # a's hand goes into b's seat for two frames of the meeting
        return crossed(key, ms) if ms in (400, 500) else pose(key, ms)

    records = review_window(window(seats, a, 0, 1000, ("HANDS_MEET",)), frames(seats, reach, 0, 1000))
    assert "PASSING_OBJECT" in kinds(records)


def _short_turn(cfg: ReviewRuleConfig | None = None):
    a, b = uuid4(), uuid4()
    seats = {a: box(0.3), b: box(0.45)}

    def pose(key, ms):  # a looks at b for 1.2 s
        return person(cx=seats[key].center[0], top=0.3, nose_dx=0.6 if key == a and 8000 <= ms < 9200 else 0.0)

    return review_window(window(seats, a, 7000, 10_000, ("HEAD_AWAY",)), frames(seats, pose, 0, 10_000), cfg)


def test_turn_shorter_than_review_turn_hold_is_ignored() -> None:
    assert "LOOK_AT_NEIGHBOR" not in kinds(_short_turn())


def test_review_turn_hold_reaches_head_rules() -> None:
    assert "LOOK_AT_NEIGHBOR" in kinds(_short_turn(ReviewRuleConfig(turn_hold_ms=1000)))
