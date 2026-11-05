from __future__ import annotations

from uuid import uuid4

from cctv_kps import person
from proctoring_ai.multi.cctv.review_detectors import ReviewObject
from proctoring_ai.multi.cctv.review_rules import ReviewFrame, assign_poses, review_window
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
    seats = {a: box(0.3), b: box(0.42)}

    def pose(key, ms):
        if key == a:
            return wristed(person(cx=0.3, top=0.3), left=(0.36, 0.42))
        return wristed(person(cx=0.42, top=0.3), left=(0.375, 0.42))

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
    pose = _meeting(a, (0.35, 0.42), (0.36, 0.42))
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


def test_hands_hidden_is_lap() -> None:
    a = uuid4()
    seats = {a: box(0.3)}
    records = review_window(window(seats, a, 0, 4000, ("HEAD_DOWN_HANDS_LOW",)),
                            frames(seats, _lap(None, None), 0, 4000))
    assert "LOOK_AT_LAP" in kinds(records)


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
