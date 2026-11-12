from __future__ import annotations

from uuid import uuid4

import pytest
from cctv_kps import person
from proctoring_ai.multi.cctv.pose_smooth import merge_flipped, smooth_frames
from proctoring_ai.multi.cctv.review_rules import ReviewFrame
from proctoring_core.observations import Box

SEAT = uuid4()
SEATS = {SEAT: Box(0.25, 0.3, 0.1, 0.2)}
STEP = 200  # the deep tier's 5 frames a second


def _frames(wrist_at, count: int = 25, missing=()) -> list[ReviewFrame]:
    out = []
    for index in range(count):
        ms = index * STEP
        kps = person(cx=0.3, wrist=wrist_at(ms))
        out.append(ReviewFrame(ms, SEATS, {} if ms in missing else {SEAT: kps}, [], 1.0))
    return out


def _wrist(frames, at_ms: int) -> tuple[float, float]:
    frame = next(item for item in frames if item.offset_ms == at_ms)
    return frame.poses[SEAT][9][0], frame.poses[SEAT][9][1]


def test_smoothing_removes_one_frame_jitter() -> None:
    jump = 2000
    frames = smooth_frames(_frames(lambda ms: (0.3, 0.45 if ms == jump else 0.40)))
    # On the 5 fps grid the jump is cut to about half on its own frame and is gone once the wrist is back.
    assert abs(_wrist(frames, jump)[1] - 0.40) < 0.03
    assert abs(_wrist(frames, jump + STEP)[1] - 0.40) < 0.02
    assert all(abs(_wrist(frames, ms)[1] - 0.40) < 0.02 for ms in range(jump + STEP, 25 * STEP, STEP))
    # Points the filter never moves: x stays put, other keypoints are untouched.
    assert all(abs(_wrist(frames, ms)[0] - 0.3) < 1e-9 for ms in range(0, 25 * STEP, STEP))


def test_smoothing_follows_a_real_move() -> None:
    move = 2000
    frames = smooth_frames(_frames(lambda ms: (0.3, 0.50 if ms >= move else 0.40)))
    assert abs(_wrist(frames, move + 1000)[1] - 0.50) < 0.01


def test_smoothing_restarts_after_missing_seat() -> None:
    gone = {2000, 2200}
    frames = smooth_frames(_frames(lambda ms: (0.3, 0.50 if ms > 2200 else 0.40), missing=gone))
    assert all(SEAT not in frames[ms // STEP].poses for ms in gone)
    # Back after the gap at the new place: the filter starts again there instead of sliding up from 0.40.
    assert _wrist(frames, 2400)[1] == pytest.approx(0.50)


def test_smoothing_passes_low_confidence_points_through() -> None:
    raw = _frames(lambda ms: (0.3, 0.45 if ms == 2000 else 0.40))
    for frame in raw:
        frame.poses[SEAT][9][2] = 0.3
    frames = smooth_frames(raw)
    assert _wrist(frames, 2000)[1] == pytest.approx(0.45)
    assert raw[10].poses[SEAT][9][1] == pytest.approx(0.45)  # the input is not changed


def _seen(kps, box) -> dict:
    return {"class": "person", "confidence": 0.9, "bbox": {"x": box[0], "y": box[1], "width": box[2], "height": box[3]},
            "keypoints": kps}


def _flip(item: dict, mirror_x: float) -> dict:
    """What the model would report for ``item`` in the mirrored crop (full-frame coordinates, not swapped
    back): x mirrored, and the person's own left side now on the image's other side."""

    b = item["bbox"]
    kps = [[2 * mirror_x - p[0], p[1], p[2]] for p in item["keypoints"]]
    for left, right in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)):
        kps[left], kps[right] = kps[right], kps[left]
    return _seen(kps, (2 * mirror_x - b["x"] - b["width"], b["y"], b["width"], b["height"]))


def test_flip_merge_swaps_left_and_right() -> None:
    mirror = 0.35
    kps = person(cx=0.3, wrist=(0.28, 0.45))  # left wrist only
    original = _seen(kps, (0.25, 0.3, 0.1, 0.25))
    # The flipped crop saw the same person with the left wrist a bit lower.
    other = [list(p) for p in kps]
    other[9] = [0.28, 0.47, 0.9]
    merged = merge_flipped([original], [_flip(_seen(other, (0.25, 0.3, 0.1, 0.25)), mirror)], mirror)
    assert len(merged) == 1
    wrist = merged[0]["keypoints"][9]
    assert wrist[0] == pytest.approx(0.28) and wrist[1] == pytest.approx(0.46)
    assert merged[0]["keypoints"][10][2] == 0.0  # the right wrist stays unseen: no left wrist leaked into it
    assert merged[0]["keypoints"][5][0] == pytest.approx(kps[5][0])


def test_flip_merge_of_symmetric_person_is_identity() -> None:
    mirror = 0.4
    original = _seen(person(cx=0.3, wrist=(0.31, 0.45)), (0.25, 0.3, 0.1, 0.25))
    merged = merge_flipped([original], [_flip(original, mirror)], mirror)
    assert len(merged) == 1
    for got, want in zip(merged[0]["keypoints"], original["keypoints"], strict=True):
        assert got == pytest.approx(want)
    assert merged[0]["bbox"] == original["bbox"]


def test_flip_merge_with_no_flipped_people_returns_originals() -> None:
    people = [_seen(person(cx=0.3), (0.25, 0.3, 0.1, 0.25)), _seen(person(cx=0.6), (0.55, 0.3, 0.1, 0.25))]
    assert merge_flipped(people, [], 0.45) == people
    # Somebody only the flipped crop saw is kept, back in place.
    lone = _seen(person(cx=0.8), (0.75, 0.3, 0.1, 0.25))
    merged = merge_flipped(people, [_flip(lone, 0.45)], 0.45)
    assert len(merged) == 3
    assert merged[2]["bbox"]["x"] == pytest.approx(0.75)
    assert merged[2]["keypoints"][5][0] == pytest.approx(lone["keypoints"][5][0])


def test_review_window_smooths_only_when_asked(monkeypatch) -> None:
    from proctoring_ai.multi.cctv import review_rules
    from proctoring_ai.multi.cctv.review_rules import ReviewRuleConfig, review_window
    from proctoring_ai.multi.cctv.suspects import SuspectWindow

    calls: list[int] = []

    def spy(frames):
        calls.append(len(frames))
        return smooth_frames(frames)

    monkeypatch.setattr(review_rules, "smooth_frames", spy)
    window = SuspectWindow(SEAT, 1, (), 0, 4000, ("OBJECT_HINT",), SEATS[SEAT], SEATS)
    frames = _frames(lambda ms: (0.3, 0.40), count=20)
    review_window(window, frames, ReviewRuleConfig(smooth_poses=False))
    assert calls == []
    review_window(window, frames, ReviewRuleConfig(smooth_poses=True))
    assert calls == [20]
