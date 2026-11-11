from __future__ import annotations

from proctoring_ai.multi.cctv.person_tracks import link_people


def _person(x, y=0.3, w=0.1, h=0.3):
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[5], kps[6] = [x + 0.02, y + 0.08, 0.9], [x + w - 0.02, y + 0.08, 0.9]
    return {"bbox": {"x": x, "y": y, "width": w, "height": h}, "keypoints": kps}


def test_still_people_keep_their_ids():
    frames = [[_person(0.1), _person(0.5)] for _ in range(50)]
    ids = link_people(frames, [i * 200 for i in range(50)])
    assert all(row == ids[0] for row in ids) and ids[0][0] != ids[0][1]


def test_a_person_lost_for_a_second_comes_back_with_the_same_id():
    frames = [[_person(0.1), _person(0.5)]] * 5 + [[_person(0.5)]] * 3 + [[_person(0.1), _person(0.5)]] * 5
    ids = link_people(frames, [i * 200 for i in range(13)])
    assert ids[7] == [ids[0][1]] and ids[10] == ids[0]


def test_people_far_apart_never_swap_and_a_newcomer_gets_a_new_id():
    frames = [[_person(0.1), _person(0.5)]] * 3 + [[_person(0.11), _person(0.5), _person(0.8)]] * 3
    ids = link_people(frames, [i * 200 for i in range(6)])
    assert ids[3][0] == ids[0][0] and ids[3][1] == ids[0][1]
    assert ids[3][2] not in ids[0]


def test_gap_longer_than_max_starts_a_new_track():
    frames = [[_person(0.1)], [_person(0.1)]]
    assert link_people(frames, [0, 5000])[1] != link_people(frames, [0, 200])[1]


def _face_person(x, y, w=0.1, h=0.3):
    person = _person(x, y, w, h)
    for index in range(5):
        person["keypoints"][index] = [x + w / 2, y + 0.02, 0.9]
    return person


def test_assign_tracked_keeps_the_real_occupant_when_a_back_row_head_drifts_in():
    from uuid import uuid4

    from proctoring_ai.multi.cctv.person_tracks import assign_tracked
    from proctoring_ai.multi.cctv.review_rules import ReviewFrame
    from proctoring_core.observations import Box

    seat = uuid4()
    box = Box(0.3, 0.3, 0.2, 0.3)
    owner, intruder = _face_person(0.35, 0.34), _face_person(0.42, 0.31)
    # 40 frames the owner alone, then 5 frames with the owner missing and a head from behind inside the box.
    frames = [ReviewFrame(i * 200, {seat: box}, {}, [], 16 / 9, [owner]) for i in range(40)]
    frames += [ReviewFrame((40 + i) * 200, {seat: box}, {}, [], 16 / 9, [intruder]) for i in range(5)]
    got = assign_tracked(frames, window_ms=20000)
    assert got[10].poses[seat] is not None and got[10].poses[seat] == owner["keypoints"]
    assert seat not in got[42].poses  # the intruder is not the occupant, so the seat has no pose then


def test_assign_tracked_without_people_keeps_frames():
    from uuid import uuid4

    from proctoring_ai.multi.cctv.person_tracks import assign_tracked
    from proctoring_ai.multi.cctv.review_rules import ReviewFrame
    from proctoring_core.observations import Box

    seat = uuid4()
    frame = ReviewFrame(0, {seat: Box(0.3, 0.3, 0.2, 0.3)}, {seat: [[0.1, 0.1, 0.9]] * 17}, [], 16 / 9)
    assert assign_tracked([frame])[0].poses == frame.poses
