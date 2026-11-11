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
