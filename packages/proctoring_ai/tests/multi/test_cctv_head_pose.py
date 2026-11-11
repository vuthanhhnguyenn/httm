from __future__ import annotations

from proctoring_ai.multi.cctv.head_pose import hhp_input, hhp_yaw_pitch


def _kps(nose, leye, reye, lear, rear):
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, (x, y) in enumerate((nose, leye, reye, lear, rear)):
        kps[index] = [x, y, 0.9 if (x, y) != (0.0, 0.0) else 0.0]
    return kps


FRONT = _kps((0.50, 0.40), (0.52, 0.38), (0.48, 0.38), (0.55, 0.40), (0.45, 0.40))
TURNED = _kps((0.40, 0.40), (0.43, 0.37), (0.41, 0.37), (0.0, 0.0), (0.46, 0.40))  # nose off to one side, one ear


def test_too_few_face_points_give_none():
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[0] = [0.5, 0.4, 0.9]
    assert hhp_input(kps) is None and hhp_yaw_pitch(kps) is None and hhp_yaw_pitch(None) is None


def test_missing_points_are_zero_and_the_rest_are_centred():
    batch = hhp_input(TURNED)
    assert batch is not None and batch.shape == (1, 15, 1)
    assert list(batch[0, 9:12, 0]) == [0.0, 0.0, 0.0]  # left ear absent
    assert max(abs(batch[0, 0::3, 0])) <= 1.0


def test_front_face_is_near_zero_and_a_turned_face_moves_the_yaw():
    front, turned = hhp_yaw_pitch(FRONT), hhp_yaw_pitch(TURNED)
    assert front is not None and turned is not None
    assert abs(front[0]) < 20
    assert abs(turned[0] - front[0]) > 15
