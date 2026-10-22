from __future__ import annotations

from proctoring_core.behavior.hands import NEAR_EAR, NEAR_MOUTH, RAPID, hand_reasons
from proctoring_core.config.schema import HandActivityConfig
from proctoring_core.observations import Box, FaceObs, HandObs

FACE = FaceObs(Box(0.42, 0.25, 0.16, 0.22), face_width=0.16,
               keypoints={"ear_left": (0.42, 0.36), "ear_right": (0.58, 0.36), "chin": (0.5, 0.47),
                          "mouth_center": (0.5, 0.42)})
CONFIG = HandActivityConfig()


def hand(wrist: tuple[float, float], tip: tuple[float, float], *, move: float = 0.0, dt: int = 0) -> HandObs:
    return HandObs(None, wrist=wrist, index_tip=tip, displacement=move, dt_ms=dt, confidence=0.9)


def test_hand_at_ear_and_over_mouth() -> None:
    assert NEAR_EAR in hand_reasons(hand((0.6, 0.42), (0.59, 0.34)), FACE, CONFIG)
    assert NEAR_MOUTH in hand_reasons(hand((0.52, 0.55), (0.5, 0.41)), FACE, CONFIG)


def test_chin_rest_writing_and_typing_are_normal() -> None:
    assert hand_reasons(hand((0.5, 0.56), (0.5, 0.47)), FACE, CONFIG) == set()          # chin on the hand
    assert hand_reasons(hand((0.49, 0.86), (0.49, 0.8), move=0.04, dt=125), FACE, CONFIG) == set()  # writing


def test_fast_movement_counts_only_above_the_chin() -> None:
    # 0.06 normalized in 125 ms = 0.48/s = 3 face widths per second.
    assert RAPID in hand_reasons(hand((0.3, 0.3), (0.3, 0.25), move=0.06, dt=125), FACE, CONFIG)
    assert RAPID not in hand_reasons(hand((0.3, 0.7), (0.3, 0.65), move=0.06, dt=125), FACE, CONFIG)


def test_without_face_geometry_nothing_is_judged() -> None:
    assert hand_reasons(hand((0.6, 0.42), (0.59, 0.34)), None, CONFIG) == set()
    assert hand_reasons(hand((0.6, 0.42), (0.59, 0.34)), FaceObs(None), CONFIG) == set()
