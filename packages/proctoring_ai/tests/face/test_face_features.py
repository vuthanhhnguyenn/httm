from __future__ import annotations

from dataclasses import dataclass

import pytest
from proctoring_ai.face.features import EYES, MOUTH, eye_features, keypoints, mouth_features


@dataclass
class P:
    x: float
    y: float


def mesh(iris_shift: float = 0.0, *, mouth_gap: float = 0.01) -> list[P]:
    points = [P(0.5, 0.5) for _ in range(478)]
    for outer, inner, upper, lower, iris in EYES:
        left = outer == 33
        x0, x1 = (0.38, 0.46) if left else (0.54, 0.62)
        points[outer], points[inner] = (P(x0, 0.4), P(x1, 0.4)) if left else (P(x1, 0.4), P(x0, 0.4))
        points[upper], points[lower] = P((x0 + x1) / 2, 0.39), P((x0 + x1) / 2, 0.415)
        points[iris] = P((x0 + x1) / 2 + iris_shift * (x1 - x0), 0.4025)
    up, low, a, b = MOUTH
    points[up], points[low] = P(0.5, 0.6), P(0.5, 0.6 + mouth_gap)
    points[a], points[b] = P(0.45, 0.605), P(0.55, 0.605)
    return points


def test_centred_and_shifted_iris() -> None:
    centred = eye_features(mesh(), 640)
    right = eye_features(mesh(0.25), 640)
    assert centred is not None and right is not None
    assert centred["h"] == pytest.approx(0.0, abs=1e-6)
    assert right["h"] == pytest.approx(0.25, abs=1e-6)  # same direction for both eyes
    assert centred["eye_width_px"] == pytest.approx(0.08 * 640)
    assert centred["openness"] == pytest.approx(0.025 / 0.08)


def test_mesh_without_iris_has_no_eye_features() -> None:
    assert eye_features(mesh()[:468], 640) is None


def test_mouth_ratio_and_keypoints() -> None:
    closed = mouth_features(mesh(mouth_gap=0.005))
    open_ = mouth_features(mesh(mouth_gap=0.04))
    assert closed is not None and open_ is not None
    assert open_["mar"] > closed["mar"]
    assert open_["mar"] == pytest.approx(0.4)
    assert set(keypoints(mesh())) == {"nose_tip", "chin", "mouth_center", "ear_left", "ear_right"}
