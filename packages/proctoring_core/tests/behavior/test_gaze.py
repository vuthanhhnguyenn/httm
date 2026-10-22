from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from proctoring_core.behavior.gaze import GazeRuleEvaluator
from proctoring_core.calibration.gaze import (
    TARGETS,
    GazeCalibrator,
    GazeProfile,
    GazeSample,
    GazeStatus,
)
from proctoring_core.config.schema import GazeConfig
from proctoring_core.observations import Box, EyeObs, FaceObs
from proctoring_core.types import EventType

BOX = Box(0.4, 0.25, 0.2, 0.25)


def face(h: float = 0.0, v: float = 0.0, *, yaw: float = 0.0, pitch: float = 0.0, width: float = 30.0,
         openness: float = 0.3, quality: float = 0.9) -> FaceObs:
    return FaceObs(BOX, yaw=yaw, pitch=pitch, quality=quality, calibration_status="calibrated",
                   gaze=EyeObs(h, v, openness, width))


def calibrated_screen() -> GazeCalibrator:
    """Synthetic candidate: eyes move 0.25 per screen width, head adds a little."""

    calibrator = GazeCalibrator()
    calibrator.begin()
    for name, (x, y) in TARGETS.items():
        calibrator.set_target(name)
        for jitter in (-0.004, 0.0, 0.004, 0.002):
            calibrator.add(face((x - 0.5) * 0.25 + jitter, (y - 0.5) * 0.2 + jitter, yaw=(x - 0.5) * 6,
                                pitch=(y - 0.5) * 5), 0.9)
    assert calibrator.finish() is GazeStatus.CALIBRATED
    return calibrator


def run(rule: GazeRuleEvaluator, profile: GazeProfile | None, faces: list[FaceObs]) -> list[dict[str, Any]]:
    session = uuid4()
    events = []
    for index, item in enumerate(faces):
        events += rule.evaluate(session_id=session, track_id=None, face=item, profile=profile,
                                monotonic_ms=1 + index * 125)
    return [item for item in events if item["active"]]


def rule(**overrides: Any) -> GazeRuleEvaluator:
    options: dict[str, Any] = {"enabled": True, **overrides}
    return GazeRuleEvaluator(GazeConfig(**options), head_turn_degrees=22, min_face_quality=0.35,
                             max_sample_gap_ms=600)


def test_screen_fit_maps_targets_and_detects_off_screen() -> None:
    profile = calibrated_screen().profile
    assert profile is not None and profile.has_screen
    x, y = profile.screen_point(GazeSample(0.125, 0.1, 3.0, 2.5)) or (0, 0)
    assert x == pytest.approx(1.0, abs=0.05) and y == pytest.approx(1.0, abs=0.05)

    corners = run(rule(), profile, [face(0.12, -0.1, yaw=3)] * 24)       # looking at a corner
    away = run(rule(), profile, [face(0.3, 0.0)] * 24)                    # eyes far right, head straight
    assert corners == []
    assert {item["event_type"] for item in away} >= {EventType.GAZE_OFF_SCREEN}
    assert away[0]["severity"].value == "LOW" and away[0]["metrics"]["experimental"] is True


def test_sideways_glance_needs_straight_head_and_duration() -> None:
    calibrator = GazeCalibrator()
    calibrator.calibrate_center_only(GazeSample(0.01, 0.0, 0.0, 0.0))
    profile = calibrator.profile
    assert profile is not None
    assert not profile.has_screen  # centre-only profile never claims off screen
    short = run(rule(), profile, [face(0.25)] * 12)          # 1.5 s < 2 s
    sustained = run(rule(), profile, [face(0.25)] * 24)      # 3 s
    head_turned = run(rule(), profile, [face(0.25, yaw=30)] * 24)

    assert short == []
    assert [item["event_type"] for item in sustained][0] is EventType.GAZE_SIDEWAYS
    assert head_turned == []


def test_uncalibrated_or_poor_view_stays_silent() -> None:
    profile = calibrated_screen().profile
    assert run(rule(), None, [face(0.4)] * 30) == []
    assert run(rule(), profile, [face(0.4, width=6)] * 30) == []        # eyes too small
    assert run(rule(), profile, [face(0.4, openness=0.05)] * 30) == []  # eyes closed / blink
    assert run(rule(off_screen_enabled=False, sideways_ratio=0.5), profile, [face(0.4)] * 30) == []


def test_calibration_refuses_missing_centre_and_bad_fit() -> None:
    calibrator = GazeCalibrator()
    calibrator.begin()
    calibrator.set_target("top_left")
    for _ in range(5):
        calibrator.add(face(-0.1, -0.1), 0.9)
    assert calibrator.finish() is GazeStatus.FAILED and calibrator.reason == "CENTER_NOT_OBSERVED"

    calibrator.begin()
    for name in TARGETS:  # the candidate never moved the eyes: targets indistinguishable
        calibrator.set_target(name)
        for _ in range(5):
            calibrator.add(face(0.0, 0.0), 0.9)
    assert calibrator.finish() is GazeStatus.FAILED
