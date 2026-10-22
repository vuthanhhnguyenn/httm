from __future__ import annotations

import math
from uuid import uuid4

from proctoring_core.behavior.talking import TalkingRuleEvaluator, mouth_oscillations
from proctoring_core.config.schema import TalkingConfig
from proctoring_core.observations import Box, FaceObs, MouthObs

STEP_MS = 125  # 8 face passes per second


def face(mar: float, *, yaw: float = 0.0, quality: float = 0.9) -> FaceObs:
    return FaceObs(Box(0.4, 0.25, 0.2, 0.25), yaw=yaw, quality=quality, mouth=MouthObs(mar))


def active_times(mars: list[float], **face_kwargs: float) -> list[int]:
    rule = TalkingRuleEvaluator(TalkingConfig(enabled=True), min_face_quality=0.35, max_sample_gap_ms=600)
    session = uuid4()
    times = []
    for index, mar in enumerate(mars):
        timestamp = 1 + index * STEP_MS
        for item in rule.evaluate(session_id=session, track_id=None, face=face(mar, **face_kwargs),
                                  monotonic_ms=timestamp):
            if item["active"]:
                times.append(timestamp)
    return times


def speech(seconds: float, hz: float = 1.7) -> list[float]:
    return [0.12 + 0.12 * max(0.0, math.sin(2 * math.pi * hz * i * STEP_MS / 1000))
            for i in range(int(seconds * 1000 / STEP_MS))]


def test_sustained_rhythmic_mouth_movement_is_possible_talking() -> None:
    times = active_times([0.05] * 16 + speech(6) + [0.05] * 16)
    assert times
    assert times[0] - 2001 >= 2500  # needs the configured duration after speech starts


def test_yawn_and_held_open_mouth_are_not_talking() -> None:
    yawn = [0.05] * 8 + [0.05 + 0.6 * math.sin(math.pi * i / 24) for i in range(24)] + [0.05] * 16
    assert active_times(yawn) == []
    assert active_times([0.7] * 40) == []


def test_short_speech_and_still_mouth_do_not_trigger() -> None:
    assert active_times([0.05] * 8 + speech(2.0) + [0.05] * 16) == []
    assert active_times([0.06 + 0.005 * (i % 2) for i in range(60)]) == []


def test_turned_or_low_quality_face_is_ignored() -> None:
    assert active_times(speech(6), yaw=60.0) == []
    assert active_times(speech(6), quality=0.2) == []


def test_oscillation_counter_counts_full_cycles() -> None:
    assert mouth_oscillations([0, 1, 0, 1, 0, 1, 0], hysteresis=0.1) == 3
    assert mouth_oscillations([0, 0.5, 1, 1, 1, 0.5, 0], hysteresis=0.1) == 1
