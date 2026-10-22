"""Scripted, model-free scenarios replayed through the real pipeline and rules.

Each scenario describes what the analyzers *would* report at a given video time.
The fakes feed those observations into ``ObservationPipeline`` so trackers, head
pose smoothing/calibration, rules, aggregation and risk all run for real.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from proctoring_ai.face.head_pose import HeadPoseEstimator
from proctoring_ai.pipeline.observation_pipeline import ObservationPipeline
from proctoring_ai.tracking.object_tracks import ObjectEvidenceTracker
from proctoring_ai.tracking.single_tracker import SingleSubjectTracker

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
SEATED = {"x": 0.3, "y": 0.2, "width": 0.4, "height": 0.75}
FACE_BOX = {"x": 0.42, "y": 0.25, "width": 0.16, "height": 0.22}
PHONE_BOX = {"x": 0.55, "y": 0.6, "width": 0.08, "height": 0.12}

Timeline = Callable[[int], Any]


def person(t: int) -> list[dict[str, Any]]:
    return [{"class": "person", "confidence": 0.92, "bbox": dict(SEATED)}]


def face(yaw: float = 0.0, pitch: float = 0.0, quality: float = 0.8) -> dict[str, Any]:
    return {"bbox": dict(FACE_BOX), "yaw": yaw, "pitch": pitch, "roll": 0.0, "pose_valid": True,
            "quality": quality, "frame_width": 640, "frame_height": 480}


def between(t: int, start_s: float, end_s: float) -> bool:
    return start_s * 1000 <= t < end_s * 1000


@dataclass(slots=True)
class Scenario:
    name: str
    duration_s: float
    people: Timeline = person
    objects: Timeline = field(default=lambda t: [])
    face: Timeline = field(default=lambda t: face())
    hands: Timeline = field(default=lambda t: [])
    quality: Timeline = field(default=lambda t: {"status": "ready", "brightness": 0.5, "entropy": 6.0,
                                                 "blur": 150.0, "occlusion_estimate": 0.0})


class _Clock:
    now_ms = 0


class FakeDetector:
    def __init__(self, scenario: Scenario, clock: _Clock) -> None:
        self.scenario, self.clock = scenario, clock

    async def detect(self, _frame: dict[str, Any]) -> list[dict[str, Any]]:
        t = self.clock.now_ms
        return [dict(item) for item in [*self.scenario.people(t), *self.scenario.objects(t)]]


class FakeFace:
    def __init__(self, scenario: Scenario, clock: _Clock) -> None:
        self.scenario, self.clock = scenario, clock
        self._estimator = HeadPoseEstimator()

    def request_calibration(self) -> None:
        self._estimator.begin_calibration()

    async def analyze(self, frame: dict[str, Any]) -> dict[str, Any]:
        raw = self.scenario.face(self.clock.now_ms)
        if raw is None:
            return {"status": "no_detection", "reason": None, "faces": []}
        smoothed = self._estimator.update(dict(raw), int(frame["captured_at_monotonic_ms"]))
        return {"status": "ready", "reason": None, "faces": [smoothed]}


class FakeHands:
    def __init__(self, scenario: Scenario, clock: _Clock) -> None:
        self.scenario, self.clock = scenario, clock

    async def analyze(self, _frame: dict[str, Any]) -> dict[str, Any]:
        hands = [dict(item) for item in self.scenario.hands(self.clock.now_ms)]
        return {"status": "ready" if hands else "no_detection", "reason": None, "hands": hands}


class FakeQuality:
    def __init__(self, scenario: Scenario, clock: _Clock) -> None:
        self.scenario, self.clock = scenario, clock

    def extract(self, _image: Any) -> dict[str, Any]:
        return dict(self.scenario.quality(self.clock.now_ms))


class ScriptedSource:
    def __init__(self, scenario: Scenario, clock: _Clock, rate_hz: float = 8.0) -> None:
        self.scenario, self.clock = scenario, clock
        self.step_ms = 1000.0 / rate_hz
        self._next = 0.0
        self._frame_id = 0
        self._image = np.zeros((2, 2, 3), dtype=np.uint8)

    async def open(self) -> None:
        self._next, self._frame_id = 0.0, 0

    async def read(self) -> dict[str, Any] | None:
        t = round(self._next)
        if t >= self.scenario.duration_s * 1000:
            return None
        self._next += self.step_ms
        self._frame_id += 1
        self.clock.now_ms = t
        return {"frame_id": self._frame_id, "image": self._image, "video_offset_ms": t,
                "captured_at_monotonic_ms": t + 1, "captured_at_utc": EPOCH + timedelta(milliseconds=t),
                "frame_size": (640, 480)}

    async def close(self) -> None:
        return None


def build(scenario: Scenario, config: dict[str, Any]) -> tuple[ScriptedSource, ObservationPipeline]:
    clock = _Clock()
    detection = config.get("detection", {})
    behavior = config.get("behavior", {})
    pipeline = ObservationPipeline(
        detector=FakeDetector(scenario, clock),
        tracker=SingleSubjectTracker(),
        face_analyzer=FakeFace(scenario, clock),
        hand_analyzer=FakeHands(scenario, clock),
        image_quality=FakeQuality(scenario, clock),
        pose_interval_ms=int(detection.get("pose_interval_ms", 250)),
        hands_interval_ms=int(detection.get("hands_interval_ms", 200)),
        person_confidence_threshold=float(detection.get("person_confidence_threshold", 0.6)),
        object_tracker=ObjectEvidenceTracker(thresholds={
            "cell phone": float(behavior.get("phone", {}).get("minimum_confidence", 0.55)),
            "book": float(behavior.get("document", {}).get("minimum_confidence", 0.7)),
        }),
    )
    return ScriptedSource(scenario, clock), pipeline


def _phone_with_gaps(t: int) -> list[dict[str, Any]]:
    # 5–15 s, two short dropouts (< max gap) and a box jump that starts a new object track.
    if not between(t, 5, 15) or between(t, 7.0, 7.4) or between(t, 11.0, 11.3):
        return []
    box = dict(PHONE_BOX) if t < 10_000 else {**PHONE_BOX, "x": 0.2, "y": 0.62}
    return [{"class": "cell phone", "confidence": 0.78, "bbox": box}]


def _leave_people(t: int) -> list[dict[str, Any]]:
    return [] if between(t, 5, 13) else person(t)


def _leave_face(t: int) -> dict[str, Any] | None:
    return None if between(t, 4.5, 13.5) else face()


def _head_face(t: int) -> dict[str, Any]:
    if between(t, 6, 9):
        return face(yaw=40.0)
    if between(t, 12, 14):
        return face(yaw=-38.0)
    if between(t, 16, 20):
        return face(pitch=30.0)
    return face(yaw=1.0 if (t // 500) % 2 else -1.0)


def _blocked_quality(t: int) -> dict[str, Any]:
    blur = 3.0 if between(t, 5, 11) else 150.0
    return {"status": "ready", "brightness": 0.05 if blur < 10 else 0.5, "entropy": 1.0, "blur": blur,
            "occlusion_estimate": 0.0}


def _two_people(t: int) -> list[dict[str, Any]]:
    people = person(t)
    if between(t, 4, 9):
        people.append({"class": "person", "confidence": 0.85,
                       "bbox": {"x": 0.72, "y": 0.1, "width": 0.25, "height": 0.6}})
    return people


FACE_GEOMETRY = {"face_width": 0.16, "keypoints": {"ear_left": (0.42, 0.36), "ear_right": (0.58, 0.36),
                                                   "chin": (0.5, 0.47), "mouth_center": (0.5, 0.42)}}


def _hands_near_face(t: int) -> list[dict[str, Any]]:
    if between(t, 2, 9):  # hand held at the ear
        return [{"bbox": {"x": 0.56, "y": 0.28, "width": 0.07, "height": 0.1}, "handedness": "Right",
                 "confidence": 0.9, "wrist": (0.6, 0.42), "index_tip": (0.59, 0.34)}]
    if between(t, 9, 14):  # chin resting on the hand: normal
        return [{"bbox": {"x": 0.46, "y": 0.45, "width": 0.08, "height": 0.1}, "handedness": "Right",
                 "confidence": 0.9, "wrist": (0.5, 0.56), "index_tip": (0.5, 0.48)}]
    return [{"bbox": {"x": 0.45, "y": 0.8, "width": 0.08, "height": 0.08}, "handedness": "Right",
             "confidence": 0.9, "wrist": (0.49, 0.86), "index_tip": (0.49, 0.8)}]


def _shifted_people(t: int) -> list[dict[str, Any]]:
    # Seated for the learning window, then standing off to the side but still in frame.
    if between(t, 6, 12):
        return [{"class": "person", "confidence": 0.9, "bbox": {"x": 0.72, "y": 0.05, "width": 0.25, "height": 0.9}}]
    return person(t)


def _glancing_face(t: int) -> dict[str, Any]:
    # Eyes centred while the neutral position is learned, then a long glance right, head still.
    h = 0.26 if between(t, 8, 12) else 0.0
    return {**face(), "gaze": {"h": h, "v": 0.0, "openness": 0.3, "eye_width_px": 30.0}}


def _talking_face(t: int) -> dict[str, Any]:
    # Rhythmic mouth opening 5–11 s; a yawn (mouth held wide) 13–15 s.
    import math

    if between(t, 5, 11):
        mar = 0.12 + 0.12 * max(0.0, math.sin(2 * math.pi * 1.7 * t / 1000))
    elif between(t, 13, 15):
        mar = 0.65
    else:
        mar = 0.05
    return {**face(), "mouth": {"mar": mar, "mouth_width": 0.08}}


SCENARIOS: dict[str, Scenario] = {
    "phone_with_gaps": Scenario("phone_with_gaps", 20, objects=_phone_with_gaps),
    "leave_seat": Scenario("leave_seat", 20, people=_leave_people, face=_leave_face),
    "head_turns": Scenario("head_turns", 23, face=_head_face),
    "camera_blocked": Scenario("camera_blocked", 16, quality=_blocked_quality,
                               people=lambda t: [] if between(t, 5, 11) else person(t),
                               face=lambda t: None if between(t, 5, 11) else face()),
    "second_person": Scenario("second_person", 14, people=_two_people),
    "hands_near_face": Scenario("hands_near_face", 16, hands=_hands_near_face,
                                face=lambda t: {**face(), **FACE_GEOMETRY}),
    "gaze_sideways": Scenario("gaze_sideways", 16, face=_glancing_face),
    "talking_and_yawn": Scenario("talking_and_yawn", 17, face=_talking_face),
    "leaves_seat_in_frame": Scenario("leaves_seat_in_frame", 16, people=_shifted_people,
                                     face=lambda t: None if between(t, 6, 12) else face()),
}
