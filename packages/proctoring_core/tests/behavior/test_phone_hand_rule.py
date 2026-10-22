from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from proctoring_core.behavior.engine import BehaviorEngine
from proctoring_core.config.schema import BehaviorConfig
from proctoring_core.events.models import ObservationBundle, ObservationHealth
from proctoring_core.observations import Observation
from proctoring_core.types import EventType

FACE = {"bbox": {"x": 0.4, "y": 0.2, "width": 0.2, "height": 0.25}, "face_width": 0.2, "quality": 0.9}
PERSON = {"class": "person", "confidence": 0.9, "bbox": {"x": 0.25, "y": 0.1, "width": 0.5, "height": 0.85}}
AT_TEMPLE = {"x": 0.37, "y": 0.28, "width": 0.04, "height": 0.05}   # glasses camera beside the eye
ON_DESK = {"x": 0.55, "y": 0.8, "width": 0.08, "height": 0.1}


def phone_active(box: dict[str, float], hands: list[dict[str, Any]], seconds: float = 3.0) -> bool:
    config = BehaviorConfig.model_validate({"phone": {"minimum_duration_seconds": 0.5}})
    engine = BehaviorEngine(uuid4(), {}, config)
    active = False
    for step in range(int(seconds * 8)):
        t = 1 + step * 125
        bundle = ObservationBundle(
            frame_id=step + 1, session_id=uuid4(), captured_at_utc=datetime.now(UTC), captured_at_monotonic_ms=t, frame_size=(640, 480),
            people=[{**PERSON, "track_id": "p"}],
            objects=[{"class": "cell phone", "confidence": 0.8, "bbox": box, "object_track_id": 1,
                      "track_confirmed": True}],
            faces=[FACE], hands=hands,
            analyzer_health={"object_detector": ObservationHealth("ready"), "face": ObservationHealth("ready"),
                             "hands": ObservationHealth("ready" if hands else "no_detection")})
        for signal in engine.evaluate(Observation.from_bundle(bundle)):
            if signal.event_type is EventType.PHONE_DETECTED and signal.active:
                active = True
    return active


def test_phone_on_face_without_hand_is_ignored() -> None:
    assert not phone_active(AT_TEMPLE, hands=[])


def test_phone_on_face_held_by_hand_is_detected() -> None:
    hand = {"wrist": (0.36, 0.42), "index_tip": (0.38, 0.3), "confidence": 0.9}
    assert phone_active(AT_TEMPLE, hands=[hand])


def test_phone_away_from_face_needs_no_hand() -> None:
    assert phone_active(ON_DESK, hands=[])
