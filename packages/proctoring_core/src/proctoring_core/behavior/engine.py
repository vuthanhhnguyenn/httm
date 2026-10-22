"""Run every behaviour rule on one typed observation.

The engine owns subject association (which face/hands belong to the single
tracked person) and analyzer-health gating; evaluators stay single-purpose.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..calibration.gaze import GazeCalibrator, GazeStatus
from ..calibration.seat import SeatZoneEstimator
from ..config.schema import BehaviorConfig
from ..observations import FaceObs, ObjectObs, Observation, PersonObs
from ..types import EventType
from .arbiter import SignalArbiter
from .gaze import GazeRuleEvaluator
from .geometry import inside_person
from .hands import HandRuleEvaluator
from .head import HeadRuleEvaluator
from .objects import ObjectRuleEvaluator
from .presence import PresenceRuleEvaluator
from .safety import SafetyRuleEvaluator
from .signals import RuleSignal
from .talking import TalkingRuleEvaluator

_KNOWN = frozenset({"ready", "no_detection"})


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


class BehaviorEngine:
    def __init__(self, session_id: Any, policy: Mapping[str, Any], behavior: BehaviorConfig) -> None:
        self.session_id = session_id
        self.behavior = behavior
        phone, document = behavior.phone, behavior.document
        self.presence = PresenceRuleEvaluator(
            missing_duration_ms=_ms(behavior.person_missing.minimum_duration_seconds),
            multiple_duration_ms=_ms(behavior.multiple_person.minimum_duration_seconds),
        )
        self.head = HeadRuleEvaluator(behavior.model_dump())
        self.objects = ObjectRuleEvaluator(
            dict(policy),
            confidence_threshold=min(phone.minimum_confidence, document.minimum_confidence),
            phone_confidence_threshold=phone.minimum_confidence,
            document_confidence_threshold=document.minimum_confidence,
            phone_min_duration_ms=_ms(phone.minimum_duration_seconds),
            document_min_duration_ms=_ms(document.minimum_duration_seconds),
            phone_release_confidence=phone.release_confidence,
            max_gap_ms=phone.maximum_gap_ms,
            phone_fast_confidence=phone.fast_confidence,
            phone_fast_duration_ms=_ms(phone.fast_duration_seconds),
            phone_fast_min_hits=phone.fast_min_hits,
            phone_max_sample_gap_ms=phone.max_sample_gap_ms,
            document_max_sample_gap_ms=document.max_sample_gap_ms,
            phone_fast_max_sample_gap_ms=phone.fast_max_sample_gap_ms,
            phone_classes=tuple(phone.classes),
            document_classes=tuple(document.classes),
            phone_episode_gap_ms=phone.episode_gap_ms,
            document_episode_gap_ms=document.episode_gap_ms,
            phone_geometry_max_height_ratio=phone.geometry_max_height_ratio,
            phone_geometry_max_area_ratio=phone.geometry_max_area_ratio,
        )
        face_rule = behavior.face_not_visible
        self.safety = SafetyRuleEvaluator(
            min_duration_ms=_ms(face_rule.minimum_duration_seconds),
            face_visibility_threshold=face_rule.visibility_threshold,
            blur_threshold=behavior.camera_blocked.blur_threshold,
            occlusion_threshold=behavior.camera_blocked.occlusion_threshold,
            max_brightness=behavior.camera_blocked.max_brightness,
            min_contrast=behavior.camera_blocked.min_contrast,
            min_edge_density=behavior.camera_blocked.min_edge_density,
            require_no_person=behavior.camera_blocked.require_no_person,
            durations_ms={
                EventType.FACE_NOT_VISIBLE: _ms(face_rule.minimum_duration_seconds),
                EventType.LEAVING_SEAT: _ms(behavior.leaving_seat.minimum_duration_seconds),
                EventType.CAMERA_BLOCKED: _ms(behavior.camera_blocked.minimum_duration_seconds),
            },
        )
        self.hands = HandRuleEvaluator(behavior.hand_activity, max_sample_gap_ms=behavior.quality.max_sample_gap_ms)
        self._margin = behavior.association.person_box_margin
        self.arbiter = SignalArbiter(behavior.hierarchy)
        self.seat = SeatZoneEstimator(learning_ms=_ms(behavior.leaving_seat.zone_learning_seconds))
        self.gaze_calibration = GazeCalibrator(min_quality=behavior.quality.face_min_quality)
        self.gaze = GazeRuleEvaluator(behavior.gaze, head_turn_degrees=behavior.head_turn.yaw_degrees,
                                      min_face_quality=behavior.quality.face_min_quality,
                                      max_sample_gap_ms=behavior.quality.max_sample_gap_ms)
        self.talking = TalkingRuleEvaluator(behavior.talking, min_face_quality=behavior.quality.face_min_quality,
                                            max_sample_gap_ms=behavior.quality.max_sample_gap_ms)
        self._center_calibration_until: int | None = None
        self._center_calibration_requested = False
        self._recent_hands: list[tuple[int, tuple[float, float]]] = []
        self._last_head_track_id: str | None = None
        self._last_hand_track_id: str | None = None

    def request_calibration(self, *, gaze_center: bool = False) -> None:
        """Re-learn the seat zone; optionally learn the neutral eye position as well."""

        self.seat.reset()
        # Gaze fits use head angles relative to the head baseline; a new baseline invalidates them.
        self.gaze_calibration = GazeCalibrator(min_quality=self.behavior.quality.face_min_quality)
        if gaze_center:
            self._center_calibration_requested = True

    # Operator-driven gaze calibration: the UI shows each target while samples are collected.
    def begin_gaze_calibration(self) -> None:
        self.gaze_calibration.begin()

    def set_gaze_target(self, name: str | None) -> None:
        self.gaze_calibration.set_target(name)

    def finish_gaze_calibration(self) -> GazeStatus:
        return self.gaze_calibration.finish()

    def cancel_gaze_calibration(self) -> None:
        """Abort collection; a previously fitted profile stays in use."""

        calibrator = self.gaze_calibration
        calibrator.set_target(None)
        if calibrator.status is GazeStatus.COLLECTING:
            calibrator.status = GazeStatus.CALIBRATED if calibrator.profile else GazeStatus.UNCALIBRATED

    def _calibrate_gaze(self, face: FaceObs | None, timestamp: int) -> None:
        quality = self.gaze.observation_quality(face)
        if self._center_calibration_requested and quality > 0:
            self._center_calibration_requested = False
            self._center_calibration_until = timestamp + _ms(self.behavior.gaze.center_calibration_seconds)
            self.gaze_calibration.begin()
            self.gaze_calibration.set_target("center")
        self.gaze_calibration.add(face, quality)
        if self._center_calibration_until is not None and timestamp >= self._center_calibration_until:
            self._center_calibration_until = None
            self.gaze_calibration.finish()

    def _leaving_seat(self, person: PersonObs | None, observation: Observation) -> bool:
        if person is None:
            return False
        config = self.behavior.leaving_seat
        self.seat.update(person.bbox, observation.monotonic_ms)
        shoulders = next((point for pose in observation.poses
                          if (point := pose.shoulder_midpoint(config.min_shoulder_visibility))), None)
        return self.seat.is_leaving(person.bbox, shoulders, margin=config.zone_margin,
                                    min_area_ratio=config.min_area_ratio)

    def _remember_hands(self, observation: Observation) -> None:
        memory = self.behavior.phone.hand_memory_ms
        now = observation.monotonic_ms
        if observation.analyzer_status("hands") in _KNOWN:
            for hand in observation.hands:
                for point in (hand.wrist, hand.index_tip, hand.bbox.center if hand.bbox else None):
                    if point is not None:
                        self._recent_hands.append((now, point))
        self._recent_hands = [(t, p) for t, p in self._recent_hands if 0 <= now - t <= memory]

    def _held_or_away_from_face(self, item: ObjectObs, face: FaceObs | None) -> bool:
        """False for a phone box on the face that no hand is holding."""

        config = self.behavior.phone
        if (not config.require_hand_near_face or item.bbox is None or face is None or face.bbox is None
                or item.key not in {name.lower() for name in config.classes}):
            return True
        width = face.face_width or face.bbox.width
        x, y = item.bbox.center
        margin = config.face_margin * width
        box = face.bbox
        near_face = (box.x - margin <= x <= box.x + box.width + margin
                     and box.y - margin <= y <= box.y + box.height + margin)
        if not near_face:
            return True
        radius = config.hand_radius * width
        return any(((px - x) ** 2 + (py - y) ** 2) ** 0.5 <= radius for _, (px, py) in self._recent_hands)

    def evaluate(self, observation: Observation) -> list[RuleSignal]:
        if observation.image_quality.frozen:
            # A stalled camera repeats one picture; it is a device fault, never fresh evidence.
            # Skipping the frame lets the sample-gap rules close conditions that were open.
            return []
        timestamp = observation.monotonic_ms
        session_id = self.session_id
        person: PersonObs | None = observation.people[0] if len(observation.people) == 1 else None
        track_id = person.track_id if person else None
        person_box = person.bbox if person else None
        raw: list[dict[str, Any]] = []

        self._remember_hands(observation)
        first_face = observation.faces[0] if observation.faces else None
        if observation.analyzer_status("object_detector") == "ready":
            raw += self.presence.evaluate(session_id=session_id, people=list(observation.people),
                                          monotonic_ms=timestamp)
            raw += self.objects.evaluate(session_id=session_id, track_id=track_id,
                                         objects=[_object_sample(item) for item in observation.objects
                                                  if self._held_or_away_from_face(item, first_face)],
                                         monotonic_ms=timestamp, frame_id=observation.frame_id)

        # Anonymous face signals remain visible, but cannot be paired with another person.
        face = observation.faces[0] if observation.faces else None
        face_matched = bool(track_id and face is not None and inside_person(face.bbox, person_box, margin=self._margin))
        head_track_id = track_id if face_matched or (track_id and face is None) else None
        if self._last_head_track_id and self._last_head_track_id != head_track_id:
            raw += self.head.evaluate(session_id=session_id, track_id=self._last_head_track_id,
                                      sample={"monotonic_ms": timestamp, "quality": 0, "pose_valid": False})
        self._last_head_track_id = head_track_id
        raw += self.head.evaluate(session_id=session_id, track_id=head_track_id,
                                  sample={**_face_sample(face or FaceObs.absent()), "monotonic_ms": timestamp})

        gaze_face = face if face_matched or (track_id is None and face is not None) else None
        self._calibrate_gaze(gaze_face, timestamp)
        raw += self.gaze.evaluate(session_id=session_id, track_id=head_track_id, face=gaze_face,
                                  profile=self.gaze_calibration.profile, monotonic_ms=timestamp)
        raw += self.talking.evaluate(session_id=session_id, track_id=head_track_id, face=gaze_face,
                                     monotonic_ms=timestamp)

        face_known = observation.analyzer_status("face") in _KNOWN
        hand_known = observation.analyzer_status("hands") in _KNOWN
        face_visible = bool(observation.faces) if face_known else True
        matched_hands = [hand for hand in observation.hands
                         if track_id and inside_person(hand.bbox, person_box, margin=self._margin)]
        hand_track_id = track_id if matched_hands or (track_id and not observation.hands) else None
        hands_for_rule = matched_hands if hand_track_id else list(observation.hands)

        if face_known or observation.has_image_quality:
            # Session-level in phase 1 (one candidate): a returning person gets a new track id,
            # and an event keyed by the old track could never be closed.
            raw += self.safety.evaluate(session_id=session_id, track_id=None, sample={
                **_person_sample(person),
                "monotonic_ms": timestamp,
                "image_quality": _quality_sample(observation),
                "person_present": bool(observation.people),
                "leaving_seat": self._leaving_seat(person, observation),
                "face_visible": face_visible,
            })
        if hand_known:
            if self._last_hand_track_id and self._last_hand_track_id != hand_track_id:
                raw += self.hands.evaluate(session_id=session_id, track_id=self._last_hand_track_id, hands=[],
                                           face=None, monotonic_ms=timestamp)
            self._last_hand_track_id = hand_track_id
            raw += self.hands.evaluate(session_id=session_id, track_id=hand_track_id, hands=hands_for_rule,
                                       face=face, monotonic_ms=timestamp)
        return self.arbiter.apply([RuleSignal.from_mapping(item) for item in raw])


def _object_sample(item: ObjectObs) -> dict[str, Any]:
    sample: dict[str, Any] = {"class": item.label, "confidence": item.confidence}
    if item.bbox is not None:
        sample["bbox"] = item.bbox.as_dict()
    if item.object_track_id is not None:
        sample["object_track_id"] = item.object_track_id
    if item.track_confirmed is not None:
        sample["track_confirmed"] = item.track_confirmed
    if item.model_name is not None:
        sample["model_name"] = item.model_name
    if item.model_version is not None:
        sample["model_version"] = item.model_version
    return sample


def _face_sample(face: FaceObs) -> dict[str, Any]:
    sample: dict[str, Any] = {"yaw": face.yaw, "pitch": face.pitch, "roll": face.roll,
                              "quality": face.quality, "pose_valid": face.pose_valid}
    if face.calibration_status is not None:
        sample["calibration_status"] = face.calibration_status
    return sample


def _person_sample(person: PersonObs | None) -> dict[str, Any]:
    if person is None:
        return {}
    sample: dict[str, Any] = {"visibility": person.visibility}
    if person.bbox is not None:
        sample["bbox"] = person.bbox.as_dict()
    return sample


def _quality_sample(observation: Observation) -> dict[str, Any]:
    quality = observation.image_quality
    values = (("blur", quality.blur), ("occlusion_estimate", quality.occlusion_estimate),
              ("brightness", quality.brightness), ("contrast", quality.contrast),
              ("edge_density", quality.edge_density))
    return {key: value for key, value in values if value is not None}


__all__ = ["BehaviorEngine"]
