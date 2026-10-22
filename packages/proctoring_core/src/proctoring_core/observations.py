"""Typed observations: the boundary between analyzers and behaviour rules.

Analyzers still emit plain mappings inside :class:`ObservationBundle`; rules only
see these dataclasses, built once per frame by :meth:`Observation.from_bundle`.
Missing analyzer values keep the defaults the rules have always assumed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from .events.models import ObservationBundle, ObservationHealth


@dataclass(frozen=True, slots=True)
class Box:
    """Normalized box; values are not clamped so geometry matches analyzer output."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @classmethod
    def parse(cls, value: Any) -> Box | None:
        if not isinstance(value, Mapping):
            return None
        try:
            return cls(float(value["x"]), float(value["y"]), float(value["width"]), float(value["height"]))
        except (KeyError, TypeError, ValueError):
            return None

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True, slots=True)
class PersonObs:
    bbox: Box | None
    confidence: float
    track_id: str | None
    visibility: float = 1.0


@dataclass(frozen=True, slots=True)
class ObjectObs:
    label: str
    confidence: float
    bbox: Box | None
    object_track_id: int | None = None
    # None means "no tracker opinion" and is treated as confirmed for entry.
    track_confirmed: bool | None = None
    model_name: str | None = None
    model_version: str | None = None

    @property
    def key(self) -> str:
        return self.label.lower()


@dataclass(frozen=True, slots=True)
class EyeObs:
    """Iris position in the eye opening (−0.5…0.5, 0 = centred; +h image right, +v down)."""

    h: float
    v: float
    openness: float
    eye_width_px: float


@dataclass(frozen=True, slots=True)
class MouthObs:
    mar: float  # inner-lip gap / mouth width


@dataclass(frozen=True, slots=True)
class FaceObs:
    bbox: Box | None
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    pose_valid: bool = True
    quality: float = 1.0
    calibration_status: str | None = None
    gaze: EyeObs | None = None
    mouth: MouthObs | None = None
    keypoints: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    face_width: float | None = None

    @classmethod
    def absent(cls) -> FaceObs:
        return cls(bbox=None, quality=0.0, pose_valid=False)


@dataclass(frozen=True, slots=True)
class HandObs:
    bbox: Box | None
    handedness: str = "unknown"
    confidence: float = 0.0
    activity: bool = False
    wrist: tuple[float, float] | None = None
    index_tip: tuple[float, float] | None = None
    hand_id: int | None = None
    # Wrist move since this hand's previous pass (normalized image units) and the time between.
    displacement: float = 0.0
    dt_ms: int = 0


@dataclass(frozen=True, slots=True)
class PoseObs:
    left_shoulder: tuple[float, float, float] | None = None   # x, y, visibility
    right_shoulder: tuple[float, float, float] | None = None

    def shoulder_midpoint(self, min_visibility: float) -> tuple[float, float] | None:
        left, right = self.left_shoulder, self.right_shoulder
        if left is None or right is None or min(left[2], right[2]) < min_visibility:
            return None
        return (left[0] + right[0]) / 2, (left[1] + right[1]) / 2


@dataclass(frozen=True, slots=True)
class ImageQualityObs:
    status: str = "not_run"
    brightness: float | None = None
    entropy: float | None = None
    blur: float | None = None
    occlusion_estimate: float | None = None
    contrast: float | None = None
    edge_density: float | None = None
    # The camera keeps delivering the same picture: a device fault, not behaviour.
    frozen: bool = False

    @property
    def available(self) -> bool:
        return self.status not in {"", "not_run"} or self.blur is not None


@dataclass(frozen=True, slots=True)
class Observation:
    frame_id: int
    session_id: UUID
    captured_at_utc: datetime
    monotonic_ms: int
    frame_size: tuple[int, int]
    people: tuple[PersonObs, ...] = ()
    objects: tuple[ObjectObs, ...] = ()
    faces: tuple[FaceObs, ...] = ()
    hands: tuple[HandObs, ...] = ()
    poses: tuple[PoseObs, ...] = ()
    image_quality: ImageQualityObs = field(default_factory=ImageQualityObs)
    analyzers: Mapping[str, ObservationHealth] = field(default_factory=dict)
    has_image_quality: bool = False

    def analyzer_status(self, name: str) -> str | None:
        health = self.analyzers.get(name)
        return health.status if health is not None else None

    @classmethod
    def from_bundle(cls, bundle: ObservationBundle) -> Observation:
        quality = bundle.image_quality or {}
        return cls(
            frame_id=bundle.frame_id,
            session_id=bundle.session_id,
            captured_at_utc=bundle.captured_at_utc,
            monotonic_ms=bundle.captured_at_monotonic_ms,
            frame_size=bundle.frame_size,
            people=tuple(_person(item) for item in bundle.people),
            objects=tuple(_object(item) for item in bundle.objects),
            faces=tuple(_face(item) for item in bundle.faces),
            hands=tuple(_hand(item) for item in bundle.hands),
            poses=tuple(_pose(item) for item in bundle.poses),
            image_quality=ImageQualityObs(
                status=str(quality.get("status", "")),
                brightness=_opt_float(quality.get("brightness")),
                entropy=_opt_float(quality.get("entropy")),
                blur=_opt_float(quality.get("blur")),
                occlusion_estimate=_opt_float(quality.get("occlusion_estimate")),
                contrast=_opt_float(quality.get("contrast")),
                edge_density=_opt_float(quality.get("edge_density")),
                frozen=bool(quality.get("frozen", False)),
            ),
            analyzers=dict(bundle.analyzer_health),
            has_image_quality=bool(quality),
        )


def _opt_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _person(item: Mapping[str, Any]) -> PersonObs:
    track = item.get("track_id")
    visibility = item.get("visibility", item.get("face_visibility", 1.0))
    return PersonObs(Box.parse(item.get("bbox")), float(item.get("confidence", 0.0)),
                     str(track) if track else None, float(visibility))


def _object(item: Mapping[str, Any]) -> ObjectObs:
    track_id = item.get("object_track_id")
    confirmed = item.get("track_confirmed")
    return ObjectObs(
        label=str(item.get("class", "")),
        confidence=float(item.get("confidence", 0.0)),
        bbox=Box.parse(item.get("bbox")),
        object_track_id=int(track_id) if track_id is not None else None,
        track_confirmed=bool(confirmed) if confirmed is not None else None,
        model_name=str(item["model_name"]) if item.get("model_name") else None,
        model_version=str(item["model_version"]) if item.get("model_version") else None,
    )


def _eye(value: Any) -> EyeObs | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return EyeObs(float(value["h"]), float(value["v"]), float(value["openness"]), float(value["eye_width_px"]))
    except (KeyError, TypeError, ValueError):
        return None


def _mouth(value: Any) -> MouthObs | None:
    if not isinstance(value, Mapping) or value.get("mar") is None:
        return None
    return MouthObs(float(value["mar"]))


def _keypoints(value: Any) -> dict[str, tuple[float, float]]:
    if not isinstance(value, Mapping):
        return {}
    points = {}
    for name, point in value.items():
        if isinstance(point, list | tuple) and len(point) >= 2:
            points[str(name)] = (float(point[0]), float(point[1]))
    return points


def _face(item: Mapping[str, Any]) -> FaceObs:
    status = item.get("calibration_status")
    return FaceObs(
        bbox=Box.parse(item.get("bbox")),
        yaw=float(item.get("yaw", 0.0)),
        pitch=float(item.get("pitch", 0.0)),
        roll=float(item.get("roll", 0.0)),
        pose_valid=bool(item.get("pose_valid", True)),
        quality=float(item.get("quality", 1.0)),
        calibration_status=str(status) if status is not None else None,
        gaze=_eye(item.get("gaze")),
        mouth=_mouth(item.get("mouth")),
        keypoints=_keypoints(item.get("keypoints")),
        face_width=_opt_float(item.get("face_width")),
    )


def _landmark(points: Any, index: int) -> tuple[float, float, float] | None:
    for point in points if isinstance(points, list) else ():
        if isinstance(point, Mapping) and int(point.get("index", -1)) == index:
            return float(point.get("x", 0.0)), float(point.get("y", 0.0)), float(point.get("visibility", 0.0))
    return None


def _pose(item: Mapping[str, Any]) -> PoseObs:
    landmarks = item.get("landmarks")
    return PoseObs(_landmark(landmarks, 11), _landmark(landmarks, 12))


def _hand(item: Mapping[str, Any]) -> HandObs:
    return HandObs(
        bbox=Box.parse(item.get("bbox")),
        handedness=str(item.get("handedness", "unknown")),
        confidence=float(item.get("confidence", 0.0)),
        activity=bool(item.get("activity", False)),
        wrist=_point(item.get("wrist")),
        index_tip=_point(item.get("index_tip")),
        hand_id=int(item["hand_id"]) if item.get("hand_id") is not None else None,
        displacement=float(item.get("displacement", 0.0)),
        dt_ms=int(item.get("dt_ms", 0)),
    )


def _point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, list | tuple) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


__all__ = [
    "Box",
    "EyeObs",
    "MouthObs",
    "FaceObs",
    "HandObs",
    "ImageQualityObs",
    "ObjectObs",
    "Observation",
    "PersonObs",
    "PoseObs",
]
