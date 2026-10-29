"""Pydantic schemas for every configurable camera, rule and runtime threshold.

Unknown keys are rejected so a misspelled threshold cannot silently change behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..types import EventType, ExamMode


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CameraConfig(StrictModel):
    source: str = Field(default="webcam", min_length=1)
    device_index: int = Field(default=0, ge=0)
    width: int = Field(default=1280, ge=1, le=7680)
    height: int = Field(default=720, ge=1, le=4320)
    target_fps: float = Field(default=30, ge=20, le=30)
    frame_queue_size: int = Field(default=2, ge=1, le=64)


class DetectionConfig(StrictModel):
    object_confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    person_confidence_threshold: float = Field(default=0.6, ge=0, le=1)
    image_size: int = Field(default=416, ge=160, le=1280)
    model_manifest: str = Field(default="./models/manifest.yaml", min_length=1)
    device: str | None = Field(default=None, min_length=1)
    pose_interval_ms: int = Field(default=250, ge=0, le=500)
    hands_interval_ms: int = Field(default=200, ge=0, le=500)
    analyzer_timeout_ms: int = Field(default=1200, ge=100, le=120_000)
    face_min_confidence: float = Field(default=0.5, ge=0, le=1)
    pose_min_confidence: float = Field(default=0.45, ge=0, le=1)
    hand_min_confidence: float = Field(default=0.4, ge=0, le=1)
    # Normalized centre displacement between two hand passes that counts as movement.
    hand_activity_movement: float = Field(default=0.025, ge=0, le=1)


class QualityConfig(StrictModel):
    """Observation gates shared by face-based rules."""

    face_min_quality: float = Field(default=0.35, ge=0, le=1)
    # Two samples further apart than this are never joined into one sustained behaviour.
    max_sample_gap_ms: int = Field(default=600, ge=50, le=10_000)


class AssociationConfig(StrictModel):
    # A face/hand belongs to the person when its centre is inside the person box widened by this share.
    person_box_margin: float = Field(default=0.1, ge=0, le=1)


class HeadTurnConfig(StrictModel):
    yaw_degrees: float = Field(default=22, ge=0, le=180)
    minimum_duration_seconds: float = Field(default=0.3, ge=0)
    fast_yaw_degrees: float = Field(default=32, ge=0, le=180)
    fast_duration_seconds: float = Field(default=0.2, ge=0.1, le=2)
    require_calibration: bool = True
    neutral_degrees: float = Field(default=12, ge=0, le=25)
    neutral_duration_seconds: float = Field(default=0.3, ge=0, le=2)
    # An active turn stays active until the angle drops this far below the threshold.
    hysteresis_degrees: float = Field(default=5, ge=0, le=30)
    # The fast path threshold is at least the normal threshold plus this margin.
    fast_margin_degrees: float = Field(default=5, ge=0, le=30)


class LookDownConfig(StrictModel):
    enabled: bool = True
    # The short high-pitch path; off where brief glances down are normal (keyboard).
    fast_enabled: bool = True
    pitch_degrees: float = Field(default=15, ge=0, le=180)
    minimum_duration_seconds: float = Field(default=0.4, ge=0)
    fast_pitch_degrees: float = Field(default=28, ge=0, le=180)
    fast_duration_seconds: float = Field(default=0.2, ge=0.1, le=2)
    fast_margin_degrees: float = Field(default=5, ge=0, le=30)


class AbnormalHeadMovementConfig(StrictModel):
    window_seconds: float = Field(default=30, gt=0)
    minimum_turn_count: int = Field(default=4, ge=1)


class PhoneConfig(StrictModel):
    minimum_confidence: float = Field(default=0.55, ge=0, le=1)
    minimum_duration_seconds: float = Field(default=0.6, ge=0)
    release_confidence: float = Field(default=0.35, ge=0, le=1)
    maximum_gap_ms: int = Field(default=600, ge=0, le=1000)
    fast_confidence: float = Field(default=0.85, ge=0.55, le=1)
    fast_duration_seconds: float = Field(default=0.2, ge=0.2, le=1)
    fast_min_hits: int = Field(default=2, ge=2, le=5)
    max_sample_gap_ms: int = Field(default=1200, ge=100, le=10_000)
    fast_max_sample_gap_ms: int = Field(default=350, ge=50, le=5_000)
    # Once confirmed, dropouts or track switches shorter than this keep the same event.
    episode_gap_ms: int = Field(default=1500, ge=0, le=10_000)
    # A phone touching the face must be held: without a hand nearby (recently seen) the box is
    # ignored, e.g. a camera clipped to glasses. Phones elsewhere (desk, lap) are unaffected.
    require_hand_near_face: bool = True
    face_margin: float = Field(default=0.35, ge=0, le=3)       # face box grown by this × face width
    hand_radius: float = Field(default=1.2, ge=0, le=5)         # hand within this × face width
    hand_memory_ms: int = Field(default=800, ge=0, le=5_000)    # hands run at a lower rate than YOLO
    # Optional geometric filter (off by default): a box much larger than a hand-held phone is
    # more likely a wear-cam or another device. Only applied to phone events.
    geometry_max_height_ratio: float | None = Field(default=None, ge=0, le=1)
    geometry_max_area_ratio: float | None = Field(default=None, ge=0, le=1)
    classes: list[str] = Field(default_factory=lambda: ["cell phone", "phone", "mobile"], min_length=1)


class DocumentConfig(StrictModel):
    minimum_confidence: float = Field(default=0.7, ge=0, le=1)
    minimum_duration_seconds: float = Field(default=1, ge=0)
    max_sample_gap_ms: int = Field(default=1200, ge=100, le=10_000)
    episode_gap_ms: int = Field(default=1500, ge=0, le=10_000)
    # COCO has no reliable class for loose paper or generic documents; only books are detected.
    classes: list[str] = Field(default_factory=lambda: ["book"], min_length=1)


class DurationRuleConfig(StrictModel):
    minimum_duration_seconds: float = Field(default=2, ge=0)


class GazeConfig(StrictModel):
    """Experimental eye-direction signals; silent without calibration or a clear view."""

    enabled: bool = True
    off_screen_enabled: bool = True
    min_eye_width_px: float = Field(default=14, ge=0, le=200)
    min_openness: float = Field(default=0.15, ge=0, le=1)
    max_head_yaw_degrees: float = Field(default=35, ge=0, le=90)
    # Normalized screen coordinates beyond [−margin, 1 + margin] count as off screen.
    off_screen_margin: float = Field(default=0.12, ge=0, le=1)
    off_screen_min_duration_seconds: float = Field(default=1.5, ge=0)
    # Iris offset from the calibrated centre (share of eye width) while the head stays straight.
    sideways_ratio: float = Field(default=0.18, ge=0, le=0.5)
    sideways_min_duration_seconds: float = Field(default=2.0, ge=0)
    # Replay/quick setup: learn the neutral eye position over this long after calibration starts.
    center_calibration_seconds: float = Field(default=2.0, ge=0.5, le=10)
    risk_cap: float = Field(default=15, ge=0, le=100)


class TalkingConfig(StrictModel):
    """Experimental mouth-rhythm signal; yawns and a held-open mouth do not count."""

    enabled: bool = True
    window_seconds: float = Field(default=2.0, ge=0.5, le=10)
    min_samples: int = Field(default=6, ge=3, le=200)
    min_oscillations: int = Field(default=3, ge=1, le=20)
    min_amplitude: float = Field(default=0.06, ge=0, le=2)
    min_std: float = Field(default=0.02, ge=0, le=1)
    yawn_mar: float = Field(default=0.55, ge=0, le=3)
    max_head_yaw_degrees: float = Field(default=40, ge=0, le=90)
    min_duration_seconds: float = Field(default=2.5, ge=0)
    risk_cap: float = Field(default=15, ge=0, le=100)


class FaceVisibilityConfig(StrictModel):
    minimum_duration_seconds: float = Field(default=2, ge=0)
    visibility_threshold: float = Field(default=0.35, ge=0, le=1)


class CameraBlockedConfig(StrictModel):
    minimum_duration_seconds: float = Field(default=2, ge=0)
    blur_threshold: float = Field(default=20, ge=0)
    occlusion_threshold: float = Field(default=0.8, ge=0, le=1)
    # Low-information picture: any of these (normalized brightness, grey std, edge share).
    max_brightness: float = Field(default=0.06, ge=0, le=1)
    min_contrast: float = Field(default=6.0, ge=0, le=128)
    min_edge_density: float = Field(default=0.002, ge=0, le=1)
    # Corroboration: a covered lens also hides the person from the detector.
    require_no_person: bool = True


class LeavingSeatConfig(StrictModel):
    minimum_duration_seconds: float = Field(default=2, ge=0)
    # The seat zone is the median person box over this long at session start / re-calibration.
    zone_learning_seconds: float = Field(default=3.0, ge=0.5, le=30)
    # Leaving: anchor outside the zone grown by this share, or box smaller than this share of it.
    zone_margin: float = Field(default=0.15, ge=0, le=2)
    min_area_ratio: float = Field(default=0.35, ge=0, le=1)
    min_shoulder_visibility: float = Field(default=0.5, ge=0, le=1)


class HandActivityConfig(StrictModel):
    """Distances are shares of the face width; speed is face widths per second."""

    minimum_duration_seconds: float = Field(default=2, ge=0)
    near_ear_radius: float = Field(default=0.45, ge=0, le=3)
    near_mouth_radius: float = Field(default=0.4, ge=0, le=3)
    # Fingertips this far above the chin are over the mouth; lower is a chin rest.
    chin_rest_margin: float = Field(default=0.12, ge=0, le=1)
    rapid_speed: float = Field(default=1.8, ge=0, le=50)
    risk_cap: float = Field(default=25, ge=0, le=100)


_FACE_SIGNALS = [
    EventType.HEAD_TURN_LEFT, EventType.HEAD_TURN_RIGHT, EventType.LOOK_DOWN,
    EventType.ABNORMAL_HEAD_MOVEMENT, EventType.POSSIBLE_TALKING, EventType.SUSPICIOUS_HAND_ACTIVITY,
    EventType.GAZE_OFF_SCREEN, EventType.GAZE_SIDEWAYS, EventType.LOOK_AT_NEIGHBOR,
]


def _default_hierarchy() -> dict[EventType, list[EventType]]:
    return {
        EventType.CAMERA_BLOCKED: [EventType.PERSON_MISSING, EventType.FACE_NOT_VISIBLE,
                                   EventType.LEAVING_SEAT, *_FACE_SIGNALS],
        EventType.PERSON_MISSING: [EventType.FACE_NOT_VISIBLE, EventType.LEAVING_SEAT, *_FACE_SIGNALS],
        EventType.LEAVING_SEAT: [EventType.FACE_NOT_VISIBLE, *_FACE_SIGNALS],
    }


class BehaviorConfig(StrictModel):
    quality: QualityConfig = Field(default_factory=QualityConfig)
    association: AssociationConfig = Field(default_factory=AssociationConfig)
    head_turn: HeadTurnConfig = Field(default_factory=HeadTurnConfig)
    look_down: LookDownConfig = Field(default_factory=LookDownConfig)
    abnormal_head_movement: AbnormalHeadMovementConfig = Field(default_factory=AbnormalHeadMovementConfig)
    phone: PhoneConfig = Field(default_factory=PhoneConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    person_missing: DurationRuleConfig = Field(default_factory=DurationRuleConfig)
    multiple_person: DurationRuleConfig = Field(default_factory=DurationRuleConfig)
    leaving_seat: LeavingSeatConfig = Field(default_factory=LeavingSeatConfig)
    camera_blocked: CameraBlockedConfig = Field(default_factory=CameraBlockedConfig)
    face_not_visible: FaceVisibilityConfig = Field(default_factory=FaceVisibilityConfig)
    hand_activity: HandActivityConfig = Field(default_factory=HandActivityConfig)
    gaze: GazeConfig = Field(default_factory=GazeConfig)
    talking: TalkingConfig = Field(default_factory=TalkingConfig)
    # Root signal → signals it explains; explained signals add no event or risk of their own.
    hierarchy: dict[EventType, list[EventType]] = Field(default_factory=_default_hierarchy)

    @model_validator(mode="after")
    def _acyclic_hierarchy(self) -> BehaviorConfig:
        from ..behavior.arbiter import validate_hierarchy

        validate_hierarchy(self.hierarchy)
        return self


class RiskConfig(StrictModel):
    maximum_score: Literal[100] = 100
    update_interval_ms: int = Field(default=250, ge=1)
    persist_epsilon: float = Field(default=1.0, ge=0)
    decay_per_second: float = Field(default=1.5, ge=0, le=10)
    recovery_half_life_seconds: float | None = Field(default=1.0, ge=0.1, le=30)
    active_event_ttl_seconds: float = Field(default=2, gt=0, le=10)
    correlation_window_seconds: float = Field(default=5, gt=0, le=30)
    correlation_confirmation_seconds: float = Field(default=0.25, ge=0, le=2)
    correlation_cap: float = Field(default=20, ge=0, le=40)
    # Live total allowed to supporting signals (gaze, talking, hands); below the HIGH band (60).
    supporting_group_cap: float = Field(default=30, ge=0, le=59)


class EvidenceConfig(StrictModel):
    evidence_frame_enabled: bool = True
    evidence_clip_enabled: bool = False
    retention_days: int = Field(default=30, gt=0)
    pre_event_seconds: float = Field(default=3, ge=0, le=30)
    post_event_seconds: float = Field(default=3, ge=0, le=30)


class RuntimeConfig(StrictModel):
    timezone: Literal["UTC"] = "UTC"
    max_active_sessions: int = Field(default=1, ge=1)
    storage_root: str = Field(default="./storage", min_length=1)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        merged[key] = (_deep_merge(current, value)
                       if isinstance(current, Mapping) and isinstance(value, Mapping) else value)
    return merged


class AppConfig(StrictModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    # Partial ``behavior`` tables merged over the defaults for each exam mode.
    exam_modes: dict[ExamMode, dict[str, Any]] = Field(default_factory=dict)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def _validate_mode_overrides(self) -> AppConfig:
        for mode in self.exam_modes:
            self.behavior_for(mode)
        return self

    def behavior_for(self, mode: ExamMode | str) -> BehaviorConfig:
        """Behaviour defaults with the exam-mode override applied and re-validated."""

        override = self.exam_modes.get(ExamMode(mode), {})
        if not override:
            return self.behavior
        return BehaviorConfig.model_validate(_deep_merge(self.behavior.model_dump(), override))


class ExamPolicyConfig(StrictModel):
    name: str = Field(default="Default exam policy", min_length=1, max_length=120)
    allow_book: bool = False
    allow_scratch_paper: bool = True
    allow_phone: bool = False
    exam_mode_default: ExamMode = ExamMode.COMPUTER
    allowed_materials: list[str] = Field(default_factory=list)
    evidence_frame_enabled: bool = True
    evidence_clip_enabled: bool = False
    pre_event_seconds: float = Field(default=3, ge=0, le=30)
    post_event_seconds: float = Field(default=3, ge=0, le=30)
    retention_days: int = Field(default=30, gt=0)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def load_config_files(config_path: str | Path, policy_path: str | Path) -> tuple[AppConfig, ExamPolicyConfig]:
    config_data = _read_yaml(Path(config_path))
    policy_data = _read_yaml(Path(policy_path))
    policy_payload = policy_data.get("exam_policy", policy_data)
    if not isinstance(policy_payload, Mapping):
        raise ValueError("exam_policy must be a mapping")
    return AppConfig.model_validate(config_data), ExamPolicyConfig.model_validate(policy_payload)


def validate_config(config: Mapping[str, Any], policy: Mapping[str, Any] | None = None) -> tuple[AppConfig, ExamPolicyConfig | None]:
    """Validate config and optional policy, rejecting out-of-schema thresholds."""

    validated_config = AppConfig.model_validate(config)
    validated_policy = ExamPolicyConfig.model_validate(policy) if policy is not None else None
    return validated_config, validated_policy


__all__ = [
    "AppConfig",
    "BehaviorConfig",
    "GazeConfig",
    "TalkingConfig",
    "ExamPolicyConfig",
    "ValidationError",
    "load_config_files",
    "validate_config",
]
