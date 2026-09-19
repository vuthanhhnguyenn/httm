"""Pydantic schemas for every configurable camera, rule and runtime threshold.

Unknown keys are rejected so a misspelled threshold cannot silently change behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CameraConfig(StrictModel):
    source: str = Field(default="webcam", min_length=1)
    device_index: int = Field(default=0, ge=0)
    width: int = Field(default=1280, ge=1, le=7680)
    height: int = Field(default=720, ge=1, le=4320)
    target_fps: float = Field(default=20, gt=0, le=120)
    frame_queue_size: int = Field(default=2, ge=1, le=64)


class DetectionConfig(StrictModel):
    object_confidence_threshold: float = Field(default=0.6, ge=0, le=1)
    person_confidence_threshold: float = Field(default=0.6, ge=0, le=1)
    model_manifest: str = Field(default="./models/manifest.yaml", min_length=1)


class HeadTurnConfig(StrictModel):
    yaw_degrees: float = Field(default=25, ge=0, le=180)
    minimum_duration_seconds: float = Field(default=1.5, ge=0)


class LookDownConfig(StrictModel):
    pitch_degrees: float = Field(default=20, ge=0, le=180)
    minimum_duration_seconds: float = Field(default=2, ge=0)


class AbnormalHeadMovementConfig(StrictModel):
    window_seconds: float = Field(default=30, gt=0)
    minimum_turn_count: int = Field(default=4, ge=1)


class PhoneConfig(StrictModel):
    minimum_confidence: float = Field(default=0.7, ge=0, le=1)
    minimum_duration_seconds: float = Field(default=1, ge=0)


class DocumentConfig(StrictModel):
    minimum_confidence: float = Field(default=0.7, ge=0, le=1)
    minimum_duration_seconds: float = Field(default=1, ge=0)


class DurationRuleConfig(StrictModel):
    minimum_duration_seconds: float = Field(default=2, ge=0)


class CountRuleConfig(StrictModel):
    minimum_duration_seconds: float = Field(default=2, ge=0)
    minimum_count: int = Field(default=2, ge=1)


class BehaviorConfig(StrictModel):
    head_turn: HeadTurnConfig = Field(default_factory=HeadTurnConfig)
    look_down: LookDownConfig = Field(default_factory=LookDownConfig)
    abnormal_head_movement: AbnormalHeadMovementConfig = Field(default_factory=AbnormalHeadMovementConfig)
    phone: PhoneConfig = Field(default_factory=PhoneConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    person_missing: DurationRuleConfig = Field(default_factory=DurationRuleConfig)
    multiple_person: DurationRuleConfig = Field(default_factory=DurationRuleConfig)
    leaving_seat: DurationRuleConfig = Field(default_factory=DurationRuleConfig)
    camera_blocked: DurationRuleConfig = Field(default_factory=DurationRuleConfig)
    face_not_visible: DurationRuleConfig = Field(default_factory=DurationRuleConfig)
    hand_activity: CountRuleConfig = Field(default_factory=CountRuleConfig)


class RiskConfig(StrictModel):
    maximum_score: Literal[100] = 100
    update_interval_ms: int = Field(default=250, ge=1)
    persist_epsilon: float = Field(default=1.0, ge=0)


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


class AppConfig(StrictModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


class ExamPolicyConfig(StrictModel):
    name: str = Field(default="Default exam policy", min_length=1, max_length=120)
    allow_book: bool = False
    allow_scratch_paper: bool = True
    allow_phone: bool = False
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
    "ExamPolicyConfig",
    "ValidationError",
    "load_config_files",
    "validate_config",
]
