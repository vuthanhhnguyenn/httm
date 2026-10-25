"""Load and validate session settings, and build the analyzer factory from them."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from proctoring_ai.acceleration import AUTO_DEVICE, cuda_status
from proctoring_ai.analyzer_factory import AnalyzerFactory
from proctoring_core.config.schema import AppConfig, DetectionConfig, ExamPolicyConfig

DEVICE_ENV = "PROCTORING_AI_DEVICE"


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Validated configuration plus the raw mappings snapshotted into the session."""

    config: AppConfig
    policy: ExamPolicyConfig
    raw_config: dict[str, Any]
    raw_policy: dict[str, Any]
    device: str | None

    @property
    def manifest_path(self) -> str:
        return self.config.detection.model_manifest


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def load_settings(repository_root: Path, *, environ: Mapping[str, str] | None = None) -> RuntimeSettings:
    """Read ``configs/default.yaml`` and ``configs/exam_policy.yaml`` and apply the device override.

    Without an override YOLO picks the first CUDA GPU when PyTorch can use it and falls back
    to CPU with a logged reason. An explicitly requested CUDA device that the running
    interpreter cannot use is an error before any camera is opened, so the operator is not
    left with a silent CPU fallback.
    """

    env = os.environ if environ is None else environ
    raw_config = _read_yaml(repository_root / "configs" / "default.yaml")
    policy_file = _read_yaml(repository_root / "configs" / "exam_policy.yaml")
    raw_policy = dict(policy_file.get("exam_policy", policy_file))
    device = env.get(DEVICE_ENV) or None
    if device == AUTO_DEVICE:
        device = None
    if device:
        if device != "cpu":
            available, detail = cuda_status()
            if not available:
                raise RuntimeError(f"CUDA chưa khả dụng trong môi trường Python đang chạy: {detail}")
        raw_config = {**raw_config, "detection": {**raw_config.get("detection", {}), "device": device}}
    return RuntimeSettings(
        config=AppConfig.model_validate(raw_config),
        policy=ExamPolicyConfig.model_validate(raw_policy),
        raw_config=raw_config,
        raw_policy=raw_policy,
        device=device,
    )


def analyzer_options(detection: DetectionConfig) -> dict[str, dict[str, Any]]:
    timeout = detection.analyzer_timeout_ms
    return {
        "face": {"timeout_ms": timeout, "min_confidence": detection.face_min_confidence},
        "pose": {"timeout_ms": timeout, "min_confidence": detection.pose_min_confidence},
        "hands": {"timeout_ms": timeout, "min_confidence": detection.hand_min_confidence,
                  "movement_threshold": detection.hand_activity_movement},
    }


def create_factory(repository_root: Path, settings: RuntimeSettings) -> AnalyzerFactory:
    detection = settings.config.detection
    return AnalyzerFactory(
        repository_root / detection.model_manifest,
        detector_image_size=detection.image_size,
        detector_device=detection.device,
        analyzer_options=analyzer_options(detection),
    )


__all__ = ["DEVICE_ENV", "RuntimeSettings", "analyzer_options", "create_factory", "load_settings"]
