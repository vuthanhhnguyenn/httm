from __future__ import annotations

from proctoring_core.config.schema import AppConfig, BehaviorConfig, EvidenceConfig


def test_experimental_signals_are_off_by_default() -> None:
    behavior = BehaviorConfig()
    assert behavior.gaze.enabled is False
    assert behavior.talking.enabled is False


def test_evidence_capture_is_off_by_default() -> None:
    # Evidence images are personal data: both config and policy must opt in.
    assert EvidenceConfig().evidence_frame_enabled is False
    assert EvidenceConfig().evidence_clip_enabled is False
    assert AppConfig().evidence.evidence_frame_enabled is False
