from __future__ import annotations

from typing import Any

from ..types import EventType, Severity
from .temporal_buffer import ConditionState, TemporalCondition, TemporalSample


class SafetyRuleEvaluator:
    def __init__(self, *, min_duration_ms: int = 1000, face_visibility_threshold: float = 0.35,
                 blur_threshold: float = 20.0, occlusion_threshold: float = 0.8,
                 durations_ms: dict[EventType, int] | None = None,
                 max_brightness: float | None = None, min_contrast: float | None = None,
                 min_edge_density: float | None = None, require_no_person: bool = False) -> None:
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_edge_density = min_edge_density
        self.require_no_person = require_no_person
        self.face_visibility_threshold = face_visibility_threshold
        self.blur_threshold = blur_threshold
        self.occlusion_threshold = occlusion_threshold
        durations = durations_ms or {}
        self._conditions = {event: TemporalCondition(min_duration_ms=durations.get(event, min_duration_ms)) for event in (EventType.LEAVING_SEAT, EventType.CAMERA_BLOCKED, EventType.FACE_NOT_VISIBLE)}

    def evaluate(self, *, session_id: Any, track_id: Any, sample: dict[str, Any]) -> list[dict[str, Any]]:
        quality = sample.get("image_quality", {}) or {}
        visibility = float(sample.get("visibility", sample.get("face_visibility", 1.0)))
        leaving = bool(sample.get("leaving_seat", False))
        blocked = self._low_information(quality) and not (
            self.require_no_person and bool(sample.get("person_present", False)))
        face_missing = visibility < self.face_visibility_threshold or not bool(sample.get("face_visible", True))
        checks = ((EventType.LEAVING_SEAT, leaving, Severity.MEDIUM), (EventType.CAMERA_BLOCKED, blocked, Severity.HIGH), (EventType.FACE_NOT_VISIBLE, face_missing, Severity.MEDIUM))
        results: list[dict[str, Any]] = []
        timestamp = int(sample.get("monotonic_ms", 0))
        for event_type, matched, severity in checks:
            condition = self._conditions[event_type]
            state = condition.update(TemporalSample(timestamp, matched, visibility if event_type is EventType.FACE_NOT_VISIBLE else 1.0))
            if state is not ConditionState.INACTIVE or condition.last_closed:
                results.append({"session_id": session_id, "track_id": track_id, "event_type": event_type, "severity": severity if state is ConditionState.ACTIVE else Severity.LOW, "confidence": 1.0 if matched else 0.0, "active": state is ConditionState.ACTIVE, "state": state.value, "metrics": {"visibility": visibility, "blur": quality.get("blur"), "occlusion_estimate": quality.get("occlusion_estimate"), **condition.current_metrics()}, "reason_codes": [f"{event_type.value}_SUSTAINED"]})
        return results

    def _low_information(self, quality: dict[str, Any]) -> bool:
        """Covered or failed lens: blurred, occluded, near-black, flat or without edges."""

        checks = (
            float(quality.get("blur", 100.0)) < self.blur_threshold,
            float(quality.get("occlusion_estimate", 0.0)) >= self.occlusion_threshold,
            self.max_brightness is not None and "brightness" in quality
            and float(quality["brightness"]) <= self.max_brightness,
            self.min_contrast is not None and "contrast" in quality
            and float(quality["contrast"]) <= self.min_contrast,
            self.min_edge_density is not None and "edge_density" in quality
            and float(quality["edge_density"]) <= self.min_edge_density,
        )
        return any(checks)
