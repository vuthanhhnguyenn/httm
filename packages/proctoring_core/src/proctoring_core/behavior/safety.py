from __future__ import annotations

from typing import Any

from ..types import EventType, Severity
from .temporal_buffer import ConditionState, TemporalCondition, TemporalSample


class SafetyRuleEvaluator:
    def __init__(self, *, min_duration_ms: int = 1000, face_visibility_threshold: float = 0.35, blur_threshold: float = 20.0) -> None:
        self.face_visibility_threshold = face_visibility_threshold
        self.blur_threshold = blur_threshold
        self._conditions = {event: TemporalCondition(min_duration_ms=min_duration_ms) for event in (EventType.LEAVING_SEAT, EventType.CAMERA_BLOCKED, EventType.FACE_NOT_VISIBLE)}

    def evaluate(self, *, session_id: Any, track_id: Any, sample: dict[str, Any]) -> list[dict[str, Any]]:
        quality = sample.get("image_quality", {}) or {}
        visibility = float(sample.get("visibility", sample.get("face_visibility", 1.0)))
        bbox = sample.get("bbox") or {}
        center_x = float(bbox.get("x", 0.0)) + float(bbox.get("width", 0.0)) / 2
        center_y = float(bbox.get("y", 0.0)) + float(bbox.get("height", 0.0)) / 2
        zone = sample.get("allowed_zone")
        outside_zone = False
        if isinstance(zone, dict):
            outside_zone = not (float(zone.get("x", 0.0)) <= center_x <= float(zone.get("x", 0.0)) + float(zone.get("width", 1.0)) and float(zone.get("y", 0.0)) <= center_y <= float(zone.get("y", 0.0)) + float(zone.get("height", 1.0)))
        leaving = bool(sample.get("leaving_seat", False)) or outside_zone or float(bbox.get("y", 0.0)) > 0.9
        blocked = float(quality.get("blur", 100.0)) < self.blur_threshold or float(quality.get("occlusion_estimate", 0.0)) >= 0.8
        face_missing = visibility < self.face_visibility_threshold or not bool(sample.get("face_visible", True))
        checks = ((EventType.LEAVING_SEAT, leaving, Severity.MEDIUM), (EventType.CAMERA_BLOCKED, blocked, Severity.HIGH), (EventType.FACE_NOT_VISIBLE, face_missing, Severity.MEDIUM))
        results: list[dict[str, Any]] = []
        timestamp = int(sample.get("monotonic_ms", 0))
        for event_type, matched, severity in checks:
            condition = self._conditions[event_type]
            state = condition.update(TemporalSample(timestamp, matched, visibility if event_type is EventType.FACE_NOT_VISIBLE else 1.0))
            if state is not ConditionState.INACTIVE or condition.last_closed:
                results.append({"session_id": session_id, "track_id": track_id, "event_type": event_type, "severity": severity if state is ConditionState.ACTIVE else Severity.LOW, "confidence": 1.0 if matched else 0.0, "active": state is ConditionState.ACTIVE, "metrics": {"visibility": visibility, "blur": quality.get("blur"), "occlusion_estimate": quality.get("occlusion_estimate"), **condition.current_metrics()}, "reason_codes": [f"{event_type.value}_SUSTAINED"]})
        return results
