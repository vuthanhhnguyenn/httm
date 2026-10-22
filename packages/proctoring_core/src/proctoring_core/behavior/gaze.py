"""Experimental gaze rules: looking off the screen, and sustained sideways glances.

Both need a gaze calibration and a clear view of the eyes; otherwise they stay
silent instead of guessing. They are supporting signals with a low risk cap.
"""

from __future__ import annotations

from typing import Any

from ..calibration.gaze import GazeProfile, GazeSample
from ..config.schema import GazeConfig
from ..observations import FaceObs
from ..types import EventType, Severity
from .temporal_buffer import ConditionState, TemporalCondition, TemporalSample


class GazeRuleEvaluator:
    def __init__(self, config: GazeConfig, *, head_turn_degrees: float, min_face_quality: float,
                 max_sample_gap_ms: int) -> None:
        self.config = config
        self.head_turn_degrees = head_turn_degrees
        self.min_face_quality = min_face_quality
        self._conditions = {
            EventType.GAZE_OFF_SCREEN: TemporalCondition(
                min_duration_ms=round(config.off_screen_min_duration_seconds * 1000),
                max_sample_gap_ms=max_sample_gap_ms),
            EventType.GAZE_SIDEWAYS: TemporalCondition(
                min_duration_ms=round(config.sideways_min_duration_seconds * 1000),
                max_sample_gap_ms=max_sample_gap_ms),
        }

    def observation_quality(self, face: FaceObs | None) -> float:
        """0 when the eyes cannot be measured reliably, else the face quality."""

        if face is None or face.gaze is None or not face.pose_valid or face.quality < self.min_face_quality:
            return 0.0
        eye = face.gaze
        if (eye.eye_width_px < self.config.min_eye_width_px or eye.openness < self.config.min_openness
                or abs(face.yaw) > self.config.max_head_yaw_degrees):
            return 0.0
        return max(0.0, min(1.0, face.quality))

    def evaluate(self, *, session_id: Any, track_id: Any, face: FaceObs | None, profile: GazeProfile | None,
                 monotonic_ms: int) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return []
        quality = self.observation_quality(face) if profile is not None else 0.0
        off_screen = sideways = False
        point: tuple[float, float] | None = None
        offset = 0.0
        if quality > 0 and face is not None and face.gaze is not None and profile is not None:
            sample = GazeSample(face.gaze.h, face.gaze.v, face.yaw, face.pitch)
            point = profile.screen_point(sample)
            margin = self.config.off_screen_margin
            off_screen = (self.config.off_screen_enabled and point is not None
                          and not (-margin <= point[0] <= 1 + margin and -margin <= point[1] <= 1 + margin))
            offset = sample.h - profile.center.h
            head_turned = abs(face.yaw - profile.center.yaw) >= self.head_turn_degrees
            sideways = abs(offset) >= self.config.sideways_ratio and not head_turned
        results: list[dict[str, Any]] = []
        for event_type, matched in ((EventType.GAZE_OFF_SCREEN, off_screen), (EventType.GAZE_SIDEWAYS, sideways)):
            condition = self._conditions[event_type]
            state = condition.update(TemporalSample(monotonic_ms, matched, quality))
            if state is ConditionState.INACTIVE and not condition.last_closed:
                continue
            metrics: dict[str, Any] = {
                "gaze_quality": round(quality, 3),
                "eye_offset": round(offset, 3),
                "risk_cap": self.config.risk_cap,
                "experimental": True,
                **condition.current_metrics(),
            }
            if point is not None:
                metrics["screen_x"], metrics["screen_y"] = round(point[0], 3), round(point[1], 3)
            results.append({
                "session_id": session_id, "track_id": track_id, "event_type": event_type,
                "severity": Severity.LOW, "confidence": quality, "active": state is ConditionState.ACTIVE,
                "state": state.value, "metrics": metrics,
                "reason_codes": [f"{event_type.value}_SUSTAINED"],
            })
        return results


__all__ = ["GazeRuleEvaluator"]
