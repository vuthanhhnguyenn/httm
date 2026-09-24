from __future__ import annotations

from collections import deque
from math import isfinite
from typing import Any

from ..types import EventType, Severity
from .temporal_buffer import ConditionState, TemporalCondition, TemporalSample


class HeadRuleEvaluator:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        head_turn = config.get("head_turn", {}) if isinstance(config.get("head_turn", {}), dict) else {}
        look_down = config.get("look_down", {}) if isinstance(config.get("look_down", {}), dict) else {}
        self.yaw_threshold = float(config.get("yaw_threshold", head_turn.get("yaw_degrees", 35.0)))
        self.fast_yaw_threshold = max(self.yaw_threshold + 5.0,
                                      float(head_turn.get("fast_yaw_degrees", 32.0)))
        self.pitch_threshold = float(config.get("pitch_threshold", look_down.get("pitch_degrees", 15.0)))
        self.min_duration_ms = int(config.get("min_duration_ms", float(head_turn.get("minimum_duration_seconds", 0.8)) * 1000))
        self.fast_duration_ms = round(float(head_turn.get("fast_duration_seconds", 0.25)) * 1000)
        self.down_duration_ms = int(config.get("min_duration_ms", float(look_down.get("minimum_duration_seconds", 0.4)) * 1000))
        self.fast_pitch_threshold = max(self.pitch_threshold + 5.0,
                                        float(look_down.get("fast_pitch_degrees", 28.0)))
        self.fast_down_duration_ms = round(float(look_down.get("fast_duration_seconds", 0.2)) * 1000)
        movement = config.get("abnormal_head_movement", {})
        self.frequency_window_ms = int(config.get("frequency_window_ms", float(movement.get("window_seconds", 30)) * 1000))
        self.minimum_turn_count = int(movement.get("minimum_turn_count", 4))
        self.require_calibration = bool(head_turn.get("require_calibration", False))
        self.neutral_degrees = float(head_turn.get("neutral_degrees", 12))
        self.neutral_duration_ms = int(float(head_turn.get("neutral_duration_seconds", 0.3)) * 1000)
        self._neutral_since: dict[str, int] = {}
        self._turn_armed: dict[str, bool] = {}
        self._last_sample_ms: dict[str, int] = {}
        self._conditions: dict[tuple[str, EventType], TemporalCondition] = {}
        self._fast_conditions: dict[tuple[str, EventType], TemporalCondition] = {}
        self._combined_active: dict[tuple[str, EventType], bool] = {}
        self._turns: dict[str, deque[int]] = {}

    def evaluate(self, *, session_id: Any, track_id: Any, sample: dict[str, Any]) -> list[dict[str, Any]]:
        key_prefix = str(track_id or "camera")
        yaw, pitch = float(sample.get("yaw", 0.0)), float(sample.get("pitch", 0.0))
        quality = float(sample.get("quality", 1))
        valid = (bool(sample.get("pose_valid", True)) and isfinite(yaw) and isfinite(pitch)
                 and isfinite(quality) and quality >= 0.35)
        if self.require_calibration and sample.get("calibration_status") != "calibrated":
            valid = False
        quality = max(0.0, min(1.0, quality)) if isfinite(quality) else 0.0
        checks = (
            (EventType.HEAD_TURN_LEFT, -yaw, self.yaw_threshold, "yaw", self.min_duration_ms),
            (EventType.HEAD_TURN_RIGHT, yaw, self.yaw_threshold, "yaw", self.min_duration_ms),
            (EventType.LOOK_DOWN, pitch, self.pitch_threshold, "pitch", self.down_duration_ms),
        )
        results: list[dict[str, Any]] = []
        timestamp = int(sample.get("monotonic_ms", 0))
        sample_gap = timestamp - self._last_sample_ms.get(key_prefix, timestamp) > 600
        if sample_gap:
            self._neutral_since.pop(key_prefix, None)
        self._last_sample_ms[key_prefix] = timestamp
        turns = self._turns.setdefault(key_prefix, deque(maxlen=256))
        if sample.get("calibration_status") == "collecting":
            turns.clear()
            self._turn_armed[key_prefix] = True
        if valid and abs(yaw) <= self.neutral_degrees:
            neutral_since = self._neutral_since.setdefault(key_prefix, timestamp)
            if timestamp - neutral_since >= self.neutral_duration_ms:
                self._turn_armed[key_prefix] = True
        else:
            self._neutral_since.pop(key_prefix, None)
        while turns and timestamp - turns[0] > self.frequency_window_ms:
            turns.popleft()
        for event_type, magnitude, threshold, metric_name, duration in checks:
            key = (key_prefix, event_type)
            condition = self._conditions.setdefault(key, TemporalCondition(min_duration_ms=duration, window_ms=self.frequency_window_ms, max_sample_gap_ms=600))
            previous = condition.state
            was_active = self._combined_active.get(key, False) and not sample_gap
            boundary = max(0.0, threshold - 5.0) if was_active or previous is ConditionState.ACTIVE else threshold
            state = condition.update(TemporalSample(timestamp, valid and magnitude >= boundary, quality, {metric_name: magnitude}))
            fast_state = ConditionState.INACTIVE
            fast_closed = False
            if metric_name in {"yaw", "pitch"}:
                fast_duration = self.fast_duration_ms if metric_name == "yaw" else self.fast_down_duration_ms
                fast_threshold = self.fast_yaw_threshold if metric_name == "yaw" else self.fast_pitch_threshold
                fast = self._fast_conditions.setdefault(
                    key, TemporalCondition(min_duration_ms=fast_duration,
                                           window_ms=self.frequency_window_ms, max_sample_gap_ms=600),
                )
                fast_state = fast.update(TemporalSample(
                    timestamp, valid and magnitude >= fast_threshold, quality,
                ))
                fast_closed = fast.last_closed
            active = (state is ConditionState.ACTIVE or fast_state is ConditionState.ACTIVE
                      or (was_active and valid and magnitude >= boundary))
            self._combined_active[key] = active
            if (active and not was_active
                    and metric_name == "yaw" and self._turn_armed.get(key_prefix, True)):
                turns.append(timestamp)
                self._turn_armed[key_prefix] = False
            if (state in {ConditionState.CANDIDATE, ConditionState.ACTIVE}
                    or fast_state in {ConditionState.CANDIDATE, ConditionState.ACTIVE}
                    or condition.last_closed or fast_closed or was_active):
                results.append({"session_id": session_id, "track_id": track_id, "event_type": event_type,
                                "severity": Severity.MEDIUM if active else Severity.LOW,
                                "confidence": quality, "active": active,
                                "metrics": {metric_name: magnitude,
                                            "fast_confirmation": fast_state is ConditionState.ACTIVE,
                                            **condition.current_metrics()},
                                "reason_codes": [f"{event_type.value}_THRESHOLD"]})
        event_type = EventType.ABNORMAL_HEAD_MOVEMENT
        condition = self._conditions.setdefault((key_prefix, event_type), TemporalCondition(min_duration_ms=0))
        state = condition.update(TemporalSample(timestamp, valid and len(turns) >= self.minimum_turn_count, quality))
        if state is not ConditionState.INACTIVE or condition.last_closed:
            results.append({"session_id": session_id, "track_id": track_id, "event_type": event_type,
                            "severity": Severity.MEDIUM, "confidence": quality,
                            "active": state is ConditionState.ACTIVE,
                            "metrics": {"turn_count": len(turns), "window_ms": self.frequency_window_ms},
                            "reason_codes": ["REPEATED_SUSTAINED_HEAD_TURNS"]})
        return results
