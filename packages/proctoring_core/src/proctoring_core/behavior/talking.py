"""Experimental "possible talking": rhythmic mouth opening over a sliding window.

Speech opens and closes the mouth repeatedly; a yawn or a held-open mouth does not.
At the analysis rate (≈ 8 face passes per second) only coarse rhythm is visible, so
this counts mean crossings rather than recognising syllables, and stays a supporting
signal with a low risk cap.
"""

from __future__ import annotations

from collections import deque
from statistics import fmean, pstdev
from typing import Any

from ..config.schema import TalkingConfig
from ..observations import FaceObs
from ..types import EventType, Severity
from .temporal_buffer import ConditionState, TemporalCondition, TemporalSample


def mouth_oscillations(values: list[float], *, hysteresis: float) -> int:
    """Number of full open/close cycles: alternating crossings of mean ± hysteresis, halved."""

    if len(values) < 3:
        return 0
    centre = fmean(values)
    side = 0
    crossings = 0
    for value in values:
        current = 1 if value > centre + hysteresis else -1 if value < centre - hysteresis else 0
        if current and side and current != side:
            crossings += 1
        if current:
            side = current
    return crossings // 2


class TalkingRuleEvaluator:
    def __init__(self, config: TalkingConfig, *, min_face_quality: float, max_sample_gap_ms: int) -> None:
        self.config = config
        self.min_face_quality = min_face_quality
        self.max_sample_gap_ms = max_sample_gap_ms
        self._window: deque[tuple[int, float]] = deque()
        self._condition = TemporalCondition(min_duration_ms=round(config.min_duration_seconds * 1000),
                                            max_sample_gap_ms=max_sample_gap_ms)

    def _usable(self, face: FaceObs | None) -> float | None:
        if (face is None or face.mouth is None or not face.pose_valid or face.quality < self.min_face_quality
                or abs(face.yaw) > self.config.max_head_yaw_degrees):
            return None
        return face.mouth.mar

    def evaluate(self, *, session_id: Any, track_id: Any, face: FaceObs | None,
                 monotonic_ms: int) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return []
        mar = self._usable(face)
        window_ms = round(self.config.window_seconds * 1000)
        if mar is None or (self._window and monotonic_ms - self._window[-1][0] > self.max_sample_gap_ms):
            self._window.clear()
        if mar is not None:
            self._window.append((monotonic_ms, mar))
        while self._window and monotonic_ms - self._window[0][0] > window_ms:
            self._window.popleft()
        values = [value for _, value in self._window]
        talking, metrics = False, {}
        if len(values) >= self.config.min_samples:
            amplitude = max(values) - min(values)
            spread = pstdev(values)
            cycles = mouth_oscillations(values, hysteresis=amplitude * 0.2)
            mean = fmean(values)
            # A yawn holds the mouth wide open: large mean with few cycles.
            yawning = mean >= self.config.yawn_mar and cycles < self.config.min_oscillations + 1
            talking = (cycles >= self.config.min_oscillations and amplitude >= self.config.min_amplitude
                       and spread >= self.config.min_std and not yawning)
            metrics = {"mouth_cycles": cycles, "mouth_amplitude": round(amplitude, 3),
                       "mouth_std": round(spread, 3), "mouth_mean": round(mean, 3)}
        confidence = max(0.0, min(1.0, face.quality)) if face is not None and mar is not None else 0.0
        state = self._condition.update(TemporalSample(monotonic_ms, talking, confidence))
        if state is ConditionState.INACTIVE and not self._condition.last_closed:
            return []
        return [{
            "session_id": session_id, "track_id": track_id, "event_type": EventType.POSSIBLE_TALKING,
            "severity": Severity.LOW, "confidence": confidence, "active": state is ConditionState.ACTIVE,
            "state": state.value,
            "metrics": {**metrics, "risk_cap": self.config.risk_cap, "experimental": True,
                        **self._condition.current_metrics()},
            "reason_codes": ["POSSIBLE_TALKING_MOUTH_RHYTHM"],
        }]


__all__ = ["TalkingRuleEvaluator", "mouth_oscillations"]
