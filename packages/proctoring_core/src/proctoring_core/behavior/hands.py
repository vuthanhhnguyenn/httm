"""Hand activity worth a look: hand at the ear, over the mouth, or waving above the chin.

Distances are measured from hand landmarks to face keypoints and scaled by face width,
so they do not depend on how far the candidate sits from the camera. Writing, typing
and resting the chin on a hand are normal and do not match. Supporting signal only.
"""

from __future__ import annotations

from math import dist
from typing import Any

from ..config.schema import HandActivityConfig
from ..observations import FaceObs, HandObs
from ..types import EventType, Severity
from .temporal_buffer import ConditionState, TemporalCondition, TemporalSample

NEAR_EAR, NEAR_MOUTH, RAPID = "HAND_NEAR_EAR", "HAND_NEAR_MOUTH", "HAND_RAPID_MOVEMENT"


def hand_reasons(hand: HandObs, face: FaceObs | None, config: HandActivityConfig) -> set[str]:
    if face is None or not face.face_width or face.face_width <= 0:
        return set()
    keypoints = face.keypoints
    chin = keypoints.get("chin")
    mouth = keypoints.get("mouth_center")
    points = [point for point in (hand.wrist, hand.index_tip) if point is not None]
    if not points or chin is None:
        return set()
    width = face.face_width
    reasons: set[str] = set()
    ears = [keypoints[name] for name in ("ear_left", "ear_right") if name in keypoints]
    if any(dist(point, ear) <= config.near_ear_radius * width for point in points for ear in ears):
        reasons.add(NEAR_EAR)
    tip = hand.index_tip
    if (mouth is not None and tip is not None and dist(tip, mouth) <= config.near_mouth_radius * width
            # Chin rest: fingertips at or below the chin line, not over the lips.
            and tip[1] < chin[1] - config.chin_rest_margin * width):
        reasons.add(NEAR_MOUTH)
    if (hand.wrist is not None and hand.wrist[1] < chin[1] and hand.dt_ms > 0
            and hand.displacement / width / (hand.dt_ms / 1000) >= config.rapid_speed):
        # Writing and typing happen below the chin; fast movement up here is signalling or reaching.
        reasons.add(RAPID)
    return reasons


class HandRuleEvaluator:
    def __init__(self, config: HandActivityConfig, *, max_sample_gap_ms: int) -> None:
        self.config = config
        self._condition = TemporalCondition(min_duration_ms=round(config.minimum_duration_seconds * 1000),
                                            max_sample_gap_ms=max_sample_gap_ms)
        self._episode_reasons: set[str] = set()

    def evaluate(self, *, session_id: Any, track_id: Any, hands: list[HandObs], face: FaceObs | None,
                 monotonic_ms: int) -> list[dict[str, Any]]:
        reasons: set[str] = set()
        for hand in hands:
            reasons |= hand_reasons(hand, face, self.config)
        confidence = max((hand.confidence for hand in hands), default=0.0)
        state = self._condition.update(TemporalSample(monotonic_ms, bool(reasons), max(0.0, min(1.0, confidence))))
        if state is ConditionState.INACTIVE:
            self._episode_reasons.clear()
            if not self._condition.last_closed:
                return []
        self._episode_reasons |= reasons
        codes = sorted(self._episode_reasons) or ["HAND_ACTIVITY"]
        return [{
            "session_id": session_id, "track_id": track_id, "event_type": EventType.SUSPICIOUS_HAND_ACTIVITY,
            "severity": Severity.LOW, "confidence": confidence, "active": state is ConditionState.ACTIVE,
            "state": state.value,
            "metrics": {"hand_count": len(hands), "risk_cap": self.config.risk_cap,
                        **self._condition.current_metrics()},
            "reason_codes": codes,
        }]


__all__ = ["NEAR_EAR", "NEAR_MOUTH", "RAPID", "HandRuleEvaluator", "hand_reasons"]
