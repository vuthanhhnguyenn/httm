"""Rule output: the typed per-frame state of one behaviour rule."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..types import EventType, Severity


class SignalState(StrEnum):
    """ACTIVE opens/updates an event; CANDIDATE (building up) and INACTIVE (ended) close an open one."""

    ACTIVE = "active"
    CANDIDATE = "candidate"
    INACTIVE = "inactive"


MetricValue = float | int | str | bool | None


@dataclass(frozen=True, slots=True)
class RuleSignal:
    event_type: EventType
    state: SignalState
    session_id: Any
    track_id: str | None
    confidence: float
    severity: Severity
    reason_codes: tuple[str, ...]
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    rule_key: str | None = None
    # Provenance of object alerts: frame, box, confidence and model of the opening detection.
    trace: Mapping[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self.state is SignalState.ACTIVE

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RuleSignal:
        """Adapter for evaluators that still build their result as a mapping."""

        track = payload.get("track_id")
        return cls(
            event_type=EventType(payload["event_type"]),
            state=_state(payload),
            session_id=payload.get("session_id"),
            track_id=str(track) if track else None,
            confidence=float(payload.get("confidence", 0.0)),
            severity=Severity(payload.get("severity", Severity.MEDIUM)),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            metrics=dict(payload.get("metrics") or {}),
            rule_key=payload.get("rule_key"),
            trace=dict(payload["object_trace"]) if isinstance(payload.get("object_trace"), Mapping) else None,
        )

    def as_mapping(self) -> dict[str, Any]:
        """Shape accepted by :class:`EventAggregator` and older adapters."""

        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "track_id": self.track_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "active": self.active,
            "metrics": dict(self.metrics),
            "reason_codes": list(self.reason_codes),
        }
        if self.rule_key is not None:
            payload["rule_key"] = self.rule_key
        if self.trace is not None:
            payload["metrics"]["object_trace"] = dict(self.trace)
        return payload


def _state(payload: Mapping[str, Any]) -> SignalState:
    if payload.get("active", True):
        return SignalState.ACTIVE
    return SignalState.CANDIDATE if payload.get("state") == "candidate" else SignalState.INACTIVE


__all__ = ["MetricValue", "RuleSignal", "SignalState"]
