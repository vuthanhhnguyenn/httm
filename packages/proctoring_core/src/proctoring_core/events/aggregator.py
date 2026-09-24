"""Merge frame-level rule signals into one open event per stable rule key."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import UUID

from ..types import EventType, Severity, ensure_utc
from .models import SuspiciousEvent


@dataclass(frozen=True, slots=True)
class EventTransition:
    action: str
    event: SuspiciousEvent


class EventAggregator:
    def __init__(self, *, policy_version: str = "default", config_version: str = "default") -> None:
        self.policy_version = policy_version
        self.config_version = config_version
        self._open: dict[tuple[UUID, UUID | None, EventType, str], SuspiciousEvent] = {}

    @staticmethod
    def _explanation(event_type: EventType, metrics: Mapping[str, Any], reason_codes: list[str]) -> str:
        details = ", ".join(f"{key}={value}" for key, value in metrics.items() if key not in {"samples"})
        suffix = f" ({details})" if details else ""
        return f"{event_type.value} triggered by {', '.join(reason_codes)}{suffix}."

    def process(self, signal: Mapping[str, Any], *, occurred_at: datetime | None = None, risk_score_after: float = 0.0) -> EventTransition | None:
        event_type = EventType(signal["event_type"])
        session_id = UUID(str(signal["session_id"]))
        raw_track = signal.get("track_id")
        track_id = UUID(str(raw_track)) if raw_track else None
        rule_key = str(signal.get("rule_key", event_type.value))
        key = (session_id, track_id, event_type, rule_key)
        now = ensure_utc(occurred_at or datetime.now(UTC))
        active = bool(signal.get("active", True))
        if active:
            confidence = max(0.0, min(1.0, float(signal.get("confidence", 0.0))))
            severity = Severity(signal.get("severity", Severity.MEDIUM))
            reason_codes = list(signal.get("reason_codes") or [f"{event_type.value}_RULE"])
            metrics = dict(signal.get("metrics") or {})
            current = self._open.get(key)
            if current is None:
                current = SuspiciousEvent(
                    session_id=session_id,
                    track_id=track_id,
                    event_type=event_type,
                    severity=severity,
                    confidence=confidence,
                    started_at=now,
                    risk_delta=float(signal.get("risk_delta", 0.0)),
                    risk_score_after=risk_score_after,
                    reason_codes=reason_codes,
                    explanation=self._explanation(event_type, metrics, reason_codes),
                    metrics=metrics,
                    policy_version=self.policy_version,
                    config_version=self.config_version,
                    rule_key=rule_key,
                )
                self._open[key] = current
                return EventTransition("opened", current)
            current.update(confidence=confidence, risk_score_after=risk_score_after, metrics=metrics)
            current.risk_delta = float(signal.get("risk_delta", current.risk_delta))
            current.explanation = self._explanation(event_type, current.metrics, current.reason_codes)
            return EventTransition("updated", current)
        current = self._open.pop(key, None)
        if current is None:
            return None
        current.close(now)
        return EventTransition("closed", current)

    def flush(self, *, occurred_at: datetime | None = None) -> list[EventTransition]:
        now = ensure_utc(occurred_at or datetime.now(UTC))
        transitions: list[EventTransition] = []
        for key in tuple(self._open):
            current = self._open.pop(key)
            current.close(now)
            transitions.append(EventTransition("closed", current))
        return transitions

    def open_events(self) -> tuple[SuspiciousEvent, ...]:
        return tuple(self._open.values())

    aggregate = process
    ingest = process
