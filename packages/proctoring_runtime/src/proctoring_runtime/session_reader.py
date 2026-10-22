"""Read a session's ``events.jsonl`` (v1 or v2) into one view per event episode.

Lines are append-only: an episode has ``opened``/``updated``/``closed`` lines, evidence
results and review labels arrive as their own records. Fields missing from older
sessions stay ``None`` and are shown as "không có dữ liệu", never guessed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from proctoring_core.events.review_models import ReviewType

MISSING = "không có dữ liệu"


@dataclass(slots=True)
class EventView:
    id: str
    type: str
    label: str
    severity: str
    opened_at: str | None
    closed_at: str | None = None
    confidence: float | None = None
    explanation: str = ""
    message: str = ""
    exam_mode: str | None = None
    object_trace: dict[str, Any] | None = None
    supporting_signals: list[str] = field(default_factory=list)
    evidence_path: str | None = None
    evidence_status: str | None = None
    review: ReviewType | None = None
    review_note: str = ""
    experimental: bool = False
    duration_ms: int | None = None
    schema_version: str = "1.0"

    @property
    def reviewed(self) -> bool:
        return self.review is not None


def review_record(event_id: str, verdict: ReviewType, *, note: str = "", reviewed_at: str,
                  episode_id: str | None = None, reviewer: str = "local-operator") -> dict[str, Any]:
    if len(note) > 500:
        raise ValueError("review note must be at most 500 characters")
    return {"schema_version": "2.0", "action": "review", "event_id": event_id, "episode_id": episode_id or event_id,
            "verdict": ReviewType(verdict).value.lower(), "note": note, "reviewed_at": reviewed_at,
            "reviewer": reviewer}


def build_views(records: Iterable[Mapping[str, Any]]) -> list[EventView]:
    views: dict[str, EventView] = {}
    order: list[str] = []
    for record in records:
        action = record.get("action")
        if action in {"opened", "correlation"}:
            event_id = str(record.get("id", ""))
            metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
            trace = record.get("object_trace") or (metrics or {}).get("object_trace")
            views[event_id] = EventView(
                id=event_id,
                type=str(record.get("type", "")),
                label=str(record.get("label_vi", record.get("type", ""))),
                severity=str(record.get("severity", "")),
                opened_at=record.get("occurred_at"),
                confidence=float(record["confidence"]) if record.get("confidence") is not None else None,
                explanation=str(record.get("explanation", "")),
                message=str(record.get("message_vi", "")),
                exam_mode=record.get("exam_mode"),
                object_trace=dict(trace) if isinstance(trace, Mapping) else None,
                supporting_signals=list(record.get("supporting_signals") or []),
                evidence_status=record.get("evidence_status"),
                experimental=bool((metrics or {}).get("experimental", False)),
                schema_version=str(record.get("schema_version", "1.0")),
            )
            order.append(event_id)
        elif action in {"updated", "closed"} and str(record.get("id", "")) in views:
            view = views[str(record["id"])]
            raw_metrics = record.get("metrics")
            updated: Mapping[str, Any] = raw_metrics if isinstance(raw_metrics, Mapping) else {}
            if isinstance(updated.get("duration_ms"), int | float):
                view.duration_ms = int(updated["duration_ms"])
            if record.get("supporting_signals"):
                view.supporting_signals = list(record["supporting_signals"])
            if action == "closed":
                view.closed_at = record.get("occurred_at")
        elif action == "evidence" and str(record.get("event_id", "")) in views:
            view = views[str(record["event_id"])]
            view.evidence_status = str(record.get("status", ""))
            if record.get("status") == "available":
                view.evidence_path = record.get("relative_path")
        elif action == "review" and str(record.get("event_id", "")) in views:
            view = views[str(record["event_id"])]
            try:
                view.review = ReviewType(str(record.get("verdict", "")).upper())
            except ValueError:
                continue
            view.review_note = str(record.get("note", ""))
    return [views[event_id] for event_id in order]


def latest_reviews(records: Iterable[Mapping[str, Any]]) -> dict[str, ReviewType]:
    return {view.id: view.review for view in build_views(records) if view.review is not None}


__all__ = ["MISSING", "EventView", "build_views", "latest_reviews", "review_record"]
