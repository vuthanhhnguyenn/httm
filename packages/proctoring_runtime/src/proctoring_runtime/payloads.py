"""Map observations, events and metrics to the dictionaries sent to the UI and JSONL."""

from __future__ import annotations

from typing import Any

from proctoring_ai.pipeline.frame_queue import LatestFrameQueue
from proctoring_ai.pipeline.metrics import PipelineMetrics
from proctoring_core.events.aggregator import EventTransition
from proctoring_core.events.correlation import CorrelationFinding
from proctoring_core.events.models import ObservationBundle
from proctoring_core.risk.models import RiskSnapshot

from .messages import event_description, event_name, level_name

HEALTHY_STATUSES = frozenset({"ready", "no_detection", "skipped"})
EVENT_SCHEMA_VERSION = "2.0"


def analysis_payload(bundle: ObservationBundle, health: dict[str, str]) -> dict[str, Any]:
    return {
        "frame_id": bundle.frame_id,
        "captured_at_monotonic_ms": bundle.captured_at_monotonic_ms,
        "objects": [
            {
                "class": str(item.get("class", "unknown")),
                "confidence": float(item.get("confidence", 0)),
                "bbox": item.get("bbox", {}),
                "track_confirmed": item.get("track_confirmed", True),
            }
            for item in bundle.objects
            if isinstance(item.get("bbox"), dict)
        ],
        "faces": [
            {key: face[key] for key in ("bbox", "yaw", "pitch", "roll", "quality", "pose_valid", "calibration_status")
             if key in face}
            for face in bundle.faces
        ],
        "poses": [
            {key: pose[key] for key in ("bbox", "landmarks", "quality") if key in pose}
            for pose in bundle.poses
        ],
        "hands": [
            {key: hand[key] for key in ("bbox", "landmarks", "handedness", "confidence") if key in hand}
            for hand in bundle.hands
        ],
        "analyzer_health": health,
    }


def metrics_snapshot(metrics: PipelineMetrics, queue: LatestFrameQueue,
                     analyzer_health: dict[str, str] | None) -> dict[str, Any]:
    snapshot: dict[str, Any] = dict(metrics.snapshot())
    snapshot["dropped_frames"] = queue.dropped_frames
    snapshot["queue_depth"] = len(queue)
    snapshot["analyzer_health"] = dict(analyzer_health or {})
    if analyzer_health and any(value not in HEALTHY_STATUSES for value in analyzer_health.values()):
        snapshot["realtime_status"] = "DEGRADED"
    return snapshot


def _with_labels(item: dict[str, Any]) -> dict[str, Any]:
    item["label_vi"] = event_name(str(item["type"]))
    item["severity_vi"] = level_name(str(item["severity"]))
    item["message_vi"] = event_description(item)
    return item


def _risk_fields(snapshot: RiskSnapshot) -> dict[str, Any]:
    return {
        "risk_score": snapshot.score_after,
        "risk_level": snapshot.level_after.value,
        "cumulative_score": snapshot.cumulative_score,
        "peak_score": snapshot.peak_score,
        "event_count": snapshot.event_count,
        "risk_reason_codes": list(snapshot.reason_codes),
    }


def event_item(transition: EventTransition, snapshot: RiskSnapshot, evidence_status: str | None) -> dict[str, Any]:
    event = transition.event
    metrics = event.metrics if isinstance(event.metrics, dict) else {}
    supporting = str(metrics.get("supporting_signals") or "")
    item: dict[str, Any] = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "id": str(event.id),
        # The aggregator keeps one event object per episode, so its id is the episode id.
        "episode_id": str(event.id),
        "action": transition.action,
        "type": event.event_type.value,
        "severity": event.severity.value,
        "confidence": event.confidence,
        "explanation": event.explanation,
        "metrics": event.metrics,
        "occurred_at": event.started_at.isoformat(),
        **_risk_fields(snapshot),
        "evidence_status": evidence_status if transition.action == "opened" else None,
        "supporting_signals": [name for name in supporting.split(",") if name],
    }
    if isinstance(metrics.get("object_trace"), dict):
        item["object_trace"] = metrics["object_trace"]
    return _with_labels(item)


def correlation_item(finding: CorrelationFinding, snapshot: RiskSnapshot) -> dict[str, Any]:
    item: dict[str, Any] = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "id": str(finding.id),
        "action": "correlation",
        "type": f"CORRELATION_{finding.rule_code}",
        "source_event_ids": [str(event_id) for event_id in finding.source_event_ids],
        "confidence": finding.confidence,
        "severity": "LOW",
        "occurred_at": finding.occurred_at.isoformat(),
        **_risk_fields(snapshot),
        "correlation_points": finding.points,
        "track_id": str(finding.track_id) if finding.track_id else None,
    }
    return _with_labels(item)


__all__ = ["analysis_payload", "correlation_item", "event_item", "metrics_snapshot"]
