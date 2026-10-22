from __future__ import annotations

from typing import Any
from uuid import uuid4

from proctoring_core.behavior.objects import ObjectRuleEvaluator
from proctoring_core.types import EventType


def phone(confidence: float = 0.7, width: float = 0.08, height: float = 0.12) -> dict[str, Any]:
    return {"class": "cell phone", "confidence": confidence,
            "bbox": {"x": 0.5, "y": 0.6, "width": width, "height": height},
            "object_track_id": 1, "track_confirmed": True,
            "model_name": "yolo11n", "model_version": "8.3.0"}


def run(objects: list[dict[str, Any]], **kwargs: Any) -> bool:
    rules = ObjectRuleEvaluator(phone_min_duration_ms=0, **kwargs)
    results = rules.evaluate(session_id=uuid4(), track_id=None, objects=objects,
                             monotonic_ms=1000, frame_id=1)
    return any(item["event_type"] is EventType.PHONE_DETECTED and item["active"] for item in results)


def test_geometry_filter_disabled_by_default_keeps_large_boxes() -> None:
    assert run([phone(width=0.6, height=0.9)])


def test_geometry_filter_rejects_boxes_too_large_for_a_hand() -> None:
    filters = {"phone_geometry_max_height_ratio": 0.4, "phone_geometry_max_area_ratio": 0.1}
    assert not run([phone(width=0.6, height=0.9)], **filters)       # height above limit
    assert not run([phone(width=0.6, height=0.2)], **filters)       # area above limit
    assert run([phone(width=0.3, height=0.3)], **filters)           # within both limits


def test_geometry_filter_does_not_touch_documents() -> None:
    rules = ObjectRuleEvaluator(document_min_duration_ms=0,
                                phone_geometry_max_height_ratio=0.05, phone_geometry_max_area_ratio=0.01)
    document = [{"class": "book", "confidence": 0.9, "bbox": {"x": 0.2, "y": 0.2, "width": 0.6, "height": 0.5},
                 "model_name": "yolo11n", "model_version": "8.3.0"}]
    results = rules.evaluate(session_id=uuid4(), track_id=None, objects=document,
                             monotonic_ms=1000, frame_id=1)
    assert any(item["event_type"] is EventType.DOCUMENT_DETECTED and item["active"] for item in results)
