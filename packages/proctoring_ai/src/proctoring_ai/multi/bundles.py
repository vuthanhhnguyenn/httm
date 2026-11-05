"""Split one analysed frame into one single-subject ObservationBundle per seat.

The phase-1 rules assume exactly one candidate per bundle; giving each seat its
own bundle (its person, face, hands/pose and nearby objects) lets every rule run
unchanged per seat.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from proctoring_core.events.models import ObservationBundle, ObservationHealth
from proctoring_core.observations import Box

from .face_pool import FaceResult
from .focus_analyzers import FocusResult
from .tracker import Seat


def _intersection(a: Box, b: Box) -> float:
    width = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
    height = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    return width * height if width > 0 and height > 0 else 0.0


def assign_objects(objects: Sequence[Mapping[str, Any]], seats: Sequence[Seat], *,
                   expand: float = 0.2) -> dict[UUID, list[dict[str, Any]]]:
    assigned: dict[UUID, list[dict[str, Any]]] = {seat.seat_id: [] for seat in seats}
    regions = [(seat.seat_id, Box(seat.bbox.x - expand * seat.bbox.width, seat.bbox.y - expand * seat.bbox.height,
                                  seat.bbox.width * (1 + 2 * expand), seat.bbox.height * (1 + 2 * expand)))
               for seat in seats]
    for item in objects:
        box = Box.parse(item.get("bbox"))
        if box is None or str(item.get("class", "")).casefold() == "person":
            continue
        best = max(((_intersection(box, region), seat_id) for seat_id, region in regions),
                   key=lambda pair: pair[0], default=(0.0, None))
        if best[1] is not None and best[0] > 0:
            assigned[best[1]].append(dict(item))
    return assigned


def _face_health(face: FaceResult | None, visible: bool) -> ObservationHealth:
    if not visible:
        return ObservationHealth("not_run", reason="SUBJECT_NOT_VISIBLE")
    if face is None:
        return ObservationHealth("not_run", reason="NOT_SCHEDULED")
    if face.status == "mismatch":
        return ObservationHealth("no_detection", face.latency_ms, "FACE_OUTSIDE_SUBJECT")
    return ObservationHealth(face.status, face.latency_ms)


def build_subject_bundle(*, frame: Mapping[str, Any], session_id: UUID, seat: Seat, visible: bool,
                         face: FaceResult | None, focus: FocusResult | None, objects: list[dict[str, Any]],
                         image_quality: dict[str, Any], detector_health: ObservationHealth) -> ObservationBundle:
    people = [{"class": "person", "bbox": seat.bbox.as_dict(), "confidence": seat.confidence,
               "track_id": str(seat.seat_id), "runtime_track_id": seat.runtime_track_id}] if visible else []
    faces = [face.face] if visible and face is not None and face.status == "ready" and face.face else []
    not_in_focus = ObservationHealth("not_run", reason="NOT_IN_FOCUS")
    health = {
        "object_detector": detector_health,
        "face": _face_health(face, visible),
        "hands": ObservationHealth(focus.hands_status, focus.latency_ms) if focus else not_in_focus,
        "pose": ObservationHealth(focus.pose_status, focus.latency_ms) if focus else not_in_focus,
    }
    return ObservationBundle(
        frame_id=int(frame.get("frame_id", 0)),
        session_id=session_id,
        captured_at_utc=frame.get("captured_at_utc", datetime.now(UTC)),
        captured_at_monotonic_ms=int(frame.get("captured_at_monotonic_ms", 0)),
        frame_size=tuple(frame.get("frame_size", (1, 1))),
        people=people,
        objects=list(objects),
        faces=faces,
        poses=list(focus.poses) if focus and visible else [],
        hands=list(focus.hands) if focus and visible else [],
        image_quality=dict(image_quality),
        analyzer_health=health,
    )


__all__ = ["assign_objects", "build_subject_bundle"]
