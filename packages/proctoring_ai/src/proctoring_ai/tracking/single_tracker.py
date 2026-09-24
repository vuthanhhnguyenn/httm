from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from proctoring_core.sessions.models import CandidateTrack
from proctoring_core.types import BoundingBox, TrackStatus


class SingleSubjectTracker:
    """Anonymous one-subject tracker; a recovered subject gets a new runtime track."""

    def __init__(self, *, max_missing_frames: int = 10) -> None:
        self.max_missing_frames = max_missing_frames
        self.track: CandidateTrack | None = None
        self._missing_frames = 0
        self._next_runtime_id = 1

    async def update(
        self,
        observations: list[dict[str, Any]],
        *,
        session_id: UUID,
        occurred_at: datetime | None = None,
    ) -> list[dict[str, Any]]:
        now = occurred_at or datetime.now(UTC)
        person = next((item for item in observations if item.get("class") == "person"), None)
        if person is None:
            self._missing_frames += 1
            if self.track is not None and self._missing_frames >= self.max_missing_frames:
                self.track.mark_lost(now)
            return []
        self._missing_frames = 0
        bbox = BoundingBox.from_mapping(person["bbox"])
        if self.track is None or self.track.status in {TrackStatus.LOST, TrackStatus.CLOSED}:
            self.track = CandidateTrack(
                uuid4(), session_id, self._next_runtime_id, current_bbox=bbox,
                visibility=person.get("confidence", 0.0), first_seen_at=now, last_seen_at=now,
            )
            self._next_runtime_id += 1
        else:
            self.track.update(bbox=bbox, visibility=person.get("confidence", 0.0), occurred_at=now)
        return [{**person, "runtime_track_id": self.track.runtime_track_id, "track_id": str(self.track.id)}]

    def close(self, occurred_at: datetime | None = None) -> None:
        if self.track is not None:
            self.track.close(occurred_at or datetime.now(UTC))
