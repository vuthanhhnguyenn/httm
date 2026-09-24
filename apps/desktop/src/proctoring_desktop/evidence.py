"""Bounded background JPEG evidence writer for analyzed camera frames."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import cv2
import numpy as np
from proctoring_core.events.models import ObservationBundle
from proctoring_core.events.review_models import EvidenceArtifact, EvidenceKind, EvidenceStatus


class EvidenceRecorder:
    def __init__(self, storage_root: Path, session_id: UUID, *, enabled: bool,
                 retention_days: int, on_result: Callable[[dict[str, Any]], None],
                 queue_size: int = 4) -> None:
        self.storage_root = storage_root.resolve()
        self.session_id = session_id
        self.enabled = enabled
        self.retention_days = max(1, retention_days)
        self.on_result = on_result
        self.session_dir = self.storage_root / "sessions" / str(session_id)
        self.evidence_dir = self.session_dir / "evidence"
        self._queue: asyncio.Queue[tuple[UUID, np.ndarray, datetime] | None] = asyncio.Queue(maxsize=queue_size)
        self._worker_task: asyncio.Task[None] | None = None
        self._callback_error: Exception | None = None

    def start(self) -> None:
        if self.enabled and self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker(), name="evidence-writer")

    def enqueue(self, event_id: UUID, frame: dict[str, Any], bundle: ObservationBundle) -> str:
        if not self.enabled:
            return "disabled"
        if (frame.get("frame_id") != bundle.frame_id
                or frame.get("captured_at_monotonic_ms") != bundle.captured_at_monotonic_ms):
            return "frame_mismatch"
        image = frame.get("image")
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            return "frame_unavailable"
        if self._queue.full():
            return "queue_full"
        # Freeze precisely the frame analyzed by the rule engine. No camera reads here.
        self._queue.put_nowait((event_id, image.copy(), bundle.captured_at_utc))
        return "pending"

    async def close(self) -> None:
        if self._worker_task is None:
            return
        await self._queue.join()
        await self._queue.put(None)
        await self._worker_task
        self._worker_task = None
        if self._callback_error is not None:
            raise RuntimeError("Không ghi được liên kết ảnh bằng chứng vào nhật ký") from self._callback_error

    async def _worker(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                event_id, image, captured_at = item
                try:
                    result = await asyncio.to_thread(self._write_frame, event_id, image, captured_at)
                except Exception as exc:
                    result = {"action": "evidence", "event_id": str(event_id), "status": "failed",
                              "reason": type(exc).__name__}
                try:
                    self.on_result(result)
                except Exception as exc:
                    # Drain all queued frames, then report the linking failure on close().
                    self._callback_error = exc
            finally:
                self._queue.task_done()

    def _write_frame(self, event_id: UUID, image: np.ndarray, captured_at: datetime) -> dict[str, Any]:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        target = self.evidence_dir / f"{event_id}.jpg"
        if target.exists():
            raise FileExistsError(target)
        encoded_ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not encoded_ok:
            raise OSError("JPEG_ENCODING_FAILED")
        payload = encoded.tobytes()
        digest = hashlib.sha256(payload).hexdigest()
        temp = self.evidence_dir / f"{event_id}.{uuid4().hex}.tmp"
        created = False
        try:
            with temp.open("xb") as stream:
                stream.write(payload)
            os.replace(temp, target)
            created = True
            retention_until = captured_at + timedelta(days=self.retention_days)
            artifact = EvidenceArtifact(
                id=uuid4(), event_id=event_id, session_id=self.session_id,
                kind=EvidenceKind.FRAME, status=EvidenceStatus.AVAILABLE,
                media_type="image/jpeg", size_bytes=len(payload), sha256=digest,
                relative_path=target.relative_to(self.storage_root).as_posix(),
                captured_from=captured_at, captured_to=captured_at,
                retention_until=retention_until,
            )
            result = {"action": "evidence", "event_id": str(event_id), "status": "available",
                      "artifact_id": str(artifact.id), "relative_path": artifact.relative_path,
                      "sha256": artifact.sha256, "size_bytes": artifact.size_bytes,
                      "captured_at": captured_at.isoformat(), "retention_until": retention_until.isoformat()}
            with (self.evidence_dir / "index.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            return result
        except Exception:
            if created:
                target.unlink(missing_ok=True)
            raise
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def purge_expired(storage_root: Path, *, now: datetime | None = None) -> int:
        """Delete only indexed JPEGs whose own retention deadline has passed."""
        root = storage_root.resolve()
        current = now or datetime.now(UTC)
        removed = 0
        for index in (root / "sessions").glob("*/evidence/index.jsonl"):
            evidence_dir = index.parent.resolve()
            if not evidence_dir.is_relative_to(root / "sessions") or not index.is_file():
                continue
            latest: dict[str, dict[str, Any]] = {}
            try:
                with index.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            record = json.loads(line)
                            latest[str(record["event_id"])] = record
                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue
            except OSError:
                continue
            for record in latest.values():
                if record.get("status") != "available":
                    continue
                try:
                    deadline = datetime.fromisoformat(record["retention_until"])
                    candidate = root / record["relative_path"]
                    if candidate.is_symlink():
                        continue
                    target = candidate.resolve()
                    expected_name = f"{UUID(str(record['event_id']))}.jpg"
                except (KeyError, TypeError, ValueError, OSError):
                    continue
                if (deadline.tzinfo is None or deadline > current
                        or target.parent != evidence_dir or target.name != expected_name):
                    continue
                try:
                    target.unlink(missing_ok=True)
                    with index.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"action": "evidence", "event_id": record["event_id"],
                                                 "status": "deleted", "deleted_at": current.isoformat()}) + "\n")
                    removed += 1
                except OSError:
                    continue
        return removed
