"""Local session storage: append-only events JSONL and the atomic risk summary."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID

from proctoring_core.config.models import canonical_sha256
from proctoring_core.risk.engine import RiskEngine
from proctoring_core.types import ExamMode


def session_dir(storage_root: Path, session_id: UUID) -> Path:
    return storage_root / "sessions" / str(session_id)


class JsonlEventRepository:
    """Append-only ``events.jsonl``; implements the ``EventRepository`` port.

    Lines are never rewritten. Review labels and evidence results are appended as
    their own records that reference the event id.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, item: Mapping[str, Any]) -> None:
        line = json.dumps(dict(item), ensure_ascii=False) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    async def save(self, event: Mapping[str, Any]) -> Mapping[str, Any]:
        self.append(event)
        return event

    async def get(self, event_id: UUID) -> Mapping[str, Any] | None:
        wanted = str(event_id)
        return next((item for item in reversed(self.records()) if item.get("id") == wanted), None)

    async def list_for_session(self, session_id: UUID, *, limit: int = 50) -> Sequence[Mapping[str, Any]]:
        return self.records()[-limit:]


SESSION_SCHEMA_VERSION = "2.0"


def write_session_manifest(target_dir: Path, session: Any, exam_mode: ExamMode, settings: Any,
                          model_versions: dict[str, dict[str, str]] | None = None,
                          calibration_profile_id: str | None = None) -> Path:
    """``session.json``: what a reviewer needs to interpret this session's events."""

    item = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "session_id": str(session.id),
        "exam_mode": ExamMode(exam_mode).value,
        "started_at": datetime.now().astimezone().isoformat(),
        "config_sha256": canonical_sha256(settings.raw_config),
        "policy_sha256": canonical_sha256(settings.raw_policy),
        "model_versions": model_versions or {},
        "calibration_profile_id": calibration_profile_id,
    }
    target = target_dir / "session.json"
    pending = target.with_suffix(".json.tmp")
    pending.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(pending, target)
    return target


def write_risk_summary(target_dir: Path, engine: RiskEngine, session_id: UUID) -> Path:
    score = engine.score(session_id)
    item = {"session_id": str(session_id), "current_score": score,
            "current_level": engine.level_for(score).value,
            **engine.history(session_id), "updated_at": datetime.now().astimezone().isoformat()}
    target = target_dir / "risk-summary.json"
    pending = target.with_suffix(".json.tmp")
    pending.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(pending, target)
    return target


__all__ = ["JsonlEventRepository", "session_dir", "write_risk_summary", "write_session_manifest"]
