"""Run a recorded video through the live analysis and rule path.

The replay uses video timestamps, a fixed analysis rate and generous analyzer
timeouts, so the same file and configuration always yield the same events.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from proctoring_core.config.schema import AppConfig
from proctoring_core.evaluation.matching import PredictedEvent
from proctoring_core.sessions.models import ExamSession
from proctoring_core.types import EventType, ExamMode, SessionMode

from .session_engine import DEFAULT_POLICY_ID, SessionRuleProcessor

REPLAY_ANALYZER_TIMEOUT_MS = 120_000


class FrameSource(Protocol):
    async def open(self) -> None: ...

    async def read(self) -> dict[str, Any] | None: ...

    async def close(self) -> None: ...


class RuleProcessor(Protocol):
    """One analysis step: observation bundle in, persisted/emitted event items out."""

    async def process(self, bundle: Any, frame: dict[str, Any]) -> list[dict[str, Any]]: ...


@dataclass(slots=True)
class ReplayResult:
    events: list[dict[str, Any]] = field(default_factory=list)
    frames_analyzed: int = 0
    last_offset_ms: int = 0
    elapsed_s: float = 0.0

    @property
    def ai_fps(self) -> float:
        return self.frames_analyzed / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def golden_lines(self) -> list[str]:
        """Canonical, id-free event stream for byte-for-byte regression checks."""

        lines = []
        for item in self.events:
            record = {
                "t": int(item["video_offset_ms"]),
                "type": item["type"],
                "action": item["action"],
                "severity": item.get("severity"),
                "risk": round(float(item.get("risk_score", 0.0)), 2),
            }
            lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
        return lines

    def predicted_events(self, end_ms: int | None = None) -> list[PredictedEvent]:
        """Pair ``opened`` with the matching ``closed`` of the same event id."""

        open_events: dict[str, PredictedEvent] = {}
        predicted: list[PredictedEvent] = []
        for item in self.events:
            action = item.get("action")
            event_type = item.get("type", "")
            if event_type not in EventType.__members__:
                continue  # correlation findings are not behaviour events
            offset = int(item["video_offset_ms"])
            if action == "opened":
                open_events[str(item["id"])] = PredictedEvent(EventType(event_type), offset, offset,
                                                              str(item.get("severity", "")))
            elif action == "closed" and str(item["id"]) in open_events:
                started = open_events.pop(str(item["id"]))
                predicted.append(PredictedEvent(started.event_type, started.start_ms,
                                                max(offset, started.start_ms), started.severity))
        final = end_ms if end_ms is not None else self.last_offset_ms
        predicted.extend(PredictedEvent(item.event_type, item.start_ms, max(final, item.start_ms), item.severity)
                         for item in open_events.values())
        return sorted(predicted, key=lambda item: (item.start_ms, item.event_type.value))


async def run_replay(
    source: FrameSource,
    pipeline: Any,
    processor: RuleProcessor,
    *,
    session_id: UUID | None = None,
    calibrate_at_ms: int | None = 0,
    on_progress: Callable[[int], None] | None = None,
) -> ReplayResult:
    """Analyse every sampled frame in order; nothing is dropped for speed."""

    session = session_id or uuid4()
    result = ReplayResult()
    last_utc = None
    calibration_requested = calibrate_at_ms is None
    started = time.perf_counter()
    await source.open()
    try:
        while (frame := await source.read()) is not None:
            offset = int(frame.get("video_offset_ms", frame["captured_at_monotonic_ms"]))
            if not calibration_requested and calibrate_at_ms is not None and offset >= calibrate_at_ms:
                face = getattr(pipeline, "face_analyzer", None)
                if face is not None and hasattr(face, "request_calibration"):
                    face.request_calibration()
                rules = getattr(processor, "rules", None)
                if rules is not None and hasattr(rules, "request_calibration"):
                    # Recorded video has no on-screen targets: learn the neutral eye position only.
                    rules.request_calibration(gaze_center=True)
                calibration_requested = True
            bundle = await pipeline.analyze(frame, session_id=session)
            for item in await processor.process(bundle, frame):
                result.events.append({**item, "video_offset_ms": offset})
            result.frames_analyzed += 1
            result.last_offset_ms = offset
            last_utc = bundle.captured_at_utc
            if on_progress is not None and result.frames_analyzed % 200 == 0:
                on_progress(offset)
    finally:
        await source.close()
    finish = getattr(processor, "finish", None)
    if finish is not None and last_utc is not None:
        result.events.extend({**item, "video_offset_ms": result.last_offset_ms} for item in finish(last_utc))
    result.elapsed_s = time.perf_counter() - started
    return result


def create_replay_processor(session_id: UUID, policy: Mapping[str, Any], raw_config: Mapping[str, Any],
                            exam_mode: ExamMode | str = ExamMode.COMPUTER) -> SessionRuleProcessor:
    """The live rule path without persistence, evidence or UI callbacks."""

    session = ExamSession.create(
        name="Phát lại",
        mode=SessionMode.SINGLE_PERSON,
        camera_source="replay",
        policy_id=DEFAULT_POLICY_ID,
        policy_snapshot={**policy, "exam_mode": ExamMode(exam_mode).value},
        config_version_id=uuid4(),
        config_snapshot=dict(raw_config),
        created_by="replay",
        session_id=session_id,
    )
    session.start("replay-start", "replay")
    session.mark_running()
    return SessionRuleProcessor.for_config(session, policy, AppConfig.model_validate(raw_config),
                                           exam_mode=ExamMode(exam_mode))


def relax_timeouts(pipeline: Any, timeout_ms: int = REPLAY_ANALYZER_TIMEOUT_MS) -> None:
    """Replay must never turn slow inference into missing observations."""

    pipeline.timeout_ms = timeout_ms
    for name in ("detector", "face_analyzer", "pose_analyzer", "hand_analyzer"):
        analyzer = getattr(pipeline, name, None)
        if analyzer is not None and hasattr(analyzer, "timeout_ms"):
            analyzer.timeout_ms = timeout_ms


def predicted_from_golden(records: list[Mapping[str, Any]], end_ms: int) -> list[PredictedEvent]:
    """Rebuild event spans from a saved golden stream (no ids: pair per type in order)."""

    open_at: dict[str, tuple[int, str]] = {}
    spans: list[PredictedEvent] = []
    for record in records:
        event_type = str(record.get("type", ""))
        if event_type not in EventType.__members__:
            continue
        if record.get("action") == "opened" and event_type not in open_at:
            open_at[event_type] = (int(record["t"]), str(record.get("severity", "")))
        elif record.get("action") == "closed" and event_type in open_at:
            start, severity = open_at.pop(event_type)
            spans.append(PredictedEvent(EventType(event_type), start, max(start, int(record["t"])), severity))
    spans.extend(PredictedEvent(EventType(name), start, max(start, end_ms), severity)
                 for name, (start, severity) in open_at.items())
    return sorted(spans, key=lambda item: (item.start_ms, item.event_type.value))


def write_golden(path: Path, result: ReplayResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(result.golden_lines()) + "\n", encoding="utf-8")


def read_golden(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = [
    "REPLAY_ANALYZER_TIMEOUT_MS",
    "create_replay_processor",
    "predicted_from_golden",
    "FrameSource",
    "ReplayResult",
    "RuleProcessor",
    "read_golden",
    "relax_timeouts",
    "run_replay",
    "write_golden",
]
