"""Run a recorded video with several candidates through per-seat rule processors.

Each seat gets its own :class:`SessionRuleProcessor`, so every phase-1 rule runs
unchanged per candidate; the neighbour rule needs all seats at once and is fed
into the seat processors as extra signals. Outputs land in one directory:
``events.jsonl``, ``evidence/``, ``annotated.mp4`` and ``timing.json``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from proctoring_ai.multi.focus import FOCUS_EVENT_TYPES
from proctoring_ai.multi.pipeline import DetectionStage, MultiFrameResult, MultiSubjectPipeline
from proctoring_ai.multi.timing import TimingRecorder
from proctoring_core.behavior.neighbor import NeighborLookEvaluator, SubjectHead
from proctoring_core.behavior.signals import RuleSignal, SignalState
from proctoring_core.config.schema import AppConfig
from proctoring_core.types import ExamMode

from .replay import FrameSource, create_replay_processor
from .session_engine import SessionRuleProcessor

cv2: Any
try:
    import cv2
except ImportError:  # pragma: no cover - exercised in environments without CV wheels
    cv2 = None


@dataclass(slots=True)
class MultiReplayResult:
    events: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    frames_analyzed: int = 0


class AnnotatedVideoWriter:
    """Draw seats, faces and open events; ASCII event codes because cv2 cannot render Vietnamese."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path = path
        self.fps = fps
        self._writer: Any = None

    def write(self, image: Any, result: MultiFrameResult, open_labels: Mapping[UUID, list[str]],
              repeat: int = 1) -> None:
        """``repeat`` holds the frame over skipped source frames so video time matches event offsets."""
        if cv2 is None or image is None:
            return
        height, width = image.shape[:2]
        if self._writer is None:
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        canvas = image.copy()

        def rect(box: Mapping[str, float], colour: tuple[int, int, int], thickness: int = 2) -> tuple[int, int]:
            x0, y0 = int(box["x"] * width), int(box["y"] * height)
            x1, y1 = int((box["x"] + box["width"]) * width), int((box["y"] + box["height"]) * height)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thickness)
            return x0, y0

        for seat in result.tracker.over_limit:
            rect(seat.bbox.as_dict(), (128, 128, 128), 1)
        for seat in result.tracker.analyzed:
            labels = open_labels.get(seat.seat_id, [])
            x0, y0 = rect(seat.bbox.as_dict(), (0, 0, 255) if labels else (0, 200, 0))
            face = result.faces.get(seat.seat_id)
            text = f"#{seat.runtime_track_id}"
            if face is not None and face.face is not None:
                rect(face.face["bbox"], (0, 220, 255), 1)
                text += f" yaw {float(face.face.get('yaw', 0)):+.0f} pitch {float(face.face.get('pitch', 0)):+.0f}"
                text += " R" if face.reused else ""
            cv2.putText(canvas, text, (x0 + 2, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            for index, label in enumerate(labels[:3]):
                cv2.putText(canvas, label, (x0 + 2, y0 + 16 + 14 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 255), 1)
        for _ in range(max(1, repeat)):
            self._writer.write(canvas)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class _SeatRules:
    """Per-seat rule processors plus the cross-seat neighbour rule."""

    def __init__(self, session_id: UUID, policy: Mapping[str, Any], raw_config: Mapping[str, Any],
                 exam_mode: ExamMode) -> None:
        self.session_id = session_id
        self.policy, self.raw_config, self.exam_mode = policy, raw_config, exam_mode
        behavior = AppConfig.model_validate(dict(raw_config)).behavior_for(exam_mode)
        self.neighbor = NeighborLookEvaluator(yaw_degrees=behavior.head_turn.yaw_degrees,
                                              min_quality=behavior.quality.face_min_quality,
                                              max_sample_gap_ms=behavior.quality.max_sample_gap_ms)
        self.processors: dict[UUID, SessionRuleProcessor] = {}
        self.runtime_ids: dict[UUID, int] = {}
        self.open_labels: dict[UUID, dict[str, str]] = {}

    def _processor(self, seat_id: UUID) -> SessionRuleProcessor:
        processor = self.processors.get(seat_id)
        if processor is None:
            processor = create_replay_processor(self.session_id, self.policy, self.raw_config, self.exam_mode)
            self.processors[seat_id] = processor
        return processor

    def _annotate(self, seat_id: UUID, item: dict[str, Any], offset: int, reused: bool) -> dict[str, Any]:
        labels = self.open_labels.setdefault(seat_id, {})
        if item.get("action") == "opened":
            labels[str(item.get("id"))] = str(item.get("type"))
        elif item.get("action") == "closed":
            labels.pop(str(item.get("id")), None)
        return {**item, "seat_id": str(seat_id), "runtime_track_id": self.runtime_ids.get(seat_id),
                "video_offset_ms": offset, "face_reused": reused}

    def _heads(self, result: MultiFrameResult) -> tuple[list[SubjectHead], list[SubjectHead]]:
        """Seats analysed this frame are sampled; seats skipped by rotation still count as neighbours."""

        heads, neighbors = [], []
        for seat in result.tracker.analyzed:
            if seat.seat_id not in result.bundles:
                neighbors.append(SubjectHead(str(seat.seat_id), seat.bbox, None, 0.0, False))
                continue
            face = result.faces.get(seat.seat_id)
            data = face.face if face is not None else None
            heads.append(SubjectHead(
                str(seat.seat_id), seat.bbox,
                float(data["yaw"]) if data is not None and "yaw" in data else None,
                float(data.get("quality", 0.0)) if data is not None else 0.0,
                data is not None and data.get("calibration_status") == "calibrated"))
        return heads, neighbors

    async def step(self, result: MultiFrameResult, frame: dict[str, Any], offset: int) -> list[tuple[UUID, dict[str, Any]]]:
        for seat in (*result.tracker.analyzed, *result.tracker.missing):
            self.runtime_ids[seat.seat_id] = seat.runtime_track_id
        extra: dict[str, list[RuleSignal]] = {}
        heads, neighbors = self._heads(result)
        for signal in self.neighbor.evaluate(session_id=self.session_id, subjects=heads, neighbors=neighbors,
                                             monotonic_ms=result.monotonic_ms):
            extra.setdefault(str(signal.track_id), []).append(signal)
        for seat in result.tracker.missing:
            # A seat nobody occupies cannot be looking at anyone.
            for signal in self.neighbor.forget(str(seat.seat_id)):
                extra.setdefault(str(signal.track_id), []).append(signal)
        emitted: list[tuple[UUID, dict[str, Any]]] = []
        for seat_id, bundle in result.bundles.items():
            processor = self._processor(seat_id)
            face = result.faces.get(seat_id)
            items = await processor.process(bundle, frame, extra_signals=extra.get(str(seat_id), ()))
            emitted += [(seat_id, self._annotate(seat_id, item, offset, bool(face and face.reused)))
                        for item in items]
        for seat in result.tracker.expired:
            self.neighbor.forget(str(seat.seat_id))
            expired = self.processors.pop(seat.seat_id, None)
            if expired is not None:
                # Closed because the seat was given up, not because the candidate came back.
                emitted += [(seat.seat_id, {**self._annotate(seat.seat_id, item, offset, False),
                                            "close_reason": "SEAT_EXPIRED", "seat_last_seen_ms": seat.last_seen_ms})
                            for item in expired.finish(frame["captured_at_utc"])]
        return emitted

    def focus_seats(self) -> list[UUID]:
        live = {SignalState.CANDIDATE, SignalState.ACTIVE}
        return [seat_id for seat_id, processor in self.processors.items()
                if any(signal.event_type in FOCUS_EVENT_TYPES and signal.state in live
                       for signal in processor.last_signals)]

    def finish(self, frame: dict[str, Any], offset: int) -> list[tuple[UUID, dict[str, Any]]]:
        emitted = []
        for seat_id, processor in self.processors.items():
            emitted += [(seat_id, self._annotate(seat_id, item, offset, False))
                        for item in processor.finish(frame["captured_at_utc"])]
        return emitted

    def labels(self) -> dict[UUID, list[str]]:
        return {seat_id: list(labels.values()) for seat_id, labels in self.open_labels.items()}


def _save_evidence(directory: Path, item: dict[str, Any], image: Any) -> str:
    if cv2 is None or image is None:
        return "failed"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        return "saved" if cv2.imwrite(str(directory / f"{item['id']}.jpg"), image) else "failed"
    except Exception:
        return "failed"


async def run_multi_replay(
    source: FrameSource,
    pipeline: MultiSubjectPipeline,
    *,
    policy: Mapping[str, Any],
    raw_config: Mapping[str, Any],
    out_dir: Path,
    mode: Literal["full", "realtime"] = "full",
    overlap: bool = True,
    write_video: bool = True,
    video_fps: float = 10.0,
    session_id: UUID | None = None,
    exam_mode: ExamMode | str = ExamMode.COMPUTER,
    run_config: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> MultiReplayResult:
    session = session_id or uuid4()
    out_dir.mkdir(parents=True, exist_ok=True)
    rules = _SeatRules(session, policy, raw_config, ExamMode(exam_mode))
    timing = TimingRecorder()
    writer = AnnotatedVideoWriter(out_dir / "annotated.mp4", video_fps) if write_video else None
    result = MultiReplayResult()
    events_file = (out_dir / "events.jsonl").open("w", encoding="utf-8")
    started = clock()
    previous_offset: int | None = None

    async def next_frame() -> dict[str, Any] | None:
        nonlocal previous_offset
        while (frame := await source.read()) is not None:
            offset = int(frame.get("video_offset_ms", frame["captured_at_monotonic_ms"]))
            period = offset - previous_offset if previous_offset is not None else 0
            previous_offset = offset
            if mode == "realtime":
                elapsed_ms = (clock() - started) * 1000
                if period > 0 and elapsed_ms - offset > period:
                    timing.add_dropped()  # a live camera would have replaced this frame already
                    continue
                if offset > elapsed_ms:
                    await sleep((offset - elapsed_ms) / 1000)
            return frame
        return None

    def record(pairs: list[tuple[UUID, dict[str, Any]]], image: Any) -> None:
        for _seat_id, item in pairs:
            if item.get("action") == "opened" and "id" in item:
                item["evidence_status"] = _save_evidence(out_dir / "evidence", item, image)
            events_file.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            result.events.append(item)

    last_frame: dict[str, Any] | None = None
    last_offset = 0
    written_offset: int | None = None
    await source.open()
    try:
        frame = await next_frame()
        pending: asyncio.Task[DetectionStage] | None = (
            asyncio.create_task(pipeline.detect(frame)) if frame is not None else None)
        while frame is not None and pending is not None:
            stage = await pending
            pending = None
            following: dict[str, Any] | None = None
            if overlap:
                following = await next_frame()
                if following is not None:
                    pending = asyncio.create_task(pipeline.detect(following))
            analysed = await pipeline.analyze_subjects(stage, session_id=session)
            offset = int(frame.get("video_offset_ms", frame["captured_at_monotonic_ms"]))
            record(await rules.step(analysed, frame, offset), frame.get("image"))
            for seat_id in rules.focus_seats():
                pipeline.mark_focus(seat_id, analysed.monotonic_ms)
            latency = ((clock() - started) * 1000 - offset if mode == "realtime"
                       else analysed.timing.total_ms)
            timing.add(analysed, latency_ms=latency)
            if writer is not None:
                repeat = 1 if written_offset is None else round((offset - written_offset) * video_fps / 1000)
                writer.write(frame.get("image"), analysed, rules.labels(), repeat=repeat)
                written_offset = offset
            result.frames_analyzed += 1
            last_frame, last_offset = frame, offset
            if not overlap:
                following = await next_frame()
                if following is not None:
                    pending = asyncio.create_task(pipeline.detect(following))
            frame = following
        if last_frame is not None:
            record(rules.finish(last_frame, last_offset), None)
    finally:
        await source.close()
        events_file.close()
        if writer is not None:
            writer.close()
        pipeline.close()
    result.timing = {**timing.summary(wall_s=clock() - started), "config": dict(run_config or {})}
    (out_dir / "timing.json").write_text(json.dumps(result.timing, ensure_ascii=False, indent=2, default=str),
                                         encoding="utf-8")
    return result


__all__ = ["AnnotatedVideoWriter", "MultiReplayResult", "run_multi_replay"]
