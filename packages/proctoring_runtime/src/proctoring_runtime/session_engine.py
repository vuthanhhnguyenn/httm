"""Headless session orchestration: capture → analysis → rules → events → risk.

``SessionEngine`` runs a live camera session; ``SessionRuleProcessor`` is the
per-frame rule path shared by live sessions and recorded-video replay. Neither
depends on Qt; the desktop app forwards the callbacks to its signals.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Lock
from typing import Any, Protocol
from uuid import UUID, uuid4

from proctoring_ai.capture.opencv_source import OpenCVFrameSource
from proctoring_ai.pipeline.frame_queue import LatestFrameQueue
from proctoring_ai.pipeline.metrics import PipelineMetrics
from proctoring_core.behavior.engine import BehaviorEngine
from proctoring_core.config.schema import AppConfig
from proctoring_core.events.aggregator import EventAggregator
from proctoring_core.events.correlation import CorrelationEngine
from proctoring_core.events.models import ObservationBundle
from proctoring_core.observations import Observation
from proctoring_core.risk.engine import RiskEngine
from proctoring_core.sessions.models import ExamSession
from proctoring_core.types import ExamMode, RiskLevel, SessionMode

from .bootstrap import RuntimeSettings, create_factory, load_settings
from .evidence import EvidenceRecorder
from .payloads import analysis_payload, correlation_item, event_item, metrics_snapshot
from .persistence import (
    JsonlEventRepository,
    session_dir,
    write_risk_summary,
    write_session_manifest,
)

DEFAULT_POLICY_ID = UUID("00000000-0000-0000-0000-000000000001")
METRICS_INTERVAL_S = 0.25

Callback = Callable[[dict[str, Any]], None]


def _ignore(_item: dict[str, Any]) -> None:
    return None


@dataclass(frozen=True, slots=True)
class SessionCallbacks:
    on_state: Callback = _ignore
    on_analysis: Callback = _ignore
    on_event: Callback = _ignore


class RiskSession(Protocol):
    """The part of :class:`ExamSession` the rule path updates."""

    @property
    def id(self) -> UUID: ...

    @property
    def current_risk_score(self) -> float: ...

    def update_risk(self, score: float, level: RiskLevel) -> Any: ...


class LatestPreview:
    """A one-frame, thread-safe handoff from camera capture to the UI paint loop."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._frame: dict[str, Any] | None = None

    def append(self, frame: dict[str, Any]) -> None:
        with self._lock:
            self._frame = dict(frame)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._frame) if self._frame is not None else None

    def clear(self) -> None:
        with self._lock:
            self._frame = None


def build_risk_engine(config: AppConfig) -> RiskEngine:
    risk = config.risk
    return RiskEngine(
        decay_per_second=risk.decay_per_second,
        active_event_ttl_seconds=risk.active_event_ttl_seconds,
        recovery_half_life_seconds=risk.recovery_half_life_seconds,
        supporting_group_cap=risk.supporting_group_cap,
    )


def build_correlation_engine(config: AppConfig) -> CorrelationEngine:
    risk = config.risk
    return CorrelationEngine(
        window_seconds=risk.correlation_window_seconds,
        confirmation_seconds=risk.correlation_confirmation_seconds,
        freshness_seconds=risk.active_event_ttl_seconds,
        session_cap=risk.correlation_cap,
    )


class SessionRuleProcessor:
    """Rules → aggregation → risk → correlation for one analysed frame."""

    def __init__(
        self,
        session: RiskSession,
        rules: BehaviorEngine,
        *,
        aggregator: EventAggregator | None = None,
        risk_engine: RiskEngine | None = None,
        correlation: CorrelationEngine | None = None,
        repository: JsonlEventRepository | None = None,
        recorder: EvidenceRecorder | None = None,
        on_event: Callback = _ignore,
        on_state: Callback = _ignore,
        exam_mode: ExamMode = ExamMode.COMPUTER,
    ) -> None:
        self.session = session
        self.exam_mode = ExamMode(exam_mode)
        self.rules = rules
        self.aggregator = aggregator or EventAggregator()
        self.risk_engine = risk_engine or RiskEngine()
        self.correlation = correlation
        self.repository = repository
        self.recorder = recorder
        self.on_event = on_event
        self.on_state = on_state

    @classmethod
    def for_config(cls, session: RiskSession, policy: Mapping[str, Any], config: AppConfig, *,
                   exam_mode: ExamMode = ExamMode.COMPUTER, **kwargs: Any) -> SessionRuleProcessor:
        rules = BehaviorEngine(session.id, policy, config.behavior_for(exam_mode))
        return cls(session, rules, risk_engine=build_risk_engine(config),
                   correlation=build_correlation_engine(config), exam_mode=exam_mode, **kwargs)

    def _emit(self, item: dict[str, Any]) -> None:
        item.setdefault("exam_mode", self.exam_mode.value)
        if self.repository is not None:
            self.repository.append(item)
        self.on_event(item)

    def record_external(self, item: dict[str, Any]) -> None:
        """Persist and forward records produced outside the rule path (evidence results)."""

        self._emit(item)

    def finish(self, occurred_at: datetime) -> list[dict[str, Any]]:
        """Close every open event when the session ends so no episode is left dangling."""

        emitted: list[dict[str, Any]] = []
        for transition in self.aggregator.flush(occurred_at=occurred_at):
            snapshot = self.risk_engine.apply_event(transition.event, occurred_at=occurred_at)
            item = event_item(transition, snapshot, None)
            self._emit(item)
            emitted.append(item)
        return emitted

    async def process(self, bundle: ObservationBundle, frame: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        session = self.session
        self.risk_engine.decay(session.id, occurred_at=bundle.captured_at_utc)
        transitions = []
        for signal in self.rules.evaluate(Observation.from_bundle(bundle)):
            transition = self.aggregator.process(signal.as_mapping(), occurred_at=bundle.captured_at_utc,
                                                 risk_score_after=session.current_risk_score)
            if transition is None:
                continue
            transitions.append(transition)
            snapshot = self.risk_engine.apply_event(transition.event, occurred_at=bundle.captured_at_utc)
            session.update_risk(snapshot.score_after, snapshot.level_after)
            evidence_status = (
                self.recorder.enqueue(transition.event.id, frame, bundle)
                if transition.action == "opened" and self.recorder is not None and frame is not None
                else "disabled"
            )
            item = event_item(transition, snapshot, evidence_status)
            self._emit(item)
            emitted.append(item)
        if self.correlation is not None:
            for finding in self.correlation.ingest(bundle, transitions):
                item = correlation_item(finding, self.risk_engine.apply_correlation(finding))
                self._emit(item)
                emitted.append(item)
        score = self.risk_engine.score(session.id)
        level = self.risk_engine.level_for(score)
        session.update_risk(score, level)
        self.on_state({"type": "risk", "score": score, "level": level.value, **self.risk_engine.history(session.id)})
        return emitted


class SessionEngine:
    """Own one live camera session until :meth:`request_stop`; errors propagate from :meth:`run`."""

    def __init__(
        self,
        *,
        repository_root: Path,
        camera_index: int,
        preview: LatestPreview,
        callbacks: SessionCallbacks | None = None,
        session_id: UUID | None = None,
        exam_mode: ExamMode = ExamMode.COMPUTER,
    ) -> None:
        self.exam_mode = ExamMode(exam_mode)
        self.repository_root = repository_root
        self.camera_index = camera_index
        self.preview = preview
        self.callbacks = callbacks or SessionCallbacks()
        self.session_id = session_id or uuid4()
        self._stop_requested = Event()
        self._head_calibration_requested = Event()
        self._analyzer_health: dict[str, str] = {}
        self._camera_frozen = False
        # UI-thread commands applied by the analysis loop between frames.
        self._commands: SimpleQueue[tuple[str, str | None]] = SimpleQueue()
        self.repository = JsonlEventRepository(
            session_dir(repository_root / "storage", self.session_id) / "events.jsonl")

    @property
    def storage_root(self) -> Path:
        return self.repository_root / "storage"

    def request_stop(self) -> None:
        self._stop_requested.set()

    def request_head_calibration(self) -> None:
        self._head_calibration_requested.set()

    def begin_gaze_calibration(self) -> None:
        self._commands.put(("gaze_begin", None))

    def set_gaze_target(self, name: str | None) -> None:
        self._commands.put(("gaze_target", name))

    def finish_gaze_calibration(self) -> None:
        self._commands.put(("gaze_finish", None))

    def cancel_gaze_calibration(self) -> None:
        self._commands.put(("gaze_cancel", None))

    def _apply_commands(self, processor: SessionRuleProcessor) -> None:
        rules = processor.rules
        while True:
            try:
                command, argument = self._commands.get_nowait()
            except Empty:
                return
            if command == "gaze_begin":
                rules.begin_gaze_calibration()
            elif command == "gaze_target":
                rules.set_gaze_target(argument)
            elif command in {"gaze_finish", "gaze_cancel"}:
                if command == "gaze_finish":
                    rules.finish_gaze_calibration()
                else:
                    rules.cancel_gaze_calibration()
                calibration = rules.gaze_calibration
                self._state({"type": "gaze_calibration", "status": calibration.status.value,
                             "reason": calibration.reason or ""})

    def _state(self, item: dict[str, Any]) -> None:
        self.callbacks.on_state(item)

    async def run(self) -> None:
        settings = load_settings(self.repository_root)
        factory = create_factory(self.repository_root, settings)
        session = self._create_session(settings)
        camera = settings.config.camera
        source = OpenCVFrameSource(self.camera_index, width=camera.width, height=camera.height,
                                   target_fps=camera.target_fps)
        queue = LatestFrameQueue(maxsize=1)
        metrics = PipelineMetrics(target_fps=camera.target_fps)
        capture_task: asyncio.Task[None] | None = None
        self._state({"type": "loading_models"})
        try:
            await source.open()
            self.repository.ensure_parent()
            write_session_manifest(self.repository.path.parent, session, self.exam_mode, settings)
            self._state({"type": "camera_ready"})
            capture_task = asyncio.create_task(self._capture_loop(source, queue, metrics), name="session-camera")
            await factory.initialize(on_progress=lambda health: self._state(
                {"type": "analyzer_health", "analyzers": health}))
            if settings.device and factory.health()["object_detector"] != "ready":
                raise RuntimeError(
                    f"YOLO không khởi tạo được trên thiết bị {settings.device}: "
                    f"{factory.health_details()['object_detector']['reason']}"
                )
            self._state({"type": "analyzer_health", "analyzers": factory.health_details()})
            session.mark_running()
            self._state({"type": "running", "session_id": str(self.session_id)})
            pipeline = factory.build_pipeline(settings.raw_config)
            await self._run_workers(pipeline, queue, metrics, session, settings, capture_task)
        finally:
            if capture_task is not None and not capture_task.done():
                capture_task.cancel()
            if capture_task is not None:
                await asyncio.gather(capture_task, return_exceptions=True)
            await source.close()
            factory.close()

    def _create_session(self, settings: RuntimeSettings) -> ExamSession:
        session = ExamSession.create(
            name=f"Phiên cục bộ {datetime.now().astimezone():%H:%M:%S}",
            mode=SessionMode.SINGLE_PERSON,
            camera_source=f"opencv:{self.camera_index}",
            policy_id=DEFAULT_POLICY_ID,
            policy_snapshot={**settings.raw_policy, "exam_mode": self.exam_mode.value},
            config_version_id=uuid4(),
            config_snapshot=settings.raw_config,
            created_by="desktop-operator",
            session_id=self.session_id,
        )
        session.start("desktop-start", "desktop-local")
        return session

    async def _run_workers(self, pipeline: Any, queue: LatestFrameQueue, metrics: PipelineMetrics,
                           session: ExamSession, settings: RuntimeSettings,
                           capture: asyncio.Task[None]) -> None:
        evidence, policy = settings.config.evidence, settings.policy
        recorder = EvidenceRecorder(
            self.storage_root, session.id,
            enabled=evidence.evidence_frame_enabled and policy.evidence_frame_enabled,
            retention_days=min(evidence.retention_days, policy.retention_days),
            on_result=self._on_evidence_result,
        )
        await asyncio.to_thread(EvidenceRecorder.purge_expired, self.storage_root)
        recorder.start()
        processor = SessionRuleProcessor.for_config(
            session, settings.raw_policy, settings.config, exam_mode=self.exam_mode,
            repository=self.repository, recorder=recorder,
            on_event=self.callbacks.on_event, on_state=self.callbacks.on_state)
        self._processor = processor
        process = asyncio.create_task(self._process_loop(pipeline, queue, metrics, processor), name="session-ai")
        stop = asyncio.create_task(self._wait_for_stop(), name="session-stop")
        try:
            done, _ = await asyncio.wait({capture, process, stop}, return_when=asyncio.FIRST_COMPLETED)
            if stop not in done:
                worker = next(task for task in (capture, process) if task in done)
                error = worker.exception()
                if error is not None:
                    raise error
                raise RuntimeError("Luồng camera hoặc AI đã dừng ngoài dự kiến")
        finally:
            for task in (capture, process, stop):
                if not task.done():
                    task.cancel()
            await asyncio.gather(capture, process, stop, return_exceptions=True)
            processor.finish(datetime.now(UTC))
            try:
                await recorder.close()
            finally:
                await asyncio.to_thread(self.write_risk_summary, processor.risk_engine)

    def _on_evidence_result(self, result: dict[str, Any]) -> None:
        self.repository.append(result)
        self.callbacks.on_event(result)

    def write_risk_summary(self, engine: RiskEngine) -> None:
        if self.repository.path.parent.is_dir():
            write_risk_summary(self.repository.path.parent, engine, self.session_id)

    async def _wait_for_stop(self) -> None:
        while not self._stop_requested.is_set():
            await asyncio.sleep(0.05)

    async def _capture_loop(self, source: OpenCVFrameSource, queue: LatestFrameQueue,
                            metrics: PipelineMetrics) -> None:
        last_metrics_at = 0.0
        last_drop_count = 0
        while not self._stop_requested.is_set():
            frame = await source.read()
            if frame is None:
                await asyncio.sleep(0.003)
                continue
            captured = dict(frame)
            self.preview.append({
                "frame_id": captured["frame_id"],
                "captured_at_monotonic_ms": captured["captured_at_monotonic_ms"],
                "image": captured.get("image"),
                "frame_size": captured["frame_size"],
            })
            metrics.record_capture()
            queue.put_nowait(captured)
            dropped = queue.dropped_frames
            if dropped > last_drop_count:
                metrics.record_drop(dropped - last_drop_count)
                last_drop_count = dropped
            now = time.monotonic()
            if now - last_metrics_at >= METRICS_INTERVAL_S:
                last_metrics_at = now
                self._state({"type": "metrics", "metrics": metrics_snapshot(metrics, queue, self._analyzer_health)})

    async def _process_loop(self, pipeline: Any, queue: LatestFrameQueue, metrics: PipelineMetrics,
                            processor: SessionRuleProcessor) -> None:
        last_metrics_at = 0.0
        while not self._stop_requested.is_set():
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            try:
                self._apply_commands(processor)
                if self._head_calibration_requested.is_set():
                    self._head_calibration_requested.clear()
                    if pipeline.face_analyzer is not None:
                        pipeline.face_analyzer.request_calibration()
                    processor.rules.request_calibration()
                bundle: ObservationBundle = await pipeline.analyze(frame, session_id=processor.session.id)
                self._analyzer_health = {name: item.status for name, item in bundle.analyzer_health.items()}
                frozen = bool(bundle.image_quality.get("frozen", False))
                if frozen != self._camera_frozen:
                    self._camera_frozen = frozen
                    self._state({"type": "camera_fault", "active": frozen, "reason": "FROZEN_FRAMES"})
                payload = analysis_payload(bundle, self._analyzer_health)
                payload["analyzer_performance"] = pipeline.performance() if hasattr(pipeline, "performance") else {}
                payload["gaze_status"] = processor.rules.gaze_calibration.status.value
                self.callbacks.on_analysis(payload)
                await processor.process(bundle, frame)
                captured_ms = int(frame.get("captured_at_monotonic_ms", 0))
                metrics.record_processed(max(0.0, time.monotonic_ns() / 1_000_000 - captured_ms))
                now = time.monotonic()
                if now - last_metrics_at >= METRICS_INTERVAL_S:
                    last_metrics_at = now
                    self._state({"type": "metrics",
                                 "metrics": metrics_snapshot(metrics, queue, self._analyzer_health)})
            finally:
                queue.task_done()


__all__ = [
    "LatestPreview",
    "RiskSession",
    "SessionCallbacks",
    "SessionEngine",
    "SessionRuleProcessor",
    "build_correlation_engine",
    "build_risk_engine",
]
