"""Local camera and analysis runtime hosted by a Qt worker thread."""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
from typing import Any
from uuid import UUID, uuid4

import yaml
from proctoring_ai.analyzer_factory import AnalyzerFactory
from proctoring_ai.capture.opencv_source import OpenCVFrameSource
from proctoring_ai.pipeline.frame_queue import LatestFrameQueue
from proctoring_ai.pipeline.metrics import PipelineMetrics
from proctoring_core.behavior.head import HeadRuleEvaluator
from proctoring_core.behavior.objects import ObjectRuleEvaluator
from proctoring_core.behavior.presence import PresenceRuleEvaluator
from proctoring_core.behavior.safety import SafetyRuleEvaluator
from proctoring_core.behavior.supporting import SupportingRuleEvaluator
from proctoring_core.events.aggregator import EventAggregator
from proctoring_core.events.correlation import CorrelationEngine
from proctoring_core.events.models import ObservationBundle
from proctoring_core.risk.engine import RiskEngine
from proctoring_core.sessions.models import ExamSession
from proctoring_core.types import SessionMode
from PySide6.QtCore import QThread, Signal

from .evidence import EvidenceRecorder
from .messages import event_description, event_name, level_name


class LatestPreview:
    """A one-frame, thread-safe handoff from camera capture to the Qt paint loop."""

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


class DesktopRuntime(QThread):
    """Capture and analyze locally; UI updates contain metadata, never full frames."""

    state_changed = Signal(dict)
    analysis_ready = Signal(dict)
    event_ready = Signal(dict)
    runtime_error = Signal(str)

    def __init__(
        self,
        *,
        camera_index: int,
        repository_root: Path,
        preview: LatestPreview,
    ) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.repository_root = repository_root
        self.preview = preview
        self.session_id = uuid4()
        self._stop_requested = Event()
        self._head_calibration_requested = Event()
        self._event_path: Path | None = None
        self._analyzer_health: dict[str, str] = {}

    def request_stop(self) -> None:
        self._stop_requested.set()

    def request_head_calibration(self) -> None:
        self._head_calibration_requested.set()

    def run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            error_log = self.repository_root / "storage" / "runtime-errors.log"
            try:
                error_log.parent.mkdir(parents=True, exist_ok=True)
                with error_log.open("a", encoding="utf-8") as stream:
                    stream.write(f"\n[{datetime.now().astimezone().isoformat()}] Lỗi giám sát\n")
                    stream.write(traceback.format_exc())
                detail = f" Chi tiết kỹ thuật đã lưu tại {error_log}."
            except OSError:
                detail = " Không ghi được tệp lỗi; hãy kiểm tra quyền ghi thư mục storage."
            reason = ("CUDA chưa sẵn sàng; hãy chạy bằng scripts/run_gpu.ps1."
                      if "CUDA" in str(exc) else "Không thể tiếp tục giám sát; hãy kiểm tra camera và model AI.")
            self.runtime_error.emit(reason + detail)
        finally:
            self.preview.clear()
            self.state_changed.emit({"type": "stopped"})

    async def _run_async(self) -> None:
        config = _read_yaml(self.repository_root / "configs" / "default.yaml")
        policy = _read_yaml(self.repository_root / "configs" / "exam_policy.yaml")
        policy = policy.get("exam_policy", policy)
        camera = config.get("camera", {})
        detection = config.get("detection", {})
        requested_device = os.environ.get("PROCTORING_AI_DEVICE")
        if requested_device:
            if requested_device != "cpu":
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA chưa khả dụng trong môi trường Python đang chạy")
            detection = {**detection, "device": requested_device}
            config = {**config, "detection": detection}
        self._event_path = self.repository_root / "storage" / "sessions" / str(self.session_id) / "events.jsonl"

        factory = AnalyzerFactory(
            self.repository_root / str(detection.get("model_manifest", "models/manifest.yaml")),
            detector_image_size=int(detection.get("image_size", 416)),
            detector_device=detection.get("device"),
        )
        config_version_id = uuid4()
        session = ExamSession.create(
            name=f"Phiên cục bộ {datetime.now().astimezone():%H:%M:%S}",
            mode=SessionMode.SINGLE_PERSON,
            camera_source=f"opencv:{self.camera_index}",
            policy_id=UUID("00000000-0000-0000-0000-000000000001"),
            policy_snapshot=policy,
            config_version_id=config_version_id,
            config_snapshot=config,
            created_by="desktop-operator",
            session_id=self.session_id,
        )
        session.start("desktop-start", "desktop-local")
        source = OpenCVFrameSource(
            self.camera_index,
            width=int(camera.get("width", 1280)),
            height=int(camera.get("height", 720)),
            target_fps=float(camera.get("target_fps", 25)),
        )
        queue = LatestFrameQueue(maxsize=1)
        metrics = PipelineMetrics(target_fps=float(camera.get("target_fps", 25)))
        capture_task: asyncio.Task[None] | None = None
        self.state_changed.emit({"type": "loading_models"})
        try:
            await source.open()
            self._event_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_changed.emit({"type": "camera_ready"})
            capture_task = asyncio.create_task(
                self._capture_loop(source, queue, metrics), name="desktop-camera"
            )
            await factory.initialize(on_progress=lambda health: self.state_changed.emit(
                {"type": "analyzer_health", "analyzers": health}
            ))
            if requested_device and factory.health()["object_detector"] != "ready":
                raise RuntimeError(
                    f"YOLO không khởi tạo được trên thiết bị {requested_device}: "
                    f"{factory.health_details()['object_detector']['reason']}"
                )
            self.state_changed.emit({"type": "analyzer_health", "analyzers": factory.health_details()})
            session.mark_running()
            self.state_changed.emit({"type": "running", "session_id": str(self.session_id)})
            pipeline = factory.build_pipeline(config)
            await self._run_workers(
                source,
                pipeline,
                queue,
                metrics,
                session,
                policy,
                config,
                capture_task=capture_task,
            )
        finally:
            if capture_task is not None and not capture_task.done():
                capture_task.cancel()
            if capture_task is not None:
                await asyncio.gather(capture_task, return_exceptions=True)
            await source.close()
            factory.close()

    async def _run_workers(
        self,
        source: OpenCVFrameSource,
        pipeline: Any,
        queue: LatestFrameQueue,
        metrics: PipelineMetrics,
        session: ExamSession,
        policy: dict[str, Any],
        config: dict[str, Any],
        *,
        capture_task: asyncio.Task[None] | None = None,
    ) -> None:
        capture = capture_task or asyncio.create_task(
            self._capture_loop(source, queue, metrics), name="desktop-camera"
        )
        evidence = config.get("evidence", {})
        enabled = bool(evidence.get("evidence_frame_enabled", True)) and bool(policy.get("evidence_frame_enabled", True))
        retention_days = min(int(evidence.get("retention_days", 30)), int(policy.get("retention_days", 30)))
        recorder = EvidenceRecorder(self.repository_root / "storage", session.id,
                                    enabled=enabled, retention_days=retention_days,
                                    on_result=self._on_evidence_result)
        await asyncio.to_thread(EvidenceRecorder.purge_expired, self.repository_root / "storage")
        recorder.start()
        risk_settings = config.get("risk", {})
        risk_engine = RiskEngine(
            decay_per_second=float(risk_settings.get("decay_per_second", 1.5)),
            active_event_ttl_seconds=float(risk_settings.get("active_event_ttl_seconds", 2)),
            recovery_half_life_seconds=risk_settings.get("recovery_half_life_seconds", 1.0),
        )
        correlation_engine = CorrelationEngine(
            window_seconds=float(risk_settings.get("correlation_window_seconds", 5)),
            confirmation_seconds=float(risk_settings.get("correlation_confirmation_seconds", 0.25)),
            freshness_seconds=float(risk_settings.get("active_event_ttl_seconds", 2)),
            session_cap=float(risk_settings.get("correlation_cap", 20)),
        )
        process = asyncio.create_task(self._process_loop(pipeline, queue, metrics, session, policy, config,
                                                         recorder, risk_engine, correlation_engine), name="desktop-ai")
        stop = asyncio.create_task(self._wait_for_stop(), name="desktop-stop")
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
            try:
                await recorder.close()
            finally:
                await asyncio.to_thread(self._write_risk_summary, risk_engine, session.id)

    async def _wait_for_stop(self) -> None:
        while not self._stop_requested.is_set():
            await asyncio.sleep(0.05)

    async def _capture_loop(
        self,
        source: OpenCVFrameSource,
        queue: LatestFrameQueue,
        metrics: PipelineMetrics,
    ) -> None:
        last_metrics_at = 0.0
        last_drop_count = 0
        while not self._stop_requested.is_set():
            frame = await source.read()
            if frame is None:
                await asyncio.sleep(0.003)
                continue
            captured = dict(frame)
            self.preview.append(
                {
                    "frame_id": captured["frame_id"],
                    "captured_at_monotonic_ms": captured["captured_at_monotonic_ms"],
                    "image": captured.get("image"),
                    "frame_size": captured["frame_size"],
                }
            )
            metrics.record_capture()
            queue.put_nowait(captured)
            dropped = queue.dropped_frames
            if dropped > last_drop_count:
                metrics.record_drop(dropped - last_drop_count)
                last_drop_count = dropped
            now = time.monotonic()
            if now - last_metrics_at >= 0.25:
                last_metrics_at = now
                self.state_changed.emit({"type": "metrics", "metrics": _metrics_snapshot(metrics, queue, self._analyzer_health)})

    async def _process_loop(
        self,
        pipeline: Any,
        queue: LatestFrameQueue,
        metrics: PipelineMetrics,
        session: ExamSession,
        policy: dict[str, Any],
        config: dict[str, Any],
        recorder: EvidenceRecorder,
        risk_engine: RiskEngine,
        correlation_engine: CorrelationEngine,
    ) -> None:
        rules = _RuleSet(session.id, policy, config)
        aggregator = EventAggregator()
        last_metrics_at = 0.0
        analyzer_health: dict[str, str] = {}
        while not self._stop_requested.is_set():
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            try:
                if self._head_calibration_requested.is_set():
                    self._head_calibration_requested.clear()
                    if pipeline.face_analyzer is not None:
                        pipeline.face_analyzer.request_calibration()
                bundle: ObservationBundle = await pipeline.analyze(frame, session_id=session.id)
                analyzer_health = {name: item.status for name, item in bundle.analyzer_health.items()}
                self._analyzer_health = analyzer_health
                payload = _analysis_payload(bundle, analyzer_health)
                payload["analyzer_performance"] = pipeline.performance() if hasattr(pipeline, "performance") else {}
                self.analysis_ready.emit(payload)
                await self._process_rules(bundle, rules, aggregator, risk_engine, session, frame, recorder,
                                          correlation_engine=correlation_engine)
                captured_ms = int(frame.get("captured_at_monotonic_ms", 0))
                latency = max(0.0, time.monotonic_ns() / 1_000_000 - captured_ms)
                metrics.record_processed(latency)
                now = time.monotonic()
                if now - last_metrics_at >= 0.25:
                    last_metrics_at = now
                    self.state_changed.emit(
                        {
                            "type": "metrics",
                            "metrics": _metrics_snapshot(metrics, queue, analyzer_health),
                        }
                    )
            finally:
                queue.task_done()

    async def _process_rules(
        self,
        bundle: ObservationBundle,
        rules: _RuleSet,
        aggregator: EventAggregator,
        risk_engine: RiskEngine,
        session: ExamSession,
        frame: dict[str, Any] | None = None,
        recorder: EvidenceRecorder | None = None,
        *,
        correlation_engine: CorrelationEngine | None = None,
    ) -> None:
        risk_engine.decay(session.id, occurred_at=bundle.captured_at_utc)
        signals = rules.evaluate(bundle)
        transitions = []
        for signal in signals:
            transition = aggregator.process(
                signal,
                occurred_at=bundle.captured_at_utc,
                risk_score_after=session.current_risk_score,
            )
            if transition is None:
                continue
            transitions.append(transition)
            snapshot = risk_engine.apply_event(transition.event, occurred_at=bundle.captured_at_utc)
            session.update_risk(snapshot.score_after, snapshot.level_after)
            evidence_status = (
                recorder.enqueue(transition.event.id, frame, bundle)
                if transition.action == "opened" and recorder is not None and frame is not None
                else "disabled"
            )
            item = {
                "id": str(transition.event.id),
                "action": transition.action,
                "type": transition.event.event_type.value,
                "severity": transition.event.severity.value,
                "confidence": transition.event.confidence,
                "explanation": transition.event.explanation,
                "metrics": transition.event.metrics,
                "occurred_at": transition.event.started_at.isoformat(),
                "risk_score": snapshot.score_after,
                "risk_level": snapshot.level_after.value,
                "cumulative_score": snapshot.cumulative_score,
                "peak_score": snapshot.peak_score,
                "event_count": snapshot.event_count,
                "risk_reason_codes": list(snapshot.reason_codes),
                "evidence_status": evidence_status if transition.action == "opened" else None,
            }
            item["label_vi"] = event_name(item["type"])
            item["severity_vi"] = level_name(item["severity"])
            item["message_vi"] = event_description(item)
            self._append_event(item)
            self.event_ready.emit(item)
        if correlation_engine is not None:
            for finding in correlation_engine.ingest(bundle, transitions):
                snapshot = risk_engine.apply_correlation(finding)
                item = {
                    "id": str(finding.id), "action": "correlation",
                    "type": f"CORRELATION_{finding.rule_code}",
                    "source_event_ids": [str(event_id) for event_id in finding.source_event_ids],
                    "confidence": finding.confidence,
                    "severity": "LOW", "occurred_at": finding.occurred_at.isoformat(),
                    "risk_score": snapshot.score_after, "risk_level": snapshot.level_after.value,
                    "cumulative_score": snapshot.cumulative_score,
                    "peak_score": snapshot.peak_score,
                    "event_count": snapshot.event_count,
                    "risk_reason_codes": list(snapshot.reason_codes),
                    "correlation_points": finding.points,
                    "track_id": str(finding.track_id) if finding.track_id else None,
                }
                item["label_vi"] = event_name(item["type"])
                item["severity_vi"] = level_name(item["severity"])
                item["message_vi"] = event_description(item)
                self._append_event(item)
                self.event_ready.emit(item)
        score = risk_engine.score(session.id)
        history = risk_engine.history(session.id)
        level = risk_engine.level_for(score)
        session.update_risk(score, level)
        self.state_changed.emit({"type": "risk", "score": score, "level": level.value, **history})

    def _on_evidence_result(self, result: dict[str, Any]) -> None:
        self._append_event(result)
        self.event_ready.emit(result)

    def _write_risk_summary(self, engine: RiskEngine, session_id: UUID) -> None:
        if self._event_path is None:
            return
        score = engine.score(session_id)
        item = {"session_id": str(session_id), "current_score": score,
                "current_level": engine.level_for(score).value,
                **engine.history(session_id), "updated_at": datetime.now().astimezone().isoformat()}
        target = self._event_path.parent / "risk-summary.json"
        pending = target.with_suffix(".json.tmp")
        pending.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(pending, target)

    def _append_event(self, item: dict[str, Any]) -> None:
        if self._event_path is None:
            return
        with self._event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


class _RuleSet:
    def __init__(self, session_id: UUID, policy: dict[str, Any], config: dict[str, Any]) -> None:
        self.session_id = session_id
        behavior = config.get("behavior", {})
        self.presence = PresenceRuleEvaluator(
            min_duration_ms=round(float(behavior.get("person_missing", {}).get("minimum_duration_seconds", 2)) * 1000)
        )
        self.head = HeadRuleEvaluator(behavior)
        phone_threshold = float(behavior.get("phone", {}).get("minimum_confidence", 0.55))
        document_threshold = float(behavior.get("document", {}).get("minimum_confidence", 0.7))
        self.objects = ObjectRuleEvaluator(
            policy,
            confidence_threshold=min(phone_threshold, document_threshold),
            phone_confidence_threshold=phone_threshold,
            document_confidence_threshold=document_threshold,
            phone_min_duration_ms=round(float(behavior.get("phone", {}).get("minimum_duration_seconds", 0.6)) * 1000),
            document_min_duration_ms=round(float(behavior.get("document", {}).get("minimum_duration_seconds", 1)) * 1000),
            phone_release_confidence=float(behavior.get("phone", {}).get("release_confidence", 0.35)),
            max_gap_ms=int(behavior.get("phone", {}).get("maximum_gap_ms", 600)),
            phone_fast_confidence=float(behavior.get("phone", {}).get("fast_confidence", 0.85)),
            phone_fast_duration_ms=round(float(behavior.get("phone", {}).get("fast_duration_seconds", 0.2)) * 1000),
            phone_fast_min_hits=int(behavior.get("phone", {}).get("fast_min_hits", 2)),
        )
        self.safety = SafetyRuleEvaluator(
            min_duration_ms=round(float(behavior.get("face_not_visible", {}).get("minimum_duration_seconds", 2)) * 1000)
        )
        self.supporting = SupportingRuleEvaluator(
            min_duration_ms=round(float(behavior.get("hand_activity", {}).get("minimum_duration_seconds", 2)) * 1000)
        )
        self._last_head_track_id: str | None = None
        self._last_hand_track_id: str | None = None

    def evaluate(self, bundle: ObservationBundle) -> list[dict[str, Any]]:
        timestamp = bundle.captured_at_monotonic_ms
        detector = bundle.analyzer_health.get("object_detector")
        face_health = bundle.analyzer_health.get("face")
        hand_health = bundle.analyzer_health.get("hands")
        track = bundle.people[0] if len(bundle.people) == 1 else {}
        track_id = track.get("track_id")
        signals: list[dict[str, Any]] = []
        if detector is not None and detector.status == "ready":
            signals.extend(
                self.presence.evaluate(session_id=self.session_id, people=bundle.people, monotonic_ms=timestamp)
            )
            signals.extend(
                self.objects.evaluate(
                    session_id=self.session_id,
                    track_id=track_id,
                    objects=bundle.objects,
                    monotonic_ms=timestamp,
                )
            )
        # Anonymous face signals remain visible, but cannot be paired with another person.
        face = bundle.faces[0] if bundle.faces else {}
        face_matched = bool(track_id and _inside_person(face.get("bbox"), track.get("bbox")))
        head_track_id = track_id if face_matched or (track_id and not face) else None
        if self._last_head_track_id and self._last_head_track_id != head_track_id:
            signals.extend(self.head.evaluate(
                session_id=self.session_id, track_id=self._last_head_track_id,
                sample={"monotonic_ms": timestamp, "quality": 0, "pose_valid": False},
            ))
        self._last_head_track_id = head_track_id
        face_sample = face or {"quality": 0.0, "pose_valid": False}
        signals.extend(self.head.evaluate(
            session_id=self.session_id, track_id=head_track_id,
            sample={**face_sample, "monotonic_ms": timestamp},
        ))
        face_known = face_health is not None and face_health.status in {"ready", "no_detection"}
        face_visible = bool(bundle.faces) if face_known else True
        hand_known = hand_health is not None and hand_health.status in {"ready", "no_detection"}
        matched_hands = [hand for hand in bundle.hands
                         if track_id and _inside_person(hand.get("bbox"), track.get("bbox"))]
        hand_track_id = track_id if matched_hands or (track_id and not bundle.hands) else None
        hands_for_rule = matched_hands if hand_track_id else bundle.hands
        hands_near_face = _hands_near_face(bundle.faces, hands_for_rule) if hand_known else False
        if face_known or bundle.image_quality:
            signals.extend(
                self.safety.evaluate(
                    session_id=self.session_id,
                    track_id=track_id,
                    sample={
                        **track,
                        "monotonic_ms": timestamp,
                        "image_quality": bundle.image_quality,
                        "face_visible": face_visible,
                    },
                )
            )
        if hand_known:
            if self._last_hand_track_id and self._last_hand_track_id != hand_track_id:
                signals.extend(self.supporting.evaluate(
                    session_id=self.session_id, track_id=self._last_hand_track_id,
                    sample={"monotonic_ms": timestamp, "hands_near_face": False, "hand_activity": False},
                ))
            self._last_hand_track_id = hand_track_id
            signals.extend(
                self.supporting.evaluate(
                    session_id=self.session_id,
                    track_id=hand_track_id,
                    sample={
                        "monotonic_ms": timestamp,
                        "hands_near_face": hands_near_face,
                        "hand_activity": any(bool(hand.get("activity")) for hand in hands_for_rule),
                    },
                )
            )
        return signals


def _inside_person(inner: Any, outer: Any) -> bool:
    if not isinstance(inner, dict) or not isinstance(outer, dict):
        return False
    try:
        x = float(inner["x"]) + float(inner["width"]) / 2
        y = float(inner["y"]) + float(inner["height"]) / 2
        px, py = float(outer["x"]), float(outer["y"])
        pw, ph = float(outer["width"]), float(outer["height"])
    except (KeyError, TypeError, ValueError):
        return False
    margin_x, margin_y = pw * 0.1, ph * 0.1
    return px - margin_x <= x <= px + pw + margin_x and py - margin_y <= y <= py + ph + margin_y


def _hands_near_face(faces: list[dict[str, Any]], hands: list[dict[str, Any]]) -> bool:
    if not faces:
        return False
    face = faces[0].get("bbox", {})
    fx = float(face.get("x", 0))
    fy = float(face.get("y", 0))
    fw = float(face.get("width", 0))
    fh = float(face.get("height", 0))
    expanded = {"x": fx - fw * 0.8, "y": fy - fh * 0.45, "width": fw * 2.6, "height": fh * 1.9}
    for hand in hands:
        bbox = hand.get("bbox", {})
        x = float(bbox.get("x", 0)) + float(bbox.get("width", 0)) / 2
        y = float(bbox.get("y", 0)) + float(bbox.get("height", 0)) / 2
        if expanded["x"] <= x <= expanded["x"] + expanded["width"] and expanded["y"] <= y <= expanded["y"] + expanded["height"]:
            return True
    return False


def _analysis_payload(bundle: ObservationBundle, health: dict[str, str]) -> dict[str, Any]:
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
            {key: face[key] for key in ("bbox", "yaw", "pitch", "roll", "quality", "pose_valid", "calibration_status") if key in face}
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


def _metrics_snapshot(
    metrics: PipelineMetrics,
    queue: LatestFrameQueue,
    analyzer_health: dict[str, str] | None,
) -> dict[str, Any]:
    snapshot = dict(metrics.snapshot())
    snapshot["dropped_frames"] = queue.dropped_frames
    snapshot["queue_depth"] = len(queue)
    snapshot["analyzer_health"] = dict(analyzer_health or {})
    if analyzer_health and any(value not in {"ready", "no_detection", "skipped"} for value in analyzer_health.values()):
        snapshot["realtime_status"] = "DEGRADED"
    return snapshot


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


__all__ = ["DesktopRuntime", "LatestPreview"]
