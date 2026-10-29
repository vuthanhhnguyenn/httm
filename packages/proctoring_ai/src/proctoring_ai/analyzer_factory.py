"""Load checksum-reviewed local models and compose the observation pipeline."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from .face.landmarks import MediaPipeFaceLandmarker
from .hands.hand_estimator import MediaPipeHandEstimator
from .multi.crop import SubjectCropper
from .multi.face_pool import PerSubjectFaceAnalyzer
from .multi.focus import FocusScheduler
from .multi.focus_analyzers import FocusAnalyzers
from .multi.motion import MotionGate
from .multi.pipeline import MultiSubjectPipeline
from .multi.tracker import MultiSubjectTracker
from .pipeline.observation_pipeline import ObservationPipeline
from .pose.pose_estimator import MediaPipePoseEstimator
from .tracking.object_tracks import ObjectEvidenceTracker
from .tracking.single_tracker import SingleSubjectTracker


class _UnavailableDetector:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def detect(self, _frame: dict[str, Any]) -> list[dict[str, Any]]:
        raise RuntimeError(self.reason)


class _UnavailableFace:
    """Stand-in per-seat face analyzer when the face model failed its checks."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def initialize(self) -> dict[str, Any]:
        return {"status": "unavailable", "reason": self.reason}

    async def analyze(self, _frame: dict[str, Any]) -> dict[str, Any]:
        return {"status": "not_run", "reason": self.reason, "faces": []}

    def auto_calibrate(self) -> str:
        return "uncalibrated"

    def close(self) -> None:
        return None


class AnalyzerFactory:
    """Own process-wide model instances and create per-session trackers/pipelines."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        detector_image_size: int = 416,
        detector_device: str | None = None,
        analyzer_options: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        # Per-analyzer constructor options keyed by "face", "pose", "hands" (thresholds, timeouts).
        self.analyzer_options = analyzer_options or {}
        self.manifest_path = Path(manifest_path).resolve()
        self.detector_image_size = detector_image_size
        self.detector_device = detector_device
        self.detector: Any | None = None
        self.face_analyzer: MediaPipeFaceLandmarker | None = None
        self.pose_analyzer: MediaPipePoseEstimator | None = None
        self.hand_analyzer: MediaPipeHandEstimator | None = None
        # Checksum-verified model files, so multi-subject sessions can open one instance per seat.
        self._model_paths: dict[str, Path] = {}
        self._status = {
            "object_detector": ("unavailable", "MODEL_NOT_CONFIGURED"),
            "face": ("unavailable", "FACE_MODEL_NOT_CONFIGURED"),
            "pose": ("unavailable", "POSE_MODEL_NOT_CONFIGURED"),
            "hands": ("unavailable", "HAND_MODEL_NOT_CONFIGURED"),
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _entries(self) -> list[dict[str, Any]]:
        payload = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8")) or {}
        entries = payload.get("models", [])
        if not isinstance(entries, list):
            raise ValueError("manifest.models must be a list")
        return [item for item in entries if isinstance(item, dict)]

    def _resolve_model(self, task: str, *, licenses: set[str]) -> tuple[Path, dict[str, Any]]:
        entry = next((item for item in self._entries() if item.get("task") == task), None)
        if entry is None:
            raise ValueError(f"no model configured for task {task}")
        if entry.get("license") not in licenses:
            raise ValueError(f"model license is not approved for task {task}")
        source = entry.get("source")
        expected_hash = str(entry.get("sha256", "")).lower()
        if not isinstance(source, str) or len(expected_hash) != 64:
            raise ValueError(f"model source and SHA-256 are required for task {task}")
        model_path = (self.manifest_path.parent / source).resolve()
        if not model_path.is_relative_to(self.manifest_path.parent):
            raise ValueError("model source must remain inside the model directory")
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if self._sha256(model_path) != expected_hash:
            raise ValueError(f"model SHA-256 does not match manifest for task {task}")
        return model_path, entry

    async def initialize(self, on_progress: Callable[[dict[str, dict[str, str]]], None] | None = None) -> dict[str, str]:
        for key, task, adapter in (
            ("object_detector", "detect", None),
            ("face", "face_landmarker", MediaPipeFaceLandmarker),
            ("pose", "pose_landmarker", MediaPipePoseEstimator),
            ("hands", "hand_landmarker", MediaPipeHandEstimator),
        ):
            self._status[key] = ("loading", "")
            if on_progress:
                on_progress(self.health_details())
            if adapter is None:
                await self._initialize_detector()
            else:
                await self._initialize_landmarker(task, key, adapter)
            if on_progress:
                on_progress(self.health_details())
        return self.health()

    async def _initialize_detector(self) -> None:
        try:
            model_path, entry = await asyncio.to_thread(
                self._resolve_model,
                "detect",
                licenses={"AGPL-3.0-only", "Enterprise"},
            )
            class_map = entry.get("class_map")
            if not isinstance(class_map, dict) or not class_map:
                raise ValueError("detector manifest must list the allowed model classes")
            from .detectors.yolo_detector import YoloObjectDetector

            self.detector = YoloObjectDetector(
                model_path,
                model_version=str(entry.get("version", "unknown")),
                model_sha256=str(entry["sha256"]),
                confidence_threshold=0.25,
                image_size=self.detector_image_size,
                device=self.detector_device,
                allowed_classes=[str(name) for name in class_map.values()],
            )
            await self.detector.initialize()
            self._status["object_detector"] = ("ready", "")
        except Exception as exc:
            self.detector = None
            self._status["object_detector"] = ("unavailable", f"{type(exc).__name__}:{exc}")

    async def _initialize_landmarker(self, task: str, key: str, adapter_type: Any) -> None:
        try:
            model_path, _ = await asyncio.to_thread(
                self._resolve_model,
                task,
                licenses={"Apache-2.0"},
            )
            adapter = adapter_type(model_path, **self.analyzer_options.get(key, {}))
            result = await adapter.initialize()
            status = str(result.get("status", "unavailable"))
            reason = str(result.get("reason") or "")
            attribute = {"face": "face_analyzer", "pose": "pose_analyzer", "hands": "hand_analyzer"}[key]
            setattr(self, attribute, adapter)
            if status == "ready":
                self._model_paths[key] = model_path
            self._status[key] = (status, reason)
        except Exception as exc:
            self._status[key] = ("unavailable", f"{type(exc).__name__}:{exc}")

    def health(self) -> dict[str, str]:
        return {name: status for name, (status, _reason) in self._status.items()}

    def health_details(self) -> dict[str, dict[str, str]]:
        details = {
            name: {"status": status, "reason": reason}
            for name, (status, reason) in self._status.items()
        }
        if self.detector is not None:
            details["object_detector"]["device"] = self.detector.active_device
        return details

    def build_pipeline(self, config: dict[str, Any] | None = None) -> ObservationPipeline:
        settings = config or {}
        detection = settings.get("detection", {})
        confidence = float(detection.get("object_confidence_threshold", 0.25))
        image_size = int(detection.get("image_size", self.detector_image_size))
        device = detection.get("device", self.detector_device)
        detector = self.detector
        if detector is None:
            detector = _UnavailableDetector(self._status["object_detector"][1])
        else:
            detector.confidence_threshold = confidence
            detector.image_size = image_size
            detector.device = str(device) if device is not None else None
        from .features.image_quality import ImageQualityExtractor
        behavior = settings.get("behavior", {})

        return ObservationPipeline(
            detector=detector,
            tracker=SingleSubjectTracker(),
            face_analyzer=self.face_analyzer,
            pose_analyzer=self.pose_analyzer,
            hand_analyzer=self.hand_analyzer,
            image_quality=ImageQualityExtractor(),
            timeout_ms=int(detection.get("analyzer_timeout_ms", 1200)),
            pose_interval_ms=int(detection.get("pose_interval_ms", 250)),
            hands_interval_ms=int(detection.get("hands_interval_ms", 200)),
            person_confidence_threshold=float(detection.get("person_confidence_threshold", 0.6)),
            object_tracker=ObjectEvidenceTracker(thresholds={
                "cell phone": float(behavior.get("phone", {}).get("minimum_confidence", 0.55)),
                "book": float(behavior.get("document", {}).get("minimum_confidence", 0.7)),
            }),
        )

    def build_multi_pipeline(self, config: dict[str, Any] | None = None, *, max_subjects: int = 6,
                             crop_size: int = 256, yolo_every: int = 1, motion_skip: bool = True,
                             focus: bool = True) -> MultiSubjectPipeline:
        """Several seated candidates: one face landmarker per seat, hands/pose for focus seats only."""

        single = self.build_pipeline(config)
        detection = (config or {}).get("detection", {})
        paths = self._model_paths

        def landmarker_factory(key: str, adapter: Any, **extra: Any) -> Callable[[], Any] | None:
            path = paths.get(key)
            if path is None:
                return None
            options = {**self.analyzer_options.get(key, {}), **extra}
            return lambda: adapter(path, **options)

        face_factory = landmarker_factory("face", MediaPipeFaceLandmarker, suggest_after_ms=3000)
        if face_factory is None:
            reason = self._status["face"][1] or "FACE_MODEL_UNAVAILABLE"
            face_factory = lambda: _UnavailableFace(reason)  # noqa: E731
        hand_factory = landmarker_factory("hands", MediaPipeHandEstimator) if focus else None
        pose_factory = landmarker_factory("pose", MediaPipePoseEstimator) if focus else None
        return MultiSubjectPipeline(
            detector=single.detector,
            tracker=MultiSubjectTracker(max_subjects=max_subjects),
            cropper=SubjectCropper(min_side=crop_size),
            faces=PerSubjectFaceAnalyzer(face_factory, max_instances=max_subjects,
                                         motion_gate=MotionGate() if motion_skip else None),
            focus_analyzers=FocusAnalyzers(hand_factory, pose_factory,
                                           hands_interval_ms=int(detection.get("hands_interval_ms", 200)),
                                           pose_interval_ms=int(detection.get("pose_interval_ms", 250))),
            scheduler=FocusScheduler(),
            image_quality=single.image_quality,
            object_tracker=single.object_tracker,
            person_confidence_threshold=single.person_confidence_threshold,
            yolo_every=yolo_every,
            timeout_ms=single.timeout_ms,
        )

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()
        for analyzer in (self.face_analyzer, self.pose_analyzer, self.hand_analyzer):
            if analyzer is not None:
                analyzer.close()


__all__ = ["AnalyzerFactory"]
