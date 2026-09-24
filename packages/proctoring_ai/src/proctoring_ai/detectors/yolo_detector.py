"""Lazy Ultralytics adapter that returns observations, never business events."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from threading import Lock
from typing import Any

from ..pipeline.single_flight import run_single_flight

try:
    from ultralytics import YOLO  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None


class YoloObjectDetector:
    def __init__(
        self,
        model_path: str | Path,
        *,
        model_version: str = "unknown",
        model_sha256: str | None = None,
        confidence_threshold: float = 0.25,
        image_size: int = 640,
        device: str | None = None,
        allowed_classes: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_version = model_version
        self.model_sha256 = model_sha256.lower() if model_sha256 else self._hash_file()
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device
        self.allowed_classes = (
            tuple(str(name).casefold() for name in allowed_classes) if allowed_classes else None
        )
        self._model: Any = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    def _hash_file(self) -> str | None:
        if not self.model_path.is_file():
            return None
        digest = hashlib.sha256()
        with self.model_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @property
    def active_device(self) -> str:
        predictor = getattr(self._model, "predictor", None)
        device = getattr(predictor, "device", None)
        return str(device) if device is not None else (self.device or "auto")

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if YOLO is None:
                raise RuntimeError("ultralytics is not installed")
            if not self.model_path.is_file():
                raise FileNotFoundError(self.model_path)
            actual_hash = self._hash_file()
            if self.model_sha256 and actual_hash != self.model_sha256:
                raise ValueError("model SHA-256 does not match the reviewed manifest")
            self._model = YOLO(str(self.model_path))

    async def initialize(self) -> None:
        """Load and validate the local model without blocking its async runtime."""

        await asyncio.to_thread(self._warmup)

    def _warmup(self) -> None:
        self._load()
        import numpy as np  # type: ignore[import-not-found]

        blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        options: dict[str, Any] = {
            "source": blank,
            "verbose": False,
            "conf": self.confidence_threshold,
            "imgsz": self.image_size,
        }
        class_ids = self._allowed_class_ids()
        if class_ids is not None:
            options["classes"] = class_ids
        if self.device is not None:
            options["device"] = self.device
        self._model.predict(**options)

    async def detect(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        return await run_single_flight(self._inference_lock, self._detect_sync, frame)

    def _allowed_class_ids(self) -> list[int] | None:
        if self.allowed_classes is None:
            return None
        self._load()
        names = self._model.names
        name_to_id = {
            str(name).casefold(): int(class_id)
            for class_id, name in (names.items() if isinstance(names, dict) else enumerate(names))
        }
        missing = set(self.allowed_classes) - name_to_id.keys()
        if missing:
            raise ValueError(f"configured detector classes are absent from model: {sorted(missing)}")
        return [name_to_id[name] for name in self.allowed_classes]

    def _detect_sync(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        self._load()
        predict_options: dict[str, Any] = {
            "source": frame["image"],
            "verbose": False,
            "conf": self.confidence_threshold,
            "imgsz": self.image_size,
        }
        class_ids = self._allowed_class_ids()
        if class_ids is not None:
            predict_options["classes"] = class_ids
        if self.device is not None:
            predict_options["device"] = self.device
        results = self._model.predict(**predict_options)
        observations: list[dict[str, Any]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", {})
            rows = boxes.data.detach().cpu().tolist()
            for row in rows:
                xyxy = row[:4]
                confidence = float(row[4])
                class_id = int(row[5])
                width, height = frame["frame_size"]
                normalized_x = max(0.0, min(1.0, xyxy[0] / width))
                normalized_y = max(0.0, min(1.0, xyxy[1] / height))
                normalized_w = max(0.0, min(1.0 - normalized_x, (xyxy[2] - xyxy[0]) / width))
                normalized_h = max(0.0, min(1.0 - normalized_y, (xyxy[3] - xyxy[1]) / height))
                label = str(names.get(class_id, class_id))
                if self.allowed_classes is not None and label.casefold() not in self.allowed_classes:
                    continue
                observations.append(
                    {
                        "class": label,
                        "class_id": class_id,
                        "confidence": confidence,
                        "bbox": {
                            "x": normalized_x,
                            "y": normalized_y,
                            "width": normalized_w,
                            "height": normalized_h,
                        },
                        "model_version": self.model_version,
                        "model_sha256": self.model_sha256,
                    }
                )
        return observations

    def close(self) -> None:
        with self._inference_lock:
            self._model = None
