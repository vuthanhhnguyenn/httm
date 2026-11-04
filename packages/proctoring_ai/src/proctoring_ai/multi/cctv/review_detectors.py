"""Pretrained models of the deep tier, listed in ``configs/review_models.yaml``.

Any Ultralytics-loadable detector (COCO, YOLOE with saved text classes, or a cheating detector someone
trained and published) plugs in with one YAML entry: its path, SHA-256 and a map from its class names
to ours (``phone``, ``paper``, ``book``). A model that is missing or fails its checksum is skipped
with the reason kept in ``status``; the others still run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from proctoring_core.observations import Box

from .crops import CropTransform

LABELS = ("phone", "paper", "book")


@dataclass(frozen=True, slots=True)
class ReviewObject:
    label: str
    confidence: float
    bbox: Box  # normalised to the full frame
    source: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    kind: str  # "pose" | "objects"
    path: Path
    class_map: Mapping[str, str]
    min_conf: float
    sha256: str | None


def load_specs(config_path: Path, models_root: Path) -> list[ModelSpec]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    specs: list[ModelSpec] = []
    for item in payload.get("models", []):
        if not item.get("enabled", True):
            continue
        class_map = {str(key): str(value) for key, value in (item.get("class_map") or {}).items()}
        if any(value not in LABELS for value in class_map.values()):
            raise ValueError(f"{item['name']}: class_map chỉ được đổi về {LABELS}")
        specs.append(ModelSpec(str(item["name"]), str(item["kind"]), models_root / str(item["path"]), class_map,
                               float(item.get("min_conf", 0.25)), item.get("sha256")))
    return specs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iou(a: Box, b: Box) -> float:
    w = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
    h = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    inter = max(0.0, w) * max(0.0, h)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _default_loader(path: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(path))


class ReviewDetectors:
    def __init__(self, specs: list[ModelSpec], device: str | None = None,
                 loader: Callable[[Path], Any] | None = None, image_size: int = 960) -> None:
        self.device, self.image_size = device, image_size
        self.status: dict[str, str] = {}
        self._models: list[tuple[ModelSpec, Any]] = []
        load = loader or _default_loader
        for spec in specs:
            if not spec.path.is_file():
                self.status[spec.name] = f"bỏ qua: không có file {spec.path}"
                continue
            if spec.sha256 and _sha256(spec.path) != str(spec.sha256).lower():
                self.status[spec.name] = "bỏ qua: SHA-256 không khớp cấu hình"
                continue
            try:
                self._models.append((spec, load(spec.path)))
            except Exception as exc:  # a broken third-party model must not stop the review
                self.status[spec.name] = f"bỏ qua: không nạp được ({exc})"
                continue
            self.status[spec.name] = "ok"

    def _predict(self, model: Any, image: Any, conf: float) -> list[Any]:
        options: dict[str, Any] = {"source": image, "conf": conf, "verbose": False, "imgsz": self.image_size}
        if self.device is not None:
            options["device"] = self.device
        return list(model.predict(**options))

    def pose(self, crop: Any, transform: CropTransform) -> list[dict[str, Any]]:
        people: list[dict[str, Any]] = []
        for spec, model in self._models:
            if spec.kind != "pose":
                continue
            for result in self._predict(model, crop, spec.min_conf):
                rows = result.boxes.data.detach().cpu().tolist()
                points = result.keypoints.data.detach().cpu().tolist() if result.keypoints is not None else []
                for row, kps in zip(rows, points, strict=False):
                    (x0, y0), (x1, y1) = transform.to_frame(row[0], row[1]), transform.to_frame(row[2], row[3])
                    people.append({"class": "person", "confidence": float(row[4]),
                                   "bbox": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
                                   "keypoints": [[*transform.to_frame(p[0], p[1]), float(p[2]) if len(p) > 2 else 1.0]
                                                 for p in kps]})
        return people

    def objects(self, crop: Any, transform: CropTransform) -> list[ReviewObject]:
        found: list[ReviewObject] = []
        for spec, model in self._models:
            if spec.kind != "objects":
                continue
            for result in self._predict(model, crop, spec.min_conf):
                names = result.names
                for row in result.boxes.data.detach().cpu().tolist():
                    label = spec.class_map.get(str(names[int(row[5])]))
                    if label is None or float(row[4]) < spec.min_conf:
                        continue
                    (x0, y0), (x1, y1) = transform.to_frame(row[0], row[1]), transform.to_frame(row[2], row[3])
                    found.append(ReviewObject(label, float(row[4]), Box(x0, y0, x1 - x0, y1 - y0), spec.name))
        kept: list[ReviewObject] = []
        for item in sorted(found, key=lambda obj: -obj.confidence):
            if not any(other.label == item.label and _iou(other.bbox, item.bbox) > 0.5 for other in kept):
                kept.append(item)
        return kept


__all__ = ["LABELS", "ModelSpec", "ReviewDetectors", "ReviewObject", "load_specs"]
