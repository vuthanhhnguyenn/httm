"""YOLO-pose adapter: person boxes plus the 17 COCO keypoints, normalised to the frame."""

from __future__ import annotations

from typing import Any

from ...detectors.yolo_detector import YoloObjectDetector

# COCO keypoint order used by YOLO-pose.
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST = 5, 6, 9, 10


def parse_pose_result(result: Any, frame_size: tuple[int, int]) -> list[dict[str, Any]]:
    """Turn one Ultralytics pose result into person observations with ``keypoints`` [[x, y, conf]] * 17."""

    width, height = frame_size
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None:
        return []
    rows = boxes.data.detach().cpu().tolist()
    points = keypoints.data.detach().cpu().tolist()
    people: list[dict[str, Any]] = []
    for row, kps in zip(rows, points, strict=False):
        x0, y0 = max(0.0, row[0] / width), max(0.0, row[1] / height)
        people.append({
            "class": "person",
            "confidence": float(row[4]),
            "bbox": {"x": x0, "y": y0, "width": max(0.0, min(1.0 - x0, (row[2] - row[0]) / width)),
                     "height": max(0.0, min(1.0 - y0, (row[3] - row[1]) / height))},
            "keypoints": [[float(p[0]) / width, float(p[1]) / height, float(p[2]) if len(p) > 2 else 1.0]
                          for p in kps],
        })
    return people


class YoloPoseDetector(YoloObjectDetector):
    def __init__(self, model_path: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allowed_classes", ["person"])
        kwargs.setdefault("image_size", 960)
        super().__init__(model_path, **kwargs)

    def _detect_sync(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        self._load()
        options: dict[str, Any] = {"source": frame["image"], "verbose": False, "conf": self.confidence_threshold,
                                   "imgsz": self.image_size, "classes": [0]}
        if self.device is not None:
            options["device"] = self.device
        people: list[dict[str, Any]] = []
        for result in self._model.predict(**options):
            people += parse_pose_result(result, frame["frame_size"])
        return people


__all__ = ["YoloPoseDetector", "parse_pose_result"]
