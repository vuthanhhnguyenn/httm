"""Confirm class-aware object tracks using fresh detections only.

Lost boxes are never returned as observations or passed to the rule engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    width = max(0.0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
    height = max(0.0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    intersection = width * height
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class _Track:
    label: str
    bbox: dict[str, float]
    last_ms: int
    high_hits: int = 0
    last_high_ms: int | None = None


class ObjectEvidenceTracker:
    def __init__(self, *, thresholds: dict[str, float] | None = None, max_gap_ms: int = 600) -> None:
        self.thresholds = thresholds or {"cell phone": 0.55, "book": 0.7}
        self.max_gap_ms = max_gap_ms
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(self, detections: list[dict[str, Any]], timestamp_ms: int) -> list[dict[str, Any]]:
        self._tracks = {key: track for key, track in self._tracks.items()
                        if 0 <= timestamp_ms - track.last_ms <= self.max_gap_ms}
        available = set(self._tracks)
        output = []
        for item in sorted(detections, key=lambda d: float(d.get("confidence", 0)), reverse=True):
            label, bbox = str(item.get("class", "")).lower(), item.get("bbox")
            if label not in self.thresholds or not isinstance(bbox, dict):
                output.append(item)
                continue
            candidates = [(box_iou(bbox, self._tracks[key].bbox), key)
                          for key in available if self._tracks[key].label == label]
            overlap, key = max(candidates, default=(0.0, -1))
            if overlap < 0.15:
                key, self._next_id = self._next_id, self._next_id + 1
                self._tracks[key] = _Track(label, dict(bbox), timestamp_ms)
            else:
                available.remove(key)
            track = self._tracks[key]
            track.last_ms, track.bbox = timestamp_ms, dict(bbox)
            if track.last_high_ms is not None and timestamp_ms - track.last_high_ms > 1500:
                track.high_hits = 0
            if float(item.get("confidence", 0)) >= self.thresholds[label]:
                track.high_hits += 1
                track.last_high_ms = timestamp_ms
            output.append({**item, "object_track_id": key, "track_confirmed": track.high_hits >= 2})
        return output
