"""Brief display-only continuity; held boxes never enter behavior evaluation."""

from __future__ import annotations

from typing import Any

from proctoring_ai.tracking.object_tracks import box_iou


class OverlayState:
    def __init__(self, *, max_age_ms: int = 800, hold_ms: int = 350) -> None:
        self.max_age_ms = max_age_ms
        self.hold_ms = hold_ms
        self.analysis: dict[str, Any] | None = None

    def update(self, result: dict[str, Any]) -> None:
        timestamp = int(result["captured_at_monotonic_ms"])
        previous = self.analysis or {}
        merged = dict(result)
        for group in ("objects", "faces"):
            old = list(previous.get(group, []))
            fresh = []
            for item in result.get(group, []):
                current = {**item, "observed_at_ms": timestamp, "held": False}
                candidates = [(box_iou(item["bbox"], candidate["bbox"]), index)
                              for index, candidate in enumerate(old)
                              if item.get("class") == candidate.get("class")
                              and timestamp - candidate["observed_at_ms"] <= self.max_age_ms]
                overlap, index = max(candidates, default=(0.0, -1))
                if overlap >= 0.2:
                    matched = old.pop(index)
                    current["bbox"] = {key: 0.75 * value + 0.25 * matched["bbox"][key]
                                       for key, value in item["bbox"].items()}
                fresh.append(current)
            fresh.extend({**item, "held": True} for item in old
                         if 0 <= timestamp - item["observed_at_ms"] <= self.hold_ms)
            merged[group] = fresh
        for group, analyzer in (("poses", "pose"), ("hands", "hands")):
            if result.get("analyzer_health", {}).get(analyzer) == "skipped":
                merged[group] = [{**item, "held": True} for item in previous.get(group, [])
                                 if 0 <= timestamp - item["observed_at_ms"] <= self.hold_ms]
            else:
                merged[group] = [{**item, "observed_at_ms": timestamp, "held": False}
                                 for item in result.get(group, [])]
        self.analysis = merged

    def snapshot(self, camera_timestamp_ms: int) -> dict[str, Any] | None:
        if self.analysis is None:
            return None
        age = camera_timestamp_ms - int(self.analysis["captured_at_monotonic_ms"])
        if not 0 <= age <= self.max_age_ms:
            return None
        result = dict(self.analysis)
        for group in ("objects", "faces", "poses", "hands"):
            result[group] = [item for item in result.get(group, [])
                             if 0 <= camera_timestamp_ms - item["observed_at_ms"] <= (
                                 self.hold_ms if group in {"poses", "hands"} else self.max_age_ms)]
        return result
