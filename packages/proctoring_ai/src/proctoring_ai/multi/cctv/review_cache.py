"""What the deep tier's models saw in each suspect window, kept so the rules can be replayed without them.

``save_window`` appends one window (its analysed frames, already in full-frame coordinates) to a
gzipped JSON-lines file; ``load_cache`` reads them all back keyed by ``(seat_id, start_ms, end_ms)``.
Numbers are rounded to 4 decimals: far below a pixel at 1280x720, and it keeps test fixtures small.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from proctoring_core.observations import Box

from .review_detectors import ReviewObject
from .review_rules import ReviewFrame
from .suspects import SuspectWindow

CacheKey = tuple[str, int, int]


def _r(value: float) -> float:
    return round(float(value), 4)


def _box(box: Box) -> list[float]:
    return [_r(box.x), _r(box.y), _r(box.width), _r(box.height)]


def _frame_json(frame: ReviewFrame) -> dict[str, Any]:
    return {"t": frame.offset_ms, "aspect": _r(frame.frame_aspect),
            "seats": {str(key): _box(box) for key, box in frame.seats.items()},
            "poses": {str(key): [[_r(v) for v in p] for p in kps] for key, kps in frame.poses.items()},
            "objects": [[obj.label, _r(obj.confidence), *_box(obj.bbox), obj.source] for obj in frame.objects],
            "people": [_person_json(person) for person in frame.people]}


def _person_json(person: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"keypoints": [[_r(v) for v in p] for p in person["keypoints"]]}
    if (box := Box.parse(person.get("bbox"))) is not None:
        out["bbox"] = {key: _r(value) for key, value in box.as_dict().items()}
    return out


def _frame_from(item: dict[str, Any]) -> ReviewFrame:
    return ReviewFrame(int(item["t"]), {UUID(key): Box(*value) for key, value in item["seats"].items()},
                       {UUID(key): [list(p) for p in kps] for key, kps in item["poses"].items()},
                       [ReviewObject(o[0], o[1], Box(o[2], o[3], o[4], o[5]), o[6]) for o in item["objects"]],
                       float(item["aspect"]), list(item.get("people", ())))


def window_key(window: SuspectWindow) -> CacheKey:
    return str(window.seat_id), window.start_ms, window.end_ms


def save_window(path: Path, window: SuspectWindow, frames: Sequence[ReviewFrame]) -> None:
    line = json.dumps({"key": list(window_key(window)), "frames": [_frame_json(f) for f in frames]})
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_cache(path: Path) -> dict[CacheKey, list[ReviewFrame]]:
    out: dict[CacheKey, list[ReviewFrame]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                seat, start, end = item["key"]
                out[(str(seat), int(start), int(end))] = [_frame_from(f) for f in item["frames"]]
    return out


__all__ = ["CacheKey", "load_cache", "save_window", "window_key"]
