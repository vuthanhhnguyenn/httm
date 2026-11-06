"""Phase 3 tiers 3–4 at run time: a trained model's probabilities become red/yellow events.

``LearnedCore.load`` reads ``model.txt`` (LightGBM) and ``meta.json`` (feature names, classes, red/yellow
thresholds) written by ``scripts/train_core.py``. ``events`` smooths each seat's per-window probability of a
class with two thresholds (open after two windows at ``p_open``, close below ``p_close``) and writes records
of the same shape as the deep-tier rules, so dedupe, the video and the report need no change.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from proctoring_core.observations import Box

from .features import FEATURE_NAMES, STEP_MS, FeatureRow

PAIR_TYPES = ("PASSING_OBJECT", "HEADS_TOGETHER")
P_OPEN, P_CLOSE = 0.5, 0.35
DOMAIN_FACTOR = 2.0


def domain_warning(meta: Mapping[str, Any], frame_size: tuple[int, int] | None,
                   seat_height_median: float | None) -> str | None:
    """A Vietnamese warning when this video's frame or seats are more than twice as big or small as what the
    model was trained on; ``None`` when they are close or nothing is known to compare."""

    def off(now: float | None, then: float | None) -> bool:
        return bool(now and then) and max(now / then, then / now) > DOMAIN_FACTOR  # type: ignore[operator]

    trained = meta.get("frame_size")
    if frame_size and trained and off(float(frame_size[0]), float(trained[0])):
        return (f"Cảnh báo: video {frame_size[0]}×{frame_size[1]} khác xa cỡ khung lúc train "
                f"({trained[0]}×{trained[1]}); kết quả lõi học có thể kém tin cậy.")
    if off(seat_height_median, meta.get("seat_height_median")):
        return (f"Cảnh báo: ghế trong video này cao {seat_height_median:.2f} khung, khác xa lúc train "
                f"({meta['seat_height_median']:.2f}); kết quả lõi học có thể kém tin cậy.")
    return None


class LearnedCore:
    def __init__(self, booster: Any, classes: Sequence[str], thresholds: Mapping[str, Sequence[float]],
                 meta: Mapping[str, Any]) -> None:
        self.booster, self.classes = booster, tuple(classes)
        self.thresholds = {name: (float(value[0]), float(value[1])) for name, value in thresholds.items()}
        self.meta = dict(meta)

    @classmethod
    def load(cls, folder: Path) -> LearnedCore:
        model, meta_path = folder / "model.txt", folder / "meta.json"
        if not model.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Thư mục model {folder} thiếu model.txt hoặc meta.json "
                                    "(train bằng scripts/train_core.py).")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        names = list(meta.get("feature_names") or [])
        if names != list(FEATURE_NAMES):
            missing = [name for name in FEATURE_NAMES if name not in names]
            extra = [name for name in names if name not in FEATURE_NAMES]
            raise ValueError(f"Model {folder} train với bộ đặc trưng khác bản code này "
                             f"(thiếu: {', '.join(missing) or 'không'}; thừa: {', '.join(extra) or 'không'}); "
                             "train lại model.")
        import lightgbm as lgb

        return cls(lgb.Booster(model_file=str(model)), meta["classes"], meta.get("thresholds", {}), meta)

    def predict(self, rows: Sequence[FeatureRow]) -> list[dict[str, float]]:
        if not rows:
            return []
        import numpy as np

        matrix = np.array([[row.values[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
        return [{name: float(p[i]) for i, name in enumerate(self.classes)} for p in self.booster.predict(matrix)]

    def events(self, rows: Sequence[FeatureRow], probs: Sequence[Mapping[str, float]], *, p_open: float = P_OPEN,
               p_close: float = P_CLOSE) -> list[dict[str, Any]]:
        boxes: dict[str, Box] = {str(row.seat_id): row.seat_box for row in rows}
        series: dict[str, list[tuple[FeatureRow, Mapping[str, float]]]] = {}
        for row, p in zip(rows, probs, strict=True):
            series.setdefault(str(row.seat_id), []).append((row, p))
        found: list[dict[str, Any]] = []  # owner, partner, type, start, end, p
        for seat, items in series.items():
            items.sort(key=lambda item: item[0].start_ms)
            for kind in self.classes:
                if kind == "NORMAL" or kind not in self.thresholds:
                    continue
                run: list[tuple[FeatureRow, float]] = []
                open_ = False
                for i, (row, p) in enumerate(items):
                    value = p.get(kind, 0.0)
                    if open_ and row.start_ms - items[i - 1][0].start_ms > STEP_MS:
                        found.append(self._event(seat, kind, run))  # the seat was not seen in between
                        open_, run = False, []
                    if not open_:
                        follows = i + 1 < len(items) and items[i + 1][0].start_ms - row.start_ms <= STEP_MS
                        nxt = items[i + 1][1].get(kind, 0.0) if follows else 0.0
                        if value >= p_open and nxt >= p_open:
                            open_, run = True, [(row, value)]
                    elif value >= p_close:
                        run.append((row, value))
                    else:
                        found.append(self._event(seat, kind, run))
                        open_, run = False, []
                if open_ and run:
                    found.append(self._event(seat, kind, run))
        records: list[dict[str, Any]] = []
        for event in self._merge(found):
            red, yellow = self.thresholds[event["type"]]
            if event["p"] >= red:
                level = "red"
            elif event["p"] >= yellow:
                level = "yellow"
            else:
                continue
            box = boxes.get(event["owner"])
            head = [round(box.x + box.width / 2, 3), round(box.y + 0.12 * box.height, 3)] if box else None
            base = {"id": str(uuid4()), "type": event["type"], "seat_id": event["owner"], "scene": "cctv",
                    "stage": "learned", "runtime_track_id": None, "reasons": ["LEARNED"],
                    **({"neighbor_seat_id": event["partner"]} if event["partner"] else {})}
            records.append({"action": "opened", "video_offset_ms": event["start"], "probability": round(event["p"], 3),
                            "level": level, "level_reason": "learned", **({"head": head} if head else {}), **base})
            records.append({"action": "closed", "video_offset_ms": event["end"],
                            "duration_ms": event["end"] - event["start"], **base})
        records.sort(key=lambda item: (item["video_offset_ms"], item["action"] != "opened"))
        return records

    @staticmethod
    def _event(seat: str, kind: str, run: Sequence[tuple[FeatureRow, float]]) -> dict[str, Any]:
        owner, partner = seat, None
        if kind in PAIR_TYPES:
            votes: dict[str, int] = {}
            for row, _p in run:
                if row.reacher:
                    votes[row.reacher] = votes.get(row.reacher, 0) + 1
            partner = next((row.partner for row, _p in run if row.partner), None)
            if votes:
                owner = max(votes, key=lambda key: votes[key])
                if owner != seat:
                    partner = seat
        return {"owner": owner, "partner": partner, "type": kind, "start": run[0][0].start_ms,
                "end": run[-1][0].end_ms, "p": max(p for _row, p in run)}

    @staticmethod
    def _merge(found: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """One event when the same pair event came from both seats' windows (overlapping in time), whichever
        seat each side named as the owner; the owner of the more probable side is kept."""

        def pair(event: dict[str, Any]) -> tuple[str, ...]:
            if event["type"] in PAIR_TYPES and event["partner"]:
                return tuple(sorted((event["owner"], event["partner"])))
            return (event["owner"], event["partner"] or "")

        out: list[dict[str, Any]] = []
        for event in sorted(found, key=lambda item: (item["type"], pair(item), item["start"])):
            last = out[-1] if out else None
            if (last is not None and last["type"] == event["type"] and pair(last) == pair(event)
                    and event["start"] <= last["end"]):
                if event["p"] > last["p"]:
                    last["owner"], last["partner"] = event["owner"], event["partner"]
                last["end"] = max(last["end"], event["end"])
                last["p"] = max(last["p"], event["p"])
            else:
                out.append(dict(event))
        return out


__all__ = ["P_CLOSE", "P_OPEN", "PAIR_TYPES", "LearnedCore", "domain_warning"]
