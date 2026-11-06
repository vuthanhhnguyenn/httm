"""Phase 3 labels: what a person (or Claude) said a seat was doing over a span of video.

A label names the seat by its box, not its id: ids change from one run of the fast tier to the next, the
place of a seat does not. ``match`` gives a feature row the label whose box overlaps its seat box by
``min_iou`` and whose span holds the middle of the row's 3 s window.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

from proctoring_core.observations import Box

from ..tracker import iou
from .features import RULE_TYPES, FeatureRow

LABELS = ("NORMAL", *RULE_TYPES, "UNSURE")
OUTLIER_FEATURES = ("yaw_dev_max", "reach_out_max", "phone_conf_max", "paper_conf_max", "wrist_below_desk_share")


@dataclass(frozen=True, slots=True)
class Label:
    video: str
    seat_box: Box
    start_ms: int
    end_ms: int
    label: str
    labeler: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class Segment:
    """A span of one seat to label. ``why``: ``rule`` (an old rule fired), ``outlier`` (a feature in the
    seat's own top 5 %) or ``quiet`` (a random calm window, for negatives)."""

    video: str
    seat_id: UUID
    seat_box: Box
    start_ms: int
    end_ms: int
    why: str


def _box(box: Box) -> list[float]:
    return [round(box.x, 6), round(box.y, 6), round(box.width, 6), round(box.height, 6)]


def save_labels(path: Path, labels: Sequence[Label]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for label in labels:
            item = asdict(label)
            item["seat_box"] = _box(label.seat_box)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_labels(path: Path) -> list[Label]:
    out = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if item["label"] not in LABELS:
            raise ValueError(f"{path}:{number}: nhãn {item['label']} không có trong {', '.join(LABELS)}")
        out.append(Label(item["video"], Box(*item["seat_box"]), int(item["start_ms"]), int(item["end_ms"]),
                         item["label"], item.get("labeler", ""), item.get("note", "")))
    return out


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0


def select_segments(rows: Sequence[FeatureRow], *, quiet_fraction: float = 0.2, seed: int = 0,
                    outlier_q: float = 0.95) -> list[Segment]:
    by_seat: dict[tuple[str, UUID], list[FeatureRow]] = {}
    for row in rows:
        by_seat.setdefault((row.video, row.seat_id), []).append(row)
    out: list[Segment] = []
    quiet: list[FeatureRow] = []
    for (_video, _seat), seat_rows in sorted(by_seat.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        seat_rows.sort(key=lambda row: row.start_ms)
        cut = {name: _quantile([row.values[name] for row in seat_rows], outlier_q) for name in OUTLIER_FEATURES}
        current: Segment | None = None
        for row in seat_rows:
            rule = any(row.values[f"rule_{name}"] > 0 for name in RULE_TYPES)
            outlier = any(row.values[name] > 0 and row.values[name] >= cut[name] for name in OUTLIER_FEATURES)
            if not (rule or outlier):
                quiet.append(row)
                continue
            why = "rule" if rule else "outlier"
            if current is not None and row.start_ms <= current.end_ms:
                current = Segment(current.video, current.seat_id, current.seat_box, current.start_ms,
                                  max(current.end_ms, row.end_ms), "rule" if "rule" in (why, current.why) else why)
                out[-1] = current
            else:
                current = Segment(row.video, row.seat_id, row.seat_box, row.start_ms, row.end_ms, why)
                out.append(current)
    picked = random.Random(seed).sample(quiet, round(quiet_fraction * len(quiet))) if quiet_fraction > 0 else []
    out += [Segment(row.video, row.seat_id, row.seat_box, row.start_ms, row.end_ms, "quiet") for row in picked]
    return out


def match(rows: Sequence[FeatureRow], labels: Sequence[Label], *, min_iou: float = 0.5
          ) -> tuple[list[tuple[FeatureRow, str]], int]:
    """Rows with a label (``UNSURE`` dropped), and how many labels found no row at all."""

    pairs: list[tuple[FeatureRow, str]] = []
    used: set[int] = set()
    for row in rows:
        middle = (row.start_ms + row.end_ms) / 2
        for index, label in enumerate(labels):
            if (label.video == row.video and label.start_ms <= middle <= label.end_ms
                    and iou(label.seat_box, row.seat_box) >= min_iou):
                used.add(index)
                if label.label != "UNSURE":
                    pairs.append((row, label.label))
                break
    unmatched = sum(1 for index, label in enumerate(labels) if index not in used and label.label != "UNSURE")
    return pairs, unmatched


__all__ = ["LABELS", "Label", "Segment", "load_labels", "match", "save_labels", "select_segments"]
