"""Phase 3: label format, which windows to label, matching labels back to feature rows, contact sheets."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from proctoring_ai.multi.cctv.features import FEATURE_NAMES, FeatureRow
from proctoring_ai.multi.cctv.labels import (
    Label,
    Segment,
    load_labels,
    match,
    save_labels,
    select_segments,
)
from proctoring_core.observations import Box

ROOT = Path(__file__).resolve().parents[4]
A, B = uuid4(), uuid4()
BOX_A, BOX_B = Box(0.2, 0.3, 0.1, 0.3), Box(0.5, 0.3, 0.1, 0.3)


def _row(seat=A, start=0, box=BOX_A, **values) -> FeatureRow:
    vals = dict.fromkeys(FEATURE_NAMES, 0.0)
    vals.update(values)
    return FeatureRow("v", seat, start, start + 3000, box, vals)


def test_select_merges_adjacent_rule_rows() -> None:
    rows = [_row(start=s * 1000, rule_LOOK_AT_LAP=1.0 if 5 <= s <= 7 else 0.0) for s in range(20)]
    rule = [seg for seg in select_segments(rows, quiet_fraction=0.0) if seg.why == "rule"]
    assert len(rule) == 1
    assert (rule[0].start_ms, rule[0].end_ms) == (5000, 10_000)


def test_quiet_sample_is_seeded() -> None:
    rows = [_row(start=s * 1000) for s in range(100)]
    first = [seg for seg in select_segments(rows, quiet_fraction=0.2, seed=3) if seg.why == "quiet"]
    again = [seg for seg in select_segments(rows, quiet_fraction=0.2, seed=3) if seg.why == "quiet"]
    assert len(first) == 20 and first == again


def test_match_by_box_iou() -> None:
    rows = [_row(start=0), _row(seat=B, start=0, box=BOX_B)]
    near = Label("v", Box(0.21, 0.3, 0.1, 0.3), 0, 3000, "LOOK_AT_LAP", "claude")
    elsewhere = Label("v", Box(0.8, 0.7, 0.1, 0.1), 0, 3000, "LOOK_AT_LAP", "claude")
    pairs, unmatched = match(rows, [near, elsewhere])
    assert [(row.seat_id, label) for row, label in pairs] == [(A, "LOOK_AT_LAP")]
    assert unmatched == 1


def test_unsure_is_dropped() -> None:
    pairs, unmatched = match([_row()], [Label("v", BOX_A, 0, 3000, "UNSURE", "claude")])
    assert pairs == [] and unmatched == 0


def test_unknown_label_raises(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    save_labels(path, [Label("v", BOX_A, 0, 3000, "NORMAL", "claude")])
    assert load_labels(path)[0].label == "NORMAL"
    path.write_text(json.dumps({"video": "v", "seat_box": [0.2, 0.3, 0.1, 0.3], "start_ms": 0, "end_ms": 3000,
                                "label": "CHEATING", "labeler": "claude"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CHEATING"):
        load_labels(path)


def _sheets_module():
    spec = importlib.util.spec_from_file_location("label_sheets", ROOT / "scripts" / "label_sheets.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sheet_written(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))  # type: ignore[attr-defined]
    for i in range(40):
        writer.write(np.full((180, 320, 3), 40 + i, np.uint8))
    writer.release()
    segment = Segment("in", A, BOX_A, 500, 3500, "rule")
    _sheets_module().write_sheets(video, [segment], {A: [BOX_A, BOX_B]}, tmp_path / "sheets")
    assert cv2.imread(str(tmp_path / "sheets" / "0000.jpg")) is not None
    lines = (tmp_path / "sheets" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["n"] == 0
