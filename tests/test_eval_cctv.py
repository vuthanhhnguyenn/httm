from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _score():
    spec = importlib.util.spec_from_file_location("eval_cctv", ROOT / "scripts" / "eval_cctv.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score


def test_events_are_matched_to_labels_by_time_and_head_region() -> None:
    labels = [{"id": "turn", "kind": "exchange", "start_s": 10, "end_s": 12, "region": [0.5, 0.5, 0.7, 0.7]},
              {"id": "pair", "kind": "exchange", "start_s": 30, "end_s": 32, "region": [0.0, 0.0, 0.2, 0.2]},
              {"id": "walk", "kind": "none", "start_s": 0, "end_s": 60}]
    events = [{"action": "opened", "type": "LOOK_AT_NEIGHBOR", "video_offset_ms": 13_500, "head": [0.6, 0.6]},
              {"action": "closed", "type": "LOOK_AT_NEIGHBOR", "video_offset_ms": 15_000, "head": [0.6, 0.6]},
              {"action": "opened", "type": "HEADS_TOGETHER", "video_offset_ms": 31_000, "head": [0.9, 0.9]}]
    result = _score()(events, labels)
    assert result["found"] == ["turn"] and result["missed"] == ["pair"]
    assert [e["video_offset_ms"] for e in result["false"]] == [31_000] and result["events"] == 2


def test_event_that_opened_early_but_spans_the_label_counts() -> None:
    labels = [{"id": "turn", "kind": "exchange", "start_s": 49, "end_s": 57, "region": [0.5, 0.5, 0.7, 0.7]}]
    events = [{"action": "opened", "id": "e1", "type": "LOOK_AT_NEIGHBOR", "video_offset_ms": 46_000, "head": [0.6, 0.6]},
              {"action": "closed", "id": "e1", "type": "LOOK_AT_NEIGHBOR", "video_offset_ms": 56_000}]
    result = _score()(events, labels)
    assert result["found"] == ["turn"] and result["false"] == []
