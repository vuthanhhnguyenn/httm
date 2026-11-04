from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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


def _module():
    spec = importlib.util.spec_from_file_location("eval_cctv", ROOT / "scripts" / "eval_cctv.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _opened(at_s: float, head: list[float]) -> dict:
    return {"action": "opened", "id": f"e{at_s}", "type": "LOOK_AT_NEIGHBOR", "video_offset_ms": int(at_s * 1000),
            "head": head}


def test_by_behavior_counts_and_default_look() -> None:
    labels = [{"id": "a", "kind": "exchange", "start_s": 0, "end_s": 5, "region": [0, 0, 1, 1]},
              {"id": "p", "kind": "exchange", "behavior": "phone", "start_s": 20, "end_s": 25, "region": [0, 0, 1, 1]}]
    events = [_opened(2.0, head=[0.5, 0.5])]
    result = _module().score(events, labels)
    assert result["by_behavior"] == {"look": {"found": 1, "total": 1}, "phone": {"found": 0, "total": 1}}


def test_coverage_needs_overlap_in_time_and_region() -> None:
    labels = [{"id": "a", "kind": "exchange", "start_s": 10, "end_s": 12, "region": [0.1, 0.1, 0.3, 0.3]},
              {"id": "b", "kind": "exchange", "start_s": 40, "end_s": 42, "region": [0.6, 0.6, 0.8, 0.8]},
              {"id": "n", "kind": "none", "start_s": 0, "end_s": 60}]
    windows = [{"start_ms": 8000, "end_ms": 13000, "bbox": {"x": 0.15, "y": 0.15, "width": 0.1, "height": 0.1}},
               {"start_ms": 40000, "end_ms": 41000, "bbox": {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1}},
               {"start_ms": 9000, "end_ms": 12000, "bbox": {"x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1}}]
    result = _module().coverage(windows, labels, duration_s=60.0)
    assert result["covered"] == ["a"] and result["uncovered"] == ["b"]
    assert result["suspect_ratio"] == pytest.approx(6.0 / 60.0)
    assert result["review_load"] == pytest.approx(9.0 / 60.0)
