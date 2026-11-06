"""Clip 1 deep-tier regression: the saved model outputs (fixture) replayed through today's rules and scored
against Nguyên's verdicts on the 23 phase 2e pictures (data/labels/EPf9O7UXRoM-review.yaml)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "packages/proctoring_ai/tests/fixtures/cctv_clip1_review"
LABELS = ROOT / "data/labels/EPf9O7UXRoM-review.yaml"


def _replay_module():
    spec = importlib.util.spec_from_file_location("replay_review", ROOT / "scripts" / "replay_review.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replay(split_ms: int | None, max_frames: int | None) -> list[dict]:
    return _replay_module().replay(FIXTURE / "suspects.jsonl", FIXTURE / "cache.json.gz", split_ms=split_ms,
                                   max_frames=max_frames)


def _score(split_ms: int | None, max_frames: int | None) -> dict:
    from proctoring_ai.multi.cctv.review_score import load_verdicts, score_review

    return score_review(_replay(split_ms, max_frames), load_verdicts(LABELS))


# offline: whole windows, every 5 fps frame. live: 10 s pieces as SuspectTracker.poll hands them out, the
# ReviewWorkerConfig frame budget, the previous piece's frames as context.
@pytest.mark.parametrize(("split_ms", "max_frames"), [(None, None), (10000, 48)], ids=["offline", "live"])
def test_clip1_phase2f(split_ms: int | None, max_frames: int | None) -> None:
    score = _score(split_ms, max_frames)
    assert score["true_covered"] == 9, score
    assert score["false_opened"] <= 7, score  # pictures with a new alert; merged repeats are in false_covered


@pytest.mark.parametrize(("split_ms", "max_frames", "mode"), [(None, None, "offline"), (10000, 48, "live")],
                         ids=["offline", "live"])
def test_clip1_opened_events_unchanged(split_ms: int | None, max_frames: int | None, mode: str) -> None:
    """Recording evidence details must not change which events open, or when (golden taken before phase 2g)."""

    golden = json.loads((FIXTURE / "opened_golden.json").read_text())[mode]
    opened = sorted([r["type"], r["video_offset_ms"]] for r in _replay(split_ms, max_frames)
                    if r.get("action") == "opened")
    assert opened == golden
