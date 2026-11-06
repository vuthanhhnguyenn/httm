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


@pytest.mark.parametrize(("split_ms", "max_frames"), [(None, None), (10000, 48)], ids=["offline", "live"])
def test_clip1_confidence_levels(split_ms: int | None, max_frames: int | None) -> None:
    """No false picture may be covered by a red alert; the 9 true ones stay covered (fast tier: no events here)."""

    from proctoring_ai.multi.cctv.confidence import assign_levels, load_confidence
    from proctoring_ai.multi.cctv.review_score import load_verdicts, score_review

    records = _replay(split_ms, max_frames)
    leveled = assign_levels(records, [], load_confidence(ROOT / "configs/confidence.yaml"))
    score = score_review(leveled, load_verdicts(LABELS))
    assert score["false_red"] == 0, score
    assert score["true_covered"] == 9, score


def _process_video_module():
    spec = importlib.util.spec_from_file_location("process_video", ROOT / "scripts" / "process_video.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clip1_process_video_passes_after_deep_tier() -> None:
    """process_video's own rules pass (per window, one deduper, REVIEW_SKIPPED for empty windows) and levels,
    from the clip 1 model outputs: every true picture on screen, no false one red, at most 7 false alerts."""

    from proctoring_ai.multi.cctv.confidence import load_confidence
    from proctoring_ai.multi.cctv.review_cache import load_cache
    from proctoring_ai.multi.cctv.review_score import load_verdicts, score_review
    from proctoring_ai.multi.cctv.suspects import SuspectWindow

    module = _process_video_module()
    windows = [SuspectWindow.from_json(json.loads(line))
               for line in (FIXTURE / "suspects.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    per_window = module.review_records(windows, load_cache(FIXTURE / "cache.json.gz"))
    records = module.finish_records(per_window, [], load_confidence(ROOT / "configs/confidence.yaml"))
    assert all("level" in r and r.get("label_vi") for r in records if r.get("action") in ("opened", "merged"))
    score = score_review(records, load_verdicts(LABELS))
    assert score["true_covered"] == 9, score
    assert score["false_red"] == 0, score
    assert score["false_opened"] <= 7, score
    golden = json.loads((FIXTURE / "opened_golden.json").read_text())["offline"]
    opened = sorted([r["type"], r["video_offset_ms"]] for r in records
                    if r.get("action") == "opened" and r["type"] != "REVIEW_SKIPPED")
    assert opened == golden  # the same events replay_review gives
