"""Phase 3: the learned core at run time (probabilities to red/yellow events)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import pytest
from proctoring_ai.multi.cctv.features import FEATURE_NAMES, FeatureRow
from proctoring_ai.multi.cctv.learned import LearnedCore, domain_warning
from proctoring_core.observations import Box

ROOT = Path(__file__).resolve().parents[4]
A, B = uuid4(), uuid4()
BOX_A, BOX_B = Box(0.2, 0.3, 0.1, 0.3), Box(0.32, 0.3, 0.1, 0.3)


def _rows(seat=A, n: int = 6, box=BOX_A, partner=None, reacher=None) -> list[FeatureRow]:
    vals = dict.fromkeys(FEATURE_NAMES, 0.0)
    return [FeatureRow("v", seat, i * 1000, i * 1000 + 3000, box, vals, partner, reacher) for i in range(n)]


def _core(thresholds=None) -> LearnedCore:
    classes = ("NORMAL", "LOOK_AT_NEIGHBOR", "PASSING_OBJECT")
    return LearnedCore(None, classes, thresholds or {"LOOK_AT_NEIGHBOR": (0.8, 0.5), "PASSING_OBJECT": (0.8, 0.5)}, {})


def _probs(kind: str, values) -> list[dict[str, float]]:
    return [{"NORMAL": 1 - p, kind: p} for p in values]


def _opened(records):
    return [r for r in records if r["action"] == "opened"]


def test_hysteresis_needs_two_windows() -> None:
    rows = _rows()
    records = _core().events(rows, _probs("LOOK_AT_NEIGHBOR", [0.6, 0.3, 0.6, 0.6, 0.4, 0.2]))
    (opened,) = _opened(records)
    (closed,) = [r for r in records if r["action"] == "closed"]
    assert opened["video_offset_ms"] == 2000 and closed["video_offset_ms"] == rows[4].end_ms
    assert opened["stage"] == "learned" and opened["seat_id"] == str(A)


def test_open_until_end() -> None:
    rows = _rows(n=3)
    records = _core().events(rows, _probs("LOOK_AT_NEIGHBOR", [0.7, 0.7, 0.7]))
    assert [r for r in records if r["action"] == "closed"][0]["video_offset_ms"] == rows[-1].end_ms


def test_levels_from_meta() -> None:
    rows = _rows(n=3)
    assert _opened(_core().events(rows, _probs("LOOK_AT_NEIGHBOR", [0.85] * 3)))[0]["level"] == "red"
    assert _opened(_core().events(rows, _probs("LOOK_AT_NEIGHBOR", [0.6] * 3)))[0]["level"] == "yellow"
    strict = _core({"LOOK_AT_NEIGHBOR": (0.9, 0.6), "PASSING_OBJECT": (0.9, 0.6)})
    assert _opened(strict.events(rows, _probs("LOOK_AT_NEIGHBOR", [0.55] * 3))) == []


def test_pair_event_goes_to_reacher() -> None:
    mine = _rows(n=3, partner=str(B), reacher=str(A))
    (event,) = _opened(_core().events(mine, _probs("PASSING_OBJECT", [0.9] * 3)))
    assert event["seat_id"] == str(A) and event["neighbor_seat_id"] == str(B)
    theirs = _rows(n=3, partner=str(B), reacher=str(B)) + _rows(B, n=3, box=BOX_B, partner=str(A), reacher=str(B))
    events = _opened(_core().events(theirs, _probs("PASSING_OBJECT", [0.9] * 6)))
    assert [e["seat_id"] for e in events] == [str(B)]
    assert events[0]["neighbor_seat_id"] == str(A)


def test_feature_mismatch_raises(tmp_path: Path) -> None:
    (tmp_path / "model.txt").write_text("x", encoding="utf-8")
    names = [name for name in FEATURE_NAMES if name != "phone_conf_max"]
    (tmp_path / "meta.json").write_text(json.dumps({"feature_names": names, "classes": ["NORMAL"], "thresholds": {}}),
                                        encoding="utf-8")
    with pytest.raises(ValueError, match="phone_conf_max"):
        LearnedCore.load(tmp_path)


def test_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LearnedCore.load(tmp_path / "khong-co")


def _train_core():
    spec = importlib.util.spec_from_file_location("train_core_l", ROOT / "scripts" / "train_core.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_roundtrip(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from test_train_core import synthetic

    module = _train_core()
    booster, classes = module.train(synthetic())
    module.save_model(tmp_path, booster, classes, {"LOOK_AT_NEIGHBOR": (0.8, 0.5)}, {})
    core = LearnedCore.load(tmp_path)
    probs = core.predict([row for row, _ in synthetic()[:5]])
    assert all(abs(sum(p.values()) - 1) < 1e-6 for p in probs)


def test_domain_warning() -> None:
    meta = {"frame_size": [1280, 720], "seat_height_median": 0.3}
    assert domain_warning(meta, (640, 360), 0.3) is None
    assert domain_warning(meta, (320, 180), 0.3)
    assert domain_warning(meta, (1280, 720), 0.1)
    assert domain_warning({}, (320, 180), 0.1) is None
