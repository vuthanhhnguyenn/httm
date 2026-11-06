"""Phase 3: training the learned core and scoring it one held-out video at a time."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from uuid import uuid4

from proctoring_ai.multi.cctv.features import FEATURE_NAMES, FeatureRow
from proctoring_core.observations import Box

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("train_core", ROOT / "scripts" / "train_core.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic(video: str = "v", normal: int = 200, look: int = 40, turn: int = 0, seed: int = 0):
    rng = random.Random(seed)
    seat = uuid4()

    def row(i: int, **values) -> FeatureRow:
        vals = {name: rng.random() * 0.1 for name in FEATURE_NAMES}
        vals.update(values)
        return FeatureRow(video, seat, i * 1000, i * 1000 + 3000, Box(0.2, 0.3, 0.1, 0.3), vals)

    pairs = [(row(i), "NORMAL") for i in range(normal)]
    pairs += [(row(normal + i, yaw_turn_share=0.8 + 0.2 * rng.random(), yaw_dev_max=0.9), "LOOK_AT_NEIGHBOR")
              for i in range(look)]
    pairs += [(row(normal + look + i, yaw_dev_max=2.0, kp_conf_mean=0.9), "TURN_AROUND") for i in range(turn)]
    return pairs


def test_train_separates_synthetic_classes() -> None:
    module = _module()
    pairs = synthetic()
    booster, classes = module.train(pairs)
    probs = module.predict(booster, classes, [row for row, _ in pairs])
    right = sum(max(p, key=p.get) == label for p, (_, label) in zip(probs, pairs, strict=True))
    assert right / len(pairs) >= 0.95


def test_turn_around_merged_when_rare() -> None:
    _booster, classes = _module().train(synthetic(turn=5))
    assert "TURN_AROUND" not in classes and "LOOK_AT_NEIGHBOR" in classes


def test_pick_thresholds() -> None:
    assert _module().pick_thresholds([0.9, 0.8, 0.6, 0.4], [True, True, False, True]) == (0.8, 0.4)


def test_pick_thresholds_never_reached() -> None:
    assert _module().pick_thresholds([0.9, 0.5], [False, False]) == (1.01, 1.01)


def test_leave_one_out_reports_each_video() -> None:
    result = _module().leave_one_out({"a": synthetic("a", seed=1), "b": synthetic("b", seed=2)})
    assert set(result) == {"a", "b"}
    for video in result.values():
        assert 0.0 <= video["LOOK_AT_NEIGHBOR"]["recall"] <= 1.0


def test_meta_lists_features(tmp_path: Path) -> None:
    module = _module()
    booster, classes = module.train(synthetic())
    module.save_model(tmp_path / "m", booster, classes, {"LOOK_AT_NEIGHBOR": (0.8, 0.5)}, {"videos": ["v"]})
    meta = json.loads((tmp_path / "m" / "meta.json").read_text(encoding="utf-8"))
    assert meta["feature_names"] == list(FEATURE_NAMES)
    assert meta["classes"] == list(classes) and meta["videos"] == ["v"]
    assert (tmp_path / "m" / "model.txt").is_file()


def test_leave_one_out_keeps_turn_around_when_the_rest_has_enough() -> None:
    result = _module().leave_one_out({"held": synthetic("held", turn=5, seed=1), "rest": synthetic("rest", turn=12, seed=2)})
    assert "TURN_AROUND" in result["held"]["_labels"]


def test_leave_one_out_skips_a_one_class_fold(capsys) -> None:
    only_normal = [(row, "NORMAL") for row, _label in synthetic("b", seed=2)]
    result = _module().leave_one_out({"a": synthetic("a", seed=1), "b": only_normal})
    assert "a" not in result and "b" in result
    assert "a" in capsys.readouterr().err
