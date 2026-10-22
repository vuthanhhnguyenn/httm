"""Byte-for-byte regression of the rule path on scripted scenarios.

Regenerate intentionally changed goldens with ``UPDATE_GOLDEN=1 pytest tests/golden``
and explain the behaviour change in the commit.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml
from proctoring_runtime.replay import create_replay_processor, run_replay

from tests.golden.scenarios import SCENARIOS, build

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).parent / "data"


def _config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    policy = yaml.safe_load((ROOT / "configs" / "exam_policy.yaml").read_text(encoding="utf-8"))
    return config, dict(policy.get("exam_policy", policy))


def replay_lines(name: str) -> list[str]:
    config, policy = _config()
    source, pipeline = build(SCENARIOS[name], config)
    session_id = uuid4()
    processor = create_replay_processor(session_id, policy, config)
    result = asyncio.run(run_replay(source, pipeline, processor, session_id=session_id))
    return result.golden_lines()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_matches_golden(name: str) -> None:
    lines = replay_lines(name)
    golden = GOLDEN_DIR / f"{name}.jsonl"
    if os.environ.get("UPDATE_GOLDEN") == "1" or not golden.exists():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text("\n".join(lines) + "\n", encoding="utf-8")
    expected = golden.read_text(encoding="utf-8").splitlines()
    assert lines == expected


def test_replay_is_deterministic() -> None:
    assert replay_lines("phone_with_gaps") == replay_lines("phone_with_gaps")
