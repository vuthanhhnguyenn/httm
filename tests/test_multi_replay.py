"""Multi-subject replay: per-seat rule processors, outputs and CLI."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import yaml
from proctoring_core.behavior.signals import RuleSignal, SignalState
from proctoring_core.events.models import ObservationBundle, ObservationHealth
from proctoring_core.types import EventType, Severity
from proctoring_runtime.replay import create_replay_processor

ROOT = Path(__file__).resolve().parents[1]


def _config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    policy = yaml.safe_load((ROOT / "configs" / "exam_policy.yaml").read_text(encoding="utf-8"))
    return config, dict(policy.get("exam_policy", policy))


def empty_bundle(session_id: UUID, *, frame_id: int, ms: int) -> ObservationBundle:
    from proctoring_ai.capture.video_file_source import REPLAY_EPOCH_UTC

    return ObservationBundle(frame_id=frame_id, session_id=session_id,
                             captured_at_utc=REPLAY_EPOCH_UTC + timedelta(milliseconds=ms),
                             captured_at_monotonic_ms=ms, frame_size=(640, 480),
                             analyzer_health={"object_detector": ObservationHealth("ready")})


def neighbor_signal(session_id: UUID, seat: str, state: SignalState) -> RuleSignal:
    return RuleSignal(EventType.LOOK_AT_NEIGHBOR, state, session_id, seat, 0.8, Severity.LOW,
                      ("HEAD_TOWARDS_NEIGHBOR",), {"risk_cap": 25.0}, rule_key="LOOK_AT_NEIGHBOR")


async def test_extra_signal_opens_event_through_aggregator() -> None:
    config, policy = _config()
    session_id = uuid4()
    processor = create_replay_processor(session_id, policy, config)
    signal = neighbor_signal(session_id, str(uuid4()), SignalState.ACTIVE)
    items = await processor.process(empty_bundle(session_id, frame_id=1, ms=100), extra_signals=[signal])
    assert [(i["type"], i["action"]) for i in items if i["type"] == "LOOK_AT_NEIGHBOR"] == [
        ("LOOK_AT_NEIGHBOR", "opened")]
    opened = next(i for i in items if i["type"] == "LOOK_AT_NEIGHBOR")
    assert opened["label_vi"] == "Quay đầu về phía thí sinh bên cạnh"


async def test_last_signals_include_candidates_and_extra() -> None:
    config, policy = _config()
    session_id = uuid4()
    processor = create_replay_processor(session_id, policy, config)
    assert processor.last_signals == ()
    candidate = neighbor_signal(session_id, str(uuid4()), SignalState.CANDIDATE)
    await processor.process(empty_bundle(session_id, frame_id=1, ms=100), extra_signals=[candidate])
    assert candidate in processor.last_signals
