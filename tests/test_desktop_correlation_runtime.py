from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from proctoring_core.events.aggregator import EventAggregator
from proctoring_core.events.correlation import CorrelationEngine
from proctoring_core.events.models import ObservationBundle, ObservationHealth
from proctoring_core.risk.engine import RiskEngine
from proctoring_core.types import EventType
from proctoring_desktop.runtime import DesktopRuntime, LatestPreview, _RuleSet


def observation(session_id, track_id, at, timestamp, *, second_person=False):
    person = {"track_id": str(track_id), "bbox": {"x": 0.2, "y": 0.1, "width": 0.6, "height": 0.8}}
    people = [person]
    if second_person:
        people.append({"track_id": str(uuid4()), "bbox": {"x": 0.8, "y": 0.1,
                                                            "width": 0.18, "height": 0.8}})
    return ObservationBundle(
        frame_id=timestamp, session_id=session_id,
        captured_at_utc=at + timedelta(milliseconds=timestamp - 1000),
        captured_at_monotonic_ms=timestamp, frame_size=(640, 480),
        people=people,
        faces=[{"yaw": 40, "pitch": 0, "quality": 0.9, "pose_valid": True,
                "bbox": {"x": 0.4, "y": 0.2, "width": 0.2, "height": 0.2}}],
        hands=[{"activity": True, "bbox": {"x": 0.4, "y": 0.45,
                                             "width": 0.1, "height": 0.1}}],
        analyzer_health={"face": ObservationHealth("ready"),
                         "hands": ObservationHealth("ready")},
    )


def fast_rules(session_id):
    return _RuleSet(session_id, {}, {"behavior": {
        "head_turn": {"minimum_duration_seconds": 0, "require_calibration": False},
        "hand_activity": {"minimum_duration_seconds": 0},
    }})


def test_face_and_hand_receive_same_track_only_when_one_person_is_located():
    session, track, at = uuid4(), uuid4(), datetime.now(UTC)
    single = fast_rules(session).evaluate(observation(session, track, at, 1000))
    relevant = [item for item in single if item["active"] and item["event_type"] in {
        EventType.HEAD_TURN_RIGHT, EventType.SUSPICIOUS_HAND_ACTIVITY}]
    assert len(relevant) == 2
    assert {item["track_id"] for item in relevant} == {str(track)}
    multiple = fast_rules(session).evaluate(observation(session, track, at, 1000, second_person=True))
    anonymous = [item for item in multiple if item["active"] and item["event_type"] in {
        EventType.HEAD_TURN_RIGHT, EventType.SUSPICIOUS_HAND_ACTIVITY}]
    assert len(anonymous) == 2
    assert all(item["track_id"] is None for item in anonymous)


@pytest.mark.asyncio
async def test_runtime_writes_one_traceable_correlation_after_confirmation(tmp_path):
    runtime = DesktopRuntime(camera_index=0, repository_root=tmp_path, preview=LatestPreview())
    session_id, track_id, at = runtime.session_id, uuid4(), datetime.now(UTC)
    runtime._event_path = tmp_path / "storage" / "sessions" / str(session_id) / "events.jsonl"
    runtime._event_path.parent.mkdir(parents=True)
    session = SimpleNamespace(id=session_id, current_risk_score=0,
                              update_risk=lambda score, level: None)
    rules, aggregator = fast_rules(session_id), EventAggregator()
    risk, correlation = RiskEngine(decay_per_second=0), CorrelationEngine()
    for timestamp in (1000, 1300, 1600):
        await runtime._process_rules(observation(session_id, track_id, at, timestamp),
                                     rules, aggregator, risk, session,
                                     correlation_engine=correlation)
    records = [json.loads(line) for line in runtime._event_path.read_text(encoding="utf-8").splitlines()]
    opened = {item["id"] for item in records if item["action"] == "opened"}
    combined = [item for item in records if item["action"] == "correlation"]
    assert len(combined) == 1
    assert combined[0]["type"] == "CORRELATION_HEAD_AND_HAND"
    assert set(combined[0]["source_event_ids"]) <= opened
    assert combined[0]["correlation_points"] == 4
    assert combined[0]["label_vi"] == "Quay đầu kèm hoạt động tay"
    assert "Giám thị" in combined[0]["message_vi"]
    assert risk.history(session_id)["correlation_points"] == 4
