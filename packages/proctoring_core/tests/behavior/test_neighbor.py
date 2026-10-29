from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from proctoring_core.behavior.neighbor import NeighborLookEvaluator, SubjectHead
from proctoring_core.behavior.signals import RuleSignal, SignalState
from proctoring_core.observations import Box
from proctoring_core.risk.engine import RiskEngine
from proctoring_core.types import EventType, Severity

LEFT = Box(0.10, 0.30, 0.15, 0.40)   # cx = 0.175
RIGHT = Box(0.30, 0.32, 0.15, 0.40)  # cx = 0.375, same row
FAR = Box(0.80, 0.30, 0.15, 0.40)    # 4.2 widths away


def run(evaluator: NeighborLookEvaluator, yaw_left: float, ms_values: Sequence[int],
        others: Sequence[tuple[str, Box, float]] = (("b", RIGHT, 0.0),)) -> list[RuleSignal]:
    session = uuid4()
    out: list[RuleSignal] = []
    for ms in ms_values:
        subjects = [SubjectHead("a", LEFT, yaw_left, 0.9, True)]
        subjects += [SubjectHead(t, b, y, 0.9, True) for t, b, y in others]
        out = evaluator.evaluate(session_id=session, subjects=subjects, monotonic_ms=ms)
    return [s for s in out if s.track_id == "a"]


def test_turn_towards_right_neighbor_activates_after_1500_ms() -> None:
    signals = run(NeighborLookEvaluator(yaw_degrees=22), 30.0, range(0, 1600, 100))
    assert signals[-1].state is SignalState.ACTIVE
    assert signals[-1].event_type is EventType.LOOK_AT_NEIGHBOR
    assert signals[-1].severity is Severity.LOW
    assert signals[-1].metrics["neighbor_track_id"] == "b"
    assert signals[-1].metrics["risk_cap"] == 25.0


def test_turn_towards_left_neighbor_activates() -> None:
    evaluator = NeighborLookEvaluator(yaw_degrees=22)
    session = uuid4()
    last: list[RuleSignal] = []
    for ms in range(0, 1600, 100):
        last = evaluator.evaluate(session_id=session, monotonic_ms=ms, subjects=[
            SubjectHead("a", LEFT, 0.0, 0.9, True), SubjectHead("b", RIGHT, -30.0, 0.9, True)])
    signals = [s for s in last if s.track_id == "b"]
    assert signals[-1].state is SignalState.ACTIVE
    assert signals[-1].metrics["neighbor_track_id"] == "a"


def test_shorter_than_1500_ms_stays_candidate() -> None:
    signals = run(NeighborLookEvaluator(yaw_degrees=22), 30.0, range(0, 1400, 100))
    assert signals[-1].state is SignalState.CANDIDATE


def test_turn_away_from_neighbor_is_ignored() -> None:
    assert run(NeighborLookEvaluator(yaw_degrees=22), -30.0, range(0, 2000, 100)) == []


def test_no_neighbor_within_2_5_widths_is_ignored() -> None:
    assert run(NeighborLookEvaluator(yaw_degrees=22), 30.0, range(0, 2000, 100),
               others=(("b", FAR, 0.0),)) == []


def test_uncalibrated_subject_never_fires() -> None:
    evaluator = NeighborLookEvaluator(yaw_degrees=22)
    session = uuid4()
    last: list[RuleSignal] = []
    for ms in range(0, 2000, 100):
        last = evaluator.evaluate(session_id=session, monotonic_ms=ms, subjects=[
            SubjectHead("a", LEFT, 30.0, 0.9, False), SubjectHead("b", RIGHT, 0.0, 0.9, True)])
    assert [s for s in last if s.track_id == "a"] == []


def test_turning_back_closes_with_inactive_signal() -> None:
    evaluator = NeighborLookEvaluator(yaw_degrees=22)
    run(evaluator, 30.0, range(0, 1600, 100))
    closed = run(evaluator, 0.0, [1700])
    assert [s.state for s in closed] == [SignalState.INACTIVE]


def test_forget_closes_open_condition() -> None:
    evaluator = NeighborLookEvaluator(yaw_degrees=22)
    run(evaluator, 30.0, range(0, 1600, 100))
    closed = evaluator.forget("a")
    assert [s.state for s in closed] == [SignalState.INACTIVE]
    assert evaluator.forget("a") == []


def test_look_at_neighbor_has_low_risk_weight() -> None:
    assert RiskEngine.DEFAULT_WEIGHTS[EventType.LOOK_AT_NEIGHBOR] == 5.0
