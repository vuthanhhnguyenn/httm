from __future__ import annotations

import pytest
from proctoring_core.behavior.arbiter import SUPPORTING_KEY, SignalArbiter, validate_hierarchy
from proctoring_core.behavior.signals import RuleSignal, SignalState
from proctoring_core.config.schema import BehaviorConfig, ValidationError
from proctoring_core.types import EventType, Severity

MISSING, FACE, TURN = EventType.PERSON_MISSING, EventType.FACE_NOT_VISIBLE, EventType.HEAD_TURN_LEFT


def signal(event_type: EventType, state: SignalState) -> RuleSignal:
    return RuleSignal(event_type, state, "s", None, 0.9, Severity.MEDIUM, ("X",))


def by_type(signals: list[RuleSignal]) -> dict[EventType, RuleSignal]:
    return {item.event_type: item for item in signals}


def test_candidate_root_holds_back_explained_signals() -> None:
    arbiter = SignalArbiter({MISSING: [FACE, TURN]})

    result = by_type(arbiter.apply([signal(MISSING, SignalState.CANDIDATE), signal(FACE, SignalState.ACTIVE)]))

    assert result[FACE].state is SignalState.INACTIVE
    assert result[MISSING].state is SignalState.CANDIDATE


def test_active_root_lists_supporting_signals_and_resets_after_episode() -> None:
    arbiter = SignalArbiter({MISSING: [FACE, TURN]})
    arbiter.apply([signal(MISSING, SignalState.CANDIDATE), signal(FACE, SignalState.ACTIVE)])

    result = by_type(arbiter.apply([signal(MISSING, SignalState.ACTIVE), signal(FACE, SignalState.ACTIVE)]))
    assert result[MISSING].metrics[SUPPORTING_KEY] == FACE.value

    arbiter.apply([signal(MISSING, SignalState.INACTIVE)])
    later = by_type(arbiter.apply([signal(MISSING, SignalState.ACTIVE)]))
    assert SUPPORTING_KEY not in later[MISSING].metrics


def test_explained_signal_stays_held_until_its_condition_resets() -> None:
    arbiter = SignalArbiter({MISSING: [FACE]})
    arbiter.apply([signal(MISSING, SignalState.ACTIVE), signal(FACE, SignalState.ACTIVE)])

    # The person returns; the face condition is still satisfied from the absence.
    still = by_type(arbiter.apply([signal(MISSING, SignalState.INACTIVE), signal(FACE, SignalState.ACTIVE)]))
    assert still[FACE].state is SignalState.INACTIVE

    arbiter.apply([signal(FACE, SignalState.INACTIVE)])  # face seen again → condition reset
    fresh = by_type(arbiter.apply([signal(FACE, SignalState.ACTIVE)]))
    assert fresh[FACE].state is SignalState.ACTIVE


def test_unrelated_signals_pass_through() -> None:
    arbiter = SignalArbiter({MISSING: [FACE]})
    phone = signal(EventType.PHONE_DETECTED, SignalState.ACTIVE)

    assert arbiter.apply([signal(MISSING, SignalState.ACTIVE), phone])[1] == phone


def test_cycles_are_rejected() -> None:
    with pytest.raises(ValueError, match="cycle"):
        validate_hierarchy({MISSING: [FACE], FACE: [MISSING]})
    with pytest.raises(ValidationError):
        BehaviorConfig.model_validate({"hierarchy": {"PERSON_MISSING": ["PERSON_MISSING"]}})
