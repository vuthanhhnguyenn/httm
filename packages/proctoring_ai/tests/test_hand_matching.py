from __future__ import annotations

import pytest
from proctoring_ai.hands.matching import HandMatcher


def test_swapped_order_keeps_identity_and_reports_true_motion() -> None:
    matcher = HandMatcher()
    first = matcher.update([("Left", (0.2, 0.7)), ("Right", (0.7, 0.7))], 0)
    # Same hands, list order swapped, each moved slightly.
    second = matcher.update([("Right", (0.71, 0.7)), ("Left", (0.2, 0.71))], 200)

    ids_first = {hand_id for hand_id, _, _ in first}
    assert {hand_id for hand_id, _, _ in second} == ids_first
    assert [round(move, 3) for _, move, _ in second] == [0.01, 0.01]
    assert all(dt == 200 for _, _, dt in second)


def test_implausible_jump_or_long_gap_starts_a_new_hand() -> None:
    matcher = HandMatcher(max_jump=0.25, max_gap_ms=1000)
    (first_id, _, _), = matcher.update([("Right", (0.1, 0.5))], 0)
    (jumped_id, move, _), = matcher.update([("Right", (0.9, 0.5))], 100)
    assert jumped_id != first_id and move == 0.0
    (later_id, _, _), = matcher.update([("Right", (0.9, 0.5))], 2000)
    assert later_id != jumped_id


def test_unknown_handedness_can_still_match() -> None:
    matcher = HandMatcher()
    (hand_id, _, _), = matcher.update([("unknown", (0.5, 0.5))], 0)
    (same_id, move, _), = matcher.update([("Left", (0.52, 0.5))], 150)
    assert same_id == hand_id and move == pytest.approx(0.02)
