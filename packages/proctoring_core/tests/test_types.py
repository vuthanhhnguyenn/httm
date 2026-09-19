from __future__ import annotations

from datetime import UTC, datetime

import pytest

from proctoring_core.types import BoundingBox, Confidence, RiskScore, UtcTimestamp


def test_bounded_value_objects() -> None:
    assert Confidence(0.75).value == 0.75
    assert RiskScore(100).value == 100
    assert BoundingBox(0, 0.1, 0.5, 0.8).as_dict()["height"] == 0.8
    assert UtcTimestamp(datetime(2026, 1, 1, tzinfo=UTC)).value.tzinfo is UTC


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_confidence_rejects_out_of_range(value: float) -> None:
    with pytest.raises(ValueError):
        Confidence(value)

