from __future__ import annotations

from proctoring_core.calibration.seat import SeatZoneEstimator
from proctoring_core.observations import Box

SEATED = Box(0.3, 0.2, 0.4, 0.75)


def learned() -> SeatZoneEstimator:
    estimator = SeatZoneEstimator(learning_ms=1000)
    for timestamp in range(0, 1200, 125):
        estimator.update(SEATED, timestamp)
    return estimator


def test_zone_is_learned_after_window_and_then_frozen() -> None:
    estimator = SeatZoneEstimator(learning_ms=1000)
    assert estimator.update(SEATED, 0) is None
    estimator = learned()
    assert estimator.zone is not None and estimator.zone.box == SEATED
    estimator.update(Box(0.0, 0.0, 0.1, 0.1), 5000)
    assert estimator.zone.box == SEATED


def test_leaving_by_position_shoulders_or_size() -> None:
    estimator = learned()
    kwargs = {"margin": 0.15, "min_area_ratio": 0.35}
    assert not estimator.is_leaving(SEATED, None, **kwargs)
    assert estimator.is_leaving(Box(0.72, 0.05, 0.25, 0.9), None, **kwargs)
    assert estimator.is_leaving(SEATED, (0.95, 0.3), **kwargs)          # shoulders outside the zone
    assert estimator.is_leaving(Box(0.45, 0.4, 0.15, 0.3), None, **kwargs)  # far smaller: walked away
    assert not estimator.is_leaving(None, None, **kwargs)               # absence is PERSON_MISSING


def test_unlearned_zone_never_reports_leaving_and_reset_relearns() -> None:
    estimator = SeatZoneEstimator(learning_ms=1000)
    assert not estimator.is_leaving(Box(0.9, 0.9, 0.05, 0.05), None, margin=0.15, min_area_ratio=0.35)
    estimator = learned()
    estimator.reset()
    assert estimator.zone is None
