from __future__ import annotations

from uuid import uuid4

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from proctoring_ai.multi.motion import MotionGate  # noqa: E402


def still() -> object:
    return np.full((120, 100, 3), 80, np.uint8)


def test_static_crop_reused_until_500_ms() -> None:
    gate = MotionGate()
    seat = uuid4()
    assert gate.should_run(seat, still(), 0)  # first sample always runs
    gate.record(seat, still(), 0)
    assert not gate.should_run(seat, still(), 499)
    assert gate.should_run(seat, still(), 500)


def test_moving_crop_runs_again() -> None:
    gate = MotionGate()
    seat = uuid4()
    gate.record(seat, still(), 0)
    moved = np.full((120, 100, 3), 80, np.uint8)
    moved[:60] = 200
    assert gate.should_run(seat, moved, 100)


def test_small_noise_stays_below_threshold() -> None:
    gate = MotionGate(threshold=4.0)
    seat = uuid4()
    gate.record(seat, still(), 0)
    assert not gate.should_run(seat, np.full((120, 100, 3), 82, np.uint8), 100)


def test_different_crop_sizes_are_compared_after_resize() -> None:
    gate = MotionGate()
    seat = uuid4()
    gate.record(seat, still(), 0)
    assert not gate.should_run(seat, np.full((240, 200, 3), 80, np.uint8), 100)


def test_forget_makes_next_sample_run() -> None:
    gate = MotionGate()
    seat = uuid4()
    gate.record(seat, still(), 0)
    gate.forget(seat)
    assert gate.should_run(seat, still(), 100)
