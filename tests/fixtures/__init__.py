"""Reusable deterministic fixtures for local runtime and core tests."""

from .clock import FakeClock
from .fake_camera import FakeCamera
from .fake_detector import FakeDetector
from .fake_tracker import FakeTracker

__all__ = ["FakeCamera", "FakeClock", "FakeDetector", "FakeTracker"]
