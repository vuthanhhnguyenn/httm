from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from proctoring_ai.pipeline.observation_pipeline import ObservationPipeline

from tests.fixtures.fake_detector import FakeDetector
from tests.fixtures.fake_tracker import FakeTracker


@pytest.mark.integration
def test_fake_single_person_pipeline_returns_health_and_normalized_observation() -> None:
    async def run() -> None:
        pipeline = ObservationPipeline(detector=FakeDetector([{"class": "person", "confidence": 0.95, "bbox": {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.7}}]), tracker=FakeTracker())
        bundle = await pipeline.analyze({"frame_id": 1, "captured_at_utc": datetime.now(UTC), "captured_at_monotonic_ms": 1, "frame_size": (640, 480)}, session_id=uuid4())
        assert bundle.frame_id == 1
        assert bundle.people
        assert bundle.analyzer_health["face"].status == "not_run"

    asyncio.run(run())


@pytest.mark.integration
def test_analyzer_timeout_is_explicit() -> None:
    class SlowAnalyzer:
        async def analyze(self, _frame):
            await asyncio.sleep(0.02)
            return {"status": "ready"}

    async def run() -> None:
        pipeline = ObservationPipeline(detector=FakeDetector(), tracker=FakeTracker(), face_analyzer=SlowAnalyzer(), timeout_ms=1)
        bundle = await pipeline.analyze({"frame_id": 1, "captured_at_utc": datetime.now(UTC), "captured_at_monotonic_ms": 1, "frame_size": (2, 2)}, session_id=uuid4())
        assert bundle.analyzer_health["face"].status == "timeout"

    asyncio.run(run())
