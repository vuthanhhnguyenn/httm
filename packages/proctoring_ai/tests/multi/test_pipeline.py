from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from proctoring_ai.multi.crop import SubjectCropper  # noqa: E402
from proctoring_ai.multi.face_pool import PerSubjectFaceAnalyzer  # noqa: E402
from proctoring_ai.multi.focus import FocusScheduler  # noqa: E402
from proctoring_ai.multi.focus_analyzers import FocusAnalyzers  # noqa: E402
from proctoring_ai.multi.pipeline import MultiSubjectPipeline  # noqa: E402
from proctoring_ai.multi.tracker import MultiSubjectTracker  # noqa: E402

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def person(x: float, c: float = 0.9) -> dict[str, Any]:
    return {"class": "person", "confidence": c, "bbox": {"x": x, "y": 0.2, "width": 0.2, "height": 0.6}}


class FakeDetector:
    def __init__(self, frames: list[list[dict[str, Any]]] | None = None, error: bool = False) -> None:
        self.frames = frames or []
        self.calls = 0
        self.error = error

    async def detect(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls += 1
        if self.error:
            raise RuntimeError("no model")
        return self.frames[min(self.calls - 1, len(self.frames) - 1)] if self.frames else []


class FakeFace:
    created: list[FakeFace] = []

    def __init__(self) -> None:
        self.closed = False
        FakeFace.created.append(self)

    async def initialize(self) -> dict[str, Any]:
        return {"status": "ready"}

    async def analyze(self, frame: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ready", "faces": [{"bbox": {"x": 0.4, "y": 0.3, "width": 0.2, "height": 0.3},
                                              "yaw": 2.0, "pitch": 1.0, "quality": 0.9, "pose_valid": True}]}

    def auto_calibrate(self) -> str:
        return "uncalibrated"

    def close(self) -> None:
        self.closed = True


def frame(index: int) -> dict[str, Any]:
    return {"frame_id": index + 1, "captured_at_monotonic_ms": index * 100,
            "captured_at_utc": EPOCH + timedelta(milliseconds=index * 100), "frame_size": (640, 480),
            "image": np.full((480, 640, 3), 100, np.uint8)}


def pipeline(detector: FakeDetector, **kwargs: Any) -> MultiSubjectPipeline:
    FakeFace.created = []
    options: dict[str, Any] = {
        "detector": detector, "tracker": MultiSubjectTracker(), "cropper": SubjectCropper(min_side=64),
        "faces": PerSubjectFaceAnalyzer(FakeFace), "focus_analyzers": FocusAnalyzers(None, None),
        "scheduler": FocusScheduler(),
    }
    options.update(kwargs)
    return MultiSubjectPipeline(**options)


async def test_two_people_get_two_bundles_with_their_own_faces() -> None:
    p = pipeline(FakeDetector([[person(0.1), person(0.6)]]))
    result = await p.analyze(frame(0), session_id=uuid4())
    assert len(result.bundles) == 2 and len(FakeFace.created) == 2
    xs = sorted(bundle.faces[0]["bbox"]["x"] for bundle in result.bundles.values())
    assert xs[0] < 0.3 < xs[1]
    for seat_id, bundle in result.bundles.items():
        assert bundle.people[0]["track_id"] == str(seat_id)
    assert result.timing.fresh_faces == 2 and result.timing.total_ms >= 0


async def test_low_confidence_person_is_not_tracked() -> None:
    p = pipeline(FakeDetector([[person(0.1), person(0.6, c=0.4)]]))
    result = await p.analyze(frame(0), session_id=uuid4())
    assert len(result.tracker.analyzed) == 1


async def test_phone_is_attached_to_nearest_seat_bundle() -> None:
    phone = {"class": "cell phone", "confidence": 0.8, "bbox": {"x": 0.15, "y": 0.5, "width": 0.03, "height": 0.05}}
    p = pipeline(FakeDetector([[person(0.1), person(0.6), phone]]))
    result = await p.analyze(frame(0), session_id=uuid4())
    owners = [bundle.people[0]["bbox"]["x"] for bundle in result.bundles.values() if bundle.objects]
    assert owners == [pytest.approx(0.1)]


async def test_yolo_every_two_reuses_detections_on_odd_frames() -> None:
    detector = FakeDetector([[person(0.1)]])
    p = pipeline(detector, yolo_every=2)
    results = [await p.analyze(frame(index), session_id=uuid4()) for index in range(4)]
    assert [r.timing.detector_reused for r in results] == [False, True, False, True]
    assert detector.calls == 2
    assert all(len(r.bundles) == 1 for r in results)


async def test_seat_skipped_by_rotation_has_no_bundle() -> None:
    scheduler = FocusScheduler(frame_budget_ms=-1.0)  # every frame counts as over budget
    p = pipeline(FakeDetector([[person(0.1)]]), scheduler=scheduler)
    session = uuid4()
    first = await p.analyze(frame(0), session_id=session)
    second = await p.analyze(frame(1), session_id=session)  # 100 ms later < 250 ms
    assert len(first.bundles) == 1 and second.bundles == {}


async def test_detector_error_yields_missing_seats_not_exception() -> None:
    detector = FakeDetector([[person(0.1)]])
    p = pipeline(detector)
    session = uuid4()
    await p.analyze(frame(0), session_id=session)
    detector.error = True
    result = await p.analyze(frame(1), session_id=session)
    assert result.tracker.analyzed == [] and len(result.tracker.missing) == 1
    bundle = next(iter(result.bundles.values()))
    assert bundle.people == [] and bundle.analyzer_health["object_detector"].status == "error"


async def test_expired_seat_releases_face_instance() -> None:
    detector = FakeDetector([[person(0.1)], []])
    p = pipeline(detector, tracker=MultiSubjectTracker(relink_window_ms=150))
    session = uuid4()
    await p.analyze(frame(0), session_id=session)
    await p.analyze(frame(1), session_id=session)
    result = await p.analyze(frame(2), session_id=session)  # 200 ms missing ≥ 150 ms
    assert len(result.tracker.expired) == 1 and FakeFace.created[0].closed
    assert result.bundles == {}


async def test_build_multi_pipeline_without_models_reports_unavailable(tmp_path: Path) -> None:
    from proctoring_ai.analyzer_factory import AnalyzerFactory

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("models: []\n", encoding="utf-8")
    factory = AnalyzerFactory(manifest)
    await factory.initialize()
    built = factory.build_multi_pipeline()
    assert isinstance(built, MultiSubjectPipeline)
    result = await built.analyze(frame(0), session_id=uuid4())
    assert result.bundles == {}
