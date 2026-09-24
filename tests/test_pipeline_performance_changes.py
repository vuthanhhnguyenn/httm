from __future__ import annotations

import asyncio
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import cv2
import numpy as np
import pytest
from proctoring_ai.features.image_quality import ImageQualityExtractor
from proctoring_ai.pipeline.observation_pipeline import ObservationPipeline, _bgr_to_rgb


@pytest.mark.parametrize("kind", ["black", "gray", "noise", "gradient"])
def test_quality_optimization_preserves_full_resolution_metrics(kind):
    if kind == "noise":
        image = np.random.default_rng(42).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    elif kind == "gradient":
        image = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (100, 1, 3))
    else:
        image = np.full((720, 1280, 3), 127 if kind == "gray" else 0, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(float)
    probabilities = histogram / histogram.sum()
    positive = probabilities[probabilities > 0]
    actual = ImageQualityExtractor().extract(image)
    assert actual["status"] == "ready"
    assert actual["brightness"] == pytest.approx(float(np.mean(gray) / 255), abs=1e-12)
    assert actual["entropy"] == pytest.approx(float(-np.sum(positive * np.log2(positive))), abs=1e-12)
    assert actual["blur"] == pytest.approx(float(cv2.Laplacian(gray, cv2.CV_64F).var()), abs=1e-8)


def test_fast_rgb_conversion_is_exact_and_does_not_modify_source():
    image = np.random.default_rng(2).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    before = image.copy()
    rgb = _bgr_to_rgb(image)
    assert np.array_equal(rgb, image[:, :, ::-1])
    assert np.array_equal(before, image)
    assert rgb.flags.c_contiguous
    assert not np.shares_memory(rgb, image)


@pytest.mark.asyncio
async def test_quality_and_detector_run_concurrently_on_same_frame():
    quality_started = Event()
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    def extract(observed):
        assert observed is image
        quality_started.set()
        return {"status": "ready", "blur": 123}

    async def detect(frame):
        assert frame["image"] is image
        assert await asyncio.to_thread(quality_started.wait, 2)
        return [{"class": "person", "confidence": 0.9}]

    face = SimpleNamespace(analyze=AsyncMock(return_value={"status": "ready", "faces": [{"yaw": 20}]}))
    pipeline = ObservationPipeline(detector=SimpleNamespace(detect=detect),
                                   tracker=SimpleNamespace(update=AsyncMock(return_value=[])),
                                   face_analyzer=face, image_quality=SimpleNamespace(extract=extract))
    result = await pipeline.analyze({"frame_id": 1, "captured_at_monotonic_ms": 100, "image": image}, session_id=uuid4())
    assert result.analyzer_health["object_detector"].status == "ready"
    assert result.frame_id == 1 and result.captured_at_monotonic_ms == 100
    assert result.image_quality["blur"] == 123
    assert result.faces == [{"yaw": 20}]
    assert np.array_equal(face.analyze.call_args.args[0]["rgb_image"], image[:, :, ::-1])


@pytest.mark.asyncio
async def test_stopping_pipeline_cancels_all_inference_waiters():
    detector_started, face_started = asyncio.Event(), asyncio.Event()
    detector_stopped, face_stopped = asyncio.Event(), asyncio.Event()

    async def detect(_frame):
        detector_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            detector_stopped.set()

    async def face(_frame):
        face_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            face_stopped.set()

    pipeline = ObservationPipeline(detector=SimpleNamespace(detect=detect),
                                   tracker=SimpleNamespace(update=AsyncMock(return_value=[])),
                                   face_analyzer=SimpleNamespace(analyze=face), timeout_ms=5000)
    task = asyncio.create_task(pipeline.analyze({"frame_id": 1, "captured_at_monotonic_ms": 100}, session_id=uuid4()))
    await asyncio.wait_for(asyncio.gather(detector_started.wait(), face_started.wait()), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert detector_stopped.is_set() and face_stopped.is_set()


def test_preview_uses_precise_polling_without_counting_duplicate_frames(monkeypatch, tmp_path):
    import time

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from proctoring_desktop.window import MainWindow
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path)
    window._timer.stop()
    try:
        assert window._timer.interval() == 16
        assert window._timer.timerType() is Qt.TimerType.PreciseTimer
        window.preview.append({"frame_id": 1, "captured_at_monotonic_ms": time.monotonic_ns() // 1_000_000,
                               "image": np.zeros((480, 640, 3), dtype=np.uint8)})
        for _ in range(10):
            window._refresh_preview()
        assert len(window._preview_times) == 1
        assert window._preview_fps == 0
        assert app is not None
    finally:
        window.close()
