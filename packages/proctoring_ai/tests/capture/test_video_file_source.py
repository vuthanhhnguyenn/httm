from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from proctoring_ai.capture.video_file_source import (  # noqa: E402
    REPLAY_EPOCH_UTC,
    VideoFileFrameSource,
)


def _write_video(path: Path, *, frames: int, fps: float) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((48, 64, 3), index % 255, dtype=np.uint8)
        writer.write(image)
    writer.release()


def _collect(source: VideoFileFrameSource) -> list[dict[str, object]]:
    async def run() -> list[dict[str, object]]:
        await source.open()
        items = []
        try:
            while (frame := await source.read()) is not None:
                items.append({key: value for key, value in frame.items() if key != "image"})
        finally:
            await source.close()
        return items

    return asyncio.run(run())


def test_samples_at_fixed_rate_with_video_timestamps(tmp_path: Path) -> None:
    video = tmp_path / "clip.avi"
    _write_video(video, frames=60, fps=30)  # 2 seconds

    frames = _collect(VideoFileFrameSource(video, analysis_rate_hz=5))

    offsets = [item["video_offset_ms"] for item in frames]
    assert offsets == [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
    assert [item["frame_id"] for item in frames] == list(range(1, 11))
    assert frames[1]["captured_at_monotonic_ms"] == 201
    assert frames[1]["captured_at_utc"] == REPLAY_EPOCH_UTC.replace(microsecond=200_000)


def test_replay_is_deterministic_and_respects_window(tmp_path: Path) -> None:
    video = tmp_path / "clip.avi"
    _write_video(video, frames=90, fps=30)

    first = _collect(VideoFileFrameSource(video, analysis_rate_hz=4, start_ms=500, end_ms=2000))
    second = _collect(VideoFileFrameSource(video, analysis_rate_hz=4, start_ms=500, end_ms=2000))

    assert first == second
    assert [item["video_offset_ms"] for item in first] == [500, 767, 1000, 1267, 1500, 1767]


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _collect(VideoFileFrameSource(tmp_path / "missing.avi"))
