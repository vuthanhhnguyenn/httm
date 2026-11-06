from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest
from proctoring_ai.capture.prefetch_source import PrefetchFrameSource
from proctoring_ai.multi.cctv.pipeline import CctvPipeline
from proctoring_ai.multi.cctv.tiling import TiledPoseDetector
from test_cctv_seat_map import body


class _Batching:
    def __init__(self):
        self.calls: list[list[tuple[int, int]]] = []

    def _people(self, frame):
        return [body(0.5)] if frame["frame_size"] != (100, 100) else [body(0.2, top=0.7)]

    async def detect(self, frame):
        raise AssertionError("must use detect_many")

    async def detect_many(self, frames):
        self.calls.append([f["frame_size"] for f in frames])
        return [self._people(f) for f in frames]


def test_tiles_one_predict_call():
    fake = _Batching()
    tiled = TiledPoseDetector(fake, every=2, cols=2, top=0.6)
    frame = {"image": np.zeros((100, 100, 3), np.uint8), "frame_size": (100, 100)}
    first = asyncio.run(tiled.detect(frame))
    second = asyncio.run(tiled.detect(frame))
    assert [len(call) for call in fake.calls] == [3, 1]
    assert len(first) == 3 and len(second) == 3


class _Slow:
    def __init__(self, result):
        self.result = result

    async def detect(self, _frame):
        await asyncio.sleep(0.05)
        return self.result


class _Tracker:
    def update(self, people, ms, frame_size=None):
        from proctoring_ai.multi.tracker import TrackerUpdate

        return TrackerUpdate([], [], [], [], 0, 0)


@pytest.mark.parametrize("parallel, low, high", [(True, 0.0, 0.09), (False, 0.1, 1.0)])
def test_parallel_objects_overlap(parallel, low, high):
    pipeline = CctvPipeline(pose_detector=_Slow([]), tracker=_Tracker(),
                            object_detector=_Slow([{"class": "cell phone"}]), phone_every=1,
                            parallel_objects=parallel)
    began = time.perf_counter()
    stage = asyncio.run(pipeline.detect({"frame_id": 1, "captured_at_monotonic_ms": 1, "frame_size": (10, 10)}))
    took = time.perf_counter() - began
    assert low <= took < high
    assert stage.detections == [{"class": "cell phone"}]


class _Source:
    def __init__(self, n, fail_at=None):
        self.n, self.fail_at, self.i, self.closed = n, fail_at, 0, False

    async def open(self):
        return None

    async def read(self):
        if self.fail_at is not None and self.i == self.fail_at:
            raise OSError("broken file")
        if self.i >= self.n:
            return None
        self.i += 1
        return {"frame_id": self.i}

    async def close(self):
        self.closed = True


def test_prefetch_order_end_and_error():
    async def run(source):
        wrapped = PrefetchFrameSource(source, max_queue=2)
        await wrapped.open()
        got = []
        try:
            while (frame := await wrapped.read()) is not None:
                got.append(frame["frame_id"])
            assert await wrapped.read() is None
        finally:
            await wrapped.close()
        return got

    source = _Source(5)
    assert asyncio.run(run(source)) == [1, 2, 3, 4, 5] and source.closed
    with pytest.raises(OSError):
        asyncio.run(run(_Source(5, fail_at=3)))
