from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import ceil
from time import monotonic


@dataclass
class PipelineMetrics:
    """Rolling capture and inference metrics based on observed monotonic timestamps."""

    target_fps: float = 25.0
    window_seconds: float = 2.0
    warmup_seconds: float = 3.0
    minimum_realtime_fps: float = 20.0
    maximum_realtime_latency_ms: float = 200.0
    max_latency_samples: int = 512
    dropped_frames: int = 0
    _capture_times: deque[float] = field(default_factory=deque)
    _processed_times: deque[float] = field(default_factory=deque)
    _latencies_ms: deque[float] = field(default_factory=deque)
    _started_at: float | None = None

    def record_capture(self, timestamp: float | None = None) -> None:
        observed_at = monotonic() if timestamp is None else timestamp
        if self._started_at is None:
            self._started_at = observed_at
        self._capture_times.append(observed_at)
        self._prune(observed_at)

    def record_drop(self, count: int = 1) -> None:
        self.dropped_frames += max(0, count)

    def record_processed(self, latency_ms: float, timestamp: float | None = None) -> None:
        observed_at = monotonic() if timestamp is None else timestamp
        if self._started_at is None:
            self._started_at = observed_at
        self._processed_times.append(observed_at)
        self._latencies_ms.append(max(0.0, latency_ms))
        while len(self._latencies_ms) > self.max_latency_samples:
            self._latencies_ms.popleft()
        self._prune(observed_at)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        for timestamps in (self._capture_times, self._processed_times):
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()

    @staticmethod
    def _fps(timestamps: deque[float]) -> float:
        if len(timestamps) < 2:
            return 0.0
        elapsed = timestamps[-1] - timestamps[0]
        return (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, ceil(percentile * len(ordered)) - 1))
        return ordered[index]

    def snapshot(self, now: float | None = None) -> dict[str, float | int | str]:
        observed_at = monotonic() if now is None else now
        self._prune(observed_at)
        capture_fps = self._fps(self._capture_times)
        processed_fps = self._fps(self._processed_times)
        latency_p50 = self._percentile(list(self._latencies_ms), 0.50)
        latency_p95 = self._percentile(list(self._latencies_ms), 0.95)
        elapsed = max(0.0, observed_at - self._started_at) if self._started_at is not None else 0.0
        measurement_window = min(self.window_seconds, elapsed)

        if elapsed < self.warmup_seconds:
            realtime_status = "WARMING_UP"
        elif (
            capture_fps >= self.minimum_realtime_fps
            and processed_fps >= self.minimum_realtime_fps
            and latency_p95 <= self.maximum_realtime_latency_ms
        ):
            realtime_status = "REALTIME"
        else:
            realtime_status = "DEGRADED"

        return {
            "target_fps": self.target_fps,
            "capture_fps": round(capture_fps, 1),
            "processed_fps": round(processed_fps, 1),
            "dropped_frames": self.dropped_frames,
            "latency_ms_p50": round(latency_p50, 2),
            "latency_ms_p95": round(latency_p95, 2),
            "realtime_status": realtime_status,
            "measurement_window_seconds": round(measurement_window, 2),
        }
