from __future__ import annotations

from proctoring_ai.pipeline.metrics import PipelineMetrics


def test_reports_measured_rolling_capture_and_processing_fps() -> None:
    metrics = PipelineMetrics(target_fps=25, warmup_seconds=3)
    for index in range(61):
        observed_at = 100.0 + index / 25
        metrics.record_capture(observed_at)
        metrics.record_processed(80, observed_at)

    snapshot = metrics.snapshot(now=102.4)

    assert snapshot["capture_fps"] == 25.0
    assert snapshot["processed_fps"] == 25.0
    assert snapshot["realtime_status"] == "WARMING_UP"

    snapshot = metrics.snapshot(now=104.0)
    assert snapshot["realtime_status"] == "REALTIME"


def test_slow_processing_reports_degraded_even_when_capture_is_fast() -> None:
    metrics = PipelineMetrics(target_fps=25, warmup_seconds=0)
    for index in range(61):
        observed_at = 100.0 + index / 25
        metrics.record_capture(observed_at)
    for index in range(26):
        observed_at = 100.0 + index / 10
        metrics.record_processed(280, observed_at)
    metrics.record_drop(35)

    snapshot = metrics.snapshot(now=102.5)

    assert snapshot["capture_fps"] >= 20
    assert snapshot["processed_fps"] == 10.0
    assert snapshot["dropped_frames"] == 35
    assert snapshot["latency_ms_p95"] == 280
    assert snapshot["realtime_status"] == "DEGRADED"
