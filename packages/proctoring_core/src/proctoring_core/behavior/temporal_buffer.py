"""Monotonic sliding-window primitives for sustained conditions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ConditionState(StrEnum):
    INACTIVE = "inactive"
    CANDIDATE = "candidate"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class TemporalSample:
    monotonic_ms: int
    value: bool
    confidence: float = 1.0
    metrics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.monotonic_ms < 0:
            raise ValueError("monotonic_ms must be non-negative")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


class TemporalCondition:
    """A bounded condition that becomes active only after minimum monotonic duration."""

    def __init__(self, *, min_duration_ms: int, window_ms: int = 30_000, max_samples: int = 256,
                 max_gap_ms: int = 0, max_sample_gap_ms: int | None = None) -> None:
        if min_duration_ms < 0 or window_ms <= 0 or max_samples < 1:
            raise ValueError("invalid temporal condition bounds")
        self.min_duration_ms = min_duration_ms
        self.window_ms = window_ms
        self.samples: deque[TemporalSample] = deque(maxlen=max_samples)
        self.state = ConditionState.INACTIVE
        self.started_at_ms: int | None = None
        self.last_closed = False
        self.max_gap_ms = max_gap_ms
        self.max_sample_gap_ms = max_sample_gap_ms
        self.last_positive_ms: int | None = None

    @property
    def duration_ms(self) -> int:
        if self.started_at_ms is None or not self.samples:
            return 0
        return max(0, self.samples[-1].monotonic_ms - self.started_at_ms)

    def update(self, sample: TemporalSample) -> ConditionState:
        self.last_closed = False
        if self.samples and sample.monotonic_ms < self.samples[-1].monotonic_ms:
            raise ValueError("temporal samples must be monotonic")
        if self.samples and self.max_sample_gap_ms is not None and sample.monotonic_ms - self.samples[-1].monotonic_ms > self.max_sample_gap_ms:
            self.last_closed = self.state is not ConditionState.INACTIVE
            self.close()
        self.samples.append(sample)
        cutoff = sample.monotonic_ms - self.window_ms
        while self.samples and self.samples[0].monotonic_ms < cutoff:
            self.samples.popleft()
        if sample.value:
            if (self.max_gap_ms > 0 and self.last_positive_ms is not None
                    and len(self.samples) > 1 and not self.samples[-2].value
                    and sample.monotonic_ms - self.last_positive_ms > self.max_gap_ms):
                self.close()
            if self.started_at_ms is None:
                self.started_at_ms = sample.monotonic_ms
                self.state = ConditionState.CANDIDATE
            if sample.monotonic_ms - self.started_at_ms >= self.min_duration_ms:
                self.state = ConditionState.ACTIVE
            self.last_positive_ms = sample.monotonic_ms
        else:
            if self.max_gap_ms > 0 and self.last_positive_ms is not None and sample.monotonic_ms - self.last_positive_ms <= self.max_gap_ms:
                # Never promote a candidate using an absent detection.
                return self.state
            self.last_closed = self.last_closed or self.state is not ConditionState.INACTIVE
            self.close()
        return self.state

    def close(self) -> ConditionState:
        self.state = ConditionState.INACTIVE
        self.started_at_ms = None
        self.last_positive_ms = None
        return self.state

    def current_metrics(self) -> dict[str, Any]:
        if not self.samples:
            return {"duration_ms": 0}
        values = [item for item in self.samples if item.value]
        confidence = sum(item.confidence for item in values) / len(values) if values else 0.0
        return {"duration_ms": self.duration_ms, "confidence": round(confidence, 4), "samples": len(values)}


# Public name used by adapters that treat the condition as a per-track ring buffer.
TemporalBuffer = TemporalCondition
