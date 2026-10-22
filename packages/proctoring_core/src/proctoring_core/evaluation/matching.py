"""Event-level scoring of predicted events against labelled intervals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median

from ..types import EventType
from .labels import LabelInterval

ANY = "ANY"


@dataclass(frozen=True, slots=True)
class PredictedEvent:
    event_type: EventType
    start_ms: int
    end_ms: int
    severity: str = ""


@dataclass(frozen=True, slots=True)
class TypeMetrics:
    """Scores for one event type, or for the ``ANY`` group.

    ``coverage`` is matched labelled intervals / labelled intervals. For intervals that
    list several types it is a lower bound per type; the ``ANY`` row is the interval recall.
    """

    key: str
    labelled: int
    tp: int
    fn: int
    fp: int
    duplicates: int
    intervals_with_duplicates: int
    labelled_ms: int
    unlabelled_ms: int
    delays_ms: tuple[int, ...]

    @property
    def coverage(self) -> float:
        return self.tp / self.labelled if self.labelled else 0.0

    @property
    def false_alerts_per_min(self) -> float:
        minutes = self.unlabelled_ms / 60_000
        return self.fp / minutes if minutes > 0 else 0.0

    @property
    def median_delay_ms(self) -> float | None:
        return float(median(self.delays_ms)) if self.delays_ms else None

    @property
    def single_event_share(self) -> float:
        """Share of detected intervals that produced exactly one event (SC-004)."""

        return 1.0 - self.intervals_with_duplicates / self.tp if self.tp else 1.0

    def merged(self, other: TypeMetrics) -> TypeMetrics:
        if other.key != self.key:
            raise ValueError("cannot merge metrics of different keys")
        return TypeMetrics(
            self.key,
            self.labelled + other.labelled,
            self.tp + other.tp,
            self.fn + other.fn,
            self.fp + other.fp,
            self.duplicates + other.duplicates,
            self.intervals_with_duplicates + other.intervals_with_duplicates,
            self.labelled_ms + other.labelled_ms,
            self.unlabelled_ms + other.unlabelled_ms,
            self.delays_ms + other.delays_ms,
        )

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "labelled": self.labelled,
            "tp": self.tp,
            "fn": self.fn,
            "fp": self.fp,
            "duplicates": self.duplicates,
            "coverage": round(self.coverage, 4),
            "false_alerts_per_min": round(self.false_alerts_per_min, 4),
            "median_delay_ms": self.median_delay_ms,
            "single_event_share": round(self.single_event_share, 4),
            "labelled_ms": self.labelled_ms,
            "unlabelled_ms": self.unlabelled_ms,
        }


def _union_ms(spans: Iterable[tuple[int, int]], duration_ms: int) -> int:
    total = 0
    current: tuple[int, int] | None = None
    for start, end in sorted((max(0, s), min(duration_ms, e)) for s, e in spans):
        if end <= start:
            continue
        if current is None or start > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = (start, end)
        else:
            current = (current[0], max(current[1], end))
    if current is not None:
        total += current[1] - current[0]
    return total


def _inside(start_ms: int, interval: LabelInterval, tolerance_ms: int) -> bool:
    return interval.start_ms - tolerance_ms <= start_ms <= interval.end_ms + tolerance_ms


def _overlaps(event: PredictedEvent, interval: LabelInterval, tolerance_ms: int) -> bool:
    return event.start_ms <= interval.end_ms + tolerance_ms and event.end_ms >= interval.start_ms - tolerance_ms


def _score(key: str, labelled: Sequence[LabelInterval], ignored: Sequence[LabelInterval],
           predicted: Sequence[PredictedEvent], duration_ms: int, tolerance_ms: int) -> TypeMetrics:
    """An interval is detected when a compatible event is on screen during it (± tolerance).

    Delay counts from the interval start to the first such event's start (0 if it was already
    open). Extra events *starting* inside one interval are duplicates (a fragmented alert).
    An event that overlaps no compatible interval and no ignore region is a false alert.
    """

    events = sorted(predicted, key=lambda item: (item.start_ms, item.end_ms, item.event_type.value))
    assigned: dict[int, int] = {}
    assigned_events: set[int] = set()
    delays: list[int] = []

    # Assign each prediction to at most one compatible label. Processing labels in time order
    # keeps the result deterministic when labelled intervals overlap.
    for label_index, interval in sorted(enumerate(labelled), key=lambda item: (item[1].start_ms, item[1].end_ms)):
        candidates = [
            event_index for event_index, event in enumerate(events)
            if event_index not in assigned_events
            and event.event_type in interval.event_types
            and _overlaps(event, interval, tolerance_ms)
        ]
        if not candidates:
            continue
        event_index = min(candidates, key=lambda index: (
            0 if _inside(events[index].start_ms, interval, tolerance_ms) else 1,
            abs(events[index].start_ms - interval.start_ms),
            events[index].start_ms,
        ))
        assigned[label_index] = event_index
        assigned_events.add(event_index)
        delays.append(max(0, events[event_index].start_ms - interval.start_ms))

    duplicate_counts: dict[int, int] = {}
    for label_index, interval in enumerate(labelled):
        compatible = [
            event_index for event_index, event in enumerate(events)
            if event.event_type in interval.event_types and _overlaps(event, interval, tolerance_ms)
        ]
        if label_index in assigned:
            # Duplicates count fragmented alerts of the *same* signal, not one detection by
            # each of several different signals sharing a multi-type label interval.
            by_type: dict[EventType, int] = {}
            for event_index in compatible:
                by_type[events[event_index].event_type] = by_type.get(
                    events[event_index].event_type, 0) + 1
            duplicate_counts[label_index] = max(0, sum(by_type.values()) - len(by_type))

    fp = sum(
        1 for event_index, event in enumerate(events)
        if event_index not in assigned_events
        and not any(event.event_type in interval.event_types
                    and _overlaps(event, interval, tolerance_ms) for interval in labelled)
        and not any(_overlaps(event, interval, 0) for interval in ignored)
    )
    spans = [(item.start_ms - tolerance_ms, item.end_ms + tolerance_ms) for item in labelled]
    ignored_spans = [(item.start_ms, item.end_ms) for item in ignored]
    covered = _union_ms([*spans, *ignored_spans], duration_ms)
    return TypeMetrics(
        key=key,
        labelled=len(labelled),
        tp=len(assigned),
        fn=len(labelled) - len(assigned),
        fp=fp,
        duplicates=sum(duplicate_counts.values()),
        intervals_with_duplicates=sum(1 for count in duplicate_counts.values() if count > 0),
        labelled_ms=_union_ms(((item.start_ms, item.end_ms) for item in labelled), duration_ms),
        unlabelled_ms=max(0, duration_ms - covered),
        delays_ms=tuple(delays),
    )


def score_case(labels: Sequence[LabelInterval], predicted: Sequence[PredictedEvent], *,
               duration_ms: int, tolerance_ms: int = 2000,
               event_types: Iterable[EventType] | None = None) -> dict[str, TypeMetrics]:
    """Score one recording. Keys are ``EventType`` values plus ``ANY``.

    A predicted event matches a labelled interval of a type it belongs to when it starts
    within ``[start - tolerance, end + tolerance]``. Extra matches of the same interval are
    duplicates. Unmatched events are false alerts unless they start inside an ignore region.
    """

    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    ignored = [item for item in labels if item.ignore]
    active = [item for item in labels if not item.ignore]
    types = set(event_types or ())
    types |= {event.event_type for event in predicted}
    for item in active:
        types |= item.event_types
    results: dict[str, TypeMetrics] = {}
    for event_type in sorted(types, key=lambda item: item.value):
        results[event_type.value] = _score(
            event_type.value,
            [item for item in active if event_type in item.event_types],
            ignored,
            [event for event in predicted if event.event_type is event_type],
            duration_ms,
            tolerance_ms,
        )
    results[ANY] = _score(ANY, active, ignored, predicted, duration_ms, tolerance_ms)
    return results


def merge_totals(cases: Iterable[Mapping[str, TypeMetrics]]) -> dict[str, TypeMetrics]:
    totals: dict[str, TypeMetrics] = {}
    for case in cases:
        for key, metrics in case.items():
            totals[key] = totals[key].merged(metrics) if key in totals else metrics
    return totals


__all__ = ["ANY", "PredictedEvent", "TypeMetrics", "merge_totals", "score_case"]
