"""Recorded-video evaluation: labels, event-level matching and reports."""

from .labels import (
    LabelError,
    LabelInterval,
    LabelMapping,
    apply_mapping,
    load_labels,
    load_mapping,
)
from .matching import ANY, PredictedEvent, TypeMetrics, merge_totals, score_case
from .report import CaseResult, EvaluationReport

__all__ = [
    "ANY",
    "CaseResult",
    "EvaluationReport",
    "LabelError",
    "LabelInterval",
    "LabelMapping",
    "PredictedEvent",
    "TypeMetrics",
    "apply_mapping",
    "load_labels",
    "load_mapping",
    "merge_totals",
    "score_case",
]
