from __future__ import annotations

from datetime import UTC, datetime

from proctoring_core.evaluation.labels import LabelInterval
from proctoring_core.evaluation.matching import ANY, PredictedEvent, score_case
from proctoring_core.evaluation.report import CaseResult, EvaluationReport
from proctoring_core.types import EventType

PHONE = EventType.PHONE_DETECTED
TALK = EventType.POSSIBLE_TALKING
TURN = EventType.HEAD_TURN_LEFT


def _interval(types: set[EventType], start_s: float, end_s: float, *, ignore: bool = False) -> LabelInterval:
    return LabelInterval(frozenset(types), int(start_s * 1000), int(end_s * 1000), "x", ignore)


def _event(event_type: EventType, start_s: float, end_s: float | None = None) -> PredictedEvent:
    return PredictedEvent(event_type, int(start_s * 1000), int((end_s or start_s + 1) * 1000))


def test_true_positive_duplicate_false_alert_and_delay() -> None:
    labels = [_interval({PHONE}, 10, 30)]
    predicted = [_event(PHONE, 12), _event(PHONE, 20), _event(PHONE, 50)]

    result = score_case(labels, predicted, duration_ms=120_000, tolerance_ms=2000)[PHONE.value]

    assert (result.tp, result.fn, result.fp, result.duplicates) == (1, 0, 1, 1)
    assert result.median_delay_ms == 2000
    assert result.single_event_share == 0.0
    # 120 s minus the labelled span widened by tolerance (8..32 s) = 96 s.
    assert result.unlabelled_ms == 96_000
    assert round(result.false_alerts_per_min, 4) == round(1 / 1.6, 4)


def test_missed_interval_and_tolerance_boundary() -> None:
    labels = [_interval({PHONE}, 10, 20), _interval({PHONE}, 40, 45)]
    predicted = [_event(PHONE, 8)]  # exactly at start - tolerance

    result = score_case(labels, predicted, duration_ms=60_000, tolerance_ms=2000)[PHONE.value]

    assert (result.tp, result.fn, result.fp) == (1, 1, 0)
    assert result.median_delay_ms == 0
    assert result.coverage == 0.5


def test_multi_type_interval_counts_once_in_any_group() -> None:
    labels = [_interval({TALK, TURN}, 10, 20)]
    predicted = [_event(TURN, 11), _event(TALK, 12)]

    scores = score_case(labels, predicted, duration_ms=60_000)

    # One labelled behaviour matched once in ANY; two different signals is not a duplicate alert.
    assert scores[TURN.value].tp == 1 and scores[TALK.value].tp == 1
    assert scores[ANY].tp == 1 and scores[ANY].duplicates == 0


def test_ignore_regions_do_not_count_as_false_alerts() -> None:
    labels = [_interval({PHONE}, 30, 40, ignore=True)]

    result = score_case(labels, [_event(PHONE, 35)], duration_ms=60_000)[PHONE.value]

    assert result.fp == 0 and result.labelled == 0
    assert result.unlabelled_ms == 50_000


def test_report_totals_and_baseline_delta() -> None:
    labels = [_interval({PHONE}, 10, 20)]
    case = CaseResult("c1", "computer", 60_000, 480, 8.0,
                      score_case(labels, [_event(PHONE, 11), _event(PHONE, 40)], duration_ms=60_000))
    baseline = {"totals": {PHONE.value: {"coverage": 1.0, "false_alerts_per_min": 2.0,
                                          "median_delay_ms": 3000.0, "single_event_share": 1.0}}}
    report = EvaluationReport("set", datetime(2026, 10, 16, tzinfo=UTC), "a" * 64, "b" * 64,
                              {"yolo11n": "8.3.0"}, {"device": "cpu"}, 8.0, [case], baseline)

    payload = report.to_dict()
    delta = payload["baseline_delta"][PHONE.value]

    assert payload["totals"][PHONE.value]["tp"] == 1
    assert delta["coverage"] == 0.0
    assert delta["median_delay_ms"] == -2000.0
    assert "Báo cáo đánh giá" in report.to_markdown()


def test_event_already_open_covers_later_interval_without_duplicates() -> None:
    # One long event spanning two separate short labeled intervals matches both. Under one-to-one
    # assignment, the event matches whichever interval comes first by time. No duplicates: only
    # the first occurrence of each predicted event counts for any statistics beyond coverage.
    labels = [_interval({PHONE}, 10, 15), _interval({PHONE}, 18, 25)]
    predicted = [PredictedEvent(PHONE, 9_000, 30_000)]  # one episode across the whole call

    result = score_case(labels, predicted, duration_ms=60_000, tolerance_ms=1000)[PHONE.value]

    # Changed from (2,0,0,0): event now matched to first label only due to one-to-one assignment.
    assert (result.tp, result.fn, result.fp, result.duplicates) == (1, 1, 0, 0)
    assert result.median_delay_ms == 0


def test_close_same_code_labels_merge_into_one_behaviour(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from proctoring_core.evaluation.labels import apply_mapping, mapping_from_dict, parse_oep

    gt = tmp_path / "gt.txt"
    gt.write_text("0305\t0308\t5\n0309\t0317\t5\n0330\t0332\t5\n0310\t0312\t2\n", encoding="utf-8")
    mapping = mapping_from_dict({"codes": {"5": {"events": ["PHONE_DETECTED"]},
                                           "2": {"events": ["POSSIBLE_TALKING"]}}})
    mapping = type(mapping)(mapping.version, mapping.codes, mapping.tolerance_ms, merge_gap_ms=3000)

    spans = [(item.source_code, item.start_ms // 1000, item.end_ms // 1000)
             for item in apply_mapping(parse_oep(gt), mapping, gt)]

    assert spans == [("5", 185, 197), ("2", 190, 192), ("5", 210, 212)]


def test_one_prediction_cannot_match_two_labelled_intervals() -> None:
    # Regression: matching previously credited the same prediction to every overlapping label.
    labels = [_interval({PHONE}, 10, 12), _interval({PHONE}, 11, 14)]
    predicted = [_event(PHONE, 10.5, 13.5)]

    result = score_case(labels, predicted, duration_ms=60_000, tolerance_ms=0)[PHONE.value]

    assert (result.tp, result.fn, result.fp, result.duplicates) == (1, 1, 0, 0)
    assert result.coverage == 0.5
