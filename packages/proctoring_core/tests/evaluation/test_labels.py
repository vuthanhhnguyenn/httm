from __future__ import annotations

from pathlib import Path

import pytest
from proctoring_core.evaluation.labels import (
    LabelError,
    apply_mapping,
    load_mapping,
    parse_csv,
    parse_oep,
)
from proctoring_core.types import EventType


def test_parse_oep_mmss_and_single_second_labels(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    gt.write_text("0001\t0008\t2\n\n0819\t0819\t1\n1330\t1417\t2", encoding="utf-8")

    labels = parse_oep(gt)

    assert [(item.start_ms, item.end_ms, item.code) for item in labels] == [
        (1000, 8000, "2"), (499_000, 500_000, "1"), (810_000, 857_000, "2"),
    ]


def test_parse_oep_reports_file_and_line(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    gt.write_text("0001\t0008\t2\nbad line\n", encoding="utf-8")

    with pytest.raises(LabelError, match=r"gt.txt:2"):
        parse_oep(gt)


def test_parse_csv_accepts_clock_and_seconds(tmp_path: Path) -> None:
    labels = tmp_path / "clip.csv"
    labels.write_text("start,end,label\n00:05,00:12.5,PHONE_DETECTED\n20,26,LOOK_DOWN\n", encoding="utf-8")

    parsed = parse_csv(labels)

    assert [(item.start_ms, item.end_ms, item.code) for item in parsed] == [
        (5000, 12500, "PHONE_DETECTED"), (20000, 26000, "LOOK_DOWN"),
    ]


def test_mapping_resolves_codes_and_event_names(tmp_path: Path) -> None:
    mapping_file = tmp_path / "map.yaml"
    mapping_file.write_text(
        "version: 1\ntolerance_ms: 1500\ncodes:\n"
        "  '2': {events: [POSSIBLE_TALKING, MULTIPLE_PERSON_DETECTED]}\n"
        "  '9': {ignore: true}\n",
        encoding="utf-8",
    )
    gt = tmp_path / "gt.txt"
    gt.write_text("0001\t0008\t2\n0010\t0012\t9\n", encoding="utf-8")
    mapping = load_mapping(mapping_file)

    intervals = apply_mapping(parse_oep(gt), mapping, gt, duration_ms=11_000)

    assert mapping.tolerance_ms == 1500
    assert intervals[0].event_types == {EventType.POSSIBLE_TALKING, EventType.MULTIPLE_PERSON_DETECTED}
    assert intervals[1].ignore and intervals[1].end_ms == 11_000


def test_unknown_code_and_label_after_video_end_fail(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    gt.write_text("0001\t0008\t7\n", encoding="utf-8")
    with pytest.raises(LabelError, match="not in the mapping"):
        apply_mapping(parse_oep(gt), None, gt)

    gt.write_text("0100\t0108\tPHONE_DETECTED\n", encoding="utf-8")
    with pytest.raises(LabelError, match="after video end"):
        apply_mapping(parse_oep(gt), None, gt, duration_ms=30_000)


def test_parse_oep_accepts_zero_padded_times(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    gt.write_text("0900\t00903\t6\n", encoding="utf-8")

    assert [(item.start_ms, item.end_ms) for item in parse_oep(gt)] == [(540_000, 543_000)]
