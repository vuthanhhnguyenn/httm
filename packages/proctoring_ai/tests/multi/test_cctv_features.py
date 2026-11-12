"""Phase 3: 3 s window features of the learned core."""

from __future__ import annotations

import json
import math
from pathlib import Path
from uuid import uuid4

from cctv_kps import person
from proctoring_ai.multi.cctv.features import FEATURE_NAMES, read_csv, video_features, write_csv
from proctoring_ai.multi.cctv.review_cache import load_cache
from proctoring_ai.multi.cctv.review_detectors import ReviewObject
from proctoring_ai.multi.cctv.review_rules import ReviewFrame
from proctoring_ai.multi.cctv.suspects import SuspectWindow
from proctoring_core.observations import Box

A, B = uuid4(), uuid4()
BOX_A, BOX_B = Box(0.2, 0.3, 0.1, 0.3), Box(0.32, 0.3, 0.1, 0.3)
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "cctv_clip1_review"


def _kps(cx: float, *, nose_dx: float = 0.0, wrist: tuple[float, float] | None = None) -> list[list[float]]:
    kps = person(cx=cx, top=0.32, nose_dx=nose_dx, shoulder_w=0.06, wrist=wrist)
    kps[0][1] = 0.30  # head up: nose well above the shoulders
    return kps


def _windows(duration_ms: int) -> list[SuspectWindow]:
    seats = {A: BOX_A, B: BOX_B}
    return [SuspectWindow(A, None, (B,), 0, duration_ms, ("ALL_SEATS",), BOX_A, seats),
            SuspectWindow(B, None, (A,), 0, duration_ms, ("ALL_SEATS",), BOX_B, seats)]


def _frames(duration_ms: int, pose_a, pose_b=None, objects=None) -> dict:
    """Same frames for both seats' windows; ``pose_a(ms)`` / ``pose_b(ms)`` give keypoints or None."""

    pose_b = pose_b or (lambda ms: _kps(0.37, wrist=(0.37, 0.5)))
    out = []
    for ms in range(0, duration_ms, 200):
        poses = {seat: kps for seat, kps in ((A, pose_a(ms)), (B, pose_b(ms))) if kps is not None}
        out.append(ReviewFrame(ms, {A: BOX_A, B: BOX_B}, poses, list(objects(ms)) if objects else [], 16 / 9))
    return {(str(A), 0, duration_ms): out, (str(B), 0, duration_ms): out}


def _rows(rows, seat, start):
    return next(row for row in rows if row.seat_id == seat and row.start_ms == start)


def test_head_turn_shows_in_yaw_dev() -> None:
    frames = _frames(20_000, lambda ms: _kps(0.25, nose_dx=0.6 if 9000 <= ms < 11000 else 0.0, wrist=(0.25, 0.5)))
    rows = video_features("v", _windows(20_000), frames)
    assert _rows(rows, A, 9000).values["yaw_turn_share"] >= 0.5
    assert _rows(rows, A, 0).values["yaw_turn_share"] == 0


def test_reach_and_wrist_gap() -> None:
    frames = _frames(12_000, lambda ms: _kps(0.25, wrist=(0.37, 0.45) if 5000 <= ms < 6000 else (0.25, 0.5)),
                     lambda ms: _kps(0.37, wrist=(0.36, 0.45)))
    row = _rows(video_features("v", _windows(12_000), frames), A, 4000)
    assert row.values["reach_out_max"] > 0.5
    assert row.values["wrist_gap_min"] < 0.2
    assert row.partner == str(B) and row.reacher == str(A)


def test_phone_near_wrist() -> None:
    phone = ReviewObject("phone", 0.7, Box(0.24, 0.49, 0.02, 0.02), "test")
    frames = _frames(6000, lambda ms: _kps(0.25, wrist=(0.25, 0.5)), objects=lambda ms: [phone])
    row = _rows(video_features("v", _windows(6000), frames), A, 0)
    assert row.values["phone_conf_max"] == 0.7
    assert row.values["phone_near_wrist_share"] > 0


def test_absent_seat_has_no_rows() -> None:
    frames = _frames(30_000, lambda ms: None if 10_000 <= ms < 20_000 else _kps(0.25, wrist=(0.25, 0.5)))
    rows = [row for row in video_features("v", _windows(30_000), frames) if row.seat_id == A]
    assert rows
    assert not any(row.start_ms >= 10_000 and row.end_ms <= 20_000 for row in rows)


def test_short_seat_uses_row_baseline() -> None:
    frames = _frames(20_000, lambda ms: _kps(0.25, nose_dx=0.5, wrist=(0.25, 0.5)) if ms < 5000 else None,
                     lambda ms: _kps(0.37, nose_dx=0.2, wrist=(0.37, 0.5)))
    rows = [row for row in video_features("v", _windows(20_000), frames) if row.seat_id == A]
    assert rows
    assert all(math.isfinite(v) for row in rows for v in row.values.values())
    assert abs(rows[0].values["yaw_dev_mean"] - 0.3) < 0.02


def test_no_nan_and_fixed_columns() -> None:
    frames = _frames(8000, lambda ms: _kps(0.25, wrist=(0.25, 0.5)))
    rows = video_features("v", _windows(8000), frames)
    assert rows
    for row in rows:
        assert tuple(row.values) == FEATURE_NAMES
        assert all(math.isfinite(v) for v in row.values.values())


def test_csv_roundtrip(tmp_path: Path) -> None:
    frames = _frames(8000, lambda ms: _kps(0.25, wrist=(0.25, 0.5)))
    rows = video_features("v", _windows(8000), frames)
    write_csv(tmp_path / "f.csv", rows)
    again = read_csv(tmp_path / "f.csv")
    assert [(r.seat_id, r.start_ms, r.end_ms, r.partner, r.reacher) for r in again] == \
        [(r.seat_id, r.start_ms, r.end_ms, r.partner, r.reacher) for r in rows]
    for a, b in zip(again, rows, strict=True):
        assert all(abs(a.values[k] - b.values[k]) < 1e-5 for k in FEATURE_NAMES)
        assert abs(a.seat_box.x - b.seat_box.x) < 1e-6


def test_clip1_fixture_runs() -> None:
    windows = [SuspectWindow.from_json(json.loads(line))
               for line in (FIXTURE / "suspects.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = video_features("EPf9O7UXRoM", windows[:4], load_cache(FIXTURE / "cache.json.gz"))
    assert rows


def test_no_reach_names_the_same_reacher_from_both_seats() -> None:
    frames = _frames(6000, lambda ms: _kps(0.25, wrist=(0.25, 0.5)), lambda ms: _kps(0.37, wrist=(0.37, 0.5)))
    rows = video_features("v", _windows(6000), frames)
    a, b = _rows(rows, A, 0), _rows(rows, B, 0)
    assert a.reacher is not None and a.reacher == b.reacher


def test_pair_features_follow_the_closest_head_not_the_first_neighbour() -> None:
    """A's wrists are hidden; A leans its head toward C. The head feature and the partner name C, not B."""

    c = uuid4()
    box_c = Box(0.08, 0.3, 0.1, 0.3)
    seats = {A: BOX_A, B: BOX_B, c: box_c}
    windows = [SuspectWindow(A, None, (B, c), 0, 12_000, ("ALL_SEATS",), BOX_A, seats),
               SuspectWindow(B, None, (A,), 0, 12_000, ("ALL_SEATS",), BOX_B, seats),
               SuspectWindow(c, None, (A,), 0, 12_000, ("ALL_SEATS",), box_c, seats)]
    out = []
    for ms in range(0, 12_000, 200):
        lean = 5000 <= ms < 8000
        poses = {A: _kps(0.17 if lean else 0.25), B: _kps(0.37), c: _kps(0.13)}
        out.append(ReviewFrame(ms, seats, poses, [], 16 / 9))
    frames = {(str(s), 0, 12_000): out for s in (A, B, c)}
    row = _rows(video_features("v", windows, frames), A, 5000)
    assert row.partner == str(c)
    assert row.values["head_gap_ratio_min"] < 0.6
