"""Phase 3 tier 1: one whole-video window per candidate seat, from the fast tier's tracks."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from proctoring_ai.multi.cctv.timeline import all_seat_windows, load_tracks

A, B, C = uuid4(), uuid4(), uuid4()


def _row(t: int, seats: dict) -> dict:
    return {"t": t, "seats": {str(key): value for key, value in seats.items()}}


def test_candidate_seats_get_one_window_each() -> None:
    rows = []
    for i in range(10):
        role_c = "walker" if i < 8 else "candidate"
        rows.append(_row(i * 200, {A: [0.1, 0.3, 0.1, 0.2, "candidate"], B: [0.4, 0.3, 0.1, 0.2, "candidate"],
                                   C: [0.7, 0.3, 0.1, 0.2, role_c]}))
    windows = all_seat_windows(rows, 5000)
    assert sorted(w.seat_id for w in windows) == sorted([A, B])
    for window in windows:
        assert window.start_ms == 0 and window.end_ms == 5000
        assert window.reasons == ("ALL_SEATS",)
    assert [str(w.seat_id) for w in windows] == sorted(str(w.seat_id) for w in windows)


def test_box_is_median() -> None:
    rows = [_row(i, {A: [x, 0.3, 0.1, 0.2, "candidate"]}) for i, x in enumerate((0.1, 0.3, 0.2))]
    (window,) = all_seat_windows(rows, 1000)
    assert abs(window.bbox.x - 0.2) < 1e-9


def test_neighbours_same_row_only() -> None:
    seats = {A: [0.1, 0.3, 0.1, 0.2, "candidate"], B: [0.2, 0.3, 0.1, 0.2, "candidate"],
             C: [0.1, 0.7, 0.1, 0.2, "candidate"]}
    windows = {w.seat_id: w for w in all_seat_windows([_row(0, seats)], 1000)}
    assert set(windows[A].seats) == {A, B}
    assert windows[A].neighbor_ids == (B,)
    assert C not in windows[A].seats


def test_load_tracks(tmp_path: Path) -> None:
    path = tmp_path / "tracks.jsonl"
    path.write_text("\n".join(json.dumps(_row(i, {A: [0.1, 0.3, 0.1, 0.2, "candidate"]})) for i in range(3)) + "\n")
    assert len(load_tracks(path)) == 3
