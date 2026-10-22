from __future__ import annotations

import pytest


@pytest.mark.parametrize("mode, expected", [
    ("computer", ["center", "top_left", "top_right", "bottom_right", "bottom_left"]),
    ("paper", ["center", "paper"]),
])
def test_dialog_drives_targets_then_finishes_within_budget(monkeypatch, mode, expected) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from proctoring_core.types import ExamMode
    from proctoring_desktop.widgets.calibration_dialog import GazeCalibrationDialog
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    targets: list[str | None] = []
    finished: list[bool] = []
    dialog = GazeCalibrationDialog(ExamMode(mode), on_target=targets.append,
                                   on_finish=lambda: finished.append(True), on_cancel=lambda: None)
    # Head calibration (~2 s) plus every target must stay under the 20 s budget (SC-007).
    assert 2000 + dialog.total_ms < 20_000
    dialog._next_target()
    while not finished:
        dialog._advance()
    assert [item for item in targets if item is not None] == expected
    assert targets[-1] is None  # sampling is always switched off before finishing
    dialog.close()
    assert app is not None
