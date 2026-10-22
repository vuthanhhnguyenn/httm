from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from proctoring_desktop.window import MainWindow
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    ini = tmp_path / "settings.ini"
    monkeypatch.setattr(MainWindow, "_settings", staticmethod(lambda: QSettings(str(ini), QSettings.Format.IniFormat)))
    app = QApplication.instance() or QApplication([])
    created = MainWindow(tmp_path)
    created._timer.stop()
    yield created
    created.close()
    assert app is not None


def test_exam_mode_defaults_to_computer_and_offers_both_modes(window) -> None:  # type: ignore[no-untyped-def]
    from proctoring_core.types import ExamMode

    labels = [window.exam_mode.itemText(index) for index in range(window.exam_mode.count())]
    assert labels == ["Thi máy", "Thi giấy"]
    assert window.selected_exam_mode() is ExamMode.COMPUTER


def test_exam_mode_is_locked_while_running_and_remembered(window) -> None:  # type: ignore[no-untyped-def]
    from proctoring_core.types import ExamMode

    window.exam_mode.setCurrentIndex(window.exam_mode.findData(ExamMode.PAPER.value))
    window._settings().setValue("session/exam_mode", window.selected_exam_mode().value)
    window._set_idle_controls(False)
    assert not window.exam_mode.isEnabled()
    window._on_state({"type": "stopped"})
    assert window.exam_mode.isEnabled()
    assert window._settings().value("session/exam_mode") == "paper"
