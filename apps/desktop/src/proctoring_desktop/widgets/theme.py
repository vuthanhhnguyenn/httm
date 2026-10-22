"""Dark operator theme and small label helpers."""

from __future__ import annotations

from PySide6.QtWidgets import QAbstractButton, QFrame, QLabel


def card() -> QFrame:
    frame = QFrame()
    frame.setObjectName("card")
    return frame


def title_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("cardTitle")
    label.setWordWrap(True)
    return label


def muted_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("mutedText")
    label.setWordWrap(True)
    return label


def set_text_if_changed(label: QLabel | QAbstractButton, value: str) -> None:
    if label.text() != value:
        label.setText(value)


def stylesheet() -> str:
    return """
        QWidget { background: #07111f; color: #edf3f8; font-family: 'Segoe UI'; font-size: 13px; }
        QLabel#eyebrow { color: #72d6cc; font-size: 10px; font-weight: 700; letter-spacing: 1px; }
        QLabel#pageTitle { color: #f5f8fb; font-size: 26px; font-weight: 700; }
        QFrame#card { background: #0c1a2b; border: 1px solid #1d344d; border-radius: 14px; padding: 10px; }
        QFrame#metricTile { background: #102238; border: 1px solid #1c344c; border-radius: 8px; }
        QScrollArea#railScroll { background: transparent; border: none; }
        QLabel#cardTitle { font-size: 15px; font-weight: 700; color: #f2f6fa; }
        QLabel#statusBadge, QLabel#mutedBadge { padding: 7px 11px; border-radius: 10px; background: #13283b; color: #a9c0d5; }
        QLabel#statusBadge { color: #88e1bb; }
        QLabel#videoSurface { background: #020812; border: 1px solid #203a55; border-radius: 12px; color: #8fa7be; font-size: 15px; }
        QLabel#mutedText { color: #91a9bf; font-size: 11px; }
        QLabel#metricValue { color: #f7fafc; font-size: 22px; font-weight: 700; }
        QLabel#metricLabel { color: #91a9bf; font-size: 11px; }
        QLabel#healthName, QLabel#healthState, QLabel#healthFps { font-size: 11px; }
        QLabel#healthFps { color: #a9c0d5; }
        QLabel#healthState[health='ready'] { color: #79e2af; }
        QLabel#healthState[health='unavailable'], QLabel#healthState[health='timeout'], QLabel#healthState[health='error'] { color: #ff9d91; }
        QLabel#healthState { color: #a9c0d5; }
        QLabel#riskValue { font-size: 24px; font-weight: 700; color: #f6c766; }
        QPushButton { border: 1px solid #29435d; border-radius: 9px; background: #11243a; padding: 10px 16px; font-weight: 650; }
        QPushButton:hover { background: #18324d; }
        QPushButton:disabled { color: #687b8d; background: #0b1725; }
        QPushButton#primaryButton { color: #06151a; background: #45d39a; border: none; }
        QPushButton#primaryButton:hover { background: #63e2ad; }
        QPushButton#stopButton { color: #ffd1cb; background: #351d29; border-color: #613343; }
        QSpinBox, QComboBox { padding: 8px; border-radius: 8px; border: 1px solid #29435d; background: #081525; }
        QListWidget { background: transparent; border: none; color: #d7e3ec; }
        QListWidget::item { padding: 7px 3px; border-bottom: 1px solid #1b2e42; }
        QScrollBar:vertical { width: 8px; background: #0a1624; }
        QScrollBar::handle:vertical { background: #29435d; border-radius: 4px; }
    """


__all__ = ["card", "muted_label", "set_text_if_changed", "stylesheet", "title_label"]
