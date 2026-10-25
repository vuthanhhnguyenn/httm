"""Graphite operator theme: one set of tokens shared by the stylesheet and the painters."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QAbstractButton, QFrame, QLabel, QWidget

CHEVRON = (Path(__file__).resolve().parent / "assets" / "chevron-down.svg").as_posix()

# Neutral graphite surfaces, a single mint accent, and semantic signal colours.
COLORS = {
    "bg": "#0a0c0f",
    "surface": "#111418",
    "raised": "#161a1f",
    "sunken": "#07090b",
    "hairline": "#1e232a",
    "hairline_strong": "#2a3038",
    "text": "#eceef1",
    "text_2": "#a3aab4",
    "text_3": "#6b737e",
    "accent": "#5ee6b5",
    "accent_hover": "#7af0c6",
    "accent_ink": "#04140d",
    "ok": "#5ee6b5",
    "info": "#7cc4fa",
    "warn": "#f2b84b",
    "danger": "#f2706a",
    "violet": "#b79cff",
    "cyan": "#67d8f0",
}
FONT_STACK = "'Segoe UI Variable Text', 'Segoe UI', 'Inter', sans-serif"
DISPLAY_STACK = "'Segoe UI Variable Display', 'Segoe UI', 'Inter', sans-serif"
# Risk level -> tone used by the score, the level pill and the event rows.
LEVEL_TONES = {"NORMAL": "ok", "LOW": "info", "MEDIUM": "warn", "HIGH": "danger", "CRITICAL": "danger"}


def card(name: str = "card") -> QFrame:
    frame = QFrame()
    frame.setObjectName(name)
    return frame


def title_label(text: str) -> QLabel:
    """Small tracked section label; the content below carries the visual weight."""

    label = QLabel(text.upper())
    label.setObjectName("cardTitle")
    return label


def muted_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("mutedText")
    label.setWordWrap(True)
    return label


def set_text_if_changed(label: QLabel | QAbstractButton, value: str) -> None:
    if label.text() != value:
        label.setText(value)


def set_tone(widget: QWidget, tone: str) -> None:
    """Switch a ``[tone=...]`` stylesheet variant, repolishing only on change."""

    if widget.property("tone") == tone:
        return
    widget.setProperty("tone", tone)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def stylesheet() -> str:
    c = COLORS
    return f"""
        QMainWindow, QWidget#root, QDialog {{ background: {c['bg']}; }}
        QWidget {{ color: {c['text']}; font-family: {FONT_STACK}; font-size: 13px; }}
        QLabel {{ background: transparent; }}
        QToolTip {{ background: {c['raised']}; color: {c['text']}; border: 1px solid {c['hairline_strong']};
                    padding: 6px 8px; border-radius: 6px; }}

        QLabel#brandMark {{ background: {c['accent']}; border-radius: 4px; }}
        QLabel#brandName {{ color: {c['text_2']}; font-size: 12px; font-weight: 600; letter-spacing: 0.4px; }}
        QLabel#eyebrow {{ color: {c['text_3']}; font-size: 11px; font-weight: 600; letter-spacing: 1.6px; }}
        QLabel#pageTitle {{ color: {c['text']}; font-family: {DISPLAY_STACK}; font-size: 28px;
                            font-weight: 600; letter-spacing: -0.4px; }}

        QFrame#card {{ background: {c['surface']}; border: 1px solid {c['hairline']}; border-radius: 16px; }}
        QFrame#toolbar {{ background: {c['surface']}; border: 1px solid {c['hairline']}; border-radius: 14px; }}
        QFrame#divider {{ background: {c['hairline']}; border: none; }}
        QFrame#metricTile {{ background: {c['raised']}; border: 1px solid {c['hairline']}; border-radius: 12px; }}
        QFrame#metaPanel {{ background: {c['surface']}; border: 1px solid {c['hairline']}; border-radius: 12px; }}
        QScrollArea#railScroll, QScrollArea#railScroll > QWidget > QWidget {{ background: transparent; border: none; }}

        QLabel#cardTitle {{ color: {c['text_3']}; font-size: 11px; font-weight: 650; letter-spacing: 1.4px; }}
        QLabel#fieldLabel {{ color: {c['text_3']}; font-size: 10px; font-weight: 650; letter-spacing: 1.2px; }}
        QLabel#dialogTitle {{ color: {c['text']}; font-family: {DISPLAY_STACK}; font-size: 22px; font-weight: 600; }}
        QLabel#mutedText {{ color: {c['text_2']}; font-size: 12px; }}
        QLabel#faintText {{ color: {c['text_3']}; font-size: 11px; }}
        QLabel#metaValue {{ color: {c['text']}; font-size: 13px; }}

        QLabel#statusBadge, QLabel#mutedBadge, QLabel#pill {{
            padding: 6px 12px; border-radius: 13px; font-size: 12px; font-weight: 600;
            background: {c['raised']}; color: {c['text_2']}; border: 1px solid {c['hairline_strong']}; }}
        QLabel#pill {{ padding: 3px 10px; border-radius: 10px; font-size: 11px; }}
        QLabel#statusBadge[tone='ok'], QLabel#mutedBadge[tone='ok'],
        QLabel#pill[tone='ok'] {{ color: {c['ok']}; border-color: #1f4a3b; background: #0f1f19; }}
        QLabel#statusBadge[tone='info'], QLabel#mutedBadge[tone='info'],
        QLabel#pill[tone='info'] {{ color: {c['info']}; border-color: #1f3a52; background: #0e1a25; }}
        QLabel#statusBadge[tone='warn'], QLabel#mutedBadge[tone='warn'],
        QLabel#pill[tone='warn'] {{ color: {c['warn']}; border-color: #4d3a18; background: #1f1a0e; }}
        QLabel#statusBadge[tone='danger'], QLabel#mutedBadge[tone='danger'],
        QLabel#pill[tone='danger'] {{ color: {c['danger']}; border-color: #55262a; background: #221213; }}

        QLabel#videoSurface {{ background: {c['sunken']}; border: 1px solid {c['hairline']}; border-radius: 12px;
                               color: {c['text_3']}; font-size: 14px; }}
        QLabel#evidenceSurface {{ background: {c['sunken']}; border: 1px solid {c['hairline']}; border-radius: 12px;
                                  color: {c['text_3']}; }}

        QLabel#metricValue {{ color: {c['text']}; font-family: {DISPLAY_STACK}; font-size: 26px; font-weight: 500;
                              letter-spacing: -0.5px; }}
        QLabel#metricLabel {{ color: {c['text_3']}; font-size: 11px; }}
        QLabel#healthName {{ color: {c['text']}; font-size: 12px; }}
        QLabel#healthFps {{ color: {c['text_2']}; font-size: 12px; font-weight: 600; }}
        QLabel#healthState {{ color: {c['text_3']}; font-size: 10px; font-weight: 700; letter-spacing: 0.8px;
                              padding: 2px 8px; border-radius: 8px; background: {c['raised']}; }}
        QLabel#healthState[health='ready'], QLabel#healthState[health='no_detection'] {{
            color: {c['ok']}; background: #0f1f19; }}
        QLabel#healthState[health='loading'], QLabel#healthState[health='busy'] {{
            color: {c['warn']}; background: #1f1a0e; }}
        QLabel#healthState[health='unavailable'], QLabel#healthState[health='timeout'],
        QLabel#healthState[health='error'] {{ color: {c['danger']}; background: #221213; }}
        QLabel#riskValue {{ font-family: {DISPLAY_STACK}; font-size: 40px; font-weight: 500; letter-spacing: -1px; }}

        QProgressBar#riskBar {{ background: {c['raised']}; border: none; border-radius: 3px; }}
        QProgressBar#riskBar::chunk {{ border-radius: 3px; background: {c['ok']}; }}
        QProgressBar#riskBar[tone='info']::chunk {{ background: {c['info']}; }}
        QProgressBar#riskBar[tone='warn']::chunk {{ background: {c['warn']}; }}
        QProgressBar#riskBar[tone='danger']::chunk {{ background: {c['danger']}; }}

        QPushButton {{ border: 1px solid {c['hairline_strong']}; border-radius: 10px; background: {c['raised']};
                       color: {c['text']}; padding: 9px 16px; font-weight: 600; }}
        QPushButton:hover {{ background: #1d2229; border-color: #3a414b; }}
        QPushButton:pressed {{ background: {c['surface']}; }}
        QPushButton:disabled {{ color: #4a515b; background: transparent; border-color: {c['hairline']}; }}
        QPushButton:checked {{ color: {c['accent']}; border-color: {c['accent']}; background: #0f1f19; }}
        QPushButton#primaryButton {{ color: {c['accent_ink']}; background: {c['accent']}; border: 1px solid {c['accent']};
                                     padding: 9px 20px; }}
        QPushButton#primaryButton:hover {{ background: {c['accent_hover']}; }}
        QPushButton#primaryButton:disabled {{ color: #2f5f4c; background: #10241c; border-color: #143327; }}
        QPushButton#stopButton {{ color: {c['danger']}; background: transparent; border-color: #55262a; }}
        QPushButton#stopButton:hover {{ background: #221213; }}
        QPushButton#stopButton:disabled {{ color: #4a515b; border-color: {c['hairline']}; }}

        QSpinBox, QComboBox {{ padding: 7px 10px; border-radius: 9px; border: 1px solid {c['hairline_strong']};
                               background: {c['sunken']}; min-width: 110px; selection-background-color: #1f4a3b; }}
        QSpinBox:hover, QComboBox:hover {{ border-color: #3a414b; }}
        QSpinBox:focus, QComboBox:focus {{ border-color: {c['accent']}; }}
        QSpinBox:disabled, QComboBox:disabled {{ color: {c['text_3']}; border-color: {c['hairline']}; }}
        QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: none; }}
        QComboBox::drop-down {{ border: none; width: 24px; }}
        QComboBox::down-arrow {{ image: url("{CHEVRON}"); width: 12px; height: 12px; }}
        QComboBox QAbstractItemView {{ background: {c['raised']}; border: 1px solid {c['hairline_strong']};
                                       border-radius: 8px; padding: 4px; outline: 0;
                                       selection-background-color: #1d2229; selection-color: {c['text']}; }}

        QListWidget {{ background: transparent; border: none; outline: 0; }}
        QListWidget::item {{ border: none; }}
        QScrollBar:vertical {{ width: 6px; background: transparent; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: {c['hairline_strong']}; border-radius: 3px; min-height: 32px; }}
        QScrollBar::handle:vertical:hover {{ background: #3a414b; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    """


__all__ = ["COLORS", "LEVEL_TONES", "card", "muted_label", "set_text_if_changed", "set_tone", "stylesheet",
           "title_label"]
