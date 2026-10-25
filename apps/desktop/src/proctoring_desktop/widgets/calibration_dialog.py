"""Full-screen gaze calibration: show targets one by one while samples are collected."""

from __future__ import annotations

from collections.abc import Callable

from proctoring_core.calibration.gaze import PAPER_TARGET
from proctoring_core.types import ExamMode
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QDialog, QWidget

from .theme import COLORS

# Screen-relative dot positions; corners are inset so the dot stays fully visible.
POSITIONS = {
    "center": (0.5, 0.5),
    "top_left": (0.06, 0.08),
    "top_right": (0.94, 0.08),
    "bottom_right": (0.94, 0.92),
    "bottom_left": (0.06, 0.92),
}
SEQUENCES = {
    ExamMode.COMPUTER: ("center", "top_left", "top_right", "bottom_right", "bottom_left"),
    ExamMode.PAPER: ("center", PAPER_TARGET),
}
SETTLE_MS = 700      # let the eyes land on the dot before sampling
COLLECT_MS = 1800    # sampling window per target


class GazeCalibrationDialog(QDialog):
    """Drives ``on_target(name | None)`` and ``on_finish()``; Esc cancels via ``on_cancel()``."""

    def __init__(self, exam_mode: ExamMode, *, on_target: Callable[[str | None], None],
                 on_finish: Callable[[], None], on_cancel: Callable[[], None],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hiệu chỉnh hướng mắt")
        self.setStyleSheet(f"background: {COLORS['sunken']};")
        self.sequence = SEQUENCES[exam_mode]
        self.on_target, self.on_finish, self.on_cancel = on_target, on_finish, on_cancel
        self.index = -1
        self.collecting = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)
        self._finished = False

    @property
    def total_ms(self) -> int:
        return len(self.sequence) * (SETTLE_MS + COLLECT_MS)

    def start(self) -> None:
        self.showFullScreen()
        self._next_target()

    def _next_target(self) -> None:
        self.on_target(None)
        self.index += 1
        self.collecting = False
        if self.index >= len(self.sequence):
            self._finished = True
            self.on_finish()
            self.accept()
            return
        self.update()
        self._timer.start(SETTLE_MS)

    def _advance(self) -> None:
        if not self.collecting:
            self.collecting = True
            self.on_target(self.sequence[self.index])
            self.update()
            self._timer.start(COLLECT_MS)
        else:
            self._next_target()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._timer.stop()
            self.on_target(None)
            self.on_cancel()
            self.reject()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if not 0 <= self.index < len(self.sequence):
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(COLORS["text"]))
        painter.setFont(QFont("Segoe UI", 15, QFont.Weight.DemiBold))
        name = self.sequence[self.index]
        step = f"Bước {self.index + 1}/{len(self.sequence)}"
        if name == PAPER_TARGET:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             f"{step}\nNhìn xuống bài thi trên bàn như khi đang làm bài.\nGiữ đầu và mắt yên.")
        else:
            x, y = POSITIONS[name]
            centre = QPointF(self.width() * x, self.height() * y)
            colour = QColor(COLORS["accent"] if self.collecting else COLORS["warn"])
            halo = QColor(colour)
            halo.setAlpha(50)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(centre, 30, 30)
            painter.setBrush(colour)
            painter.drawEllipse(centre, 14, 14)
            painter.setBrush(QColor(COLORS["sunken"]))
            painter.drawEllipse(centre, 4, 4)
            painter.setPen(QColor(COLORS["text_2"]))
            painter.drawText(self.rect().adjusted(0, 0, 0, -40), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                             f"{step} · Chỉ nhìn vào chấm tròn, giữ đầu yên · Esc để hủy")
        painter.end()


__all__ = ["GazeCalibrationDialog", "SEQUENCES"]
