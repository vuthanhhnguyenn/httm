"""Event detail for review: analysed frame with its box, provenance, and the reviewer's label."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from proctoring_core.events.review_models import ReviewType
from proctoring_runtime.messages import event_description, event_name, level_name, local_clock
from proctoring_runtime.session_reader import MISSING
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .painting import draw_box, object_name
from .preview import rounded
from .theme import COLORS, LEVEL_TONES, set_tone

REVIEW_LABELS = {
    ReviewType.VALID_SIGNAL: "Hợp lệ",
    ReviewType.FALSE_POSITIVE: "Báo nhầm",
    ReviewType.INCONCLUSIVE: "Không kết luận",
}
EXAM_MODE_LABELS = {"paper": "Thi giấy", "computer": "Thi máy"}


def _text(value: Any) -> str:
    return MISSING if value in (None, "") else str(value)


def load_evidence(storage_root: Path, relative: str | None) -> QPixmap | None:
    """Only JPEGs inside ``storage/sessions`` may be opened."""

    if not relative:
        return None
    storage = storage_root.resolve()
    path = (storage / relative).resolve()
    if not path.is_relative_to(storage / "sessions") or path.suffix.lower() != ".jpg":
        return None
    pixmap = QPixmap(str(path))
    return None if pixmap.isNull() else pixmap


class EventDetailDialog(QDialog):
    def __init__(self, item: dict[str, Any], pixmap: QPixmap | None, *,
                 on_review: Callable[[ReviewType], None], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.item = item
        self.on_review = on_review
        event_type = str(item.get("type", ""))
        experimental = bool((item.get("metrics") or {}).get("experimental"))
        self.setWindowTitle("Chi tiết sự kiện · giám thị rà soát")
        self.setMinimumWidth(620)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(14)

        header = QHBoxLayout()
        header.setSpacing(10)
        titles = QVBoxLayout()
        titles.setSpacing(4)
        eyebrow = QLabel("CHI TIẾT SỰ KIỆN")
        eyebrow.setObjectName("cardTitle")
        titles.addWidget(eyebrow)
        heading = QLabel(event_name(event_type))
        heading.setObjectName("dialogTitle")
        titles.addWidget(heading)
        header.addLayout(titles, 1)
        if experimental:
            tag = QLabel("Thử nghiệm")
            tag.setObjectName("pill")
            set_tone(tag, "info")
            header.addWidget(tag, 0, Qt.AlignmentFlag.AlignBottom)
        severity = str(item.get("severity", ""))
        level = QLabel(level_name(severity))
        level.setObjectName("pill")
        set_tone(level, LEVEL_TONES.get(severity, "info"))
        header.addWidget(level, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(header)

        self.image = QLabel("Không có ảnh đúng khung phân tích.")
        self.image.setObjectName("evidenceSurface")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(480, 300)
        trace = item.get("object_trace") if isinstance(item.get("object_trace"), dict) else None
        if pixmap is not None:
            frame = pixmap.scaled(900, 600, Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
            if trace and isinstance(trace.get("bbox"), dict):
                painter = QPainter(frame)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                draw_box(painter, trace["bbox"], frame.width(), frame.height(), QColor(COLORS["danger"]),
                         f"AI gợi ý · {object_name(str(trace.get('label', '')))} · "
                         f"{float(trace.get('confidence', 0)):.0%}")
                painter.end()
            self.image.setPixmap(rounded(frame, 11))
        layout.addWidget(self.image, 1)

        meta = QFrame()
        meta.setObjectName("metaPanel")
        grid = QGridLayout(meta)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        model = trace and " ".join(str(trace[key]) for key in ("model_name", "model_version") if trace.get(key))
        supporting = ", ".join(event_name(str(code)) for code in item.get("supporting_signals") or [])
        rows = (
            ("Thời điểm", local_clock(str(item.get("occurred_at", "")))),
            ("Mức độ", level_name(severity)),
            ("Độ tin cậy tín hiệu", f"{float(item['confidence']):.0%}" if item.get("confidence") is not None else MISSING),
            ("Kiểu bài thi", EXAM_MODE_LABELS.get(str(item.get("exam_mode", "")), MISSING)),
            ("Mô hình", _text(model)),
            ("Khung phân tích", _text(trace.get("frame_id") if trace else None)),
            ("Tín hiệu kèm theo", supporting or "không có"),
        )
        for position, (label, value) in enumerate(rows):
            row, column = (position // 2, (position % 2) * 2) if position < 6 else (3, 0)
            name = QLabel(label)
            name.setObjectName("faintText")
            text = QLabel(value)
            text.setObjectName("metaValue")
            text.setWordWrap(True)
            grid.addWidget(name, row, column)
            grid.addWidget(text, row, column + 1, 1, 3 if position == 6 else 1)
        layout.addWidget(meta)
        explanation = QLabel(event_description(item))
        explanation.setObjectName("mutedText")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.status = QLabel(self._review_text(item.get("review")))
        self.status.setObjectName("pill")
        actions.addWidget(self.status)
        actions.addStretch(1)
        self.review_buttons: dict[ReviewType, QPushButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for verdict, label in REVIEW_LABELS.items():
            button = QPushButton(label)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, value=verdict: self._review(value))
            group.addButton(button)
            actions.addWidget(button)
            self.review_buttons[verdict] = button
        layout.addLayout(actions)
        footnote = QLabel("Nhãn rà soát được lưu cùng sự kiện; không thay đổi điểm đã ghi.")
        footnote.setObjectName("faintText")
        layout.addWidget(footnote)
        self._show_review(item.get("review"))

    def _show_review(self, review: Any) -> None:
        self.status.setText(self._review_text(review))
        try:
            verdict = ReviewType(str(review).upper())
        except ValueError:
            set_tone(self.status, "")
            return
        set_tone(self.status, "ok")
        if verdict in self.review_buttons:
            self.review_buttons[verdict].setChecked(True)

    @staticmethod
    def _review_text(review: Any) -> str:
        try:
            return f"Đã rà soát: {REVIEW_LABELS[ReviewType(str(review).upper())]}"
        except (KeyError, ValueError):
            return "Chưa rà soát"

    def _review(self, verdict: ReviewType) -> None:
        self.on_review(verdict)
        self.item["review"] = verdict.value
        self._show_review(verdict.value)


__all__ = ["EventDetailDialog", "load_evidence"]
