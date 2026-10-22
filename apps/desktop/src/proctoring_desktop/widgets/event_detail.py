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
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .painting import draw_box, object_name

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
        layout = QVBoxLayout(self)
        heading = QLabel(event_name(event_type) + ("  · Thử nghiệm" if experimental else ""))
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)

        self.image = QLabel("Không có ảnh đúng khung phân tích.")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(480, 300)
        trace = item.get("object_trace") if isinstance(item.get("object_trace"), dict) else None
        if pixmap is not None:
            frame = pixmap.scaled(900, 600, Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
            if trace and isinstance(trace.get("bbox"), dict):
                painter = QPainter(frame)
                draw_box(painter, trace["bbox"], frame.width(), frame.height(), QColor("#ff6b6b"),
                         f"AI gợi ý · {object_name(str(trace.get('label', '')))} · "
                         f"{float(trace.get('confidence', 0)):.0%}")
                painter.end()
            self.image.setPixmap(frame)
        layout.addWidget(self.image)

        grid = QGridLayout()
        model = trace and " ".join(str(trace[key]) for key in ("model_name", "model_version") if trace.get(key))
        supporting = ", ".join(event_name(str(code)) for code in item.get("supporting_signals") or [])
        rows = (
            ("Thời điểm", local_clock(str(item.get("occurred_at", "")))),
            ("Mức độ", level_name(str(item.get("severity", "")))),
            ("Độ tin cậy tín hiệu", f"{float(item['confidence']):.0%}" if item.get("confidence") is not None else MISSING),
            ("Kiểu bài thi", EXAM_MODE_LABELS.get(str(item.get("exam_mode", "")), MISSING)),
            ("Mô hình", _text(model)),
            ("Khung phân tích", _text(trace.get("frame_id") if trace else None)),
            ("Tín hiệu kèm theo", supporting or "không có"),
        )
        for row, (label, value) in enumerate(rows):
            name = QLabel(label)
            name.setObjectName("mutedText")
            grid.addWidget(name, row, 0)
            grid.addWidget(QLabel(value), row, 1)
        layout.addLayout(grid)
        explanation = QLabel(event_description(item))
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.status = QLabel(self._review_text(item.get("review")))
        self.status.setObjectName("mutedText")
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.review_buttons: dict[ReviewType, QPushButton] = {}
        for verdict, label in REVIEW_LABELS.items():
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, value=verdict: self._review(value))
            buttons.addWidget(button)
            self.review_buttons[verdict] = button
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Nhãn rà soát được lưu cùng sự kiện; không thay đổi điểm đã ghi."))

    @staticmethod
    def _review_text(review: Any) -> str:
        try:
            return f"Đã rà soát: {REVIEW_LABELS[ReviewType(str(review).upper())]}"
        except (KeyError, ValueError):
            return "Chưa rà soát"

    def _review(self, verdict: ReviewType) -> None:
        self.on_review(verdict)
        self.item["review"] = verdict.value
        self.status.setText(self._review_text(verdict.value))


__all__ = ["EventDetailDialog", "load_evidence"]
