"""Session event list: one row per opened episode or correlation."""

from __future__ import annotations

from typing import Any

from proctoring_runtime.messages import event_description, event_name, level_name, local_clock
from PySide6.QtCore import QModelIndex, QPersistentModelIndex, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
)

from .theme import COLORS, LEVEL_TONES, card, title_label

MAX_ROWS = 40
RECORD_ROLE = Qt.ItemDataRole.UserRole + 1  # the event record shown in the detail dialog
DISPLAY_ROLE = Qt.ItemDataRole.UserRole + 2  # pre-formatted fields painted by EventRowDelegate
ROW_HEIGHT = 58
EVIDENCE_NOTES = {
    "pending": "\nĐang lưu ảnh bằng chứng…",
    "disabled": "\nLưu ảnh bằng chứng đang tắt.",
    "queue_full": "\nHàng đợi ảnh đã đầy; khung này chưa được lưu.",
    "frame_mismatch": "\nKhông thể gắn đúng khung phân tích với sự kiện.",
    "frame_unavailable": "\nKhung ảnh phân tích không khả dụng.",
}


class EventListPanel:
    def __init__(self) -> None:
        self.frame = card()
        layout = QVBoxLayout(self.frame)
        layout.setContentsMargins(18, 16, 12, 12)
        layout.setSpacing(10)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 6, 0)
        header.addWidget(title_label("Sự kiện phiên này"))
        header.addStretch(1)
        self.count = QLabel("0")
        self.count.setObjectName("pill")
        header.addWidget(self.count)
        layout.addLayout(header)
        self.list = QListWidget()
        self.list.setMinimumHeight(180)
        self.list.setSpacing(2)
        self.list.setMouseTracking(True)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setItemDelegate(EventRowDelegate(self.list))
        layout.addWidget(self.list)
        hint = QLabel("Nhấp đúp một sự kiện để xem chi tiết và gắn nhãn.")
        hint.setObjectName("faintText")
        layout.addWidget(hint)
        self.rows: dict[str, QListWidgetItem] = {}

    def clear(self) -> None:
        self.list.clear()
        self.rows.clear()
        self.count.setText("0")

    def apply_evidence(self, item: dict[str, Any]) -> None:
        row = self.rows.get(str(item.get("event_id", "")))
        if row is None:
            return
        if item.get("status") == "available":
            row.setData(Qt.ItemDataRole.UserRole, str(item.get("relative_path", "")))
            row.setToolTip(row.toolTip() + "\nẢnh đã lưu. Nhấp đúp để xem bằng chứng.")
        else:
            row.setToolTip(row.toolTip() + "\nKhông lưu được ảnh bằng chứng; xem nhật ký kỹ thuật.")

    def mark_reviewed(self, event_id: str, label: str) -> None:
        row = self.rows.get(event_id)
        if row is None:
            return
        base = row.text().split("   ·   đã rà soát")[0]
        row.setText(f"{base}   ·   đã rà soát: {label}")
        display = dict(row.data(DISPLAY_ROLE) or {})
        display["review"] = label
        row.setData(DISPLAY_ROLE, display)

    def add(self, item: dict[str, Any]) -> None:
        event_type = str(item.get("type", "EVENT"))
        severity = str(item.get("severity", "LOW"))
        experimental = "  · Thử nghiệm" if (item.get("metrics") or {}).get("experimental") else ""
        row = QListWidgetItem(f"{local_clock(str(item.get('occurred_at', '')))}   "
                              f"{event_name(event_type)}{experimental}   ·   {level_name(severity)}")
        row.setData(RECORD_ROLE, dict(item))
        row.setData(DISPLAY_ROLE, {"time": local_clock(str(item.get("occurred_at", ""))),
                                   "name": event_name(event_type), "experimental": bool(experimental),
                                   "level": level_name(severity), "tone": LEVEL_TONES.get(severity, "info"),
                                   "review": ""})
        row.setSizeHint(QSize(0, ROW_HEIGHT))
        sources = item.get("source_event_ids") or []
        source_note = "\nSự kiện nguồn: " + ", ".join(str(source) for source in sources) if sources else ""
        row.setToolTip(event_description(item) + EVIDENCE_NOTES.get(str(item.get("evidence_status", "")), "")
                       + source_note)
        self.list.insertItem(0, row)
        self.rows[str(item.get("id", ""))] = row
        while self.list.count() > MAX_ROWS:
            removed = self.list.takeItem(self.list.count() - 1)
            self.rows = {key: value for key, value in self.rows.items() if value is not removed}
        self.count.setText(str(len(self.rows)))


class EventRowDelegate(QStyledItemDelegate):
    """Two-line event row: name on top, time · level · review below, with a severity rail."""

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex) -> QSize:
        return QSize(option.rect.width(), ROW_HEIGHT)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem,
              index: QModelIndex | QPersistentModelIndex) -> None:
        display = index.data(DISPLAY_ROLE)
        if not isinstance(display, dict):
            super().paint(painter, option, index)
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(option.rect).adjusted(0, 1, -6, -1)
        state = option.state
        if state & QStyle.StateFlag.State_Selected or state & QStyle.StateFlag.State_MouseOver:
            path = QPainterPath()
            path.addRoundedRect(rect, 10, 10)
            painter.fillPath(path, QColor(COLORS["raised"]))
        tone = QColor(COLORS.get(str(display.get("tone")), COLORS["info"]))
        painter.fillRect(QRectF(rect.left() + 6, rect.top() + 12, 3, rect.height() - 24), tone)

        left = rect.left() + 20
        name_font = QFont(option.font)
        name_font.setPixelSize(13)
        name_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(name_font)
        painter.setPen(QColor(COLORS["text"]))
        name = str(display.get("name", ""))
        top_line = QRectF(left, rect.top() + 9, rect.width() - 88, 20)
        painter.drawText(top_line, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, int(top_line.width())))

        meta_font = QFont(option.font)
        meta_font.setPixelSize(11)
        painter.setFont(meta_font)
        painter.setPen(QColor(COLORS["text_3"]))
        painter.drawText(QRectF(rect.right() - 80, rect.top() + 9, 68, 20),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(display.get("time", "")))

        x = left
        bottom = QRectF(x, rect.top() + 30, rect.width(), 18)
        parts = [(str(display.get("level", "")), tone)]
        if display.get("experimental"):
            parts.append(("Thử nghiệm", QColor(COLORS["violet"])))
        if display.get("review"):
            parts.append((f"Đã rà soát: {display['review']}", QColor(COLORS["text_2"])))
        for position, (text, colour) in enumerate(parts):
            if position:
                painter.setPen(QColor(COLORS["text_3"]))
                painter.drawText(bottom.adjusted(x - left, 0, 0, 0),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "·")
                x += painter.fontMetrics().horizontalAdvance("·") + 8
            painter.setPen(colour)
            painter.drawText(QRectF(x, bottom.top(), rect.right() - x - 8, bottom.height()),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
            x += painter.fontMetrics().horizontalAdvance(text) + 8
        painter.restore()


__all__ = ["EventListPanel", "EventRowDelegate"]
