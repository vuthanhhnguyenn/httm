"""Session event list: one row per opened episode or correlation."""

from __future__ import annotations

from typing import Any

from proctoring_runtime.messages import event_description, event_name, level_name, local_clock
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout

from .theme import card, title_label

MAX_ROWS = 40
RECORD_ROLE = Qt.ItemDataRole.UserRole + 1  # the event record shown in the detail dialog
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
        layout.addWidget(title_label("Sự kiện phiên này"))
        self.list = QListWidget()
        self.list.setMinimumHeight(150)
        layout.addWidget(self.list)
        self.rows: dict[str, QListWidgetItem] = {}

    def clear(self) -> None:
        self.list.clear()
        self.rows.clear()

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

    def add(self, item: dict[str, Any]) -> None:
        event_type = str(item.get("type", "EVENT"))
        severity = str(item.get("severity", "LOW"))
        experimental = "  · Thử nghiệm" if (item.get("metrics") or {}).get("experimental") else ""
        row = QListWidgetItem(f"{local_clock(str(item.get('occurred_at', '')))}   "
                              f"{event_name(event_type)}{experimental}   ·   {level_name(severity)}")
        row.setData(RECORD_ROLE, dict(item))
        sources = item.get("source_event_ids") or []
        source_note = "\nSự kiện nguồn: " + ", ".join(str(source) for source in sources) if sources else ""
        row.setToolTip(event_description(item) + EVIDENCE_NOTES.get(str(item.get("evidence_status", "")), "")
                       + source_note)
        row.setForeground(QColor("#ffbf69" if severity in {"HIGH", "CRITICAL"} else "#8bd3dd"))
        self.list.insertItem(0, row)
        self.rows[str(item.get("id", ""))] = row
        while self.list.count() > MAX_ROWS:
            removed = self.list.takeItem(self.list.count() - 1)
            self.rows = {key: value for key, value in self.rows.items() if value is not removed}


__all__ = ["EventListPanel"]
