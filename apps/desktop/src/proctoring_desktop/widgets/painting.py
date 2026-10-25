"""Draw analysed boxes, landmarks and the AI age badge on a preview pixmap."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap

from .theme import COLORS

POSE_CONNECTIONS = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
                    (23, 25), (25, 27), (24, 26), (26, 28))
OBJECT_NAMES = {"person": "Người", "cell phone": "Điện thoại", "book": "Sách"}


def object_name(label: str) -> str:
    return OBJECT_NAMES.get(label.lower(), label)


def paint_overlay(pixmap: QPixmap, analysis: dict[str, Any], age_ms: int) -> None:
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    width, height = pixmap.width(), pixmap.height()
    for item in analysis.get("objects", []):
        label = str(item.get("class", "vật thể"))
        if label.casefold() not in OBJECT_NAMES:
            continue
        color = (QColor(COLORS["accent"]) if label.lower() == "person"
                 else QColor(COLORS["danger"]) if "phone" in label.lower() else QColor(COLORS["warn"]))
        pending = item.get("track_confirmed") is False
        suffix = " · đang xác nhận" if pending else " · tạm giữ" if item.get("held") else ""
        if pending:
            color = QColor(COLORS["warn"])
        draw_box(painter, item.get("bbox", {}), width, height, color,
                 f"{object_name(label)} · {float(item.get('confidence', 0)):.0%}{suffix}",
                 dashed=pending or bool(item.get("held")))
    for face in analysis.get("faces", []):
        pose = f"Mặt · xoay {float(face.get('yaw', 0)):.0f}° · cúi/ngẩng {float(face.get('pitch', 0)):.0f}°"
        if not face.get("pose_valid", True):
            pose = "Mặt · chưa đo được hướng đầu"
        if face.get("held"):
            pose += " · tạm giữ"
        draw_box(painter, face.get("bbox", {}), width, height, QColor(COLORS["cyan"]), pose, thickness=2,
                 dashed=bool(face.get("held")))
    for pose in analysis.get("poses", []):
        draw_pose(painter, pose.get("landmarks", []), width, height, dashed=bool(pose.get("held")))
    for hand in analysis.get("hands", []):
        held = bool(hand.get("held"))
        draw_box(painter, hand.get("bbox", {}), width, height, QColor(COLORS["violet"]),
                 "Bàn tay · tạm giữ" if held else "Bàn tay", dashed=held)
    painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
    text = f"AI · khung {analysis.get('frame_id', '—')} · trễ {max(0, age_ms)} ms"
    badge = QRectF(14, 14, painter.fontMetrics().horizontalAdvance(text) + 36, 26)
    path = QPainterPath()
    path.addRoundedRect(badge, 13, 13)
    painter.fillPath(path, QColor(10, 12, 15, 200))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(COLORS["accent"]))
    painter.drawEllipse(badge.left() + 14, badge.center().y() - 3, 6, 6)
    painter.setPen(QColor(COLORS["text"]))
    painter.drawText(badge.adjusted(26, 0, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
    painter.end()


def draw_box(painter: QPainter, bbox: dict[str, Any], width: int, height: int, color: QColor, label: str, *,
             thickness: int = 2, dashed: bool = False) -> None:
    x = max(0.0, min(1.0, float(bbox.get("x", 0))))
    y = max(0.0, min(1.0, float(bbox.get("y", 0))))
    box_width = max(0.0, min(1.0 - x, float(bbox.get("width", 0))))
    box_height = max(0.0, min(1.0 - y, float(bbox.get("height", 0))))
    if box_width <= 0 or box_height <= 0:
        return
    left, top = round(x * width), round(y * height)
    rect_width, rect_height = round(box_width * width), round(box_height * height)
    # A faint full outline plus solid corner brackets keeps the subject readable under the box.
    outline = QColor(color)
    outline.setAlpha(150)
    pen = QPen(outline, 1)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(left, top, rect_width, rect_height)
    if not dashed:
        corner = max(6, min(18, rect_width // 4, rect_height // 4))
        bracket = QPen(color, thickness + 1)
        bracket.setCapStyle(Qt.PenCapStyle.SquareCap)
        painter.setPen(bracket)
        right, bottom = left + rect_width, top + rect_height
        for x, y, dx, dy in ((left, top, 1, 1), (right, top, -1, 1), (left, bottom, 1, -1), (right, bottom, -1, -1)):
            painter.drawLine(x, y, x + dx * corner, y)
            painter.drawLine(x, y, x, y + dy * corner)
    painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
    text_width = painter.fontMetrics().horizontalAdvance(label) + 16
    text_y = max(26, top - 4)
    tag = QRectF(left, text_y - 22, text_width, 22)
    path = QPainterPath()
    path.addRoundedRect(tag, 6, 6)
    painter.fillPath(path, QColor(color.red(), color.green(), color.blue(), 235))
    painter.setPen(QColor(COLORS["accent_ink"]))
    painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, label)


def draw_pose(painter: QPainter, landmarks: list[dict[str, Any]], width: int, height: int, *,
              dashed: bool = False) -> None:
    points = {int(item.get("index", -1)): item for item in landmarks if float(item.get("visibility", 0)) >= 0.35}
    colour = QColor(COLORS["accent"])
    colour.setAlpha(200)
    pen = QPen(colour, 2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    for start, end in POSE_CONNECTIONS:
        if start not in points or end not in points:
            continue
        a, b = points[start], points[end]
        painter.drawLine(round(float(a["x"]) * width), round(float(a["y"]) * height),
                         round(float(b["x"]) * width), round(float(b["y"]) * height))


__all__ = ["draw_box", "draw_pose", "object_name", "paint_overlay"]
