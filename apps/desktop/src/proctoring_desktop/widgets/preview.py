"""Live camera card: paints the latest frame with the freshest matching AI overlay."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout

from ..overlay import OverlayState
from .painting import paint_overlay
from .theme import card, muted_label, title_label


class CameraPanel:
    def __init__(self) -> None:
        self.frame = card()
        layout = QVBoxLayout(self.frame)
        header = QHBoxLayout()
        header.addWidget(title_label("Webcam trực tiếp"))
        header.addStretch(1)
        self.status = QLabel("Camera chưa mở")
        self.status.setObjectName("mutedBadge")
        header.addWidget(self.status)
        layout.addLayout(header)
        self.video = QLabel("Nhấn ‘Bắt đầu giám sát’ để mở webcam")
        self.video.setObjectName("videoSurface")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(640, 430)
        layout.addWidget(self.video, 1)
        self.note = muted_label("Hình ảnh và AI được xử lý trên máy này.")
        layout.addWidget(self.note)

    def render(self, frame: dict[str, Any], overlay: OverlayState) -> bool:
        """Paint ``frame``; returns False when it carries no drawable image."""

        image_data = frame.get("image")
        if image_data is None:
            return False
        height, width = image_data.shape[:2]
        image = QImage(image_data.data, width, height, int(image_data.strides[0]),
                       QImage.Format.Format_BGR888).copy()
        if image.isNull():
            return False
        image = image.scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.FastTransformation)
        pixmap = QPixmap.fromImage(image)
        captured_at_ms = int(frame.get("captured_at_monotonic_ms", 0))
        analysis = overlay.snapshot(captured_at_ms)
        if analysis is not None:
            age_ms = captured_at_ms - int(analysis.get("captured_at_monotonic_ms", 0))
            if 0 <= age_ms <= overlay.max_age_ms:
                paint_overlay(pixmap, analysis, age_ms)
        self.video.setPixmap(pixmap)
        return True


__all__ = ["CameraPanel"]
