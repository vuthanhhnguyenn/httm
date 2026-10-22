"""PySide6 operator window: assembles the panels and routes runtime updates to them."""

from __future__ import annotations

import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from proctoring_core.events.review_models import ReviewType
from proctoring_core.types import ExamMode
from proctoring_runtime.persistence import JsonlEventRepository, session_dir
from proctoring_runtime.session_reader import review_record
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .overlay import OverlayState
from .runtime import DesktopRuntime, LatestPreview
from .widgets.calibration_dialog import GazeCalibrationDialog
from .widgets.event_detail import REVIEW_LABELS, EventDetailDialog, load_evidence
from .widgets.event_list import RECORD_ROLE, EventListPanel
from .widgets.preview import CameraPanel
from .widgets.status_panel import HealthPanel, MetricsPanel, RiskPanel
from .widgets.theme import card, set_text_if_changed, stylesheet

STALE_FRAME_MS = 1000
EXAM_MODES: tuple[tuple[ExamMode, str], ...] = ((ExamMode.COMPUTER, "Thi máy"), (ExamMode.PAPER, "Thi giấy"))
SETTINGS_EXAM_MODE = "session/exam_mode"
CALIBRATION_NOTES = {
    "collecting": "Hiệu chỉnh: nhìn thẳng màn hình, giữ đầu yên khoảng 2 giây.",
    "failed": "Chưa hiệu chỉnh được: giữ đầu yên rồi nhấn Hiệu chỉnh hướng đầu lần nữa.",
    "uncalibrated": "Chưa bật cảnh báo quay đầu: hãy nhìn thẳng và nhấn Hiệu chỉnh hướng đầu. "
                    "Điện thoại vẫn được kiểm tra.",
    "calibrated": "Đã hiệu chỉnh hướng đầu. Khung nét đứt là kết quả tạm giữ; AI FPS đo riêng với preview.",
    "suggested": "Đầu thí sinh đã ổn định. Nếu thí sinh đang nhìn thẳng màn hình, nhấn "
                 "‘Chấp nhận hiệu chỉnh đề xuất’; nếu không, yêu cầu nhìn thẳng rồi hiệu chỉnh.",
}
CALIBRATE_LABELS = {"suggested": "Chấp nhận hiệu chỉnh đề xuất"}
GAZE_STATUS_TEXT = {
    "uncalibrated": "Hướng mắt: chưa hiệu chỉnh",
    "collecting": "Hướng mắt: đang hiệu chỉnh",
    "calibrated": "Hướng mắt: đã hiệu chỉnh (thử nghiệm)",
    "failed": "Hướng mắt: hiệu chỉnh thất bại — thử lại",
}
GAZE_FAILURE_TEXT = {
    "CENTER_NOT_OBSERVED": "không đo được mắt khi nhìn tâm màn hình",
    "TARGETS_NOT_DISTINCT": "mắt gần như không đổi giữa các điểm",
    "FIT_RESIDUAL_TOO_HIGH": "các điểm nhìn không nhất quán",
}


class MainWindow(QMainWindow):
    def __init__(self, repository_root: Path) -> None:
        super().__init__()
        self.repository_root = repository_root
        self.preview = LatestPreview()
        self.runtime: DesktopRuntime | None = None
        self._last_frame_id: int | None = None
        self._analysis: dict[str, Any] | None = None
        self._overlay = OverlayState()
        self._calibration_status = "uncalibrated"
        self._metrics: dict[str, Any] = {}
        self._risk_score = 0.0
        self._risk_level = "NORMAL"
        self._risk_history: dict[str, Any] = {}
        self._preview_times: deque[float] = deque()
        self._preview_fps = 0.0
        self._last_analyzers: dict[str, dict[str, str]] = {}
        self._analyzer_performance: dict[str, dict[str, Any]] = {}
        self._last_error = ""
        self._evidence_dialogs: list[QDialog] = []
        self._camera_fault = False
        self._gaze_status = "uncalibrated"
        self._gaze_dialog: GazeCalibrationDialog | None = None
        self._session_id: UUID | None = None

        self.setWindowTitle("Smart Exam Proctoring · Giám sát cục bộ")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(stylesheet())
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._refresh_preview)
        self._timer.start()

    def _build_ui(self) -> None:
        root = QWidget()
        page = QVBoxLayout(root)
        page.setContentsMargins(28, 22, 28, 24)
        page.setSpacing(18)
        page.addLayout(self._build_header())
        page.addWidget(self._build_controls())

        content = QHBoxLayout()
        content.setSpacing(18)
        self.camera_panel = CameraPanel()
        self.camera_card = self.camera_panel.frame
        self.video = self.camera_panel.video
        self.camera_status = self.camera_panel.status
        self.preview_note = self.camera_panel.note
        content.addWidget(self.camera_card, 7)

        self.metrics_panel = MetricsPanel()
        self.metric_values = self.metrics_panel.values
        self.realtime_status = self.metrics_panel.realtime_status
        self.health_panel = HealthPanel()
        self.health_labels = self.health_panel.labels
        self.health_fps_labels = self.health_panel.fps_labels
        self.risk_panel = RiskPanel()
        self.risk_value = self.risk_panel.value
        self.risk_history_label = self.risk_panel.history
        self.event_panel = EventListPanel()
        self.events = self.event_panel.list
        self.events.itemDoubleClicked.connect(self._open_evidence)

        rail = QVBoxLayout()
        rail.setSpacing(14)
        for panel in (self.metrics_panel, self.health_panel, self.risk_panel):
            rail.addWidget(panel.frame)
        rail.addWidget(self.event_panel.frame, 1)
        rail_container = QWidget()
        rail_container.setLayout(rail)
        rail_scroll = QScrollArea()
        rail_scroll.setObjectName("railScroll")
        rail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        rail_scroll.setWidgetResizable(True)
        rail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rail_scroll.setWidget(rail_container)
        content.addWidget(rail_scroll, 3)
        page.addLayout(content, 1)
        self.setCentralWidget(root)

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        heading = QVBoxLayout()
        eyebrow = QLabel("GIÁM THỊ · CHẠY TRÊN MÁY NÀY")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Giám sát phiên thi")
        title.setObjectName("pageTitle")
        heading.addWidget(eyebrow)
        heading.addWidget(title)
        header.addLayout(heading)
        header.addStretch(1)
        self.app_status = QLabel("Sẵn sàng")
        self.app_status.setObjectName("statusBadge")
        header.addWidget(self.app_status)
        return header

    def _build_controls(self) -> QFrame:
        controls = card()
        row = QHBoxLayout(controls)
        label = QLabel("Camera")
        label.setObjectName("cardTitle")
        row.addWidget(label)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 16)
        self.camera_index.setValue(0)
        self.camera_index.setPrefix("Thiết bị ")
        row.addWidget(self.camera_index)
        exam_label = QLabel("Kiểu bài thi")
        exam_label.setObjectName("cardTitle")
        row.addWidget(exam_label)
        self.exam_mode = QComboBox()
        self.exam_mode.setToolTip("Thi giấy: cúi nhìn bài là bình thường. "
                                  "Thi máy: chỉ cúi đầu kéo dài mới cần rà soát. "
                                  "Không đổi được khi phiên đang chạy.")
        for mode, mode_label in EXAM_MODES:
            self.exam_mode.addItem(mode_label, mode.value)
        remembered = str(self._settings().value(SETTINGS_EXAM_MODE, ExamMode.COMPUTER.value))
        index = self.exam_mode.findData(remembered)
        self.exam_mode.setCurrentIndex(index if index >= 0 else 0)
        row.addWidget(self.exam_mode)
        row.addStretch(1)
        self.calibrate_button = QPushButton("Hiệu chỉnh hướng đầu")
        self.calibrate_button.setToolTip("Nhìn thẳng màn hình và giữ đầu yên khoảng 2 giây.")
        self.calibrate_button.setEnabled(False)
        self.calibrate_button.clicked.connect(self._calibrate_head)
        row.addWidget(self.calibrate_button)
        self.gaze_button = QPushButton("Hiệu chỉnh hướng mắt")
        self.gaze_button.setToolTip("Sau khi hiệu chỉnh hướng đầu: thí sinh nhìn lần lượt các chấm "
                                    "trên màn hình (thi máy) hoặc tâm màn hình rồi bài thi (thi giấy).")
        self.gaze_button.setEnabled(False)
        self.gaze_button.clicked.connect(self._calibrate_gaze)
        row.addWidget(self.gaze_button)
        self.start_button = QPushButton("Bắt đầu giám sát")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._start)
        row.addWidget(self.start_button)
        self.stop_button = QPushButton("Dừng phiên")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        row.addWidget(self.stop_button)
        return controls

    @staticmethod
    def _settings() -> QSettings:
        return QSettings("SmartExamProctoring", "Desktop")

    def selected_exam_mode(self) -> ExamMode:
        return ExamMode(str(self.exam_mode.currentData()))

    def _set_idle_controls(self, idle: bool) -> None:
        self.start_button.setEnabled(idle)
        self.stop_button.setEnabled(not idle)
        self.camera_index.setEnabled(idle)
        # The exam mode is part of the session policy snapshot and cannot change mid-session.
        self.exam_mode.setEnabled(idle)

    def _start(self) -> None:
        if self.runtime is not None and self.runtime.isRunning():
            return
        self.event_panel.clear()
        self._analysis = None
        self._overlay = OverlayState()
        self._calibration_status = "uncalibrated"
        self._metrics = {}
        self._last_analyzers = {}
        self._analyzer_performance = {}
        self._last_error = ""
        self._risk_score, self._risk_level = 0.0, "NORMAL"
        self._risk_history = {}
        self._refresh_risk()
        self.preview.clear()
        self._last_frame_id = None
        self._preview_times.clear()
        self._preview_fps = 0.0
        exam_mode = self.selected_exam_mode()
        self._settings().setValue(SETTINGS_EXAM_MODE, exam_mode.value)
        self.runtime = DesktopRuntime(camera_index=self.camera_index.value(), repository_root=self.repository_root,
                                      preview=self.preview, exam_mode=exam_mode)
        self.runtime.state_changed.connect(self._on_state)
        self.runtime.analysis_ready.connect(self._on_analysis)
        self.runtime.event_ready.connect(self._on_event)
        self.runtime.runtime_error.connect(self._on_error)
        self._set_idle_controls(False)
        self.app_status.setText("Đang khởi tạo")
        self.runtime.start()

    def _calibrate_head(self) -> None:
        if self.runtime is not None and self.runtime.isRunning():
            self._calibration_status = "collecting"
            self.runtime.request_head_calibration()

    def _calibrate_gaze(self) -> None:
        runtime = self.runtime
        if runtime is None or not runtime.isRunning():
            return
        runtime.begin_gaze_calibration()
        self._gaze_status = "collecting"
        self._gaze_dialog = GazeCalibrationDialog(
            self.selected_exam_mode(), on_target=runtime.set_gaze_target,
            on_finish=runtime.finish_gaze_calibration, on_cancel=runtime.cancel_gaze_calibration, parent=self)
        self._gaze_dialog.start()

    def _stop(self) -> None:
        if self.runtime is None or not self.runtime.isRunning():
            return
        self.stop_button.setEnabled(False)
        self.calibrate_button.setEnabled(False)
        self.app_status.setText("Đang dừng")
        self.runtime.request_stop()

    def _on_state(self, update: dict[str, Any]) -> None:
        kind = update.get("type")
        if kind == "loading_models":
            self.app_status.setText("Đang tải model AI")
            self.realtime_status.setText("Đang khởi tạo các bộ phân tích…")
        elif kind == "camera_ready":
            self.camera_status.setText("Webcam trực tiếp · đang tải model AI")
        elif kind == "analyzer_health":
            self._last_analyzers = update.get("analyzers", {})
            self._refresh_health()
        elif kind == "running":
            self._session_id = UUID(str(update["session_id"])) if update.get("session_id") else None
            self.app_status.setText("Đang giám sát")
            self.camera_status.setText("Camera đã mở")
            self.calibrate_button.setEnabled(self._last_analyzers.get("face", {}).get("status") == "ready")
        elif kind == "metrics":
            self._metrics = update.get("metrics", {})
            self._refresh_metrics()
        elif kind == "risk":
            self._risk_score = float(update.get("score", self._risk_score))
            self._risk_level = str(update.get("level", self._risk_level))
            self._risk_history = update
            self._refresh_risk()
        elif kind == "gaze_calibration":
            self._gaze_status = str(update.get("status", "uncalibrated"))
            reason = GAZE_FAILURE_TEXT.get(str(update.get("reason", "")), "")
            self.preview_note.setText(GAZE_STATUS_TEXT.get(self._gaze_status, "") + (f": {reason}." if reason else "."))
        elif kind == "camera_fault":
            self._camera_fault = bool(update.get("active"))
            if self._camera_fault:
                self.camera_status.setText("Camera có thể bị treo · tạm dừng đánh giá hành vi")
        elif kind == "stopped":
            self.calibrate_button.setEnabled(False)
            self.gaze_button.setEnabled(False)
            self.app_status.setText("Cần kiểm tra" if self._last_error else "Đã dừng")
            self.camera_status.setText("Camera chưa mở")
            self._set_idle_controls(True)

    def _on_analysis(self, result: dict[str, Any]) -> None:
        self._analysis = result
        self._overlay.update(result)
        faces = result.get("faces", [])
        if faces:
            self._calibration_status = faces[0].get("calibration_status", "uncalibrated")
            set_text_if_changed(self.calibrate_button,
                                CALIBRATE_LABELS.get(self._calibration_status, "Hiệu chỉnh hướng đầu"))
        self._gaze_status = str(result.get("gaze_status", self._gaze_status))
        running = self.runtime is not None and self.runtime.isRunning()
        self.gaze_button.setEnabled(running and self._calibration_status == "calibrated"
                                    and self._gaze_status != "collecting")
        self.gaze_button.setToolTip(GAZE_STATUS_TEXT.get(self._gaze_status, ""))
        self._analyzer_performance = result.get("analyzer_performance", {})
        self._last_analyzers = {
            key: {
                "status": value,
                "reason": self._analyzer_performance.get(key, {}).get(
                    "reason", self._last_analyzers.get(key, {}).get("reason", "")),
                "device": self._last_analyzers.get(key, {}).get("device", ""),
            }
            for key, value in result.get("analyzer_health", {}).items()
        }
        self._refresh_health()

    def _on_event(self, item: dict[str, Any]) -> None:
        if item.get("action") == "evidence":
            self.event_panel.apply_evidence(item)
            return
        self._risk_score = float(item.get("risk_score", self._risk_score))
        self._risk_level = str(item.get("risk_level", self._risk_level))
        self._risk_history = item
        self._refresh_risk()
        if item.get("action") in {"opened", "correlation"}:
            self.event_panel.add(item)

    def _open_evidence(self, row: QListWidgetItem) -> None:
        """Open the review detail: analysed frame (if saved), provenance and review buttons."""

        item = row.data(RECORD_ROLE)
        item = dict(item) if isinstance(item, dict) else {}
        relative = row.data(Qt.ItemDataRole.UserRole)
        pixmap = load_evidence(self.repository_root / "storage", relative if isinstance(relative, str) else None)
        if isinstance(relative, str) and relative and pixmap is None:
            QMessageBox.warning(self, "Ảnh không còn khả dụng", "Không mở được ảnh; có thể ảnh đã hết thời hạn lưu.")
        event_id = str(item.get("id", ""))
        dialog = EventDetailDialog(item, pixmap, parent=self,
                                   on_review=lambda verdict: self._review_event(event_id, item, verdict))
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._evidence_dialogs.append(dialog)
        dialog.finished.connect(lambda _code, current=dialog: self._evidence_dialogs.remove(current))
        dialog.show()

    def _review_event(self, event_id: str, item: dict[str, Any], verdict: ReviewType) -> None:
        if not event_id or self._session_id is None:
            return
        repository = JsonlEventRepository(session_dir(self.repository_root / "storage", self._session_id)
                                          / "events.jsonl")
        repository.ensure_parent()
        repository.append(review_record(event_id, verdict, episode_id=str(item.get("episode_id") or event_id),
                                        reviewed_at=datetime.now(UTC).isoformat()))
        self.event_panel.mark_reviewed(event_id, REVIEW_LABELS[verdict])

    def _on_error(self, message: str) -> None:
        self._last_error = message
        self.calibrate_button.setEnabled(False)
        self.app_status.setText("Cần kiểm tra")
        self.camera_status.setText("Camera chưa mở")
        self.preview_note.setText(message)
        self._set_idle_controls(True)

    def _refresh_health(self) -> None:
        self.health_panel.refresh(self._last_analyzers, self._analyzer_performance)

    def _refresh_metrics(self) -> None:
        self.metrics_panel.refresh(self._metrics, self._preview_fps)

    def _refresh_risk(self) -> None:
        self.risk_panel.refresh(self._risk_score, self._risk_level, self._risk_history)

    def _refresh_preview(self) -> None:
        frame = self.preview.latest()
        if frame is None:
            return
        frame_id = int(frame.get("frame_id", -1))
        captured_at_ms = int(frame.get("captured_at_monotonic_ms", 0))
        frame_age_ms = max(0, time.monotonic_ns() // 1_000_000 - captured_at_ms)
        if frame_id == self._last_frame_id:
            if frame_age_ms > STALE_FRAME_MS:
                set_text_if_changed(self.camera_status, "Khung webcam đã cũ · kiểm tra camera")
            return
        self._last_frame_id = frame_id
        if not self.camera_panel.render(frame, self._overlay):
            return
        now = time.monotonic()
        self._preview_times.append(now)
        while self._preview_times and now - self._preview_times[0] > 2.0:
            self._preview_times.popleft()
        if len(self._preview_times) > 1:
            elapsed = self._preview_times[-1] - self._preview_times[0]
            self._preview_fps = (len(self._preview_times) - 1) / elapsed if elapsed > 0 else 0.0
        self._refresh_metrics()
        if self._camera_fault:
            set_text_if_changed(self.camera_status, "Camera có thể bị treo · tạm dừng đánh giá hành vi")
        elif frame_age_ms > STALE_FRAME_MS:
            set_text_if_changed(self.camera_status, "Khung webcam đã cũ · kiểm tra camera")
        else:
            set_text_if_changed(self.camera_status, f"Đang hiển thị · {self._preview_fps:.1f} FPS")
        set_text_if_changed(self.preview_note, self._last_error or CALIBRATION_NOTES.get(
            self._calibration_status, "Khung AI hiển thị kết quả gần nhất và độ trễ."))

    def closeEvent(self, event: Any) -> None:
        if self.runtime is not None and self.runtime.isRunning():
            self.runtime.request_stop()
            self.runtime.wait(8000)
        super().closeEvent(event)


__all__ = ["MainWindow"]
