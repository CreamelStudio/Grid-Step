from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from .calibration import compute_homography, default_floor_corners, warp_floor
from .camera_manager import CameraFrame, LatestFrameCamera, make_placeholder_frame
from .config import AppSettings, DEFAULT_SETTINGS_PATH, load_settings, save_settings
from .detector import DetectionResult, FloorDetector
from .floor_plane import analyze_floor_plane
from .grid_mapper import GridMapper
from .output_manager import OutputManager
from .setup_wizard import SetupWizard
from .state_manager import GridStateManager
from .utils import FpsCounter, PerfStats, measure_ms


def bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


class CameraCanvas(QWidget):
    points_changed = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(520, 360)
        self.frame: np.ndarray | None = None
        self.points: list[list[float]] = []
        self._drag_index: int | None = None
        self._image_rect = QRectF()
        self.setMouseTracking(True)

    def set_frame(self, frame: np.ndarray, points: list[list[float]]) -> None:
        self.frame = frame
        self.points = [list(p) for p in points]
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0f172a"))
        if self.frame is None:
            painter.setPen(QColor("#dbeafe"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for camera")
            return

        pixmap = bgr_to_pixmap(self.frame)
        self._image_rect = self._fit_rect(pixmap.width(), pixmap.height())
        painter.drawPixmap(self._image_rect.toRect(), pixmap)

        if len(self.points) >= 2:
            widget_points = [self._image_to_widget(p) for p in self.points]
            painter.setRenderHint(QPainter.Antialiasing, True)
            if len(widget_points) == 4:
                painter.setBrush(QColor(56, 189, 248, 48))
                painter.setPen(QPen(QColor("#38bdf8"), 2))
                painter.drawPolygon(widget_points)
            else:
                painter.setPen(QPen(QColor("#38bdf8"), 2))
                for a, b in zip(widget_points, widget_points[1:]):
                    painter.drawLine(a, b)

            for idx, point in enumerate(widget_points):
                painter.setBrush(QColor("#f8fafc"))
                painter.setPen(QPen(QColor("#0ea5e9"), 3))
                painter.drawEllipse(point, 8, 8)
                painter.setPen(QColor("#0f172a"))
                painter.drawText(point + QPointF(11, -7), str(idx + 1))

        painter.setPen(QColor("#e2e8f0"))
        painter.drawText(16, 24, "Click four floor corners. Drag handles to refine.")

    def mousePressEvent(self, event) -> None:
        if self.frame is None or event.button() != Qt.LeftButton:
            return
        image_point = self._widget_to_image(event.position())
        if image_point is None:
            return
        nearest = self._nearest_point(event.position())
        if nearest is not None:
            self._drag_index = nearest
        elif len(self.points) < 4:
            self.points.append([image_point.x(), image_point.y()])
            self._drag_index = len(self.points) - 1
            self.points_changed.emit(self.points)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_index is None:
            return
        image_point = self._widget_to_image(event.position())
        if image_point is None:
            return
        self.points[self._drag_index] = [image_point.x(), image_point.y()]
        self.points_changed.emit(self.points)
        self.update()

    def mouseReleaseEvent(self, _event) -> None:
        self._drag_index = None

    def reset_points(self) -> None:
        self.points = []
        self.points_changed.emit(self.points)
        self.update()

    def _nearest_point(self, position: QPointF) -> int | None:
        for idx, point in enumerate(self.points):
            widget_point = self._image_to_widget(point)
            if (widget_point - position).manhattanLength() <= 20:
                return idx
        return None

    def _fit_rect(self, image_w: int, image_h: int) -> QRectF:
        if image_w <= 0 or image_h <= 0:
            return QRectF()
        scale = min(self.width() / image_w, self.height() / image_h)
        width = image_w * scale
        height = image_h * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _image_to_widget(self, point: list[float]) -> QPointF:
        if self.frame is None or self._image_rect.width() <= 0:
            return QPointF()
        height, width = self.frame.shape[:2]
        return QPointF(
            self._image_rect.left() + point[0] * self._image_rect.width() / width,
            self._image_rect.top() + point[1] * self._image_rect.height() / height,
        )

    def _widget_to_image(self, point: QPointF) -> QPointF | None:
        if self.frame is None or not self._image_rect.contains(point):
            return None
        height, width = self.frame.shape[:2]
        x = (point.x() - self._image_rect.left()) * width / self._image_rect.width()
        y = (point.y() - self._image_rect.top()) * height / self._image_rect.height()
        return QPointF(max(0, min(width - 1, x)), max(0, min(height - 1, y)))


class TopDownCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(520, 360)
        self.frame: np.ndarray | None = None
        self.confidence: np.ndarray | None = None
        self.states: np.ndarray | None = None
        self.rows = 4
        self.columns = 4
        self.status = "Floor Plane Preview"
        self._image_rect = QRectF()

    def set_data(
        self,
        frame: np.ndarray | None,
        rows: int,
        columns: int,
        confidence: np.ndarray | None = None,
        states: np.ndarray | None = None,
        status: str = "",
    ) -> None:
        self.frame = frame
        self.rows = rows
        self.columns = columns
        self.confidence = confidence
        self.states = states
        if status:
            self.status = status
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#020617"))
        if self.frame is None:
            painter.setPen(QColor("#dbeafe"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Calibrated floor preview")
            return

        pixmap = bgr_to_pixmap(self.frame)
        scale = min(self.width() / pixmap.width(), self.height() / pixmap.height())
        width = pixmap.width() * scale
        height = pixmap.height() * scale
        self._image_rect = QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)
        painter.drawPixmap(self._image_rect.toRect(), pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._draw_grid(painter)
        painter.setPen(QColor("#e2e8f0"))
        painter.drawText(16, 24, self.status)

    def _draw_grid(self, painter: QPainter) -> None:
        rect = self._image_rect
        cell_w = rect.width() / max(self.columns, 1)
        cell_h = rect.height() / max(self.rows, 1)
        for row in range(self.rows):
            for col in range(self.columns):
                x = rect.left() + col * cell_w
                y = rect.top() + row * cell_h
                cell_rect = QRectF(x, y, cell_w, cell_h)
                conf = 0.0 if self.confidence is None else float(self.confidence[row, col])
                state = "" if self.states is None else str(self.states[row, col])
                if state == "PRESSED" or conf >= 0.5:
                    painter.fillRect(cell_rect, QColor(34, 197, 94, 105))
                elif conf > 0.15:
                    painter.fillRect(cell_rect, QColor(250, 204, 21, 70))
                painter.setPen(QPen(QColor("#e0f2fe"), 1))
                painter.drawRect(cell_rect)
                painter.setPen(QColor("#f8fafc"))
                painter.drawText(cell_rect.adjusted(5, 4, -5, -4), Qt.AlignTop | Qt.AlignLeft, f"{row},{col}")
                if conf > 0.01:
                    painter.drawText(cell_rect.adjusted(5, 4, -5, -4), Qt.AlignBottom | Qt.AlignRight, f"{conf:.2f}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.camera = LatestFrameCamera()
        self.detector = FloorDetector(self.settings)
        self.state_manager = GridStateManager(
            self.settings.grid_rows,
            self.settings.grid_columns,
            self.settings.smoothing_frames,
            self.settings.release_frames,
        )
        self.output = OutputManager()
        self.background: np.ndarray | None = None
        self.current_topdown: np.ndarray | None = None
        self.current_confidence = np.zeros((self.settings.grid_rows, self.settings.grid_columns), dtype=np.float32)
        self.detection_running = False
        self.has_camera_frame = False
        self.camera_fps = FpsCounter()
        self.processing_fps = FpsCounter()
        self.perf = PerfStats()

        self.setWindowTitle("Grid Step")
        self.resize(1320, 860)
        self._build_ui()
        self._apply_settings_to_widgets()

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def closeEvent(self, event) -> None:
        self.camera.stop()
        self.detector.close()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        self.step_label = QLabel("Camera > Floor Area > Floor Plane > Grid > Background > Detection")
        self.step_label.setAlignment(Qt.AlignCenter)
        self.step_label.setStyleSheet("font-weight: 700; color: #dbeafe; padding: 8px;")
        main.addWidget(self.step_label)

        view_row = QHBoxLayout()
        self.camera_canvas = CameraCanvas()
        self.topdown_canvas = TopDownCanvas()
        self.camera_canvas.points_changed.connect(self._set_floor_corners)
        view_row.addWidget(self.camera_canvas, 1)
        view_row.addWidget(self.topdown_canvas, 1)
        main.addLayout(view_row, 1)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #cbd5e1; padding: 4px;")
        main.addWidget(self.status_label)

        main.addWidget(self._build_controls())
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0f172a; color: #f8fafc; }
            QGroupBox { border: 1px solid #334155; border-radius: 6px; margin-top: 8px; padding-top: 14px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background: #182235; color: #f8fafc; border: 1px solid #334155; border-radius: 5px; padding: 5px;
            }
            QPushButton:hover { background: #243047; }
            QCheckBox { spacing: 8px; }
            """
        )

    def _build_controls(self) -> QWidget:
        box = QGroupBox("Controls")
        layout = QGridLayout(box)

        self.camera_combo = QComboBox()
        for cam_id in range(4):
            self.camera_combo.addItem(f"Camera {cam_id}", cam_id)

        self.resolution_combo = QComboBox()
        for item in ["640x480", "1280x720", "1920x1080"]:
            self.resolution_combo.addItem(item)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(15, 120)

        self.start_camera_btn = QPushButton("Start Camera")
        self.stop_camera_btn = QPushButton("Stop")
        self.wizard_btn = QPushButton("Setup Wizard")
        self.reset_btn = QPushButton("Reset Corners")
        self.capture_bg_btn = QPushButton("Capture Background")
        self.start_detection_btn = QPushButton("Start Detection")
        self.save_btn = QPushButton("Save Settings")
        self.load_btn = QPushButton("Load Settings")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Pixel", "Shadow", "Hybrid"])
        self.ai_check = QCheckBox("AI Detection")
        self.model_path = QLineEdit()
        self.browse_model_btn = QPushButton("Browse")

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 16)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 16)
        self.pixel_threshold_spin = QSpinBox()
        self.pixel_threshold_spin.setRange(1, 255)
        self.shadow_threshold_spin = QSpinBox()
        self.shadow_threshold_spin.setRange(1, 255)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 10000)
        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(1, 10)
        self.floor_w_spin = QDoubleSpinBox()
        self.floor_w_spin.setRange(0.1, 100.0)
        self.floor_w_spin.setSuffix(" m")
        self.floor_h_spin = QDoubleSpinBox()
        self.floor_h_spin.setRange(0.1, 100.0)
        self.floor_h_spin.setSuffix(" m")

        widgets = [
            ("Camera", self.camera_combo),
            ("Resolution", self.resolution_combo),
            ("FPS", self.fps_spin),
            ("Rows", self.rows_spin),
            ("Columns", self.cols_spin),
            ("Mode", self.mode_combo),
            ("Pixel Threshold", self.pixel_threshold_spin),
            ("Shadow Threshold", self.shadow_threshold_spin),
            ("Min Area", self.min_area_spin),
            ("Smoothing", self.smoothing_spin),
            ("Floor Width", self.floor_w_spin),
            ("Floor Height", self.floor_h_spin),
        ]
        for idx, (label, widget) in enumerate(widgets):
            row = idx // 6
            col = (idx % 6) * 2
            layout.addWidget(QLabel(label), row, col)
            layout.addWidget(widget, row, col + 1)

        layout.addWidget(self.ai_check, 2, 0, 1, 2)
        layout.addWidget(QLabel("ONNX Model"), 2, 2)
        layout.addWidget(self.model_path, 2, 3, 1, 4)
        layout.addWidget(self.browse_model_btn, 2, 7)

        buttons = [
            self.start_camera_btn,
            self.stop_camera_btn,
            self.wizard_btn,
            self.reset_btn,
            self.capture_bg_btn,
            self.start_detection_btn,
            self.save_btn,
            self.load_btn,
        ]
        for idx, button in enumerate(buttons):
            layout.addWidget(button, 3, idx)

        self.start_camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn.clicked.connect(self.camera.stop)
        self.wizard_btn.clicked.connect(self.open_wizard)
        self.reset_btn.clicked.connect(self.reset_corners)
        self.capture_bg_btn.clicked.connect(self.capture_background)
        self.start_detection_btn.clicked.connect(self.toggle_detection)
        self.save_btn.clicked.connect(self.save_settings)
        self.load_btn.clicked.connect(self.load_settings)
        self.browse_model_btn.clicked.connect(self.browse_model)
        for widget in [
            self.rows_spin,
            self.cols_spin,
            self.pixel_threshold_spin,
            self.shadow_threshold_spin,
            self.min_area_spin,
            self.smoothing_spin,
            self.floor_w_spin,
            self.floor_h_spin,
        ]:
            widget.valueChanged.connect(self._read_widgets_to_settings)
        self.mode_combo.currentTextChanged.connect(self._read_widgets_to_settings)
        self.ai_check.toggled.connect(self._read_widgets_to_settings)
        self.model_path.textChanged.connect(self._read_widgets_to_settings)
        return box

    def _apply_settings_to_widgets(self) -> None:
        self.fps_spin.setValue(self.settings.target_fps)
        self.rows_spin.setValue(self.settings.grid_rows)
        self.cols_spin.setValue(self.settings.grid_columns)
        self.mode_combo.setCurrentText(self.settings.detection_mode if self.settings.detection_mode in ["Pixel", "Shadow", "Hybrid"] else "Hybrid")
        self.ai_check.setChecked(self.settings.ai_detection)
        self.model_path.setText(self.settings.onnx_model_path)
        self.pixel_threshold_spin.setValue(self.settings.pixel_threshold)
        self.shadow_threshold_spin.setValue(self.settings.shadow_threshold)
        self.min_area_spin.setValue(self.settings.min_area)
        self.smoothing_spin.setValue(self.settings.smoothing_frames)
        self.floor_w_spin.setValue(self.settings.floor_width_m)
        self.floor_h_spin.setValue(self.settings.floor_height_m)
        resolution = f"{self.settings.resolution_width}x{self.settings.resolution_height}"
        index = self.resolution_combo.findText(resolution)
        if index >= 0:
            self.resolution_combo.setCurrentIndex(index)

    def _read_widgets_to_settings(self) -> None:
        resolution = self.resolution_combo.currentText().split("x")
        self.settings.camera_id = int(self.camera_combo.currentData())
        self.settings.resolution_width = int(resolution[0])
        self.settings.resolution_height = int(resolution[1])
        self.settings.target_fps = self.fps_spin.value()
        self.settings.grid_rows = self.rows_spin.value()
        self.settings.grid_columns = self.cols_spin.value()
        self.settings.detection_mode = self.mode_combo.currentText()
        self.settings.ai_detection = self.ai_check.isChecked()
        self.settings.onnx_model_path = self.model_path.text()
        self.settings.pixel_threshold = self.pixel_threshold_spin.value()
        self.settings.shadow_threshold = self.shadow_threshold_spin.value()
        self.settings.min_area = self.min_area_spin.value()
        self.settings.smoothing_frames = self.smoothing_spin.value()
        self.settings.floor_width_m = self.floor_w_spin.value()
        self.settings.floor_height_m = self.floor_h_spin.value()
        self.state_manager.smoothing_frames = self.settings.smoothing_frames
        self.state_manager.resize(self.settings.grid_rows, self.settings.grid_columns)
        self.detector.update_settings(self.settings)

    def start_camera(self) -> None:
        self._read_widgets_to_settings()
        self.detection_running = False
        self.background = None
        self.start_detection_btn.setText("Start Detection")
        self.state_manager.resize(self.settings.grid_rows, self.settings.grid_columns)
        ok = self.camera.start(
            self.settings.camera_id,
            self.settings.resolution,
            self.settings.target_fps,
        )
        if not ok:
            QMessageBox.warning(self, "Camera", self.camera.last_error or "Camera could not be opened")

    def open_wizard(self) -> None:
        wizard = SetupWizard(self)
        wizard.reset_requested.connect(self.reset_corners)
        wizard.capture_background_requested.connect(self.capture_background)
        wizard.start_detection_requested.connect(lambda: self.toggle_detection(force_start=True))
        wizard.exec()

    def reset_corners(self) -> None:
        self.settings.floor_corners = []
        self.camera_canvas.reset_points()

    def capture_background(self) -> None:
        if not self.has_camera_frame:
            QMessageBox.warning(self, "Background", "Start the camera and wait for a live frame before capturing background.")
            return
        if len(self.settings.floor_corners) != 4:
            QMessageBox.warning(self, "Background", "Select four floor corners before capturing background.")
            return
        if self.current_topdown is None:
            QMessageBox.warning(self, "Background", "Calibrate the floor area before capturing background.")
            return
        self.background = self._capture_stable_background()
        self.current_confidence = np.zeros((self.settings.grid_rows, self.settings.grid_columns), dtype=np.float32)
        self.state_manager.resize(self.settings.grid_rows, self.settings.grid_columns)
        self.status_label.setText("Stable background captured from calibrated top-down floor.")

    def toggle_detection(self, force_start: bool = False) -> None:
        if self.detection_running and not force_start:
            self.detection_running = False
            self.start_detection_btn.setText("Start Detection")
            return
        if self.background is None:
            QMessageBox.warning(self, "Detection", "Capture an empty floor background before starting detection.")
            return
        if not self.has_camera_frame or self.current_topdown is None:
            QMessageBox.warning(self, "Detection", "Start the camera and calibrate the floor before starting detection.")
            return
        if self.background.shape != self.current_topdown.shape:
            QMessageBox.warning(self, "Detection", "Camera or calibration changed. Capture the empty floor background again.")
            return
        self._read_widgets_to_settings()
        self.state_manager.resize(self.settings.grid_rows, self.settings.grid_columns)
        self.detection_running = True
        self.start_detection_btn.setText("Stop Detection")

    def save_settings(self) -> None:
        self._read_widgets_to_settings()
        save_settings(self.settings, DEFAULT_SETTINGS_PATH)
        self.status_label.setText(f"Settings saved to {DEFAULT_SETTINGS_PATH}")

    def load_settings(self) -> None:
        self.settings = load_settings(DEFAULT_SETTINGS_PATH)
        self.detector.update_settings(self.settings)
        self._apply_settings_to_widgets()
        self.status_label.setText("Settings loaded.")

    def browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select ONNX model", str(Path.cwd()), "ONNX Models (*.onnx)")
        if path:
            self.model_path.setText(path)

    def _set_floor_corners(self, points: list[list[float]]) -> None:
        old_corners = self.settings.floor_corners
        self.settings.floor_corners = [[float(x), float(y)] for x, y in points]
        if self.background is not None and old_corners != self.settings.floor_corners:
            self.background = None
            self.detection_running = False
            self.start_detection_btn.setText("Start Detection")
            self.status_label.setText("Floor corners changed. Capture the empty floor background again.")

    def _capture_stable_background(self, samples: int = 8, timeout_s: float = 1.0) -> np.ndarray:
        if self.current_topdown is None:
            raise RuntimeError("No calibrated frame to capture")

        try:
            homography, _ = compute_homography(self.settings.floor_corners, self.settings.topdown_size)
        except Exception:
            return self.current_topdown.copy()

        frames: list[np.ndarray] = []
        seen_index: int | None = None
        deadline = time.perf_counter() + timeout_s
        while len(frames) < samples and time.perf_counter() < deadline:
            frame_obj = self.camera.read_latest()
            if frame_obj is not None and frame_obj.index != seen_index:
                seen_index = frame_obj.index
                frames.append(warp_floor(frame_obj.image, homography, self.settings.topdown_size))
            QApplication.processEvents()
            time.sleep(0.015)

        if len(frames) < 2:
            return self.current_topdown.copy()
        return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

    def _tick(self) -> None:
        start = time.perf_counter()
        frame_obj = self.camera.read_latest()
        if frame_obj is None:
            self.has_camera_frame = False
            frame = make_placeholder_frame(*self.settings.resolution)
        else:
            self.has_camera_frame = True
            frame = frame_obj.image
            self.perf.camera_fps = self.camera_fps.tick(frame_obj.timestamp)

        if not self.settings.floor_corners and frame_obj is not None:
            self.settings.floor_corners = default_floor_corners(frame.shape)

        self.camera_canvas.set_frame(frame, self.settings.floor_corners)
        self._update_topdown(frame)
        self.perf.render_ms = (time.perf_counter() - start) * 1000.0

    def _update_topdown(self, frame: np.ndarray) -> None:
        status = "Select four floor corners"
        if len(self.settings.floor_corners) == 4:
            try:
                homography, ordered = compute_homography(self.settings.floor_corners, self.settings.topdown_size)
                self.settings.floor_corners = ordered.tolist()
                self.settings.homography_matrix = homography.tolist()
                self.current_topdown = warp_floor(frame, homography, self.settings.topdown_size)
                plane = analyze_floor_plane(
                    self.settings.floor_corners,
                    self.settings.grid_rows,
                    self.settings.grid_columns,
                    self.settings.floor_width_m,
                    self.settings.floor_height_m,
                )
                status = (
                    f"Floor Plane Calibrated | quality {plane.quality_score}/100 | "
                    f"cell {plane.cell_width_m:.2f}m x {plane.cell_height_m:.2f}m"
                )
                if plane.warning:
                    status += f" | {plane.warning}"
            except Exception as exc:
                self.current_topdown = None
                status = f"Calibration failed: {exc}"
        else:
            self.current_topdown = None

        detection_result: DetectionResult | None = None
        if self.detection_running and self.current_topdown is not None and self.background is not None:
            with measure_ms({}, "ignored"):
                detection_result = self.detector.process(self.current_topdown, self.background)
            self.perf.processing_fps = self.processing_fps.tick()
            self.perf.pixel_ms = detection_result.timings_ms.get("pixel_ms", 0.0)
            self.perf.shadow_ms = detection_result.timings_ms.get("shadow_ms", 0.0)
            self.perf.onnx_ms = detection_result.timings_ms.get("onnx_ms", 0.0)
            self.current_confidence = detection_result.final_confidence
            events = self.state_manager.update(self.current_confidence, 0.5)
            state = self.state_manager.as_json(self.current_confidence)
            self.output.publish(state, events.pressed, events.released)
            if detection_result.warning:
                status += f" | {detection_result.warning}"

        states = self.state_manager.states
        pressed = self.state_manager.pressed_cells()
        if self.detection_running:
            status += f" | mode {self.settings.detection_mode} | pressed {pressed if pressed else 'none'}"
        elif self.background is None:
            status += " | background not captured"
        else:
            status += " | background ready"
        confidence = self.current_confidence if detection_result is not None else None
        self.topdown_canvas.set_data(
            self.current_topdown,
            self.settings.grid_rows,
            self.settings.grid_columns,
            confidence,
            states,
            status,
        )
        self.status_label.setText(
            f"{status} | Camera FPS {self.perf.camera_fps:.1f} | Processing FPS {self.perf.processing_fps:.1f} | "
            f"Pixel {self.perf.pixel_ms:.1f}ms | Shadow {self.perf.shadow_ms:.1f}ms | ONNX {self.perf.onnx_ms:.1f}ms | "
            f"Render {self.perf.render_ms:.1f}ms"
        )


def run_app() -> int:
    app = QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
