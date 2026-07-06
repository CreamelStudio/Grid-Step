from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
)


class SetupWizard(QDialog):
    step_changed = Signal(int)
    capture_background_requested = Signal()
    start_detection_requested = Signal()
    reset_requested = Signal()

    STEPS = [
        ("Select Camera", "Select the camera, resolution, and FPS. 60 FPS is preferred for low latency."),
        (
            "Set Floor Area",
            "Click the four real floor corners on the camera image. Drag handles to adjust them.",
        ),
        (
            "Adjust Floor Plane",
            "Check the top-down preview. If the floor is heavily skewed, adjust camera angle or corners.",
        ),
        ("Set Grid", "Choose rows and columns. The preview updates in real time."),
        ("Capture Empty Floor", "Leave the floor empty, then capture the reference background."),
        ("Start Detection", "Pick Pixel, Shadow, or Hybrid mode, then start real-time detection."),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Grid Step Setup Wizard")
        self.setMinimumSize(560, 360)
        self.current_step = 0

        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress = QProgressBar()
        self.progress.setRange(0, len(self.STEPS) - 1)

        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 26px; font-weight: 700;")

        self.body_label = QLabel()
        self.body_label.setAlignment(Qt.AlignCenter)
        self.body_label.setWordWrap(True)
        self.body_label.setStyleSheet("font-size: 15px; color: #d7dde8;")

        self.back_button = QPushButton("Back")
        self.next_button = QPushButton("Next")
        self.reset_button = QPushButton("Reset")
        self.confirm_button = QPushButton("Confirm")

        button_row = QHBoxLayout()
        button_row.addWidget(self.back_button)
        button_row.addWidget(self.next_button)
        button_row.addStretch(1)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.confirm_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress)
        layout.addStretch(1)
        layout.addWidget(self.title_label)
        layout.addWidget(self.body_label)
        layout.addStretch(1)
        layout.addLayout(button_row)

        self.setStyleSheet(
            """
            QDialog { background: #111827; color: #f8fafc; }
            QProgressBar {
                height: 8px; border: 0; border-radius: 4px; background: #253044;
            }
            QProgressBar::chunk { border-radius: 4px; background: #38bdf8; }
            QPushButton {
                background: #243047; color: #f8fafc; border: 1px solid #40516d;
                border-radius: 6px; padding: 9px 16px;
            }
            QPushButton:hover { background: #334155; }
            """
        )

        self.back_button.clicked.connect(self.back)
        self.next_button.clicked.connect(self.next)
        self.reset_button.clicked.connect(self.reset_requested.emit)
        self.confirm_button.clicked.connect(self.confirm_current_step)
        self._render()

    def next(self) -> None:
        self.current_step = min(self.current_step + 1, len(self.STEPS) - 1)
        self._render()

    def back(self) -> None:
        self.current_step = max(self.current_step - 1, 0)
        self._render()

    def confirm_current_step(self) -> None:
        if self.current_step == 4:
            self.capture_background_requested.emit()
        elif self.current_step == 5:
            self.start_detection_requested.emit()
        else:
            self.next()

    def _render(self) -> None:
        names = [name for name, _ in self.STEPS]
        self.progress_label.setText("  >  ".join(names))
        title, body = self.STEPS[self.current_step]
        self.title_label.setText(f"Step {self.current_step + 1}: {title}")
        self.body_label.setText(body)
        self.progress.setValue(self.current_step)
        self.back_button.setEnabled(self.current_step > 0)
        self.next_button.setEnabled(self.current_step < len(self.STEPS) - 1)
        self.confirm_button.setText(
            "Capture Background" if self.current_step == 4 else "Start Detection" if self.current_step == 5 else "Confirm"
        )
        self.step_changed.emit(self.current_step)
