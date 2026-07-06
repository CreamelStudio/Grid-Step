from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraFrame:
    image: np.ndarray
    timestamp: float
    index: int


class LatestFrameCamera:
    """Reads camera frames on a worker thread and keeps only the newest one."""

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: CameraFrame | None = None
        self._running = threading.Event()
        self._frame_index = 0
        self.last_error = ""

    @staticmethod
    def list_cameras(limit: int = 8) -> list[int]:
        found: list[int] = []
        for camera_id in range(limit):
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                found.append(camera_id)
            cap.release()
        return found

    def start(
        self,
        camera_id: int = 0,
        resolution: tuple[int, int] = (1280, 720),
        fps: int = 60,
    ) -> bool:
        self.stop()
        cap = cv2.VideoCapture(camera_id, cv2.CAP_ANY)
        if not cap.isOpened():
            self.last_error = f"Camera {camera_id} could not be opened"
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._capture = cap
        self._running.set()
        self._thread = threading.Thread(target=self._reader_loop, name="grid-step-camera", daemon=True)
        self._thread.start()
        self.last_error = ""
        return True

    def stop(self) -> None:
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._capture is not None:
            self._capture.release()
        self._capture = None

    def read_latest(self) -> Optional[CameraFrame]:
        with self._lock:
            if self._latest is None:
                return None
            return CameraFrame(self._latest.image.copy(), self._latest.timestamp, self._latest.index)

    def _reader_loop(self) -> None:
        while self._running.is_set() and self._capture is not None:
            ok, frame = self._capture.read()
            if not ok:
                self.last_error = "Camera frame read failed"
                time.sleep(0.01)
                continue
            self._frame_index += 1
            with self._lock:
                self._latest = CameraFrame(frame, time.perf_counter(), self._frame_index)

    def __enter__(self) -> "LatestFrameCamera":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def make_placeholder_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (28, 28, 32)
    cv2.putText(
        image,
        "No camera frame",
        (40, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (210, 210, 220),
        2,
        cv2.LINE_AA,
    )
    return image
