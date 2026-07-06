from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Full, Queue
from typing import Any

import cv2
import numpy as np


@dataclass
class OnnxDetectionResult:
    boxes: list[tuple[float, float, float, float, float]] = field(default_factory=list)
    mask: np.ndarray | None = None
    latency_ms: float = 0.0
    warning: str = ""
    timestamp: float = 0.0


class AsyncOnnxDetector:
    """Optional non-blocking ONNX detector.

    The main OpenCV pipeline never waits for this worker. When AI detection is
    enabled, frames are offered to a size-1 queue and stale inference requests
    are dropped.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.enabled = False
        self.warning = ""
        self._session: Any | None = None
        self._input_name = ""
        self._input_shape: tuple[int, int] = (320, 320)
        self._queue: Queue[np.ndarray] = Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._latest = OnnxDetectionResult(warning="AI detection is off")

    def start(self) -> bool:
        if self.enabled:
            return True
        model = Path(self.model_path)
        if not model.exists():
            self.warning = "ONNX model not found"
            self._latest = OnnxDetectionResult(warning=self.warning)
            return False
        try:
            import onnxruntime as ort
        except Exception:
            self.warning = "ONNX Runtime is not installed"
            self._latest = OnnxDetectionResult(warning=self.warning)
            return False

        try:
            self._session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            input_meta = self._session.get_inputs()[0]
            self._input_name = input_meta.name
            shape = input_meta.shape
            if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                self._input_shape = (shape[3], shape[2])
        except Exception as exc:
            self.warning = f"ONNX model load failed: {exc}"
            self._latest = OnnxDetectionResult(warning=self.warning)
            return False

        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="grid-step-onnx", daemon=True)
        self._thread.start()
        self.enabled = True
        self.warning = ""
        return True

    def stop(self) -> None:
        self.enabled = False
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._session = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break
        self._latest = OnnxDetectionResult(warning="AI detection is off")

    def submit(self, frame_bgr: np.ndarray) -> None:
        if not self.enabled:
            return
        item = frame_bgr.copy()
        try:
            self._queue.put_nowait(item)
        except Full:
            try:
                self._queue.get_nowait()
            except Exception:
                pass
            try:
                self._queue.put_nowait(item)
            except Full:
                pass

    def latest(self) -> OnnxDetectionResult:
        with self._lock:
            result = self._latest
            mask = None if result.mask is None else result.mask.copy()
            return OnnxDetectionResult(list(result.boxes), mask, result.latency_ms, result.warning, result.timestamp)

    def _loop(self) -> None:
        while self._running.is_set():
            try:
                frame = self._queue.get(timeout=0.1)
            except Exception:
                continue
            start = time.perf_counter()
            result = self._infer(frame)
            result.latency_ms = (time.perf_counter() - start) * 1000.0
            result.timestamp = time.perf_counter()
            with self._lock:
                self._latest = result

    def _infer(self, frame_bgr: np.ndarray) -> OnnxDetectionResult:
        if self._session is None:
            return OnnxDetectionResult(warning="ONNX session is not ready")

        original_h, original_w = frame_bgr.shape[:2]
        input_w, input_h = self._input_shape
        resized = cv2.resize(frame_bgr, (input_w, input_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]

        try:
            outputs = self._session.run(None, {self._input_name: tensor})
        except Exception as exc:
            return OnnxDetectionResult(warning=f"ONNX inference failed: {exc}")

        return self._parse_outputs(outputs, (original_w, original_h), (input_w, input_h))

    def _parse_outputs(
        self,
        outputs: list[np.ndarray],
        original_size: tuple[int, int],
        input_size: tuple[int, int],
    ) -> OnnxDetectionResult:
        boxes: list[tuple[float, float, float, float, float]] = []
        mask: np.ndarray | None = None
        original_w, original_h = original_size
        input_w, input_h = input_size

        for output in outputs:
            arr = np.asarray(output)
            squeezed = np.squeeze(arr)
            if squeezed.ndim == 2 and squeezed.shape[-1] >= 5:
                for row in squeezed:
                    conf = float(row[4])
                    if conf < self.confidence_threshold:
                        continue
                    x0, y0, x1, y1 = [float(v) for v in row[:4]]
                    if max(x0, y0, x1, y1) <= 1.5:
                        x0 *= original_w
                        x1 *= original_w
                        y0 *= original_h
                        y1 *= original_h
                    else:
                        x0 *= original_w / input_w
                        x1 *= original_w / input_w
                        y0 *= original_h / input_h
                        y1 *= original_h / input_h
                    boxes.append((x0, y0, x1, y1, conf))
            elif squeezed.ndim == 2:
                normalized = cv2.normalize(squeezed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                mask = cv2.resize(normalized, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        return OnnxDetectionResult(boxes=boxes, mask=mask)
