from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import AppSettings
from .grid_mapper import GridMapper
from .onnx_detector import AsyncOnnxDetector
from .utils import measure_ms


@dataclass
class DetectionResult:
    pixel_confidence: np.ndarray
    shadow_confidence: np.ndarray
    ai_confidence: np.ndarray
    final_confidence: np.ndarray
    pixel_mask: np.ndarray
    shadow_mask: np.ndarray
    ai_mask: np.ndarray
    timings_ms: dict[str, float] = field(default_factory=dict)
    warning: str = ""


class FloorDetector:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.onnx = AsyncOnnxDetector(settings.onnx_model_path, settings.ai_confidence_threshold)
        self._last_ai_enabled = False

    def close(self) -> None:
        self.onnx.stop()

    def update_settings(self, settings: AppSettings) -> None:
        restart_ai = (
            settings.onnx_model_path != self.settings.onnx_model_path
            or settings.ai_confidence_threshold != self.settings.ai_confidence_threshold
        )
        self.settings = settings
        if restart_ai:
            self.onnx.stop()
            self.onnx = AsyncOnnxDetector(settings.onnx_model_path, settings.ai_confidence_threshold)
            self._last_ai_enabled = False

    def process(self, current_bgr: np.ndarray, background_bgr: np.ndarray) -> DetectionResult:
        rows = self.settings.grid_rows
        cols = self.settings.grid_columns
        mode = self.settings.detection_mode.lower()
        use_pixel = mode in ("pixel", "hybrid")
        use_shadow = mode in ("shadow", "hybrid")
        mapper = GridMapper(rows, cols, (current_bgr.shape[1], current_bgr.shape[0]))
        timings: dict[str, float] = {}

        if self.settings.ai_detection and not self.onnx.enabled:
            self.onnx.start()
        elif not self.settings.ai_detection and self.onnx.enabled:
            self.onnx.stop()
        self._last_ai_enabled = self.onnx.enabled

        with measure_ms(timings, "pixel_ms"):
            pixel_mask = self._pixel_diff_mask(current_bgr, background_bgr)
            pixel_warning = self._global_change_warning(pixel_mask, "pixel")
            if use_pixel and pixel_warning:
                pixel_mask = np.zeros_like(pixel_mask)
            pixel_conf = mapper.confidence_from_mask(pixel_mask)

        with measure_ms(timings, "shadow_ms"):
            shadow_mask = self._shadow_mask(current_bgr, background_bgr)
            shadow_warning = self._global_change_warning(shadow_mask, "shadow")
            if use_shadow and shadow_warning:
                shadow_mask = np.zeros_like(shadow_mask)
            shadow_conf = mapper.confidence_from_mask(shadow_mask)

        ai_mask = np.zeros(current_bgr.shape[:2], dtype=np.uint8)
        ai_conf = np.zeros((rows, cols), dtype=np.float32)
        warnings = []
        if use_pixel and pixel_warning:
            warnings.append(pixel_warning)
        if use_shadow and shadow_warning:
            warnings.append(shadow_warning)
        if self.settings.ai_detection:
            self.onnx.submit(current_bgr)
            ai_result = self.onnx.latest()
            timings["onnx_ms"] = ai_result.latency_ms
            if ai_result.warning:
                warnings.append(ai_result.warning)
            if ai_result.mask is not None:
                ai_mask = ai_result.mask
            elif ai_result.boxes:
                ai_mask = mapper.mask_from_boxes(ai_result.boxes)
            ai_conf = mapper.confidence_from_mask(ai_mask)
        else:
            timings["onnx_ms"] = 0.0

        final_conf = self._combine(pixel_conf, shadow_conf, ai_conf)
        return DetectionResult(
            pixel_conf,
            shadow_conf,
            ai_conf,
            final_conf,
            pixel_mask,
            shadow_mask,
            ai_mask,
            timings,
            " | ".join(warnings),
        )

    def _pixel_diff_mask(self, current_bgr: np.ndarray, background_bgr: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(current_bgr, background_bgr)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, self.settings.pixel_threshold, 255, cv2.THRESH_BINARY)
        return self._clean_mask(mask)

    def _shadow_mask(self, current_bgr: np.ndarray, background_bgr: np.ndarray) -> np.ndarray:
        current_lab = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2LAB)
        background_lab = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2LAB)
        current_l = current_lab[:, :, 0].astype(np.int16)
        background_l = background_lab[:, :, 0].astype(np.int16)
        global_delta = int(np.mean(current_l - background_l))
        compensated_background = background_l + global_delta
        darker = compensated_background - current_l
        mask = np.where(darker > self.settings.shadow_threshold, 255, 0).astype(np.uint8)
        return self._clean_mask(mask)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        if self.settings.min_area <= 1:
            return cleaned

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(cleaned)
        for contour in contours:
            if cv2.contourArea(contour) >= self.settings.min_area:
                cv2.drawContours(filtered, [contour], -1, 255, -1)
        return filtered

    def _global_change_warning(self, mask: np.ndarray, label: str) -> str:
        changed_ratio = float(np.count_nonzero(mask) / max(mask.size, 1))
        if changed_ratio >= self.settings.background_change_guard:
            return (
                f"{label} background mismatch {changed_ratio:.0%}; "
                "recapture empty floor or recalibrate corners"
            )
        return ""

    def _combine(self, pixel_conf: np.ndarray, shadow_conf: np.ndarray, ai_conf: np.ndarray) -> np.ndarray:
        mode = self.settings.detection_mode.lower()
        pixel_score = np.clip(pixel_conf / max(self.settings.pixel_cell_threshold, 1e-6), 0.0, 1.0)
        shadow_score = np.clip(shadow_conf / max(self.settings.shadow_cell_threshold, 1e-6), 0.0, 1.0)
        ai_score = np.clip(ai_conf / max(self.settings.ai_confidence_threshold, 1e-6), 0.0, 1.0)

        if mode == "pixel":
            return pixel_score.astype(np.float32)
        if mode == "shadow":
            return shadow_score.astype(np.float32)

        if self.settings.ai_detection:
            final = pixel_score * 0.45 + shadow_score * 0.35 + ai_score * 0.20
        else:
            total = 0.45 + 0.35
            final = pixel_score * (0.45 / total) + shadow_score * (0.35 / total)
        return np.clip(final, 0.0, 1.0).astype(np.float32)
