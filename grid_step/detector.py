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
    shoe_confidence: np.ndarray
    motion_confidence: np.ndarray
    ai_confidence: np.ndarray
    final_confidence: np.ndarray
    pixel_mask: np.ndarray
    shadow_mask: np.ndarray
    shoe_mask: np.ndarray
    motion_mask: np.ndarray
    ai_mask: np.ndarray
    timings_ms: dict[str, float] = field(default_factory=dict)
    warning: str = ""


class FloorDetector:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.onnx = AsyncOnnxDetector(settings.onnx_model_path, settings.ai_confidence_threshold)
        self._last_ai_enabled = False
        self._previous_gray: np.ndarray | None = None
        self._previous_occupancy: np.ndarray | None = None
        self._motion_spike_hold: np.ndarray | None = None

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
        self._ensure_history_shape(settings.grid_rows, settings.grid_columns)

    def process(self, current_bgr: np.ndarray, background_bgr: np.ndarray) -> DetectionResult:
        rows = self.settings.grid_rows
        cols = self.settings.grid_columns
        mode = self.settings.detection_mode.lower()
        use_pixel = mode in ("pixel", "hybrid", "touch")
        use_shadow = mode in ("shadow", "hybrid", "touch")
        self._ensure_history_shape(rows, cols)
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

        with measure_ms(timings, "shoe_ms"):
            shoe_mask = self._shoe_color_mask(current_bgr)
            shoe_conf = mapper.confidence_from_mask(shoe_mask)

        with measure_ms(timings, "shadow_ms"):
            shadow_mask = self._shadow_mask(current_bgr, background_bgr)
            shadow_mask = self._keep_shadow_near_shoe(shadow_mask, shoe_mask)
            shadow_warning = self._global_change_warning(shadow_mask, "shadow")
            if use_shadow and shadow_warning:
                shadow_mask = np.zeros_like(shadow_mask)
            shadow_conf = mapper.confidence_from_mask(shadow_mask)

        with measure_ms(timings, "motion_ms"):
            motion_mask = self._motion_mask(current_bgr)
            motion_conf = mapper.confidence_from_mask(motion_mask)

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

        final_conf = self._combine(pixel_conf, shadow_conf, shoe_conf, motion_conf, ai_conf)
        return DetectionResult(
            pixel_conf,
            shadow_conf,
            shoe_conf,
            motion_conf,
            ai_conf,
            final_conf,
            pixel_mask,
            shadow_mask,
            shoe_mask,
            motion_mask,
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

    def _motion_mask(self, current_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
        if self._previous_gray is None or self._previous_gray.shape != gray.shape:
            self._previous_gray = gray
            return np.zeros_like(gray, dtype=np.uint8)

        diff = cv2.absdiff(gray, self._previous_gray)
        self._previous_gray = gray
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)
        _, mask = cv2.threshold(blurred, self.settings.motion_threshold, 255, cv2.THRESH_BINARY)
        return self._clean_mask(mask)

    def _shoe_color_mask(self, current_bgr: np.ndarray) -> np.ndarray:
        if len(self.settings.shoe_color_bgr) != 3:
            return np.zeros(current_bgr.shape[:2], dtype=np.uint8)

        target_bgr = np.array([[self.settings.shoe_color_bgr]], dtype=np.uint8)
        target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
        current_lab = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        delta = current_lab - target_lab
        distance = np.sqrt(np.sum(delta * delta, axis=2))
        lab_match = distance <= self.settings.shoe_color_threshold

        target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)[0, 0]
        current_hsv = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
        hue_delta = np.abs(current_hsv[:, :, 0] - int(target_hsv[0]))
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        sat_delta = np.abs(current_hsv[:, :, 1] - int(target_hsv[1]))
        val_delta = np.abs(current_hsv[:, :, 2] - int(target_hsv[2]))

        if int(target_hsv[1]) >= 35:
            hsv_match = (
                (hue_delta <= self.settings.shoe_hue_threshold)
                & (sat_delta <= self.settings.shoe_saturation_threshold)
                & (val_delta <= self.settings.shoe_value_threshold)
            )
            mask_bool = lab_match & hsv_match
        else:
            mask_bool = lab_match & (val_delta <= self.settings.shoe_value_threshold)

        mask = np.where(mask_bool, 255, 0).astype(np.uint8)
        return self._clean_mask(mask)

    def _keep_shadow_near_shoe(self, shadow_mask: np.ndarray, shoe_mask: np.ndarray) -> np.ndarray:
        if len(self.settings.shoe_color_bgr) != 3 or not np.any(shoe_mask):
            return shadow_mask
        radius = max(3, int(self.settings.shadow_near_shoe_radius))
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        near_shoe = cv2.dilate(shoe_mask, kernel, iterations=1)
        return cv2.bitwise_and(shadow_mask, near_shoe)

    def _global_change_warning(self, mask: np.ndarray, label: str) -> str:
        changed_ratio = float(np.count_nonzero(mask) / max(mask.size, 1))
        if changed_ratio >= self.settings.background_change_guard:
            return (
                f"{label} background mismatch {changed_ratio:.0%}; "
                "recapture empty floor or recalibrate corners"
            )
        return ""

    def _combine(
        self,
        pixel_conf: np.ndarray,
        shadow_conf: np.ndarray,
        shoe_conf: np.ndarray,
        motion_conf: np.ndarray,
        ai_conf: np.ndarray,
    ) -> np.ndarray:
        mode = self.settings.detection_mode.lower()
        pixel_score = np.clip(pixel_conf / max(self.settings.pixel_cell_threshold, 1e-6), 0.0, 1.0)
        shadow_score = np.clip(shadow_conf / max(self.settings.shadow_cell_threshold, 1e-6), 0.0, 1.0)
        shoe_score = np.clip(shoe_conf / max(self.settings.shoe_cell_threshold, 1e-6), 0.0, 1.0)
        motion_score = np.clip(motion_conf / max(self.settings.motion_cell_threshold, 1e-6), 0.0, 1.0)
        ai_score = np.clip(ai_conf / max(self.settings.ai_confidence_threshold, 1e-6), 0.0, 1.0)

        if mode == "pixel":
            return pixel_score.astype(np.float32)
        if mode == "shadow":
            return shadow_score.astype(np.float32)
        if mode == "touch":
            return self._touch_score(pixel_score, shadow_score, shoe_score, motion_score)

        if self.settings.ai_detection:
            final = pixel_score * 0.45 + shadow_score * 0.35 + ai_score * 0.20
        else:
            total = 0.45 + 0.35
            final = pixel_score * (0.45 / total) + shadow_score * (0.35 / total)
        return np.clip(final, 0.0, 1.0).astype(np.float32)

    def _touch_score(
        self,
        occupancy_score: np.ndarray,
        shadow_score: np.ndarray,
        shoe_score: np.ndarray,
        motion_score: np.ndarray,
    ) -> np.ndarray:
        self._ensure_history_shape(*occupancy_score.shape)
        assert self._previous_occupancy is not None
        assert self._motion_spike_hold is not None

        has_shoe_color = len(self.settings.shoe_color_bgr) == 3
        if has_shoe_color:
            occupancy_score = np.maximum(occupancy_score * 0.35, shoe_score)
            shoe_gate = shoe_score >= 0.18
        else:
            shoe_gate = occupancy_score >= 0.25

        occupancy_rise = np.clip(occupancy_score - self._previous_occupancy, 0.0, 1.0)
        moving_in_cell = shoe_gate & (motion_score >= 0.30)
        self._motion_spike_hold[moving_in_cell] = self.settings.touch_spike_hold_frames
        self._motion_spike_hold[~moving_in_cell] = np.maximum(self._motion_spike_hold[~moving_in_cell] - 1, 0)

        recently_moved = self._motion_spike_hold > 0
        motion_stopped = (
            recently_moved
            & shoe_gate
            & (motion_score <= self.settings.motion_stop_threshold)
        )
        landing_spike = shoe_gate & (occupancy_rise >= 0.16) & (motion_score >= 0.16)
        stable_contact = shoe_gate & (shadow_score >= 0.20) & (motion_score <= 0.20)

        final = (
            occupancy_score * 0.30
            + shoe_score * 0.22
            + shadow_score * 0.10
            + motion_stopped.astype(np.float32) * 0.30
            + landing_spike.astype(np.float32) * 0.06
            + stable_contact.astype(np.float32) * 0.02
        )
        if has_shoe_color:
            final = np.where(shoe_gate, final, 0.0)
        self._previous_occupancy = occupancy_score.copy()
        return np.clip(final, 0.0, 1.0).astype(np.float32)

    def _ensure_history_shape(self, rows: int, columns: int) -> None:
        shape = (rows, columns)
        if self._previous_occupancy is None or self._previous_occupancy.shape != shape:
            self._previous_occupancy = np.zeros(shape, dtype=np.float32)
        if self._motion_spike_hold is None or self._motion_spike_hold.shape != shape:
            self._motion_spike_hold = np.zeros(shape, dtype=np.int16)
