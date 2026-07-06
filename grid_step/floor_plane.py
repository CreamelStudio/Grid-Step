from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import order_points
from .utils import clamp


@dataclass
class FloorPlaneInfo:
    quality_score: int
    warning: str
    cell_width_m: float
    cell_height_m: float
    area_px: float
    aspect_ratio: float


def analyze_floor_plane(
    corners: list[list[float]] | np.ndarray,
    rows: int,
    columns: int,
    floor_width_m: float,
    floor_height_m: float,
) -> FloorPlaneInfo:
    if len(corners) != 4:
        return FloorPlaneInfo(0, "Select four floor corners", 0.0, 0.0, 0.0, 0.0)

    pts = order_points(corners)
    area = abs(float(cv2.contourArea(pts)))
    top = np.linalg.norm(pts[1] - pts[0])
    bottom = np.linalg.norm(pts[2] - pts[3])
    left = np.linalg.norm(pts[3] - pts[0])
    right = np.linalg.norm(pts[2] - pts[1])
    avg_w = max((top + bottom) * 0.5, 1.0)
    avg_h = max((left + right) * 0.5, 1.0)
    perspective_skew = abs(top - bottom) / avg_w + abs(left - right) / avg_h
    rectangularity = clamp(1.0 - perspective_skew * 0.65, 0.0, 1.0)
    min_area_score = clamp(area / 35_000.0, 0.0, 1.0)
    quality = int(round(100 * rectangularity * min_area_score))

    warning = ""
    if quality < 45:
        warning = "Floor area is too skewed or too small. Adjust camera angle or corners."
    elif quality < 70:
        warning = "Floor area is usable, but calibration quality is moderate."

    return FloorPlaneInfo(
        quality_score=quality,
        warning=warning,
        cell_width_m=floor_width_m / max(columns, 1),
        cell_height_m=floor_height_m / max(rows, 1),
        area_px=area,
        aspect_ratio=avg_w / avg_h,
    )
