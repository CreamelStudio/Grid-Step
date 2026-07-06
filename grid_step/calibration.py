from __future__ import annotations

import cv2
import numpy as np


def order_points(points: np.ndarray | list[list[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Expected four 2D points")

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def destination_points(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def compute_homography(
    source_points: np.ndarray | list[list[float]],
    topdown_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    ordered = order_points(source_points)
    matrix = cv2.getPerspectiveTransform(ordered, destination_points(topdown_size))
    return matrix, ordered


def warp_floor(frame: np.ndarray, homography: np.ndarray, topdown_size: tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(frame, homography, topdown_size, flags=cv2.INTER_LINEAR)


def default_floor_corners(frame_shape: tuple[int, ...]) -> list[list[float]]:
    height, width = frame_shape[:2]
    margin_x = width * 0.18
    margin_y = height * 0.18
    return [
        [margin_x, margin_y],
        [width - margin_x, margin_y],
        [width - margin_x, height - margin_y],
        [margin_x, height - margin_y],
    ]
