from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int


class GridMapper:
    def __init__(self, rows: int, columns: int, size: tuple[int, int]) -> None:
        self.rows = max(1, int(rows))
        self.columns = max(1, int(columns))
        self.width, self.height = size
        self.cells = self._build_cells()

    def _build_cells(self) -> list[GridCell]:
        cells: list[GridCell] = []
        for row in range(self.rows):
            y0 = round(row * self.height / self.rows)
            y1 = round((row + 1) * self.height / self.rows)
            for col in range(self.columns):
                x0 = round(col * self.width / self.columns)
                x1 = round((col + 1) * self.width / self.columns)
                cells.append(GridCell(row, col, x0, y0, x1, y1))
        return cells

    def confidence_from_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        binary = mask > 0
        confidence = np.zeros((self.rows, self.columns), dtype=np.float32)
        for cell in self.cells:
            roi = binary[cell.y0 : cell.y1, cell.x0 : cell.x1]
            if roi.size:
                confidence[cell.row, cell.col] = float(np.count_nonzero(roi) / roi.size)
        return confidence

    def mask_from_boxes(self, boxes: list[tuple[float, float, float, float, float]]) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for x0, y0, x1, y1, conf in boxes:
            value = int(max(0.0, min(1.0, conf)) * 255)
            cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), value, -1)
        return mask

    def pressed_cells(self, confidence: np.ndarray, threshold: float) -> list[tuple[int, int]]:
        rows, cols = np.where(confidence >= threshold)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]
