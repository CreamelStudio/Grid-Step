from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class CellState(str, Enum):
    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    PRESSED = "PRESSED"
    RELEASED = "RELEASED"


@dataclass
class StateEvents:
    pressed: list[tuple[int, int]]
    released: list[tuple[int, int]]


class GridStateManager:
    def __init__(self, rows: int, columns: int, smoothing_frames: int = 2, release_frames: int = 2) -> None:
        self.rows = max(1, int(rows))
        self.columns = max(1, int(columns))
        self.smoothing_frames = max(1, int(smoothing_frames))
        self.release_frames = max(1, int(release_frames))
        self.states = np.full((self.rows, self.columns), CellState.IDLE.value, dtype=object)
        self._hit_counts = np.zeros((self.rows, self.columns), dtype=np.int16)
        self._miss_counts = np.zeros((self.rows, self.columns), dtype=np.int16)

    def resize(self, rows: int, columns: int) -> None:
        if rows == self.rows and columns == self.columns:
            return
        self.__init__(rows, columns, self.smoothing_frames, self.release_frames)

    def update(self, confidence: np.ndarray, threshold: float) -> StateEvents:
        active = confidence >= threshold
        pressed_events: list[tuple[int, int]] = []
        released_events: list[tuple[int, int]] = []

        for row in range(self.rows):
            for col in range(self.columns):
                state = CellState(self.states[row, col])
                if active[row, col]:
                    self._hit_counts[row, col] += 1
                    self._miss_counts[row, col] = 0
                    if state in (CellState.IDLE, CellState.RELEASED):
                        self.states[row, col] = CellState.CANDIDATE.value
                    if self._hit_counts[row, col] >= self.smoothing_frames and state != CellState.PRESSED:
                        self.states[row, col] = CellState.PRESSED.value
                        pressed_events.append((row, col))
                else:
                    self._hit_counts[row, col] = 0
                    if state == CellState.PRESSED:
                        self._miss_counts[row, col] += 1
                        if self._miss_counts[row, col] >= self.release_frames:
                            self.states[row, col] = CellState.RELEASED.value
                            released_events.append((row, col))
                    elif state == CellState.RELEASED:
                        self.states[row, col] = CellState.IDLE.value
                        self._miss_counts[row, col] = 0
                    else:
                        self.states[row, col] = CellState.IDLE.value
                        self._miss_counts[row, col] = 0

        return StateEvents(pressed_events, released_events)

    def pressed_cells(self) -> list[tuple[int, int]]:
        rows, cols = np.where(self.states == CellState.PRESSED.value)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]

    def as_json(self, confidence: np.ndarray) -> dict[str, object]:
        return {
            "pressed": [{"row": r, "col": c} for r, c in self.pressed_cells()],
            "states": self.states.tolist(),
            "confidence": np.round(confidence, 4).tolist(),
        }
