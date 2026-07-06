from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


class FpsCounter:
    def __init__(self, window: int = 60) -> None:
        self.samples: deque[float] = deque(maxlen=window)
        self._last: float | None = None

    def tick(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        if self._last is not None:
            delta = max(now - self._last, 1e-9)
            self.samples.append(1.0 / delta)
        self._last = now
        return self.value

    @property
    def value(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)


@dataclass
class PerfStats:
    camera_fps: float = 0.0
    processing_fps: float = 0.0
    total_latency_ms: float = 0.0
    pixel_ms: float = 0.0
    shadow_ms: float = 0.0
    onnx_ms: float = 0.0
    render_ms: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


@contextmanager
def measure_ms(target: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        target[key] = (time.perf_counter() - start) * 1000.0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
