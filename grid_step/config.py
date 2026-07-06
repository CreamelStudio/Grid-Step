from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS_PATH = Path(__file__).with_name("settings.json")


@dataclass
class AppSettings:
    camera_id: int = 0
    resolution_width: int = 1280
    resolution_height: int = 720
    target_fps: int = 60
    floor_corners: list[list[float]] = field(default_factory=list)
    homography_matrix: list[list[float]] = field(default_factory=list)
    grid_rows: int = 4
    grid_columns: int = 4
    floor_width_m: float = 2.0
    floor_height_m: float = 2.0
    topdown_width: int = 800
    topdown_height: int = 800
    pixel_threshold: int = 35
    pixel_cell_threshold: float = 0.06
    shadow_threshold: int = 22
    shadow_cell_threshold: float = 0.05
    background_change_guard: float = 0.85
    motion_threshold: int = 18
    motion_cell_threshold: float = 0.025
    motion_stop_threshold: float = 0.22
    hover_confidence_threshold: float = 0.28
    press_confidence_threshold: float = 0.62
    touch_spike_hold_frames: int = 4
    shoe_color_bgr: list[int] = field(default_factory=list)
    shoe_color_threshold: int = 24
    shoe_hue_threshold: int = 10
    shoe_saturation_threshold: int = 45
    shoe_value_threshold: int = 55
    shoe_cell_threshold: float = 0.018
    shadow_near_shoe_radius: int = 42
    ai_detection: bool = False
    onnx_model_path: str = "models/foot_detector.onnx"
    ai_confidence_threshold: float = 0.5
    min_area: int = 120
    smoothing_frames: int = 1
    release_frames: int = 2
    detection_mode: str = "Touch"
    grid_line_thickness: int = 2
    show_grid: bool = True
    processing_scale: float = 1.0

    @property
    def resolution(self) -> tuple[int, int]:
        return self.resolution_width, self.resolution_height

    @property
    def topdown_size(self) -> tuple[int, int]:
        return self.topdown_width, self.topdown_height


def _coerce_settings(raw: dict[str, Any]) -> AppSettings:
    defaults = asdict(AppSettings())
    defaults.update({k: v for k, v in raw.items() if k in defaults})
    return AppSettings(**defaults)


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> AppSettings:
    settings_path = Path(path)
    if not settings_path.exists():
        return AppSettings()

    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AppSettings()
    return _coerce_settings(raw)


def save_settings(settings: AppSettings, path: str | Path = DEFAULT_SETTINGS_PATH) -> None:
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(asdict(settings), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
