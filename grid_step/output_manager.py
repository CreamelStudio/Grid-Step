from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OutputManager:
    last_state: dict[str, Any] = field(default_factory=dict)

    def publish(self, state: dict[str, Any], pressed: list[tuple[int, int]], released: list[tuple[int, int]]) -> None:
        self.last_state = state
        if pressed:
            print(f"PRESSED: {pressed}", flush=True)
        if released:
            print(f"RELEASED: {released}", flush=True)

    def current_json(self) -> str:
        return json.dumps(self.last_state, ensure_ascii=False)
