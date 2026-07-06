from __future__ import annotations

import sys


def main() -> int:
    try:
        from .gui import run_app
    except ImportError as exc:
        print(f"Grid Step requires PySide6, OpenCV, and NumPy. Import failed: {exc}", file=sys.stderr)
        return 1
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
