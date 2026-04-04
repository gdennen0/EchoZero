from __future__ import annotations

from pathlib import Path

from .main_window import run_foundry_ui


if __name__ == "__main__":
    raise SystemExit(run_foundry_ui(Path.cwd()))
