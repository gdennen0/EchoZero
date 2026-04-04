from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_foundry_window_smoke(tmp_path: Path):
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception:
        pytest.skip("PyQt6 not installed")

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from echozero.foundry.ui import FoundryWindow

    app = QApplication.instance() or QApplication([])
    window = FoundryWindow(tmp_path)
    try:
        assert window.windowTitle().startswith("EchoZero Foundry")
        assert window._app is not None
    finally:
        window.close()
