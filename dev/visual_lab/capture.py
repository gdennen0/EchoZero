"""Visual Lab capture: screenshot helpers for lab previews.
Exists to use Peekaboo when available while keeping a deterministic Qt fallback.
Preview runner and tests call this module; EchoZero runtime paths do not.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess

from PyQt6.QtWidgets import QWidget

DEFAULT_OUTPUT_DIR = Path("artifacts") / "visual-lab"
PEEKABOO_PATH = Path("/opt/homebrew/bin/peekaboo")


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """Result metadata for one Visual Lab screenshot capture."""

    path: Path
    backend: str


def capture_widget(
    widget: QWidget,
    output_path: str | Path,
    *,
    prefer_peekaboo: bool = True,
    window_title: str = "EchoZero Visual Lab",
    peekaboo_timeout_seconds: float = 8.0,
) -> CaptureResult:
    """Capture a preview widget to disk using Peekaboo or the Qt grab fallback."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if prefer_peekaboo and is_peekaboo_available() and widget.isVisible():
        if _capture_with_peekaboo(
            destination,
            window_title=window_title,
            timeout_seconds=peekaboo_timeout_seconds,
        ):
            return CaptureResult(path=destination, backend="peekaboo")

    _capture_with_qt_grab(widget, destination)
    return CaptureResult(path=destination, backend="qt-grab")


def default_capture_path(name: str = "visual_lab_preview.png") -> Path:
    """Return the default ignored artifact path for a Visual Lab capture."""
    return DEFAULT_OUTPUT_DIR / name


def is_peekaboo_available() -> bool:
    """Return whether the preferred local Peekaboo CLI is executable."""
    return PEEKABOO_PATH.exists() and os.access(PEEKABOO_PATH, os.X_OK)


def _capture_with_peekaboo(
    destination: Path,
    *,
    window_title: str,
    timeout_seconds: float,
) -> bool:
    command = [
        str(PEEKABOO_PATH),
        "image",
        "--mode",
        "window",
        "--app",
        f"PID:{os.getpid()}",
        "--window-title",
        window_title,
        "--path",
        str(destination),
        "--format",
        "png",
    ]
    if shutil.which(str(PEEKABOO_PATH)) is None and not PEEKABOO_PATH.exists():
        return False

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0 and destination.exists() and destination.stat().st_size > 0


def _capture_with_qt_grab(widget: QWidget, destination: Path) -> None:
    pixmap = widget.grab()
    if pixmap.isNull():
        raise RuntimeError("Qt widget grab produced an empty pixmap")
    if not pixmap.save(str(destination), "PNG"):
        raise RuntimeError(f"Could not save Visual Lab capture to {destination}")
