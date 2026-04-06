"""Artifact helpers for screenshots and optional video capture."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .driver import E2EDriver


def capture_screenshot(driver: E2EDriver, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return driver.capture_screenshot(target)


@dataclass(slots=True)
class VideoCaptureHandle:
    output_path: Path
    command: tuple[str, ...] | None = None
    process: subprocess.Popen[str] | None = None

    def stop(self) -> Path:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
        return self.output_path


def start_video_capture(
    output_path: str | Path,
    *,
    command_template: Sequence[str] | None = None,
    context: dict[str, Any] | None = None,
) -> VideoCaptureHandle:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if command_template is None:
        return VideoCaptureHandle(output_path=target)

    format_context = {"output": str(target)}
    if context:
        format_context.update({key: str(value) for key, value in context.items()})
    command = tuple(part.format(**format_context) for part in command_template)
    process = subprocess.Popen(command, text=True)
    return VideoCaptureHandle(output_path=target, command=command, process=process)
