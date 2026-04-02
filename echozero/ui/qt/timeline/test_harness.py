"""Reusable test harness for the new read-only timeline shell.

Provides:
- demo presentation construction
- screenshot capture
- optional multiple state variants for quick visual regression checks
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

from PyQt6.QtWidgets import QApplication

from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.presentation.models import TimelinePresentation
from echozero.ui.qt.timeline.demo_app import build_demo_timeline
from echozero.ui.qt.timeline.widget import TimelineWidget


def build_demo_presentation() -> TimelinePresentation:
    timeline, session = build_demo_timeline()
    return TimelineAssembler().assemble(timeline, session)


def build_variant_presentations() -> dict[str, TimelinePresentation]:
    base = build_demo_presentation()
    return {
        "default": base,
        "stopped": replace(base, is_playing=False, playhead=0.25),
        "scrolled": replace(base, scroll_x=120.0, playhead=2.25),
    }


def capture_presentation_screenshot(
    presentation: TimelinePresentation,
    output_path: str | Path,
    *,
    width: int = 1400,
    height: int = 420,
) -> Path:
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(presentation)
    widget.resize(width, height)
    widget.show()
    app.processEvents()
    pixmap = widget.grab()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(str(output))
    widget.close()
    app.processEvents()
    return output.resolve()


def capture_demo_variants(output_dir: str | Path) -> list[Path]:
    output_root = Path(output_dir)
    paths: list[Path] = []
    for name, presentation in build_variant_presentations().items():
        path = capture_presentation_screenshot(
            presentation,
            output_root / f"timeline_{name}.png",
        )
        paths.append(path)
    return paths
