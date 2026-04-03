"""Reusable test harness for the Stage Zero timeline shell."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

from PyQt6.QtWidgets import QApplication

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.timeline.intents import Pause, Seek, ToggleTakeSelector
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.widget import TimelineWidget


def build_demo_presentation() -> TimelinePresentation:
    demo = build_demo_app()
    return demo.presentation()


def build_variant_presentations() -> dict[str, TimelinePresentation]:
    demo = build_demo_app()
    default = demo.presentation()
    stopped = demo.dispatch(Pause())
    scrolled = replace(stopped, scroll_x=220.0, playhead=2.25, current_time_label='00:02.25')

    demo_take_lanes = build_demo_app()
    take_lanes_open = demo_take_lanes.dispatch(ToggleTakeSelector(demo_take_lanes.presentation().layers[0].layer_id))

    demo_seek = build_demo_app()
    sought = demo_seek.dispatch(Seek(3.4))

    zoomed_in = replace(default, pixels_per_second=320.0)
    zoomed_out = replace(default, pixels_per_second=90.0)

    return {
        'default': default,
        'stopped': stopped,
        'scrolled': scrolled,
        'take_lanes_open': take_lanes_open,
        'seeked': sought,
        'zoomed_in': zoomed_in,
        'zoomed_out': zoomed_out,
    }


def estimate_full_window_height(presentation: TimelinePresentation, *, minimum: int = 720) -> int:
    """Estimate a screenshot window height that avoids vertical clipping."""
    ruler_height = 28
    ruler_gap = 8
    main_row_height = 72
    take_row_height = 44
    canvas_bottom_padding = 12
    transport_height = 44
    hscroll_height = 20

    take_rows = sum(len(layer.takes) for layer in presentation.layers if layer.is_expanded)
    canvas_height = ruler_height + ruler_gap + (len(presentation.layers) * main_row_height) + (take_rows * take_row_height) + canvas_bottom_padding
    full_height = transport_height + hscroll_height + canvas_height
    return max(minimum, int(full_height + 8))


def capture_presentation_screenshot(
    presentation: TimelinePresentation,
    output_path: str | Path,
    *,
    width: int = 1440,
    height: int | None = None,
) -> Path:
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(presentation)
    target_height = estimate_full_window_height(presentation) if height is None else height
    widget.resize(width, target_height)
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
        path = capture_presentation_screenshot(presentation, output_root / f'timeline_{name}.png')
        paths.append(path)
    return paths
