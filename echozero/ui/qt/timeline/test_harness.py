"""Timeline test harness: Deterministic Stage Zero shell harness for tests.
Exists to stand up demo-backed widget states quickly for test and review surfaces.
Never treat this module as the canonical runtime or automation control plane.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.timeline.intents import Pause, Seek, ToggleLayerExpanded
from echozero.ui.FEEL import RULER_HEIGHT_PX, TAKE_ROW_HEIGHT_PX, TIMELINE_TRANSPORT_HEIGHT_PX
from echozero.ui.qt.font_bootstrap import ensure_qt_fonts_available
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.layer_height_config import timeline_layer_height_config
from echozero.ui.qt.timeline.widget import TimelineWidget


def build_demo_presentation() -> TimelinePresentation:
    demo = build_demo_app()
    return demo.presentation()


def build_variant_presentations() -> dict[str, TimelinePresentation]:
    demo = build_demo_app()
    default = demo.presentation()
    stopped = demo.dispatch(Pause())
    scrolled = replace(stopped, scroll_x=220.0, playhead=2.25, current_time_label="00:02.25")

    demo_take_lanes = build_demo_app()
    take_lanes_open = demo_take_lanes.dispatch(
        ToggleLayerExpanded(demo_take_lanes.presentation().layers[0].layer_id)
    )

    demo_seek = build_demo_app()
    sought = demo_seek.dispatch(Seek(3.4))

    zoomed_in = replace(default, pixels_per_second=320.0)
    zoomed_out = replace(default, pixels_per_second=90.0)

    return {
        "default": default,
        "stopped": stopped,
        "scrolled": scrolled,
        "take_lanes_open": take_lanes_open,
        "seeked": sought,
        "zoomed_in": zoomed_in,
        "zoomed_out": zoomed_out,
    }


def estimate_full_window_height(presentation: TimelinePresentation, *, minimum: int = 720) -> int:
    """Estimate a screenshot window height that avoids vertical clipping."""
    height_config = timeline_layer_height_config()
    ruler_height = RULER_HEIGHT_PX
    ruler_gap = 8
    default_main_row_height = height_config.default_main_row_height_px
    main_row_heights_by_kind = dict(height_config.layer_kind_main_row_height_px)
    take_row_height = height_config.take_row_height_px or TAKE_ROW_HEIGHT_PX
    canvas_bottom_padding = 12
    transport_height = TIMELINE_TRANSPORT_HEIGHT_PX
    hscroll_height = 20

    take_rows = sum(len(layer.takes) for layer in presentation.layers if layer.is_expanded)
    main_rows_height = sum(
        int(main_row_heights_by_kind.get(layer.kind, default_main_row_height))
        for layer in presentation.layers
    )
    canvas_height = (
        ruler_height
        + ruler_gap
        + main_rows_height
        + (take_rows * take_row_height)
        + canvas_bottom_padding
    )
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
    ensure_qt_fonts_available(app)
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
        path = capture_presentation_screenshot(presentation, output_root / f"timeline_{name}.png")
        paths.append(path)
    return paths
