"""Object-info preview rendering tests.
Exists to keep compact waveform preview coverage separate from the wider timeline shell cases.
Connects inspector preview resampling to a stable short-clip regression proof.
"""

import numpy as np
import pytest
from PyQt6.QtCore import QRect
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from echozero.ui.qt.timeline.object_info_panel_preview import (
    EventPreviewState,
    EventPreviewWaveform,
    audio_event_preview_variant_label,
    audio_event_preview_variants,
    build_waveform_envelope_points,
    clip_waveform_columns,
)
from echozero.ui.qt.timeline.style import TIMELINE_STYLE
from echozero.ui.qt.timeline.waveform_cache import CachedWaveform


def test_clip_waveform_columns_resamples_short_clip_across_preview_width():
    cached = CachedWaveform(
        sample_rate=4,
        window_size=1,
        peaks=np.array(
            [
                [-0.1, 0.1],
                [-0.8, 0.6],
                [-0.3, 0.2],
                [-0.5, 0.9],
            ],
            dtype=np.float32,
        ),
    )

    columns = clip_waveform_columns(
        cached,
        start_seconds=0.25,
        end_seconds=0.50,
        column_count=8,
    )

    assert len(columns) == 8
    for vmin, vmax in columns:
        assert vmin == pytest.approx(-0.8)
        assert vmax == pytest.approx(0.6)


def test_clip_waveform_columns_blends_multiple_peak_windows_per_column():
    cached = CachedWaveform(
        sample_rate=4,
        window_size=1,
        peaks=np.array(
            [
                [-0.2, 0.1],
                [-0.8, 0.4],
                [-0.3, 0.9],
                [-0.1, 0.2],
            ],
            dtype=np.float32,
        ),
    )

    columns = clip_waveform_columns(
        cached,
        start_seconds=0.0,
        end_seconds=1.0,
        column_count=2,
    )

    assert columns[0][0] == pytest.approx(-0.8)
    assert columns[0][1] == pytest.approx(0.4)
    assert columns[1][0] == pytest.approx(-0.3)
    assert columns[1][1] == pytest.approx(0.9)


def test_build_waveform_envelope_points_tracks_outline_without_center_fill():
    rect = QRect(6, 6, 228, 48)
    peak_columns = [(-1.0, 1.0)] * 8

    top_envelope, bottom_envelope = build_waveform_envelope_points(
        rect,
        peak_columns=peak_columns,
    )

    assert len(top_envelope) == len(peak_columns)
    assert len(bottom_envelope) == len(peak_columns)
    assert top_envelope[0].y() < rect.center().y()
    assert bottom_envelope[0].y() > rect.center().y()
    assert all(point.y() == top_envelope[0].y() for point in top_envelope)
    assert all(point.y() == bottom_envelope[0].y() for point in bottom_envelope)


def test_event_preview_waveform_renders_visible_outline_for_dense_clip():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    widget = EventPreviewWaveform()
    widget.resize(282, 60)
    widget.set_preview(
        EventPreviewState(
            layer_id="layer",
            take_id="take",
            event_id="event",
            source_ref="drums.wav",
            source_audio_path="drums.wav",
            waveform_key="wave",
            start_seconds=0.25,
            end_seconds=0.33,
            duration_seconds=0.08,
        )
    )
    cached = CachedWaveform(
        sample_rate=44100,
        window_size=256,
        peaks=np.array([[-1.0, 0.55]] * 173, dtype=np.float32),
    )
    widget._resolve_cached_waveform = lambda preview: cached

    widget.show()
    app.processEvents()
    widget.update()
    app.processEvents()
    image = widget.grab().toImage().convertToFormat(QImage.Format.Format_ARGB32)

    background = TIMELINE_STYLE.object_palette.button_bg_hex.lower()
    rect = widget.rect().adjusted(6, 6, -6, -6)

    assert any(
        image.pixelColor(x, 18).name().lower() != background
        for x in range(rect.left(), rect.right() + 1)
    )
    assert any(
        image.pixelColor(x, 48).name().lower() != background
        for x in range(rect.left(), rect.right() + 1)
    )


def test_audio_event_preview_variant_catalog_is_user_facing():
    assert audio_event_preview_variants() == ("bars", "filled", "outline")
    assert audio_event_preview_variant_label("bars") == "Bars"
    assert audio_event_preview_variant_label("filled") == "Fill"
    assert audio_event_preview_variant_label("outline") == "Outline"


def test_event_preview_waveform_variant_changes_center_body_render():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    widget = EventPreviewWaveform()
    widget.resize(282, 60)
    widget.set_preview(
        EventPreviewState(
            layer_id="layer",
            take_id="take",
            event_id="event",
            source_ref="drums.wav",
            source_audio_path="drums.wav",
            waveform_key="wave",
            start_seconds=0.25,
            end_seconds=0.33,
            duration_seconds=0.08,
        )
    )
    cached = CachedWaveform(
        sample_rate=44100,
        window_size=256,
        peaks=np.array([[-1.0, 0.55]] * 173, dtype=np.float32),
    )
    widget._resolve_cached_waveform = lambda preview: cached
    widget.show()

    def _grab_body_color(variant: str) -> str:
        widget.set_variant(variant)
        app.processEvents()
        widget.update()
        app.processEvents()
        image = widget.grab().toImage().convertToFormat(QImage.Format.Format_ARGB32)
        return image.pixelColor(120, 30).name().lower()

    background = TIMELINE_STYLE.object_palette.button_bg_hex.lower()
    bars_color = _grab_body_color("bars")
    filled_color = _grab_body_color("filled")
    outline_color = _grab_body_color("outline")

    assert bars_color != background
    assert filled_color != background
    assert outline_color == background
    assert bars_color != filled_color
