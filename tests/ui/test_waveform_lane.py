import pytest
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QPainter

from echozero.ui.qt.timeline.blocks.waveform_lane import (
    WaveformLaneBlock,
    WaveformLanePresentation,
    iter_compacted_waveform_columns,
    visible_waveform_columns,
)
from echozero.ui.qt.timeline.waveform_cache import CachedWaveform


def _presentation(scroll_x: float) -> WaveformLanePresentation:
    return WaveformLanePresentation(
        color_hex="#9b87f5",
        row_height=72,
        pixels_per_second=180.0,
        scroll_x=scroll_x,
        header_width=320,
        width=1440,
        dimmed=False,
    )


def test_visible_waveform_columns_fill_viewport_with_no_scroll():
    p = _presentation(scroll_x=0.0)
    cols = visible_waveform_columns(p, step=14)
    xs = [x for x, _ in cols]

    assert xs
    assert min(xs) <= p.header_width + 8
    assert max(xs) >= p.width - 8 - 14


def test_visible_waveform_columns_fill_viewport_after_horizontal_scroll():
    p = _presentation(scroll_x=1080.0)
    cols = visible_waveform_columns(p, step=14)
    xs = [x for x, _ in cols]

    assert xs
    assert min(xs) <= p.header_width + 8
    assert max(xs) >= p.width - 8 - 14


def test_visible_waveform_columns_shift_sample_indices_when_scrolled():
    no_scroll = visible_waveform_columns(_presentation(scroll_x=0.0), step=14)
    far_scroll = visible_waveform_columns(_presentation(scroll_x=560.0), step=14)

    assert no_scroll and far_scroll
    assert no_scroll[0][1] != far_scroll[0][1]


def test_waveform_lane_does_not_render_synthetic_fallback_when_cache_is_missing(monkeypatch):
    app = QApplication.instance() or QApplication([])
    assert app is not None
    image = QImage(1440, 96, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    block = WaveformLaneBlock()
    presented = _presentation(scroll_x=0.0)
    presented.waveform_key = "missing-waveform"
    presented.source_audio_path = None

    called = {"value": False}

    def _fail_visible_columns(*args, **kwargs):
        called["value"] = True
        raise AssertionError("Synthetic fallback columns should never be used")

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.blocks.waveform_lane.visible_waveform_columns",
        _fail_visible_columns,
    )

    try:
        block.paint(painter, 8, presented)
    finally:
        painter.end()

    assert called["value"] is False


def test_compacted_waveform_columns_merge_many_peaks_into_single_pixel_column():
    cached = CachedWaveform(
        sample_rate=1000,
        window_size=100,
        peaks=np.array(
            [
                [-0.1, 0.1],
                [-0.5, 0.2],
                [-0.3, 0.7],
                [-0.2, 0.4],
            ],
            dtype=np.float32,
        ),
    )

    cols = list(
        iter_compacted_waveform_columns(
            cached=cached,
            start_idx=0,
            end_idx=3,
            pixels_per_second=3.0,
            scroll_x=0.0,
            content_start_x=320.0,
        )
    )

    assert len(cols) == 1
    x, vmin, vmax = cols[0]
    assert x == 320
    assert vmin == pytest.approx(-0.5, abs=1e-6)
    assert vmax == pytest.approx(0.7, abs=1e-6)


def test_compacted_waveform_columns_preserve_per_peak_columns_when_zoomed_in():
    cached = CachedWaveform(
        sample_rate=1000,
        window_size=100,
        peaks=np.array(
            [
                [-0.1, 0.1],
                [-0.2, 0.2],
                [-0.3, 0.3],
            ],
            dtype=np.float32,
        ),
    )

    cols = list(
        iter_compacted_waveform_columns(
            cached=cached,
            start_idx=0,
            end_idx=2,
            pixels_per_second=40.0,
            scroll_x=0.0,
            content_start_x=320.0,
        )
    )

    assert [x for x, _, _ in cols] == [320, 324, 328]
    assert cols[0][1] == pytest.approx(-0.1, abs=1e-6)
    assert cols[0][2] == pytest.approx(0.1, abs=1e-6)
    assert cols[1][1] == pytest.approx(-0.2, abs=1e-6)
    assert cols[1][2] == pytest.approx(0.2, abs=1e-6)
    assert cols[2][1] == pytest.approx(-0.3, abs=1e-6)
    assert cols[2][2] == pytest.approx(0.3, abs=1e-6)
