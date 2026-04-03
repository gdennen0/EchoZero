from echozero.ui.qt.timeline.blocks.waveform_lane import (
    WaveformLanePresentation,
    visible_waveform_columns,
)


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
