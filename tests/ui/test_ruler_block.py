from echozero.ui.qt.timeline.blocks.ruler import (
    absolute_timeline_x_for_view_x,
    seek_time_for_x,
    timeline_x_for_time,
    visible_ruler_seconds,
)
from echozero.ui.qt.timeline.blocks.waveform_lane import waveform_x_for_time


def test_visible_ruler_seconds_starts_near_zero_without_scroll():
    marks = visible_ruler_seconds(
        scroll_x=0.0,
        pixels_per_second=180.0,
        content_width=900.0,
        content_start_x=320.0,
    )

    assert marks
    assert marks[0][0] == 0


def test_visible_ruler_seconds_reflects_horizontal_scroll_offset():
    marks = visible_ruler_seconds(
        scroll_x=1080.0,
        pixels_per_second=180.0,
        content_width=900.0,
        content_start_x=320.0,
    )

    seconds = [second for second, _ in marks]
    assert min(seconds) >= 5
    assert 6 in seconds


def test_visible_ruler_seconds_screen_x_remains_in_content_band():
    content_start_x = 320.0
    content_width = 900.0
    pps = 180.0

    marks = visible_ruler_seconds(
        scroll_x=720.0,
        pixels_per_second=pps,
        content_width=content_width,
        content_start_x=content_start_x,
    )

    assert marks
    for _, x in marks:
        assert (content_start_x - pps) <= x <= (content_start_x + content_width + pps)


def test_seek_time_for_x_maps_ruler_x_to_timeline_time():
    assert seek_time_for_x(
        520.0,
        scroll_x=120.0,
        pixels_per_second=100.0,
        content_start_x=320.0,
    ) == 3.2


def test_absolute_timeline_x_for_view_x_includes_scroll_offset():
    assert absolute_timeline_x_for_view_x(
        520.0,
        scroll_x=180.0,
        content_start_x=320.0,
    ) == 380.0


def test_timeline_x_for_time_inverts_seek_mapping():
    x = timeline_x_for_time(
        4.5,
        scroll_x=80.0,
        pixels_per_second=100.0,
        content_start_x=320.0,
    )

    assert x == 690.0
    assert seek_time_for_x(
        x,
        scroll_x=80.0,
        pixels_per_second=100.0,
        content_start_x=320.0,
    ) == 4.5


def test_waveform_x_for_time_matches_ruler_mapping():
    assert waveform_x_for_time(
        4.5,
        scroll_x=80.0,
        pixels_per_second=100.0,
        content_start_x=320.0,
    ) == timeline_x_for_time(
        4.5,
        scroll_x=80.0,
        pixels_per_second=100.0,
        content_start_x=320.0,
    )
