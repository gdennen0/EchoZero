from echozero.ui.qt.timeline.blocks.ruler import (
    absolute_timeline_x_for_view_x,
    seek_time_for_x,
    timeline_x_for_time,
    visible_ruler_marks,
    visible_ruler_seconds,
)
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.ids import TimelineId
from echozero.ui.FEEL import RULER_MIN_TICK_SPACING_PX
from echozero.ui.qt.timeline.time_grid import TimelineGridMode, visible_grid_lines
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
        assert content_start_x <= x <= (content_start_x + content_width)


def test_visible_ruler_seconds_respects_min_major_tick_spacing_when_zoomed_out():
    marks = visible_ruler_seconds(
        scroll_x=0.0,
        pixels_per_second=20.0,
        content_width=900.0,
        content_start_x=320.0,
    )

    assert len(marks) > 1
    deltas = [right_x - left_x for (_, left_x), (_, right_x) in zip(marks, marks[1:])]
    assert all(delta >= float(RULER_MIN_TICK_SPACING_PX) for delta in deltas)


def test_seek_time_for_x_maps_ruler_x_to_timeline_time():
    assert (
        seek_time_for_x(
            520.0,
            scroll_x=120.0,
            pixels_per_second=100.0,
            content_start_x=320.0,
        )
        == 3.2
    )


def test_absolute_timeline_x_for_view_x_includes_scroll_offset():
    assert (
        absolute_timeline_x_for_view_x(
            520.0,
            scroll_x=180.0,
            content_start_x=320.0,
        )
        == 380.0
    )


def test_timeline_x_for_time_inverts_seek_mapping():
    x = timeline_x_for_time(
        4.5,
        scroll_x=80.0,
        pixels_per_second=100.0,
        content_start_x=320.0,
    )

    assert x == 690.0
    assert (
        seek_time_for_x(
            x,
            scroll_x=80.0,
            pixels_per_second=100.0,
            content_start_x=320.0,
        )
        == 4.5
    )


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


def test_visible_grid_lines_extend_before_first_beat_anchor():
    lines = visible_grid_lines(
        scroll_x=0.0,
        pixels_per_second=160.0,
        content_width=500.0,
        mode=TimelineGridMode.BEAT,
        bpm=120.0,
        beat_anchor_seconds=2.0,
    )

    time_seconds = [round(line.time_seconds, 3) for line in lines]
    assert 0.0 in time_seconds
    assert 1.5 in time_seconds
    assert 2.0 in time_seconds


def test_visible_ruler_marks_skip_negative_bar_labels_before_anchor():
    presentation = TimelinePresentation(
        timeline_id=TimelineId("timeline_ruler"),
        title="Ruler",
        bpm=120.0,
        beat_anchor_seconds=2.0,
        pixels_per_second=160.0,
        scroll_x=0.0,
    )

    marks = visible_ruler_marks(
        presentation=presentation,
        content_width=800.0,
        content_start_x=320.0,
        mode=TimelineGridMode.BEAT,
    )

    assert marks
    assert marks[0][0] == "1|1"
    assert marks[0][1] == 640.0


def test_visible_ruler_marks_use_seconds_when_grid_mode_is_not_beat():
    presentation = TimelinePresentation(
        timeline_id=TimelineId("timeline_ruler_seconds"),
        title="Ruler",
        bpm=120.0,
        beat_anchor_seconds=2.0,
        pixels_per_second=160.0,
        scroll_x=0.0,
    )

    marks = visible_ruler_marks(
        presentation=presentation,
        content_width=800.0,
        content_start_x=320.0,
        mode=TimelineGridMode.AUTO,
    )

    assert marks
    assert marks[0][0] == "0"
