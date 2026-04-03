from echozero.ui.qt.timeline.blocks.ruler import visible_ruler_seconds


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
