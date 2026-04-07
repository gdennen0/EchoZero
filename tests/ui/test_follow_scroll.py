from dataclasses import replace

from echozero.application.shared.enums import FollowMode
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.widget import compute_follow_scroll_x


def test_follow_off_keeps_scroll():
    p = replace(build_demo_app().presentation(), follow_mode=FollowMode.OFF, scroll_x=222.0)
    assert compute_follow_scroll_x(p, viewport_width=1200) == 222.0


def test_follow_page_moves_when_playhead_offscreen_right():
    p = replace(
        build_demo_app().presentation(),
        follow_mode=FollowMode.PAGE,
        is_playing=True,
        playhead=28.0,
        pixels_per_second=180.0,
        scroll_x=0.0,
    )
    assert compute_follow_scroll_x(p, viewport_width=1200) > 0.0


def test_follow_does_not_auto_scroll_when_not_playing():
    p = replace(
        build_demo_app().presentation(),
        follow_mode=FollowMode.PAGE,
        is_playing=False,
        playhead=28.0,
        pixels_per_second=180.0,
        scroll_x=123.0,
    )
    assert compute_follow_scroll_x(p, viewport_width=1200) == 123.0


def test_follow_center_targets_midpoint_and_clamps():
    p = replace(
        build_demo_app().presentation(),
        follow_mode=FollowMode.CENTER,
        is_playing=True,
        playhead=20.0,
        pixels_per_second=200.0,
        scroll_x=0.0,
    )
    next_scroll = compute_follow_scroll_x(p, viewport_width=1200)
    assert next_scroll > 0.0

    p2 = replace(p, playhead=0.0)
    assert compute_follow_scroll_x(p2, viewport_width=1200) == 0.0


def test_follow_smooth_targets_75_percent_position_not_center():
    p_center = replace(
        build_demo_app().presentation(),
        follow_mode=FollowMode.CENTER,
        is_playing=True,
        playhead=20.0,
        pixels_per_second=200.0,
        scroll_x=0.0,
    )
    p_smooth = replace(p_center, follow_mode=FollowMode.SMOOTH)

    center_scroll = compute_follow_scroll_x(p_center, viewport_width=1200)
    smooth_scroll = compute_follow_scroll_x(p_smooth, viewport_width=1200)

    # Smooth mode should keep more look-ahead than centered mode.
    assert smooth_scroll < center_scroll
    assert smooth_scroll > 0.0
