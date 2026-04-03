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
        playhead=28.0,
        pixels_per_second=180.0,
        scroll_x=0.0,
    )
    assert compute_follow_scroll_x(p, viewport_width=1200) > 0.0


def test_follow_center_targets_midpoint_and_clamps():
    p = replace(
        build_demo_app().presentation(),
        follow_mode=FollowMode.CENTER,
        playhead=20.0,
        pixels_per_second=200.0,
        scroll_x=0.0,
    )
    next_scroll = compute_follow_scroll_x(p, viewport_width=1200)
    assert next_scroll > 0.0

    p2 = replace(p, playhead=0.0)
    assert compute_follow_scroll_x(p2, viewport_width=1200) == 0.0
