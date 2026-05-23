"""
Video placement policy tests for application-owned reference-video edits.
Exists to keep video trim and loop semantics out of Qt gesture code.
Connects timeline interaction behavior to stable application-level placement rules.
"""

from echozero.application.timeline.video_placement import (
    VideoPlacement,
    VideoPlacementEditMode,
    edit_video_placement,
)


def test_video_loop_back_edit_extends_item_past_source_duration() -> None:
    placement = VideoPlacement(
        start_seconds=1.0,
        trim_start_seconds=0.0,
        visible_duration_seconds=8.0,
        source_duration_seconds=8.0,
        loop_enabled=False,
    )

    edited = edit_video_placement(
        placement,
        mode=VideoPlacementEditMode.LOOP_BACK,
        delta_seconds=25.0,
    )

    assert edited.loop_enabled is True
    assert edited.visible_duration_seconds == 33.0
    assert edited.start_seconds == 1.0


def test_video_trim_front_preserves_right_edge_and_source_offset() -> None:
    placement = VideoPlacement(
        start_seconds=1.0,
        trim_start_seconds=0.0,
        visible_duration_seconds=8.0,
        source_duration_seconds=8.0,
        loop_enabled=False,
    )

    edited = edit_video_placement(
        placement,
        mode=VideoPlacementEditMode.TRIM_FRONT,
        delta_seconds=0.5,
    )

    assert edited.start_seconds == 1.5
    assert edited.trim_start_seconds == 0.5
    assert edited.visible_duration_seconds == 7.5
    assert edited.loop_enabled is False


def test_video_loop_front_extends_item_left_without_changing_source_offset() -> None:
    placement = VideoPlacement(
        start_seconds=1.0,
        trim_start_seconds=1.0,
        visible_duration_seconds=7.0,
        source_duration_seconds=8.0,
        loop_enabled=False,
    )

    edited = edit_video_placement(
        placement,
        mode=VideoPlacementEditMode.LOOP_FRONT,
        delta_seconds=-5.0,
    )

    assert edited.loop_enabled is True
    assert edited.start_seconds == -4.0
    assert edited.trim_start_seconds == 1.0
    assert edited.visible_duration_seconds == 12.0


def test_video_placement_preserves_oversized_duration_when_looping() -> None:
    placement = VideoPlacement(
        start_seconds=1.0,
        trim_start_seconds=2.0,
        visible_duration_seconds=999.0,
        source_duration_seconds=8.0,
        loop_enabled=True,
    )

    normalized = placement.normalized()

    assert normalized.visible_duration_seconds == 999.0
    assert normalized.loop_enabled is True


def test_video_placement_clamps_oversized_duration_when_not_looping() -> None:
    placement = VideoPlacement(
        start_seconds=1.0,
        trim_start_seconds=2.0,
        visible_duration_seconds=999.0,
        source_duration_seconds=8.0,
        loop_enabled=False,
    )

    normalized = placement.normalized()

    assert normalized.visible_duration_seconds == 6.0
    assert normalized.loop_enabled is False
