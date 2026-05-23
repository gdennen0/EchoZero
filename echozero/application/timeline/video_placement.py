"""
Video placement editing policy for song-level reference media.
Exists so Qt gestures do not own trim, move, or loop semantics.
Connects timeline UI interactions to application-level video placement truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class VideoPlacementEditMode(StrEnum):
    """Direct manipulation modes for a timeline video reference."""

    MOVE = "move"
    TRIM_FRONT = "trim_front"
    TRIM_BACK = "trim_back"
    LOOP_FRONT = "loop_front"
    LOOP_BACK = "loop_back"


@dataclass(frozen=True, slots=True)
class VideoPlacement:
    """One song-version video placement in timeline and source-media seconds."""

    start_seconds: float
    trim_start_seconds: float
    visible_duration_seconds: float
    source_duration_seconds: float
    loop_enabled: bool = False

    @property
    def max_visible_duration_seconds(self) -> float:
        """Return the longest non-looped visible object duration."""

        return max(0.0, self.source_duration_seconds - self.trim_start_seconds)

    @property
    def loop_cycle_seconds(self) -> float:
        """Return one source-media cycle for repeated playback."""

        return self.max_visible_duration_seconds

    def normalized(self) -> VideoPlacement:
        """Return a bounded placement with sane trim and duration values."""

        source_duration = max(0.0, float(self.source_duration_seconds))
        trim_start = min(max(0.0, float(self.trim_start_seconds)), source_duration)
        max_visible_duration = max(0.0, source_duration - trim_start)
        visible_duration = max(0.0, float(self.visible_duration_seconds))
        if visible_duration <= 0.0:
            visible_duration = max_visible_duration
        if not self.loop_enabled:
            visible_duration = min(visible_duration, max_visible_duration)
        return VideoPlacement(
            start_seconds=float(self.start_seconds),
            trim_start_seconds=trim_start,
            visible_duration_seconds=visible_duration,
            source_duration_seconds=source_duration,
            loop_enabled=bool(self.loop_enabled),
        )


def edit_video_placement(
    placement: VideoPlacement,
    *,
    mode: VideoPlacementEditMode | str,
    delta_seconds: float,
    minimum_duration_seconds: float = 0.05,
) -> VideoPlacement:
    """Apply a direct manipulation edit to a video placement."""

    edit_mode = VideoPlacementEditMode(str(mode))
    current = placement.normalized()
    min_duration = max(0.001, float(minimum_duration_seconds))
    start = current.start_seconds
    trim_start = current.trim_start_seconds
    visible_duration = current.visible_duration_seconds
    loop_enabled = current.loop_enabled
    delta = float(delta_seconds)

    if edit_mode is VideoPlacementEditMode.MOVE:
        start += delta
    elif edit_mode is VideoPlacementEditMode.TRIM_FRONT:
        max_delta = max(-trim_start, visible_duration - min_duration)
        applied = min(max(delta, -trim_start), max_delta)
        start += applied
        trim_start += applied
        visible_duration -= applied
    elif edit_mode is VideoPlacementEditMode.TRIM_BACK:
        max_visible_duration = max(min_duration, current.source_duration_seconds - trim_start)
        visible_duration = min(
            max(min_duration, visible_duration + delta),
            max_visible_duration,
        )
    elif edit_mode is VideoPlacementEditMode.LOOP_BACK:
        visible_duration = max(min_duration, visible_duration + delta)
        loop_enabled = True
    elif edit_mode is VideoPlacementEditMode.LOOP_FRONT:
        applied = min(delta, visible_duration - min_duration)
        start += applied
        visible_duration -= applied
        loop_enabled = True

    return VideoPlacement(
        start_seconds=start,
        trim_start_seconds=trim_start,
        visible_duration_seconds=visible_duration,
        source_duration_seconds=current.source_duration_seconds,
        loop_enabled=loop_enabled,
    ).normalized()
