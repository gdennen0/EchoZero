"""
Video timeline mapping for song-level reference media.
Exists to keep video offset sync rules independent from Qt playback widgets.
Connects timeline presentation layers to media-second positions for video controllers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from echozero.application.presentation.models import TimelinePresentation


@dataclass(frozen=True, slots=True)
class VideoTimelineMapping:
    """Maps song timeline seconds to media seconds for one video clip."""

    video_path: str
    start_seconds: float
    duration_seconds: float

    def media_seconds_for_song_time(self, song_seconds: float) -> float:
        """Return the clamped media position for a song timeline position."""

        return max(0.0, min(self.duration_seconds, float(song_seconds) - self.start_seconds))

    def contains_song_time(self, song_seconds: float) -> bool:
        """Return whether the song timeline position is inside the video range."""

        media_seconds = float(song_seconds) - self.start_seconds
        return 0.0 <= media_seconds <= self.duration_seconds


def video_mapping_from_presentation(
    presentation: TimelinePresentation,
) -> VideoTimelineMapping | None:
    """Return the first valid video mapping from a timeline presentation."""

    for layer in presentation.layers:
        if getattr(layer, "reference_kind", None) != "video":
            continue
        video_path = str(getattr(layer, "video_path", "") or "").strip()
        if not video_path:
            continue
        path = Path(video_path)
        if not path.exists():
            continue
        return VideoTimelineMapping(
            video_path=str(path),
            start_seconds=float(getattr(layer, "video_start_seconds", 0.0)),
            duration_seconds=max(0.0, float(getattr(layer, "video_duration_seconds", 0.0))),
        )
    return None
