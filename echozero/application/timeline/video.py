"""
Video timeline mapping for song-level reference media.
Exists to keep video offset sync rules independent from Qt playback widgets.
Connects timeline presentation layers to media-second positions for video controllers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.timeline.video_placement import VideoPlacement


@dataclass(frozen=True, slots=True)
class VideoTimelineMapping:
    """Maps song timeline seconds to media seconds for one video clip."""

    video_path: str
    start_seconds: float
    duration_seconds: float
    trim_start_seconds: float = 0.0
    source_duration_seconds: float | None = None
    loop_enabled: bool = False
    media_available: bool = True
    unavailable_reason: str = ""

    def media_seconds_for_song_time(self, song_seconds: float) -> float:
        """Return the media position for a song timeline position."""

        elapsed_seconds = float(song_seconds) - self.start_seconds
        if elapsed_seconds <= 0.0 or self.duration_seconds <= 0.0:
            return max(0.0, self.trim_start_seconds)
        if self.loop_enabled:
            cycle_seconds = self.loop_cycle_seconds
            if cycle_seconds <= 0.0:
                return max(0.0, self.trim_start_seconds)
            return self.trim_start_seconds + (elapsed_seconds % cycle_seconds)
        return self.trim_start_seconds + min(self.duration_seconds, elapsed_seconds)

    def contains_song_time(self, song_seconds: float) -> bool:
        """Return whether the song timeline position is inside the video range."""

        media_seconds = float(song_seconds) - self.start_seconds
        return 0.0 <= media_seconds <= self.duration_seconds

    @property
    def loop_cycle_seconds(self) -> float:
        """Return the source-media length used by loop playback."""

        if self.source_duration_seconds is None:
            return max(0.0, self.duration_seconds)
        return max(0.0, float(self.source_duration_seconds) - self.trim_start_seconds)


@dataclass(frozen=True, slots=True)
class VideoClockDecision:
    """One requested media-player action for the current audio-clock sample."""

    should_play: bool
    media_seconds: float
    should_seek: bool


@dataclass(slots=True)
class VideoClockSync:
    """Computes video media actions from the authoritative song transport clock."""

    drift_threshold_seconds: float = 0.08

    def decision(
        self,
        mapping: VideoTimelineMapping | None,
        *,
        song_seconds: float,
        audio_is_playing: bool,
        media_seconds: float,
    ) -> VideoClockDecision:
        """Return how a video player should follow the song transport clock."""

        if mapping is None:
            return VideoClockDecision(
                should_play=False,
                media_seconds=0.0,
                should_seek=abs(float(media_seconds)) > self.drift_threshold_seconds,
            )
        target_seconds = mapping.media_seconds_for_song_time(song_seconds)
        should_play = bool(audio_is_playing and mapping.contains_song_time(song_seconds))
        return VideoClockDecision(
            should_play=should_play,
            media_seconds=target_seconds,
            should_seek=abs(float(media_seconds) - target_seconds)
            > self.drift_threshold_seconds,
        )


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
        media_available = path.exists()
        return VideoTimelineMapping(
            video_path=str(path),
            start_seconds=float(getattr(layer, "video_start_seconds", 0.0)),
            duration_seconds=_visible_duration_seconds(layer),
            trim_start_seconds=max(
                0.0,
                float(getattr(layer, "video_trim_start_seconds", 0.0)),
            ),
            source_duration_seconds=max(
                0.0,
                float(getattr(layer, "video_duration_seconds", 0.0)),
            ),
            loop_enabled=bool(getattr(layer, "video_loop_enabled", False)),
            media_available=media_available,
            unavailable_reason="" if media_available else f"Missing video file: {path}",
        )
    return None


def _visible_duration_seconds(layer: object) -> float:
    placement = VideoPlacement(
        start_seconds=float(getattr(layer, "video_start_seconds", 0.0)),
        trim_start_seconds=float(getattr(layer, "video_trim_start_seconds", 0.0)),
        visible_duration_seconds=float(
            getattr(layer, "video_visible_duration_seconds", 0.0)
        ),
        source_duration_seconds=float(getattr(layer, "video_duration_seconds", 0.0)),
        loop_enabled=bool(getattr(layer, "video_loop_enabled", False)),
    )
    return placement.normalized().visible_duration_seconds
