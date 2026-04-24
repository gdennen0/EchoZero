"""Compatibility import surface for timeline playback runtime types.
Exists to keep the Qt timeline modules on one stable playback import boundary.
Connects timeline UI callers to the canonical application playback runtime types.
"""

from echozero.application.playback.models import PlaybackTimingSnapshot as RuntimeAudioTimingSnapshot
from echozero.application.playback.runtime import PresentationPlaybackRuntime as TimelineRuntimeAudioController

__all__ = [
    "RuntimeAudioTimingSnapshot",
    "TimelineRuntimeAudioController",
]
