"""Timeline playback controller imports for the Qt runtime.
Exists to keep the Qt timeline modules on one stable playback import boundary.
Connects timeline UI callers to the canonical application playback controller types.
"""

from echozero.application.playback.models import PlaybackTimingSnapshot as RuntimeAudioTimingSnapshot
from echozero.application.playback.runtime import (
    PlaybackController as TimelinePlaybackController,
)

TimelineRuntimeAudioController = TimelinePlaybackController

__all__ = [
    "RuntimeAudioTimingSnapshot",
    "TimelinePlaybackController",
    "TimelineRuntimeAudioController",
]
