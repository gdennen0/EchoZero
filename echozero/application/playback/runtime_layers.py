"""
Compatibility imports for legacy runtime-audio layer names.
Exists because older EZ playback call sites still import the pre-track naming surface.
Connects those callers to the canonical playback-track planning types.
"""

from echozero.application.playback.tracks import PlaybackTrack as RuntimeAudioLayer
from echozero.application.playback.tracks import PlaybackTrackBuilder as RuntimeLayerPlanner
from echozero.application.playback.tracks import PlaybackTrackPlan

__all__ = [
    "PlaybackTrackPlan",
    "RuntimeAudioLayer",
    "RuntimeLayerPlanner",
]
