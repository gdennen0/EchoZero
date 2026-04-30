"""
Audio runtime primitives for EchoZero playback.
Exists because the app needs one public import surface for transport, mixing, and output backends.
Connects playback callers to the canonical engine, layer, clock, and backend contracts.
"""

from echozero.audio.clock import Clock, ClockSubscriber, LoopRegion
from echozero.audio.crossfade import CrossfadeBuffer, build_equal_power_curves
from echozero.audio.output_backend import AudioOutputBackend, AudioOutputConfig
from echozero.audio.transport import Transport, TransportState
from echozero.audio.layer import AudioLayer, AudioTrack, resample_buffer
from echozero.audio.mixer import Mixer
from echozero.audio.engine import AudioEngine
from echozero.audio.sounddevice_backend import SounddeviceBackend

__all__ = [
    "AudioEngine",
    "AudioLayer",
    "AudioTrack",
    "AudioOutputBackend",
    "AudioOutputConfig",
    "Clock",
    "ClockSubscriber",
    "LoopRegion",
    "Mixer",
    "SounddeviceBackend",
    "Transport",
    "TransportState",
    "resample_buffer",
]
