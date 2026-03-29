"""
Audio engine: DAW-grade playback with sample-accurate clock.
Process-agnostic — lives in UI process for zero-jitter playhead sync,
but has no dependencies on Qt or pipeline engine.
"""

from echozero.audio.clock import Clock, ClockSubscriber
from echozero.audio.transport import Transport, TransportState
from echozero.audio.layer import AudioLayer
from echozero.audio.mixer import Mixer
from echozero.audio.engine import AudioEngine

__all__ = [
    "AudioEngine",
    "AudioLayer",
    "Clock",
    "ClockSubscriber",
    "Mixer",
    "Transport",
    "TransportState",
]
