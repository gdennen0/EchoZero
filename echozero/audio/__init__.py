"""
Audio engine: DAW-grade playback with sample-accurate clock.
Process-agnostic — lives in UI process for zero-jitter playhead sync,
but has no dependencies on Qt or pipeline engine.

Ship-ready: lock-free callback, zero allocations, clipping protection,
auto-stop, sample rate conversion, thread-safe subscriber management.
"""

from echozero.audio.clock import Clock, ClockSubscriber, LoopRegion
from echozero.audio.transport import Transport, TransportState
from echozero.audio.layer import AudioLayer, resample_buffer
from echozero.audio.mixer import Mixer
from echozero.audio.engine import AudioEngine

__all__ = [
    "AudioEngine",
    "AudioLayer",
    "Clock",
    "ClockSubscriber",
    "LoopRegion",
    "Mixer",
    "Transport",
    "TransportState",
    "resample_buffer",
]
