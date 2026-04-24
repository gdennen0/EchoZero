"""Shared helpers for audio-engine support cases.
Exists to keep fake streams and common subscribers out of the compatibility wrapper.
Connects the behavior-owned audio-engine support modules to one stable shared seam.
"""

from __future__ import annotations

import numpy as np
import pytest

from echozero.audio.clock import Clock, LoopRegion
from echozero.audio.crossfade import CrossfadeBuffer, build_equal_power_curves
from echozero.audio.transport import Transport, TransportState
from echozero.audio.layer import AudioLayer, resample_buffer
from echozero.audio.mixer import Mixer
from echozero.audio.engine import AudioEngine, _resolve_output_defaults, _resolve_stream_defaults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(samples: int = 44100, freq: float = 440.0, sr: int = 44100) -> np.ndarray:
    """Generate a mono sine wave."""
    t = np.arange(samples, dtype=np.float32) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class FakeStream:
    """Mock audio stream for testing without sounddevice."""

    def __init__(self, **kwargs):
        self.callback = kwargs.get("callback")
        self.device = kwargs.get("device")
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def fake_stream_factory(**kwargs):
    return FakeStream(**kwargs)


class RecordingSubscriber:
    """Clock subscriber that records every tick."""

    def __init__(self):
        self.ticks: list[tuple[int, int]] = []

    def on_clock_tick(self, position_samples: int, sample_rate: int) -> None:
        self.ticks.append((position_samples, sample_rate))


# ===========================================================================
# Clock tests
# ===========================================================================


class TestClock:
    def test_initial_position_is_zero(self) -> None:
        clock = Clock(44100)
        assert clock.position == 0
        assert clock.position_seconds == 0.0

    def test_advance_returns_pre_advance_position(self) -> None:
        clock = Clock(44100)
        pos = clock.advance(256)

__all__ = [name for name in globals() if not name.startswith("__")]
