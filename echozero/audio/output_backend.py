"""
Audio output backend contracts for EchoZero runtime playback.
Exists because the engine should own transport/mix state while delegating device I/O to one adapter.
Connects `AudioEngine` to concrete libraries such as sounddevice through one stable stream contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np


DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BUFFER_SIZE = 256
DEFAULT_CHANNELS = 1

StreamCallback = Callable[[np.ndarray, int, Any, Any], None]


@dataclass(slots=True, frozen=True)
class AudioOutputConfig:
    """Resolved device and stream configuration for one engine instance."""

    sample_rate: int
    channels: int
    buffer_size: int = DEFAULT_BUFFER_SIZE
    blocksize: int = 0
    latency: str | float = "high"
    prime_output_buffers_using_stream_callback: bool = True
    output_device: int | str | None = None


class AudioOutputStream(Protocol):
    """One live output stream created by a backend adapter."""

    latency: Any

    def start(self) -> None:
        """Start the output stream."""

    def stop(self) -> None:
        """Stop the output stream."""

    def close(self) -> None:
        """Release the output stream."""


class AudioOutputBackend(Protocol):
    """Backend adapter that opens one audio output stream for the engine."""

    name: str

    def resolve_output_config(
        self,
        *,
        sample_rate: int | None,
        channels: int | None,
        buffer_size: int,
        output_device: int | str | None,
        stream_blocksize: int | None,
        stream_latency: str | float | None,
        prime_output_buffers_using_stream_callback: bool,
    ) -> AudioOutputConfig:
        """Resolve one concrete output configuration for a new engine."""

    def open_output_stream(
        self,
        callback: StreamCallback,
        config: AudioOutputConfig,
    ) -> AudioOutputStream:
        """Create one output stream bound to the supplied callback."""
