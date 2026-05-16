"""
Fake audio output backend for runtime playback diagnostics.
Exists so app-path tests can exercise output callbacks without speaker hardware.
Connects playback engines to deterministic streams that tests can render manually.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from echozero.audio.output_backend import AudioOutputConfig, StreamCallback

DEFAULT_FAKE_BLOCK_FRAMES = 256


class FakeOutputStream:
    """In-process output stream that exposes the engine callback for diagnostics."""

    def __init__(self, callback: StreamCallback, *, latency: str | float = 0.0) -> None:
        self.callback = callback
        self.latency = latency
        self.started = False
        self.closed = False

    def start(self) -> None:
        """Mark the fake stream active."""

        self.started = True

    def stop(self) -> None:
        """Mark the fake stream inactive."""

        self.started = False

    def close(self) -> None:
        """Mark the fake stream closed."""

        self.closed = True


class FakeOutputBackend:
    """Audio backend that resolves output config without opening hardware."""

    name = "fake-output"

    def __init__(self, *, default_sample_rate: int = 44100, default_channels: int = 2) -> None:
        self.default_sample_rate = max(1, int(default_sample_rate))
        self.default_channels = max(1, int(default_channels))
        self.streams: list[FakeOutputStream] = []

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
        """Resolve a deterministic fake output configuration."""

        resolved_sample_rate = sample_rate or self.default_sample_rate
        resolved_channels = channels or self.default_channels
        return AudioOutputConfig(
            sample_rate=int(resolved_sample_rate),
            channels=int(resolved_channels),
            buffer_size=int(buffer_size),
            blocksize=int(stream_blocksize or DEFAULT_FAKE_BLOCK_FRAMES),
            latency=stream_latency or 0.0,
            prime_output_buffers_using_stream_callback=(
                prime_output_buffers_using_stream_callback
            ),
            output_device=output_device,
            requested_output_device=output_device,
            resolved_output_device=output_device,
            resolved_output_device_name="Fake Output",
            requested_sample_rate=sample_rate,
            requested_channels=channels,
            device_max_output_channels=int(resolved_channels),
            hardware_resolution_reason="injected-fake-output",
            sample_rate_resolution_reason="requested" if sample_rate is not None else "default",
            channel_resolution_reason="requested" if channels is not None else "default",
        )

    def open_output_stream(
        self,
        callback: StreamCallback,
        config: AudioOutputConfig,
    ) -> FakeOutputStream:
        """Create one fake stream and retain it for manual callback execution."""

        stream = FakeOutputStream(callback, latency=config.latency)
        self.streams.append(stream)
        return stream


@dataclass(slots=True, frozen=True)
class RenderedBlockStats:
    """Summary statistics for one manually rendered fake-output callback block."""

    frames: int
    channels: int
    peak_abs: float
    rms: float
    max_discontinuity: float


def render_fake_output_blocks(
    stream: FakeOutputStream,
    *,
    channels: int,
    block_frames: int = DEFAULT_FAKE_BLOCK_FRAMES,
    block_count: int = 1,
) -> list[RenderedBlockStats]:
    """Render callback blocks from a fake stream and return output statistics."""

    stats: list[RenderedBlockStats] = []
    previous_tail: npt.NDArray[np.float32] | None = None
    for _index in range(max(1, int(block_count))):
        outdata: npt.NDArray[np.float32] = np.zeros(
            (max(1, int(block_frames)), max(1, int(channels))),
            dtype=np.float32,
        )
        stream.callback(outdata, int(block_frames), None, None)
        block = np.asarray(outdata, dtype=np.float32)
        if block.shape[0] > 1:
            within_delta = float(np.max(np.abs(np.diff(block, axis=0))))
        else:
            within_delta = 0.0
        if previous_tail is not None and block.shape[0] > 0:
            boundary_delta = float(np.max(np.abs(block[0] - previous_tail)))
        else:
            boundary_delta = 0.0
        previous_tail = np.array(block[-1], dtype=np.float32, copy=True)
        stats.append(
            RenderedBlockStats(
                frames=int(block.shape[0]),
                channels=int(block.shape[1]) if block.ndim == 2 else 1,
                peak_abs=float(np.max(np.abs(block))) if block.size else 0.0,
                rms=float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0,
                max_discontinuity=max(within_delta, boundary_delta),
            )
        )
    return stats


__all__ = [
    "DEFAULT_FAKE_BLOCK_FRAMES",
    "FakeOutputBackend",
    "FakeOutputStream",
    "RenderedBlockStats",
    "render_fake_output_blocks",
]
