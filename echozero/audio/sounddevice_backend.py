"""
Sounddevice output backend for EchoZero runtime playback.
Exists because `AudioEngine` should depend on one adapter instead of raw `sounddevice` details.
Connects the engine's callback contract to PortAudio-backed output streams and device defaults.
"""

from __future__ import annotations

from typing import Any, Callable

from echozero.audio.output_backend import (
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    AudioOutputBackend,
    AudioOutputConfig,
    StreamCallback,
)


_AUTO_SAMPLE_RATE_PREFERENCE = (48000, 44100, 96000, 88200, 32000)


def _coerce_sample_rate(value: Any, *, default: int = DEFAULT_SAMPLE_RATE) -> int:
    """Convert one backend-provided sample-rate value to a positive integer."""

    try:
        resolved = int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)
    return max(1, resolved)


def _is_output_sample_rate_supported(
    sounddevice_module: Any,
    *,
    output_device: int | str | None,
    channels: int,
    sample_rate: int,
) -> bool:
    """Return True when the device accepts this output format."""

    check_output_settings = getattr(sounddevice_module, "check_output_settings", None)
    if not callable(check_output_settings):
        return False
    try:
        check_output_settings(
            device=output_device,
            channels=max(1, int(channels)),
            dtype="float32",
            samplerate=int(sample_rate),
        )
    except Exception:
        return False
    return True


def _select_auto_output_sample_rate(
    sounddevice_module: Any,
    *,
    output_device: int | str | None,
    channels: int,
    default_sample_rate: int,
) -> int:
    """Choose a stable auto sample rate for the current output device."""

    resolved_default = _coerce_sample_rate(default_sample_rate)
    candidates: list[int] = []
    for candidate in (resolved_default, *_AUTO_SAMPLE_RATE_PREFERENCE):
        sample_rate = _coerce_sample_rate(candidate)
        if sample_rate not in candidates:
            candidates.append(sample_rate)

    for sample_rate in candidates:
        if _is_output_sample_rate_supported(
            sounddevice_module,
            output_device=output_device,
            channels=channels,
            sample_rate=sample_rate,
        ):
            return sample_rate
    return resolved_default


def _resolve_output_defaults(
    stream_factory: Callable[..., Any] | None,
    *,
    output_device: int | str | None = None,
    sounddevice_module: Any | None = None,
) -> tuple[int, int]:
    """Prefer the real output-device format unless one test stream is injected."""

    if stream_factory is not None:
        return DEFAULT_SAMPLE_RATE, DEFAULT_CHANNELS

    try:
        sounddevice = sounddevice_module or __import__("sounddevice")
        resolved_output_device = (
            sounddevice.default.device[1] if output_device is None else output_device
        )
        device_info = sounddevice.query_devices(resolved_output_device)
        max_output_channels = int(device_info.get("max_output_channels", DEFAULT_CHANNELS))
        channels = 2 if max_output_channels >= 2 else max(1, max_output_channels)
        sample_rate = _select_auto_output_sample_rate(
            sounddevice,
            output_device=resolved_output_device,
            channels=channels,
            default_sample_rate=device_info.get("default_samplerate", DEFAULT_SAMPLE_RATE),
        )
        return sample_rate, max(1, channels)
    except Exception:
        return DEFAULT_SAMPLE_RATE, DEFAULT_CHANNELS


def _resolve_stream_defaults(
    stream_factory: Callable[..., Any] | None,
    *,
    buffer_size: int,
    blocksize: int | None,
    latency: str | float | None,
    prime_output_buffers_using_stream_callback: bool,
) -> tuple[int, str | float, bool]:
    """Choose safer real-device stream settings without perturbing injected tests."""

    _ = buffer_size
    if stream_factory is not None:
        resolved_blocksize = 0 if blocksize is None else int(blocksize)
        resolved_latency: str | float = "low" if latency is None else latency
        return (
            resolved_blocksize,
            resolved_latency,
            bool(prime_output_buffers_using_stream_callback),
        )

    resolved_blocksize = 0 if blocksize is None else int(blocksize)
    resolved_latency = "high" if latency is None else latency
    return (
        max(0, resolved_blocksize),
        resolved_latency,
        bool(prime_output_buffers_using_stream_callback),
    )


class SounddeviceBackend(AudioOutputBackend):
    """Sounddevice adapter that resolves device defaults and opens output streams."""

    name = "sounddevice"

    def __init__(
        self,
        *,
        stream_factory: Callable[..., Any] | None = None,
        sounddevice_module: Any | None = None,
    ) -> None:
        self._stream_factory = stream_factory
        self._sounddevice_module = sounddevice_module

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
        """Resolve one stable sounddevice output configuration."""

        resolved_sample_rate, resolved_channels = _resolve_output_defaults(
            self._stream_factory,
            output_device=output_device,
            sounddevice_module=self._sounddevice(),
        )
        blocksize, latency, prime_output = _resolve_stream_defaults(
            self._stream_factory,
            buffer_size=buffer_size,
            blocksize=stream_blocksize,
            latency=stream_latency,
            prime_output_buffers_using_stream_callback=(
                prime_output_buffers_using_stream_callback
            ),
        )
        return AudioOutputConfig(
            sample_rate=int(sample_rate or resolved_sample_rate),
            channels=int(channels or resolved_channels),
            buffer_size=int(buffer_size),
            blocksize=blocksize,
            latency=latency,
            prime_output_buffers_using_stream_callback=prime_output,
            output_device=output_device,
        )

    def open_output_stream(
        self,
        callback: StreamCallback,
        config: AudioOutputConfig,
    ) -> Any:
        """Create one sounddevice output stream from the resolved config."""

        stream_kwargs = {
            "samplerate": config.sample_rate,
            "blocksize": config.blocksize,
            "channels": config.channels,
            "dtype": "float32",
            "latency": config.latency,
            "prime_output_buffers_using_stream_callback": (
                config.prime_output_buffers_using_stream_callback
            ),
            "callback": callback,
        }
        if config.output_device is not None:
            stream_kwargs["device"] = config.output_device
        if self._stream_factory is not None:
            return self._stream_factory(**stream_kwargs)
        return self._sounddevice().OutputStream(**stream_kwargs)

    def _sounddevice(self) -> Any:
        if self._sounddevice_module is not None:
            return self._sounddevice_module
        return __import__("sounddevice")
