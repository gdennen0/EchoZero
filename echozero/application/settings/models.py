"""App settings models for machine-local EchoZero preferences.
Exists because audio device and OSC wiring belong to the app environment, not project truth.
Connects persisted local preferences to launch-time and runtime configuration resolution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AudioLatencyProfile(str, Enum):
    """User-facing latency policy for audio output streams."""

    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


@dataclass(slots=True, frozen=True)
class AudioOutputPreferences:
    """Machine-local audio output preferences for the runtime engine."""

    output_device: str | None = None
    sample_rate: int | None = None
    output_channels: int | None = None
    latency_profile: AudioLatencyProfile = AudioLatencyProfile.AUTO
    blocksize: int | None = None
    prime_output_buffers_using_stream_callback: bool = True


@dataclass(slots=True, frozen=True)
class OscReceivePreferences:
    """Saved OSC receive endpoint settings for one machine-local integration."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 0


@dataclass(slots=True, frozen=True)
class OscSendPreferences:
    """Saved OSC send endpoint settings for one machine-local integration."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int | None = None


@dataclass(slots=True, frozen=True)
class MA3OscPreferences:
    """Machine-local MA3 OSC send, receive, and protocol preferences."""

    receive: OscReceivePreferences = field(default_factory=OscReceivePreferences)
    send: OscSendPreferences = field(default_factory=OscSendPreferences)


@dataclass(slots=True, frozen=True)
class AppPreferences:
    """Top-level machine-local application preferences."""

    audio_output: AudioOutputPreferences = field(default_factory=AudioOutputPreferences)
    ma3_osc: MA3OscPreferences = field(default_factory=MA3OscPreferences)


@dataclass(slots=True, frozen=True)
class AppSettingsLaunchOverrides:
    """Launch-time overrides that take precedence over saved app preferences."""

    ma3_osc_listen_host: str | None = None
    ma3_osc_listen_port: int | None = None
    ma3_osc_command_host: str | None = None
    ma3_osc_command_port: int | None = None


@dataclass(slots=True, frozen=True)
class AudioOutputRuntimeConfig:
    """Resolved runtime audio configuration for AudioEngine creation."""

    output_device: int | str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    stream_latency: str | float | None = None
    stream_blocksize: int | None = None
    prime_output_buffers_using_stream_callback: bool = True


@dataclass(slots=True, frozen=True)
class OscReceiveRuntimeConfig:
    """Resolved OSC receive binding used to start one runtime listener."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 0
    path: str = "/ez/message"


@dataclass(slots=True, frozen=True)
class OscSendRuntimeConfig:
    """Resolved OSC send target used to start one runtime sender."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int | None = None
    path: str = "/cmd"


@dataclass(slots=True, frozen=True)
class MA3OscRuntimeConfig:
    """Resolved runtime MA3 OSC configuration for bridge startup."""

    receive: OscReceiveRuntimeConfig = field(default_factory=OscReceiveRuntimeConfig)
    send: OscSendRuntimeConfig = field(default_factory=OscSendRuntimeConfig)

    @property
    def is_enabled(self) -> bool:
        """True when any MA3 OSC runtime path should be initialized."""

        return self.receive.enabled or self.send.enabled


@dataclass(slots=True, frozen=True)
class AppSettingsUpdateResult:
    """Save result describing which runtime areas changed."""

    preferences: AppPreferences
    audio_changed: bool = False
    osc_changed: bool = False
    restart_required: bool = False
    restart_reasons: tuple[str, ...] = ()


def app_preferences_to_dict(preferences: AppPreferences) -> dict[str, Any]:
    """Serialize app preferences into a JSON-compatible mapping."""

    return asdict(preferences)


def app_preferences_from_dict(payload: dict[str, Any] | None) -> AppPreferences:
    """Deserialize app preferences from a JSON-compatible mapping."""

    data = payload or {}
    audio = _mapping_or_empty(data.get("audio_output"))
    osc = _mapping_or_empty(data.get("ma3_osc"))
    receive = _mapping_or_empty(osc.get("receive"))
    send = _mapping_or_empty(osc.get("send"))

    return AppPreferences(
        audio_output=AudioOutputPreferences(
            output_device=_coerce_optional_text(audio.get("output_device")),
            sample_rate=_coerce_optional_positive_int(audio.get("sample_rate")),
            output_channels=_coerce_optional_positive_int(audio.get("output_channels")),
            latency_profile=_coerce_latency_profile(audio.get("latency_profile")),
            blocksize=_coerce_optional_positive_int(audio.get("blocksize")),
            prime_output_buffers_using_stream_callback=bool(
                audio.get("prime_output_buffers_using_stream_callback", True)
            ),
        ),
        ma3_osc=MA3OscPreferences(
            receive=OscReceivePreferences(
                enabled=bool(receive.get("enabled", osc.get("listen_enabled", False))),
                host=_coerce_text(
                    receive.get("host", osc.get("listen_host")),
                    default="127.0.0.1",
                ),
                port=_coerce_non_negative_int(
                    receive.get("port", osc.get("listen_port")),
                    default=0,
                ),
            ),
            send=OscSendPreferences(
                enabled=bool(send.get("enabled", osc.get("command_enabled", False))),
                host=_coerce_text(
                    send.get("host", osc.get("command_host")),
                    default="127.0.0.1",
                ),
                port=_coerce_optional_positive_int(
                    send.get("port", osc.get("command_port"))
                ),
            ),
        ),
    )


def _mapping_or_empty(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _coerce_latency_profile(value: object) -> AudioLatencyProfile:
    try:
        return AudioLatencyProfile(str(value or AudioLatencyProfile.AUTO.value).strip().lower())
    except ValueError:
        return AudioLatencyProfile.AUTO


def _coerce_optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _coerce_text(value: object, *, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _coerce_non_negative_int(value: object, *, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _coerce_optional_positive_int(value: object) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None
