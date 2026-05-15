"""App settings models for machine-local EchoZero preferences.
Exists because audio device and OSC wiring belong to the app environment, not project truth.
Connects persisted local preferences to launch-time and runtime configuration resolution.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from echozero.application.timeline.object_actions.descriptors import (
    ActionDescriptor,
    action_descriptors,
    descriptor_for_action,
)
from echozero.output_routing import (
    DEFAULT_MASTER_OUTPUT_BUS,
    canonical_master_output_buses,
)

_IMPORT_ACTION_PRIORITY: dict[str, int] = {
    "timeline.extract_stems": 0,
    "timeline.extract_song_drum_events": 1,
    "timeline.extract_drum_events": 2,
    "timeline.extract_classified_drums": 3,
}


def import_safe_pipeline_action_descriptors() -> tuple[ActionDescriptor, ...]:
    """Return the pipeline actions that can safely run unattended during import."""

    descriptors = [
        descriptor
        for descriptor in action_descriptors()
        if _is_import_safe_pipeline_action(descriptor)
    ]
    descriptors.sort(
        key=lambda descriptor: (
            _IMPORT_ACTION_PRIORITY.get(descriptor.action_id, 1000),
            descriptor.label,
            descriptor.action_id,
        )
    )
    return tuple(descriptors)


def canonical_import_pipeline_action_ids(
    action_ids: Iterable[str] | None,
) -> tuple[str, ...]:
    """Canonicalize and dedupe import pipeline action IDs to import-safe actions."""

    if action_ids is None:
        return ()
    allowed_action_ids = {
        descriptor.action_id for descriptor in import_safe_pipeline_action_descriptors()
    }
    resolved: list[str] = []
    seen: set[str] = set()
    for action_id in action_ids:
        text = str(action_id).strip()
        if not text:
            continue
        descriptor = descriptor_for_action(text)
        if descriptor is None:
            continue
        canonical_id = descriptor.action_id
        if canonical_id not in allowed_action_ids or canonical_id in seen:
            continue
        seen.add(canonical_id)
        resolved.append(canonical_id)
    return tuple(resolved)


def _is_import_safe_pipeline_action(descriptor: ActionDescriptor) -> bool:
    if descriptor.action_id not in _IMPORT_ACTION_PRIORITY:
        return False
    if "layer" not in descriptor.object_types:
        return False
    if descriptor.workflow_id is None or descriptor.pipeline_template_id is None:
        return False
    params_schema = descriptor.params_schema or {}
    if set(params_schema.keys()) != {"layer_id"}:
        return False
    return str(params_schema.get("layer_id")).strip().lower() == "required"


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
    master_output_bus: str = DEFAULT_MASTER_OUTPUT_BUS
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
class SongImportPreferences:
    """Machine-local defaults for song/version import behavior."""

    strip_ltc_timecode: bool = True
    pipeline_action_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pipeline_action_ids",
            canonical_import_pipeline_action_ids(self.pipeline_action_ids),
        )


@dataclass(slots=True, frozen=True)
class AppPreferences:
    """Top-level machine-local application preferences."""

    audio_output: AudioOutputPreferences = field(default_factory=AudioOutputPreferences)
    ma3_osc: MA3OscPreferences = field(default_factory=MA3OscPreferences)
    song_import: SongImportPreferences = field(default_factory=SongImportPreferences)
    pipeline_defaults_by_template: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_project_paths: tuple[str, ...] = ()


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
    master_output_bus: str = DEFAULT_MASTER_OUTPUT_BUS
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
    song_import_changed: bool = False
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
    song_import = _mapping_or_empty(data.get("song_import"))
    pipeline_action_ids = canonical_import_pipeline_action_ids(
        _coerce_pipeline_action_ids(
            song_import.get("pipeline_action_ids", song_import.get("action_ids"))
        )
    )

    return AppPreferences(
        audio_output=AudioOutputPreferences(
            output_device=_coerce_optional_text(audio.get("output_device")),
            sample_rate=_coerce_optional_positive_int(audio.get("sample_rate")),
            output_channels=_coerce_optional_positive_int(audio.get("output_channels")),
            master_output_bus=_coerce_output_bus_tokens(audio.get("master_output_bus")),
            latency_profile=_coerce_latency_profile(audio.get("latency_profile")),
            blocksize=_coerce_optional_positive_int(audio.get("blocksize")),
            prime_output_buffers_using_stream_callback=bool(
                audio.get("prime_output_buffers_using_stream_callback", True)
            ),
        ),
        ma3_osc=MA3OscPreferences(
            receive=OscReceivePreferences(
                enabled=bool(receive.get("enabled", False)),
                host=_coerce_text(receive.get("host"), default="127.0.0.1"),
                port=_coerce_non_negative_int(receive.get("port"), default=0),
            ),
            send=OscSendPreferences(
                enabled=bool(send.get("enabled", False)),
                host=_coerce_text(send.get("host"), default="127.0.0.1"),
                port=_coerce_optional_positive_int(send.get("port")),
            ),
        ),
        song_import=SongImportPreferences(
            strip_ltc_timecode=bool(song_import.get("strip_ltc_timecode", True)),
            pipeline_action_ids=pipeline_action_ids,
        ),
        pipeline_defaults_by_template=_coerce_pipeline_defaults_by_template(
            data.get("pipeline_defaults_by_template")
        ),
        recent_project_paths=_coerce_recent_project_paths(data.get("recent_project_paths")),
    )


def _mapping_or_empty(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _coerce_output_bus_tokens(value: object) -> str:
    tokens = canonical_master_output_buses(value)
    return ",".join(tokens) if tokens else DEFAULT_MASTER_OUTPUT_BUS


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


def _coerce_pipeline_action_ids(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",")]
        return tuple(token for token in tokens if token)
    if isinstance(value, (list, tuple, set)):
        resolved: list[str] = []
        for token in value:
            text = str(token).strip()
            if text:
                resolved.append(text)
        return tuple(resolved)
    return ()


def _coerce_pipeline_defaults_by_template(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    resolved: dict[str, dict[str, Any]] = {}
    for raw_template_id, raw_defaults in value.items():
        template_id = str(raw_template_id).strip()
        if not template_id or not isinstance(raw_defaults, dict):
            continue
        normalized = {
            str(raw_key).strip(): raw_value
            for raw_key, raw_value in raw_defaults.items()
            if str(raw_key).strip()
        }
        resolved[template_id] = normalized
    return resolved


def _coerce_recent_project_paths(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    entries: list[str] = []
    for candidate in value:
        text = str(candidate or "").strip()
        if text:
            entries.append(text)
    return tuple(entries)
