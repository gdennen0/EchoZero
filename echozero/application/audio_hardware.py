"""
Audio hardware application contracts and coordination.
Exists so device selection state is typed before runtime-specific adapters apply it.
Connects app requests to resolved hardware snapshots without importing UI or playback.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum

from echozero.application.playback.models import PlaybackState
from echozero.application.settings import AudioOutputRuntimeConfig


class AudioHardwareHealth(str, Enum):
    """App-visible health summary for the current audio hardware selection."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class AudioHardwareGenerationStatus(str, Enum):
    """Lifecycle states for one resolved hardware generation."""
    QUEUED = "queued"
    APPLYING = "applying"
    ACTIVE = "active"
    STALE = "stale"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioHardwareOperationStatus(str, Enum):
    """Lifecycle states for app-visible hardware apply operations."""
    APPLYING = "applying"
    APPLIED = "applied"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioHardwareDiagnosticSeverity(str, Enum):
    """Severity for one hardware diagnostic record."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True, frozen=True)
class RequestedAudioHardware:
    """Operator-requested audio hardware preferences before adapter resolution."""
    device_id: str | None = None
    device_name: str | None = None
    backend_name: str | None = None
    sample_rate: int | None = None
    channel_count: int | None = None
    block_size: int | None = None
    latency_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_id", _optional_text(self.device_id))
        object.__setattr__(self, "device_name", _optional_text(self.device_name))
        object.__setattr__(self, "backend_name", _optional_text(self.backend_name))
        object.__setattr__(self, "sample_rate", _positive_int_or_none(self.sample_rate))
        object.__setattr__(self, "channel_count", _positive_int_or_none(self.channel_count))
        object.__setattr__(self, "block_size", _positive_int_or_none(self.block_size))
        object.__setattr__(self, "latency_seconds", _non_negative_float_or_none(self.latency_seconds))


@dataclass(slots=True, frozen=True)
class ResolvedAudioHardware:
    """Audio hardware settings after an adapter has selected concrete capabilities."""
    device_id: str
    device_name: str
    backend_name: str
    sample_rate: int
    channel_count: int
    block_size: int | None = None
    latency_seconds: float | None = None
    is_default_device: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_id", str(self.device_id or "").strip())
        object.__setattr__(self, "device_name", str(self.device_name or "").strip())
        object.__setattr__(self, "backend_name", str(self.backend_name or "").strip())
        object.__setattr__(self, "sample_rate", max(1, int(self.sample_rate or 0)))
        object.__setattr__(self, "channel_count", max(1, int(self.channel_count or 0)))
        object.__setattr__(self, "block_size", _positive_int_or_none(self.block_size))
        object.__setattr__(self, "latency_seconds", _non_negative_float_or_none(self.latency_seconds))

    def signature(self) -> tuple[object, ...]:
        """Return a stable value tuple for staleness and snapshot comparisons."""

        return (
            self.device_id,
            self.device_name,
            self.backend_name,
            self.sample_rate,
            self.channel_count,
            self.block_size,
            self.latency_seconds,
            self.is_default_device,
        )

@dataclass(slots=True, frozen=True)
class AudioHardwareDiagnostic:
    """One human-readable diagnostic emitted while resolving audio hardware."""
    severity: AudioHardwareDiagnosticSeverity
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", _coerce_diagnostic_severity(self.severity))
        object.__setattr__(self, "code", str(self.code or "").strip())
        object.__setattr__(self, "message", str(self.message or "").strip())


@dataclass(slots=True, frozen=True)
class AudioHardwareApplyResult:
    """Resolved hardware plus diagnostics returned by an adapter apply function."""
    resolved: ResolvedAudioHardware | None
    diagnostics: tuple[AudioHardwareDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics or ()))


AudioHardwareApplyReturn = AudioHardwareApplyResult | ResolvedAudioHardware
AudioHardwareApplyCallable = Callable[[RequestedAudioHardware], AudioHardwareApplyReturn]


@dataclass(slots=True, frozen=True)
class AudioHardwareGenerationState:
    """State for one attempted audio hardware generation."""
    generation_id: str
    request_id: str
    requested: RequestedAudioHardware
    status: AudioHardwareGenerationStatus
    resolved: ResolvedAudioHardware | None = None
    health: AudioHardwareHealth = AudioHardwareHealth.UNKNOWN
    diagnostics: tuple[AudioHardwareDiagnostic, ...] = ()
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(slots=True, frozen=True)
class AudioHardwareOperationState:
    """App-visible operation state for applying a hardware request."""
    operation_id: str
    request_id: str
    generation_id: str
    status: AudioHardwareOperationStatus
    message: str = ""
    started_at: float = 0.0
    updated_at: float = 0.0
    finished_at: float | None = None
    diagnostics: tuple[AudioHardwareDiagnostic, ...] = ()
    error: str | None = None


@dataclass(slots=True, frozen=True)
class AudioHardwareSnapshot:
    """Point-in-time audio hardware state safe for application polling."""
    revision: int
    requested: RequestedAudioHardware = field(default_factory=RequestedAudioHardware)
    resolved: ResolvedAudioHardware | None = None
    health: AudioHardwareHealth = AudioHardwareHealth.UNKNOWN
    generation: AudioHardwareGenerationState | None = None
    operation: AudioHardwareOperationState | None = None
    diagnostics: tuple[AudioHardwareDiagnostic, ...] = ()


class AudioHardwareCoordinator:
    """Coordinates app hardware requests around a supplied apply callable."""
    def __init__(
        self,
        apply_hardware: AudioHardwareApplyCallable,
        *,
        clock: Callable[[], float] | None = None,
        initial_snapshot: AudioHardwareSnapshot | None = None,
    ) -> None:
        self._apply_hardware = apply_hardware
        self._clock = clock or time.time
        self._snapshot = initial_snapshot or AudioHardwareSnapshot(revision=0)

    def apply_request(
        self,
        request: RequestedAudioHardware,
        *,
        request_id: str = "",
        operation_id: str = "",
        generation_id: str = "",
    ) -> AudioHardwareSnapshot:
        """Apply a request and return the resulting hardware snapshot."""
        revision = self._snapshot.revision + 1
        resolved_request = replace(request)
        request_id = str(request_id or f"audio-hardware-request-{revision}").strip()
        generation_id = str(generation_id or f"audio-hardware-generation-{revision}").strip()
        operation_id = str(operation_id or f"audio-hardware-operation-{revision}").strip()
        started_at = _non_negative_float(self._clock())

        try:
            apply_result = _normalize_apply_result(self._apply_hardware(resolved_request))
            resolved = apply_result.resolved
            diagnostics = apply_result.diagnostics
            health = audio_hardware_health_from_diagnostics(diagnostics, resolved=resolved)
            generation_status = AudioHardwareGenerationStatus.ACTIVE
            operation_status = AudioHardwareOperationStatus.APPLIED
            message = "Audio hardware applied"
            error = None
        except Exception as exc:
            diagnostic = AudioHardwareDiagnostic(
                severity=AudioHardwareDiagnosticSeverity.ERROR,
                code="apply_failed",
                message=str(exc) or exc.__class__.__name__,
            )
            resolved = None
            diagnostics = (diagnostic,)
            health = AudioHardwareHealth.UNAVAILABLE
            generation_status = AudioHardwareGenerationStatus.FAILED
            operation_status = AudioHardwareOperationStatus.FAILED
            message = "Audio hardware apply failed"
            error = diagnostic.message

        return self._store_snapshot(
            revision,
            resolved_request,
            resolved,
            diagnostics,
            health,
            generation_status,
            operation_status,
            request_id,
            generation_id,
            operation_id,
            started_at,
            message,
            error,
        )

    def latest_snapshot(self) -> AudioHardwareSnapshot:
        """Return the most recent audio hardware snapshot."""
        return self._snapshot

    def _store_snapshot(
        self,
        revision: int,
        requested: RequestedAudioHardware,
        resolved: ResolvedAudioHardware | None,
        diagnostics: tuple[AudioHardwareDiagnostic, ...],
        health: AudioHardwareHealth,
        generation_status: AudioHardwareGenerationStatus,
        operation_status: AudioHardwareOperationStatus,
        request_id: str,
        generation_id: str,
        operation_id: str,
        started_at: float,
        message: str,
        error: str | None,
    ) -> AudioHardwareSnapshot:
        finished_at = _non_negative_float(self._clock())
        generation = AudioHardwareGenerationState(
            generation_id, request_id, requested, generation_status,
            resolved, health, diagnostics, started_at, finished_at
        )
        operation = AudioHardwareOperationState(
            operation_id, request_id, generation_id, operation_status, message,
            started_at, finished_at, finished_at, diagnostics, error
        )
        self._snapshot = AudioHardwareSnapshot(revision, requested, resolved, health, generation, operation, diagnostics)
        return self._snapshot


def audio_hardware_health_from_diagnostics(
    diagnostics: tuple[AudioHardwareDiagnostic, ...],
    *,
    resolved: ResolvedAudioHardware | None = None,
) -> AudioHardwareHealth:
    """Summarize diagnostics and resolved hardware into one health value."""
    severities = {diagnostic.severity for diagnostic in diagnostics}
    if AudioHardwareDiagnosticSeverity.ERROR in severities:
        return AudioHardwareHealth.UNAVAILABLE
    if AudioHardwareDiagnosticSeverity.WARNING in severities:
        return AudioHardwareHealth.DEGRADED
    if resolved is None:
        return AudioHardwareHealth.UNKNOWN
    return AudioHardwareHealth.HEALTHY


def requested_audio_hardware_from_runtime_config(
    config: AudioOutputRuntimeConfig | None,
) -> RequestedAudioHardware:
    """Build an app hardware request from saved runtime audio settings."""

    if config is None:
        return RequestedAudioHardware(device_name="System Default", backend_name="sounddevice")
    return RequestedAudioHardware(
        device_id=_optional_text(config.output_device),
        device_name=("System Default" if config.output_device is None else None),
        backend_name="sounddevice",
        sample_rate=config.sample_rate,
        channel_count=config.channels,
        block_size=config.stream_blocksize,
    )


def resolved_audio_hardware_from_playback_state(
    state: PlaybackState,
) -> ResolvedAudioHardware | None:
    """Build a resolved hardware record from playback diagnostics."""

    diagnostics = state.diagnostics
    sample_rate = int(state.output_sample_rate or 0)
    channel_count = int(state.output_channels or 0)
    if sample_rate <= 0 or channel_count <= 0:
        return None
    device_id = (
        _optional_text(diagnostics.resolved_output_device)
        or _optional_text(diagnostics.output_device)
        or "default"
    )
    device_name = (
        _optional_text(diagnostics.output_device_name)
        or _optional_text(diagnostics.resolved_output_device)
        or "System Default"
    )
    return ResolvedAudioHardware(
        device_id=device_id,
        device_name=device_name,
        backend_name=str(state.backend_name or "sounddevice"),
        sample_rate=sample_rate,
        channel_count=channel_count,
        block_size=diagnostics.stream_blocksize or None,
        is_default_device=diagnostics.output_device in {None, "", "default"},
    )


def audio_hardware_diagnostics_from_playback_state(
    state: PlaybackState,
) -> tuple[AudioHardwareDiagnostic, ...]:
    """Translate playback diagnostics into hardware-control-room diagnostics."""

    diagnostics = state.diagnostics
    records: list[AudioHardwareDiagnostic] = []
    reason = str(diagnostics.hardware_resolution_reason or "").strip()
    if reason and reason != "system-default":
        severity = (
            AudioHardwareDiagnosticSeverity.WARNING
            if "fallback" in reason or "unavailable" in reason
            else AudioHardwareDiagnosticSeverity.INFO
        )
        records.append(
            AudioHardwareDiagnostic(
                severity=severity,
                code="hardware_resolution",
                message=reason,
            )
        )
    for code, value in (
        ("sample_rate_resolution", diagnostics.sample_rate_resolution_reason),
        ("channel_resolution", diagnostics.channel_resolution_reason),
    ):
        text = str(value or "").strip()
        if text and text not in {"requested", "device-auto", "resolved"}:
            records.append(
                AudioHardwareDiagnostic(
                    severity=AudioHardwareDiagnosticSeverity.WARNING,
                    code=code,
                    message=text,
                )
            )
    if diagnostics.last_ipc_error:
        records.append(
            AudioHardwareDiagnostic(
                severity=AudioHardwareDiagnosticSeverity.WARNING,
                code="ipc_error",
                message=str(diagnostics.last_ipc_error),
            )
        )
    route_summary = str(diagnostics.route_resolution_summary or "").strip()
    if route_summary and route_summary != "routes-fit-hardware":
        records.append(
            AudioHardwareDiagnostic(
                severity=AudioHardwareDiagnosticSeverity.WARNING,
                code="route_resolution",
                message=route_summary,
            )
        )
    return tuple(records)


def _normalize_apply_result(value: AudioHardwareApplyReturn) -> AudioHardwareApplyResult:
    if isinstance(value, AudioHardwareApplyResult):
        return value
    if isinstance(value, ResolvedAudioHardware):
        return AudioHardwareApplyResult(resolved=value)
    raise TypeError(f"Unsupported audio hardware apply result: {type(value).__name__}")


def _coerce_diagnostic_severity(value: AudioHardwareDiagnosticSeverity | str) -> AudioHardwareDiagnosticSeverity:
    return value if isinstance(value, AudioHardwareDiagnosticSeverity) else AudioHardwareDiagnosticSeverity(str(value))


def _optional_text(value: object | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _positive_int_or_none(value: int | None) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    return normalized if normalized > 0 else None


def _non_negative_float_or_none(value: float | int | None) -> float | None:
    return None if value is None else max(0.0, float(value))


def _non_negative_float(value: float | int) -> float:
    return max(0.0, float(value or 0.0))
