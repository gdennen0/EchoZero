"""
Audio hardware application contract tests.
Exists to lock requested and resolved hardware state before adapter integrations.
Connects coordinator apply behavior to immutable operation and snapshot records.
"""

from echozero.application.audio_hardware import (
    AudioHardwareApplyResult,
    AudioHardwareCoordinator,
    AudioHardwareDiagnostic,
    AudioHardwareDiagnosticSeverity,
    AudioHardwareGenerationStatus,
    AudioHardwareHealth,
    AudioHardwareOperationStatus,
    RequestedAudioHardware,
    ResolvedAudioHardware,
    audio_hardware_diagnostics_from_playback_state,
    audio_hardware_health_from_diagnostics,
    requested_audio_hardware_from_runtime_config,
)
from echozero.application.playback.models import PlaybackDiagnostics, PlaybackState
from echozero.application.settings import AudioOutputRuntimeConfig


def test_requested_audio_hardware_normalizes_optional_values() -> None:
    request = RequestedAudioHardware(
        device_id=" device-1 ",
        device_name=" ",
        backend_name=" CoreAudio ",
        sample_rate=-1,
        channel_count=0,
        block_size=256,
        latency_seconds=-0.5,
    )

    assert request.device_id == "device-1"
    assert request.device_name is None
    assert request.backend_name == "CoreAudio"
    assert request.sample_rate is None
    assert request.channel_count is None
    assert request.block_size == 256
    assert request.latency_seconds == 0.0


def test_resolved_audio_hardware_normalizes_supported_values() -> None:
    resolved = ResolvedAudioHardware(
        device_id="out-1",
        device_name="Main Out",
        backend_name="CoreAudio",
        sample_rate=48000,
        channel_count=2,
        block_size=0,
        latency_seconds=0.012,
        is_default_device=True,
    )

    assert resolved.block_size is None
    assert resolved.sample_rate == 48000
    assert resolved.channel_count == 2
    assert resolved.is_default_device is True


def test_diagnostics_summarize_health() -> None:
    warning = AudioHardwareDiagnostic(
        severity="warning",
        code="sample_rate_adjusted",
        message="Using nearest supported sample rate.",
    )

    assert warning.severity is AudioHardwareDiagnosticSeverity.WARNING
    assert audio_hardware_health_from_diagnostics((warning,)) is AudioHardwareHealth.DEGRADED
    assert audio_hardware_health_from_diagnostics((), resolved=None) is AudioHardwareHealth.UNKNOWN


def test_route_degradation_marks_audio_hardware_degraded() -> None:
    records = audio_hardware_diagnostics_from_playback_state(
        PlaybackState(
            diagnostics=PlaybackDiagnostics(
                route_resolution_summary=(
                    "routes-exceed-hardware;routes-degraded:layer:outputs_3_4->outside-hardware"
                )
            )
        )
    )

    assert [(record.severity, record.code) for record in records] == [
        (AudioHardwareDiagnosticSeverity.WARNING, "route_resolution")
    ]
    assert audio_hardware_health_from_diagnostics(records) is AudioHardwareHealth.DEGRADED


def test_coordinator_applies_request_and_records_snapshot_state() -> None:
    calls: list[RequestedAudioHardware] = []
    resolved = ResolvedAudioHardware("out-1", "Main Out", "CoreAudio", 48000, 2, 256, 0.01)

    def apply_hardware(request: RequestedAudioHardware) -> AudioHardwareApplyResult:
        calls.append(request)
        return AudioHardwareApplyResult(resolved=resolved)

    coordinator = AudioHardwareCoordinator(apply_hardware, clock=lambda: 10.0)
    snapshot = coordinator.apply_request(
        RequestedAudioHardware(device_id=" out-1 ", sample_rate=48000),
        request_id="request-1",
        operation_id="operation-1",
        generation_id="generation-1",
    )

    assert calls == [RequestedAudioHardware(device_id="out-1", sample_rate=48000)]
    assert snapshot.revision == 1
    assert snapshot.resolved is resolved
    assert snapshot.health is AudioHardwareHealth.HEALTHY
    assert snapshot.generation is not None
    assert snapshot.generation.status is AudioHardwareGenerationStatus.ACTIVE
    assert snapshot.operation is not None
    assert snapshot.operation.status is AudioHardwareOperationStatus.APPLIED
    assert coordinator.latest_snapshot() is snapshot


def test_coordinator_records_failed_apply_as_unavailable_snapshot() -> None:
    def apply_hardware(request: RequestedAudioHardware) -> ResolvedAudioHardware:
        raise RuntimeError(f"{request.device_id} unavailable")

    coordinator = AudioHardwareCoordinator(apply_hardware, clock=lambda: 20.0)
    snapshot = coordinator.apply_request(RequestedAudioHardware(device_id="out-2"))

    assert snapshot.resolved is None
    assert snapshot.health is AudioHardwareHealth.UNAVAILABLE
    assert snapshot.generation is not None
    assert snapshot.generation.status is AudioHardwareGenerationStatus.FAILED
    assert snapshot.operation is not None
    assert snapshot.operation.status is AudioHardwareOperationStatus.FAILED
    assert snapshot.operation.error == "out-2 unavailable"
    assert snapshot.diagnostics[0].code == "apply_failed"


def test_requested_audio_hardware_from_runtime_config_keeps_system_default_as_intent() -> None:
    request = requested_audio_hardware_from_runtime_config(
        AudioOutputRuntimeConfig(
            output_device=None,
            sample_rate=None,
            channels=None,
            stream_blocksize=None,
        )
    )

    assert request.device_id is None
    assert request.device_name == "System Default"
    assert request.backend_name == "sounddevice"
    assert request.sample_rate is None
    assert request.channel_count is None
