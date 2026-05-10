"""Playback application models."""

from dataclasses import dataclass, field

from echozero.application.shared.enums import PlaybackMode, PlaybackStatus
from echozero.application.shared.ids import LayerId, TakeId


@dataclass(slots=True)
class PlaybackSource:
    layer_id: LayerId
    take_id: TakeId | None = None
    source_ref: str | None = None
    mode: PlaybackMode = PlaybackMode.NONE


@dataclass(slots=True)
class LayerPlaybackState:
    mode: PlaybackMode = PlaybackMode.NONE
    enabled: bool = False
    armed_source_ref: str | None = None
    preloaded: bool = False
    supports_scrub: bool = False
    supports_loop: bool = True


@dataclass(slots=True)
class PlaybackDiagnostics:
    glitch_count: int = 0
    last_audio_status: str | None = None
    output_device: str | None = None
    stream_latency: str | float | None = None
    stream_blocksize: int = 0
    prime_output_buffers_using_stream_callback: bool = True
    last_transition: str = ""
    transition_state: str = "stopped"
    last_track_sync_reason: str = ""
    ramp_samples_remaining: int = 0
    last_discontinuity_reason: str | None = None
    last_ramp_reason: str | None = None
    timecode_mode: str = "internal_generated"
    timecode_lock_state: str = "locked"
    drift_ppm: float | None = None
    drift_ms: float | None = None
    structural_rebuild_count: int = 0
    coalesced_edit_count: int = 0
    last_structural_rebuild_ms: float = 0.0
    max_structural_rebuild_ms: float = 0.0
    audio_process_connected: bool = False
    audio_process_pid: int | None = None
    ipc_rtt_ms: float = 0.0
    last_ipc_error: str | None = None
    latency_profile: str = ""
    latency_profile_switch_count: int = 0
    last_latency_profile_reason: str | None = None
    local_projection_build_ms: float = 0.0
    local_sync_classify_ms: float = 0.0
    last_local_sync_change_kind: str = ""
    last_ipc_command: str = ""
    rt_command_queue_depth: int = 0
    rt_last_apply_latency_ms: float = 0.0
    rt_last_seek_apply_latency_ms: float = 0.0
    device_reinit_count: int = 0


@dataclass(slots=True)
class PlaybackState:
    status: PlaybackStatus = PlaybackStatus.STOPPED
    active_sources: list[PlaybackSource] = field(default_factory=list)
    latency_ms: float = 0.0
    backend_name: str = "unconfigured"
    active_layer_id: LayerId | None = None
    active_take_id: TakeId | None = None
    output_sample_rate: int = 0
    output_channels: int = 0
    diagnostics: PlaybackDiagnostics = field(default_factory=PlaybackDiagnostics)


@dataclass(slots=True, frozen=True)
class PlaybackTimingSnapshot:
    audible_time_seconds: float
    clock_time_seconds: float
    snapshot_monotonic_seconds: float | None
    is_playing: bool
    sample_position: int = 0
    frame_index: int = 0
    timecode_label: str = ""
    display_label: str = ""
    timecode_mode: str = "internal_generated"
    timecode_lock_state: str = "locked"
    drift_ppm: float | None = None
    drift_ms: float | None = None
