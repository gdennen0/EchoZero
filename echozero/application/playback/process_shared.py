"""
Playback IPC contract helpers for process-isolated runtime audio.
Exists because the app and child playback process need one versioned, typed wire contract.
Connects HTTP/WebSocket command payloads to playback runtime state and diagnostics models.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from echozero.application.playback.models import (
    PlaybackDiagnostics,
    PlaybackSource,
    PlaybackState,
    PlaybackTimingSnapshot,
)
from echozero.application.shared.enums import PlaybackMode, PlaybackStatus

PLAYBACK_IPC_VERSION = "1"
PLAYBACK_IPC_TOKEN_HEADER = "x-ez-playback-token"
PLAYBACK_IPC_HOST = "127.0.0.1"
PLAYBACK_IPC_COMMAND_PATH = "/v1/command"
PLAYBACK_IPC_HEALTH_PATH = "/v1/health"
PLAYBACK_IPC_WS_PATH = "/v1/events"


class PlaybackIpcError(RuntimeError):
    """Raised when playback IPC transport or protocol handling fails."""


def encode_timing_snapshot(snapshot: PlaybackTimingSnapshot) -> dict[str, object]:
    """Encode one timing snapshot into a JSON-compatible mapping."""

    return {
        "audible_time_seconds": float(snapshot.audible_time_seconds),
        "clock_time_seconds": float(snapshot.clock_time_seconds),
        "snapshot_monotonic_seconds": (
            float(snapshot.snapshot_monotonic_seconds)
            if snapshot.snapshot_monotonic_seconds is not None
            else None
        ),
        "is_playing": bool(snapshot.is_playing),
        "sample_position": int(snapshot.sample_position),
        "frame_index": int(snapshot.frame_index),
        "timecode_label": str(snapshot.timecode_label),
        "display_label": str(snapshot.display_label),
        "timecode_mode": str(snapshot.timecode_mode),
        "timecode_lock_state": str(snapshot.timecode_lock_state),
        "drift_ppm": float(snapshot.drift_ppm) if snapshot.drift_ppm is not None else None,
        "drift_ms": float(snapshot.drift_ms) if snapshot.drift_ms is not None else None,
    }


def decode_timing_snapshot(payload: dict[str, object]) -> PlaybackTimingSnapshot:
    """Decode one timing snapshot mapping into a typed model."""

    return PlaybackTimingSnapshot(
        audible_time_seconds=float(payload.get("audible_time_seconds", 0.0) or 0.0),
        clock_time_seconds=float(payload.get("clock_time_seconds", 0.0) or 0.0),
        snapshot_monotonic_seconds=(
            float(payload.get("snapshot_monotonic_seconds"))
            if payload.get("snapshot_monotonic_seconds") is not None
            else None
        ),
        is_playing=bool(payload.get("is_playing", False)),
        sample_position=int(payload.get("sample_position", 0) or 0),
        frame_index=int(payload.get("frame_index", 0) or 0),
        timecode_label=str(payload.get("timecode_label", "") or ""),
        display_label=str(payload.get("display_label", "") or ""),
        timecode_mode=str(
            payload.get("timecode_mode", "internal_generated") or "internal_generated"
        ),
        timecode_lock_state=str(payload.get("timecode_lock_state", "locked") or "locked"),
        drift_ppm=(
            float(payload.get("drift_ppm")) if payload.get("drift_ppm") is not None else None
        ),
        drift_ms=(float(payload.get("drift_ms")) if payload.get("drift_ms") is not None else None),
    )


def encode_playback_state(state: PlaybackState) -> dict[str, object]:
    """Encode one playback state into a JSON-compatible mapping."""

    return {
        "status": str(state.status.value),
        "active_sources": [
            {
                "layer_id": str(source.layer_id),
                "take_id": str(source.take_id) if source.take_id is not None else None,
                "source_ref": source.source_ref,
                "mode": str(source.mode.value),
            }
            for source in state.active_sources
        ],
        "latency_ms": float(state.latency_ms),
        "backend_name": str(state.backend_name),
        "active_layer_id": (
            str(state.active_layer_id) if state.active_layer_id is not None else None
        ),
        "active_take_id": str(state.active_take_id) if state.active_take_id is not None else None,
        "output_sample_rate": int(state.output_sample_rate),
        "output_channels": int(state.output_channels),
        "diagnostics": asdict(state.diagnostics),
    }


def decode_playback_state(payload: dict[str, object]) -> PlaybackState:
    """Decode one playback state mapping into a typed model."""

    diagnostics_payload = payload.get("diagnostics", {})
    diagnostics = PlaybackDiagnostics(
        glitch_count=int(_mapping_get(diagnostics_payload, "glitch_count", 0)),
        last_audio_status=_mapping_get(diagnostics_payload, "last_audio_status", None),
        output_device=_mapping_get(diagnostics_payload, "output_device", None),
        stream_latency=_mapping_get(diagnostics_payload, "stream_latency", None),
        stream_blocksize=int(_mapping_get(diagnostics_payload, "stream_blocksize", 0) or 0),
        prime_output_buffers_using_stream_callback=bool(
            _mapping_get(diagnostics_payload, "prime_output_buffers_using_stream_callback", True)
        ),
        last_transition=str(_mapping_get(diagnostics_payload, "last_transition", "") or ""),
        transition_state=str(
            _mapping_get(diagnostics_payload, "transition_state", "stopped") or "stopped"
        ),
        last_track_sync_reason=str(
            _mapping_get(diagnostics_payload, "last_track_sync_reason", "") or ""
        ),
        ramp_samples_remaining=int(
            _mapping_get(diagnostics_payload, "ramp_samples_remaining", 0) or 0
        ),
        last_discontinuity_reason=(
            str(_mapping_get(diagnostics_payload, "last_discontinuity_reason", None))
            if _mapping_get(diagnostics_payload, "last_discontinuity_reason", None) is not None
            else None
        ),
        last_ramp_reason=(
            str(_mapping_get(diagnostics_payload, "last_ramp_reason", None))
            if _mapping_get(diagnostics_payload, "last_ramp_reason", None) is not None
            else None
        ),
        timecode_mode=str(
            _mapping_get(diagnostics_payload, "timecode_mode", "internal_generated")
            or "internal_generated"
        ),
        timecode_lock_state=str(
            _mapping_get(diagnostics_payload, "timecode_lock_state", "locked") or "locked"
        ),
        drift_ppm=(
            float(_mapping_get(diagnostics_payload, "drift_ppm", 0.0))
            if _mapping_get(diagnostics_payload, "drift_ppm", None) is not None
            else None
        ),
        drift_ms=(
            float(_mapping_get(diagnostics_payload, "drift_ms", 0.0))
            if _mapping_get(diagnostics_payload, "drift_ms", None) is not None
            else None
        ),
        structural_rebuild_count=int(
            _mapping_get(diagnostics_payload, "structural_rebuild_count", 0) or 0
        ),
        coalesced_edit_count=int(
            _mapping_get(diagnostics_payload, "coalesced_edit_count", 0) or 0
        ),
        last_structural_rebuild_ms=float(
            _mapping_get(diagnostics_payload, "last_structural_rebuild_ms", 0.0) or 0.0
        ),
        max_structural_rebuild_ms=float(
            _mapping_get(diagnostics_payload, "max_structural_rebuild_ms", 0.0) or 0.0
        ),
        audio_process_connected=bool(
            _mapping_get(diagnostics_payload, "audio_process_connected", False)
        ),
        audio_process_pid=(
            int(_mapping_get(diagnostics_payload, "audio_process_pid", 0) or 0)
            if _mapping_get(diagnostics_payload, "audio_process_pid", None) is not None
            else None
        ),
        ipc_rtt_ms=float(_mapping_get(diagnostics_payload, "ipc_rtt_ms", 0.0) or 0.0),
        last_ipc_error=(
            str(_mapping_get(diagnostics_payload, "last_ipc_error"))
            if _mapping_get(diagnostics_payload, "last_ipc_error", None) is not None
            else None
        ),
        latency_profile=str(_mapping_get(diagnostics_payload, "latency_profile", "") or ""),
        latency_profile_switch_count=int(
            _mapping_get(diagnostics_payload, "latency_profile_switch_count", 0) or 0
        ),
        last_latency_profile_reason=(
            str(_mapping_get(diagnostics_payload, "last_latency_profile_reason"))
            if _mapping_get(diagnostics_payload, "last_latency_profile_reason", None) is not None
            else None
        ),
        rt_command_queue_depth=int(
            _mapping_get(diagnostics_payload, "rt_command_queue_depth", 0) or 0
        ),
        rt_last_apply_latency_ms=float(
            _mapping_get(diagnostics_payload, "rt_last_apply_latency_ms", 0.0) or 0.0
        ),
        rt_last_seek_apply_latency_ms=float(
            _mapping_get(diagnostics_payload, "rt_last_seek_apply_latency_ms", 0.0) or 0.0
        ),
        device_reinit_count=int(_mapping_get(diagnostics_payload, "device_reinit_count", 0) or 0),
    )
    active_sources_payload = payload.get("active_sources", ()) or ()
    active_sources = [
        PlaybackSource(
            layer_id=str(item.get("layer_id", "") or ""),
            take_id=str(item.get("take_id")) if item.get("take_id") else None,
            source_ref=str(item.get("source_ref")) if item.get("source_ref") else None,
            mode=_coerce_playback_mode(item.get("mode")),
        )
        for item in active_sources_payload
        if isinstance(item, dict)
    ]
    return PlaybackState(
        status=_coerce_playback_status(payload.get("status")),
        active_sources=active_sources,
        latency_ms=float(payload.get("latency_ms", 0.0) or 0.0),
        backend_name=str(payload.get("backend_name", "") or ""),
        active_layer_id=(
            str(payload.get("active_layer_id")) if payload.get("active_layer_id") else None
        ),
        active_take_id=(
            str(payload.get("active_take_id")) if payload.get("active_take_id") else None
        ),
        output_sample_rate=int(payload.get("output_sample_rate", 0) or 0),
        output_channels=int(payload.get("output_channels", 0) or 0),
        diagnostics=diagnostics,
    )


def _coerce_playback_mode(value: object) -> PlaybackMode:
    try:
        return PlaybackMode(str(value or "").strip().lower())
    except ValueError:
        return PlaybackMode.NONE


def _coerce_playback_status(value: object) -> PlaybackStatus:
    try:
        return PlaybackStatus(str(value or "").strip().lower())
    except ValueError:
        return PlaybackStatus.STOPPED


def _mapping_get(mapping: object, key: str, default: object) -> Any:
    if not isinstance(mapping, dict):
        return default
    return mapping.get(key, default)


__all__ = [
    "PLAYBACK_IPC_COMMAND_PATH",
    "PLAYBACK_IPC_HEALTH_PATH",
    "PLAYBACK_IPC_HOST",
    "PLAYBACK_IPC_TOKEN_HEADER",
    "PLAYBACK_IPC_VERSION",
    "PLAYBACK_IPC_WS_PATH",
    "PlaybackIpcError",
    "decode_playback_state",
    "decode_timing_snapshot",
    "encode_playback_state",
    "encode_timing_snapshot",
]
