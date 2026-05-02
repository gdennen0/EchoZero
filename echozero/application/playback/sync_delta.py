"""
Playback sync delta classifier for app-side runtime audio decisions.
Exists because the UI must decide playback sync work locally without blocking on runtime IPC.
Connects compact playback projections to one shared none/mix/structure change model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

from echozero.application.playback.sync_projection import PlaybackSyncPayload
from echozero.application.playback.track_identity import (
    event_slice_signature,
    sanitize_output_bus_for_channels,
)
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.enums import LayerKind, PlaybackMode


class PlaybackChangeKind(StrEnum):
    """Kinds of app-side playback runtime change."""

    NONE = "none"
    MIX_ONLY = "mix_only"
    STRUCTURE = "structure"


@dataclass(slots=True, frozen=True)
class PlaybackSyncDelta:
    """One classified playback-sync transition."""

    payload: PlaybackSyncPayload
    change_kind: PlaybackChangeKind
    projection_build_ms: float
    classify_ms: float


def classify_playback_sync_change(
    previous_payload: PlaybackSyncPayload | None,
    next_source: TimelinePresentation | PlaybackSyncPayload,
) -> PlaybackSyncDelta:
    """Classify one playback change without touching the runtime-audio backend."""

    build_started = perf_counter()
    if isinstance(next_source, PlaybackSyncPayload):
        next_payload = next_source
        projection_build_ms = 0.0
    else:
        next_payload = PlaybackSyncPayload.from_presentation(next_source)
        projection_build_ms = max(0.0, (perf_counter() - build_started) * 1000.0)

    classify_started = perf_counter()
    if previous_payload is None:
        next_structure = _structure_signature(next_payload)
        next_mix = _mix_signature(next_payload)
        change_kind = (
            PlaybackChangeKind.STRUCTURE
            if next_structure or next_mix
            else PlaybackChangeKind.NONE
        )
    else:
        previous_structure = _structure_signature(previous_payload)
        next_structure = _structure_signature(next_payload)
        if previous_structure != next_structure:
            change_kind = PlaybackChangeKind.STRUCTURE
        else:
            previous_mix = _mix_signature(previous_payload)
            next_mix = _mix_signature(next_payload)
            if previous_mix != next_mix:
                change_kind = PlaybackChangeKind.MIX_ONLY
            else:
                change_kind = PlaybackChangeKind.NONE
    classify_ms = max(0.0, (perf_counter() - classify_started) * 1000.0)

    return PlaybackSyncDelta(
        payload=next_payload,
        change_kind=change_kind,
        projection_build_ms=projection_build_ms,
        classify_ms=classify_ms,
    )


def _structure_signature(payload: PlaybackSyncPayload) -> tuple[tuple[str, str], ...]:
    tracks = _select_playback_tracks(payload)
    return tuple(
        (track.track_id, track.source_key)
        for track in tracks
    )


def _mix_signature(payload: PlaybackSyncPayload) -> tuple[tuple[str, str], ...]:
    tracks = _select_playback_tracks(payload)
    return tuple(
        (
            track.track_id,
            (
                f"{int(track.muted)}|{track.gain_db:.6f}|"
                f"{track.output_bus or 'outputs_1_2'}|{payload.playback_output_channels}"
            ),
        )
        for track in tracks
    )


@dataclass(slots=True, frozen=True)
class _SyncTrackIdentity:
    track_id: str
    source_key: str
    gain_db: float
    muted: bool
    output_bus: str | None


def _select_playback_tracks(payload: PlaybackSyncPayload) -> tuple[_SyncTrackIdentity, ...]:
    playable_layers = [layer for layer in payload.layers if _layer_has_playable_source(layer)]
    if not playable_layers:
        return ()

    has_soloed_layers = any(bool(layer.soloed) for layer in playable_layers)
    selected_layer_id = payload.selected_layer_id
    selected_take_id = payload.selected_take_id
    tracks: list[_SyncTrackIdentity] = []
    seen_track_ids: set[str] = set()

    for layer in playable_layers:
        if str(layer.layer_id) == str(selected_layer_id) and selected_take_id is not None:
            identity = _track_identity_for_target(
                payload,
                layer_id=str(layer.layer_id),
                take_id=str(selected_take_id),
            )
        else:
            identity = _track_identity_from_layer(
                payload,
                layer,
            )
        if identity is None or identity.track_id in seen_track_ids:
            continue
        effective_muted = bool(layer.muted) or (has_soloed_layers and not bool(layer.soloed))
        tracks.append(
            _SyncTrackIdentity(
                track_id=identity.track_id,
                source_key=identity.source_key,
                gain_db=identity.gain_db,
                muted=effective_muted,
                output_bus=identity.output_bus,
            )
        )
        seen_track_ids.add(identity.track_id)

    return tuple(tracks)


def _track_identity_for_target(
    payload: PlaybackSyncPayload,
    *,
    layer_id: str,
    take_id: str | None,
) -> _SyncTrackIdentity | None:
    for layer in payload.layers:
        if str(layer.layer_id) != layer_id:
            continue
        if take_id is not None:
            for take in layer.takes:
                if str(take.take_id) == take_id:
                    identity = _track_identity_from_take(payload, layer, take)
                    if identity is not None:
                        return identity
        return _track_identity_from_layer(payload, layer)
    return None


def _track_identity_from_layer(
    payload: PlaybackSyncPayload,
    layer,
) -> _SyncTrackIdentity | None:
    if layer.source_audio_path and layer.kind is not LayerKind.EVENT:
        return _SyncTrackIdentity(
            track_id=str(layer.layer_id),
            source_key=f"audio:{layer.source_audio_path}",
            gain_db=float(layer.gain_db),
            muted=bool(layer.muted),
            output_bus=sanitize_output_bus_for_channels(
                layer.output_bus,
                playback_output_channels=payload.playback_output_channels,
            ),
        )
    if not _is_event_track_source(layer):
        return None
    return _event_track_identity(
        payload,
        track_id=str(layer.layer_id),
        gain_db=float(layer.gain_db),
        muted=bool(layer.muted),
        output_bus=layer.output_bus,
        playback_source_ref=str(layer.playback_source_ref or ""),
        events=layer.events,
    )


def _track_identity_from_take(
    payload: PlaybackSyncPayload,
    layer,
    take,
) -> _SyncTrackIdentity | None:
    layer_id = str(layer.layer_id)
    take_id = str(take.take_id)
    if take.source_audio_path and layer.kind is not LayerKind.EVENT:
        return _SyncTrackIdentity(
            track_id=f"{layer_id}:{take_id}",
            source_key=f"audio:{take.source_audio_path}",
            gain_db=float(layer.gain_db),
            muted=bool(layer.muted),
            output_bus=sanitize_output_bus_for_channels(
                layer.output_bus,
                playback_output_channels=payload.playback_output_channels,
            ),
        )
    if not _is_event_track_source(take):
        return None
    return _event_track_identity(
        payload,
        track_id=f"{layer_id}:{take_id}",
        gain_db=float(layer.gain_db),
        muted=bool(layer.muted),
        output_bus=layer.output_bus,
        playback_source_ref=str(take.playback_source_ref or ""),
        events=take.events,
    )


def _event_track_identity(
    payload: PlaybackSyncPayload,
    *,
    track_id: str,
    gain_db: float,
    muted: bool,
    output_bus: str | None,
    playback_source_ref: str,
    events: tuple[object, ...] | list[object],
) -> _SyncTrackIdentity:
    source_key = f"event:{playback_source_ref}:{event_slice_signature(list(events))}"
    return _SyncTrackIdentity(
        track_id=track_id,
        source_key=source_key,
        gain_db=gain_db,
        muted=muted,
        output_bus=sanitize_output_bus_for_channels(
            output_bus,
            playback_output_channels=payload.playback_output_channels,
        ),
    )


def _layer_has_playable_source(layer) -> bool:
    has_continuous_source = bool(layer.source_audio_path and layer.kind is not LayerKind.EVENT)
    return bool(has_continuous_source or _is_event_track_source(layer))


def _is_event_track_source(layer) -> bool:
    return bool(
        layer.kind is LayerKind.EVENT
        and bool(layer.playback_enabled)
        and layer.playback_mode == PlaybackMode.EVENT_SLICE
        and bool(layer.playback_source_ref)
    )
