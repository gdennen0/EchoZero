"""
Audible playback intent contracts for EchoZero.
Exists so UI selection and presentation details cannot silently become playback truth.
Connects timeline-derived playback projections to graph preparation and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

from echozero.application.playback.sync_projection import (
    PlaybackSyncEventProjection,
    PlaybackSyncLayerProjection,
    PlaybackSyncPayload,
)
from echozero.application.playback.track_identity import (
    event_slice_signature,
    sanitize_output_bus_for_channels,
)
from echozero.application.shared.enums import LayerKind, PlaybackMode


@dataclass(slots=True, frozen=True)
class AudibleEventSpec:
    """One scheduled source region in timeline time."""

    start_seconds: float
    end_seconds: float
    muted: bool = False

    @property
    def duration_seconds(self) -> float:
        return max(0.0, float(self.end_seconds) - float(self.start_seconds))


@dataclass(slots=True, frozen=True)
class AudibleLayerSpec:
    """One layer's app-authored audio intent before graph preparation."""

    layer_id: str
    source_key: str
    source_ref: str
    mode: PlaybackMode
    output_bus: str | None
    gain_db: float
    muted: bool
    soloed: bool
    events: tuple[AudibleEventSpec, ...] = ()

    @property
    def structure_signature(self) -> tuple[str, str]:
        return self.layer_id, self.source_key

    @property
    def mix_signature(self) -> tuple[str, str]:
        return (
            self.layer_id,
            (
                f"{int(self.muted)}|{int(self.soloed)}|{self.gain_db:.6f}|"
                f"{self.output_bus or ''}"
            ),
        )


@dataclass(slots=True, frozen=True)
class AudibleIntent:
    """Immutable playback intent extracted from app truth."""

    layers: tuple[AudibleLayerSpec, ...]
    playback_output_channels: int = 0

    @property
    def structure_signature(self) -> tuple[tuple[str, str], ...]:
        return tuple(layer.structure_signature for layer in self.layers)

    @property
    def mix_signature(self) -> tuple[tuple[str, str], ...]:
        return tuple(layer.mix_signature for layer in self.layers)


def audible_intent_from_sync_payload(payload: PlaybackSyncPayload) -> AudibleIntent:
    """Build audible intent from the compact playback projection.

    Selection fields on the payload are intentionally ignored. A take can become audible
    only through a future explicit audition/playback-focus command, not by being selected.
    """

    layers = tuple(
        layer_spec
        for layer in payload.layers
        for layer_spec in (_audible_layer_from_projection(layer, payload),)
        if layer_spec is not None
    )
    has_soloed_layers = any(layer.soloed for layer in layers)
    if not has_soloed_layers:
        return AudibleIntent(
            layers=layers,
            playback_output_channels=max(0, int(payload.playback_output_channels)),
        )
    return AudibleIntent(
        layers=tuple(
            layer if layer.soloed else _muted_copy(layer)
            for layer in layers
        ),
        playback_output_channels=max(0, int(payload.playback_output_channels)),
    )


def _audible_layer_from_projection(
    layer: PlaybackSyncLayerProjection,
    payload: PlaybackSyncPayload,
) -> AudibleLayerSpec | None:
    source_ref = _audio_source_ref(layer)
    if source_ref and layer.kind is not LayerKind.EVENT:
        return AudibleLayerSpec(
            layer_id=str(layer.layer_id),
            source_key=f"audio:{source_ref}",
            source_ref=source_ref,
            mode=PlaybackMode.CONTINUOUS_AUDIO,
            output_bus=sanitize_output_bus_for_channels(
                layer.output_bus,
                playback_output_channels=payload.playback_output_channels,
            ),
            gain_db=float(layer.gain_db),
            muted=bool(layer.muted),
            soloed=bool(layer.soloed),
        )
    if not _is_event_slice_layer(layer):
        return None
    events = tuple(
        AudibleEventSpec(
            start_seconds=float(event.start),
            end_seconds=float(event.end),
            muted=bool(event.muted) or "demoted" in event.badges,
        )
        for event in layer.events
    )
    playback_source_ref = _event_source_ref(layer)
    return AudibleLayerSpec(
        layer_id=str(layer.layer_id),
        source_key=f"event:{playback_source_ref}:{event_slice_signature(list(layer.events))}",
        source_ref=playback_source_ref,
        mode=PlaybackMode.EVENT_SLICE,
        output_bus=sanitize_output_bus_for_channels(
            layer.output_bus,
            playback_output_channels=payload.playback_output_channels,
        ),
        gain_db=float(layer.gain_db),
        muted=bool(layer.muted),
        soloed=bool(layer.soloed),
        events=events,
    )


def _muted_copy(layer: AudibleLayerSpec) -> AudibleLayerSpec:
    return AudibleLayerSpec(
        layer_id=layer.layer_id,
        source_key=layer.source_key,
        source_ref=layer.source_ref,
        mode=layer.mode,
        output_bus=layer.output_bus,
        gain_db=layer.gain_db,
        muted=True,
        soloed=layer.soloed,
        events=layer.events,
    )


def _is_event_slice_layer(layer: PlaybackSyncLayerProjection) -> bool:
    return bool(
        layer.kind is LayerKind.EVENT
        and bool(layer.playback_enabled)
        and layer.playback_mode == PlaybackMode.EVENT_SLICE
        and bool(_event_source_ref(layer))
    )


def _audio_source_ref(item: object) -> str | None:
    source_audio_path = getattr(item, "source_audio_path", None)
    if source_audio_path:
        return str(source_audio_path)
    source_content_ref = getattr(item, "source_content_ref", None)
    locator = getattr(source_content_ref, "locator", None)
    if locator:
        return str(locator)
    return None


def _event_source_ref(item: object) -> str:
    source_content_ref = getattr(item, "source_content_ref", None)
    locator = getattr(source_content_ref, "locator", None)
    if locator:
        return str(locator)
    playback_source_ref = getattr(item, "playback_source_ref", None)
    if playback_source_ref:
        return str(playback_source_ref)
    source_audio_path = getattr(item, "source_audio_path", None)
    if source_audio_path:
        return str(source_audio_path)
    return ""


__all__ = [
    "AudibleEventSpec",
    "AudibleIntent",
    "AudibleLayerSpec",
    "audible_intent_from_sync_payload",
]
