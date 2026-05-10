"""
Playback sync projection: compact transport payload for runtime-audio sync.
Exists because process-isolated playback should not ship full UI presentation blobs across IPC.
Connects timeline presentation state to a minimal playback-only projection used by runtime sync.
"""

from __future__ import annotations

from dataclasses import dataclass

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import LayerKind, PlaybackMode


@dataclass(slots=True, frozen=True)
class PlaybackSyncEventProjection:
    """Compact event projection needed for event-slice signature and render."""

    start: float
    muted: bool
    badges: tuple[str, ...]

    @classmethod
    def from_event(cls, event: EventPresentation) -> "PlaybackSyncEventProjection":
        return cls(
            start=float(getattr(event, "start", 0.0)),
            muted=bool(getattr(event, "muted", False)),
            badges=tuple(str(badge) for badge in getattr(event, "badges", ()) or ()),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlaybackSyncEventProjection":
        return cls(
            start=float(payload.get("start", 0.0) or 0.0),
            muted=bool(payload.get("muted", False)),
            badges=tuple(str(item) for item in payload.get("badges", ()) or ()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "start": float(self.start),
            "muted": bool(self.muted),
            "badges": list(self.badges),
        }


@dataclass(slots=True, frozen=True)
class PlaybackSyncTakeProjection:
    """Compact take projection containing only playback-relevant fields."""

    take_id: str
    name: str
    source_audio_path: str | None
    playback_source_ref: str | None
    events: tuple[PlaybackSyncEventProjection, ...]

    @classmethod
    def from_take(cls, take: TakeLanePresentation) -> "PlaybackSyncTakeProjection":
        return cls(
            take_id=str(take.take_id),
            name=str(take.name),
            source_audio_path=(
                str(take.source_audio_path) if getattr(take, "source_audio_path", None) else None
            ),
            playback_source_ref=(
                str(take.playback_source_ref)
                if getattr(take, "playback_source_ref", None)
                else None
            ),
            events=tuple(PlaybackSyncEventProjection.from_event(event) for event in take.events),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlaybackSyncTakeProjection":
        events_payload = payload.get("events", ()) or ()
        return cls(
            take_id=str(payload.get("take_id", "") or ""),
            name=str(payload.get("name", "") or ""),
            source_audio_path=(
                str(payload.get("source_audio_path")) if payload.get("source_audio_path") else None
            ),
            playback_source_ref=(
                str(payload.get("playback_source_ref"))
                if payload.get("playback_source_ref")
                else None
            ),
            events=tuple(
                PlaybackSyncEventProjection.from_dict(item)
                for item in events_payload
                if isinstance(item, dict)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "take_id": self.take_id,
            "name": self.name,
            "source_audio_path": self.source_audio_path,
            "playback_source_ref": self.playback_source_ref,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(slots=True, frozen=True)
class PlaybackSyncLayerProjection:
    """Compact layer projection used by playback runtime sync."""

    layer_id: str
    title: str
    kind: LayerKind
    main_take_id: str | None
    muted: bool
    soloed: bool
    gain_db: float
    output_bus: str | None
    source_audio_path: str | None
    playback_enabled: bool
    playback_mode: PlaybackMode
    playback_source_ref: str | None
    events: tuple[PlaybackSyncEventProjection, ...]
    takes: tuple[PlaybackSyncTakeProjection, ...]

    @classmethod
    def from_layer(cls, layer: LayerPresentation) -> "PlaybackSyncLayerProjection":
        return cls(
            layer_id=str(layer.layer_id),
            title=str(layer.title),
            kind=_coerce_layer_kind(getattr(layer, "kind", LayerKind.EVENT)),
            main_take_id=str(layer.main_take_id) if layer.main_take_id is not None else None,
            muted=bool(layer.muted),
            soloed=bool(layer.soloed),
            gain_db=float(layer.gain_db),
            output_bus=str(layer.output_bus) if layer.output_bus else None,
            source_audio_path=str(layer.source_audio_path) if layer.source_audio_path else None,
            playback_enabled=bool(layer.playback_enabled),
            playback_mode=_coerce_playback_mode(
                getattr(layer, "playback_mode", PlaybackMode.NONE)
            ),
            playback_source_ref=(
                str(layer.playback_source_ref) if layer.playback_source_ref else None
            ),
            events=tuple(PlaybackSyncEventProjection.from_event(event) for event in layer.events),
            takes=tuple(PlaybackSyncTakeProjection.from_take(take) for take in layer.takes),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlaybackSyncLayerProjection":
        events_payload = payload.get("events", ()) or ()
        takes_payload = payload.get("takes", ()) or ()
        return cls(
            layer_id=str(payload.get("layer_id", "") or ""),
            title=str(payload.get("title", "") or ""),
            kind=_coerce_layer_kind(payload.get("kind", LayerKind.EVENT.value)),
            main_take_id=str(payload.get("main_take_id")) if payload.get("main_take_id") else None,
            muted=bool(payload.get("muted", False)),
            soloed=bool(payload.get("soloed", False)),
            gain_db=float(payload.get("gain_db", 0.0) or 0.0),
            output_bus=str(payload.get("output_bus")) if payload.get("output_bus") else None,
            source_audio_path=(
                str(payload.get("source_audio_path")) if payload.get("source_audio_path") else None
            ),
            playback_enabled=bool(payload.get("playback_enabled", False)),
            playback_mode=_coerce_playback_mode(
                payload.get("playback_mode", PlaybackMode.NONE.value)
            ),
            playback_source_ref=(
                str(payload.get("playback_source_ref"))
                if payload.get("playback_source_ref")
                else None
            ),
            events=tuple(
                PlaybackSyncEventProjection.from_dict(item)
                for item in events_payload
                if isinstance(item, dict)
            ),
            takes=tuple(
                PlaybackSyncTakeProjection.from_dict(item)
                for item in takes_payload
                if isinstance(item, dict)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "title": self.title,
            "kind": self.kind.value,
            "main_take_id": self.main_take_id,
            "muted": self.muted,
            "soloed": self.soloed,
            "gain_db": self.gain_db,
            "output_bus": self.output_bus,
            "source_audio_path": self.source_audio_path,
            "playback_enabled": self.playback_enabled,
            "playback_mode": self.playback_mode.value,
            "playback_source_ref": self.playback_source_ref,
            "events": [event.to_dict() for event in self.events],
            "takes": [take.to_dict() for take in self.takes],
        }

    def to_runtime_layer(self) -> "RuntimeLayerProjection":
        return RuntimeLayerProjection(
            layer_id=self.layer_id,
            title=self.title,
            kind=self.kind,
            main_take_id=self.main_take_id,
            muted=self.muted,
            soloed=self.soloed,
            gain_db=self.gain_db,
            output_bus=self.output_bus,
            source_audio_path=self.source_audio_path,
            playback_enabled=self.playback_enabled,
            playback_mode=self.playback_mode,
            playback_source_ref=self.playback_source_ref,
            events=_to_runtime_events(self.events),
            takes=_to_runtime_takes(self.takes),
        )


@dataclass(slots=True, frozen=True)
class PlaybackSyncPayload:
    """Compact playback-only sync payload for process-isolated runtime IPC."""

    layers: tuple[PlaybackSyncLayerProjection, ...]
    selected_layer_id: str | None
    selected_take_id: str | None
    playback_output_channels: int

    @classmethod
    def from_presentation(cls, presentation: TimelinePresentation) -> "PlaybackSyncPayload":
        return cls(
            layers=tuple(
                PlaybackSyncLayerProjection.from_layer(layer) for layer in presentation.layers
            ),
            selected_layer_id=(
                str(presentation.selected_layer_id)
                if presentation.selected_layer_id is not None
                else None
            ),
            selected_take_id=(
                str(presentation.selected_take_id)
                if presentation.selected_take_id is not None
                else None
            ),
            playback_output_channels=max(
                0, int(getattr(presentation, "playback_output_channels", 0) or 0)
            ),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlaybackSyncPayload":
        layers_payload = payload.get("layers", ()) or ()
        return cls(
            layers=tuple(
                PlaybackSyncLayerProjection.from_dict(item)
                for item in layers_payload
                if isinstance(item, dict)
            ),
            selected_layer_id=(
                str(payload.get("selected_layer_id")) if payload.get("selected_layer_id") else None
            ),
            selected_take_id=(
                str(payload.get("selected_take_id")) if payload.get("selected_take_id") else None
            ),
            playback_output_channels=max(0, int(payload.get("playback_output_channels", 0) or 0)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "selected_layer_id": self.selected_layer_id,
            "selected_take_id": self.selected_take_id,
            "playback_output_channels": int(self.playback_output_channels),
        }

    def to_runtime_projection(self) -> "RuntimeSyncProjection":
        return RuntimeSyncProjection(
            layers=[layer.to_runtime_layer() for layer in self.layers],
            selected_layer_id=self.selected_layer_id,
            selected_take_id=self.selected_take_id,
            playback_output_channels=max(0, int(self.playback_output_channels)),
        )


@dataclass(slots=True)
class RuntimeSyncProjection:
    """Runtime-facing playback projection consumed by PlaybackTrackBuilder."""

    layers: list[RuntimeLayerProjection]
    selected_layer_id: str | None
    selected_take_id: str | None
    playback_output_channels: int


@dataclass(slots=True)
class RuntimeLayerProjection:
    """Runtime playback layer projection with builder-compatible field names."""

    layer_id: str
    title: str
    kind: LayerKind
    main_take_id: str | None
    muted: bool
    soloed: bool
    gain_db: float
    output_bus: str | None
    source_audio_path: str | None
    playback_enabled: bool
    playback_mode: PlaybackMode
    playback_source_ref: str | None
    events: list[RuntimeEventProjection]
    takes: list[RuntimeTakeProjection]


@dataclass(slots=True)
class RuntimeTakeProjection:
    """Runtime playback take projection with builder-compatible field names."""

    take_id: str
    name: str
    source_audio_path: str | None
    playback_source_ref: str | None
    events: list[RuntimeEventProjection]


@dataclass(slots=True)
class RuntimeEventProjection:
    """Runtime playback event projection with builder-compatible field names."""

    start: float
    muted: bool
    badges: list[str]


def _coerce_layer_kind(value: object) -> LayerKind:
    if isinstance(value, LayerKind):
        return value
    try:
        return LayerKind(str(value or "").strip().lower())
    except ValueError:
        return LayerKind.EVENT


def _coerce_playback_mode(value: object) -> PlaybackMode:
    if isinstance(value, PlaybackMode):
        return value
    try:
        return PlaybackMode(str(value or "").strip().lower())
    except ValueError:
        return PlaybackMode.NONE


def _to_runtime_events(
    events: tuple[PlaybackSyncEventProjection, ...],
) -> list[RuntimeEventProjection]:
    return [
        RuntimeEventProjection(
            start=float(event.start),
            muted=bool(event.muted),
            badges=list(event.badges),
        )
        for event in events
    ]


def _to_runtime_takes(
    takes: tuple[PlaybackSyncTakeProjection, ...],
) -> list[RuntimeTakeProjection]:
    return [
        RuntimeTakeProjection(
            take_id=take.take_id,
            name=take.name,
            source_audio_path=take.source_audio_path,
            playback_source_ref=take.playback_source_ref,
            events=_to_runtime_events(take.events),
        )
        for take in takes
    ]


__all__ = [
    "PlaybackSyncEventProjection",
    "PlaybackSyncLayerProjection",
    "PlaybackSyncPayload",
    "PlaybackSyncTakeProjection",
    "RuntimeEventProjection",
    "RuntimeLayerProjection",
    "RuntimeSyncProjection",
    "RuntimeTakeProjection",
]
