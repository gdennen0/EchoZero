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
class PlaybackState:
    status: PlaybackStatus = PlaybackStatus.STOPPED
    active_sources: list[PlaybackSource] = field(default_factory=list)
    latency_ms: float = 0.0
    backend_name: str = "unconfigured"
