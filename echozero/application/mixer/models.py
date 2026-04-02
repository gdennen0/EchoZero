"""Mixer application models."""

from dataclasses import dataclass, field

from echozero.application.shared.ids import LayerId


@dataclass(slots=True)
class LayerMixerState:
    mute: bool = False
    solo: bool = False
    gain_db: float = 0.0
    pan: float = 0.0
    output_bus: str | None = None


@dataclass(slots=True)
class AudibilityState:
    layer_id: LayerId
    is_audible: bool
    reason: str


@dataclass(slots=True)
class MixerState:
    master_gain_db: float = 0.0
    layer_states: dict[LayerId, LayerMixerState] = field(default_factory=dict)
