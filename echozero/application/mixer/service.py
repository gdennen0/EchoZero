"""Mixer service contract for audibility and balance resolution."""

from abc import ABC, abstractmethod

from echozero.application.shared.ids import LayerId
from echozero.application.mixer.models import MixerState, LayerMixerState, AudibilityState
from echozero.application.timeline.models import Layer


class MixerService(ABC):
    """Owns mute/solo/gain/pan changes and effective audibility resolution."""

    @abstractmethod
    def get_state(self) -> MixerState:
        """Return the current mixer state snapshot."""
        raise NotImplementedError

    @abstractmethod
    def set_layer_state(self, layer_id: LayerId, state: LayerMixerState) -> MixerState:
        raise NotImplementedError

    @abstractmethod
    def set_mute(self, layer_id: LayerId, muted: bool) -> MixerState:
        raise NotImplementedError

    @abstractmethod
    def set_solo(self, layer_id: LayerId, soloed: bool) -> MixerState:
        raise NotImplementedError

    @abstractmethod
    def set_gain(self, layer_id: LayerId, gain_db: float) -> MixerState:
        raise NotImplementedError

    @abstractmethod
    def set_pan(self, layer_id: LayerId, pan: float) -> MixerState:
        raise NotImplementedError

    @abstractmethod
    def resolve_audibility(self, layers: list[Layer]) -> list[AudibilityState]:
        raise NotImplementedError
