"""Playback service contract for runtime source execution coordination."""

from abc import ABC, abstractmethod

from echozero.application.playback.models import PlaybackState
from echozero.application.timeline.models import Timeline
from echozero.application.transport.models import TransportState
from echozero.application.mixer.models import AudibilityState
from echozero.application.sync.models import SyncState


class PlaybackService(ABC):
    """Owns runtime playback source preparation and backend-facing coordination."""

    @abstractmethod
    def get_state(self) -> PlaybackState:
        """Return the current playback state snapshot."""
        raise NotImplementedError

    @abstractmethod
    def prepare(self, timeline: Timeline) -> PlaybackState:
        """Prepare playback sources for the current timeline state."""
        raise NotImplementedError

    @abstractmethod
    def update_runtime(
        self,
        timeline: Timeline,
        transport: TransportState,
        audibility: list[AudibilityState],
        sync: SyncState,
    ) -> PlaybackState:
        """Update runtime playback according to current app state."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> PlaybackState:
        """Stop runtime playback output."""
        raise NotImplementedError
