"""Transport service contract for EchoZero application runtime."""

from abc import ABC, abstractmethod

from echozero.application.shared.ranges import TimeRange
from echozero.application.transport.models import TransportState


class TransportService(ABC):
    """Owns movement through time: play, pause, stop, seek, and loop state."""

    @abstractmethod
    def get_state(self) -> TransportState:
        """Return the current transport state snapshot."""
        raise NotImplementedError

    @abstractmethod
    def play(self) -> TransportState:
        raise NotImplementedError

    @abstractmethod
    def pause(self) -> TransportState:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> TransportState:
        raise NotImplementedError

    @abstractmethod
    def seek(self, position: float) -> TransportState:
        raise NotImplementedError

    @abstractmethod
    def set_loop(self, loop_region: TimeRange | None, enabled: bool = True) -> TransportState:
        raise NotImplementedError
