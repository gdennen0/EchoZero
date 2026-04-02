"""Sync service contract for external timing/control relationships."""

from abc import ABC, abstractmethod

from echozero.application.shared.enums import SyncMode
from echozero.application.sync.models import SyncState
from echozero.application.transport.models import TransportState


class SyncService(ABC):
    """Owns sync mode, connection state, and external/internal timing alignment."""

    @abstractmethod
    def get_state(self) -> SyncState:
        """Return the current sync state snapshot."""
        raise NotImplementedError

    @abstractmethod
    def set_mode(self, mode: SyncMode) -> SyncState:
        raise NotImplementedError

    @abstractmethod
    def connect(self) -> SyncState:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> SyncState:
        raise NotImplementedError

    @abstractmethod
    def align_transport(self, transport: TransportState) -> TransportState:
        """Return a transport state aligned to current sync constraints if needed."""
        raise NotImplementedError
