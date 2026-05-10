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

    def create_static_preset(
        self,
        *,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        value_command: str,
    ) -> dict[str, object]:
        """Create one MA3 static preset when the active sync provider supports authoring."""
        raise NotImplementedError

    def create_phaser_preset(
        self,
        *,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        step_preset_refs: list[list[str]] | list[str] | tuple[list[str], ...] | tuple[str, ...],
        speed_bpm: float | None = None,
    ) -> dict[str, object]:
        """Create one MA3 phaser preset from explicit step references."""
        raise NotImplementedError

    def create_recipe_preset(
        self,
        *,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        source_preset_ref: str,
        selection_mode: str = "Strict",
    ) -> dict[str, object]:
        """Create one MA3 recipe preset from an explicit source preset reference."""
        raise NotImplementedError

    def edit_static_preset(
        self,
        *,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        value_command: str,
    ) -> dict[str, object]:
        """Replace one MA3 static preset when the active sync provider supports authoring."""
        raise NotImplementedError

    def edit_phaser_preset(
        self,
        *,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        step_preset_refs: list[list[str]] | list[str] | tuple[list[str], ...] | tuple[str, ...],
        speed_bpm: float | None = None,
    ) -> dict[str, object]:
        """Replace one MA3 phaser preset from explicit step references."""
        raise NotImplementedError

    def edit_recipe_preset(
        self,
        *,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        source_preset_ref: str,
        selection_mode: str = "Strict",
    ) -> dict[str, object]:
        """Replace one MA3 recipe preset from an explicit source preset reference."""
        raise NotImplementedError
