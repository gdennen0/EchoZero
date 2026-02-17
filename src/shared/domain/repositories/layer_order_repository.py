"""
Layer Order Repository Interface

Defines the contract for persisted layer order per block.
"""
from abc import ABC, abstractmethod
from typing import Optional

from src.shared.domain.entities.layer_order import LayerOrder


class LayerOrderRepository(ABC):
    """Repository interface for layer order persistence."""

    @abstractmethod
    def get_order(self, block_id: str) -> Optional[LayerOrder]:
        """Return stored LayerOrder for block, or None if missing."""
        raise NotImplementedError

    @abstractmethod
    def set_order(self, layer_order: LayerOrder) -> None:
        """Persist layer order for block (overwrite)."""
        raise NotImplementedError

    @abstractmethod
    def clear_order(self, block_id: str) -> None:
        """Clear layer order for block."""
        raise NotImplementedError
