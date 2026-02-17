"""Block Local State Repository Interface

Defines the contract for persisted per-block local input state.

Local state stores references only:
  { input_port: data_item_id | [data_item_id, ...] }
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


LocalInputs = Dict[str, Any]


class BlockLocalStateRepository(ABC):
    """Repository interface for per-block local input references."""

    @abstractmethod
    def get_inputs(self, block_id: str) -> Optional[LocalInputs]:
        """Return stored local inputs mapping, or None if missing."""
        raise NotImplementedError

    @abstractmethod
    def set_inputs(self, block_id: str, inputs: LocalInputs) -> None:
        """Persist local inputs mapping for block (overwrite)."""
        raise NotImplementedError

    @abstractmethod
    def clear_inputs(self, block_id: str) -> None:
        """Clear local inputs for block."""
        raise NotImplementedError
