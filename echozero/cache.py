"""
ExecutionCache: Stores evaluation outputs keyed by (block_id, port_name).
Exists because the coordinator needs to track which blocks have fresh outputs and pass data between blocks.
Used by the Coordinator to cache block outputs and invalidate downstream when inputs change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from echozero.domain.graph import Graph


@dataclass(frozen=True)
class CachedOutput:
    """A single cached output value from a block execution."""

    block_id: str
    port_name: str
    value: Any
    produced_at: float
    execution_id: str


class ExecutionCache:
    """Stores block outputs keyed by (block_id, port_name) with invalidation support."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], CachedOutput] = {}

    def store(
        self,
        block_id: str,
        port_name: str,
        value: Any,
        execution_id: str,
    ) -> None:
        """Cache an output value for a block's port."""
        self._entries[(block_id, port_name)] = CachedOutput(
            block_id=block_id,
            port_name=port_name,
            value=value,
            produced_at=time.time(),
            execution_id=execution_id,
        )

    def get(self, block_id: str, port_name: str) -> CachedOutput | None:
        """Retrieve a cached output, or None if not present."""
        return self._entries.get((block_id, port_name))

    def has_valid_output(self, block_id: str, port_name: str) -> bool:
        """Check if a valid cached output exists for a block's port."""
        return (block_id, port_name) in self._entries

    def get_all(self, block_id: str) -> dict[str, CachedOutput]:
        """Return all cached outputs for a block, keyed by port name."""
        return {
            port_name: entry
            for (bid, port_name), entry in self._entries.items()
            if bid == block_id
        }

    def invalidate(self, block_id: str) -> None:
        """Remove all cached outputs for a block."""
        keys_to_remove = [
            key for key in self._entries if key[0] == block_id
        ]
        for key in keys_to_remove:
            del self._entries[key]

    def invalidate_downstream(self, block_id: str, graph: Graph) -> set[str]:
        """Invalidate a block and all its downstream dependents. Returns affected block IDs."""
        affected = graph.downstream_of(block_id)
        affected.add(block_id)
        for bid in affected:
            self.invalidate(bid)
        return affected

    def clear(self) -> None:
        """Remove all cached entries."""
        self._entries.clear()
