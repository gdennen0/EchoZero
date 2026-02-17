"""
Layer order entities.

Represents the persisted ordering of timeline layers for a block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class LayerKey:
    """Unique layer identifier for ordering purposes."""
    group_name: Optional[str]
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_name": self.group_name,
            "name": self.name,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LayerKey":
        return LayerKey(
            group_name=data.get("group_name") or data.get("group_id"),
            name=data.get("name") or "",
        )


@dataclass
class LayerOrder:
    """Persisted layer order for a block."""
    block_id: str
    order: List[LayerKey]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "order": [key.to_dict() for key in self.order],
        }
