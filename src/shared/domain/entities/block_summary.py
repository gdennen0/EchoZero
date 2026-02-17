from dataclasses import dataclass


@dataclass
class BlockSummary:
    """Minimal block metadata used for lazy listing."""
    id: str
    name: str
    type: str

