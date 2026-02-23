from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class BlockSummary:
    """Minimal block metadata used for lazy listing."""
    id: str
    name: str
    type: str
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

