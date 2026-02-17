from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class DataItemSummary:
    """Minimal metadata for a data item used in lazy listings."""
    id: str
    block_id: str
    name: str
    type: str
    created_at: datetime
    file_path: Optional[str]

