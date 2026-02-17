from dataclasses import dataclass


@dataclass
class ConnectionSummary:
    """Minimal connection metadata used for lazy listing."""
    id: str
    source_block_id: str
    source_output_name: str
    target_block_id: str
    target_input_name: str

