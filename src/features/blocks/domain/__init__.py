"""
Domain layer for blocks feature.

Contains:
- Block entity
- Port entity
- PortDirection value object
- BlockStatus value object
- BlockRepository interface
"""
from src.features.blocks.domain.block import Block
from src.features.blocks.domain.port import Port
from src.features.blocks.domain.port_direction import PortDirection
from src.features.blocks.domain.block_status import BlockStatus, BlockStatusLevel
from src.features.blocks.domain.block_repository import BlockRepository

__all__ = [
    'Block',
    'Port',
    'PortDirection',
    'BlockStatus',
    'BlockStatusLevel',
    'BlockRepository',
]
