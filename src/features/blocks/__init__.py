"""
Blocks feature module.

Usage:
    from src.features.blocks.domain import Block, Port, PortDirection, BlockStatus
    from src.features.blocks.application import BlockService, BlockStatusService
    from src.features.blocks.infrastructure import SQLiteBlockRepository
"""
# Only export domain by default - application and infrastructure via submodules
from src.features.blocks.domain import (
    Block,
    Port,
    PortDirection,
    BlockStatus,
    BlockRepository,
)

__all__ = [
    'Block',
    'Port',
    'PortDirection',
    'BlockStatus',
    'BlockRepository',
]
