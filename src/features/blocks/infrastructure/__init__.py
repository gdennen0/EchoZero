"""
Infrastructure layer for blocks feature.

Contains:
- SQLiteBlockRepository - SQLite implementation of BlockRepository
"""
from src.features.blocks.infrastructure.block_repository_impl import SQLiteBlockRepository

__all__ = [
    'SQLiteBlockRepository',
]
