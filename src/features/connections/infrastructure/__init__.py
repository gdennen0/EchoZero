"""
Infrastructure layer for connections feature.

Contains:
- SQLiteConnectionRepository - SQLite implementation of ConnectionRepository
"""
from src.features.connections.infrastructure.connection_repository_impl import SQLiteConnectionRepository

__all__ = [
    'SQLiteConnectionRepository',
]
