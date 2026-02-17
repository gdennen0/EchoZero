"""
Connections feature module.

Usage:
    from src.features.connections.domain import Connection, ConnectionSummary
    from src.features.connections.application import ConnectionService
    from src.features.connections.infrastructure import SQLiteConnectionRepository
"""
# Only export domain by default - application and infrastructure via submodules
from src.features.connections.domain import (
    Connection,
    ConnectionSummary,
    ConnectionRepository,
)

__all__ = [
    'Connection',
    'ConnectionSummary',
    'ConnectionRepository',
]
