"""
Domain layer for connections feature.

Contains:
- Connection entity
- ConnectionSummary value object
- ConnectionRepository interface
"""
from src.features.connections.domain.connection import Connection
from src.features.connections.domain.connection_summary import ConnectionSummary
from src.features.connections.domain.connection_repository import ConnectionRepository

__all__ = [
    'Connection',
    'ConnectionSummary',
    'ConnectionRepository',
]
