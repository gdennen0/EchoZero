"""
Shared status module.

Provides unified status publishing and subscription patterns:
- StatusPublisher: Base class for components that publish status updates
- Status: Standard status data structure
- StatusLevel: Enumeration of status levels
"""
from .status_publisher import (
    StatusPublisher,
    Status,
    StatusLevel,
    StatusSubscriber,
)

__all__ = [
    'StatusPublisher',
    'Status',
    'StatusLevel',
    'StatusSubscriber',
]
