"""
Infrastructure layer for MA3 (grandMA3) feature.

The MA3 feature primarily uses OSC for communication.
Infrastructure components include the OSC parser.
"""

from src.features.ma3.infrastructure.osc_parser import (
    OSCParser,
    get_osc_parser,
)

__all__ = [
    'OSCParser',
    'get_osc_parser',
]
