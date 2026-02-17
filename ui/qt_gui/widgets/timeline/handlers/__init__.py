"""
Timeline Layer Handlers
=======================

Strategy pattern implementation for layer-type-specific behavior.

Each layer type (regular, sync) has its own handler that implements
standardized operations. This enables:
- Per-layer reload without affecting other layers
- Different behaviors for different layer types
- Isolated failure boundaries
- Easy addition of new layer types

Usage:
    from ui.qt_gui.widgets.timeline.handlers import (
        LayerHandler,
        RegularLayerHandler,
        SyncLayerHandler,
        get_handler_for_layer,
    )
"""

from .base import LayerHandler
from .regular import RegularLayerHandler
from .sync import SyncLayerHandler
from .registry import LayerHandlerRegistry, get_handler_for_layer

__all__ = [
    'LayerHandler',
    'RegularLayerHandler', 
    'SyncLayerHandler',
    'LayerHandlerRegistry',
    'get_handler_for_layer',
]
