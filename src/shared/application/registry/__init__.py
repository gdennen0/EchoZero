"""
Shared registry module.

Provides unified registration patterns for components:
- ModuleRegistry: Generic registry for any component type
- register_component: Decorator for component registration
"""
from .module_registry import (
    ModuleRegistry,
    ComponentMetadata,
    register_component,
    get_registry,
)

__all__ = [
    'ModuleRegistry',
    'ComponentMetadata',
    'register_component',
    'get_registry',
]
