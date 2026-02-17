"""
Execution feature module.

Usage:
    from src.features.execution.application import BlockExecutionEngine, topological_sort_blocks
"""
# Application-only feature, no domain layer
from src.features.execution.application import (
    BlockExecutionEngine,
    topological_sort_blocks,
    ProgressTracker,
)

__all__ = [
    'BlockExecutionEngine',
    'topological_sort_blocks',
    'ProgressTracker',
]
