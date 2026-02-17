"""
SQLite persistence implementation

Core database utilities. Repository implementations have moved to feature modules:
- src.features.projects.infrastructure.SQLiteProjectRepository
- src.features.blocks.infrastructure.SQLiteBlockRepository
- src.features.connections.infrastructure.SQLiteConnectionRepository
"""
from src.infrastructure.persistence.sqlite.database import Database

__all__ = [
    'Database',
]


# Backwards compatibility - lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'SQLiteProjectRepository':
        from src.features.projects.infrastructure import SQLiteProjectRepository
        return SQLiteProjectRepository
    if name == 'SQLiteBlockRepository':
        from src.features.blocks.infrastructure import SQLiteBlockRepository
        return SQLiteBlockRepository
    if name == 'SQLiteConnectionRepository':
        from src.features.connections.infrastructure import SQLiteConnectionRepository
        return SQLiteConnectionRepository
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
