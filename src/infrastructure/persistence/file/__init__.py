"""
File-based persistence implementations.

Note: ActionSetFileRepository has moved to src.features.projects.infrastructure
"""
# Backwards compatibility - lazy import
def __getattr__(name):
    if name in ('ActionSetFileRepository', 'get_action_set_file_repo'):
        from src.features.projects.infrastructure.action_set_file_repository import ActionSetFileRepository, get_action_set_file_repo
        if name == 'ActionSetFileRepository':
            return ActionSetFileRepository
        return get_action_set_file_repo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['ActionSetFileRepository', 'get_action_set_file_repo']
