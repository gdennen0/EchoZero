"""
Application API Layer

Provides a unified facade for all application operations.
Used by CLI, GUI (future), and external APIs.
"""
from .result_types import CommandResult, ResultStatus
from .application_facade import ApplicationFacade

__all__ = [
    "CommandResult",
    "ResultStatus",
    "ApplicationFacade",
]



