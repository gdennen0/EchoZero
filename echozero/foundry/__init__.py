"""EZ2 Foundry package (standalone training application lane)."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import FoundryApp

__all__ = ["FoundryApp"]


def __getattr__(name: str):
    if name == "FoundryApp":
        return import_module("echozero.foundry.app").FoundryApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
