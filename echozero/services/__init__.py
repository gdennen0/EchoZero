"""
Services layer: Application-level orchestration between engine and persistence.
Exists because the engine computes and persistence stores — services bridge both.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from echozero.services.orchestrator import AnalysisResult, Orchestrator
from echozero.services.setlist import SetlistProcessor, SetlistResult

if TYPE_CHECKING:
    from echozero.services.foundry_orchestrator import FoundryOrchestrator

# Legacy import shim for external consumers; internal code should use Orchestrator.
AnalysisService = Orchestrator

__all__ = [
    "Orchestrator",
    "AnalysisService",
    "AnalysisResult",
    "SetlistProcessor",
    "SetlistResult",
    "FoundryOrchestrator",
]


def __getattr__(name: str):
    if name == "FoundryOrchestrator":
        return import_module("echozero.services.foundry_orchestrator").FoundryOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
