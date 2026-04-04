"""
Services layer: Application-level orchestration between engine and persistence.
Exists because the engine computes and persistence stores — services bridge both.
"""

from echozero.services.foundry_orchestrator import FoundryOrchestrator
from echozero.services.orchestrator import AnalysisResult, Orchestrator
from echozero.services.setlist import SetlistProcessor, SetlistResult

# Backward-compat alias
AnalysisService = Orchestrator

__all__ = [
    "Orchestrator",
    "AnalysisService",  # backward compat
    "AnalysisResult",
    "SetlistProcessor",
    "SetlistResult",
    "FoundryOrchestrator",
]
