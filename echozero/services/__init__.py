"""
Services layer: Application-level orchestration between engine and persistence.
Exists because the engine computes and persistence stores — services bridge both.
"""

from echozero.services.analysis import AnalysisResult, AnalysisService
from echozero.services.setlist import SetlistProcessor, SetlistResult

__all__ = [
    "AnalysisService",
    "AnalysisResult",
    "SetlistProcessor",
    "SetlistResult",
]
