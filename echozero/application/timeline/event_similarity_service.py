"""
Compatibility wrappers for the legacy event-similarity API.
Exists to keep the old shape-specific import path working while the canonical contract moves to generic event comparison.
Connects existing callers to the generic comparison service using the shape-envelope mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from echozero.application.timeline.event_comparison_service import (
    EventComparisonCandidateRecord,
    EventComparisonRequest,
    EventComparisonScoredCandidate,
    EventComparisonService,
)
from echozero.application.timeline.event_similarity_audio import ShapeNormalizationSettings

SimilarityCandidateRecord = EventComparisonCandidateRecord
SimilarityScoredCandidate = EventComparisonScoredCandidate


@dataclass(slots=True)
class SimilaritySelectionRequest(EventComparisonRequest):
    """Legacy shape-specific request wrapper."""

    normalization_settings: ShapeNormalizationSettings = field(
        default_factory=ShapeNormalizationSettings
    )

    def __post_init__(self) -> None:
        self.comparison_mode = "shape_envelope"
        self.comparison_settings = self.normalization_settings
        super().__post_init__()


class EventSimilarityService(EventComparisonService):
    """Compatibility alias for the legacy shape-comparison service."""

    def select_similar_event_refs(self, **kwargs):
        return self.select_matching_event_refs(**kwargs)
