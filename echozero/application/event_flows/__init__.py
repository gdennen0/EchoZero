"""
Event-flow contracts for compact app-facing extraction controls.
Exists because user-facing flows should compile to pipeline knobs without leaking raw DSP settings.
Connects timeline action settings to pipeline templates while the engine remains pipeline-data only.
"""

from .drum_event_extraction_v2 import (
    DrumEventExtractionRequest,
    DrumEventExtractionResult,
    DrumEventLabelLane,
    DrumEventLayerTakeDraft,
    build_drum_event_extraction_result,
    build_drum_event_layer_take_drafts,
)
from .drum_events import (
    DRUM_EVENT_LABELS,
    DRUM_EVENT_SENSITIVITY_OPTIONS,
    CompactDrumEventSettings,
    DrumEventModelReadiness,
    apply_drum_event_sensitivity_preset,
    compile_drum_event_sensitivity_knobs,
    drum_event_type_options,
    model_readiness_from_fields,
    normalize_drum_event_labels,
)

__all__ = [
    "DRUM_EVENT_LABELS",
    "DRUM_EVENT_SENSITIVITY_OPTIONS",
    "CompactDrumEventSettings",
    "DrumEventExtractionRequest",
    "DrumEventExtractionResult",
    "DrumEventLabelLane",
    "DrumEventLayerTakeDraft",
    "DrumEventModelReadiness",
    "apply_drum_event_sensitivity_preset",
    "build_drum_event_extraction_result",
    "build_drum_event_layer_take_drafts",
    "compile_drum_event_sensitivity_knobs",
    "drum_event_type_options",
    "model_readiness_from_fields",
    "normalize_drum_event_labels",
]
