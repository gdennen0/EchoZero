"""Model evolution review-truth records.
Exists to keep user-fixed Event timing distinct from model-ready audio windows.
Connects timeline review decisions to Foundry runtime sample materialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FixedEventTruth:
    """One user-confirmed Event that may seed future model training."""

    truth_id: str
    label: str
    source_audio_path: Path
    event_start_seconds: float
    event_end_seconds: float
    anchor_seconds: float | None = None
    project_ref: str | None = None
    song_id: str | None = None
    song_version_id: str | None = None
    layer_id: str | None = None
    event_id: str | None = None
    decision_kind: str | None = None
    review_outcome: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_label(self) -> str:
        """Return the canonical lower-case class label for training."""
        return self.label.strip().lower()

    @property
    def event_duration_ms(self) -> float:
        """Return the visual/semantic Event duration, not the model sample duration."""
        return max(0.0, float(self.event_end_seconds) - float(self.event_start_seconds)) * 1000.0

    @property
    def anchor(self) -> float:
        """Return the training anchor, defaulting to the Event start/onset."""
        if self.anchor_seconds is not None:
            return max(0.0, float(self.anchor_seconds))
        return max(0.0, float(self.event_start_seconds))

    def provenance(self) -> dict[str, Any]:
        """Serialize durable review-truth metadata for a dataset sample."""
        payload: dict[str, Any] = {
            "kind": "model_evolution_runtime_sample",
            "truth_id": self.truth_id,
            "label": self.normalized_label,
            "source_audio_path": str(self.source_audio_path.expanduser()),
            "event_start_seconds": float(self.event_start_seconds),
            "event_end_seconds": float(self.event_end_seconds),
            "event_duration_ms": self.event_duration_ms,
            "anchor_seconds": self.anchor,
        }
        optional = {
            "project_ref": self.project_ref,
            "song_id": self.song_id,
            "song_version_id": self.song_version_id,
            "layer_id": self.layer_id,
            "event_id": self.event_id,
            "decision_kind": self.decision_kind,
            "review_outcome": self.review_outcome,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

