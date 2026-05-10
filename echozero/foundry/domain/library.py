"""Foundry sample-library records for durable local training inputs.
Exists to keep reusable reviewed samples separate from one-off dataset and run state.
Connects local sample accumulation to the thin dataset-version kickoff path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class LibrarySampleState(StrEnum):
    """Review bucket for one library sample."""

    APPROVED = "approved"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


@dataclass(slots=True)
class SampleLibraryRecord:
    """One durable training example captured into the local library."""

    id: str
    audio_ref: str
    label: str
    state: LibrarySampleState
    source_type: str
    source_ref: str
    provenance: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    content_hash: str = ""
    duration_ms: float | None = None
    review_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    reviewed_at: datetime | None = None
