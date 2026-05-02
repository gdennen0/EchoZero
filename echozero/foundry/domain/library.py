"""Library-first Foundry domain records for local training growth.
Exists to make sample accumulation, snapshotting, and promotion first-class concepts.
Connects durable sample curation to training candidates and future contribution seams.
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
    HARD_NEGATIVE = "hard_negative"
    EDGE_CASE = "edge_case"


class ContributionPolicy(StrEnum):
    """Sharing posture for one library record."""

    LOCAL_ONLY = "local_only"
    CONTRIBUTABLE = "contributable"
    EXCLUDED = "excluded"


class TrainingRecipeName(StrEnum):
    """Named training presets exposed to operators."""

    QUICK = "quick"
    BALANCED = "balanced"
    MONSTER = "monster"


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
    contribution_policy: ContributionPolicy = ContributionPolicy.LOCAL_ONLY
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    reviewed_at: datetime | None = None


@dataclass(slots=True)
class LibrarySnapshot:
    """Frozen slice of approved library samples for one training candidate."""

    id: str
    name: str
    sample_ids: list[str]
    sample_count: int
    class_counts: dict[str, int]
    source_summary: dict[str, int]
    filters: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class ModelCandidateRecord:
    """One challenger model produced from a library snapshot."""

    id: str
    snapshot_id: str
    recipe_name: str
    dataset_id: str
    dataset_version_id: str
    run_id: str
    artifact_id: str | None = None
    eval_report_id: str | None = None
    status: str = "queued"
    metrics: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class ChampionModelRecord:
    """Current promoted model for one local scope."""

    scope: str
    candidate_id: str
    artifact_id: str
    promoted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    previous_candidate_id: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContributionRecord:
    """Contribution-ready export metadata for one shareable sample."""

    id: str
    sample_id: str
    policy: ContributionPolicy
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
