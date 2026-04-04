from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class TrainRunStatus(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    EVALUATING = "evaluating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class CurationState(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Dataset:
    id: str
    name: str
    source_kind: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class DatasetSample:
    sample_id: str
    audio_ref: str
    label: str
    duration_ms: float | None = None
    source_provenance: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)
    curation_state: CurationState = CurationState.UNKNOWN


@dataclass(slots=True)
class DatasetVersion:
    id: str
    dataset_id: str
    version: int
    manifest_hash: str
    sample_rate: int
    audio_standard: str
    class_map: list[str]
    samples: list[DatasetSample] = field(default_factory=list)
    split_plan: dict[str, Any] = field(default_factory=dict)
    balance_plan: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class EvalReport:
    id: str
    run_id: str
    classification_mode: str
    metrics: dict[str, Any]
    threshold_policy: dict[str, Any] | None = None
    confusion: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class TrainRun:
    id: str
    dataset_version_id: str
    status: TrainRunStatus
    spec: dict[str, Any]
    spec_hash: str
    backend: str = "pytorch"
    device: str = "cpu"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def run_dir(self, base_dir: Path) -> Path:
        return base_dir / "foundry" / "runs" / self.id


@dataclass(slots=True)
class ModelArtifact:
    id: str
    run_id: str
    artifact_version: str
    path: Path
    sha256: str
    manifest: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class CompatibilityReport:
    artifact_id: str
    consumer: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))
