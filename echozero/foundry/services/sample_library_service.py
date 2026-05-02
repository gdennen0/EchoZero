"""Sample library service for local-first training accumulation.
Exists to keep durable sample curation separate from one-off dataset or run state.
Connects reviewed examples and imported samples to library-first training workflows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from echozero.foundry.domain import (
    DatasetSample,
    DatasetVersion,
    LibrarySampleState,
    SampleLibraryRecord,
)
from echozero.foundry.persistence import SampleLibraryRepository


class SampleLibraryService:
    """Persist and summarize reusable local training samples."""

    def __init__(
        self,
        root: Path,
        *,
        repository: SampleLibraryRepository | None = None,
    ) -> None:
        self._root = root
        self._repository = repository or SampleLibraryRepository(root)

    def record_dataset_version(
        self,
        version: DatasetVersion,
        *,
        state: LibrarySampleState = LibrarySampleState.APPROVED,
    ) -> list[SampleLibraryRecord]:
        """Copy one dataset version into the durable sample library."""
        records = [
            self._save_or_replace_record(
                SampleLibraryRecord(
                    id=f"lib_{uuid4().hex[:12]}",
                    audio_ref=sample.audio_ref,
                    label=sample.label,
                    state=state,
                    source_type=str(version.lineage.get("kind", version.manifest.get("schema", "dataset_version"))),
                    source_ref=version.id,
                    provenance={
                        "dataset_id": version.dataset_id,
                        "dataset_version_id": version.id,
                        "sample_id": sample.sample_id,
                        "source_provenance": dict(sample.source_provenance),
                    },
                    quality_flags=list(sample.quality_flags),
                    tags=self._build_tags(sample),
                    content_hash=sample.content_hash,
                    duration_ms=sample.duration_ms,
                    review_count=1,
                    reviewed_at=datetime.now(UTC),
                )
            )
            for sample in version.samples
        ]
        return records

    def list_samples(self) -> list[SampleLibraryRecord]:
        """Return every library sample."""
        return self._repository.list()

    def list_samples_by_state(self, state: LibrarySampleState) -> list[SampleLibraryRecord]:
        """Return library samples for one review bucket."""
        return [record for record in self._repository.list() if record.state is state]

    def summarize(self) -> dict[str, object]:
        """Return a compact operator-facing summary of library state."""
        records = self._repository.list()
        state_counts = Counter(record.state.value for record in records)
        class_counts = Counter(record.label for record in records if record.state is LibrarySampleState.APPROVED)
        source_counts = Counter(record.source_type for record in records)
        return {
            "sample_count": len(records),
            "approved_count": state_counts.get(LibrarySampleState.APPROVED.value, 0),
            "state_counts": dict(sorted(state_counts.items())),
            "approved_class_counts": dict(sorted(class_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
        }

    def _save_or_replace_record(self, candidate: SampleLibraryRecord) -> SampleLibraryRecord:
        existing = self._find_existing(candidate)
        if existing is None:
            return self._repository.save(candidate)
        merged = replace(
            existing,
            label=candidate.label,
            state=candidate.state,
            source_type=candidate.source_type,
            source_ref=candidate.source_ref,
            provenance=candidate.provenance,
            quality_flags=candidate.quality_flags,
            tags=candidate.tags,
            duration_ms=candidate.duration_ms,
            review_count=max(existing.review_count, candidate.review_count),
            reviewed_at=candidate.reviewed_at or existing.reviewed_at,
        )
        return self._repository.save(merged)

    def _find_existing(self, candidate: SampleLibraryRecord) -> SampleLibraryRecord | None:
        for record in self._repository.list():
            if record.content_hash and candidate.content_hash and record.content_hash != candidate.content_hash:
                continue
            if record.audio_ref != candidate.audio_ref:
                continue
            if record.label != candidate.label:
                continue
            return record
        return None

    @staticmethod
    def _build_tags(sample: DatasetSample) -> list[str]:
        tags = [sample.label]
        if sample.is_synthetic:
            tags.append("synthetic")
        if sample.group_id:
            tags.append(sample.group_id)
        return tags
