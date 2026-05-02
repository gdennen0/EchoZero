"""Snapshot service for frozen training slices of the sample library.
Exists to keep training inputs reproducible while the durable library keeps growing.
Connects approved library samples to candidate training runs.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from uuid import uuid4

from echozero.foundry.domain import LibrarySampleState, LibrarySnapshot, SampleLibraryRecord
from echozero.foundry.persistence import LibrarySnapshotRepository


class SnapshotService:
    """Create reproducible training snapshots from the local sample library."""

    def __init__(
        self,
        root: Path,
        *,
        repository: LibrarySnapshotRepository | None = None,
    ) -> None:
        self._root = root
        self._repository = repository or LibrarySnapshotRepository(root)

    def create_snapshot(
        self,
        *,
        name: str,
        samples: list[SampleLibraryRecord],
        provenance: dict[str, object] | None = None,
        filters: dict[str, object] | None = None,
    ) -> LibrarySnapshot:
        """Persist one snapshot from the approved library records passed in."""
        approved = [record for record in samples if record.state is LibrarySampleState.APPROVED]
        class_counts = Counter(record.label for record in approved)
        source_summary = Counter(record.source_type for record in approved)
        snapshot = LibrarySnapshot(
            id=f"snap_{uuid4().hex[:12]}",
            name=name,
            sample_ids=[record.id for record in approved],
            sample_count=len(approved),
            class_counts=dict(sorted(class_counts.items())),
            source_summary=dict(sorted(source_summary.items())),
            provenance=dict(provenance or {}),
            filters=dict(filters or {}),
        )
        return self._repository.save(snapshot)

    def get_snapshot(self, snapshot_id: str) -> LibrarySnapshot | None:
        """Return one snapshot when present."""
        return self._repository.get(snapshot_id)

    def list_snapshots(self) -> list[LibrarySnapshot]:
        """Return all persisted snapshots."""
        return self._repository.list()
