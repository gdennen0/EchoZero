"""Data records for review sample doctor runs.
Exists to keep the repair service focused on workflow instead of payload shapes.
Connects doctor audit actions, reports, and CLI JSON output through typed records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ReviewSampleDoctorResult:
    """Summarizes a doctor run over a shared review-sample export."""

    source_root: Path
    output_root: Path
    clean_root: Path
    quarantine_root: Path
    report_path: Path
    report: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable doctor result."""
        return {
            "source_root": str(self.source_root),
            "output_root": str(self.output_root),
            "clean_root": str(self.clean_root),
            "quarantine_root": str(self.quarantine_root),
            "report_path": str(self.report_path),
            "report": self.report,
        }


@dataclass(frozen=True, slots=True)
class DoctorSample:
    """One discovered audio clip plus resolved doctor metadata."""

    source_path: Path
    source_relative_path: str
    folder_label: str
    target_label: str
    content_hash: str
    duration_seconds: float | None
    frames: int | None
    sample_rate: int | None
    manifest_row: dict[str, Any] | None
    decision_kind: str | None
    review_polarity: str | None
    quality_flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DoctorAction:
    """One file copy decision made by the review sample doctor."""

    sample: DoctorSample
    action: str
    reason: str
    output_path: Path | None
