"""Review import helpers for folder-backed phone verification sessions.
Exists because review-session services should stay focused on persistence and filtering, not filesystem crawling.
Connects host audio folders to typed review items with deterministic ordering and skip reporting.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from echozero.foundry.domain import ReviewItem, ReviewPolarity
from echozero.foundry.services.audio_source_validation import InvalidAudioSourceError, inspect_audio_source

_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}


@dataclass(frozen=True, slots=True)
class FolderReviewImport:
    """Result of scanning one folder into review items plus skip diagnostics."""

    items: list[ReviewItem]
    skipped_sources: list[dict[str, str]]
    skipped_reason_counts: Counter[str]


def import_review_items_from_folder(
    base: Path,
    *,
    target_class: str | None,
    polarity: ReviewPolarity,
) -> FolderReviewImport:
    """Return review items imported from one host folder."""
    resolved_target_class = (target_class or "").strip() or None
    if resolved_target_class is None:
        class_dirs = sorted(path for path in base.iterdir() if path.is_dir())
        if not class_dirs:
            raise ValueError("Review folder import without --target-class requires class-named subdirectories.")
        file_candidates = [file for class_dir in class_dirs for file in sorted(class_dir.rglob("*"))]
    else:
        file_candidates = sorted(base.rglob("*"))

    items: list[ReviewItem] = []
    skipped_sources: list[dict[str, str]] = []
    skipped_reason_counts: Counter[str] = Counter()
    for file in file_candidates:
        if not file.is_file() or file.suffix.lower() not in _AUDIO_SUFFIXES:
            continue
        relative_path = file.relative_to(base).as_posix()
        predicted_label = resolved_target_class or relative_path.split("/", 1)[0]
        try:
            inspect_audio_source(file)
        except InvalidAudioSourceError as exc:
            skipped_reason_counts[exc.code] += 1
            skipped_sources.append(
                {
                    "path": str(file.resolve()),
                    "relative_path": relative_path,
                    "filename": file.name,
                    "predicted_label": predicted_label,
                    "reason_code": exc.code,
                    "reason": str(exc),
                }
            )
            continue
        items.append(
            ReviewItem(
                item_id=f"ri_{uuid4().hex[:12]}",
                audio_path=str(file.resolve()),
                predicted_label=predicted_label,
                target_class=predicted_label,
                polarity=polarity,
                score=None,
                source_provenance={
                    "kind": "review_folder_import",
                    "path": str(file.resolve()),
                    "source_root": str(base.resolve()),
                    "relative_path": relative_path,
                    "filename": file.name,
                    "predicted_label": predicted_label,
                },
            )
        )
    return FolderReviewImport(
        items=items,
        skipped_sources=skipped_sources,
        skipped_reason_counts=skipped_reason_counts,
    )
