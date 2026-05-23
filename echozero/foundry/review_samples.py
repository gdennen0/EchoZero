"""Canonical shared review-sample export contract.
Exists because training clips must be self-explanatory without reading UI code.
Connects review decisions, local clip layout, and Foundry shared-sample ingestion.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from echozero.foundry.domain.review import ReviewDecisionKind
from echozero.foundry.services.review_event_state import normalize_review_label


class ReviewSampleTrainingRole(StrEnum):
    """Declares how a reviewed clip should be consumed by model training."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


def review_sample_training_role(
    decision_kind: ReviewDecisionKind | str,
) -> ReviewSampleTrainingRole:
    """Resolve the training role for one human review decision."""

    normalized = str(decision_kind).strip().lower()
    if normalized == ReviewDecisionKind.REJECTED.value:
        return ReviewSampleTrainingRole.NEGATIVE
    return ReviewSampleTrainingRole.POSITIVE


def review_sample_label_dir(
    export_root: Path,
    *,
    class_label: str,
    training_role: ReviewSampleTrainingRole,
) -> Path:
    """Return the canonical folder for one review-sample class and role."""

    return export_root / training_role.value / _safe_segment(normalize_review_label(class_label))


def review_sample_target_label(
    *,
    class_label: str,
    training_role: ReviewSampleTrainingRole,
) -> str:
    """Return the effective label a generic training pool should learn."""

    if training_role is ReviewSampleTrainingRole.NEGATIVE:
        return "other"
    return normalize_review_label(class_label)


def legacy_review_sample_label_dir(export_root: Path, *, class_label: str) -> Path:
    """Return the pre-canonical class folder used by older shared exports."""

    return export_root / _safe_segment(normalize_review_label(class_label))


def _safe_segment(value: str) -> str:
    text = str(value).strip() or "event"
    safe = "".join(character if character.isalnum() else "_" for character in text)
    safe = safe.strip("_")
    return safe or "event"
