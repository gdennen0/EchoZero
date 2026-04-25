"""Review queue filtering for project-backed and imported review sessions.
Exists because review-entry surfaces should share one deterministic batch-selection rule.
Connects review-item collections to confidence-threshold and batch-limit workflows.
"""

from __future__ import annotations

from echozero.foundry.domain.review import ReviewItem


def select_review_items(
    items: list[ReviewItem],
    *,
    review_mode: str = "all_events",
    questionable_score_threshold: float | None = None,
    item_limit: int | None = None,
) -> list[ReviewItem]:
    """Return one deterministic review-item slice for the requested batch settings."""

    normalized_mode = str(review_mode).strip() or "all_events"
    if normalized_mode not in {"all_events", "questionables"}:
        raise ValueError("review_mode must be 'all_events' or 'questionables'")
    if questionable_score_threshold is not None and not 0.0 <= questionable_score_threshold <= 1.0:
        raise ValueError("questionable_score_threshold must be between 0.0 and 1.0")
    if item_limit is not None and item_limit <= 0:
        raise ValueError("item_limit must be a positive integer")

    selected = list(items)
    if normalized_mode == "questionables":
        if questionable_score_threshold is None:
            raise ValueError("questionable_score_threshold is required for questionables review mode")
        selected = _select_questionable_items(
            selected,
            questionable_score_threshold=questionable_score_threshold,
        )

    if item_limit is not None:
        selected = selected[:item_limit]
    return selected


def _select_questionable_items(
    items: list[ReviewItem],
    *,
    questionable_score_threshold: float,
) -> list[ReviewItem]:
    ranked_items: list[tuple[bool, float, int, ReviewItem]] = []
    for index, item in enumerate(items):
        if item.score is not None and item.score > questionable_score_threshold:
            continue
        ranked_items.append(
            (
                item.score is not None,
                float(item.score) if item.score is not None else -1.0,
                index,
                item,
            )
        )
    ranked_items.sort(key=lambda row: row[:3])
    return [item for *_ignored, item in ranked_items]


__all__ = ["select_review_items"]
