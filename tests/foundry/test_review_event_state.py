"""Review event-state tests for detection-first promotion fallback."""

from __future__ import annotations

from echozero.foundry.domain.review import ReviewOutcome
from echozero.foundry.services.review_event_state import canonical_review_state


def test_canonical_review_state_uses_detection_promotion_state_when_review_is_missing() -> None:
    state = canonical_review_state(
        origin="binary_drum_classify:kick",
        metadata={
            "detection": {
                "promotion_state": "demoted",
                "threshold_passed": True,
            }
        },
    )

    assert state.promotion_state == "demoted"
    assert state.review_state == "unreviewed"
    assert state.review_outcome == ReviewOutcome.PENDING


def test_canonical_review_state_derives_promotion_from_detection_threshold() -> None:
    demoted = canonical_review_state(
        origin="binary_drum_classify:kick",
        metadata={"detection": {"threshold_passed": False}},
    )
    promoted = canonical_review_state(
        origin="binary_drum_classify:kick",
        metadata={"detection": {"threshold_passed": True}},
    )

    assert demoted.promotion_state == "demoted"
    assert promoted.promotion_state == "promoted"
