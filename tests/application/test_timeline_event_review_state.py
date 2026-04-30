"""Timeline Event review/promotion metadata fallback tests."""

from __future__ import annotations

from echozero.application.shared.ids import EventId, TakeId
from echozero.application.timeline.models import Event


def test_event_promotion_state_uses_detection_promotion_when_review_missing() -> None:
    event = Event(
        id=EventId("evt_1"),
        take_id=TakeId("take_1"),
        start=1.0,
        end=1.1,
        metadata={"detection": {"promotion_state": "demoted"}},
    )

    assert event.promotion_state == "demoted"
    assert event.review_state == "unreviewed"


def test_event_promotion_state_derives_from_detection_threshold_when_needed() -> None:
    demoted = Event(
        id=EventId("evt_demoted"),
        take_id=TakeId("take_1"),
        start=1.0,
        end=1.1,
        metadata={"detection": {"threshold_passed": False}},
    )
    promoted = Event(
        id=EventId("evt_promoted"),
        take_id=TakeId("take_1"),
        start=1.0,
        end=1.1,
        metadata={"detection": {"threshold_passed": True}},
    )

    assert demoted.promotion_state == "demoted"
    assert promoted.promotion_state == "promoted"


def test_event_review_payload_overrides_detection_and_ignores_legacy_top_level_keys() -> None:
    event = Event(
        id=EventId("evt_2"),
        take_id=TakeId("take_1"),
        start=1.0,
        end=1.1,
        metadata={
            "promotion_state": "demoted",
            "review_state": "corrected",
            "review": {
                "promotion_state": "promoted",
                "review_state": "signed_off",
            },
            "detection": {
                "promotion_state": "demoted",
                "threshold_passed": False,
            },
        },
    )

    assert event.promotion_state == "promoted"
    assert event.review_state == "signed_off"


def test_event_without_review_or_detection_legacy_keys_falls_back_to_defaults() -> None:
    event = Event(
        id=EventId("evt_3"),
        take_id=TakeId("take_1"),
        start=1.0,
        end=1.1,
        metadata={
            "promotion_state": "demoted",
            "review_state": "corrected",
        },
    )

    assert event.promotion_state == "promoted"
    assert event.review_state == "unreviewed"
