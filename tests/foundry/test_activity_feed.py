from __future__ import annotations

from echozero.domain.events import (
    FoundryArtifactFinalizedEvent,
    FoundryArtifactValidatedEvent,
    FoundryRunCreatedEvent,
    FoundryRunStartedEvent,
    create_event_id,
)
from echozero.event_bus import EventBus
from echozero.foundry.presentation import FoundryActivityFeed


def test_foundry_activity_feed_captures_lifecycle_events():
    bus = EventBus()
    feed = FoundryActivityFeed(bus)

    bus.publish(
        FoundryRunCreatedEvent(
            event_id=create_event_id(),
            timestamp=1.0,
            correlation_id="run_1",
            run_id="run_1",
            dataset_version_id="dsv_1",
            status="queued",
        )
    )
    bus.publish(
        FoundryRunStartedEvent(
            event_id=create_event_id(),
            timestamp=2.0,
            correlation_id="run_1",
            run_id="run_1",
            status="running",
        )
    )
    bus.publish(
        FoundryArtifactFinalizedEvent(
            event_id=create_event_id(),
            timestamp=3.0,
            correlation_id="run_1",
            artifact_id="art_1",
            run_id="run_1",
        )
    )
    bus.publish(
        FoundryArtifactValidatedEvent(
            event_id=create_event_id(),
            timestamp=4.0,
            correlation_id="art_1",
            artifact_id="art_1",
            consumer="PyTorchAudioClassify",
            ok=True,
            error_count=0,
            warning_count=1,
        )
    )

    assert len(feed.items) == 4
    assert feed.latest_run_status["run_1"] == "running"
    assert feed.items[-1].kind == "artifact_validated"


def test_foundry_activity_feed_dispose_unsubscribes_handlers():
    bus = EventBus()
    feed = FoundryActivityFeed(bus)
    feed.dispose()

    bus.publish(
        FoundryRunCreatedEvent(
            event_id=create_event_id(),
            timestamp=1.0,
            correlation_id="run_2",
            run_id="run_2",
            dataset_version_id="dsv_2",
            status="queued",
        )
    )

    assert feed.items == []
