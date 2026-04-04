from __future__ import annotations

from echozero.application.presentation.foundry_viewmodel import FoundryActivityViewModel
from echozero.domain.events import (
    FoundryArtifactFinalizedEvent,
    FoundryArtifactValidatedEvent,
    FoundryRunCreatedEvent,
    FoundryRunStartedEvent,
    create_event_id,
)
from echozero.event_bus import EventBus


def test_foundry_viewmodel_captures_lifecycle_events():
    bus = EventBus()
    vm = FoundryActivityViewModel(bus)

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

    assert len(vm.items) == 4
    assert vm.latest_run_status["run_1"] == "running"
    assert vm.items[-1].kind == "artifact_validated"


def test_foundry_viewmodel_dispose_unsubscribes_handlers():
    bus = EventBus()
    vm = FoundryActivityViewModel(bus)
    vm.dispose()

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

    assert vm.items == []
