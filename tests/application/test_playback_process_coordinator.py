"""Playback process coordinator tests.
Exists to prove transport and prepare lanes stay separated inside the playback service.
Connects queued IPC work to live-controller routing, seek coalescing, and generation state.
"""

from __future__ import annotations

import time

from echozero.application.playback.coordination import (
    PlaybackGenerationStatus,
    TransportCommand,
    TransportCommandAction,
)
from echozero.application.playback.process_coordinator import PlaybackProcessCoordinator
from echozero.application.playback.sync_projection import RuntimeSyncProjection


class _FakeController:
    def __init__(self) -> None:
        self.commands: list[TransportCommand] = []
        self.structure_generations: list[int] = []
        self.mix_calls = 0
        self._next_generation = 1
        self._outcomes: dict[int, str] = {}

    def enqueue_transport_command(self, command: TransportCommand) -> None:
        self.commands.append(command)

    def enqueue_structure_prepare(self, _projection: RuntimeSyncProjection) -> int:
        generation = self._next_generation
        self._next_generation += 1
        self.structure_generations.append(generation)
        return generation

    def sync_mix_state(self, _projection: RuntimeSyncProjection) -> None:
        self.mix_calls += 1

    def drain_pending_structure_sync(self) -> None:
        return

    def generation_outcome(self, generation: int) -> str | None:
        return self._outcomes.get(int(generation))

    def finish_generation(self, generation: int, outcome: str = "applied") -> None:
        self._outcomes[int(generation)] = str(outcome)


def _projection() -> RuntimeSyncProjection:
    return RuntimeSyncProjection(
        layers=[],
        selected_layer_id=None,
        selected_take_id=None,
        playback_output_channels=0,
    )


def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_coordinator_routes_transport_to_current_controller_after_swap() -> None:
    first = _FakeController()
    second = _FakeController()
    current = {"controller": first}
    events: list[tuple[str, dict[str, object]]] = []
    coordinator = PlaybackProcessCoordinator(
        lambda: current["controller"],
        publish_event=lambda event_type, payload: events.append((event_type, payload)),
    )
    try:
        coordinator.enqueue_transport_command(TransportCommand(action=TransportCommandAction.PLAY))
        coordinator.drain_pending_transport_commands()

        current["controller"] = second
        coordinator.enqueue_transport_command(TransportCommand(action=TransportCommandAction.STOP))
        coordinator.drain_pending_transport_commands()

        assert [command.action for command in first.commands] == [TransportCommandAction.PLAY]
        assert [command.action for command in second.commands] == [TransportCommandAction.STOP]
    finally:
        coordinator.shutdown()


def test_coordinator_coalesces_seek_bursts_to_latest_command() -> None:
    controller = _FakeController()
    coordinator = PlaybackProcessCoordinator(
        controller,
        publish_event=lambda _event_type, _payload: None,
    )
    try:
        coordinator.enqueue_transport_command(
            TransportCommand(action=TransportCommandAction.SEEK, position_seconds=1.0)
        )
        coordinator.enqueue_transport_command(
            TransportCommand(action=TransportCommandAction.SCRUB_UPDATE, position_seconds=2.0)
        )
        coordinator.enqueue_transport_command(
            TransportCommand(action=TransportCommandAction.SCRUB_COMMIT, position_seconds=3.0)
        )

        coordinator.drain_pending_transport_commands()

        assert controller.commands[-1].position_seconds == 3.0
        assert coordinator.coalesced_seek_count >= 1
    finally:
        coordinator.shutdown()


def test_coordinator_publishes_generation_ready_after_controller_outcome() -> None:
    controller = _FakeController()
    events: list[tuple[str, dict[str, object]]] = []
    coordinator = PlaybackProcessCoordinator(
        controller,
        publish_event=lambda event_type, payload: events.append((event_type, payload)),
    )
    try:
        work = coordinator.enqueue_structure_sync(_projection())

        coordinator.drain_pending_structure_sync()
        assert controller.structure_generations == [1]
        assert not any(
            payload["generation"] == work.generation
            and payload["status"] == PlaybackGenerationStatus.READY.value
            for event_type, payload in events
            if event_type == "playback-generation"
        )

        controller.finish_generation(1)
        coordinator.drain_pending_structure_sync()

        assert any(
            payload["generation"] == work.generation
            and payload["status"] == PlaybackGenerationStatus.READY.value
            for event_type, payload in events
            if event_type == "playback-generation"
        )
    finally:
        coordinator.shutdown()


def test_coordinator_keeps_mix_update_additive_and_ready() -> None:
    controller = _FakeController()
    events: list[tuple[str, dict[str, object]]] = []
    coordinator = PlaybackProcessCoordinator(
        controller,
        publish_event=lambda event_type, payload: events.append((event_type, payload)),
    )
    try:
        work = coordinator.enqueue_mix_sync(_projection())
        coordinator.drain_pending_structure_sync()

        assert controller.mix_calls == 1
        assert any(
            payload["generation"] == work.generation
            and payload["status"] == PlaybackGenerationStatus.READY.value
            for event_type, payload in events
            if event_type == "playback-generation"
        )
    finally:
        coordinator.shutdown()
