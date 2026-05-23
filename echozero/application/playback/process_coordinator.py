"""
Playback process coordinator for command lane separation.
Exists so HTTP IPC accepts work without running every playback task inline.
Connects high-priority transport commands and lower-priority preparation work to PlaybackController.
"""

from __future__ import annotations

import queue
from collections.abc import Callable
from dataclasses import dataclass

from echozero.application.playback.coordination import (
    PlaybackGenerationStatus,
    TransportCommand,
    TransportCommandAction,
)
from echozero.application.playback.runtime import PlaybackController
from echozero.application.playback.sync_projection import RuntimeSyncProjection


@dataclass(slots=True, frozen=True)
class EnqueuedPlaybackWork:
    """Accepted playback work metadata returned to IPC callers."""

    generation: int = 0
    accepted: bool = True


class PlaybackProcessCoordinator:
    """Coordinates transport and preparation work inside the playback process."""

    def __init__(
        self,
        controller: PlaybackController | Callable[[], PlaybackController],
        *,
        publish_event: Callable[[str, dict[str, object]], None],
    ) -> None:
        if callable(controller):
            self._controller_getter = controller
        else:
            self._controller_getter = lambda: controller
        self._publish_event = publish_event
        self._transport_queue: queue.Queue[TransportCommand | None] = queue.Queue(maxsize=512)
        self._pending_structure_sync: tuple[int, RuntimeSyncProjection] | None = None
        self._pending_mix_sync: tuple[int, RuntimeSyncProjection] | None = None
        self._controller_generation_map: dict[int, int] = {}
        self._next_generation = 1
        self._latest_requested_generation = 0
        self._latest_ready_generation = 0
        self._coalesced_seek_count = 0

    @property
    def latest_requested_generation(self) -> int:
        return int(self._latest_requested_generation)

    @property
    def latest_ready_generation(self) -> int:
        return int(self._latest_ready_generation)

    @property
    def transport_queue_depth(self) -> int:
        return int(self._transport_queue.qsize())

    @property
    def coalesced_seek_count(self) -> int:
        return int(self._coalesced_seek_count)

    def enqueue_transport_command(self, command: TransportCommand) -> EnqueuedPlaybackWork:
        """Accept one high-priority transport command."""

        try:
            self._transport_queue.put_nowait(command)
        except queue.Full:
            _ = self._drop_oldest_transport_command()
            self._transport_queue.put_nowait(command)
        self._publish_event(
            "transport-command-accepted",
            {
                "action": command.action.value,
                "command_id": command.command_id,
                "position_seconds": command.position_seconds,
                "queue_depth": self.transport_queue_depth,
            },
        )
        return EnqueuedPlaybackWork()

    def enqueue_structure_sync(self, projection: RuntimeSyncProjection) -> EnqueuedPlaybackWork:
        """Queue playback graph preparation and return its generation."""

        generation = self._next_preparation_generation()
        previous = self._pending_structure_sync
        if previous is not None and previous[0] != generation:
            self._publish_generation(previous[0], PlaybackGenerationStatus.CANCELLED)
        self._pending_structure_sync = (generation, projection)
        return EnqueuedPlaybackWork(generation=generation)

    def enqueue_mix_sync(self, projection: RuntimeSyncProjection) -> EnqueuedPlaybackWork:
        """Queue mix-state preparation and return its generation."""

        generation = self._next_preparation_generation()
        previous = self._pending_mix_sync
        if previous is not None and previous[0] != generation:
            self._publish_generation(previous[0], PlaybackGenerationStatus.CANCELLED)
        self._pending_mix_sync = (generation, projection)
        return EnqueuedPlaybackWork(generation=generation)

    def drain_pending_structure_sync(self) -> None:
        """Queue/apply lower-priority graph work from the service owner thread."""

        controller = self._controller_getter()
        structure_sync = self._pending_structure_sync
        mix_sync = self._pending_mix_sync
        self._pending_structure_sync = None
        self._pending_mix_sync = None
        if structure_sync is not None:
            generation, projection = structure_sync
            self._queue_structure_prepare(controller, generation, projection)
        if mix_sync is not None:
            generation, projection = mix_sync
            self._apply_mix_sync(controller, generation, projection)
        controller.drain_pending_structure_sync()
        self._publish_completed_controller_generations(controller)

    def drain_pending_transport_commands(self) -> None:
        """Apply accepted transport commands from the service owner thread."""

        commands: list[TransportCommand] = []
        while True:
            try:
                command = self._transport_queue.get_nowait()
            except queue.Empty:
                break
            if command is not None:
                commands.append(command)
        pending_seek: TransportCommand | None = None
        for command in commands:
            if command.action in {
                TransportCommandAction.SEEK,
                TransportCommandAction.SCRUB_UPDATE,
                TransportCommandAction.SCRUB_COMMIT,
            }:
                if pending_seek is not None:
                    self._coalesced_seek_count += 1
                pending_seek = command
                continue
            if pending_seek is not None:
                self._apply_transport_command(pending_seek)
                pending_seek = None
            self._apply_transport_command(command)
        if pending_seek is not None:
            self._apply_transport_command(pending_seek)

    def shutdown(self) -> None:
        """Stop workers without shutting down the owned playback controller."""

        self.drain_pending_transport_commands()

    def _apply_transport_command(self, command: TransportCommand) -> None:
        try:
            self._controller_getter().enqueue_transport_command(command)
        except Exception as exc:
            self._publish_event(
                "transport-command-failed",
                {
                    "action": command.action.value,
                    "command_id": command.command_id,
                    "position_seconds": command.position_seconds,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "queue_depth": self.transport_queue_depth,
                },
            )
            return
        self._publish_event(
            "transport-command-applied",
            {
                "action": command.action.value,
                "command_id": command.command_id,
                "position_seconds": command.position_seconds,
                "queue_depth": self.transport_queue_depth,
                "coalesced_seek_count": int(self._coalesced_seek_count),
            },
        )

    def _queue_structure_prepare(
        self,
        controller: PlaybackController,
        generation: int,
        projection: RuntimeSyncProjection,
    ) -> None:
        self._publish_generation(generation, PlaybackGenerationStatus.PREPARING)
        try:
            enqueue_prepare = getattr(controller, "enqueue_structure_prepare", None)
            if callable(enqueue_prepare):
                controller_generation = int(enqueue_prepare(projection) or 0)
                if controller_generation > 0:
                    self._controller_generation_map[controller_generation] = generation
                    return
            controller.sync_structure_state(projection)
        except Exception as exc:
            self._publish_generation(generation, PlaybackGenerationStatus.FAILED, error=str(exc))
            return
        self._latest_ready_generation = max(int(self._latest_ready_generation), int(generation))
        self._publish_generation(generation, PlaybackGenerationStatus.READY)

    def _apply_mix_sync(
        self,
        controller: PlaybackController,
        generation: int,
        projection: RuntimeSyncProjection,
    ) -> None:
        self._publish_generation(generation, PlaybackGenerationStatus.PREPARING)
        try:
            controller.sync_mix_state(projection)
        except Exception as exc:
            self._publish_generation(generation, PlaybackGenerationStatus.FAILED, error=str(exc))
            return
        self._latest_ready_generation = max(int(self._latest_ready_generation), int(generation))
        self._publish_generation(generation, PlaybackGenerationStatus.READY)

    def _publish_completed_controller_generations(self, controller: PlaybackController) -> None:
        if not self._controller_generation_map:
            return
        generation_outcome = getattr(controller, "generation_outcome", None)
        if not callable(generation_outcome):
            return
        for controller_generation, operation_generation in list(
            self._controller_generation_map.items()
        ):
            outcome = generation_outcome(controller_generation)
            if not outcome:
                continue
            self._controller_generation_map.pop(controller_generation, None)
            if outcome == "applied":
                self._latest_ready_generation = max(
                    int(self._latest_ready_generation),
                    int(operation_generation),
                )
                self._publish_generation(operation_generation, PlaybackGenerationStatus.READY)
            elif outcome in {"stale-dropped"}:
                self._publish_generation(operation_generation, PlaybackGenerationStatus.STALE)
            elif outcome in {"cancelled"}:
                self._publish_generation(operation_generation, PlaybackGenerationStatus.CANCELLED)
            else:
                self._publish_generation(operation_generation, PlaybackGenerationStatus.FAILED)

    def _next_preparation_generation(self) -> int:
        generation = int(self._next_generation)
        self._next_generation += 1
        self._latest_requested_generation = generation
        self._publish_generation(generation, PlaybackGenerationStatus.QUEUED)
        return generation

    def _publish_generation(
        self,
        generation: int,
        status: PlaybackGenerationStatus,
        *,
        error: str = "",
    ) -> None:
        self._publish_event(
            "playback-generation",
            {
                "generation": int(generation),
                "status": status.value,
                "error": str(error or ""),
            },
        )

    def _drop_oldest_transport_command(self) -> bool:
        try:
            _ = self._transport_queue.get_nowait()
        except queue.Empty:
            return False
        return True


__all__ = ["EnqueuedPlaybackWork", "PlaybackProcessCoordinator"]
