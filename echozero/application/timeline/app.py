"""
TimelineApplication: Runtime composition for the timeline application contract.
Exists to keep canonical app state in Timeline plus Session, not mutable presentation blobs.
Connects orchestrator, queries, and runtime-audio side effects behind one app-facing surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.intents import (
    DisableSync,
    EnableSync,
    Pause,
    Play,
    Seek,
    SetGain,
    SetLayerMute,
    SetLayerOutputBus,
    SetLayerSolo,
    Stop,
    TimelineIntent,
)
from echozero.application.timeline.orchestrator import (
    MA3TransferWorkspaceService,
    TimelineOrchestrator,
    TimelineMutator,
)
from echozero.application.timeline.queries import TimelineQueries
from echozero.application.timeline.models import Timeline


@dataclass(slots=True)
class TimelineApplication:
    """Compose timeline state, command handling, querying, and runtime side effects."""

    timeline: Timeline
    session: Session
    orchestrator: TimelineOrchestrator
    queries: TimelineQueries
    sync_service: SyncService
    runtime_audio: object | None = None
    runtime_video: object | None = None
    presentation_enricher: Callable[[TimelinePresentation], TimelinePresentation] | None = None

    def presentation(self) -> TimelinePresentation:
        presentation = self.queries.get_presentation(self.timeline, self.session)
        return self._enrich_presentation(presentation)

    def dispatch(self, intent: TimelineIntent) -> TimelinePresentation:
        self._apply_runtime_audio_before_dispatch(intent)
        presentation = self.orchestrator.handle(self.timeline, intent)
        presentation = self._enrich_presentation(presentation)
        self._apply_runtime_audio_after_dispatch(intent, presentation)
        self._apply_runtime_video_for_transport_intent(intent, presentation)
        self._sync_runtime_state_for_transport_intent(intent, presentation)
        return presentation

    @property
    def timeline_mutator(self) -> TimelineMutator:
        """Expose the canonical mutation owner for app-shell collaborators."""

        self.orchestrator._sync_owners()
        return self.orchestrator.mutator

    @property
    def transfer_workspace_service(self) -> MA3TransferWorkspaceService:
        """Expose the MA3 transfer workspace owner for transfer-facing flows."""

        self.orchestrator._sync_owners()
        return self.orchestrator.transfer_workspace

    @property
    def ma3_transfer_workspace(self) -> MA3TransferWorkspaceService:
        """Compatibility alias for the explicit MA3 transfer workspace owner."""

        return self.transfer_workspace_service

    def replace_timeline(self, timeline: Timeline) -> None:
        self.timeline = timeline
        self.session.active_timeline_id = timeline.id

    def enable_sync(self, mode) -> SyncState:
        self.dispatch(EnableSync(mode=mode))
        return self.session.sync_state

    def disable_sync(self) -> SyncState:
        self.dispatch(DisableSync())
        return self.session.sync_state

    def _enrich_presentation(self, presentation: TimelinePresentation) -> TimelinePresentation:
        if self.presentation_enricher is None:
            return presentation
        return self.presentation_enricher(presentation)

    def _apply_runtime_audio_before_dispatch(self, intent: TimelineIntent) -> None:
        runtime_audio = self.runtime_audio
        if runtime_audio is None:
            return

        if isinstance(intent, Play):
            sync_structure_state = getattr(runtime_audio, "sync_structure_state", None)
            if callable(sync_structure_state):
                sync_structure_state(self.presentation())
            else:
                sync_presentation = getattr(runtime_audio, "sync_presentation", None)
                if callable(sync_presentation):
                    sync_presentation(self.presentation())
                else:
                    runtime_audio.build_for_presentation(self.presentation())
            runtime_audio.play()
        elif isinstance(intent, Pause):
            runtime_audio.pause()
        elif isinstance(intent, Stop):
            runtime_audio.stop()
        elif isinstance(intent, Seek):
            runtime_audio.seek(intent.position)

    def _apply_runtime_audio_after_dispatch(
        self,
        intent: TimelineIntent,
        presentation: TimelinePresentation,
    ) -> None:
        runtime_audio = self.runtime_audio
        if runtime_audio is None:
            return

        if isinstance(
            intent,
            (
                SetGain,
                SetLayerMute,
                SetLayerSolo,
                SetLayerOutputBus,
            ),
        ):
            sync_mix_state = getattr(runtime_audio, "sync_mix_state", None)
            if callable(sync_mix_state):
                sync_mix_state(presentation)
            else:
                runtime_audio.apply_mix_state(presentation)

    def _sync_runtime_state_for_transport_intent(
        self,
        intent: TimelineIntent,
        presentation: TimelinePresentation,
    ) -> None:
        runtime_audio = self.runtime_audio
        if runtime_audio is None:
            return
        if not isinstance(intent, (Play, Pause, Stop, Seek)):
            return
        if hasattr(runtime_audio, "snapshot_state"):
            self.session.playback_state = runtime_audio.snapshot_state(presentation)

    def _apply_runtime_video_for_transport_intent(
        self,
        intent: TimelineIntent,
        presentation: TimelinePresentation,
    ) -> None:
        runtime_video = self.runtime_video
        if runtime_video is None or not isinstance(intent, (Play, Pause, Stop, Seek)):
            return
        sync_presentation = getattr(runtime_video, "sync_presentation", None)
        if callable(sync_presentation):
            sync_presentation(presentation)
        if isinstance(intent, Play):
            runtime_video.play(float(presentation.playhead))
        elif isinstance(intent, Pause):
            runtime_video.pause(float(presentation.playhead))
        elif isinstance(intent, Stop):
            runtime_video.stop()
        elif isinstance(intent, Seek):
            runtime_video.seek(float(intent.position))
