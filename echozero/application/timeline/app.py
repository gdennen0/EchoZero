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
from echozero.application.timeline.orchestrator import TimelineOrchestrator
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
    presentation_enricher: Callable[[TimelinePresentation], TimelinePresentation] | None = None

    def presentation(self) -> TimelinePresentation:
        presentation = self.queries.get_presentation(self.timeline, self.session)
        return self._enrich_presentation(presentation)

    def dispatch(self, intent: TimelineIntent) -> TimelinePresentation:
        self._apply_runtime_audio_before_dispatch(intent)
        presentation = self.orchestrator.handle(self.timeline, intent)
        presentation = self._enrich_presentation(presentation)
        self._apply_runtime_audio_after_dispatch(intent, presentation)
        self._sync_runtime_state(presentation)
        return presentation

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

    def _sync_runtime_state(self, presentation: TimelinePresentation) -> None:
        runtime_audio = self.runtime_audio
        if runtime_audio is None:
            return

        if hasattr(runtime_audio, "snapshot_state"):
            self.session.playback_state = runtime_audio.snapshot_state(presentation)
