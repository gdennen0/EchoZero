"""Timeline orchestration for the new EchoZero application layer."""

from dataclasses import dataclass

from echozero.application.mixer.service import MixerService
from echozero.application.playback.service import PlaybackService
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.service import SessionService
from echozero.application.sync.service import SyncService
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.intents import (
    TimelineIntent,
    SelectEvent,
    SelectLayer,
    SelectTake,
    ToggleLayerExpanded,
    ToggleTakeSelector,
    Play,
    Pause,
    Seek,
    ToggleMute,
    ToggleSolo,
    SetGain,
    EnableSync,
    DisableSync,
)
from echozero.application.timeline.models import Timeline
from echozero.application.transport.service import TransportService


@dataclass(slots=True)
class TimelineOrchestrator:
    """Coordinates timeline intents across sibling application services."""

    session_service: SessionService
    transport_service: TransportService
    mixer_service: MixerService
    playback_service: PlaybackService
    sync_service: SyncService
    assembler: TimelineAssembler

    def handle(self, timeline: Timeline, intent: TimelineIntent) -> TimelinePresentation:
        if isinstance(intent, SelectLayer):
            timeline.selection.selected_layer_id = intent.layer_id
            if intent.layer_id is None:
                timeline.selection.selected_take_id = None
                timeline.selection.selected_event_ids = []

        elif isinstance(intent, SelectTake):
            self._handle_select_take(timeline, intent.layer_id, intent.take_id)

        elif isinstance(intent, SelectEvent):
            timeline.selection.selected_layer_id = intent.layer_id
            timeline.selection.selected_event_ids = [intent.event_id] if intent.event_id is not None else []

        elif isinstance(intent, ToggleLayerExpanded):
            layer = self._find_layer(timeline, intent.layer_id)
            layer.presentation_hints.collapsed = not layer.presentation_hints.collapsed

        elif isinstance(intent, ToggleTakeSelector):
            layer = self._find_layer(timeline, intent.layer_id)
            layer.presentation_hints.take_selector_expanded = (
                not layer.presentation_hints.take_selector_expanded
            )

        elif isinstance(intent, Play):
            self.transport_service.play()

        elif isinstance(intent, Pause):
            self.transport_service.pause()

        elif isinstance(intent, Seek):
            self.transport_service.seek(intent.position)

        elif isinstance(intent, ToggleMute):
            layer = self._find_layer(timeline, intent.layer_id)
            self.mixer_service.set_mute(intent.layer_id, not layer.mixer.mute)
            layer.mixer.mute = not layer.mixer.mute

        elif isinstance(intent, ToggleSolo):
            layer = self._find_layer(timeline, intent.layer_id)
            self.mixer_service.set_solo(intent.layer_id, not layer.mixer.solo)
            layer.mixer.solo = not layer.mixer.solo

        elif isinstance(intent, SetGain):
            layer = self._find_layer(timeline, intent.layer_id)
            self.mixer_service.set_gain(intent.layer_id, intent.gain_db)
            layer.mixer.gain_db = intent.gain_db

        elif isinstance(intent, EnableSync):
            sync_state = self.sync_service.set_mode(intent.mode)
            self.sync_service.connect()
            session = self.session_service.get_session()
            session.sync_state = sync_state

        elif isinstance(intent, DisableSync):
            session = self.session_service.get_session()
            session.sync_state = self.sync_service.disconnect()

        session = self.session_service.get_session()
        audibility = self.mixer_service.resolve_audibility(timeline.layers)
        self.playback_service.update_runtime(
            timeline=timeline,
            transport=session.transport_state,
            audibility=audibility,
            sync=session.sync_state,
        )
        return self.assembler.assemble(timeline=timeline, session=session)

    def _handle_select_take(self, timeline: Timeline, layer_id, take_id) -> None:
        layer = self._find_layer(timeline, layer_id)
        layer.active_take_id = take_id
        timeline.selection.selected_layer_id = layer_id
        timeline.selection.selected_take_id = take_id
        timeline.selection.selected_event_ids = []

    def _find_layer(self, timeline: Timeline, layer_id):
        for layer in timeline.layers:
            if layer.id == layer_id:
                return layer
        raise ValueError(f"Layer not found: {layer_id}")
