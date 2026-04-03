"""Timeline orchestration for the new EchoZero application layer."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from echozero.application.mixer.service import MixerService
from echozero.application.playback.service import PlaybackService
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.service import SessionService
from echozero.application.sync.service import SyncService
if TYPE_CHECKING:
    from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.intents import (
    TimelineIntent,
    SelectEvent,
    SelectLayer,
    SelectTake,
    ToggleLayerExpanded,
    ToggleTakeSelector,
    TriggerTakeAction,
    Play,
    Pause,
    Seek,
    ToggleMute,
    ToggleSolo,
    SetGain,
    EnableSync,
    DisableSync,
)
from echozero.application.timeline.models import Timeline, Layer, Take, Event
from echozero.application.transport.service import TransportService


@dataclass(slots=True)
class TimelineOrchestrator:
    """Coordinates timeline intents across sibling application services."""

    session_service: SessionService
    transport_service: TransportService
    mixer_service: MixerService
    playback_service: PlaybackService
    sync_service: SyncService
    assembler: 'TimelineAssembler'

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

        elif isinstance(intent, TriggerTakeAction):
            self._handle_trigger_take_action(
                timeline,
                layer_id=intent.layer_id,
                take_id=intent.take_id,
                action_id=intent.action_id,
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

    def _handle_trigger_take_action(self, timeline: Timeline, layer_id, take_id, action_id: str) -> None:
        layer = self._find_layer(timeline, layer_id)
        source_take = self._find_take(layer, take_id)
        main_take = self._main_take(layer)
        if source_take is None or main_take is None:
            return

        normalized = (action_id or "").strip().lower()
        if normalized in {"overwrite_main", "promote_take"}:
            if source_take.id == main_take.id:
                return
            main_take.events = self._clone_events_for_target(source_take.events, main_take)
        elif normalized == "merge_main":
            if source_take.id == main_take.id:
                return
            merged = list(main_take.events)
            merged.extend(self._clone_events_for_target(source_take.events, main_take))
            main_take.events = sorted(merged, key=lambda event: (event.start, event.end, str(event.id)))
        else:
            return

        layer.active_take_id = main_take.id
        timeline.selection.selected_layer_id = layer.id
        timeline.selection.selected_take_id = main_take.id
        timeline.selection.selected_event_ids = []

    @staticmethod
    def _clone_events_for_target(events: list[Event], target_take: Take) -> list[Event]:
        clones: list[Event] = []
        for idx, event in enumerate(events, start=1):
            clones.append(
                Event(
                    id=f"{target_take.id}:from:{event.id}:{idx}",
                    take_id=target_take.id,
                    start=event.start,
                    end=event.end,
                    payload_ref=event.payload_ref,
                    label=event.label,
                    color=event.color,
                    muted=event.muted,
                )
            )
        return clones

    @staticmethod
    def _main_take(layer: Layer) -> Take | None:
        if layer.takes:
            return layer.takes[0]
        return None

    @staticmethod
    def _find_take(layer: Layer, take_id) -> Take | None:
        for take in layer.takes:
            if take.id == take_id:
                return take
        return None

    def _find_layer(self, timeline: Timeline, layer_id):
        for layer in timeline.layers:
            if layer.id == layer_id:
                return layer
        raise ValueError(f"Layer not found: {layer_id}")
