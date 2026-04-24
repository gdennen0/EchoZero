"""Demo-only timeline runtime for fixture app dispatch.
Exists to keep support-only intent dispatch and sync toggles out of the demo app builder entrypoint.
Never use this runtime as the canonical timeline application surface.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.shared.enums import SyncMode
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.intents import (
    ClearSelection,
    CreateEvent,
    DeleteEvents,
    DuplicateSelectedEvents,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    OpenPullFromMA3Dialog,
    OpenPushToMA3Dialog,
    Pause,
    Play,
    Seek,
    SelectEvent,
    SelectLayer,
    SelectTake,
    SetActivePlaybackTarget,
    SetGain,
    SetSelectedEvents,
    Stop,
    TimelineIntent,
    ToggleLayerExpanded,
    TriggerTakeAction,
)
from echozero.ui.qt.timeline.demo_app_mutations import (
    apply_take_action,
    clear_selection,
    create_demo_event,
    delete_demo_events,
    duplicate_selected_events,
    format_demo_time,
    move_selected_events,
    nudge_selected_events,
    open_pull_workspace,
    open_push_workspace,
    select_event,
    set_selected_events,
)
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController


@dataclass(slots=True)
class DemoTimelineApp:
    presentation_state: TimelinePresentation
    session: Session
    sync_service: SyncService
    runtime_audio: TimelineRuntimeAudioController | None = None

    def presentation(self) -> TimelinePresentation:
        return self.presentation_state

    def _sync_runtime_state(self) -> None:
        if self.runtime_audio is not None:
            self.session.transport_state.playhead = self.runtime_audio.current_time_seconds()
            self.session.transport_state.is_playing = self.runtime_audio.is_playing()
            if hasattr(self.runtime_audio, "snapshot_state"):
                self.session.playback_state = self.runtime_audio.snapshot_state(
                    self.presentation_state
                )

    def dispatch(self, intent: TimelineIntent) -> TimelinePresentation:
        if isinstance(intent, Pause):
            self.session.transport_state.is_playing = False
            if self.runtime_audio is not None:
                self.runtime_audio.pause()
        elif isinstance(intent, Play):
            if self.runtime_audio is not None:
                self.runtime_audio.build_for_presentation(self.presentation_state)
                self.runtime_audio.play()
            self.session.transport_state.is_playing = True
        elif isinstance(intent, Stop):
            if self.runtime_audio is not None:
                self.runtime_audio.stop()
            self.session.transport_state.is_playing = False
            self.session.transport_state.playhead = 0.0
        elif isinstance(intent, Seek):
            if self.runtime_audio is not None:
                self.runtime_audio.seek(intent.position)
            self.session.transport_state.playhead = max(0.0, intent.position)
        elif isinstance(intent, SetGain):
            layers = []
            for layer in self.presentation_state.layers:
                if layer.layer_id == intent.layer_id:
                    layers.append(replace(layer, gain_db=float(intent.gain_db)))
                else:
                    layers.append(layer)
            self.presentation_state = replace(self.presentation_state, layers=layers)
            if self.runtime_audio is not None:
                self.runtime_audio.apply_mix_state(self.presentation_state)
        elif isinstance(intent, ToggleLayerExpanded):
            layers = []
            for layer in self.presentation_state.layers:
                if layer.layer_id == intent.layer_id:
                    layers.append(replace(layer, is_expanded=not layer.is_expanded))
                else:
                    layers.append(layer)
            self.presentation_state = replace(self.presentation_state, layers=layers)
        elif isinstance(intent, SelectLayer):
            selected_ids = [intent.layer_id] if intent.layer_id is not None else []
            if intent.mode == "toggle" and intent.layer_id is not None:
                current_ids = list(self.presentation_state.selected_layer_ids) or (
                    [self.presentation_state.selected_layer_id]
                    if self.presentation_state.selected_layer_id is not None
                    else []
                )
                if intent.layer_id in current_ids:
                    selected_ids = [
                        layer_id for layer_id in current_ids if layer_id != intent.layer_id
                    ]
                else:
                    selected_ids = [*current_ids, intent.layer_id]
            elif intent.mode == "range" and intent.layer_id is not None:
                ordered_ids = [layer.layer_id for layer in self.presentation_state.layers]
                anchor_id = self.presentation_state.selected_layer_id or intent.layer_id
                low, high = sorted(
                    (ordered_ids.index(anchor_id), ordered_ids.index(intent.layer_id))
                )
                selected_ids = ordered_ids[low : high + 1]
            layers = [
                replace(layer, is_selected=(layer.layer_id in selected_ids))
                for layer in self.presentation_state.layers
            ]
            self.presentation_state = replace(
                self.presentation_state,
                layers=layers,
                selected_layer_id=intent.layer_id if selected_ids else None,
                selected_layer_ids=selected_ids,
                selected_take_id=None,
                selected_event_ids=[],
            )
        elif isinstance(intent, SelectTake):
            layers = [
                replace(layer, is_selected=(layer.layer_id == intent.layer_id))
                for layer in self.presentation_state.layers
            ]
            self.presentation_state = replace(
                self.presentation_state,
                layers=layers,
                selected_layer_id=intent.layer_id,
                selected_layer_ids=[intent.layer_id],
                selected_take_id=intent.take_id,
                selected_event_ids=[],
            )
        elif isinstance(intent, SetActivePlaybackTarget):
            self.presentation_state = replace(
                self.presentation_state,
                active_playback_layer_id=intent.layer_id,
                active_playback_take_id=intent.take_id,
            )
            if self.runtime_audio is not None:
                self.runtime_audio.apply_mix_state(self.presentation_state)
        elif isinstance(intent, SelectEvent):
            layers = select_event(
                self.presentation_state.layers,
                layer_id=intent.layer_id,
                take_id=intent.take_id,
                event_id=intent.event_id,
            )
            selected_ids = [] if intent.event_id is None else [intent.event_id]
            self.presentation_state = replace(
                self.presentation_state,
                layers=layers,
                selected_layer_id=intent.layer_id,
                selected_layer_ids=[intent.layer_id],
                selected_take_id=intent.take_id,
                selected_event_ids=selected_ids,
            )
        elif isinstance(intent, SetSelectedEvents):
            self.presentation_state = replace(
                self.presentation_state,
                layers=set_selected_events(
                    self.presentation_state.layers,
                    selected_event_ids=list(intent.event_ids),
                ),
                selected_layer_id=intent.anchor_layer_id,
                selected_layer_ids=list(
                    intent.selected_layer_ids
                    or ([] if intent.anchor_layer_id is None else [intent.anchor_layer_id])
                ),
                selected_take_id=intent.anchor_take_id,
                selected_event_ids=list(intent.event_ids),
            )
        elif isinstance(intent, CreateEvent):
            self.presentation_state = create_demo_event(
                self.presentation_state,
                layer_id=intent.layer_id,
                take_id=intent.take_id,
                start=float(intent.time_range.start),
                end=float(intent.time_range.end),
                label=intent.label,
            )
        elif isinstance(intent, DeleteEvents):
            self.presentation_state = delete_demo_events(
                self.presentation_state,
                event_ids=list(intent.event_ids),
            )
        elif isinstance(intent, ClearSelection):
            self.presentation_state = replace(
                self.presentation_state,
                layers=clear_selection(self.presentation_state.layers),
                selected_layer_id=None,
                selected_layer_ids=[],
                selected_take_id=None,
                selected_event_ids=[],
            )
        elif isinstance(intent, TriggerTakeAction):
            self.presentation_state = replace(
                self.presentation_state,
                layers=apply_take_action(
                    self.presentation_state,
                    intent.layer_id,
                    intent.take_id,
                    intent.action_id,
                ),
            )
        elif isinstance(intent, NudgeSelectedEvents):
            self.presentation_state = nudge_selected_events(
                self.presentation_state,
                direction=int(intent.direction),
                steps=int(intent.steps),
            )
        elif isinstance(intent, MoveSelectedEvents):
            self.presentation_state = move_selected_events(
                self.presentation_state,
                delta_seconds=float(intent.delta_seconds),
                target_layer_id=intent.target_layer_id,
            )
        elif isinstance(intent, DuplicateSelectedEvents):
            self.presentation_state = duplicate_selected_events(
                self.presentation_state,
                steps=int(intent.steps),
            )
        elif isinstance(intent, OpenPushToMA3Dialog):
            self.presentation_state = open_push_workspace(
                self.presentation_state,
                self.sync_service,
                selected_event_ids=list(intent.selection_event_ids),
            )
        elif isinstance(intent, OpenPullFromMA3Dialog):
            self.presentation_state = open_pull_workspace(
                self.presentation_state,
                self.sync_service,
            )
        self._sync_runtime_state()

        self.presentation_state = replace(
            self.presentation_state,
            is_playing=self.session.transport_state.is_playing,
            playhead=self.session.transport_state.playhead,
            current_time_label=format_demo_time(self.session.transport_state.playhead),
        )
        return self.presentation_state

    def enable_sync(self, mode: SyncMode = SyncMode.MA3) -> SyncState:
        state = self.sync_service.set_mode(mode)
        self.session.sync_state = state
        self.session.sync_state = self.sync_service.connect()
        return self.session.sync_state

    def disable_sync(self) -> SyncState:
        self.session.sync_state = self.sync_service.disconnect()
        return self.session.sync_state


__all__ = ["DemoTimelineApp"]
