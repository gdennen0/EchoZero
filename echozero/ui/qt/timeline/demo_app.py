"""Demo-only timeline fixture app for tests and screenshots.
Exists to build synthetic timeline states without project/runtime wiring.
Never use this module in the canonical user launch or app-shell runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.presentation.models import (
    BatchTransferPlanPresentation,
    BatchTransferPlanRowPresentation,
    EventPresentation,
    LayerPresentation,
    LayerStatusPresentation,
    ManualPullEventOptionPresentation,
    ManualPullFlowPresentation,
    ManualPullTargetOptionPresentation,
    ManualPullTrackOptionPresentation,
    ManualPushFlowPresentation,
    ManualPushTrackOptionPresentation,
    TimelinePresentation,
)
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import FollowMode, PlaybackStatus, SyncMode
from echozero.application.shared.ids import EventId, ProjectId, SessionId, SongId, SongVersionId
from echozero.application.sync.adapters import InMemorySyncService, MA3SyncAdapter, MA3SyncBridge
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
from echozero.application.transport.models import TransportState
from echozero.application.transport.service import TransportService
from echozero.ui.qt.timeline.fixture_loader import load_realistic_timeline_fixture
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController


class DemoSessionService(SessionService):
    def __init__(self, session: Session):
        self._session = session

    def get_session(self) -> Session:
        return self._session

    def set_active_song(self, song_id):
        self._session.active_song_id = song_id
        return self._session

    def set_active_song_version(self, song_version_id):
        self._session.active_song_version_id = song_version_id
        return self._session

    def set_active_timeline(self, timeline_id):
        self._session.active_timeline_id = timeline_id
        return self._session


class DemoTransportService(TransportService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> TransportState:
        return self._session.transport_state

    def play(self) -> TransportState:
        self._session.transport_state.is_playing = True
        return self._session.transport_state

    def pause(self) -> TransportState:
        self._session.transport_state.is_playing = False
        return self._session.transport_state

    def stop(self) -> TransportState:
        self._session.transport_state.is_playing = False
        self._session.transport_state.playhead = 0.0
        return self._session.transport_state

    def seek(self, position: float) -> TransportState:
        self._session.transport_state.playhead = max(0.0, position)
        return self._session.transport_state

    def set_loop(self, loop_region, enabled: bool = True) -> TransportState:
        self._session.transport_state.loop_region = loop_region if enabled else None
        return self._session.transport_state


class DemoMixerService(MixerService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> MixerState:
        return self._session.mixer_state

    def set_layer_state(self, layer_id, state: LayerMixerState) -> MixerState:
        self._session.mixer_state.layer_states[layer_id] = state
        return self._session.mixer_state

    def set_mute(self, layer_id, muted: bool) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.mute = muted
        return self._session.mixer_state

    def set_solo(self, layer_id, soloed: bool) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.solo = soloed
        return self._session.mixer_state

    def set_gain(self, layer_id, gain_db: float) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.gain_db = gain_db
        return self._session.mixer_state

    def set_pan(self, layer_id, pan: float) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.pan = pan
        return self._session.mixer_state

    def resolve_audibility(self, layers) -> list[AudibilityState]:
        resolved: list[AudibilityState] = []
        for layer in layers:
            resolved.append(
                AudibilityState(layer_id=layer.layer_id, is_audible=True, reason="normal")
            )
        return resolved


class DemoPlaybackService(PlaybackService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> PlaybackState:
        return self._session.playback_state

    def prepare(self, timeline) -> PlaybackState:
        return self._session.playback_state

    def update_runtime(
        self, timeline, transport: TransportState, audibility, sync: SyncState
    ) -> PlaybackState:
        self._session.playback_state.status = (
            PlaybackStatus.PLAYING if transport.is_playing else PlaybackStatus.STOPPED
        )
        return self._session.playback_state

    def stop(self) -> PlaybackState:
        self._session.playback_state.status = PlaybackStatus.STOPPED
        return self._session.playback_state


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
            layers = _select_event(
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
                layers=_set_selected_events(
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
            self.presentation_state = _create_demo_event(
                self.presentation_state,
                layer_id=intent.layer_id,
                take_id=intent.take_id,
                start=float(intent.time_range.start),
                end=float(intent.time_range.end),
                label=intent.label,
            )
        elif isinstance(intent, DeleteEvents):
            self.presentation_state = _delete_demo_events(
                self.presentation_state,
                event_ids=list(intent.event_ids),
            )
        elif isinstance(intent, ClearSelection):
            layers = _clear_selection(self.presentation_state.layers)
            self.presentation_state = replace(
                self.presentation_state,
                layers=layers,
                selected_layer_id=None,
                selected_layer_ids=[],
                selected_take_id=None,
                selected_event_ids=[],
            )
        elif isinstance(intent, TriggerTakeAction):
            self.presentation_state = replace(
                self.presentation_state,
                layers=_apply_take_action(
                    self.presentation_state.layers,
                    intent.layer_id,
                    intent.take_id,
                    intent.action_id,
                ),
            )
        elif isinstance(intent, NudgeSelectedEvents):
            self.presentation_state = _nudge_selected_events(
                self.presentation_state,
                direction=int(intent.direction),
                steps=int(intent.steps),
            )
        elif isinstance(intent, MoveSelectedEvents):
            self.presentation_state = _move_selected_events(
                self.presentation_state,
                delta_seconds=float(intent.delta_seconds),
                target_layer_id=intent.target_layer_id,
            )
        elif isinstance(intent, DuplicateSelectedEvents):
            self.presentation_state = _duplicate_selected_events(
                self.presentation_state,
                steps=int(intent.steps),
            )
        elif isinstance(intent, OpenPushToMA3Dialog):
            self.presentation_state = _open_push_workspace(
                self.presentation_state,
                self.sync_service,
                selected_event_ids=list(intent.selection_event_ids),
            )
        elif isinstance(intent, OpenPullFromMA3Dialog):
            self.presentation_state = _open_pull_workspace(
                self.presentation_state,
                self.sync_service,
            )
        self._sync_runtime_state()

        self.presentation_state = replace(
            self.presentation_state,
            is_playing=self.session.transport_state.is_playing,
            playhead=self.session.transport_state.playhead,
            current_time_label=_fmt_time(self.session.transport_state.playhead),
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


def _fmt_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


def _nudge_selected_events(
    presentation: TimelinePresentation,
    *,
    direction: int,
    steps: int,
) -> TimelinePresentation:
    delta = 0.01 * max(1, steps) * (-1 if direction < 0 else 1)
    selected_ids = set(presentation.selected_event_ids)
    layers: list[LayerPresentation] = []
    for layer in presentation.layers:
        updated_events = []
        for event in layer.events:
            if event.event_id in selected_ids:
                start = max(0.0, event.start + delta)
                duration = max(event.duration, 0.01)
                updated_events.append(replace(event, start=start, end=start + duration))
            else:
                updated_events.append(event)
        layers.append(replace(layer, events=updated_events))
    return replace(presentation, layers=layers)


def _duplicate_selected_events(
    presentation: TimelinePresentation,
    *,
    steps: int,
) -> TimelinePresentation:
    selected_ids = set(presentation.selected_event_ids)
    if not selected_ids:
        return presentation
    delta = 0.05 * max(1, steps)
    layers: list[LayerPresentation] = []
    for layer in presentation.layers:
        next_events = list(layer.events)
        for index, event in enumerate(layer.events, start=1):
            if event.event_id not in selected_ids:
                continue
            duration = max(event.duration, 0.01)
            clone_start = event.start + delta
            next_events.append(
                replace(
                    event,
                    event_id=EventId(f"{event.event_id}_dup_{index}_{steps}"),
                    start=clone_start,
                    end=clone_start + duration,
                    is_selected=False,
                )
            )
        next_events.sort(
            key=lambda candidate: (candidate.start, candidate.end, str(candidate.event_id))
        )
        layers.append(replace(layer, events=next_events))
    return replace(presentation, layers=layers)


def _move_selected_events(
    presentation: TimelinePresentation,
    *,
    delta_seconds: float,
    target_layer_id,
) -> TimelinePresentation:
    selected_ids = set(presentation.selected_event_ids)
    if not selected_ids:
        return presentation

    source_layer_id = presentation.selected_layer_id
    updated_layers: list[LayerPresentation] = []
    moved_events: list[EventPresentation] = []

    for layer in presentation.layers:
        next_events: list[EventPresentation] = []
        for event in layer.events:
            if event.event_id not in selected_ids:
                next_events.append(event)
                continue
            duration = max(event.duration, 0.01)
            moved_events.append(
                replace(
                    event,
                    start=max(0.0, event.start + delta_seconds),
                    end=max(0.0, event.start + delta_seconds) + duration,
                    is_selected=True,
                )
            )
        if target_layer_id is not None and layer.layer_id == target_layer_id:
            next_events.extend(moved_events)
            next_events.sort(
                key=lambda candidate: (candidate.start, candidate.end, str(candidate.event_id))
            )
        elif target_layer_id is None and layer.layer_id == source_layer_id:
            next_events.extend(moved_events)
            next_events.sort(
                key=lambda candidate: (candidate.start, candidate.end, str(candidate.event_id))
            )
        updated_layers.append(replace(layer, events=next_events))

    selected_layer_id = target_layer_id if target_layer_id is not None else source_layer_id
    return replace(
        presentation,
        layers=updated_layers,
        selected_layer_id=selected_layer_id,
        selected_layer_ids=[] if selected_layer_id is None else [selected_layer_id],
    )


def _open_push_workspace(
    presentation: TimelinePresentation,
    sync_service: SyncService,
    *,
    selected_event_ids: list[EventId],
) -> TimelinePresentation:
    selected_layer_ids = list(presentation.selected_layer_ids) or (
        [presentation.selected_layer_id] if presentation.selected_layer_id is not None else []
    )
    track_options = [
        ManualPushTrackOptionPresentation(
            coord=str(option.coord),
            name=str(option.name),
            note=option.note,
            event_count=option.event_count,
        )
        for option in _sync_list(sync_service, "list_push_track_options")
    ]
    target_layer_id = presentation.selected_layer_id
    layers: list[LayerPresentation] = []
    for layer in presentation.layers:
        if layer.layer_id == target_layer_id:
            layers.append(
                replace(
                    layer,
                    push_selection_count=len(selected_event_ids),
                    push_row_status="blocked",
                    push_row_issue="Select an MA3 target track",
                )
            )
        else:
            layers.append(layer)
    batch_plan = BatchTransferPlanPresentation(
        plan_id="push:timeline_selection",
        operation_type="push",
        rows=(
            [
                BatchTransferPlanRowPresentation(
                    row_id=f"push:{target_layer_id}",
                    direction="push",
                    source_label=next(
                        (layer.title for layer in layers if layer.layer_id == target_layer_id),
                        "Selection",
                    ),
                    target_label="Unmapped",
                    source_layer_id=target_layer_id,
                    selected_event_ids=list(selected_event_ids),
                    selected_count=len(selected_event_ids),
                    status="blocked",
                    issue="Select an MA3 target track",
                )
            ]
            if target_layer_id is not None
            else []
        ),
        blocked_count=1 if target_layer_id is not None else 0,
    )
    return replace(
        presentation,
        layers=layers,
        manual_push_flow=ManualPushFlowPresentation(
            dialog_open=False,
            push_mode_active=True,
            selected_layer_ids=selected_layer_ids,
            available_tracks=track_options,
            transfer_mode="merge",
        ),
        batch_transfer_plan=batch_plan,
    )


def _open_pull_workspace(
    presentation: TimelinePresentation,
    sync_service: SyncService,
) -> TimelinePresentation:
    track_options = [
        ManualPullTrackOptionPresentation(
            coord=str(option.coord),
            name=str(option.name),
            note=option.note,
            event_count=option.event_count,
        )
        for option in _sync_list(sync_service, "list_pull_track_options")
    ]
    target_options = [
        ManualPullTargetOptionPresentation(layer_id=layer.layer_id, name=layer.title)
        for layer in presentation.layers
        if layer.kind.name == "EVENT"
    ]
    return replace(
        presentation,
        manual_pull_flow=ManualPullFlowPresentation(
            dialog_open=False,
            workspace_active=True,
            available_tracks=track_options,
            available_events=(
                [
                    ManualPullEventOptionPresentation(
                        event_id=str(option.event_id),
                        label=str(option.label),
                        start=option.start,
                        end=option.end,
                    )
                    for option in _sync_list(
                        sync_service, "list_pull_source_events", "tc1_tg2_tr3"
                    )
                ]
                if track_options
                else []
            ),
            available_target_layers=target_options,
        ),
        batch_transfer_plan=BatchTransferPlanPresentation(
            plan_id="pull:timeline_selection",
            operation_type="pull",
        ),
    )


def _sync_list(sync_service: SyncService, method_name: str, *args):
    method = getattr(sync_service, method_name, None)
    if not callable(method):
        return []
    return list(method(*args))


def _apply_take_action(
    layers: list[LayerPresentation],
    layer_id,
    take_id,
    action_id: str,
) -> list[LayerPresentation]:
    updated: list[LayerPresentation] = []
    for layer in layers:
        if layer.layer_id != layer_id:
            updated.append(layer)
            continue

        take = next((candidate for candidate in layer.takes if candidate.take_id == take_id), None)
        if take is None:
            updated.append(layer)
            continue

        next_events = list(layer.events)
        if action_id in {"overwrite_main", "promote_take"}:
            next_events = _clone_events_for_main(take.events, suffix="ow")
        elif action_id == "merge_main":
            merged = list(layer.events)
            merged.extend(_clone_events_for_main(take.events, suffix="mg"))
            next_events = sorted(merged, key=lambda e: (e.start, e.end))

        next_status = replace(layer.status, stale=False, manually_modified=True)
        updated.append(replace(layer, events=next_events, status=next_status))
    return updated


def _clone_events_for_main(
    events: list[EventPresentation], *, suffix: str
) -> list[EventPresentation]:
    clones: list[EventPresentation] = []
    for idx, event in enumerate(events, start=1):
        clones.append(
            replace(
                event,
                event_id=EventId(f"{event.event_id}_{suffix}_{idx}"),
                is_selected=False,
            )
        )
    return clones


def _clear_selection(layers: list[LayerPresentation]) -> list[LayerPresentation]:
    updated: list[LayerPresentation] = []
    for layer in layers:
        updated.append(
            replace(
                layer,
                is_selected=False,
                events=[replace(event, is_selected=False) for event in layer.events],
                takes=[
                    replace(
                        take, events=[replace(event, is_selected=False) for event in take.events]
                    )
                    for take in layer.takes
                ],
            )
        )
    return updated


def _select_event(
    layers: list[LayerPresentation],
    *,
    layer_id,
    take_id,
    event_id,
) -> list[LayerPresentation]:
    updated: list[LayerPresentation] = []
    for layer in layers:
        is_layer_selected = layer.layer_id == layer_id
        events = [
            replace(
                event,
                is_selected=is_layer_selected and take_id is None and event.event_id == event_id,
            )
            for event in layer.events
        ]
        takes = []
        for take in layer.takes:
            takes.append(
                replace(
                    take,
                    events=[
                        replace(
                            event,
                            is_selected=is_layer_selected
                            and take.take_id == take_id
                            and event.event_id == event_id,
                        )
                        for event in take.events
                    ],
                )
            )
        updated.append(replace(layer, is_selected=is_layer_selected, events=events, takes=takes))
    return updated


def _set_selected_events(
    layers: list[LayerPresentation],
    *,
    selected_event_ids: list[EventId],
) -> list[LayerPresentation]:
    selected_lookup = set(selected_event_ids)
    updated_layers: list[LayerPresentation] = []
    for layer in layers:
        updated_layers.append(
            replace(
                layer,
                is_selected=any(event.event_id in selected_lookup for event in layer.events)
                or any(
                    event.event_id in selected_lookup
                    for take in layer.takes
                    for event in take.events
                ),
                events=[
                    replace(event, is_selected=event.event_id in selected_lookup)
                    for event in layer.events
                ],
                takes=[
                    replace(
                        take,
                        events=[
                            replace(event, is_selected=event.event_id in selected_lookup)
                            for event in take.events
                        ],
                    )
                    for take in layer.takes
                ],
            )
        )
    return updated_layers


def _create_demo_event(
    presentation: TimelinePresentation,
    *,
    layer_id,
    take_id,
    start: float,
    end: float,
    label: str,
) -> TimelinePresentation:
    target_event_id = None
    target_take_id = take_id
    updated_layers: list[LayerPresentation] = []
    for layer in presentation.layers:
        if layer.layer_id != layer_id:
            updated_layers.append(layer)
            continue

        target_take_id = take_id if take_id is not None else layer.main_take_id
        if take_id in (None, layer.main_take_id):
            target_event_id = _next_demo_event_id(
                layer.main_take_id or EventId(f"{layer.layer_id}:main"), layer.events
            )
            created = EventPresentation(
                event_id=target_event_id,
                start=start,
                end=end,
                label=label,
                is_selected=True,
            )
            updated_layers.append(
                replace(
                    layer,
                    is_selected=True,
                    events=sorted(
                        [replace(event, is_selected=False) for event in layer.events] + [created],
                        key=lambda event: (event.start, event.end, str(event.event_id)),
                    ),
                    takes=[
                        replace(
                            take,
                            events=[replace(event, is_selected=False) for event in take.events],
                        )
                        for take in layer.takes
                    ],
                )
            )
            continue

        updated_takes: list[TakeLanePresentation] = []
        for take in layer.takes:
            if take.take_id != take_id:
                updated_takes.append(
                    replace(
                        take,
                        events=[replace(event, is_selected=False) for event in take.events],
                    )
                )
                continue
            target_event_id = _next_demo_event_id(take.take_id, take.events)
            created = EventPresentation(
                event_id=target_event_id,
                start=start,
                end=end,
                label=label,
                is_selected=True,
            )
            updated_takes.append(
                replace(
                    take,
                    events=sorted(
                        [replace(event, is_selected=False) for event in take.events] + [created],
                        key=lambda event: (event.start, event.end, str(event.event_id)),
                    ),
                )
            )
        updated_layers.append(
            replace(
                layer,
                is_selected=True,
                events=[replace(event, is_selected=False) for event in layer.events],
                takes=updated_takes,
            )
        )

    return replace(
        presentation,
        layers=updated_layers,
        selected_layer_id=layer_id,
        selected_layer_ids=[layer_id],
        selected_take_id=target_take_id,
        selected_event_ids=[] if target_event_id is None else [target_event_id],
    )


def _delete_demo_events(
    presentation: TimelinePresentation,
    *,
    event_ids: list[EventId],
) -> TimelinePresentation:
    delete_lookup = set(event_ids)
    layers = [
        replace(
            layer,
            events=[
                replace(event, is_selected=False)
                for event in layer.events
                if event.event_id not in delete_lookup
            ],
            takes=[
                replace(
                    take,
                    events=[
                        replace(event, is_selected=False)
                        for event in take.events
                        if event.event_id not in delete_lookup
                    ],
                )
                for take in layer.takes
            ],
        )
        for layer in presentation.layers
    ]
    return replace(
        presentation,
        layers=layers,
        selected_take_id=None,
        selected_event_ids=[],
    )


def _next_demo_event_id(take_id, events: list[EventPresentation]) -> EventId:
    existing = {str(event.event_id) for event in events}
    index = 1
    while True:
        candidate = EventId(f"{take_id}:event:{index}")
        if str(candidate) not in existing:
            return candidate
        index += 1


def build_demo_presentation() -> TimelinePresentation:
    """Build the synthetic fixture presentation used by timeline-only tests."""
    return load_realistic_timeline_fixture()


def build_demo_app(
    *,
    sync_bridge: MA3SyncBridge | None = None,
    sync_service: SyncService | None = None,
) -> DemoTimelineApp:
    """Build the synthetic demo app used by timeline fixture tests only."""
    presentation = build_demo_presentation()
    session = Session(
        id=SessionId("session_demo"),
        project_id=ProjectId("project_demo"),
        active_song_id=SongId("song_demo"),
        active_song_version_id=SongVersionId("song_version_demo"),
        active_timeline_id=presentation.timeline_id,
        transport_state=TransportState(
            is_playing=presentation.is_playing,
            playhead=presentation.playhead,
            follow_mode=presentation.follow_mode,
        ),
        mixer_state=MixerState(),
        playback_state=PlaybackState(
            status=PlaybackStatus.PLAYING if presentation.is_playing else PlaybackStatus.STOPPED,
            backend_name="demo",
        ),
        sync_state=SyncState(mode=SyncMode.MA3, connected=True, target_ref="show_manager"),
    )

    runtime_sync_service: SyncService
    if sync_service is not None:
        runtime_sync_service = sync_service
    elif sync_bridge is not None:
        runtime_sync_service = MA3SyncAdapter(
            sync_bridge, state=session.sync_state, target_ref="show_manager"
        )
    else:
        runtime_sync_service = InMemorySyncService(session.sync_state)

    return DemoTimelineApp(
        presentation_state=presentation,
        session=session,
        sync_service=runtime_sync_service,
    )


def build_real_data_demo_app(
    audio_path: str | Path,
    *,
    working_root: str | Path,
    song_title: str = "Doechii Nissan Altima",
    runtime_audio: TimelineRuntimeAudioController | None = None,
) -> tuple[DemoTimelineApp, object]:
    """Build a demo app around real data fixtures for explicit demo/test flows only."""
    from echozero.ui.qt.timeline.real_data_fixture import build_real_data_presentation

    presentation, summary = build_real_data_presentation(
        audio_path=audio_path,
        working_root=working_root,
        song_title=song_title,
    )
    app = build_demo_app()
    app.presentation_state = presentation
    app.session.active_timeline_id = presentation.timeline_id
    app.session.transport_state.is_playing = presentation.is_playing
    app.session.transport_state.playhead = presentation.playhead
    app.runtime_audio = runtime_audio or TimelineRuntimeAudioController()
    app.runtime_audio.build_for_presentation(presentation)
    return app, summary
