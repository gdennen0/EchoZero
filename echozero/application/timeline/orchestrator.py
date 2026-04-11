"""Timeline orchestration for the new EchoZero application layer."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from echozero.application.mixer.service import MixerService
from echozero.application.playback.service import PlaybackService
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import (
    ManualPullDiffPreview,
    ManualPullEventOption,
    ManualPullTargetOption,
    ManualPullTrackOption,
    ManualPushDiffPreview,
    ManualPushTrackOption,
)
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import LayerKind
from echozero.application.sync.diff_service import SyncDiffService
from echozero.application.sync.models import LiveSyncState
from echozero.application.sync.service import SyncService
if TYPE_CHECKING:
    from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    ClearLayerLiveSyncPauseReason,
    ClearSelection,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    DisableExperimentalLiveSync,
    DuplicateSelectedEvents,
    DisableSync,
    EnableExperimentalLiveSync,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    OpenPullFromMA3Dialog,
    SelectAllEvents,
    SelectEvent,
    SelectLayer,
    SelectPullSourceEvents,
    SelectPullSourceTrack,
    SelectPullTargetLayer,
    SelectTake,
    EnableSync,
    OpenPushToMA3Dialog,
    Play,
    Pause,
    Seek,
    SetPullSourceEvents,
    SetPullTrackOptions,
    SelectPushTargetTrack,
    SetGain,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    SetPushTrackOptions,
    Stop,
    TimelineIntent,
    ToggleLayerExpanded,
    ToggleMute,
    ToggleSolo,
    TriggerTakeAction,
)
from echozero.application.timeline.models import Timeline, Layer, Take, Event
from echozero.application.transport.service import TransportService

_KEYBOARD_STEP_SECONDS = 1.0 / 30.0
_RECONNECT_REARM_REQUIRED_REASON = "Live sync reconnected; explicit re-arm required"


@dataclass(slots=True)
class TimelineOrchestrator:
    """Coordinates timeline intents across sibling application services."""

    session_service: SessionService
    transport_service: TransportService
    mixer_service: MixerService
    playback_service: PlaybackService
    sync_service: SyncService
    assembler: 'TimelineAssembler'
    diff_service: SyncDiffService = field(default_factory=SyncDiffService)

    def handle(self, timeline: Timeline, intent: TimelineIntent) -> TimelinePresentation:
        if isinstance(intent, SelectLayer):
            timeline.selection.selected_layer_id = intent.layer_id
            timeline.selection.selected_take_id = None
            timeline.selection.selected_event_ids = []

        elif isinstance(intent, SelectTake):
            self._handle_select_take(timeline, intent.layer_id, intent.take_id)

        elif isinstance(intent, SelectEvent):
            self._handle_select_event(
                timeline,
                layer_id=intent.layer_id,
                take_id=intent.take_id,
                event_id=intent.event_id,
                mode=intent.mode,
            )

        elif isinstance(intent, ClearSelection):
            timeline.selection.selected_take_id = None
            timeline.selection.selected_event_ids = []

        elif isinstance(intent, SelectAllEvents):
            self._handle_select_all_events(timeline)

        elif isinstance(intent, MoveSelectedEvents):
            self._handle_move_selected_events(
                timeline,
                delta_seconds=float(intent.delta_seconds),
                target_layer_id=intent.target_layer_id,
            )

        elif isinstance(intent, ToggleLayerExpanded):
            layer = self._find_layer(timeline, intent.layer_id)
            layer.presentation_hints.expanded = not layer.presentation_hints.expanded

        elif isinstance(intent, TriggerTakeAction):
            self._handle_trigger_take_action(
                timeline,
                layer_id=intent.layer_id,
                take_id=intent.take_id,
                action_id=intent.action_id,
            )

        elif isinstance(intent, NudgeSelectedEvents):
            self._handle_nudge_selected_events(
                timeline,
                direction=intent.direction,
                steps=intent.steps,
            )

        elif isinstance(intent, DuplicateSelectedEvents):
            self._handle_duplicate_selected_events(timeline, steps=intent.steps)

        elif isinstance(intent, Play):
            self.transport_service.play()

        elif isinstance(intent, Pause):
            self.transport_service.pause()

        elif isinstance(intent, Stop):
            self.transport_service.stop()

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
            session = self.session_service.get_session()
            try:
                sync_state = self.sync_service.connect()
            except Exception:
                # Keep session state in lockstep with sync service error state.
                session.sync_state = self.sync_service.get_state()
                raise
            self._pause_armed_write_layers_on_reconnect(timeline)
            session.sync_state = sync_state

        elif isinstance(intent, DisableSync):
            session = self.session_service.get_session()
            session.sync_state = self.sync_service.disconnect()

        elif isinstance(intent, EnableExperimentalLiveSync):
            session = self.session_service.get_session()
            session.sync_state.experimental_live_sync_enabled = True

        elif isinstance(intent, DisableExperimentalLiveSync):
            session = self.session_service.get_session()
            session.sync_state.experimental_live_sync_enabled = False
            self._reset_live_sync_guardrails(timeline)

        elif isinstance(intent, SetLayerLiveSyncState):
            session = self.session_service.get_session()
            if (
                not session.sync_state.experimental_live_sync_enabled
                and intent.live_sync_state is not LiveSyncState.OFF
            ):
                raise ValueError(
                    "SetLayerLiveSyncState requires experimental live sync to be enabled for non-off states"
                )
            layer = self._find_layer(timeline, intent.layer_id)
            layer.sync.live_sync_state = intent.live_sync_state
            if intent.live_sync_state is LiveSyncState.OFF:
                layer.sync.live_sync_pause_reason = None

        elif isinstance(intent, SetLayerLiveSyncPauseReason):
            layer = self._find_layer(timeline, intent.layer_id)
            layer.sync.live_sync_state = LiveSyncState.PAUSED
            layer.sync.live_sync_pause_reason = intent.pause_reason

        elif isinstance(intent, ClearLayerLiveSyncPauseReason):
            layer = self._find_layer(timeline, intent.layer_id)
            layer.sync.live_sync_pause_reason = None

        elif isinstance(intent, OpenPushToMA3Dialog):
            session = self.session_service.get_session()
            session.manual_push_flow.dialog_open = True
            session.manual_push_flow.selected_event_ids = list(intent.selection_event_ids)
            session.manual_push_flow.target_track_coord = None
            session.manual_push_flow.diff_gate_open = False
            session.manual_push_flow.diff_preview = None
            session.manual_push_flow.available_tracks = self._load_manual_push_track_options()

        elif isinstance(intent, SetPushTrackOptions):
            session = self.session_service.get_session()
            session.manual_push_flow.available_tracks = list(intent.tracks)

        elif isinstance(intent, SelectPushTargetTrack):
            session = self.session_service.get_session()
            available_coords = {
                track.coord
                for track in session.manual_push_flow.available_tracks
            }
            if intent.target_track_coord not in available_coords:
                raise ValueError(
                    f"SelectPushTargetTrack target_track_coord not found in available_tracks: "
                    f"{intent.target_track_coord}"
                )
            session.manual_push_flow.target_track_coord = intent.target_track_coord

        elif isinstance(intent, ConfirmPushToMA3):
            session = self.session_service.get_session()
            target_track = self._manual_push_track_by_coord(
                session.manual_push_flow.available_tracks,
                intent.target_track_coord,
            )
            selected_events = self._selected_events_by_ids(timeline, intent.selected_event_ids)
            diff_summary, diff_rows = self.diff_service.build_push_preview_rows(
                selected_events=selected_events,
                target_track_name=target_track.name,
                target_track_coord=target_track.coord,
            )
            session.manual_push_flow.dialog_open = False
            session.manual_push_flow.selected_event_ids = list(intent.selected_event_ids)
            session.manual_push_flow.target_track_coord = intent.target_track_coord
            session.manual_push_flow.diff_gate_open = True
            session.manual_push_flow.diff_preview = ManualPushDiffPreview(
                selected_count=len(intent.selected_event_ids),
                target_track_coord=target_track.coord,
                target_track_name=target_track.name,
                target_track_note=target_track.note,
                target_track_event_count=target_track.event_count,
                diff_summary=diff_summary,
                diff_rows=diff_rows,
            )

        elif isinstance(intent, OpenPullFromMA3Dialog):
            session = self.session_service.get_session()
            session.manual_pull_flow.dialog_open = True
            session.manual_pull_flow.available_tracks = self._load_manual_pull_track_options()
            session.manual_pull_flow.source_track_coord = None
            session.manual_pull_flow.available_events = []
            session.manual_pull_flow.selected_ma3_event_ids = []
            session.manual_pull_flow.available_target_layers = self._load_manual_pull_target_options(timeline)
            session.manual_pull_flow.target_layer_id = None
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None

        elif isinstance(intent, SetPullTrackOptions):
            session = self.session_service.get_session()
            session.manual_pull_flow.available_tracks = list(intent.tracks)

        elif isinstance(intent, SelectPullSourceTrack):
            session = self.session_service.get_session()
            source_track = self._manual_pull_track_by_coord(
                session.manual_pull_flow.available_tracks,
                intent.source_track_coord,
                action_name="SelectPullSourceTrack",
            )
            session.manual_pull_flow.source_track_coord = source_track.coord
            session.manual_pull_flow.available_events = self._load_manual_pull_event_options(source_track.coord)
            session.manual_pull_flow.selected_ma3_event_ids = []
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None

        elif isinstance(intent, SetPullSourceEvents):
            session = self.session_service.get_session()
            session.manual_pull_flow.available_events = list(intent.events)

        elif isinstance(intent, SelectPullSourceEvents):
            session = self.session_service.get_session()
            available_event_ids = {
                event.event_id
                for event in session.manual_pull_flow.available_events
            }
            unknown_event_ids = [
                event_id
                for event_id in intent.selected_ma3_event_ids
                if event_id not in available_event_ids
            ]
            if unknown_event_ids:
                raise ValueError(
                    "SelectPullSourceEvents selected_ma3_event_ids not found in available_events: "
                    + ", ".join(unknown_event_ids)
                )
            session.manual_pull_flow.selected_ma3_event_ids = list(intent.selected_ma3_event_ids)
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None

        elif isinstance(intent, SelectPullTargetLayer):
            session = self.session_service.get_session()
            target_layer = self._manual_pull_target_layer_by_id(
                session.manual_pull_flow.available_target_layers,
                intent.target_layer_id,
                action_name="SelectPullTargetLayer",
            )
            session.manual_pull_flow.target_layer_id = target_layer.layer_id
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None

        elif isinstance(intent, ConfirmPullFromMA3):
            session = self.session_service.get_session()
            source_track = self._manual_pull_track_by_coord(
                session.manual_pull_flow.available_tracks,
                intent.source_track_coord,
                action_name="ConfirmPullFromMA3",
            )
            target_layer = self._manual_pull_target_layer_by_id(
                session.manual_pull_flow.available_target_layers,
                intent.target_layer_id,
                action_name="ConfirmPullFromMA3",
            )
            available_event_ids = {
                event.event_id
                for event in session.manual_pull_flow.available_events
            }
            unknown_event_ids = [
                event_id
                for event_id in intent.selected_ma3_event_ids
                if event_id not in available_event_ids
            ]
            if unknown_event_ids:
                raise ValueError(
                    "ConfirmPullFromMA3 selected_ma3_event_ids not found in available_events: "
                    + ", ".join(unknown_event_ids)
                )
            selected_events = self._manual_pull_selected_events_by_ids(
                available_events=session.manual_pull_flow.available_events,
                selected_ids=intent.selected_ma3_event_ids,
                action_name="ConfirmPullFromMA3",
            )
            diff_summary, diff_rows = self.diff_service.build_pull_preview_rows(
                selected_events=selected_events,
                target_layer_name=target_layer.name,
            )

            session.manual_pull_flow.dialog_open = False
            session.manual_pull_flow.source_track_coord = source_track.coord
            session.manual_pull_flow.selected_ma3_event_ids = list(intent.selected_ma3_event_ids)
            session.manual_pull_flow.target_layer_id = target_layer.layer_id
            session.manual_pull_flow.diff_gate_open = True
            session.manual_pull_flow.diff_preview = ManualPullDiffPreview(
                selected_count=len(intent.selected_ma3_event_ids),
                source_track_coord=source_track.coord,
                source_track_name=source_track.name,
                source_track_note=source_track.note,
                source_track_event_count=source_track.event_count,
                target_layer_id=target_layer.layer_id,
                target_layer_name=target_layer.name,
                import_mode=intent.import_mode,
                diff_summary=diff_summary,
                diff_rows=diff_rows,
            )

        elif isinstance(intent, ApplyPullFromMA3):
            session = self.session_service.get_session()
            flow = session.manual_pull_flow
            preview = flow.diff_preview
            if not flow.diff_gate_open or preview is None:
                raise ValueError("ApplyPullFromMA3 requires an open diff preview")
            if flow.target_layer_id is None:
                raise ValueError("ApplyPullFromMA3 requires a selected target_layer_id")

            target_layer = self._find_layer(timeline, flow.target_layer_id)
            source_track = self._manual_pull_track_by_coord(
                flow.available_tracks,
                flow.source_track_coord or preview.source_track_coord,
                action_name="ApplyPullFromMA3",
            )
            selected_events = self._manual_pull_selected_events(flow)

            imported_take = self._build_manual_pull_take(
                layer=target_layer,
                source_track=source_track,
                selected_events=selected_events,
            )
            target_layer.takes.append(imported_take)
            self._sort_take_events(imported_take)

            timeline.selection.selected_layer_id = target_layer.id
            timeline.selection.selected_take_id = imported_take.id
            timeline.selection.selected_event_ids = [event.id for event in imported_take.events]

            flow.diff_gate_open = False
            flow.diff_preview = None

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
        # Selection only. Selecting a take must never change timeline truth.
        self._find_layer(timeline, layer_id)
        timeline.selection.selected_layer_id = layer_id
        timeline.selection.selected_take_id = take_id
        timeline.selection.selected_event_ids = []

    def _handle_select_event(self, timeline: Timeline, layer_id, take_id, event_id, mode: str) -> None:
        layer = self._find_layer(timeline, layer_id)
        if event_id is None:
            timeline.selection.selected_layer_id = layer.id
            timeline.selection.selected_take_id = take_id
            timeline.selection.selected_event_ids = []
            return

        mode_normalized = (mode or "replace").strip().lower()
        selected_ids = list(timeline.selection.selected_event_ids)

        if mode_normalized == "replace":
            selected_ids = [event_id]
        elif mode_normalized == "additive":
            if event_id not in selected_ids:
                selected_ids.append(event_id)
        elif mode_normalized == "toggle":
            if event_id in selected_ids:
                selected_ids = [selected_id for selected_id in selected_ids if selected_id != event_id]
            else:
                selected_ids.append(event_id)
        else:
            raise ValueError(f"Unsupported selection mode: {mode}")

        timeline.selection.selected_layer_id = layer.id
        timeline.selection.selected_event_ids = selected_ids
        timeline.selection.selected_take_id = take_id if selected_ids else None

    def _handle_select_all_events(self, timeline: Timeline) -> None:
        selected_layer_id = timeline.selection.selected_layer_id
        target_layers: list[Layer]
        if selected_layer_id is not None:
            layer = self._find_layer(timeline, selected_layer_id)
            target_layers = [layer]
        else:
            target_layers = [layer for layer in timeline.layers if layer.presentation_hints.visible and not layer.presentation_hints.locked]

        selected_event_ids: list = []
        selected_take_id = None
        for layer in target_layers:
            if not layer.presentation_hints.visible or layer.presentation_hints.locked:
                continue
            for take in layer.takes:
                if take.events and selected_take_id is None:
                    selected_take_id = take.id
                selected_event_ids.extend(event.id for event in take.events)

        timeline.selection.selected_event_ids = selected_event_ids
        timeline.selection.selected_take_id = selected_take_id

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

        timeline.selection.selected_layer_id = layer.id
        timeline.selection.selected_take_id = main_take.id
        timeline.selection.selected_event_ids = []

    def _handle_move_selected_events(self, timeline: Timeline, delta_seconds: float, target_layer_id) -> None:
        selected_ids = list(timeline.selection.selected_event_ids)
        if not selected_ids:
            return

        records = self._selected_event_records(timeline, selected_ids)
        if not records:
            timeline.selection.selected_event_ids = []
            timeline.selection.selected_take_id = None
            return

        applied_delta = max(float(delta_seconds), -min(record.event.start for record in records))
        source_layer_ids = {record.layer.id for record in records}
        source_layer_id = timeline.selection.selected_layer_id
        transfer_target = None

        if target_layer_id is not None and target_layer_id not in source_layer_ids:
            transfer_target = self._find_layer(timeline, target_layer_id)
            if (
                transfer_target.kind != records[0].layer.kind
                or transfer_target.kind.value != "event"
                or transfer_target.presentation_hints.locked
                or not transfer_target.presentation_hints.visible
            ):
                return

        affected_takes: dict[object, Take] = {}
        if transfer_target is None:
            for record in records:
                record.event.start += applied_delta
                record.event.end += applied_delta
                affected_takes[record.take.id] = record.take
            self._sort_take_events(*affected_takes.values())
            if source_layer_id is None and records:
                source_layer_id = records[0].layer.id
            timeline.selection.selected_layer_id = source_layer_id
            timeline.selection.selected_take_id = self._resolve_selected_take_id(
                self._find_layer(timeline, source_layer_id) if source_layer_id is not None else records[0].layer,
                selected_ids,
                fallback_take_id=timeline.selection.selected_take_id,
            )
            return

        target_take = self._main_take(transfer_target)
        if target_take is None:
            return

        for record in records:
            record.take.events = [candidate for candidate in record.take.events if candidate.id != record.event.id]
            affected_takes[record.take.id] = record.take
            record.event.start += applied_delta
            record.event.end += applied_delta
            record.event.take_id = target_take.id
            target_take.events.append(record.event)

        affected_takes[target_take.id] = target_take
        self._sort_take_events(*affected_takes.values())
        timeline.selection.selected_layer_id = transfer_target.id
        timeline.selection.selected_take_id = target_take.id
        timeline.selection.selected_event_ids = [record.event.id for record in records]

    @dataclass(slots=True)
    class _EventRecord:
        layer: Layer
        take: Take
        event: Event

    def _selected_event_records(self, timeline: Timeline, selected_ids: list) -> list[_EventRecord]:
        selected_lookup = set(selected_ids)
        order = {event_id: idx for idx, event_id in enumerate(selected_ids)}
        records: list[TimelineOrchestrator._EventRecord] = []
        for layer in timeline.layers:
            for take in layer.takes:
                for event in take.events:
                    if event.id in selected_lookup:
                        records.append(self._EventRecord(layer=layer, take=take, event=event))
        records.sort(key=lambda record: order.get(record.event.id, len(order)))
        return records

    @staticmethod
    def _resolve_selected_take_id(layer: Layer, selected_ids: list, fallback_take_id=None):
        selected_lookup = set(selected_ids)
        if fallback_take_id is not None:
            fallback = TimelineOrchestrator._find_take(layer, fallback_take_id)
            if fallback is not None and any(event.id in selected_lookup for event in fallback.events):
                return fallback_take_id

        for take in layer.takes:
            if any(event.id in selected_lookup for event in take.events):
                return take.id
        return None

    @staticmethod
    def _sort_take_events(*takes: Take) -> None:
        for take in takes:
            take.events = sorted(take.events, key=lambda event: (event.start, event.end, str(event.id)))

    def _handle_nudge_selected_events(self, timeline: Timeline, direction: int, steps: int) -> None:
        if direction == 0 or steps <= 0:
            return

        selected = self._selected_events(timeline)
        if not selected:
            return

        delta = float(direction) * float(steps) * _KEYBOARD_STEP_SECONDS
        for _layer, take, event in selected:
            duration = event.duration
            next_start = max(0.0, event.start + delta)
            event.start = next_start
            event.end = next_start + duration
            take.events = self._sorted_events(take.events)

    def _handle_duplicate_selected_events(self, timeline: Timeline, steps: int) -> None:
        if steps <= 0:
            return

        selected = self._selected_events(timeline)
        if not selected:
            return

        delta = float(steps) * _KEYBOARD_STEP_SECONDS
        existing_ids = self._all_event_ids(timeline)
        duplicated_ids: list = []
        selected_layer_id = timeline.selection.selected_layer_id
        selected_take_id = timeline.selection.selected_take_id

        for layer, take, event in selected:
            duplicate_id = self._next_duplicate_event_id(take, event, existing_ids)
            duplicate = Event(
                id=duplicate_id,
                take_id=take.id,
                start=event.start + delta,
                end=event.end + delta,
                payload_ref=event.payload_ref,
                label=event.label,
                color=event.color,
                muted=event.muted,
            )
            take.events = self._sorted_events([*take.events, duplicate])
            existing_ids.add(str(duplicate.id))
            duplicated_ids.append(duplicate.id)
            selected_layer_id = layer.id
            selected_take_id = take.id

        timeline.selection.selected_layer_id = selected_layer_id
        timeline.selection.selected_take_id = selected_take_id
        timeline.selection.selected_event_ids = duplicated_ids

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

    @staticmethod
    def _sorted_events(events: list[Event]) -> list[Event]:
        return sorted(events, key=lambda event: (event.start, event.end, str(event.id)))

    @staticmethod
    def _all_event_ids(timeline: Timeline) -> set[str]:
        return {
            str(event.id)
            for layer in timeline.layers
            for take in layer.takes
            for event in take.events
        }

    @staticmethod
    def _next_duplicate_event_id(take: Take, event: Event, existing_ids: set[str]):
        index = 1
        while True:
            candidate = f"{take.id}:dup:{event.id}:{index}"
            if candidate not in existing_ids:
                return candidate
            index += 1

    def _selected_events(self, timeline: Timeline) -> list[tuple[Layer, Take, Event]]:
        selected_ids = set(timeline.selection.selected_event_ids)
        if not selected_ids:
            return []

        selected_order = {
            str(event_id): index for index, event_id in enumerate(timeline.selection.selected_event_ids)
        }
        selected: list[tuple[Layer, Take, Event]] = []
        for layer in timeline.layers:
            for take in layer.takes:
                for event in take.events:
                    if event.id in selected_ids:
                        selected.append((layer, take, event))

        selected.sort(key=lambda item: selected_order[str(item[2].id)])
        return selected

    @staticmethod
    def _selected_events_by_ids(timeline: Timeline, selected_ids: list) -> list[Event]:
        selected_lookup = set(selected_ids)
        selected_order = {
            str(event_id): index for index, event_id in enumerate(selected_ids)
        }
        selected: list[Event] = []
        for layer in timeline.layers:
            for take in layer.takes:
                for event in take.events:
                    if event.id in selected_lookup:
                        selected.append(event)

        selected.sort(key=lambda event: selected_order[str(event.id)])
        return selected

    @staticmethod
    def _manual_pull_selected_events(flow) -> list[ManualPullEventOption]:
        return TimelineOrchestrator._manual_pull_selected_events_by_ids(
            available_events=flow.available_events,
            selected_ids=flow.selected_ma3_event_ids,
            action_name="ApplyPullFromMA3",
        )

    @staticmethod
    def _manual_pull_selected_events_by_ids(
        *,
        available_events: list[ManualPullEventOption],
        selected_ids: list[str],
        action_name: str,
    ) -> list[ManualPullEventOption]:
        available_by_id = {
            event.event_id: event
            for event in available_events
        }
        selected_events: list[ManualPullEventOption] = []
        missing_event_ids: list[str] = []
        for event_id in selected_ids:
            selected_event = available_by_id.get(event_id)
            if selected_event is None:
                missing_event_ids.append(event_id)
                continue
            selected_events.append(selected_event)
        if missing_event_ids:
            raise ValueError(
                f"{action_name} selected_ma3_event_ids not found in available_events: "
                + ", ".join(missing_event_ids)
            )
        return selected_events

    def _build_manual_pull_take(
        self,
        *,
        layer: Layer,
        source_track: ManualPullTrackOption,
        selected_events: list[ManualPullEventOption],
    ) -> Take:
        take_id = self._next_manual_pull_take_id(layer)
        imported_events = [
            self._build_manual_pull_event(
                take_id=take_id,
                source_track=source_track,
                source_event=source_event,
                order_index=index,
            )
            for index, source_event in enumerate(selected_events, start=1)
        ]
        return Take(
            id=take_id,
            layer_id=layer.id,
            name=f"MA3 Pull - {source_track.name}",
            events=imported_events,
            source_ref=source_track.coord,
        )

    def _build_manual_pull_event(
        self,
        *,
        take_id,
        source_track: ManualPullTrackOption,
        source_event: ManualPullEventOption,
        order_index: int,
    ) -> Event:
        start, end = self.diff_service.resolve_pull_event_range(source_event, order_index=order_index)
        return Event(
            id=f"{take_id}:ma3:{source_track.coord}:{source_event.event_id}:{order_index}",
            take_id=take_id,
            start=start,
            end=end,
            label=source_event.label,
            payload_ref=source_event.event_id,
        )

    @staticmethod
    def _next_manual_pull_take_id(layer: Layer):
        existing_ids = {str(take.id) for take in layer.takes}
        index = 1
        while True:
            candidate = f"{layer.id}:ma3_pull:{index}"
            if candidate not in existing_ids:
                return candidate
            index += 1

    def _load_manual_push_track_options(self) -> list[ManualPushTrackOption]:
        provider = self.sync_service
        if hasattr(provider, "list_push_track_options"):
            raw_tracks = provider.list_push_track_options()
        elif hasattr(provider, "get_available_ma3_tracks"):
            raw_tracks = provider.get_available_ma3_tracks()
        else:
            return []

        return [
            self._normalize_manual_push_track_option(raw_track)
            for raw_track in raw_tracks or []
        ]

    def _load_manual_pull_track_options(self) -> list[ManualPullTrackOption]:
        provider = self.sync_service
        if hasattr(provider, "list_pull_track_options"):
            raw_tracks = provider.list_pull_track_options()
        elif hasattr(provider, "get_available_ma3_tracks"):
            raw_tracks = provider.get_available_ma3_tracks()
        else:
            return []

        return [
            self._normalize_manual_pull_track_option(raw_track)
            for raw_track in raw_tracks or []
        ]

    def _load_manual_pull_event_options(self, source_track_coord: str) -> list[ManualPullEventOption]:
        provider = self.sync_service
        if hasattr(provider, "list_pull_source_events"):
            raw_events = provider.list_pull_source_events(source_track_coord)
        elif hasattr(provider, "list_ma3_track_events"):
            raw_events = provider.list_ma3_track_events(source_track_coord)
        elif hasattr(provider, "get_available_ma3_events"):
            raw_events = provider.get_available_ma3_events(source_track_coord)
        else:
            return []

        return [
            self._normalize_manual_pull_event_option(raw_event)
            for raw_event in raw_events or []
        ]

    @staticmethod
    def _load_manual_pull_target_options(timeline: Timeline) -> list[ManualPullTargetOption]:
        return [
            ManualPullTargetOption(layer_id=layer.id, name=layer.name)
            for layer in sorted(timeline.layers, key=lambda value: value.order_index)
            if layer.kind == LayerKind.EVENT
            and layer.presentation_hints.visible
            and not layer.presentation_hints.locked
        ]

    @staticmethod
    def _normalize_manual_push_track_option(raw_track: Any) -> ManualPushTrackOption:
        if isinstance(raw_track, ManualPushTrackOption):
            return raw_track

        coord = TimelineOrchestrator._track_option_value(raw_track, "coord")
        name = TimelineOrchestrator._track_option_value(raw_track, "name")
        note = TimelineOrchestrator._track_option_value(raw_track, "note")
        event_count = TimelineOrchestrator._track_option_value(raw_track, "event_count")

        return ManualPushTrackOption(
            coord=str(coord or ""),
            name=str(name or ""),
            note=None if note is None else str(note),
            event_count=TimelineOrchestrator._coerce_optional_int(event_count),
        )

    @staticmethod
    def _normalize_manual_pull_track_option(raw_track: Any) -> ManualPullTrackOption:
        if isinstance(raw_track, ManualPullTrackOption):
            return raw_track

        coord = TimelineOrchestrator._track_option_value(raw_track, "coord")
        name = TimelineOrchestrator._track_option_value(raw_track, "name")
        note = TimelineOrchestrator._track_option_value(raw_track, "note")
        event_count = TimelineOrchestrator._track_option_value(raw_track, "event_count")

        return ManualPullTrackOption(
            coord=str(coord or ""),
            name=str(name or ""),
            note=None if note is None else str(note),
            event_count=TimelineOrchestrator._coerce_optional_int(event_count),
        )

    @staticmethod
    def _normalize_manual_pull_event_option(raw_event: Any) -> ManualPullEventOption:
        if isinstance(raw_event, ManualPullEventOption):
            return raw_event

        event_id = TimelineOrchestrator._track_option_value(raw_event, "event_id")
        label = TimelineOrchestrator._track_option_value(raw_event, "label")
        start = TimelineOrchestrator._track_option_value(raw_event, "start")
        end = TimelineOrchestrator._track_option_value(raw_event, "end")

        return ManualPullEventOption(
            event_id=str(event_id or ""),
            label=str(label or event_id or ""),
            start=None if start is None else float(start),
            end=None if end is None else float(end),
        )

    @staticmethod
    def _track_option_value(raw_track: Any, key: str) -> Any:
        if isinstance(raw_track, dict):
            return raw_track.get(key)
        return getattr(raw_track, key, None)

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    @staticmethod
    def _manual_push_track_by_coord(
        available_tracks: list[ManualPushTrackOption],
        target_track_coord: str,
    ) -> ManualPushTrackOption:
        for track in available_tracks:
            if track.coord == target_track_coord:
                return track
        raise ValueError(
            f"ConfirmPushToMA3 target_track_coord not found in available_tracks: "
            f"{target_track_coord}"
        )

    @staticmethod
    def _manual_pull_track_by_coord(
        available_tracks: list[ManualPullTrackOption],
        source_track_coord: str,
        action_name: str,
    ) -> ManualPullTrackOption:
        for track in available_tracks:
            if track.coord == source_track_coord:
                return track
        raise ValueError(
            f"{action_name} source_track_coord not found in available_tracks: "
            f"{source_track_coord}"
        )

    @staticmethod
    def _manual_pull_target_layer_by_id(
        available_targets: list[ManualPullTargetOption],
        target_layer_id,
        action_name: str,
    ) -> ManualPullTargetOption:
        for target in available_targets:
            if target.layer_id == target_layer_id:
                return target
        raise ValueError(
            f"{action_name} target_layer_id not found in available_target_layers: "
            f"{target_layer_id}"
        )

    def _find_layer(self, timeline: Timeline, layer_id):
        for layer in timeline.layers:
            if layer.id == layer_id:
                return layer
        raise ValueError(f"Layer not found: {layer_id}")

    @staticmethod
    def _reset_live_sync_guardrails(timeline: Timeline) -> None:
        for layer in timeline.layers:
            layer.sync.live_sync_state = LiveSyncState.OFF
            layer.sync.live_sync_pause_reason = None
            layer.sync.live_sync_divergent = False

    @staticmethod
    def _pause_armed_write_layers_on_reconnect(timeline: Timeline) -> None:
        for layer in timeline.layers:
            if layer.sync.live_sync_state is LiveSyncState.ARMED_WRITE:
                layer.sync.live_sync_state = LiveSyncState.PAUSED
                layer.sync.live_sync_pause_reason = _RECONNECT_REARM_REQUIRED_REASON
