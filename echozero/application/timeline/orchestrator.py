"""Timeline orchestration for the new EchoZero application layer."""

import inspect
import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from echozero.application.mixer.service import MixerService
from echozero.application.playback.service import PlaybackService
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import (
    BatchTransferPlanRowState,
    BatchTransferPlanState,
    ManualPullDiffPreview,
    ManualPullEventOption,
    ManualPullTargetOption,
    ManualPullTrackOption,
    ManualPushDiffPreview,
    ManualPushTrackOption,
    TransferPresetState,
)
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import LayerId, TakeId
from echozero.application.sync.diff_service import SyncDiffService
from echozero.application.sync.models import LiveSyncState
from echozero.application.sync.service import SyncService
if TYPE_CHECKING:
    from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    ApplyTransferPreset,
    ApplyTransferPlan,
    CancelTransferPlan,
    ClearLayerLiveSyncPauseReason,
    ClearSelection,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    DeleteTransferPreset,
    DisableExperimentalLiveSync,
    DuplicateSelectedEvents,
    DisableSync,
    EnableExperimentalLiveSync,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    ExitPullFromMA3Workspace,
    OpenPullFromMA3Dialog,
    SelectAllEvents,
    SelectEvent,
    SelectLayer,
    SelectPullSourceEvents,
    SelectPullSourceTracks,
    SelectPullSourceTrack,
    SelectPullTargetLayer,
    SelectTake,
    EnableSync,
    ExitPushToMA3Mode,
    OpenPushToMA3Dialog,
    Pause,
    Play,
    PreviewTransferPlan,
    SaveTransferPreset,
    Seek,
    SetPullImportMode,
    SetPullSourceEvents,
    SetPullTrackOptions,
    SetPushTransferMode,
    SelectPushTargetTrack,
    SetGain,
    SetActivePlaybackTarget,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    SetPushTrackOptions,
    Stop,
    TimelineIntent,
    ToggleLayerExpanded,
    TriggerTakeAction,
)
from echozero.application.timeline.models import Timeline, Layer, Take, Event
from echozero.application.transport.service import TransportService

_KEYBOARD_STEP_SECONDS = 1.0 / 30.0
_RECONNECT_REARM_REQUIRED_REASON = "Live sync reconnected; explicit re-arm required"
_PULL_TARGET_CREATE_NEW_LAYER_ID = LayerId("__manual_pull__:create_new_layer")
_PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_ID = LayerId("__manual_pull__:create_new_layer_per_source_track")
_PULL_TARGET_CREATE_NEW_LAYER_NAME = "+ Create New Layer..."
_PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_NAME = "+ Create New Layer Per Source Track..."


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
            self._handle_select_layer(timeline, intent.layer_id, mode=intent.mode)

        elif isinstance(intent, SelectTake):
            self._handle_select_take(timeline, intent.layer_id, intent.take_id)

        elif isinstance(intent, SetActivePlaybackTarget):
            self._handle_set_active_playback_target(
                timeline,
                layer_id=intent.layer_id,
                take_id=intent.take_id,
            )

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
            timeline.selection.selected_event_ids = list(intent.selection_event_ids)
            session.manual_push_flow.selected_layer_ids = self._selected_layer_scope(timeline)
            session.manual_push_flow.dialog_open = False
            session.manual_push_flow.push_mode_active = True
            session.manual_push_flow.selected_event_ids = list(intent.selection_event_ids)
            session.manual_push_flow.target_track_coord = None
            session.manual_push_flow.transfer_mode = "merge"
            session.manual_push_flow.diff_gate_open = False
            session.manual_push_flow.diff_preview = None
            session.manual_push_flow.available_tracks = self._load_manual_push_track_options()
            self._rebuild_push_transfer_plan(timeline, session)

        elif isinstance(intent, ExitPushToMA3Mode):
            session = self.session_service.get_session()
            session.manual_push_flow.dialog_open = False
            session.manual_push_flow.push_mode_active = False
            session.manual_push_flow.selected_layer_ids = []
            session.manual_push_flow.selected_event_ids = []
            session.manual_push_flow.target_track_coord = None
            session.manual_push_flow.transfer_mode = "merge"
            session.manual_push_flow.diff_gate_open = False
            session.manual_push_flow.diff_preview = None
            session.batch_transfer_plan = None

        elif isinstance(intent, SetPushTrackOptions):
            session = self.session_service.get_session()
            session.manual_push_flow.available_tracks = list(intent.tracks)
            if session.manual_push_flow.push_mode_active:
                self._rebuild_push_transfer_plan(timeline, session)

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
            if intent.layer_id is not None:
                layer = self._find_layer(timeline, intent.layer_id)
                layer.sync.ma3_track_coord = intent.target_track_coord
            elif timeline.selection.selected_layer_id is not None and session.manual_push_flow.push_mode_active:
                layer = self._find_layer(timeline, timeline.selection.selected_layer_id)
                layer.sync.ma3_track_coord = intent.target_track_coord
            if session.manual_push_flow.push_mode_active:
                self._rebuild_push_transfer_plan(timeline, session)

        elif isinstance(intent, SetPushTransferMode):
            session = self.session_service.get_session()
            session.manual_push_flow.transfer_mode = intent.mode

        elif isinstance(intent, ConfirmPushToMA3):
            session = self.session_service.get_session()
            target_track = self._manual_push_track_by_coord(
                session.manual_push_flow.available_tracks,
                intent.target_track_coord,
            )
            selected_events = self._selected_events_by_ids(timeline, intent.selected_event_ids)
            selected_event_ids = [event.id for event in selected_events]
            selected_event_lookup = set(selected_event_ids)
            if session.manual_push_flow.push_mode_active:
                for layer_id in self._selected_layer_scope(timeline):
                    layer = self._find_layer(timeline, layer_id)
                    main_take = self._main_take(layer)
                    if main_take is not None and any(event.id in selected_event_lookup for event in main_take.events):
                        layer.sync.ma3_track_coord = target_track.coord
            diff_summary, diff_rows = self.diff_service.build_push_preview_rows(
                selected_events=selected_events,
                target_track_name=target_track.name,
                target_track_coord=target_track.coord,
            )
            session.manual_push_flow.dialog_open = False
            session.manual_push_flow.selected_event_ids = selected_event_ids
            session.manual_push_flow.target_track_coord = intent.target_track_coord
            session.manual_push_flow.diff_gate_open = True
            session.manual_push_flow.diff_preview = ManualPushDiffPreview(
                selected_count=len(selected_event_ids),
                target_track_coord=target_track.coord,
                target_track_name=target_track.name,
                target_track_note=target_track.note,
                target_track_event_count=target_track.event_count,
                diff_summary=diff_summary,
                diff_rows=diff_rows,
            )
            if session.manual_push_flow.push_mode_active:
                self._rebuild_push_transfer_plan(timeline, session)

        elif isinstance(intent, OpenPullFromMA3Dialog):
            session = self.session_service.get_session()
            session.manual_pull_flow.dialog_open = False
            session.manual_pull_flow.workspace_active = True
            session.manual_pull_flow.available_tracks = self._load_manual_pull_track_options()
            session.manual_pull_flow.selected_source_track_coords = []
            session.manual_pull_flow.active_source_track_coord = None
            session.manual_pull_flow.source_track_coord = None
            session.manual_pull_flow.available_events = []
            session.manual_pull_flow.selected_ma3_event_ids = []
            session.manual_pull_flow.selected_ma3_event_ids_by_track = {}
            session.manual_pull_flow.import_mode = "new_take"
            session.manual_pull_flow.import_mode_by_source_track = {}
            session.manual_pull_flow.available_target_layers = self._load_manual_pull_target_options(timeline)
            session.manual_pull_flow.target_layer_id = None
            session.manual_pull_flow.target_layer_id_by_source_track = {}
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None
            self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, ExitPullFromMA3Workspace):
            session = self.session_service.get_session()
            session.manual_pull_flow.dialog_open = False
            session.manual_pull_flow.workspace_active = False
            session.manual_pull_flow.available_tracks = []
            session.manual_pull_flow.selected_source_track_coords = []
            session.manual_pull_flow.active_source_track_coord = None
            session.manual_pull_flow.source_track_coord = None
            session.manual_pull_flow.available_events = []
            session.manual_pull_flow.selected_ma3_event_ids = []
            session.manual_pull_flow.selected_ma3_event_ids_by_track = {}
            session.manual_pull_flow.import_mode = "new_take"
            session.manual_pull_flow.import_mode_by_source_track = {}
            session.manual_pull_flow.available_target_layers = []
            session.manual_pull_flow.target_layer_id = None
            session.manual_pull_flow.target_layer_id_by_source_track = {}
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None
            if session.batch_transfer_plan is not None and session.batch_transfer_plan.operation_type == "pull":
                session.batch_transfer_plan = None

        elif isinstance(intent, SetPullTrackOptions):
            session = self.session_service.get_session()
            session.manual_pull_flow.available_tracks = list(intent.tracks)
            if session.manual_pull_flow.workspace_active:
                selected_coords = {
                    track.coord
                    for track in session.manual_pull_flow.available_tracks
                }
                session.manual_pull_flow.selected_source_track_coords = [
                    coord
                    for coord in session.manual_pull_flow.selected_source_track_coords
                    if coord in selected_coords
                ]
                if session.manual_pull_flow.active_source_track_coord not in selected_coords:
                    session.manual_pull_flow.active_source_track_coord = None
                    session.manual_pull_flow.source_track_coord = None
                    session.manual_pull_flow.available_events = []
                    session.manual_pull_flow.selected_ma3_event_ids = []
                    session.manual_pull_flow.import_mode = "new_take"
                self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, SelectPullSourceTracks):
            session = self.session_service.get_session()
            available_by_coord = {
                track.coord: track
                for track in session.manual_pull_flow.available_tracks
            }
            ordered_coords: list[str] = []
            unknown_coords: list[str] = []
            for coord in intent.source_track_coords:
                if coord not in available_by_coord:
                    unknown_coords.append(coord)
                    continue
                if coord not in ordered_coords:
                    ordered_coords.append(coord)
            if unknown_coords:
                raise ValueError(
                    "SelectPullSourceTracks source_track_coords not found in available_tracks: "
                    + ", ".join(unknown_coords)
                )
            session.manual_pull_flow.selected_source_track_coords = ordered_coords
            if session.manual_pull_flow.active_source_track_coord not in ordered_coords:
                active_coord = ordered_coords[0]
                session.manual_pull_flow.active_source_track_coord = active_coord
                session.manual_pull_flow.source_track_coord = active_coord
                session.manual_pull_flow.available_events = self._load_manual_pull_event_options(active_coord)
                session.manual_pull_flow.selected_ma3_event_ids = list(
                    session.manual_pull_flow.selected_ma3_event_ids_by_track.get(active_coord, [])
                )
                session.manual_pull_flow.import_mode = session.manual_pull_flow.import_mode_by_source_track.get(
                    active_coord,
                    "new_take",
                )
                session.manual_pull_flow.target_layer_id = session.manual_pull_flow.target_layer_id_by_source_track.get(
                    active_coord
                )
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None
            self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, SelectPullSourceTrack):
            session = self.session_service.get_session()
            source_track = self._manual_pull_track_by_coord(
                session.manual_pull_flow.available_tracks,
                intent.source_track_coord,
                action_name="SelectPullSourceTrack",
            )
            if source_track.coord not in session.manual_pull_flow.selected_source_track_coords:
                session.manual_pull_flow.selected_source_track_coords = [
                    *session.manual_pull_flow.selected_source_track_coords,
                    source_track.coord,
                ]
            session.manual_pull_flow.active_source_track_coord = source_track.coord
            session.manual_pull_flow.source_track_coord = source_track.coord
            session.manual_pull_flow.available_events = self._load_manual_pull_event_options(source_track.coord)
            session.manual_pull_flow.selected_ma3_event_ids = list(
                session.manual_pull_flow.selected_ma3_event_ids_by_track.get(source_track.coord, [])
            )
            session.manual_pull_flow.import_mode = session.manual_pull_flow.import_mode_by_source_track.get(
                source_track.coord,
                "new_take",
            )
            session.manual_pull_flow.target_layer_id = session.manual_pull_flow.target_layer_id_by_source_track.get(
                source_track.coord
            )
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None
            if session.manual_pull_flow.workspace_active:
                self._rebuild_pull_transfer_plan(timeline, session)

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
            active_coord = session.manual_pull_flow.active_source_track_coord or session.manual_pull_flow.source_track_coord
            if active_coord:
                session.manual_pull_flow.selected_ma3_event_ids_by_track[active_coord] = list(
                    intent.selected_ma3_event_ids
                )
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None
            if session.manual_pull_flow.workspace_active:
                self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, SelectPullTargetLayer):
            session = self.session_service.get_session()
            target_layer = self._manual_pull_target_layer_by_id(
                session.manual_pull_flow.available_target_layers,
                intent.target_layer_id,
                action_name="SelectPullTargetLayer",
            )
            session.manual_pull_flow.target_layer_id = target_layer.layer_id
            active_coord = session.manual_pull_flow.active_source_track_coord or session.manual_pull_flow.source_track_coord
            if target_layer.layer_id == _PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_ID:
                for coord in session.manual_pull_flow.selected_source_track_coords:
                    session.manual_pull_flow.target_layer_id_by_source_track[coord] = target_layer.layer_id
            elif active_coord:
                session.manual_pull_flow.target_layer_id_by_source_track[active_coord] = target_layer.layer_id
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None
            if session.manual_pull_flow.workspace_active:
                self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, SetPullImportMode):
            session = self.session_service.get_session()
            session.manual_pull_flow.import_mode = intent.import_mode
            active_coord = session.manual_pull_flow.active_source_track_coord or session.manual_pull_flow.source_track_coord
            if active_coord:
                session.manual_pull_flow.import_mode_by_source_track[active_coord] = intent.import_mode
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None
            if session.manual_pull_flow.workspace_active:
                self._rebuild_pull_transfer_plan(timeline, session)

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
            session.manual_pull_flow.workspace_active = True
            if source_track.coord not in session.manual_pull_flow.selected_source_track_coords:
                session.manual_pull_flow.selected_source_track_coords = [
                    *session.manual_pull_flow.selected_source_track_coords,
                    source_track.coord,
                ]
            session.manual_pull_flow.active_source_track_coord = source_track.coord
            session.manual_pull_flow.source_track_coord = source_track.coord
            session.manual_pull_flow.selected_ma3_event_ids = list(intent.selected_ma3_event_ids)
            session.manual_pull_flow.selected_ma3_event_ids_by_track[source_track.coord] = list(
                intent.selected_ma3_event_ids
            )
            session.manual_pull_flow.import_mode = intent.import_mode
            session.manual_pull_flow.import_mode_by_source_track[source_track.coord] = intent.import_mode
            session.manual_pull_flow.target_layer_id = target_layer.layer_id
            session.manual_pull_flow.target_layer_id_by_source_track[source_track.coord] = target_layer.layer_id
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
            if session.manual_pull_flow.workspace_active:
                self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, ApplyPullFromMA3):
            session = self.session_service.get_session()
            flow = session.manual_pull_flow
            preview = flow.diff_preview
            if not flow.diff_gate_open or preview is None:
                raise ValueError("ApplyPullFromMA3 requires an open diff preview")
            if flow.target_layer_id is None:
                raise ValueError("ApplyPullFromMA3 requires a selected target_layer_id")

            source_track = self._manual_pull_track_by_coord(
                flow.available_tracks,
                flow.source_track_coord or preview.source_track_coord,
                action_name="ApplyPullFromMA3",
            )
            selected_events = self._manual_pull_selected_events(flow)
            target_layer = self._resolve_manual_pull_target_layer(
                timeline,
                target_layer_id=flow.target_layer_id,
                source_track=source_track,
            )
            selected_take_id, selected_event_ids = self._apply_manual_pull_import(
                target_layer=target_layer,
                source_track=source_track,
                selected_events=selected_events,
                import_mode=preview.import_mode,
            )

            timeline.selection.selected_layer_id = target_layer.id
            timeline.selection.selected_layer_ids = [target_layer.id]
            timeline.selection.selected_take_id = selected_take_id
            timeline.selection.selected_event_ids = selected_event_ids

            flow.diff_gate_open = False
            flow.diff_preview = None
            if flow.workspace_active:
                self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, SaveTransferPreset):
            session = self.session_service.get_session()
            self._save_transfer_preset(timeline, session, intent.name)

        elif isinstance(intent, ApplyTransferPreset):
            session = self.session_service.get_session()
            self._apply_transfer_preset(timeline, session, intent.preset_id)

        elif isinstance(intent, DeleteTransferPreset):
            session = self.session_service.get_session()
            self._delete_transfer_preset(session, intent.preset_id)

        elif isinstance(intent, PreviewTransferPlan):
            session = self.session_service.get_session()
            plan = self._require_active_transfer_plan(
                session,
                intent.plan_id,
                action_name="PreviewTransferPlan",
            )
            self._preview_transfer_plan(timeline, session, plan)

        elif isinstance(intent, ApplyTransferPlan):
            session = self.session_service.get_session()
            plan = self._require_active_transfer_plan(
                session,
                intent.plan_id,
                action_name="ApplyTransferPlan",
            )
            self._apply_transfer_plan(timeline, session, plan)

        elif isinstance(intent, CancelTransferPlan):
            session = self.session_service.get_session()
            plan = self._require_active_transfer_plan(
                session,
                intent.plan_id,
                action_name="CancelTransferPlan",
            )
            self._cancel_transfer_plan(session, plan)

        session = self.session_service.get_session()
        if session.manual_push_flow.push_mode_active:
            session.manual_push_flow.selected_layer_ids = self._selected_layer_scope(timeline)
            session.manual_push_flow.selected_event_ids = list(timeline.selection.selected_event_ids)
            if not self._plan_counters_locked(session.batch_transfer_plan):
                self._rebuild_push_transfer_plan(timeline, session)
        audibility = self.mixer_service.resolve_audibility(timeline.layers)
        self.playback_service.update_runtime(
            timeline=timeline,
            transport=session.transport_state,
            audibility=audibility,
            sync=session.sync_state,
        )
        return self.assembler.assemble(timeline=timeline, session=session)

    def _handle_select_layer(self, timeline: Timeline, layer_id, *, mode: str) -> None:
        if layer_id is None:
            timeline.selection.selected_layer_id = None
            timeline.selection.selected_layer_ids = []
            timeline.selection.selected_take_id = None
            timeline.selection.selected_event_ids = []
            return

        self._find_layer(timeline, layer_id)
        mode_normalized = (mode or "replace").strip().lower()
        current_ids = list(dict.fromkeys(timeline.selection.selected_layer_ids))
        if not current_ids and timeline.selection.selected_layer_id is not None:
            current_ids = [timeline.selection.selected_layer_id]

        if mode_normalized == "replace":
            next_ids = [layer_id]
        elif mode_normalized == "toggle":
            if layer_id in current_ids:
                next_ids = [candidate for candidate in current_ids if candidate != layer_id]
            else:
                next_ids = [*current_ids, layer_id]
        elif mode_normalized == "range":
            ordered_ids = [layer.id for layer in sorted(timeline.layers, key=lambda value: value.order_index)]
            anchor_id = current_ids[0] if current_ids else timeline.selection.selected_layer_id
            if anchor_id is None:
                anchor_id = layer_id
            if anchor_id not in ordered_ids:
                anchor_id = layer_id
            start = ordered_ids.index(anchor_id)
            end = ordered_ids.index(layer_id)
            low, high = sorted((start, end))
            next_ids = ordered_ids[low : high + 1]
        else:
            raise ValueError(f"Unsupported layer selection mode: {mode}")

        timeline.selection.selected_layer_id = layer_id if next_ids else None
        timeline.selection.selected_layer_ids = next_ids
        timeline.selection.selected_take_id = None
        timeline.selection.selected_event_ids = []

    def _handle_select_take(self, timeline: Timeline, layer_id, take_id) -> None:
        # Selection only. Selecting a take must never change timeline truth.
        layer = self._find_layer(timeline, layer_id)
        if take_id is not None:
            self._find_take(layer, take_id)
        timeline.selection.selected_layer_id = layer_id
        timeline.selection.selected_layer_ids = [layer_id]
        timeline.selection.selected_take_id = take_id
        timeline.selection.selected_event_ids = []

    def _handle_set_active_playback_target(self, timeline: Timeline, layer_id, take_id) -> None:
        if layer_id is None:
            timeline.playback_target.layer_id = None
            timeline.playback_target.take_id = None
            return

        layer = self._find_layer(timeline, layer_id)
        if take_id is not None:
            self._find_take(layer, take_id)
        timeline.playback_target.layer_id = layer.id
        timeline.playback_target.take_id = take_id

    def _handle_select_event(self, timeline: Timeline, layer_id, take_id, event_id, mode: str) -> None:
        layer = self._find_layer(timeline, layer_id)
        if event_id is None:
            timeline.selection.selected_layer_id = layer.id
            timeline.selection.selected_layer_ids = [layer.id]
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
        timeline.selection.selected_layer_ids = [layer.id]
        timeline.selection.selected_event_ids = selected_ids
        timeline.selection.selected_take_id = take_id if selected_ids else None

    def _handle_select_all_events(self, timeline: Timeline) -> None:
        selected_layer_ids = self._selected_layer_scope(timeline)
        target_layers: list[Layer]
        if selected_layer_ids:
            target_layers = [self._find_layer(timeline, layer_id) for layer_id in selected_layer_ids]
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
        timeline.selection.selected_layer_ids = [layer.id]
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
            timeline.selection.selected_layer_ids = [] if source_layer_id is None else [source_layer_id]
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
        timeline.selection.selected_layer_ids = [transfer_target.id]
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
        timeline.selection.selected_layer_ids = [] if selected_layer_id is None else [selected_layer_id]
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
            main_take = TimelineOrchestrator._main_take(layer)
            if main_take is None:
                continue
            for event in main_take.events:
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
        existing_event_ids: set[str] = set()
        imported_events = [
            self._build_manual_pull_event(
                take_id=take_id,
                source_track=source_track,
                source_event=source_event,
                order_index=index,
                existing_event_ids=existing_event_ids,
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
        existing_event_ids: set[str] | None = None,
    ) -> Event:
        start, end = self.diff_service.resolve_pull_event_range(source_event, order_index=order_index)
        event_id = self._next_manual_pull_event_id(
            take_id=take_id,
            source_track_coord=source_track.coord,
            source_event_id=source_event.event_id,
            order_index=order_index,
            existing_event_ids=existing_event_ids,
        )
        return Event(
            id=event_id,
            take_id=take_id,
            start=start,
            end=end,
            label=source_event.label,
            payload_ref=source_event.event_id,
        )

    @staticmethod
    def _next_manual_pull_event_id(
        *,
        take_id,
        source_track_coord: str,
        source_event_id: str,
        order_index: int,
        existing_event_ids: set[str] | None,
    ) -> str:
        reserved = existing_event_ids if existing_event_ids is not None else set()
        base_id = f"{take_id}:ma3:{source_track_coord}:{source_event_id}:{order_index}"
        if base_id not in reserved:
            reserved.add(base_id)
            return base_id
        suffix = 2
        while True:
            candidate = f"{base_id}:{suffix}"
            if candidate not in reserved:
                reserved.add(candidate)
                return candidate
            suffix += 1

    def _require_active_transfer_plan(self, session, plan_id: str, *, action_name: str) -> BatchTransferPlanState:
        plan = session.batch_transfer_plan
        if plan is None:
            raise ValueError(f"{action_name} requires an active batch transfer plan")
        if plan.plan_id != plan_id:
            raise ValueError(
                f"{action_name} plan_id does not match active batch transfer plan: "
                f"expected {plan.plan_id}, got {plan_id}"
            )
        return plan

    def _preview_transfer_plan(self, timeline: Timeline, session, plan: BatchTransferPlanState) -> None:
        preview_rows: list[BatchTransferPlanRowState] = []
        for row in plan.rows:
            if row.status == "blocked":
                preview_rows.append(self._copy_plan_row(row))
                continue
            if row.direction == "push":
                preview_rows.append(self._preview_push_plan_row(timeline, row))
                continue
            if row.direction == "pull":
                preview_rows.append(self._preview_pull_plan_row(row))
                continue
            preview_rows.append(self._copy_plan_row(row, issue=f"Unsupported transfer direction: {row.direction}"))

        plan.rows = preview_rows
        self._refresh_plan_counters(plan)

    def _apply_transfer_plan(self, timeline: Timeline, session, plan: BatchTransferPlanState) -> None:
        applied_rows: list[BatchTransferPlanRowState] = []
        stop_execution = False
        for row in plan.rows:
            if row.status == "blocked":
                applied_rows.append(self._copy_plan_row(row))
                continue
            if stop_execution:
                applied_rows.append(self._copy_plan_row(row))
                continue
            if row.status != "ready":
                applied_rows.append(self._copy_plan_row(row))
                continue
            try:
                if row.direction == "pull":
                    applied_row = self._apply_pull_plan_row(timeline, row)
                elif row.direction == "push":
                    applied_row = self._apply_push_plan_row(timeline, row)
                else:
                    applied_row = self._copy_plan_row(
                        row,
                        status="failed",
                        issue=f"Unsupported transfer direction: {row.direction}",
                    )
                applied_rows.append(applied_row)
                if applied_row.status == "failed":
                    stop_execution = True
            except Exception as exc:
                applied_rows.append(
                    self._copy_plan_row(
                        row,
                        status="failed",
                        issue=self._deterministic_issue_text(exc),
                    )
                )
                stop_execution = True

        plan.rows = applied_rows
        self._refresh_plan_counters(plan)
        self._clear_plan_diff_gates(session)

    def _cancel_transfer_plan(self, session, plan: BatchTransferPlanState) -> None:
        if plan.operation_type in {"push", "mixed"}:
            self._reset_manual_push_flow(session)
        if plan.operation_type in {"pull", "mixed"}:
            self._reset_manual_pull_flow(session)
        session.batch_transfer_plan = None

    def _preview_push_plan_row(self, timeline: Timeline, row: BatchTransferPlanRowState) -> BatchTransferPlanRowState:
        target_track = self._manual_push_track_by_coord(
            self._load_manual_push_track_options(),
            row.target_track_coord or "",
        )
        selected_events = self._selected_events_by_ids(timeline, list(row.selected_event_ids))
        self.diff_service.build_push_preview_rows(
            selected_events=selected_events,
            target_track_name=target_track.name,
            target_track_coord=target_track.coord,
        )
        return self._copy_plan_row(row)

    def _preview_pull_plan_row(self, row: BatchTransferPlanRowState) -> BatchTransferPlanRowState:
        if row.target_layer_id is None:
            return self._copy_plan_row(row)
        selected_events = self._manual_pull_selected_events_by_ids(
            available_events=self._load_manual_pull_event_options(row.source_track_coord or ""),
            selected_ids=list(row.selected_ma3_event_ids),
            action_name="PreviewTransferPlan",
        )
        self.diff_service.build_pull_preview_rows(
            selected_events=selected_events,
            target_layer_name=str(row.target_label or row.target_layer_id),
        )
        return self._copy_plan_row(row)

    def _apply_pull_plan_row(self, timeline: Timeline, row: BatchTransferPlanRowState) -> BatchTransferPlanRowState:
        source_track = self._manual_pull_track_by_coord(
            self._load_manual_pull_track_options(),
            row.source_track_coord or "",
            action_name="ApplyTransferPlan",
        )
        target_layer = self._resolve_manual_pull_target_layer(
            timeline,
            target_layer_id=row.target_layer_id,
            source_track=source_track,
        )
        selected_events = self._manual_pull_selected_events_by_ids(
            available_events=self._load_manual_pull_event_options(source_track.coord),
            selected_ids=list(row.selected_ma3_event_ids),
            action_name="ApplyTransferPlan",
        )
        selected_take_id, selected_event_ids = self._apply_manual_pull_import(
            target_layer=target_layer,
            source_track=source_track,
            selected_events=selected_events,
            import_mode=row.import_mode,
        )

        timeline.selection.selected_layer_id = target_layer.id
        timeline.selection.selected_layer_ids = [target_layer.id]
        timeline.selection.selected_take_id = selected_take_id
        timeline.selection.selected_event_ids = selected_event_ids
        return self._copy_plan_row(row, status="applied", issue=None)

    def _apply_push_plan_row(self, timeline: Timeline, row: BatchTransferPlanRowState) -> BatchTransferPlanRowState:
        selected_events = self._selected_events_by_ids(timeline, list(row.selected_event_ids))
        if not selected_events:
            return self._copy_plan_row(
                row,
                status="failed",
                issue="No main-take events selected for push",
            )
        apply_push = getattr(self.sync_service, "apply_push_transfer", None)
        if callable(apply_push):
            self._invoke_push_apply(
                apply_push,
                target_track_coord=row.target_track_coord,
                selected_events=selected_events,
            )
            return self._copy_plan_row(row, status="applied", issue=None)
        execute_push = getattr(self.sync_service, "execute_push_transfer", None)
        if callable(execute_push):
            self._invoke_push_apply(
                execute_push,
                target_track_coord=row.target_track_coord,
                selected_events=selected_events,
            )
            return self._copy_plan_row(row, status="applied", issue=None)
        return self._copy_plan_row(
            row,
            status="failed",
            issue="Push execution endpoint unavailable",
        )

    def _invoke_push_apply(self, callback, *, target_track_coord, selected_events) -> None:
        kwargs = {
            "target_track_coord": target_track_coord,
            "selected_events": selected_events,
        }
        transfer_mode = self.session_service.get_session().manual_push_flow.transfer_mode
        try:
            parameters = inspect.signature(callback).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "transfer_mode" in parameters:
            kwargs["transfer_mode"] = transfer_mode
        elif "mode" in parameters:
            kwargs["mode"] = transfer_mode
        callback(**kwargs)

    @staticmethod
    def _copy_plan_row(
        row: BatchTransferPlanRowState,
        *,
        status: str | None = None,
        issue: str | None = ...,
    ) -> BatchTransferPlanRowState:
        copied = BatchTransferPlanRowState(
            row_id=row.row_id,
            direction=row.direction,
            source_label=row.source_label,
            target_label=row.target_label,
            source_layer_id=row.source_layer_id,
            source_track_coord=row.source_track_coord,
            target_track_coord=row.target_track_coord,
            target_layer_id=row.target_layer_id,
            import_mode=row.import_mode,
            selected_event_ids=list(row.selected_event_ids),
            selected_ma3_event_ids=list(row.selected_ma3_event_ids),
            selected_count=row.selected_count,
            status=row.status if status is None else status,
            issue=row.issue,
        )
        if issue is not ...:
            copied.issue = issue
        return copied

    @staticmethod
    def _refresh_plan_counters(plan: BatchTransferPlanState) -> None:
        plan.draft_count = sum(1 for row in plan.rows if row.status == "draft")
        plan.ready_count = sum(1 for row in plan.rows if row.status == "ready")
        plan.blocked_count = sum(1 for row in plan.rows if row.status == "blocked")
        plan.applied_count = sum(1 for row in plan.rows if row.status == "applied")
        plan.failed_count = sum(1 for row in plan.rows if row.status == "failed")

    @staticmethod
    def _deterministic_issue_text(exc: Exception) -> str:
        message = str(exc).strip()
        return message or exc.__class__.__name__

    @staticmethod
    def _plan_counters_locked(plan: BatchTransferPlanState | None) -> bool:
        if plan is None:
            return False
        return (plan.applied_count + plan.failed_count) > 0

    @staticmethod
    def _clear_plan_diff_gates(session) -> None:
        session.manual_push_flow.diff_gate_open = False
        session.manual_push_flow.diff_preview = None
        session.manual_pull_flow.diff_gate_open = False
        session.manual_pull_flow.diff_preview = None

    @staticmethod
    def _reset_manual_push_flow(session) -> None:
        session.manual_push_flow.dialog_open = False
        session.manual_push_flow.push_mode_active = False
        session.manual_push_flow.selected_event_ids = []
        session.manual_push_flow.available_tracks = []
        session.manual_push_flow.target_track_coord = None
        session.manual_push_flow.transfer_mode = "merge"
        session.manual_push_flow.diff_gate_open = False
        session.manual_push_flow.diff_preview = None

    @staticmethod
    def _reset_manual_pull_flow(session) -> None:
        session.manual_pull_flow.dialog_open = False
        session.manual_pull_flow.workspace_active = False
        session.manual_pull_flow.available_tracks = []
        session.manual_pull_flow.selected_source_track_coords = []
        session.manual_pull_flow.active_source_track_coord = None
        session.manual_pull_flow.source_track_coord = None
        session.manual_pull_flow.available_events = []
        session.manual_pull_flow.selected_ma3_event_ids = []
        session.manual_pull_flow.selected_ma3_event_ids_by_track = {}
        session.manual_pull_flow.import_mode = "new_take"
        session.manual_pull_flow.import_mode_by_source_track = {}
        session.manual_pull_flow.available_target_layers = []
        session.manual_pull_flow.target_layer_id = None
        session.manual_pull_flow.target_layer_id_by_source_track = {}
        session.manual_pull_flow.diff_gate_open = False
        session.manual_pull_flow.diff_preview = None

    def _save_transfer_preset(self, timeline: Timeline, session, name: str) -> None:
        push_mapping = self._capture_push_preset_mapping(timeline, session)
        pull_mapping = self._capture_pull_preset_mapping(session)
        preset_id = self._next_transfer_preset_id(session, name)
        session.transfer_presets.append(
            TransferPresetState(
                preset_id=preset_id,
                name=name,
                push_target_mapping_by_layer_id=push_mapping,
                pull_target_mapping_by_source_track=pull_mapping,
            )
        )

    def _apply_transfer_preset(self, timeline: Timeline, session, preset_id: str) -> None:
        preset = self._require_transfer_preset(session, preset_id, action_name="ApplyTransferPreset")

        if session.manual_push_flow.push_mode_active:
            for layer in timeline.layers:
                target_track_coord = preset.push_target_mapping_by_layer_id.get(layer.id)
                if target_track_coord:
                    layer.sync.ma3_track_coord = target_track_coord
            selected_layer_scope = self._selected_layer_scope(timeline)
            if selected_layer_scope:
                session.manual_push_flow.target_track_coord = preset.push_target_mapping_by_layer_id.get(
                    timeline.selection.selected_layer_id or selected_layer_scope[0]
                )
            self._rebuild_push_transfer_plan(timeline, session)

        if session.manual_pull_flow.workspace_active:
            selected_coords = {
                coord
                for coord in session.manual_pull_flow.selected_source_track_coords
            }
            available_target_ids = {
                target.layer_id
                for target in session.manual_pull_flow.available_target_layers
            }
            next_mapping: dict[str, object] = {}
            for source_track_coord, target_layer_id in preset.pull_target_mapping_by_source_track.items():
                if source_track_coord in selected_coords and target_layer_id in available_target_ids:
                    next_mapping[source_track_coord] = target_layer_id
            session.manual_pull_flow.target_layer_id_by_source_track = next_mapping
            active_coord = session.manual_pull_flow.active_source_track_coord or session.manual_pull_flow.source_track_coord
            session.manual_pull_flow.target_layer_id = (
                next_mapping.get(active_coord)
                if active_coord is not None
                else None
            )
            self._rebuild_pull_transfer_plan(timeline, session)

    @staticmethod
    def _delete_transfer_preset(session, preset_id: str) -> None:
        preset = TimelineOrchestrator._require_transfer_preset(
            session,
            preset_id,
            action_name="DeleteTransferPreset",
        )
        session.transfer_presets = [
            candidate
            for candidate in session.transfer_presets
            if candidate.preset_id != preset.preset_id
        ]

    @staticmethod
    def _capture_push_preset_mapping(timeline: Timeline, session) -> dict[object, str]:
        plan = session.batch_transfer_plan
        if plan is not None and plan.operation_type in {"push", "mixed"}:
            return {
                row.source_layer_id: row.target_track_coord
                for row in plan.rows
                if row.direction == "push"
                and row.source_layer_id is not None
                and row.target_track_coord
            }
        return {
            layer.id: layer.sync.ma3_track_coord
            for layer in timeline.layers
            if layer.sync.ma3_track_coord
        }

    @staticmethod
    def _capture_pull_preset_mapping(session) -> dict[str, object]:
        plan = session.batch_transfer_plan
        if plan is not None and plan.operation_type in {"pull", "mixed"}:
            return {
                row.source_track_coord: row.target_layer_id
                for row in plan.rows
                if row.direction == "pull"
                and row.source_track_coord
                and row.target_layer_id is not None
            }
        return dict(session.manual_pull_flow.target_layer_id_by_source_track)

    @staticmethod
    def _slugify_transfer_preset_name(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
        return slug or "preset"

    @classmethod
    def _next_transfer_preset_id(cls, session, name: str) -> str:
        base_slug = cls._slugify_transfer_preset_name(name)
        existing_ids = {
            preset.preset_id
            for preset in session.transfer_presets
        }
        if base_slug not in existing_ids:
            return base_slug
        counter = 2
        while True:
            candidate = f"{base_slug}-{counter}"
            if candidate not in existing_ids:
                return candidate
            counter += 1

    @staticmethod
    def _require_transfer_preset(session, preset_id: str, *, action_name: str) -> TransferPresetState:
        for preset in session.transfer_presets:
            if preset.preset_id == preset_id:
                return preset
        raise ValueError(f"{action_name} preset_id not found: {preset_id}")

    def _rebuild_push_transfer_plan(self, timeline: Timeline, session) -> None:
        rows = self._build_push_transfer_plan_rows(timeline)
        if not rows:
            session.batch_transfer_plan = None
            return

        ready_count = sum(1 for row in rows if row.status == "ready")
        blocked_count = sum(1 for row in rows if row.status == "blocked")
        session.batch_transfer_plan = BatchTransferPlanState(
            plan_id=f"push:{timeline.id}",
            operation_type="push",
            rows=rows,
            ready_count=ready_count,
            blocked_count=blocked_count,
        )

    def _build_push_transfer_plan_rows(self, timeline: Timeline) -> list[BatchTransferPlanRowState]:
        grouped_records = self._selected_event_records_by_layer(timeline)
        if not grouped_records:
            return []

        available_tracks = self._load_manual_push_track_options()
        rows: list[BatchTransferPlanRowState] = []
        for layer, records in grouped_records:
            main_take = self._main_take(layer)
            main_event_ids = {
                event.id
                for event in (main_take.events if main_take is not None else [])
            }
            selected_event_ids = [
                record.event.id
                for record in records
                if record.event.id in main_event_ids
            ]
            target_track_coord = layer.sync.ma3_track_coord
            target_track = None
            if target_track_coord:
                target_track = self._manual_push_track_option_by_coord(
                    available_tracks,
                    target_track_coord,
                )
            if not selected_event_ids:
                status = "blocked"
                issue = "Select main-take events to push"
            elif not target_track_coord:
                status = "blocked"
                issue = "Select an MA3 target track"
            else:
                status = "ready"
                issue = None
            target_label = (
                self._format_manual_push_target_label(target_track)
                if target_track is not None
                else (target_track_coord or "Unmapped")
            )
            rows.append(
                BatchTransferPlanRowState(
                    row_id=f"push:{layer.id}",
                    direction="push",
                    source_label=layer.name,
                    target_label=target_label,
                    source_layer_id=layer.id,
                    target_track_coord=target_track_coord,
                    selected_event_ids=selected_event_ids,
                    selected_count=len(selected_event_ids),
                    status=status,
                    issue=issue,
                )
            )

        rows.sort(key=lambda row: (row.source_label.lower(), row.row_id))
        return rows

    def _rebuild_pull_transfer_plan(self, timeline: Timeline, session) -> None:
        rows = self._build_pull_transfer_plan_rows(timeline, session)
        if not rows:
            if session.batch_transfer_plan is not None and session.batch_transfer_plan.operation_type == "pull":
                session.batch_transfer_plan = None
            return

        ready_count = sum(1 for row in rows if row.status == "ready")
        blocked_count = sum(1 for row in rows if row.status == "blocked")
        draft_count = sum(1 for row in rows if row.status == "draft")
        session.batch_transfer_plan = BatchTransferPlanState(
            plan_id=f"pull:{timeline.id}",
            operation_type="pull",
            rows=rows,
            draft_count=draft_count,
            ready_count=ready_count,
            blocked_count=blocked_count,
        )

    def _build_pull_transfer_plan_rows(self, timeline: Timeline, session) -> list[BatchTransferPlanRowState]:
        flow = session.manual_pull_flow
        if not flow.workspace_active:
            return []

        tracks_by_coord = {
            track.coord: track
            for track in flow.available_tracks
        }
        track_order = {
            track.coord: index
            for index, track in enumerate(flow.available_tracks)
        }
        target_labels = {
            target.layer_id: target.name
            for target in flow.available_target_layers
        }
        selected_coords = [
            coord
            for coord in flow.selected_source_track_coords
            if coord in tracks_by_coord
        ]
        selected_coords.sort(key=lambda coord: (track_order.get(coord, 0), coord))

        rows: list[BatchTransferPlanRowState] = []
        for coord in selected_coords:
            track = tracks_by_coord[coord]
            selected_event_ids = list(flow.selected_ma3_event_ids_by_track.get(coord, []))
            target_layer_id = flow.target_layer_id_by_source_track.get(coord)
            import_mode = flow.import_mode_by_source_track.get(coord, flow.import_mode or "new_take")
            if not selected_event_ids and target_layer_id is None:
                status = "blocked"
                issue = "Select source events and target layer mapping"
            elif not selected_event_ids:
                status = "blocked"
                issue = "Select source events"
            elif target_layer_id is None:
                status = "blocked"
                issue = "Select target layer mapping"
            else:
                status = "ready"
                issue = None
            target_label = (
                target_labels.get(target_layer_id, str(target_layer_id))
                if target_layer_id is not None
                else "Unmapped"
            )
            rows.append(
                BatchTransferPlanRowState(
                    row_id=f"pull:{coord}",
                    direction="pull",
                    source_label=f"{track.name} ({track.coord})",
                    target_label=target_label,
                    source_track_coord=coord,
                    target_layer_id=target_layer_id,
                    import_mode=import_mode,
                    selected_ma3_event_ids=selected_event_ids,
                    selected_count=len(selected_event_ids),
                    status=status,
                    issue=issue,
                )
            )

        return rows

    def _selected_event_records_by_layer(self, timeline: Timeline) -> list[tuple[Layer, list[_EventRecord]]]:
        records = self._selected_event_records(timeline, list(timeline.selection.selected_event_ids))
        explicit_selected_layer_scope = set(dict.fromkeys(timeline.selection.selected_layer_ids))
        if explicit_selected_layer_scope:
            records = [record for record in records if record.layer.id in explicit_selected_layer_scope]
        grouped: dict[object, list[TimelineOrchestrator._EventRecord]] = {}
        layer_order: dict[object, int] = {}
        for index, layer in enumerate(sorted(timeline.layers, key=lambda value: value.order_index)):
            layer_order[layer.id] = index
        layer_lookup: dict[object, Layer] = {}
        for record in records:
            layer_lookup[record.layer.id] = record.layer
            grouped.setdefault(record.layer.id, []).append(record)
        ordered_layer_ids = sorted(grouped.keys(), key=lambda layer_id: layer_order.get(layer_id, 0))
        return [(layer_lookup[layer_id], grouped[layer_id]) for layer_id in ordered_layer_ids]

    @staticmethod
    def _selected_layer_scope(timeline: Timeline) -> list[LayerId]:
        selected_layer_ids = list(dict.fromkeys(timeline.selection.selected_layer_ids))
        if selected_layer_ids:
            return selected_layer_ids
        if timeline.selection.selected_layer_id is not None:
            return [timeline.selection.selected_layer_id]
        return []

    @staticmethod
    def _manual_push_track_option_by_coord(
        available_tracks: list[ManualPushTrackOption],
        target_track_coord: str | None,
    ) -> ManualPushTrackOption | None:
        if not target_track_coord:
            return None
        for track in available_tracks:
            if track.coord == target_track_coord:
                return track
        return None

    @staticmethod
    def _format_manual_push_target_label(track: ManualPushTrackOption) -> str:
        if track.note:
            return f"{track.name} ({track.coord}) - {track.note}"
        return f"{track.name} ({track.coord})"

    @staticmethod
    def _next_manual_pull_take_id(layer: Layer):
        existing_ids = {str(take.id) for take in layer.takes}
        index = 1
        while True:
            candidate = f"{layer.id}:ma3_pull:{index}"
            if candidate not in existing_ids:
                return candidate
            index += 1

    def _apply_manual_pull_import(
        self,
        *,
        target_layer: Layer,
        source_track: ManualPullTrackOption,
        selected_events: list[ManualPullEventOption],
        import_mode: str,
    ) -> tuple[TakeId, list]:
        if import_mode == "main":
            target_take = self._resolve_or_create_manual_pull_main_take(target_layer)
            existing_event_ids = {str(event.id) for event in target_take.events}
            imported_events = [
                self._build_manual_pull_event(
                    take_id=target_take.id,
                    source_track=source_track,
                    source_event=source_event,
                    order_index=index,
                    existing_event_ids=existing_event_ids,
                )
                for index, source_event in enumerate(selected_events, start=1)
            ]
            target_take.events.extend(imported_events)
            self._sort_take_events(target_take)
            return target_take.id, [event.id for event in imported_events]

        imported_take = self._build_manual_pull_take(
            layer=target_layer,
            source_track=source_track,
            selected_events=selected_events,
        )
        target_layer.takes.append(imported_take)
        self._sort_take_events(imported_take)
        return imported_take.id, [event.id for event in imported_take.events]

    @staticmethod
    def _resolve_or_create_manual_pull_main_take(layer: Layer) -> Take:
        main_take = TimelineOrchestrator._main_take(layer)
        if main_take is not None:
            return main_take
        main_take = Take(
            id=TakeId(f"{layer.id}:main"),
            layer_id=layer.id,
            name="Main",
        )
        layer.takes.insert(0, main_take)
        return main_take

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
        targets = [
            ManualPullTargetOption(layer_id=layer.id, name=layer.name)
            for layer in sorted(timeline.layers, key=lambda value: value.order_index)
            if layer.kind == LayerKind.EVENT
            and layer.presentation_hints.visible
            and not layer.presentation_hints.locked
        ]
        targets.extend(
            [
                ManualPullTargetOption(
                    layer_id=_PULL_TARGET_CREATE_NEW_LAYER_ID,
                    name=_PULL_TARGET_CREATE_NEW_LAYER_NAME,
                ),
                ManualPullTargetOption(
                    layer_id=_PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_ID,
                    name=_PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_NAME,
                ),
            ]
        )
        return targets

    @staticmethod
    def _is_manual_pull_synthetic_target(target_layer_id) -> bool:
        return target_layer_id in {
            _PULL_TARGET_CREATE_NEW_LAYER_ID,
            _PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_ID,
        }

    def _resolve_manual_pull_target_layer(
        self,
        timeline: Timeline,
        *,
        target_layer_id,
        source_track: ManualPullTrackOption,
    ) -> Layer:
        if self._is_manual_pull_synthetic_target(target_layer_id):
            return self._create_manual_pull_target_layer(timeline, source_track=source_track)
        return self._find_layer(timeline, target_layer_id)

    def _create_manual_pull_target_layer(self, timeline: Timeline, *, source_track: ManualPullTrackOption) -> Layer:
        layer_name = self._next_manual_pull_layer_name(timeline, source_track.name)
        layer_id = self._next_manual_pull_layer_id(timeline, source_track)
        main_take_id = TakeId(f"{layer_id}:main")
        new_layer = Layer(
            id=layer_id,
            timeline_id=timeline.id,
            name=layer_name,
            kind=LayerKind.EVENT,
            order_index=self._next_timeline_layer_order_index(timeline),
            takes=[
                Take(
                    id=main_take_id,
                    layer_id=layer_id,
                    name="Main",
                )
            ],
        )
        timeline.layers.append(new_layer)
        return new_layer

    @staticmethod
    def _next_timeline_layer_order_index(timeline: Timeline) -> int:
        if not timeline.layers:
            return 0
        return max(int(layer.order_index) for layer in timeline.layers) + 1

    def _next_manual_pull_layer_id(self, timeline: Timeline, source_track: ManualPullTrackOption):
        base_slug = self._manual_pull_layer_slug(source_track)
        existing_ids = {str(layer.id) for layer in timeline.layers}
        index = 1
        while True:
            suffix = "" if index == 1 else f"_{index}"
            candidate = LayerId(f"layer_ma3_pull_{base_slug}{suffix}")
            if str(candidate) not in existing_ids:
                return candidate
            index += 1

    def _next_manual_pull_layer_name(self, timeline: Timeline, source_name: str) -> str:
        base_name = source_name.strip() or "Imported Layer"
        existing_names = {layer.name for layer in timeline.layers}
        if base_name not in existing_names:
            return base_name
        index = 2
        while True:
            candidate = f"{base_name} {index}"
            if candidate not in existing_names:
                return candidate
            index += 1

    @staticmethod
    def _manual_pull_layer_slug(source_track: ManualPullTrackOption) -> str:
        raw = f"{source_track.name}_{source_track.coord}".strip().lower()
        normalized = unicodedata.normalize("NFKD", raw)
        ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
        slug = re.sub(r"[^a-z0-9]+", "_", ascii_only).strip("_")
        return slug or "import"

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
