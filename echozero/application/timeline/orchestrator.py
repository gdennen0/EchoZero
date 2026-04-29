"""Timeline orchestrator for the canonical EchoZero application layer.
Exists to mutate timeline and session state in response to timeline intents.
Connects the timeline app contract to selection, editing, transfer, and sync flows.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from echozero.application.mixer.models import LayerMixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.service import PlaybackService
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.cue_numbers import cue_number_text
from echozero.application.session.models import (
    ManualPullDiffPreview,
    ManualPullEventOption,
    ManualPullFlowState,
    ManualPullTrackGroupOption,
    ManualPullTrackOption,
    ManualPullTimecodeOption,
    ManualPushDiffPreview,
    Session,
)
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import FollowMode, LayerKind
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.application.shared.ids import EventId, LayerId, RegionId, TakeId
from echozero.application.shared.ranges import TimeRange
from echozero.application.sync.diff_service import SyncDiffService
from echozero.application.sync.models import LiveSyncState, coerce_live_sync_state
from echozero.application.sync.service import SyncService
from echozero.application.timeline.ma3_push_intents import (
    AssignMA3TrackSequence,
    CreateMA3Timecode,
    CreateMA3Track,
    CreateMA3TrackGroup,
    CreateMA3Sequence,
    PrepareMA3TrackForPush,
    PushLayerToMA3,
    RefreshMA3Sequences,
    RefreshMA3PushTracks,
    SetLayerMA3Route,
)
from echozero.application.timeline.orchestrator_ma3_push_mixin import (
    TimelineOrchestratorMA3PushMixin,
)
from echozero.application.timeline.orchestrator_selection_mixin import (
    TimelineOrchestratorSelectionMixin,
)
from echozero.application.timeline.orchestrator_manual_pull_import_mixin import (
    TimelineOrchestratorManualPullImportMixin,
)
from echozero.application.timeline.orchestrator_sync_preset_mixin import (
    TimelineOrchestratorSyncPresetMixin,
)
from echozero.application.timeline.orchestrator_transfer_plan_mixin import (
    TimelineOrchestratorTransferPlanMixin,
)
from echozero.application.timeline.orchestrator_transfer_lookup_mixin import (
    _PULL_TARGET_CREATE_NEW_LAYER_ID,
    _PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_ID,
    _PULL_TARGET_CREATE_NEW_SECTION_LAYER_ID,
    TimelineOrchestratorTransferLookupMixin,
)

if TYPE_CHECKING:
    from echozero.application.timeline.assembler import TimelineAssembler

from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    ApplyTransferPlan,
    ApplyTransferPreset,
    CancelTransferPlan,
    ClearLayerLiveSyncPauseReason,
    ClearSelection,
    CreateRegion,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    CreateEvent,
    DeleteRegion,
    DeleteEvents,
    DeleteTransferPreset,
    DisableExperimentalLiveSync,
    DisableSync,
    DuplicateSelectedEvents,
    EnableExperimentalLiveSync,
    EnableSync,
    ExitPullFromMA3Workspace,
    ExitPushToMA3Mode,
    MoveSelectedEventsToAdjacentLayer,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    ReorderLayer,
    OpenPullFromMA3Dialog,
    OpenPushToMA3Dialog,
    Pause,
    Play,
    PreviewTransferPlan,
    ReplaceSectionCues,
    RenumberEventCueNumbers,
    SaveTransferPreset,
    Seek,
    SelectAllEvents,
    SelectAdjacentEventInSelectedLayer,
    SelectAdjacentLayer,
    SelectEveryOtherEvents,
    SelectEvent,
    SelectLayer,
    SelectPullSourceEvents,
    SelectPullSourceTimecode,
    SelectPullSourceTrack,
    SelectPullSourceTrackGroup,
    SelectPullSourceTracks,
    SelectPullTargetLayer,
    SelectPushTargetTrack,
    SelectTake,
    SetActivePlaybackTarget,
    SetGain,
    SetLayerOutputBus,
    SetFollowCursorEnabled,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    SetPullImportMode,
    SetPullSourceEvents,
    SetPullTrackOptions,
    SetPushTrackOptions,
    SetPushTransferMode,
    SetSelectedEvents,
    SelectRegion,
    Stop,
    TimelineIntent,
    ToggleLayerExpanded,
    TriggerTakeAction,
    TrimEvent,
    UpdateEventLabel,
    UpdateRegion,
)
from echozero.application.timeline.models import (
    Event,
    EventRef,
    Layer,
    Take,
    Timeline,
    TimelineRegion,
    derive_section_cues_from_layers,
)
from echozero.application.transport.service import TransportService

_KEYBOARD_STEP_SECONDS = 1.0 / 30.0


@dataclass(slots=True)
class TimelineOrchestrator(
    TimelineOrchestratorSelectionMixin,
    TimelineOrchestratorMA3PushMixin,
    TimelineOrchestratorSyncPresetMixin,
    TimelineOrchestratorTransferLookupMixin,
    TimelineOrchestratorManualPullImportMixin,
    TimelineOrchestratorTransferPlanMixin,
):
    """Coordinates timeline intents across sibling application services."""

    session_service: SessionService
    transport_service: TransportService
    mixer_service: MixerService
    playback_service: PlaybackService
    sync_service: SyncService
    assembler: "TimelineAssembler"
    diff_service: SyncDiffService = field(default_factory=SyncDiffService)

    def handle(self, timeline: Timeline, intent: TimelineIntent) -> TimelinePresentation:
        if isinstance(intent, SelectLayer):
            self._handle_select_layer(timeline, intent.layer_id, mode=intent.mode)

        elif isinstance(intent, SelectAdjacentLayer):
            self._handle_select_adjacent_layer(timeline, direction=intent.direction)

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

        elif isinstance(intent, SelectAdjacentEventInSelectedLayer):
            self._handle_select_adjacent_event_in_selected_layer(
                timeline,
                direction=intent.direction,
                include_demoted=bool(intent.include_demoted),
            )

        elif isinstance(intent, ClearSelection):
            timeline.selection.selected_take_id = None
            timeline.selection.selected_event_refs = []
            timeline.selection.selected_event_ids = []
            timeline.selection.selected_region_id = None

        elif isinstance(intent, SelectAllEvents):
            self._handle_select_all_events(timeline)

        elif isinstance(intent, SetSelectedEvents):
            self._handle_set_selected_events(
                timeline,
                event_ids=list(intent.event_ids),
                event_refs=list(intent.event_refs or []),
                anchor_layer_id=intent.anchor_layer_id,
                anchor_take_id=intent.anchor_take_id,
                selected_layer_ids=list(intent.selected_layer_ids or []),
            )

        elif isinstance(intent, SelectRegion):
            if intent.region_id is None:
                timeline.selection.selected_region_id = None
            else:
                self._require_region(timeline, intent.region_id)
                timeline.selection.selected_region_id = intent.region_id

        elif isinstance(intent, CreateRegion):
            self._handle_create_region(
                timeline,
                time_range=intent.time_range,
                label=intent.label,
                color=intent.color,
                kind=intent.kind,
            )

        elif isinstance(intent, UpdateRegion):
            self._handle_update_region(
                timeline,
                region_id=intent.region_id,
                time_range=intent.time_range,
                label=intent.label,
                color=intent.color,
                kind=intent.kind,
            )

        elif isinstance(intent, DeleteRegion):
            self._handle_delete_region(
                timeline,
                region_id=intent.region_id,
            )

        elif isinstance(intent, SelectEveryOtherEvents):
            self._handle_select_every_other_events(
                timeline,
                scope=intent.scope,
            )

        elif isinstance(intent, RenumberEventCueNumbers):
            self._handle_renumber_event_cue_numbers(
                timeline,
                scope=intent.scope,
                start_at=intent.start_at,
                step=intent.step,
            )

        elif isinstance(intent, CreateEvent):
            self._handle_create_event(
                timeline,
                layer_id=intent.layer_id,
                take_id=intent.take_id,
                time_range=intent.time_range,
                label=intent.label,
                cue_number=intent.cue_number,
                source_event_id=intent.source_event_id,
                payload_ref=intent.payload_ref,
                color=intent.color,
            )

        elif isinstance(intent, TrimEvent):
            self._handle_trim_event(
                timeline,
                event_id=intent.event_id,
                new_range=intent.new_range,
            )

        elif isinstance(intent, UpdateEventLabel):
            self._handle_update_event_label(
                timeline,
                event_id=intent.event_id,
                label=intent.label,
                layer_id=intent.layer_id,
                take_id=intent.take_id,
            )

        elif isinstance(intent, ReplaceSectionCues):
            self._handle_replace_section_cues(
                timeline,
                cues=list(intent.cues),
            )

        elif isinstance(intent, DeleteEvents):
            self._handle_delete_events(
                timeline,
                event_ids=list(intent.event_ids),
                event_refs=list(intent.event_refs or []),
            )

        elif isinstance(intent, MoveSelectedEvents):
            self._handle_move_selected_events(
                timeline,
                delta_seconds=float(intent.delta_seconds),
                target_layer_id=intent.target_layer_id,
                copy_selected=bool(intent.copy_selected),
            )

        elif isinstance(intent, MoveSelectedEventsToAdjacentLayer):
            self._handle_move_selected_events_to_adjacent_layer(
                timeline,
                direction=intent.direction,
            )

        elif isinstance(intent, ReorderLayer):
            self._handle_reorder_layer(
                timeline,
                source_layer_id=intent.source_layer_id,
                target_after_layer_id=intent.target_after_layer_id,
                insert_at_start=intent.insert_at_start,
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

        elif isinstance(intent, SetFollowCursorEnabled):
            session = self.session_service.get_session()
            if intent.enabled:
                if session.transport_state.follow_mode == FollowMode.OFF:
                    session.transport_state.follow_mode = FollowMode.CENTER
            else:
                session.transport_state.follow_mode = FollowMode.OFF

        elif isinstance(intent, SetGain):
            layer = self._find_layer(timeline, intent.layer_id)
            self.mixer_service.set_gain(intent.layer_id, intent.gain_db)
            layer.mixer.gain_db = intent.gain_db

        elif isinstance(intent, SetLayerOutputBus):
            layer = self._find_layer(timeline, intent.layer_id)
            layer.mixer.output_bus = intent.output_bus
            mixer_state = self.mixer_service.get_state()
            state = mixer_state.layer_states.setdefault(
                intent.layer_id,
                LayerMixerState(),
            )
            state.output_bus = intent.output_bus

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
            layer.sync.live_sync_state = coerce_live_sync_state(intent.live_sync_state)
            if intent.live_sync_state is LiveSyncState.OFF:
                layer.sync.live_sync_pause_reason = None

        elif isinstance(intent, SetLayerLiveSyncPauseReason):
            layer = self._find_layer(timeline, intent.layer_id)
            layer.sync.live_sync_state = LiveSyncState.PAUSED
            layer.sync.live_sync_pause_reason = intent.pause_reason

        elif isinstance(intent, ClearLayerLiveSyncPauseReason):
            layer = self._find_layer(timeline, intent.layer_id)
            layer.sync.live_sync_pause_reason = None

        elif isinstance(intent, RefreshMA3PushTracks):
            self._handle_refresh_ma3_push_tracks(intent)

        elif isinstance(intent, RefreshMA3Sequences):
            self._handle_refresh_ma3_sequences(intent)

        elif isinstance(intent, AssignMA3TrackSequence):
            self._handle_assign_ma3_track_sequence(intent)

        elif isinstance(intent, CreateMA3Sequence):
            self._handle_create_ma3_sequence(intent)

        elif isinstance(intent, CreateMA3Timecode):
            self._handle_create_ma3_timecode(intent)

        elif isinstance(intent, CreateMA3TrackGroup):
            self._handle_create_ma3_track_group(intent)

        elif isinstance(intent, CreateMA3Track):
            self._handle_create_ma3_track(intent)

        elif isinstance(intent, PrepareMA3TrackForPush):
            self._handle_prepare_ma3_track_for_push(intent)

        elif isinstance(intent, SetLayerMA3Route):
            self._handle_set_layer_ma3_route(
                timeline,
                layer_id=intent.layer_id,
                target_track_coord=intent.target_track_coord,
                sequence_action=intent.sequence_action,
            )

        elif isinstance(intent, PushLayerToMA3):
            self._handle_push_layer_to_ma3(timeline, intent)

        elif isinstance(intent, OpenPushToMA3Dialog):
            session = self.session_service.get_session()
            self._set_selected_event_refs(
                timeline,
                self._resolve_event_refs_by_ids(
                    timeline,
                    list(intent.selection_event_ids),
                    preferred_layer_ids=self._selected_layer_scope(timeline),
                    preferred_take_id=timeline.selection.selected_take_id,
                ),
            )
            session.manual_push_flow.selected_layer_ids = self._selected_layer_scope(timeline)
            session.manual_push_flow.dialog_open = False
            session.manual_push_flow.push_mode_active = True
            session.manual_push_flow.selected_event_ids = list(intent.selection_event_ids)
            session.manual_push_flow.target_track_coord = None
            session.manual_push_flow.transfer_mode = "merge"
            session.manual_push_flow.diff_gate_open = False
            session.manual_push_flow.diff_preview = None
            self._refresh_manual_push_tracks()
            self._rebuild_push_transfer_plan(timeline, session)

        elif isinstance(intent, ExitPushToMA3Mode):
            session = self.session_service.get_session()
            session.manual_push_flow.dialog_open = False
            session.manual_push_flow.push_mode_active = False
            session.manual_push_flow.selected_layer_ids = []
            session.manual_push_flow.selected_event_ids = []
            session.manual_push_flow.available_timecodes = []
            session.manual_push_flow.selected_timecode_no = None
            session.manual_push_flow.available_track_groups = []
            session.manual_push_flow.selected_track_group_no = None
            session.manual_push_flow.available_tracks = []
            session.manual_push_flow.available_sequences = []
            session.manual_push_flow.current_song_sequence_range = None
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
            available_coords = {track.coord for track in session.manual_push_flow.available_tracks}
            if intent.target_track_coord not in available_coords:
                raise ValueError(
                    f"SelectPushTargetTrack target_track_coord not found in available_tracks: "
                    f"{intent.target_track_coord}"
                )
            session.manual_push_flow.target_track_coord = intent.target_track_coord
            if intent.layer_id is not None:
                layer = self._find_layer(timeline, intent.layer_id)
                layer.sync.ma3_track_coord = intent.target_track_coord
            elif (
                timeline.selection.selected_layer_id is not None
                and session.manual_push_flow.push_mode_active
            ):
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
                    if main_take is not None and any(
                        event.id in selected_event_lookup for event in main_take.events
                    ):
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
            flow = session.manual_pull_flow
            flow.dialog_open = False
            flow.workspace_active = True
            flow.available_timecodes = self._load_manual_pull_timecode_options()
            flow.available_target_layers = self._load_manual_pull_target_options(timeline)
            flow.selected_timecode_no = self._default_manual_pull_timecode_no(
                timeline,
                available_timecodes=flow.available_timecodes,
            )
            flow.available_track_groups = (
                self._load_manual_pull_track_group_options(timecode_no=flow.selected_timecode_no)
                if flow.selected_timecode_no is not None
                else []
            )
            flow.selected_track_group_no = self._default_manual_pull_track_group_no(
                timeline,
                selected_timecode_no=flow.selected_timecode_no,
                available_track_groups=flow.available_track_groups,
            )
            flow.available_tracks = (
                self._load_manual_pull_track_options(
                    timecode_no=flow.selected_timecode_no,
                )
                if flow.selected_timecode_no is not None
                else []
            )
            flow.selected_source_track_coords = []
            flow.active_source_track_coord = None
            flow.source_track_coord = None
            flow.available_events = []
            flow.selected_ma3_event_ids = []
            flow.selected_ma3_event_ids_by_track = {}
            flow.import_mode = "new_take"
            flow.import_mode_by_source_track = {}
            flow.target_layer_id = None
            flow.target_layer_id_by_source_track = {}
            flow.diff_gate_open = False
            flow.diff_preview = None
            default_source_track_coord = self._default_manual_pull_source_track_coord(
                timeline,
                session,
                available_tracks=flow.available_tracks,
                preferred_track_group_no=flow.selected_track_group_no,
            )
            if default_source_track_coord is not None:
                self._set_active_manual_pull_source_track(
                    timeline,
                    session,
                    source_track_coord=default_source_track_coord,
                )
            else:
                self._refresh_manual_pull_target_options(timeline, session)
            self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, ExitPullFromMA3Workspace):
            session = self.session_service.get_session()
            flow = session.manual_pull_flow
            flow.dialog_open = False
            flow.workspace_active = False
            flow.available_timecodes = []
            flow.selected_timecode_no = None
            flow.available_track_groups = []
            flow.selected_track_group_no = None
            flow.available_tracks = []
            flow.selected_source_track_coords = []
            flow.active_source_track_coord = None
            flow.source_track_coord = None
            flow.available_events = []
            flow.selected_ma3_event_ids = []
            flow.selected_ma3_event_ids_by_track = {}
            flow.import_mode = "new_take"
            flow.import_mode_by_source_track = {}
            flow.available_target_layers = []
            flow.target_layer_id = None
            flow.target_layer_id_by_source_track = {}
            flow.diff_gate_open = False
            flow.diff_preview = None
            if (
                session.batch_transfer_plan is not None
                and session.batch_transfer_plan.operation_type == "pull"
            ):
                session.batch_transfer_plan = None

        elif isinstance(intent, SetPullTrackOptions):
            session = self.session_service.get_session()
            session.manual_pull_flow.available_tracks = list(intent.tracks)
            if session.manual_pull_flow.workspace_active:
                selected_coords = {
                    track.coord for track in session.manual_pull_flow.available_tracks
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
                    session.manual_pull_flow.target_layer_id = None
                self._refresh_manual_pull_target_options(timeline, session)
                self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, SelectPullSourceTimecode):
            session = self.session_service.get_session()
            flow = session.manual_pull_flow
            available_timecodes = {timecode.number for timecode in flow.available_timecodes}
            if intent.timecode_no not in available_timecodes:
                raise ValueError(
                    "SelectPullSourceTimecode timecode_no not found in available_timecodes: "
                    f"{intent.timecode_no}"
                )
            flow.selected_timecode_no = intent.timecode_no
            flow.available_track_groups = self._load_manual_pull_track_group_options(
                timecode_no=intent.timecode_no
            )
            flow.selected_track_group_no = self._default_manual_pull_track_group_no(
                timeline,
                selected_timecode_no=flow.selected_timecode_no,
                available_track_groups=flow.available_track_groups,
            )
            flow.available_tracks = (
                self._load_manual_pull_track_options(
                    timecode_no=flow.selected_timecode_no,
                )
                if flow.selected_timecode_no is not None
                else []
            )
            default_source_track_coord = self._default_manual_pull_source_track_coord(
                timeline,
                session,
                available_tracks=flow.available_tracks,
                preferred_track_group_no=flow.selected_track_group_no,
            )
            if default_source_track_coord is None:
                self._clear_manual_pull_source_selection(session)
                self._refresh_manual_pull_target_options(timeline, session)
            else:
                self._set_active_manual_pull_source_track(
                    timeline,
                    session,
                    source_track_coord=default_source_track_coord,
                )
            self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, SelectPullSourceTrackGroup):
            session = self.session_service.get_session()
            flow = session.manual_pull_flow
            available_track_groups = {group.number for group in flow.available_track_groups}
            if intent.track_group_no not in available_track_groups:
                raise ValueError(
                    "SelectPullSourceTrackGroup track_group_no not found in available_track_groups: "
                    f"{intent.track_group_no}"
                )
            flow.selected_track_group_no = intent.track_group_no
            flow.available_tracks = (
                self._load_manual_pull_track_options(
                    timecode_no=flow.selected_timecode_no,
                )
                if flow.selected_timecode_no is not None
                else []
            )
            default_source_track_coord = self._default_manual_pull_source_track_coord(
                timeline,
                session,
                available_tracks=flow.available_tracks,
                preferred_track_group_no=flow.selected_track_group_no,
            )
            if default_source_track_coord is None:
                self._clear_manual_pull_source_selection(session)
                self._refresh_manual_pull_target_options(timeline, session)
            else:
                self._set_active_manual_pull_source_track(
                    timeline,
                    session,
                    source_track_coord=default_source_track_coord,
                )
            self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, SelectPullSourceTracks):
            session = self.session_service.get_session()
            available_by_coord = {
                track.coord: track for track in session.manual_pull_flow.available_tracks
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
            self._refresh_manual_pull_target_options(timeline, session)
            if session.manual_pull_flow.active_source_track_coord not in ordered_coords:
                next_active_coord = ordered_coords[0]
                session.manual_pull_flow.active_source_track_coord = next_active_coord
                session.manual_pull_flow.source_track_coord = next_active_coord
                session.manual_pull_flow.available_events = self._load_manual_pull_event_options(
                    next_active_coord
                )
                session.manual_pull_flow.selected_ma3_event_ids = list(
                    session.manual_pull_flow.selected_ma3_event_ids_by_track.get(next_active_coord, [])
                )
                session.manual_pull_flow.import_mode = (
                    session.manual_pull_flow.import_mode_by_source_track.get(
                        next_active_coord,
                        "new_take",
                    )
                )
                session.manual_pull_flow.target_layer_id = (
                    session.manual_pull_flow.target_layer_id_by_source_track.get(next_active_coord)
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
            self._set_active_manual_pull_source_track(
                timeline,
                session,
                source_track_coord=source_track.coord,
            )
            session.manual_pull_flow.selected_track_group_no = self._ma3_track_coord_track_group_no(
                source_track.coord
            )
            if session.manual_pull_flow.workspace_active:
                self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, SetPullSourceEvents):
            session = self.session_service.get_session()
            session.manual_pull_flow.available_events = list(intent.events)

        elif isinstance(intent, SelectPullSourceEvents):
            session = self.session_service.get_session()
            available_event_ids = {
                event.event_id for event in session.manual_pull_flow.available_events
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
            active_source_coord = (
                session.manual_pull_flow.active_source_track_coord
                or session.manual_pull_flow.source_track_coord
            )
            if active_source_coord:
                session.manual_pull_flow.selected_ma3_event_ids_by_track[active_source_coord] = list(
                    intent.selected_ma3_event_ids
                )
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None
            if session.manual_pull_flow.workspace_active:
                self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, SelectPullTargetLayer):
            session = self.session_service.get_session()
            target_option = self._manual_pull_target_layer_by_id(
                session.manual_pull_flow.available_target_layers,
                intent.target_layer_id,
                action_name="SelectPullTargetLayer",
            )
            session.manual_pull_flow.target_layer_id = target_option.layer_id
            derived_import_mode = self._manual_pull_import_mode_for_target_layer(
                timeline,
                target_option.layer_id,
            )
            active_source_coord = (
                session.manual_pull_flow.active_source_track_coord
                or session.manual_pull_flow.source_track_coord
            )
            if target_option.layer_id == _PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_ID:
                for coord in session.manual_pull_flow.selected_source_track_coords:
                    session.manual_pull_flow.target_layer_id_by_source_track[coord] = (
                        target_option.layer_id
                    )
                    session.manual_pull_flow.import_mode_by_source_track[coord] = (
                        derived_import_mode
                    )
            elif active_source_coord:
                session.manual_pull_flow.target_layer_id_by_source_track[active_source_coord] = (
                    target_option.layer_id
                )
                session.manual_pull_flow.import_mode_by_source_track[active_source_coord] = (
                    derived_import_mode
                )
            session.manual_pull_flow.import_mode = derived_import_mode
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None
            if session.manual_pull_flow.workspace_active:
                self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, SetPullImportMode):
            session = self.session_service.get_session()
            active_source_coord = (
                session.manual_pull_flow.active_source_track_coord
                or session.manual_pull_flow.source_track_coord
            )
            resolved_target_layer_id = (
                session.manual_pull_flow.target_layer_id_by_source_track.get(active_source_coord)
                if active_source_coord is not None
                else session.manual_pull_flow.target_layer_id
            )
            resolved_import_mode = (
                self._manual_pull_import_mode_for_target_layer(
                    timeline,
                    resolved_target_layer_id,
                )
                if resolved_target_layer_id is not None
                else intent.import_mode
            )
            session.manual_pull_flow.import_mode = resolved_import_mode
            if active_source_coord:
                session.manual_pull_flow.import_mode_by_source_track[active_source_coord] = (
                    resolved_import_mode
                )
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
            target_option = self._manual_pull_target_layer_by_id(
                session.manual_pull_flow.available_target_layers,
                intent.target_layer_id,
                action_name="ConfirmPullFromMA3",
            )
            available_event_ids = {
                event.event_id for event in session.manual_pull_flow.available_events
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
            session.manual_pull_flow.dialog_open = False
            session.manual_pull_flow.workspace_active = True
            session.manual_pull_flow.selected_source_track_coords = [source_track.coord]
            session.manual_pull_flow.active_source_track_coord = source_track.coord
            session.manual_pull_flow.source_track_coord = source_track.coord
            session.manual_pull_flow.selected_ma3_event_ids = list(intent.selected_ma3_event_ids)
            session.manual_pull_flow.selected_ma3_event_ids_by_track[source_track.coord] = list(
                intent.selected_ma3_event_ids
            )
            derived_import_mode = self._manual_pull_import_mode_for_target_layer(
                timeline,
                target_option.layer_id,
            )
            session.manual_pull_flow.import_mode = derived_import_mode
            session.manual_pull_flow.import_mode_by_source_track[source_track.coord] = (
                derived_import_mode
            )
            session.manual_pull_flow.target_layer_id = target_option.layer_id
            session.manual_pull_flow.target_layer_id_by_source_track[source_track.coord] = (
                target_option.layer_id
            )
            session.manual_pull_flow.diff_gate_open = False
            session.manual_pull_flow.diff_preview = None
            if session.manual_pull_flow.workspace_active:
                self._rebuild_pull_transfer_plan(timeline, session)

        elif isinstance(intent, ApplyPullFromMA3):
            session = self.session_service.get_session()
            flow = session.manual_pull_flow
            if not flow.source_track_coord:
                raise ValueError("ApplyPullFromMA3 requires a selected source_track_coord")
            if not flow.selected_ma3_event_ids:
                raise ValueError("ApplyPullFromMA3 requires selected MA3 events")
            if flow.target_layer_id is None:
                raise ValueError("ApplyPullFromMA3 requires a selected target_layer_id")

            source_track = self._manual_pull_track_by_coord(
                flow.available_tracks,
                flow.source_track_coord,
                action_name="ApplyPullFromMA3",
            )
            selected_pull_events = self._manual_pull_selected_events(flow)
            resolved_target_layer = self._resolve_manual_pull_target_layer(
                timeline,
                target_layer_id=flow.target_layer_id,
                source_track=source_track,
            )
            resolved_import_mode = self._manual_pull_import_mode_for_target_layer(
                timeline,
                flow.target_layer_id,
            )
            flow.import_mode = resolved_import_mode
            selected_take_id, selected_event_ids = self._apply_manual_pull_import(
                target_layer=resolved_target_layer,
                source_track=source_track,
                selected_events=selected_pull_events,
                import_mode=resolved_import_mode,
            )

            flow.target_layer_id_by_source_track[source_track.coord] = resolved_target_layer.id
            flow.target_layer_id = resolved_target_layer.id
            flow.import_mode_by_source_track[source_track.coord] = (
                self._manual_pull_import_mode_for_target_layer(
                    timeline,
                    resolved_target_layer.id,
                )
            )
            flow.import_mode = flow.import_mode_by_source_track[source_track.coord]
            self._refresh_manual_pull_target_options(timeline, session)

            timeline.selection.selected_layer_id = resolved_target_layer.id
            timeline.selection.selected_layer_ids = [resolved_target_layer.id]
            timeline.selection.selected_take_id = selected_take_id
            self._set_selected_event_refs(
                timeline,
                self._resolve_event_refs_by_ids(
                    timeline,
                    selected_event_ids,
                    preferred_layer_ids=[resolved_target_layer.id],
                    preferred_take_id=selected_take_id,
                ),
            )

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
            session.manual_push_flow.selected_event_ids = list(
                timeline.selection.selected_event_ids
            )
            if not self._plan_counters_locked(session.batch_transfer_plan):
                self._rebuild_push_transfer_plan(timeline, session)
        timeline.section_cues = derive_section_cues_from_layers(timeline.layers)
        audibility = self.mixer_service.resolve_audibility(timeline.layers)
        self.playback_service.update_runtime(
            timeline=timeline,
            transport=session.transport_state,
            audibility=audibility,
            sync=session.sync_state,
        )
        return self.assembler.assemble(timeline=timeline, session=session)

    @staticmethod
    def _manual_pull_selected_events(flow: ManualPullFlowState) -> list[ManualPullEventOption]:
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
        available_by_id = {event.event_id: event for event in available_events}
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

    def _set_active_manual_pull_source_track(
        self,
        timeline: Timeline,
        session: Session,
        *,
        source_track_coord: str,
    ) -> None:
        flow = session.manual_pull_flow
        source_track = self._manual_pull_track_by_coord(
            flow.available_tracks,
            source_track_coord,
            action_name="_set_active_manual_pull_source_track",
        )
        selected_source_track_coords = list(flow.selected_source_track_coords)
        if source_track.coord not in selected_source_track_coords:
            selected_source_track_coords = [source_track.coord]
        flow.selected_source_track_coords = selected_source_track_coords or [source_track.coord]
        flow.active_source_track_coord = source_track.coord
        flow.source_track_coord = source_track.coord
        self._refresh_manual_pull_target_options(timeline, session)
        flow.available_events = self._load_manual_pull_event_options(source_track.coord)
        selected_ids = list(flow.selected_ma3_event_ids_by_track.get(source_track.coord, []))
        if not selected_ids:
            selected_ids = [event.event_id for event in flow.available_events]
        flow.selected_ma3_event_ids = selected_ids
        flow.selected_ma3_event_ids_by_track[source_track.coord] = list(selected_ids)
        target_layer_id = flow.target_layer_id_by_source_track.get(source_track.coord)
        valid_target_ids = {target.layer_id for target in flow.available_target_layers}
        if target_layer_id not in valid_target_ids:
            target_layer_id = self._default_manual_pull_target_layer_id(
                timeline,
                source_track_coord=source_track.coord,
            )
            flow.target_layer_id_by_source_track[source_track.coord] = target_layer_id
        flow.target_layer_id = target_layer_id
        import_mode = self._manual_pull_import_mode_for_target_layer(
            timeline,
            target_layer_id,
        )
        flow.import_mode = import_mode
        flow.import_mode_by_source_track[source_track.coord] = import_mode
        flow.diff_gate_open = False
        flow.diff_preview = None

    @staticmethod
    def _clear_manual_pull_source_selection(session: Session) -> None:
        flow = session.manual_pull_flow
        flow.selected_source_track_coords = []
        flow.active_source_track_coord = None
        flow.source_track_coord = None
        flow.available_events = []
        flow.selected_ma3_event_ids = []
        flow.target_layer_id = None
        flow.import_mode = "new_take"
        flow.diff_gate_open = False
        flow.diff_preview = None

    def _refresh_manual_pull_target_options(
        self,
        timeline: Timeline,
        session: Session,
    ) -> None:
        flow = session.manual_pull_flow
        flow.available_target_layers = self._load_manual_pull_target_options(
            timeline,
            include_create_per_source_track=len(flow.selected_source_track_coords) > 1,
        )
        valid_target_ids = {target.layer_id for target in flow.available_target_layers}
        flow.target_layer_id_by_source_track = {
            coord: target_layer_id
            for coord, target_layer_id in flow.target_layer_id_by_source_track.items()
            if target_layer_id in valid_target_ids
        }
        active_source_coord = flow.active_source_track_coord or flow.source_track_coord
        if active_source_coord is None:
            if flow.target_layer_id not in valid_target_ids:
                flow.target_layer_id = None
            return
        target_layer_id = flow.target_layer_id_by_source_track.get(active_source_coord)
        if target_layer_id is None:
            target_layer_id = self._default_manual_pull_target_layer_id(
                timeline,
                source_track_coord=active_source_coord,
            )
            flow.target_layer_id_by_source_track[active_source_coord] = target_layer_id
        flow.target_layer_id = target_layer_id
        flow.import_mode = self._manual_pull_import_mode_for_target_layer(
            timeline,
            target_layer_id,
        )
        flow.import_mode_by_source_track[active_source_coord] = flow.import_mode

    @staticmethod
    def _manual_pull_import_mode_for_target_layer(
        timeline: Timeline,
        target_layer_id: LayerId | None,
    ) -> str:
        if target_layer_id in {
            _PULL_TARGET_CREATE_NEW_LAYER_ID,
            _PULL_TARGET_CREATE_NEW_SECTION_LAYER_ID,
            _PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_ID,
        }:
            return "main"
        if target_layer_id is not None:
            for layer in timeline.layers:
                if layer.id == target_layer_id and layer.kind is LayerKind.SECTION:
                    return "main"
        return "new_take"

    def _default_manual_pull_timecode_no(
        self,
        timeline: Timeline,
        *,
        available_timecodes: list[ManualPullTimecodeOption],
    ) -> int | None:
        if not available_timecodes:
            return None
        preferred = self._ma3_track_coord_timecode_no(
            self._selected_layer_ma3_track_coord(timeline)
        )
        if preferred is None:
            preferred = self._active_song_version_ma3_timecode_pool_no()
        available_numbers = {timecode.number for timecode in available_timecodes}
        if preferred in available_numbers:
            return preferred
        return available_timecodes[0].number

    def _default_manual_pull_track_group_no(
        self,
        timeline: Timeline,
        *,
        selected_timecode_no: int | None,
        available_track_groups: list[ManualPullTrackGroupOption],
    ) -> int | None:
        if not available_track_groups:
            return None
        preferred_coord = self._selected_layer_ma3_track_coord(timeline)
        preferred_timecode_no = self._ma3_track_coord_timecode_no(preferred_coord)
        preferred_group_no = (
            self._ma3_track_coord_track_group_no(preferred_coord)
            if preferred_timecode_no == selected_timecode_no
            else None
        )
        available_numbers = {group.number for group in available_track_groups}
        if preferred_group_no in available_numbers:
            return preferred_group_no
        for group in available_track_groups:
            if (group.track_count or 0) > 0:
                return group.number
        return available_track_groups[0].number

    def _default_manual_pull_source_track_coord(
        self,
        timeline: Timeline,
        session: Session,
        *,
        available_tracks: list[ManualPullTrackOption],
        preferred_track_group_no: int | None = None,
    ) -> str | None:
        if not available_tracks:
            return None
        scoped_tracks = available_tracks
        if preferred_track_group_no is not None:
            scoped_tracks = [
                track
                for track in available_tracks
                if self._ma3_track_coord_track_group_no(track.coord) == preferred_track_group_no
            ]
        candidate_tracks = scoped_tracks or available_tracks
        available_by_coord = {track.coord: track for track in candidate_tracks}
        preferred_coord = self._selected_layer_ma3_track_coord(timeline)
        if preferred_coord in available_by_coord:
            return preferred_coord
        active_coord = session.manual_pull_flow.active_source_track_coord
        if active_coord in available_by_coord:
            return active_coord
        return candidate_tracks[0].coord

    def _default_manual_pull_target_layer_id(
        self,
        timeline: Timeline,
        *,
        source_track_coord: str,
    ) -> LayerId:
        linked_layer_id = self._linked_manual_pull_layer_id(
            timeline,
            source_track_coord=source_track_coord,
        )
        if linked_layer_id is not None:
            return linked_layer_id
        selected_layer_id = self._selected_manual_pull_target_layer_id(timeline)
        if selected_layer_id is not None:
            return selected_layer_id
        return _PULL_TARGET_CREATE_NEW_LAYER_ID

    @staticmethod
    def _selected_manual_pull_target_layer_id(timeline: Timeline) -> LayerId | None:
        selected_layer_id = timeline.selection.selected_layer_id
        if selected_layer_id is None:
            return None
        for layer in timeline.layers:
            if layer.id != selected_layer_id:
                continue
            if (
                is_event_like_layer_kind(layer.kind)
                and layer.presentation_hints.visible
                and not layer.presentation_hints.locked
            ):
                return layer.id
            return None
        return None

    @staticmethod
    def _selected_layer_ma3_track_coord(timeline: Timeline) -> str | None:
        selected_layer_id = timeline.selection.selected_layer_id
        if selected_layer_id is None:
            return None
        for layer in timeline.layers:
            if layer.id == selected_layer_id:
                return str(layer.sync.ma3_track_coord or "").strip() or None
        return None

    @staticmethod
    def _linked_manual_pull_layer_id(
        timeline: Timeline,
        *,
        source_track_coord: str,
    ) -> LayerId | None:
        for layer in timeline.layers:
            if str(layer.sync.ma3_track_coord or "").strip() == source_track_coord:
                return layer.id
        return None

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
        take_id: TakeId,
        source_track: ManualPullTrackOption,
        source_event: ManualPullEventOption,
        order_index: int,
        existing_event_ids: set[str] | None = None,
    ) -> Event:
        start, end = self.diff_service.resolve_pull_event_range(
            source_event, order_index=order_index
        )
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
            origin="ma3_pull",
            classifications={"label": source_event.label} if source_event.label else {},
            metadata={},
            cue_number=source_event.cue_number or 1,
            label=source_event.label,
            cue_ref=source_event.cue_ref
            or cue_number_text(source_event.cue_number),
            color=source_event.color,
            notes=source_event.notes,
            payload_ref=source_event.payload_ref or source_event.event_id,
        )

    @staticmethod
    def _next_manual_pull_event_id(
        *,
        take_id: TakeId,
        source_track_coord: str,
        source_event_id: str,
        order_index: int,
        existing_event_ids: set[str] | None,
    ) -> EventId:
        reserved = existing_event_ids if existing_event_ids is not None else set()
        base_id = f"{take_id}:ma3:{source_track_coord}:{source_event_id}:{order_index}"
        if base_id not in reserved:
            reserved.add(base_id)
            return EventId(base_id)
        suffix = 2
        while True:
            candidate = f"{base_id}:{suffix}"
            if candidate not in reserved:
                reserved.add(candidate)
                return EventId(candidate)
            suffix += 1

    def _handle_create_region(
        self,
        timeline: Timeline,
        *,
        time_range: TimeRange,
        label: str,
        color: str | None,
        kind: str,
    ) -> None:
        region = TimelineRegion(
            id=self._next_region_id(timeline),
            start=float(time_range.start),
            end=float(time_range.end),
            label=label,
            color=color,
            kind=kind,
            order_index=len(timeline.regions),
        )
        timeline.regions = self._sorted_regions([*timeline.regions, region])
        timeline.selection.selected_region_id = region.id
        timeline.end = max(float(timeline.end), float(region.end))

    def _handle_update_region(
        self,
        timeline: Timeline,
        *,
        region_id: RegionId,
        time_range: TimeRange,
        label: str,
        color: str | None,
        kind: str,
    ) -> None:
        existing = self._require_region(timeline, region_id)
        updated = TimelineRegion(
            id=existing.id,
            start=float(time_range.start),
            end=float(time_range.end),
            label=label,
            color=color,
            kind=kind,
            order_index=existing.order_index,
        )
        timeline.regions = self._sorted_regions(
            [
                updated if region.id == existing.id else region
                for region in timeline.regions
            ]
        )
        timeline.selection.selected_region_id = updated.id
        timeline.end = max(float(timeline.end), float(updated.end))

    def _handle_delete_region(
        self,
        timeline: Timeline,
        *,
        region_id: RegionId,
    ) -> None:
        self._require_region(timeline, region_id)
        timeline.regions = [
            region for region in timeline.regions if region.id != region_id
        ]
        if timeline.selection.selected_region_id == region_id:
            timeline.selection.selected_region_id = None
        timeline.regions = self._sorted_regions(list(timeline.regions))

    @staticmethod
    def _sorted_regions(regions: list[TimelineRegion]) -> list[TimelineRegion]:
        sorted_regions = sorted(
            regions,
            key=lambda region: (
                float(region.start),
                float(region.end),
                int(region.order_index),
                str(region.id),
            ),
        )
        return [
            TimelineRegion(
                id=region.id,
                start=float(region.start),
                end=float(region.end),
                label=region.label,
                color=region.color,
                kind=region.kind,
                order_index=index,
            )
            for index, region in enumerate(sorted_regions)
        ]

    @staticmethod
    def _next_region_id(timeline: Timeline) -> RegionId:
        existing_ids = {str(region.id) for region in timeline.regions}
        counter = 1
        while True:
            candidate = f"region_{counter}"
            if candidate not in existing_ids:
                return RegionId(candidate)
            counter += 1

    @staticmethod
    def _require_region(timeline: Timeline, region_id: RegionId) -> TimelineRegion:
        for region in timeline.regions:
            if region.id == region_id:
                return region
        raise ValueError(f"Region not found: {region_id}")
