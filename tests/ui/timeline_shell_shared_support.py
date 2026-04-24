"""Shared fixtures and harnesses for timeline-shell support cases.
Exists to keep common timeline presentations, runtime harnesses, and Qt hit-test helpers out of the compatibility wrapper.
Connects behavior-owned support case modules to one stable pool of test fixtures and interaction helpers.
"""

from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QContextMenuEvent, QImage, QMouseEvent, QPainter
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from echozero.application.presentation.inspector_contract import (
    InspectorAction,
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
    render_inspector_contract_text,
)
from echozero.application.presentation.models import (
    BatchTransferPlanPresentation,
    BatchTransferPlanRowPresentation,
    EventPresentation,
    LayerHeaderControlPresentation,
    LayerPresentation,
    LayerStatusPresentation,
    ManualPullDiffPreviewPresentation,
    ManualPullEventOptionPresentation,
    ManualPullFlowPresentation,
    ManualPullTargetOptionPresentation,
    ManualPullTrackOptionPresentation,
    ManualPushDiffPreviewPresentation,
    ManualPushFlowPresentation,
    ManualPushTimecodeOptionPresentation,
    ManualPushTrackGroupOptionPresentation,
    ManualPushSequenceOptionPresentation,
    ManualPushSequenceRangePresentation,
    ManualPushTrackOptionPresentation,
    PipelineRunBannerPresentation,
    SongOptionPresentation,
    SongVersionOptionPresentation,
    TakeLanePresentation,
    TimelinePresentation,
    TransferPresetPresentation,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId, TimelineId
from echozero.application.sync.models import LiveSyncState
from echozero.application.timeline.event_batch_scope import EventBatchScope
from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    ApplyTransferPlan,
    ApplyTransferPreset,
    CancelTransferPlan,
    ClearLayerLiveSyncPauseReason,
    ClearSelection,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    CreateRegion,
    CreateEvent,
    DeleteEvents,
    DeleteTransferPreset,
    DuplicateSelectedEvents,
    ExitPullFromMA3Workspace,
    ExitPushToMA3Mode,
    MoveSelectedEventsToAdjacentLayer,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    OpenPullFromMA3Dialog,
    OpenPushToMA3Dialog,
    Pause,
    Play,
    PreviewTransferPlan,
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
    SelectPullSourceTrack,
    SelectPullSourceTracks,
    SelectPullTargetLayer,
    SelectPushTargetTrack,
    SetActivePlaybackTarget,
    SetGain,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    SetPullImportMode,
    SetPushTransferMode,
    SetSelectedEvents,
    Stop,
    ToggleLayerExpanded,
)
from echozero.application.timeline.ma3_push_intents import (
    AssignMA3TrackSequence,
    CreateMA3Sequence,
    MA3PushApplyMode,
    MA3PushScope,
    MA3PushTargetMode,
    MA3SequenceCreationMode,
    MA3SequenceRefreshRangeMode,
    PushLayerToMA3,
    RefreshMA3Sequences,
    RefreshMA3PushTracks,
    SetLayerMA3Route,
)
from echozero.application.timeline.models import EventRef
from echozero.ui.qt.timeline.blocks.layer_header import HeaderSlots, LayerHeaderBlock
from echozero.ui.qt.timeline.blocks.ruler import timeline_x_for_time
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.test_harness import (
    build_variant_presentations,
    estimate_full_window_height,
)
from echozero.ui.qt.timeline.widget import (
    ManualPullTimelineDialog,
    ManualPullTimelineSelectionResult,
    TimelineWidget,
    compute_scroll_bounds,
    estimate_timeline_span_seconds,
)


def _selection_test_presentation() -> TimelinePresentation:
    layer_id = LayerId("layer_kick")
    main_take_id = TakeId("take_main")
    alt_take_id = TakeId("take_alt")
    return TimelinePresentation(
        timeline_id=TimelineId("timeline_selection"),
        title="Selection",
        layers=[
            LayerPresentation(
                layer_id=layer_id,
                title="Kick",
                main_take_id=main_take_id,
                kind=LayerKind.EVENT,
                is_expanded=True,
                events=[
                    EventPresentation(
                        event_id=EventId("main_evt"),
                        start=1.0,
                        end=1.5,
                        label="Main",
                    )
                ],
                takes=[
                    TakeLanePresentation(
                        take_id=alt_take_id,
                        name="Take 2",
                        kind=LayerKind.EVENT,
                        events=[
                            EventPresentation(
                                event_id=EventId("take_evt"),
                                start=2.0,
                                end=2.5,
                                label="Take",
                            )
                        ],
                    )
                ],
                status=LayerStatusPresentation(),
            )
        ],
        pixels_per_second=100.0,
        end_time_label="00:05.00",
    )


def _audio_pipeline_presentation() -> TimelinePresentation:
    return TimelinePresentation(
        timeline_id=TimelineId("timeline_audio_pipeline"),
        title="Audio Pipeline",
        layers=[
            LayerPresentation(
                layer_id=LayerId("layer_song"),
                title="Song",
                main_take_id=TakeId("take_song"),
                kind=LayerKind.AUDIO,
                is_selected=True,
                badges=["main", "audio"],
                status=LayerStatusPresentation(),
            ),
            LayerPresentation(
                layer_id=LayerId("layer_drums"),
                title="Drums",
                main_take_id=TakeId("take_drums"),
                kind=LayerKind.AUDIO,
                badges=["main", "audio", "drums"],
                status=LayerStatusPresentation(),
            ),
        ],
        selected_layer_id=LayerId("layer_song"),
        selected_layer_ids=[LayerId("layer_song")],
        end_time_label="00:05.00",
    )


def _song_switching_presentation() -> TimelinePresentation:
    presentation = _audio_pipeline_presentation()
    presentation.active_song_id = "song_alpha"
    presentation.active_song_title = "Alpha Song"
    presentation.active_song_version_id = "song_version_festival"
    presentation.active_song_version_label = "Festival Edit"
    presentation.available_songs = [
        SongOptionPresentation(
            song_id="song_alpha",
            title="Alpha Song",
            is_active=True,
            active_version_id="song_version_festival",
            active_version_label="Festival Edit",
            version_count=2,
            versions=[
                SongVersionOptionPresentation(
                    song_version_id="song_version_original",
                    label="Original",
                ),
                SongVersionOptionPresentation(
                    song_version_id="song_version_festival",
                    label="Festival Edit",
                    is_active=True,
                ),
            ],
        ),
        SongOptionPresentation(
            song_id="song_beta",
            title="Beta Song",
            active_version_id="song_version_beta",
            active_version_label="Original",
            version_count=1,
            versions=[
                SongVersionOptionPresentation(
                    song_version_id="song_version_beta",
                    label="Original",
                    is_active=True,
                )
            ],
        ),
    ]
    presentation.available_song_versions = [
        SongVersionOptionPresentation(song_version_id="song_version_original", label="Original"),
        SongVersionOptionPresentation(
            song_version_id="song_version_festival",
            label="Festival Edit",
            is_active=True,
        ),
    ]
    return presentation


def _no_takes_presentation() -> TimelinePresentation:
    base = _selection_test_presentation()
    layer = base.layers[0]
    return replace(
        base,
        layers=[replace(layer, takes=[])],
    )


def _drag_test_presentation() -> TimelinePresentation:
    source = _selection_test_presentation()
    target_layer_id = LayerId("layer_snare")
    return replace(
        source,
        selected_layer_id=LayerId("layer_kick"),
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt")],
        layers=[
            replace(
                source.layers[0],
                events=[
                    replace(source.layers[0].events[0], is_selected=True),
                ],
            ),
            LayerPresentation(
                layer_id=target_layer_id,
                title="Snare",
                main_take_id=TakeId("take_snare_main"),
                kind=LayerKind.EVENT,
                is_expanded=False,
                events=[
                    EventPresentation(
                        event_id=EventId("snare_evt"),
                        start=3.0,
                        end=3.5,
                        label="Snare",
                    )
                ],
                status=LayerStatusPresentation(),
            ),
        ],
    )


def _multi_layer_selection_presentation() -> TimelinePresentation:
    base = _selection_test_presentation()
    return replace(
        base,
        layers=[
            base.layers[0],
            LayerPresentation(
                layer_id=LayerId("layer_snare"),
                title="Snare",
                main_take_id=TakeId("take_snare"),
                kind=LayerKind.EVENT,
                events=[
                    EventPresentation(
                        event_id=EventId("snare_evt"),
                        start=3.0,
                        end=3.5,
                        label="Snare",
                    )
                ],
                status=LayerStatusPresentation(),
            ),
            LayerPresentation(
                layer_id=LayerId("layer_hat"),
                title="Hat",
                main_take_id=TakeId("take_hat"),
                kind=LayerKind.EVENT,
                events=[
                    EventPresentation(
                        event_id=EventId("hat_evt"),
                        start=4.0,
                        end=4.5,
                        label="Hat",
                    )
                ],
                status=LayerStatusPresentation(),
            ),
        ],
    )


def _no_takes_layer_presentation() -> TimelinePresentation:
    return TimelinePresentation(
        timeline_id=TimelineId("timeline_empty"),
        title="Empty",
        layers=[
            LayerPresentation(
                layer_id=LayerId("layer_empty"),
                title="Empty",
                main_take_id=None,
                kind=LayerKind.EVENT,
                events=[],
                takes=[],
                status=LayerStatusPresentation(),
            )
        ],
        pixels_per_second=100.0,
        end_time_label="00:05.00",
    )


class _SelectionInspectorHarness:
    def __init__(self, presentation: TimelinePresentation):
        self._presentation = presentation

    def presentation(self) -> TimelinePresentation:
        return self._presentation

    def dispatch(self, intent):
        if isinstance(intent, SelectLayer):
            selected_ids = [intent.layer_id] if intent.layer_id is not None else []
            if intent.mode == "toggle" and intent.layer_id is not None:
                current_ids = list(self._presentation.selected_layer_ids) or (
                    [self._presentation.selected_layer_id]
                    if self._presentation.selected_layer_id is not None
                    else []
                )
                if intent.layer_id in current_ids:
                    selected_ids = [
                        layer_id for layer_id in current_ids if layer_id != intent.layer_id
                    ]
                else:
                    selected_ids = [*current_ids, intent.layer_id]
            elif intent.mode == "range" and intent.layer_id is not None:
                ordered_ids = [layer.layer_id for layer in self._presentation.layers]
                anchor_id = self._presentation.selected_layer_id or intent.layer_id
                low, high = sorted(
                    (ordered_ids.index(anchor_id), ordered_ids.index(intent.layer_id))
                )
                selected_ids = ordered_ids[low : high + 1]
            self._presentation = replace(
                self._presentation,
                layers=[
                    replace(layer, is_selected=(layer.layer_id in selected_ids))
                    for layer in self._presentation.layers
                ],
                selected_layer_id=intent.layer_id if selected_ids else None,
                selected_layer_ids=selected_ids,
                selected_take_id=None,
                selected_event_ids=[],
            )
            return self._presentation

        if isinstance(intent, SelectEvent):
            layers = []
            for layer in self._presentation.layers:
                is_target_layer = layer.layer_id == intent.layer_id
                layers.append(
                    replace(
                        layer,
                        is_selected=is_target_layer,
                        events=[
                            replace(
                                event,
                                is_selected=(
                                    is_target_layer
                                    and intent.take_id == layer.main_take_id
                                    and event.event_id == intent.event_id
                                ),
                            )
                            for event in layer.events
                        ],
                        takes=[
                            replace(
                                take,
                                events=[
                                    replace(
                                        event,
                                        is_selected=(
                                            is_target_layer
                                            and take.take_id == intent.take_id
                                            and event.event_id == intent.event_id
                                        ),
                                    )
                                    for event in take.events
                                ],
                            )
                            for take in layer.takes
                        ],
                    )
                )
            self._presentation = replace(
                self._presentation,
                layers=layers,
                selected_layer_id=intent.layer_id,
                selected_layer_ids=[intent.layer_id],
                selected_take_id=intent.take_id,
                selected_event_ids=[] if intent.event_id is None else [intent.event_id],
            )
            return self._presentation

        if isinstance(intent, ClearSelection):
            self._presentation = replace(
                self._presentation,
                layers=[
                    replace(
                        layer,
                        is_selected=False,
                        events=[replace(event, is_selected=False) for event in layer.events],
                        takes=[
                            replace(
                                take,
                                events=[
                                    replace(event, is_selected=False) for event in take.events
                                ],
                            )
                            for take in layer.takes
                        ],
                    )
                    for layer in self._presentation.layers
                ],
                selected_layer_id=None,
                selected_layer_ids=[],
                selected_take_id=None,
                selected_event_ids=[],
            )
            return self._presentation

        return self._presentation


class _ManualPushHarness:
    def __init__(self, presentation: TimelinePresentation):
        self._presentation = presentation
        self.intents: list[object] = []

    def presentation(self) -> TimelinePresentation:
        return self._presentation

    def dispatch(self, intent):
        self.intents.append(intent)
        if isinstance(intent, RefreshMA3PushTracks):
            available_tracks = [
                ManualPushTrackOptionPresentation(
                    coord="tc1_tg2_tr3",
                    name="Track 3",
                    note="Bass",
                    event_count=8,
                    sequence_no=12,
                ),
                ManualPushTrackOptionPresentation(
                    coord="tc1_tg2_tr9",
                    name="Track 9",
                    note="Alt",
                    event_count=2,
                    sequence_no=None,
                ),
            ]
            if intent.timecode_no is not None:
                available_tracks = [
                    track
                    for track in available_tracks
                    if track.coord.startswith(f"tc{int(intent.timecode_no)}_")
                ]
            if intent.track_group_no is not None:
                available_tracks = [
                    track
                    for track in available_tracks
                    if f"_tg{int(intent.track_group_no)}_" in track.coord
                ]
            self._presentation = replace(
                self._presentation,
                manual_push_flow=replace(
                    self._presentation.manual_push_flow,
                    available_timecodes=[
                        ManualPushTimecodeOptionPresentation(number=1, name=None),
                    ],
                    selected_timecode_no=(
                        int(intent.timecode_no)
                        if intent.timecode_no is not None
                        else 1
                    ),
                    available_track_groups=[
                        ManualPushTrackGroupOptionPresentation(
                            number=2,
                            name="Group 2",
                            track_count=2,
                        ),
                    ],
                    selected_track_group_no=(
                        int(intent.track_group_no)
                        if intent.track_group_no is not None
                        else 2
                    ),
                    available_tracks=available_tracks,
                ),
            )
            return self._presentation
        if isinstance(intent, RefreshMA3Sequences):
            available_sequences = [
                ManualPushSequenceOptionPresentation(number=12, name="Song A"),
                ManualPushSequenceOptionPresentation(number=15, name="Lead Stack"),
            ]
            if intent.range_mode is MA3SequenceRefreshRangeMode.CURRENT_SONG:
                available_sequences = [available_sequences[0]]
            self._presentation = replace(
                self._presentation,
                manual_push_flow=replace(
                    self._presentation.manual_push_flow,
                    available_sequences=available_sequences,
                    current_song_sequence_range=ManualPushSequenceRangePresentation(
                        start=12,
                        end=111,
                        song_label="Song A",
                    ),
                ),
            )
            return self._presentation
        if isinstance(intent, SetLayerMA3Route):
            assigned_sequence_no: int | None = None
            if isinstance(intent.sequence_action, AssignMA3TrackSequence):
                assigned_sequence_no = intent.sequence_action.sequence_no
            elif isinstance(intent.sequence_action, CreateMA3Sequence):
                assigned_sequence_no = (
                    200
                    if intent.sequence_action.creation_mode
                    is MA3SequenceCreationMode.NEXT_AVAILABLE
                    else 45
                )
            self._presentation = replace(
                self._presentation,
                layers=[
                    replace(
                        layer,
                        sync_target_label=(
                            intent.target_track_coord
                            if layer.layer_id == intent.layer_id
                            else layer.sync_target_label
                        ),
                    )
                    for layer in self._presentation.layers
                ],
                manual_push_flow=replace(
                    self._presentation.manual_push_flow,
                    available_tracks=[
                        replace(
                            track,
                            sequence_no=(
                                assigned_sequence_no
                                if track.coord == intent.target_track_coord
                                and assigned_sequence_no is not None
                                else track.sequence_no
                            ),
                        )
                        for track in self._presentation.manual_push_flow.available_tracks
                    ],
                ),
            )
            return self._presentation
        if isinstance(intent, PushLayerToMA3):
            return self._presentation
        if isinstance(intent, OpenPushToMA3Dialog):
            self._presentation = replace(
                self._presentation,
                layers=[
                    replace(
                        layer,
                        push_selection_count=(
                            len(self._presentation.selected_event_ids)
                            if layer.layer_id == self._presentation.selected_layer_id
                            else 0
                        ),
                        push_row_status=(
                            "blocked"
                            if layer.layer_id == self._presentation.selected_layer_id
                            else ""
                        ),
                        push_row_issue=(
                            "Select an MA3 target track"
                            if layer.layer_id == self._presentation.selected_layer_id
                            else ""
                        ),
                    )
                    for layer in self._presentation.layers
                ],
                manual_push_flow=ManualPushFlowPresentation(
                    dialog_open=False,
                    push_mode_active=True,
                    selected_layer_ids=list(self._presentation.selected_layer_ids)
                    or [LayerId("layer_kick")],
                    available_tracks=[
                        ManualPushTrackOptionPresentation(
                            coord="tc1_tg2_tr3",
                            name="Track 3",
                            note="Bass",
                            event_count=8,
                            sequence_no=12,
                        )
                    ],
                    target_track_coord=None,
                    diff_gate_open=False,
                    diff_preview=None,
                ),
                batch_transfer_plan=BatchTransferPlanPresentation(
                    plan_id="push:timeline_selection",
                    operation_type="push",
                    rows=[
                        BatchTransferPlanRowPresentation(
                            row_id="push:layer_kick",
                            direction="push",
                            source_label="Kick",
                            target_label="Unmapped",
                            source_layer_id=LayerId("layer_kick"),
                            selected_event_ids=list(self._presentation.selected_event_ids),
                            selected_count=len(self._presentation.selected_event_ids),
                            status="blocked",
                            issue="Select an MA3 target track",
                        )
                    ],
                    blocked_count=1,
                ),
            )
            return self._presentation
        if isinstance(intent, SelectPushTargetTrack):
            selected_event_ids = list(self._presentation.selected_event_ids)
            if not selected_event_ids and self._presentation.batch_transfer_plan is not None:
                selected_event_ids = list(
                    self._presentation.batch_transfer_plan.rows[0].selected_event_ids
                )
            self._presentation = replace(
                self._presentation,
                layers=[
                    replace(
                        layer,
                        push_target_label=(
                            "Track 3 (tc1_tg2_tr3) - Bass"
                            if layer.layer_id == self._presentation.selected_layer_id
                            else layer.push_target_label
                        ),
                        push_selection_count=(
                            len(selected_event_ids)
                            if layer.layer_id == self._presentation.selected_layer_id
                            else layer.push_selection_count
                        ),
                        push_row_status=(
                            "ready"
                            if layer.layer_id == self._presentation.selected_layer_id
                            else layer.push_row_status
                        ),
                        push_row_issue=(
                            ""
                            if layer.layer_id == self._presentation.selected_layer_id
                            else layer.push_row_issue
                        ),
                    )
                    for layer in self._presentation.layers
                ],
                manual_push_flow=replace(
                    self._presentation.manual_push_flow,
                    target_track_coord=intent.target_track_coord,
                ),
                batch_transfer_plan=BatchTransferPlanPresentation(
                    plan_id="push:timeline_selection",
                    operation_type="push",
                    rows=[
                        BatchTransferPlanRowPresentation(
                            row_id="push:layer_kick",
                            direction="push",
                            source_label="Kick",
                            target_label="Track 3 (tc1_tg2_tr3) - Bass",
                            source_layer_id=LayerId("layer_kick"),
                            target_track_coord=intent.target_track_coord,
                            selected_event_ids=selected_event_ids,
                            selected_count=len(selected_event_ids),
                            status="ready",
                        )
                    ],
                    ready_count=1,
                ),
            )
            return self._presentation
        if isinstance(intent, SetPushTransferMode):
            self._presentation = replace(
                self._presentation,
                manual_push_flow=replace(
                    self._presentation.manual_push_flow,
                    transfer_mode=intent.mode,
                ),
            )
            return self._presentation
        if isinstance(intent, ConfirmPushToMA3):
            self._presentation = replace(
                self._presentation,
                layers=[
                    replace(
                        layer,
                        push_target_label=(
                            "Track 3 (tc1_tg2_tr3) - Bass"
                            if layer.layer_id == self._presentation.selected_layer_id
                            else layer.push_target_label
                        ),
                        push_selection_count=(
                            len(intent.selected_event_ids)
                            if layer.layer_id == self._presentation.selected_layer_id
                            else layer.push_selection_count
                        ),
                        push_row_status=(
                            "ready"
                            if layer.layer_id == self._presentation.selected_layer_id
                            else layer.push_row_status
                        ),
                        push_row_issue=(
                            ""
                            if layer.layer_id == self._presentation.selected_layer_id
                            else layer.push_row_issue
                        ),
                    )
                    for layer in self._presentation.layers
                ],
                manual_push_flow=ManualPushFlowPresentation(
                    dialog_open=False,
                    push_mode_active=True,
                    selected_layer_ids=list(
                        self._presentation.manual_push_flow.selected_layer_ids
                    ),
                    available_tracks=list(self._presentation.manual_push_flow.available_tracks),
                    available_sequences=list(
                        self._presentation.manual_push_flow.available_sequences
                    ),
                    current_song_sequence_range=(
                        self._presentation.manual_push_flow.current_song_sequence_range
                    ),
                    target_track_coord=intent.target_track_coord,
                    diff_gate_open=True,
                    diff_preview=ManualPushDiffPreviewPresentation(
                        selected_count=len(intent.selected_event_ids),
                        target_track_coord=intent.target_track_coord,
                        target_track_name="Track 3",
                        target_track_note="Bass",
                        target_track_event_count=8,
                    ),
                ),
            )
            return self._presentation
        if isinstance(intent, SaveTransferPreset):
            next_index = len(self._presentation.transfer_presets) + 1
            self._presentation = replace(
                self._presentation,
                transfer_presets=[
                    *self._presentation.transfer_presets,
                    TransferPresetPresentation(
                        preset_id=f"preset-{next_index}",
                        name=intent.name,
                        push_target_mapping_by_layer_id={LayerId("layer_kick"): "tc1_tg2_tr3"},
                    ),
                ],
            )
            return self._presentation
        if isinstance(intent, ApplyTransferPreset):
            return self._presentation
        if isinstance(intent, DeleteTransferPreset):
            self._presentation = replace(
                self._presentation,
                transfer_presets=[
                    preset
                    for preset in self._presentation.transfer_presets
                    if preset.preset_id != intent.preset_id
                ],
            )
            return self._presentation
        if isinstance(intent, PreviewTransferPlan):
            return self._presentation
        if isinstance(intent, ApplyTransferPlan):
            plan = self._presentation.batch_transfer_plan
            rows = (
                []
                if plan is None
                else [
                    replace(
                        row,
                        status="applied" if row.status == "ready" else row.status,
                        issue=None if row.status == "ready" else row.issue,
                    )
                    for row in plan.rows
                ]
            )
            self._presentation = replace(
                self._presentation,
                batch_transfer_plan=(
                    None
                    if plan is None
                    else replace(
                        plan,
                        rows=rows,
                        ready_count=0,
                        applied_count=sum(1 for row in rows if row.status == "applied"),
                        failed_count=sum(1 for row in rows if row.status == "failed"),
                        blocked_count=sum(1 for row in rows if row.status == "blocked"),
                    )
                ),
            )
            return self._presentation
        if isinstance(intent, CancelTransferPlan):
            self._presentation = replace(
                self._presentation,
                manual_push_flow=ManualPushFlowPresentation(),
                batch_transfer_plan=None,
            )
            return self._presentation
        if isinstance(intent, ExitPushToMA3Mode):
            self._presentation = replace(
                self._presentation,
                manual_push_flow=ManualPushFlowPresentation(),
                batch_transfer_plan=None,
            )
            return self._presentation
        return self._presentation


class _ManualPullHarness:
    def __init__(self, presentation: TimelinePresentation):
        self._presentation = presentation
        self.intents: list[object] = []

    def presentation(self) -> TimelinePresentation:
        return self._presentation

    def dispatch(self, intent):
        self.intents.append(intent)
        if isinstance(intent, OpenPullFromMA3Dialog):
            self._presentation = replace(
                self._presentation,
                manual_pull_flow=ManualPullFlowPresentation(
                    dialog_open=False,
                    workspace_active=True,
                    available_tracks=[
                        ManualPullTrackOptionPresentation(
                            coord="tc1_tg2_tr3",
                            name="Track 3",
                            note="Lead",
                            event_count=2,
                        )
                    ],
                    import_mode="new_take",
                    available_target_layers=[
                        ManualPullTargetOptionPresentation(
                            layer_id=LayerId("layer_kick"),
                            name="Kick",
                        )
                    ],
                ),
            )
            return self._presentation
        if isinstance(intent, SelectPullSourceTracks):
            self._presentation = replace(
                self._presentation,
                manual_pull_flow=replace(
                    self._presentation.manual_pull_flow,
                    selected_source_track_coords=list(intent.source_track_coords),
                    active_source_track_coord=intent.source_track_coords[-1],
                ),
                batch_transfer_plan=BatchTransferPlanPresentation(
                    plan_id="pull:timeline_selection",
                    operation_type="pull",
                    rows=[
                        BatchTransferPlanRowPresentation(
                            row_id="pull:tc1_tg2_tr3",
                            direction="pull",
                            source_label="Track 3 (tc1_tg2_tr3)",
                            target_label="Unmapped",
                            source_track_coord="tc1_tg2_tr3",
                            import_mode=self._presentation.manual_pull_flow.import_mode,
                            selected_count=0,
                            status="blocked",
                            issue="Select source events and target layer mapping",
                        )
                    ],
                    blocked_count=1,
                ),
            )
            return self._presentation
        if isinstance(intent, SelectPullSourceTrack):
            self._presentation = replace(
                self._presentation,
                manual_pull_flow=replace(
                    self._presentation.manual_pull_flow,
                    selected_source_track_coords=(
                        list(self._presentation.manual_pull_flow.selected_source_track_coords)
                        if self._presentation.manual_pull_flow.selected_source_track_coords
                        else [intent.source_track_coord]
                    ),
                    active_source_track_coord=intent.source_track_coord,
                    source_track_coord=intent.source_track_coord,
                    available_events=[
                        ManualPullEventOptionPresentation(
                            event_id="ma3_evt_1",
                            label="Cue 1",
                            start=1.0,
                            end=1.5,
                        ),
                        ManualPullEventOptionPresentation(
                            event_id="ma3_evt_2",
                            label="Cue 2",
                            start=2.0,
                            end=2.5,
                        ),
                    ],
                ),
            )
            return self._presentation
        if isinstance(intent, SelectPullSourceEvents):
            self._presentation = replace(
                self._presentation,
                manual_pull_flow=replace(
                    self._presentation.manual_pull_flow,
                    selected_ma3_event_ids=list(intent.selected_ma3_event_ids),
                    selected_ma3_event_ids_by_track={
                        **self._presentation.manual_pull_flow.selected_ma3_event_ids_by_track,
                        self._presentation.manual_pull_flow.active_source_track_coord: list(
                            intent.selected_ma3_event_ids
                        ),
                    },
                ),
                batch_transfer_plan=BatchTransferPlanPresentation(
                    plan_id="pull:timeline_selection",
                    operation_type="pull",
                    rows=[
                        BatchTransferPlanRowPresentation(
                            row_id="pull:tc1_tg2_tr3",
                            direction="pull",
                            source_label="Track 3 (tc1_tg2_tr3)",
                            target_label="Unmapped",
                            source_track_coord="tc1_tg2_tr3",
                            import_mode=self._presentation.manual_pull_flow.import_mode,
                            selected_ma3_event_ids=list(intent.selected_ma3_event_ids),
                            selected_count=len(intent.selected_ma3_event_ids),
                            status="blocked",
                            issue="Select target layer mapping",
                        )
                    ],
                    blocked_count=1,
                ),
            )
            return self._presentation
        if isinstance(intent, SetPullImportMode):
            active_coord = self._presentation.manual_pull_flow.active_source_track_coord
            self._presentation = replace(
                self._presentation,
                manual_pull_flow=replace(
                    self._presentation.manual_pull_flow,
                    import_mode=intent.import_mode,
                    import_mode_by_source_track=(
                        self._presentation.manual_pull_flow.import_mode_by_source_track
                        if active_coord is None
                        else {
                            **self._presentation.manual_pull_flow.import_mode_by_source_track,
                            active_coord: intent.import_mode,
                        }
                    ),
                ),
                batch_transfer_plan=BatchTransferPlanPresentation(
                    plan_id="pull:timeline_selection",
                    operation_type="pull",
                    rows=[
                        BatchTransferPlanRowPresentation(
                            row_id="pull:tc1_tg2_tr3",
                            direction="pull",
                            source_label="Track 3 (tc1_tg2_tr3)",
                            target_label=(
                                "Kick"
                                if self._presentation.manual_pull_flow.target_layer_id
                                == LayerId("layer_kick")
                                else "Unmapped"
                            ),
                            source_track_coord="tc1_tg2_tr3",
                            target_layer_id=self._presentation.manual_pull_flow.target_layer_id,
                            import_mode=intent.import_mode,
                            selected_ma3_event_ids=list(
                                self._presentation.manual_pull_flow.selected_ma3_event_ids
                            ),
                            selected_count=len(
                                self._presentation.manual_pull_flow.selected_ma3_event_ids
                            ),
                            status=(
                                "ready"
                                if self._presentation.manual_pull_flow.target_layer_id is not None
                                and self._presentation.manual_pull_flow.selected_ma3_event_ids
                                else "blocked"
                            ),
                            issue=(
                                None
                                if self._presentation.manual_pull_flow.target_layer_id is not None
                                and self._presentation.manual_pull_flow.selected_ma3_event_ids
                                else "Select target layer mapping"
                            ),
                        )
                    ],
                    ready_count=(
                        1
                        if self._presentation.manual_pull_flow.target_layer_id is not None
                        and self._presentation.manual_pull_flow.selected_ma3_event_ids
                        else 0
                    ),
                    blocked_count=(
                        0
                        if self._presentation.manual_pull_flow.target_layer_id is not None
                        and self._presentation.manual_pull_flow.selected_ma3_event_ids
                        else 1
                    ),
                ),
            )
            return self._presentation
        if isinstance(intent, SelectPullTargetLayer):
            self._presentation = replace(
                self._presentation,
                manual_pull_flow=replace(
                    self._presentation.manual_pull_flow,
                    target_layer_id=intent.target_layer_id,
                    target_layer_id_by_source_track={
                        **self._presentation.manual_pull_flow.target_layer_id_by_source_track,
                        self._presentation.manual_pull_flow.active_source_track_coord: intent.target_layer_id,
                    },
                ),
                layers=[
                    replace(
                        layer,
                        pull_target_label=(
                            "Kick"
                            if layer.layer_id == intent.target_layer_id
                            else layer.pull_target_label
                        ),
                        pull_selection_count=(
                            len(self._presentation.manual_pull_flow.selected_ma3_event_ids)
                            if layer.layer_id == intent.target_layer_id
                            else layer.pull_selection_count
                        ),
                        pull_row_status=(
                            "ready"
                            if layer.layer_id == intent.target_layer_id
                            else layer.pull_row_status
                        ),
                        pull_row_issue=(
                            ""
                            if layer.layer_id == intent.target_layer_id
                            else layer.pull_row_issue
                        ),
                    )
                    for layer in self._presentation.layers
                ],
                batch_transfer_plan=BatchTransferPlanPresentation(
                    plan_id="pull:timeline_selection",
                    operation_type="pull",
                    rows=[
                        BatchTransferPlanRowPresentation(
                            row_id="pull:tc1_tg2_tr3",
                            direction="pull",
                            source_label="Track 3 (tc1_tg2_tr3)",
                            target_label="Kick",
                            source_track_coord="tc1_tg2_tr3",
                            target_layer_id=intent.target_layer_id,
                            import_mode=self._presentation.manual_pull_flow.import_mode,
                            selected_ma3_event_ids=list(
                                self._presentation.manual_pull_flow.selected_ma3_event_ids
                            ),
                            selected_count=len(
                                self._presentation.manual_pull_flow.selected_ma3_event_ids
                            ),
                            status="ready",
                        )
                    ],
                    ready_count=1,
                ),
            )
            return self._presentation
        if isinstance(intent, ConfirmPullFromMA3):
            self._presentation = replace(
                self._presentation,
                manual_pull_flow=ManualPullFlowPresentation(
                    dialog_open=False,
                    workspace_active=True,
                    available_tracks=list(self._presentation.manual_pull_flow.available_tracks),
                    selected_source_track_coords=list(
                        self._presentation.manual_pull_flow.selected_source_track_coords
                    ),
                    active_source_track_coord=intent.source_track_coord,
                    source_track_coord=intent.source_track_coord,
                    available_events=list(self._presentation.manual_pull_flow.available_events),
                    selected_ma3_event_ids=list(intent.selected_ma3_event_ids),
                    selected_ma3_event_ids_by_track={
                        **self._presentation.manual_pull_flow.selected_ma3_event_ids_by_track,
                        intent.source_track_coord: list(intent.selected_ma3_event_ids),
                    },
                    import_mode=intent.import_mode,
                    import_mode_by_source_track={
                        **self._presentation.manual_pull_flow.import_mode_by_source_track,
                        intent.source_track_coord: intent.import_mode,
                    },
                    available_target_layers=list(
                        self._presentation.manual_pull_flow.available_target_layers
                    ),
                    target_layer_id=intent.target_layer_id,
                    target_layer_id_by_source_track={
                        **self._presentation.manual_pull_flow.target_layer_id_by_source_track,
                        intent.source_track_coord: intent.target_layer_id,
                    },
                    diff_gate_open=True,
                    diff_preview=ManualPullDiffPreviewPresentation(
                        selected_count=len(intent.selected_ma3_event_ids),
                        source_track_coord=intent.source_track_coord,
                        source_track_name="Track 3",
                        source_track_note="Lead",
                        source_track_event_count=2,
                        target_layer_id=intent.target_layer_id,
                        target_layer_name="Kick",
                        import_mode=intent.import_mode,
                    ),
                ),
            )
            return self._presentation
        if isinstance(intent, PreviewTransferPlan):
            return self._presentation
        if isinstance(intent, ApplyTransferPlan):
            plan = self._presentation.batch_transfer_plan
            rows = (
                []
                if plan is None
                else [
                    replace(
                        row,
                        status="applied" if row.status == "ready" else row.status,
                        issue=None if row.status == "ready" else row.issue,
                    )
                    for row in plan.rows
                ]
            )
            self._presentation = replace(
                self._presentation,
                batch_transfer_plan=(
                    None
                    if plan is None
                    else replace(
                        plan,
                        rows=rows,
                        ready_count=0,
                        applied_count=sum(1 for row in rows if row.status == "applied"),
                        failed_count=sum(1 for row in rows if row.status == "failed"),
                        blocked_count=sum(1 for row in rows if row.status == "blocked"),
                    )
                ),
            )
            return self._presentation
        if isinstance(intent, CancelTransferPlan):
            self._presentation = replace(
                self._presentation,
                manual_pull_flow=ManualPullFlowPresentation(),
                batch_transfer_plan=None,
            )
            return self._presentation
        if isinstance(intent, ApplyPullFromMA3):
            self._presentation = replace(
                self._presentation,
                manual_pull_flow=replace(
                    self._presentation.manual_pull_flow,
                    diff_gate_open=False,
                    diff_preview=None,
                ),
            )
            return self._presentation
        if isinstance(intent, ExitPullFromMA3Workspace):
            self._presentation = replace(
                self._presentation,
                manual_pull_flow=ManualPullFlowPresentation(),
                batch_transfer_plan=None,
            )
            return self._presentation
        return self._presentation


def _render_for_hit_testing(widget: TimelineWidget) -> None:
    widget.resize(1200, 320)
    widget.show()
    widget.activateWindow()
    widget.setFocus()
    widget.repaint()
    QApplication.processEvents()
    widget._canvas.repaint()
    QApplication.processEvents()


def _click_event_rect(
    widget: TimelineWidget,
    event_id: str,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    for rect, _, _, candidate_event_id in widget._canvas._event_rects:
        if str(candidate_event_id) == event_id:
            center = rect.center().toPoint()
            QTest.mouseClick(
                widget._canvas,
                Qt.MouseButton.LeftButton,
                modifiers,
                QPoint(center.x(), center.y()),
            )
            QApplication.processEvents()
            return
    raise AssertionError(f"Missing event rect for {event_id}")


def _click_rect(
    widget: TimelineWidget, rect, modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier
) -> None:
    center = rect.center().toPoint()
    QTest.mouseClick(
        widget._canvas, Qt.MouseButton.LeftButton, modifiers, QPoint(center.x(), center.y())
    )
    QApplication.processEvents()


def _click_transport_rect(widget: TimelineWidget, key: str) -> None:
    rect = widget._transport._control_rects[key]
    center = rect.center().toPoint()
    QTest.mouseClick(
        widget._transport,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(center.x(), center.y()),
    )
    QApplication.processEvents()


def _mouse_drag(target, points: list[QPoint]) -> None:
    first = points[0]
    QApplication.sendEvent(
        target,
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(first),
            QPointF(first),
            QPointF(first),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    for point in points[1:]:
        QApplication.sendEvent(
            target,
            QMouseEvent(
                QEvent.Type.MouseMove,
                QPointF(point),
                QPointF(point),
                QPointF(point),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
    last = points[-1]
    QApplication.sendEvent(
        target,
        QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(last),
            QPointF(last),
            QPointF(last),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    QApplication.processEvents()


def _seek_tracking_widget(presentation: TimelinePresentation) -> tuple[TimelineWidget, list[Seek]]:
    intents: list[Seek] = []

    def _on_intent(intent):
        if isinstance(intent, Seek):
            intents.append(intent)
            return replace(presentation, playhead=intent.position)
        return presentation

    return TimelineWidget(presentation, on_intent=_on_intent), intents


__all__ = [name for name in globals() if not name.startswith("__")]
