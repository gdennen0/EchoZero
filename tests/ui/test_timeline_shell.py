from dataclasses import replace

from PyQt6.QtCore import QPoint, QPointF, QEvent
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMouseEvent
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
    LayerPresentation,
    ManualPullDiffPreviewPresentation,
    ManualPullEventOptionPresentation,
    ManualPullFlowPresentation,
    ManualPullTargetOptionPresentation,
    ManualPullTrackOptionPresentation,
    ManualPushDiffPreviewPresentation,
    ManualPushFlowPresentation,
    ManualPushTrackOptionPresentation,
    LayerStatusPresentation,
    TakeLanePresentation,
    TimelinePresentation,
    TransferPresetPresentation,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId, TimelineId
from echozero.application.sync.models import LiveSyncState
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
    DuplicateSelectedEvents,
    ExitPullFromMA3Workspace,
    ExitPushToMA3Mode,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    OpenPullFromMA3Dialog,
    OpenPushToMA3Dialog,
    Pause,
    Play,
    PreviewTransferPlan,
    SaveTransferPreset,
    Seek,
    SetPullImportMode,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    SetPushTransferMode,
    SelectAllEvents,
    SelectEvent,
    SelectLayer,
    SelectPullSourceEvents,
    SelectPullSourceTracks,
    SelectPullSourceTrack,
    SelectPullTargetLayer,
    SelectPushTargetTrack,
    Stop,
    ToggleLayerExpanded,
    ToggleMute,
    ToggleSolo,
    SetGain,
)
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.test_harness import build_variant_presentations, estimate_full_window_height
from echozero.ui.qt.timeline.blocks.ruler import timeline_x_for_time
from echozero.ui.qt.timeline.widget import (
    ManualPullTimelineDialog,
    ManualPullTimelineSelectionResult,
    TimelineWidget,
    compute_scroll_bounds,
    estimate_timeline_span_seconds,
)


def test_demo_variants_include_take_lanes_open_and_zoom_states():
    variants = build_variant_presentations()
    assert 'take_lanes_open' in variants
    assert 'zoomed_in' in variants
    assert 'zoomed_out' in variants


def test_play_pause_seek_intents_update_presentation():
    demo = build_demo_app()

    stopped = demo.dispatch(Pause())
    assert stopped.is_playing is False

    moved = demo.dispatch(Seek(4.25))
    assert moved.playhead == 4.25

    playing = demo.dispatch(Play())
    assert playing.is_playing is True

    stopped = demo.dispatch(Stop())
    assert stopped.is_playing is False
    assert stopped.playhead == 0.0


def test_realistic_fixture_contains_song_stems_and_drum_classifiers():
    demo = build_demo_app()
    presentation = demo.presentation()
    titles = {layer.title for layer in presentation.layers}

    assert {'Song', 'Drums', 'Bass', 'Vocals', 'Other', 'Kick', 'Snare', 'HiHat', 'Clap'} <= titles


def test_take_lanes_exist_without_inline_action_requirements():
    demo = build_demo_app()
    presentation = demo.presentation()
    drums = next(layer for layer in presentation.layers if layer.title == 'Drums')
    kick = next(layer for layer in presentation.layers if layer.title == 'Kick')

    assert len(drums.takes) >= 1
    assert drums.takes[0].kind.name == 'AUDIO'
    assert len(kick.takes) >= 1
    assert kick.takes[0].kind.name == 'EVENT'


def test_toggle_layer_expansion_round_trips():
    demo = build_demo_app()
    song = next(layer for layer in demo.presentation().layers if layer.title == 'Song')

    expanded = demo.dispatch(ToggleLayerExpanded(song.layer_id))
    expanded_song = next(layer for layer in expanded.layers if layer.title == 'Song')
    assert expanded_song.is_expanded is True

    collapsed = demo.dispatch(ToggleLayerExpanded(song.layer_id))
    collapsed_song = next(layer for layer in collapsed.layers if layer.title == 'Song')
    assert collapsed_song.is_expanded is False


def test_fixture_has_muted_and_soloed_layers_for_daw_state_rendering():
    demo = build_demo_app()
    presentation = demo.presentation()
    assert any(layer.muted for layer in presentation.layers)
    assert any(layer.soloed for layer in presentation.layers)


def test_timeline_span_estimate_uses_events_and_end_label():
    demo = build_demo_app()
    presentation = demo.presentation()

    span = estimate_timeline_span_seconds(presentation)

    assert span >= 8.0


def test_scroll_bounds_grow_with_zoom_level():
    demo = build_demo_app()
    base = demo.presentation()

    _, base_max = compute_scroll_bounds(base, viewport_width=900)
    zoomed_in = replace(base, pixels_per_second=320.0)
    _, zoomed_max = compute_scroll_bounds(zoomed_in, viewport_width=900)

    assert base_max > 0
    assert zoomed_max > base_max


def test_fixture_exposes_stale_manual_and_sync_signals():
    presentation = build_demo_app().presentation()

    assert any(layer.status.stale for layer in presentation.layers)
    assert any(layer.status.manually_modified for layer in presentation.layers)
    assert any("sync" in layer.title.lower() or layer.status.sync_label for layer in presentation.layers)


def test_fixture_keeps_unique_event_ids_across_main_and_takes():
    presentation = build_demo_app().presentation()

    ids: set[str] = set()
    for layer in presentation.layers:
        for event in layer.events:
            assert str(event.event_id) not in ids
            ids.add(str(event.event_id))
        for take in layer.takes:
            for event in take.events:
                assert str(event.event_id) not in ids
                ids.add(str(event.event_id))


def test_estimate_full_window_height_expanded_fixture_exceeds_default_capture_height():
    presentation = build_demo_app().presentation()
    assert estimate_full_window_height(presentation) > 720


def test_ruler_is_separate_widget_from_scroll_canvas():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(build_demo_app().presentation())
    try:
        assert widget._scroll.widget() is widget._canvas
        assert widget._ruler.parent() is not widget._scroll
        assert widget._ruler.parent() is not widget._canvas
    finally:
        widget.close()


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


def test_pipeline_context_actions_include_phase1_ids():
    presentation = _audio_pipeline_presentation()

    empty_contract = build_timeline_inspector_contract(presentation)
    song_contract = build_timeline_inspector_contract(
        presentation,
        hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=LayerId("layer_song")),
    )
    drums_contract = build_timeline_inspector_contract(
        presentation,
        hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=LayerId("layer_drums")),
    )

    empty_action_ids = {
        action.action_id
        for section in empty_contract.context_sections
        for action in section.actions
    }
    song_action_ids = {
        action.action_id
        for section in song_contract.context_sections
        for action in section.actions
    }
    drums_action_ids = {
        action.action_id
        for section in drums_contract.context_sections
        for action in section.actions
    }

    assert "add_song_from_path" in empty_action_ids
    assert "extract_stems" in song_action_ids
    assert "extract_drum_events" not in song_action_ids
    assert "classify_drum_events" not in song_action_ids
    assert "extract_stems" in drums_action_ids
    assert "extract_drum_events" in drums_action_ids
    assert "classify_drum_events" in drums_action_ids


def test_contract_add_song_action_calls_runtime(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []
            self._presentation = _audio_pipeline_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.calls.append((title, audio_path))
            self._presentation = replace(self._presentation, title=title)
            return self._presentation

    runtime = _Runtime()
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getText",
        lambda *args, **kwargs: ("Imported Song", True),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: ("C:/audio/import.wav", "Audio Files"),
    )
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(InspectorAction(action_id="add_song_from_path", label="Add Song From Path"))

        assert runtime.calls == [("Imported Song", "C:/audio/import.wav")]
        assert widget.presentation.title == "Imported Song"
    finally:
        widget.close()
        app.processEvents()


def test_contract_extract_pipeline_action_warns_when_not_implemented(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.runtime_audio = None
            self._presentation = _audio_pipeline_presentation()

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def extract_stems(self, layer_id):
            raise NotImplementedError(f"extract_stems pending for {layer_id}")

    warnings: list[str] = []
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.warning",
        lambda _parent, _title, message: warnings.append(message),
    )
    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._handle_runtime_pipeline_action(
            "extract_stems",
            {"layer_id": LayerId("layer_song")},
        )

        assert handled is True
        assert warnings == ["extract_stems pending for layer_song"]
    finally:
        widget.close()
        app.processEvents()


def test_contract_classify_pipeline_action_prompts_for_model_and_calls_runtime(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.runtime_audio = None
            self._presentation = _audio_pipeline_presentation()
            self.calls: list[tuple[object, str]] = []

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def classify_drum_events(self, layer_id, model_path):
            self.calls.append((layer_id, model_path))
            return self._presentation

    runtime = _Runtime()
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: ("C:/models/drums.pth", "PyTorch Models"),
    )
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._handle_runtime_pipeline_action(
            "classify_drum_events",
            {"layer_id": LayerId("layer_drums")},
        )

        assert handled is True
        assert runtime.calls == [(LayerId("layer_drums"), "C:/models/drums.pth")]
    finally:
        widget.close()
        app.processEvents()


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
                    [self._presentation.selected_layer_id] if self._presentation.selected_layer_id is not None else []
                )
                if intent.layer_id in current_ids:
                    selected_ids = [layer_id for layer_id in current_ids if layer_id != intent.layer_id]
                else:
                    selected_ids = [*current_ids, intent.layer_id]
            elif intent.mode == "range" and intent.layer_id is not None:
                ordered_ids = [layer.layer_id for layer in self._presentation.layers]
                anchor_id = self._presentation.selected_layer_id or intent.layer_id
                low, high = sorted((ordered_ids.index(anchor_id), ordered_ids.index(intent.layer_id)))
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
                            replace(take, events=[replace(event, is_selected=False) for event in take.events])
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
        if isinstance(intent, OpenPushToMA3Dialog):
            self._presentation = replace(
                self._presentation,
                layers=[
                    replace(
                        layer,
                        push_selection_count=len(self._presentation.selected_event_ids) if layer.layer_id == self._presentation.selected_layer_id else 0,
                        push_row_status="blocked" if layer.layer_id == self._presentation.selected_layer_id else "",
                        push_row_issue="Select an MA3 target track" if layer.layer_id == self._presentation.selected_layer_id else "",
                    )
                    for layer in self._presentation.layers
                ],
                manual_push_flow=ManualPushFlowPresentation(
                    dialog_open=False,
                    push_mode_active=True,
                    selected_layer_ids=list(self._presentation.selected_layer_ids) or [LayerId("layer_kick")],
                    available_tracks=[
                        ManualPushTrackOptionPresentation(
                            coord="tc1_tg2_tr3",
                            name="Track 3",
                            note="Bass",
                            event_count=8,
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
                selected_event_ids = list(self._presentation.batch_transfer_plan.rows[0].selected_event_ids)
            self._presentation = replace(
                self._presentation,
                layers=[
                    replace(
                        layer,
                        push_target_label="Track 3 (tc1_tg2_tr3) - Bass" if layer.layer_id == self._presentation.selected_layer_id else layer.push_target_label,
                        push_selection_count=len(selected_event_ids) if layer.layer_id == self._presentation.selected_layer_id else layer.push_selection_count,
                        push_row_status="ready" if layer.layer_id == self._presentation.selected_layer_id else layer.push_row_status,
                        push_row_issue="" if layer.layer_id == self._presentation.selected_layer_id else layer.push_row_issue,
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
                        push_target_label="Track 3 (tc1_tg2_tr3) - Bass" if layer.layer_id == self._presentation.selected_layer_id else layer.push_target_label,
                        push_selection_count=len(intent.selected_event_ids) if layer.layer_id == self._presentation.selected_layer_id else layer.push_selection_count,
                        push_row_status="ready" if layer.layer_id == self._presentation.selected_layer_id else layer.push_row_status,
                        push_row_issue="" if layer.layer_id == self._presentation.selected_layer_id else layer.push_row_issue,
                    )
                    for layer in self._presentation.layers
                ],
                manual_push_flow=ManualPushFlowPresentation(
                    dialog_open=False,
                    push_mode_active=True,
                    selected_layer_ids=list(self._presentation.manual_push_flow.selected_layer_ids),
                    available_tracks=list(self._presentation.manual_push_flow.available_tracks),
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
            rows = [] if plan is None else [
                replace(
                    row,
                    status="applied" if row.status == "ready" else row.status,
                    issue=None if row.status == "ready" else row.issue,
                )
                for row in plan.rows
            ]
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
                        self._presentation.manual_pull_flow.active_source_track_coord: list(intent.selected_ma3_event_ids),
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
                                if self._presentation.manual_pull_flow.target_layer_id == LayerId("layer_kick")
                                else "Unmapped"
                            ),
                            source_track_coord="tc1_tg2_tr3",
                            target_layer_id=self._presentation.manual_pull_flow.target_layer_id,
                            import_mode=intent.import_mode,
                            selected_ma3_event_ids=list(self._presentation.manual_pull_flow.selected_ma3_event_ids),
                            selected_count=len(self._presentation.manual_pull_flow.selected_ma3_event_ids),
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
                        pull_target_label="Kick" if layer.layer_id == intent.target_layer_id else layer.pull_target_label,
                        pull_selection_count=(
                            len(self._presentation.manual_pull_flow.selected_ma3_event_ids)
                            if layer.layer_id == intent.target_layer_id
                            else layer.pull_selection_count
                        ),
                        pull_row_status="ready" if layer.layer_id == intent.target_layer_id else layer.pull_row_status,
                        pull_row_issue="" if layer.layer_id == intent.target_layer_id else layer.pull_row_issue,
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
                            selected_ma3_event_ids=list(self._presentation.manual_pull_flow.selected_ma3_event_ids),
                            selected_count=len(self._presentation.manual_pull_flow.selected_ma3_event_ids),
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
                    selected_source_track_coords=list(self._presentation.manual_pull_flow.selected_source_track_coords),
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
                    available_target_layers=list(self._presentation.manual_pull_flow.available_target_layers),
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
            rows = [] if plan is None else [
                replace(
                    row,
                    status="applied" if row.status == "ready" else row.status,
                    issue=None if row.status == "ready" else row.issue,
                )
                for row in plan.rows
            ]
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


def _click_event_rect(widget: TimelineWidget, event_id: str, modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier) -> None:
    for rect, _, _, candidate_event_id in widget._canvas._event_rects:
        if str(candidate_event_id) == event_id:
            center = rect.center().toPoint()
            QTest.mouseClick(widget._canvas, Qt.MouseButton.LeftButton, modifiers, QPoint(center.x(), center.y()))
            QApplication.processEvents()
            return
    raise AssertionError(f"Missing event rect for {event_id}")


def _click_rect(widget: TimelineWidget, rect, modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier) -> None:
    center = rect.center().toPoint()
    QTest.mouseClick(widget._canvas, Qt.MouseButton.LeftButton, modifiers, QPoint(center.x(), center.y()))
    QApplication.processEvents()


def _click_transport_rect(widget: TimelineWidget, key: str) -> None:
    rect = widget._transport._control_rects[key]
    center = rect.center().toPoint()
    QTest.mouseClick(widget._transport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(center.x(), center.y()))
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

def test_main_row_event_click_dispatches_main_take_identity():
    app = QApplication.instance() or QApplication([])
    intents: list[SelectEvent] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)
        _click_event_rect(widget, "main_evt")

        assert len(intents) == 1
        assert intents[0] == SelectEvent(
            layer_id=LayerId("layer_kick"),
            take_id=TakeId("take_main"),
            event_id=EventId("main_evt"),
            mode="replace",
        )
    finally:
        widget.close()
        app.processEvents()


def test_object_info_panel_shows_empty_state_without_selection():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_selection_test_presentation())
    try:
        _render_for_hit_testing(widget)

        assert widget._object_info.text() == "No timeline object selected."
    finally:
        widget.close()
        app.processEvents()


def test_object_info_panel_updates_for_layer_selection():
    app = QApplication.instance() or QApplication([])
    harness = _SelectionInspectorHarness(_selection_test_presentation())
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    try:
        _render_for_hit_testing(widget)

        rect, _ = widget._canvas._header_select_rects[0]
        _click_rect(widget, rect)

        info = widget._object_info.text()
        assert "Layer Kick" in info
        assert "id: layer_kick" in info
        assert "kind: EVENT" in info
        assert "main take: take_main" in info
        assert "takes: 2" in info
        assert "status flags: none" in info
    finally:
        widget.close()
        app.processEvents()


def test_object_info_panel_updates_for_main_lane_event_selection():
    app = QApplication.instance() or QApplication([])
    harness = _SelectionInspectorHarness(_selection_test_presentation())
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    try:
        _render_for_hit_testing(widget)
        _click_event_rect(widget, "main_evt")

        info = widget._object_info.text()
        assert "Event Main" in info
        assert "id: main_evt" in info
        assert "start: 1.00s" in info
        assert "end: 1.50s" in info
        assert "duration: 0.50s" in info
        assert "layer: Kick" in info
        assert "take: Main take (take_main)" in info
    finally:
        widget.close()
        app.processEvents()


def test_take_lane_event_click_dispatches_take_identity():
    app = QApplication.instance() or QApplication([])
    intents: list[SelectEvent] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)
        _click_event_rect(widget, "take_evt")

        assert len(intents) == 1
        assert intents[0] == SelectEvent(
            layer_id=LayerId("layer_kick"),
            take_id=TakeId("take_alt"),
            event_id=EventId("take_evt"),
            mode="replace",
        )
    finally:
        widget.close()
        app.processEvents()


def test_object_info_panel_updates_for_take_lane_event_selection():
    app = QApplication.instance() or QApplication([])
    harness = _SelectionInspectorHarness(_selection_test_presentation())
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    try:
        _render_for_hit_testing(widget)
        _click_event_rect(widget, "take_evt")

        info = widget._object_info.text()
        assert "Event Take" in info
        assert "id: take_evt" in info
        assert "start: 2.00s" in info
        assert "end: 2.50s" in info
        assert "duration: 0.50s" in info
        assert "layer: Kick" in info
        assert "take: Take 2 (take_alt)" in info
    finally:
        widget.close()
        app.processEvents()


def test_object_info_panel_renders_selected_layer_contract_text():
    app = QApplication.instance() or QApplication([])
    presentation = replace(_selection_test_presentation(), selected_layer_id=LayerId("layer_kick"))
    widget = TimelineWidget(presentation)
    try:
        _render_for_hit_testing(widget)

        contract = build_timeline_inspector_contract(widget.presentation)

        assert widget._object_info.contract() == contract
        assert widget._object_info.text() == render_inspector_contract_text(contract)
    finally:
        widget.close()
        app.processEvents()


def test_object_info_panel_remains_contract_rendered_through_selection_transition_sequence():
    app = QApplication.instance() or QApplication([])
    harness = _SelectionInspectorHarness(_selection_test_presentation())
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    try:
        _render_for_hit_testing(widget)

        expected = build_timeline_inspector_contract(widget.presentation)
        assert widget._object_info.text() == render_inspector_contract_text(expected)

        header_rect, _ = widget._canvas._header_select_rects[0]
        _click_rect(widget, header_rect)
        expected = build_timeline_inspector_contract(widget.presentation)
        assert widget._object_info.text() == render_inspector_contract_text(expected)

        _click_event_rect(widget, "main_evt")
        expected = build_timeline_inspector_contract(widget.presentation)
        assert widget._object_info.text() == render_inspector_contract_text(expected)

        QTest.keyClick(widget._canvas, Qt.Key.Key_Escape)
        QApplication.processEvents()
        expected = build_timeline_inspector_contract(widget.presentation)
        assert widget._object_info.text() == render_inspector_contract_text(expected)
    finally:
        widget.close()
        app.processEvents()


def test_object_info_panel_keeps_no_takes_indication_for_empty_layer():
    app = QApplication.instance() or QApplication([])
    presentation = replace(_no_takes_layer_presentation(), selected_layer_id=LayerId("layer_empty"))
    widget = TimelineWidget(presentation)
    try:
        _render_for_hit_testing(widget)

        info = widget._object_info.text()
        assert "Layer Empty" in info
        assert "main take: none" in info
        assert "takes: none" in info
    finally:
        widget.close()
        app.processEvents()


def test_no_takes_layer_context_menu_excludes_take_actions():
    app = QApplication.instance() or QApplication([])
    presentation = _no_takes_layer_presentation()
    widget = TimelineWidget(presentation)
    try:
        _render_for_hit_testing(widget)

        contract = build_timeline_inspector_contract(
            widget.presentation,
            hit_target=TimelineInspectorHitTarget(
                kind="layer",
                layer_id=LayerId("layer_empty"),
                time_seconds=1.25,
            ),
        )
        menu = widget._canvas._build_context_menu(contract)
        action_ids = [action.action_id for section in contract.context_sections for action in section.actions]
        menu_labels = [action.text() for action in menu.actions() if not action.isSeparator()]

        assert "overwrite_main" not in action_ids
        assert "merge_main" not in action_ids
        assert "Overwrite Main" not in menu_labels
        assert "Merge Main" not in menu_labels
    finally:
        widget.close()
        app.processEvents()


def test_no_takes_layer_has_no_toggle_takes_intent_path():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _no_takes_layer_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        assert widget._canvas._toggle_rects == []
    finally:
        widget.close()
        app.processEvents()

    assert intents == []


def test_context_menu_uses_contract_actions_for_take_event_hit_target():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_selection_test_presentation())
    try:
        _render_for_hit_testing(widget)

        for rect, layer_id, take_id, event_id in widget._canvas._event_rects:
            if str(event_id) == "take_evt":
                hit_target = TimelineInspectorHitTarget(
                    kind="event",
                    layer_id=layer_id,
                    take_id=take_id,
                    event_id=event_id,
                    time_seconds=widget._canvas._seek_time_at_x(rect.center().x()),
                )
                break
        else:
            raise AssertionError("Missing event rect for take_evt")

        contract = build_timeline_inspector_contract(widget.presentation, hit_target=hit_target)
        menu = widget._canvas._build_context_menu(contract)
        menu_labels = [action.text() for action in menu.actions() if not action.isSeparator()]
        contract_labels = [action.label for section in contract.context_sections for action in section.actions]

        assert menu_labels == contract_labels
        assert "Overwrite Main" in menu_labels
        assert "Merge Main" in menu_labels
    finally:
        widget.close()
        app.processEvents()


def test_selection_contract_exposes_push_to_ma3_action():
    presentation = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt")],
        layers=[
            replace(
                _selection_test_presentation().layers[0],
                events=[
                    replace(
                        _selection_test_presentation().layers[0].events[0],
                        is_selected=True,
                    )
                ],
            )
        ],
    )

    contract = build_timeline_inspector_contract(presentation)
    action_ids = [action.action_id for section in contract.context_sections for action in section.actions]

    assert "push_to_ma3" in action_ids


def test_empty_contract_exposes_pull_from_ma3_action():
    contract = build_timeline_inspector_contract(_selection_test_presentation())
    action_ids = [action.action_id for section in contract.context_sections for action in section.actions]

    assert "pull_from_ma3" in action_ids


def test_layer_contract_exposes_sync_transfer_section_and_batch_placeholder_action():
    presentation = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _selection_test_presentation().layers[0],
                sync_target_label="tc1_tg2_tr3",
            )
        ],
        batch_transfer_plan=BatchTransferPlanPresentation(
            plan_id="plan_123",
            operation_type="push",
            rows=[
                BatchTransferPlanRowPresentation(
                    row_id="row_1",
                    direction="push",
                    source_label="Kick",
                    target_label="Track 3",
                    selected_count=1,
                    status="ready",
                )
            ],
            ready_count=1,
        ),
    )

    contract = build_timeline_inspector_contract(presentation)
    section_ids = [section.section_id for section in contract.context_sections]
    action_ids = [action.action_id for section in contract.context_sections for action in section.actions]

    assert "sync-transfer" in section_ids
    assert "open_batch_transfer_workspace" in action_ids
    assert "pull_from_ma3" in action_ids
    assert {"preview_transfer_plan", "apply_transfer_plan", "cancel_transfer_plan"} <= set(action_ids)


def test_live_sync_actions_hidden_when_experimental_flag_disabled():
    presentation = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _selection_test_presentation().layers[0],
                live_sync_state=LiveSyncState.OBSERVE,
                live_sync_pause_reason="operator pause",
                live_sync_divergent=True,
            )
        ],
    )

    contract = build_timeline_inspector_contract(presentation)
    section_ids = [section.section_id for section in contract.context_sections]
    action_ids = [action.action_id for section in contract.context_sections for action in section.actions]

    assert "live-sync" not in section_ids
    assert "live_sync_set_off" not in action_ids


def test_live_sync_action_dispatches_state_and_pause_intents(monkeypatch):
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = replace(
        _selection_test_presentation(),
        experimental_live_sync_enabled=True,
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _selection_test_presentation().layers[0],
                live_sync_state=LiveSyncState.PAUSED,
                live_sync_pause_reason="operator pause",
                live_sync_divergent=True,
            )
        ],
    )
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        contract = build_timeline_inspector_contract(widget.presentation)
        actions = {
            action.action_id: action
            for section in contract.context_sections
            for action in section.actions
        }

        widget._trigger_contract_action(actions["live_sync_set_off"])
        widget._trigger_contract_action(actions["live_sync_set_observe"])
        widget._trigger_contract_action(actions["live_sync_set_pause_reason"])
        widget._trigger_contract_action(actions["live_sync_clear_pause_reason"])

        assert intents == [
            SetLayerLiveSyncState(layer_id=LayerId("layer_kick"), live_sync_state=LiveSyncState.OFF),
            SetLayerLiveSyncState(layer_id=LayerId("layer_kick"), live_sync_state=LiveSyncState.OBSERVE),
            SetLayerLiveSyncPauseReason(layer_id=LayerId("layer_kick"), pause_reason="operator pause"),
            ClearLayerLiveSyncPauseReason(layer_id=LayerId("layer_kick")),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_live_sync_armed_write_requires_confirmation_before_dispatch(monkeypatch):
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = replace(
        _selection_test_presentation(),
        experimental_live_sync_enabled=True,
        selected_layer_id=LayerId("layer_kick"),
    )
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        contract = build_timeline_inspector_contract(widget.presentation)
        armed_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "live_sync_set_armed_write"
        )

        monkeypatch.setattr(
            "echozero.ui.qt.timeline.widget.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.StandardButton.No,
        )
        widget._trigger_contract_action(armed_action)
        assert intents == []

        monkeypatch.setattr(
            "echozero.ui.qt.timeline.widget.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
        )
        widget._trigger_contract_action(armed_action)
        assert intents == [
            SetLayerLiveSyncState(
                layer_id=LayerId("layer_kick"),
                live_sync_state=LiveSyncState.ARMED_WRITE,
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_pull_from_ma3_action_enters_pull_workspace_mode(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[replace(_selection_test_presentation().layers[0], is_selected=True)],
    )
    harness = _ManualPullHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    try:
        _render_for_hit_testing(widget)

        contract = build_timeline_inspector_contract(widget.presentation)
        pull_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "pull_from_ma3"
        )

        widget._trigger_contract_action(pull_action)

        assert harness.intents == [OpenPullFromMA3Dialog()]
        assert widget.presentation.manual_pull_flow.workspace_active is True
        assert widget.presentation.manual_pull_flow.available_tracks == [
            ManualPullTrackOptionPresentation(
                coord="tc1_tg2_tr3",
                name="Track 3",
                note="Lead",
                event_count=2,
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_select_pull_source_events_action_opens_timeline_popup_and_dispatches_event_and_target_selection(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[replace(_selection_test_presentation().layers[0], is_selected=True)],
    )
    harness = _ManualPullHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    popup_calls: list[str] = []
    picks = iter([("Track 3 (tc1_tg2_tr3) - Lead [2 events]", True)])
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: next(picks),
    )
    monkeypatch.setattr(
        TimelineWidget,
        "_open_manual_pull_timeline_popup",
        lambda self, flow: (
            popup_calls.append(flow.active_source_track_coord or "")
            or ManualPullTimelineSelectionResult(
                selected_event_ids=["ma3_evt_1", "ma3_evt_2"],
                target_layer_id=LayerId("layer_kick"),
                import_mode="main",
            )
        ),
    )
    try:
        _render_for_hit_testing(widget)

        pull_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "pull_from_ma3"
        )
        widget._trigger_contract_action(pull_action)
        widget._trigger_contract_action(
            next(
                action
                for section in build_timeline_inspector_contract(widget.presentation).context_sections
                for action in section.actions
                if action.action_id == "select_pull_source_tracks"
            )
        )
        widget._trigger_contract_action(
            next(
                action
                for section in build_timeline_inspector_contract(widget.presentation).context_sections
                for action in section.actions
                if action.action_id == "select_pull_source_events"
            )
        )

        assert popup_calls == ["tc1_tg2_tr3"]
        assert harness.intents == [
            OpenPullFromMA3Dialog(),
            SelectPullSourceTracks(source_track_coords=["tc1_tg2_tr3"]),
            SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"),
            SelectPullSourceEvents(selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2"]),
            SelectPullTargetLayer(target_layer_id=LayerId("layer_kick")),
            SetPullImportMode(import_mode="main"),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_pull_workspace_actions_dispatch_selection_mapping_preview_and_exit(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[replace(_selection_test_presentation().layers[0], is_selected=True)],
    )
    harness = _ManualPullHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    summaries: list[tuple[str, str, str]] = []
    picks = iter([("Track 3 (tc1_tg2_tr3) - Lead [2 events]", True)])
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: next(picks),
    )
    monkeypatch.setattr(
        TimelineWidget,
        "_open_manual_pull_timeline_popup",
        lambda self, flow: ManualPullTimelineSelectionResult(
            selected_event_ids=["ma3_evt_1", "ma3_evt_2"],
            target_layer_id=LayerId("layer_kick"),
            import_mode="main",
        ),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.information",
        lambda parent, title, text: summaries.append((parent.__class__.__name__, title, text)),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.No,
    )
    try:
        _render_for_hit_testing(widget)

        contract = build_timeline_inspector_contract(widget.presentation)
        pull_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "pull_from_ma3"
        )
        widget._trigger_contract_action(pull_action)

        contract = build_timeline_inspector_contract(widget.presentation)
        select_tracks_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "select_pull_source_tracks"
        )
        select_events_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "select_pull_source_events"
        )
        set_target_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "set_pull_target_layer_mapping"
        )

        widget._trigger_contract_action(select_tracks_action)
        widget._trigger_contract_action(select_events_action)

        contract = build_timeline_inspector_contract(widget.presentation)
        preview_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "preview_pull_diff"
        )
        exit_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "exit_pull_workspace"
        )

        widget._trigger_contract_action(preview_action)
        widget._trigger_contract_action(exit_action)

        assert harness.intents == [
            OpenPullFromMA3Dialog(),
            SelectPullSourceTracks(source_track_coords=["tc1_tg2_tr3"]),
            SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"),
            SelectPullSourceEvents(selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2"]),
            SelectPullTargetLayer(target_layer_id=LayerId("layer_kick")),
            SetPullImportMode(import_mode="main"),
            SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"),
            ConfirmPullFromMA3(
                source_track_coord="tc1_tg2_tr3",
                selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2"],
                target_layer_id=LayerId("layer_kick"),
                import_mode="main",
            ),
            ExitPullFromMA3Workspace(),
        ]
        assert widget.presentation.manual_pull_flow.workspace_active is False
        assert widget.presentation.batch_transfer_plan is None
        assert summaries == [
            (
                "TimelineWidget",
                "Pull Diff Preview",
                "Prepared diff preview for 2 selected events.\n\n"
                "Source track: Track 3 (tc1_tg2_tr3)\n"
                "Target layer: Kick\n"
                "No MA3 import has been started in this step.",
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_preview_pull_diff_does_not_auto_apply(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[replace(_selection_test_presentation().layers[0], is_selected=True)],
    )
    harness = _ManualPullHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    summaries: list[tuple[str, str, str]] = []
    picks = iter([("Track 3 (tc1_tg2_tr3) - Lead [2 events]", True)])
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: next(picks),
    )
    monkeypatch.setattr(
        TimelineWidget,
        "_open_manual_pull_timeline_popup",
        lambda self, flow: ManualPullTimelineSelectionResult(
            selected_event_ids=["ma3_evt_1", "ma3_evt_2"],
            target_layer_id=LayerId("layer_kick"),
            import_mode="new_take",
        ),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.information",
        lambda parent, title, text: summaries.append((parent.__class__.__name__, title, text)),
    )
    try:
        _render_for_hit_testing(widget)

        pull_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "pull_from_ma3"
        )
        widget._trigger_contract_action(pull_action)
        contract = build_timeline_inspector_contract(widget.presentation)
        widget._trigger_contract_action(
            next(
                action
                for section in contract.context_sections
                for action in section.actions
                if action.action_id == "select_pull_source_tracks"
            )
        )
        widget._trigger_contract_action(
            next(
                action
                for section in build_timeline_inspector_contract(widget.presentation).context_sections
                for action in section.actions
                if action.action_id == "select_pull_source_events"
            )
        )
        widget._trigger_contract_action(
            next(
                action
                for section in build_timeline_inspector_contract(widget.presentation).context_sections
                for action in section.actions
                if action.action_id == "preview_pull_diff"
            )
        )

        assert ApplyPullFromMA3() not in harness.intents
        assert widget.presentation.manual_pull_flow.diff_gate_open is True
        assert widget.presentation.manual_pull_flow.diff_preview == ManualPullDiffPreviewPresentation(
            selected_count=2,
            source_track_coord="tc1_tg2_tr3",
            source_track_name="Track 3",
            source_track_note="Lead",
            source_track_event_count=2,
            target_layer_id=LayerId("layer_kick"),
            target_layer_name="Kick",
            import_mode="new_take",
        )
        assert summaries == [
            (
                "TimelineWidget",
                "Pull Diff Preview",
                "Prepared diff preview for 2 selected events.\n\n"
                "Source track: Track 3 (tc1_tg2_tr3)\n"
                "Target layer: Kick\n"
                "No MA3 import has been started in this step.",
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_push_to_ma3_action_enters_push_mode_only(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt")],
        layers=[
            replace(
                _selection_test_presentation().layers[0],
                events=[
                    replace(
                        _selection_test_presentation().layers[0].events[0],
                        is_selected=True,
                    )
                ],
            )
        ],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    try:
        _render_for_hit_testing(widget)

        contract = build_timeline_inspector_contract(widget.presentation)
        push_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "push_to_ma3"
        )

        widget._trigger_contract_action(push_action)

        assert harness.intents == [
            OpenPushToMA3Dialog(selection_event_ids=[EventId("main_evt")]),
        ]
        assert widget.presentation.manual_push_flow.dialog_open is False
        assert widget.presentation.manual_push_flow.push_mode_active is True
        assert widget.presentation.manual_push_flow.diff_gate_open is False
        assert widget.presentation.batch_transfer_plan is not None
        assert widget.presentation.batch_transfer_plan.rows[0].status == "blocked"
    finally:
        widget.close()
        app.processEvents()


def test_push_mode_actions_dispatch_target_preview_and_exit(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_take_id=None,
        selected_event_ids=[],
        layers=[
            replace(
                _selection_test_presentation().layers[0],
                push_selection_count=1,
            )
        ],
        manual_push_flow=ManualPushFlowPresentation(
            dialog_open=False,
            push_mode_active=True,
            available_tracks=[
                ManualPushTrackOptionPresentation(
                    coord="tc1_tg2_tr3",
                    name="Track 3",
                    note="Bass",
                    event_count=8,
                )
            ],
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
                    selected_event_ids=[EventId("main_evt")],
                    selected_count=1,
                    status="blocked",
                    issue="Select an MA3 target track",
                )
            ],
            blocked_count=1,
        ),
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    summaries: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: ("Track 3 (tc1_tg2_tr3) - Bass [8 existing]", True),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.information",
        lambda parent, title, text: summaries.append((parent.__class__.__name__, title, text)),
    )
    try:
        _render_for_hit_testing(widget)
        contract = build_timeline_inspector_contract(widget.presentation)
        select_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "select_push_target_track"
        )
        exit_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "exit_push_mode"
        )

        widget._trigger_contract_action(select_action)
        contract = build_timeline_inspector_contract(widget.presentation)
        preview_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "preview_push_diff"
        )
        widget._trigger_contract_action(preview_action)
        widget._trigger_contract_action(exit_action)

        assert harness.intents == [
            SelectPushTargetTrack(target_track_coord="tc1_tg2_tr3", layer_id=LayerId("layer_kick")),
            ConfirmPushToMA3(
                target_track_coord="tc1_tg2_tr3",
                selected_event_ids=[EventId("main_evt")],
            ),
            ExitPushToMA3Mode(),
        ]
        assert summaries == [
            (
                "TimelineWidget",
                "Push Diff Preview",
                "Prepared diff preview for 1 selected event.\n\n"
                "Target track: Track 3 (tc1_tg2_tr3)\n"
                "No MA3 transfer has been started in this step.",
            )
        ]
        assert widget.presentation.manual_push_flow.push_mode_active is False
    finally:
        widget.close()
        app.processEvents()


def test_transfer_plan_actions_dispatch_preview_apply_and_cancel(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt")],
        layers=[
            replace(
                _selection_test_presentation().layers[0],
                events=[replace(_selection_test_presentation().layers[0].events[0], is_selected=True)],
                push_selection_count=1,
                push_row_status="ready",
            )
        ],
        manual_push_flow=ManualPushFlowPresentation(
            dialog_open=False,
            push_mode_active=True,
            available_tracks=[
                ManualPushTrackOptionPresentation(
                    coord="tc1_tg2_tr3",
                    name="Track 3",
                    note="Bass",
                    event_count=8,
                )
            ],
            target_track_coord="tc1_tg2_tr3",
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
                    target_track_coord="tc1_tg2_tr3",
                    selected_event_ids=[EventId("main_evt")],
                    selected_count=1,
                    status="ready",
                )
            ],
            ready_count=1,
        ),
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    summaries: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.information",
        lambda parent, title, text: summaries.append((parent.__class__.__name__, title, text)),
    )
    try:
        _render_for_hit_testing(widget)
        contract = build_timeline_inspector_contract(widget.presentation)
        preview_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "preview_transfer_plan"
        )
        apply_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "apply_transfer_plan"
        )
        cancel_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "cancel_transfer_plan"
        )

        widget._trigger_contract_action(preview_action)
        widget._trigger_contract_action(apply_action)
        widget._trigger_contract_action(cancel_action)

        assert harness.intents == [
            PreviewTransferPlan(plan_id="push:timeline_selection"),
            ApplyTransferPlan(plan_id="push:timeline_selection"),
            CancelTransferPlan(plan_id="push:timeline_selection"),
        ]
        assert summaries == [
            (
                "TimelineWidget",
                "Transfer Plan Preview",
                "Push plan preview complete.\n\nRows: 1\nReady: 1\nBlocked: 0\nApplied: 0\nFailed: 0",
            ),
            (
                "TimelineWidget",
                "Transfer Plan Results",
                "Push plan apply complete.\n\nRows: 1\nApplied: 1\nFailed: 0\nBlocked: 0",
            ),
        ]
        assert widget.presentation.batch_transfer_plan is None
    finally:
        widget.close()
        app.processEvents()


def test_transfer_preset_actions_are_hidden_from_primary_transfer_surface(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt")],
        layers=[
            replace(
                _selection_test_presentation().layers[0],
                sync_target_label="tc1_tg2_tr3",
            )
        ],
        manual_push_flow=ManualPushFlowPresentation(
            dialog_open=False,
            push_mode_active=True,
        ),
        transfer_presets=[
            TransferPresetPresentation(
                preset_id="preset-1",
                name="Drums",
                push_target_mapping_by_layer_id={LayerId("layer_kick"): "tc1_tg2_tr3"},
            )
        ],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    try:
        _render_for_hit_testing(widget)
        contract = build_timeline_inspector_contract(widget.presentation)
        action_ids = {action.action_id for section in contract.context_sections for action in section.actions}

        assert "save_transfer_preset" not in action_ids
        assert "apply_transfer_preset" not in action_ids
        assert "delete_transfer_preset" not in action_ids
        assert harness.intents == []
    finally:
        widget.close()
        app.processEvents()


def test_push_mode_contract_actions_dispatch_select_all_unselect_all_and_transfer_mode(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        manual_push_flow=ManualPushFlowPresentation(
            push_mode_active=True,
            transfer_mode="merge",
            available_tracks=[
                ManualPushTrackOptionPresentation(coord="tc1_tg2_tr3", name="Track 3", note="Bass"),
            ],
        ),
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: ("Overwrite", True),
    )
    try:
        _render_for_hit_testing(widget)
        widget._trigger_contract_action(InspectorAction(action_id="push_select_all_events", label="Select All Events"))
        widget._trigger_contract_action(InspectorAction(action_id="push_unselect_all_events", label="Unselect All Events"))
        widget._trigger_contract_action(InspectorAction(action_id="set_push_transfer_mode", label="Set Push Transfer Mode"))

        assert harness.intents == [
            SelectAllEvents(),
            ClearSelection(),
            SetPushTransferMode(mode="overwrite"),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_apply_transfer_plan_prompts_for_each_unmapped_push_row_before_dispatch(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_event_ids=[EventId("main_evt")],
        manual_push_flow=ManualPushFlowPresentation(
            push_mode_active=True,
            available_tracks=[
                ManualPushTrackOptionPresentation(coord="tc1_tg2_tr3", name="Track 3", note="Bass"),
                ManualPushTrackOptionPresentation(coord="tc1_tg2_tr4", name="Track 4", note="Lead"),
            ],
        ),
        batch_transfer_plan=BatchTransferPlanPresentation(
            plan_id="push:timeline_selection",
            operation_type="push",
            rows=[
                BatchTransferPlanRowPresentation(
                    row_id="push:layer_a",
                    direction="push",
                    source_label="A Layer",
                    source_layer_id=LayerId("layer_a"),
                    target_label="Unmapped",
                    selected_event_ids=[EventId("main_evt")],
                    selected_count=1,
                    status="blocked",
                    issue="Select an MA3 target track",
                ),
                BatchTransferPlanRowPresentation(
                    row_id="push:layer_b",
                    direction="push",
                    source_label="B Layer",
                    source_layer_id=LayerId("layer_b"),
                    target_label="Unmapped",
                    selected_event_ids=[EventId("main_evt")],
                    selected_count=1,
                    status="blocked",
                    issue="Select an MA3 target track",
                ),
            ],
            blocked_count=2,
        ),
    )
    intents: list[object] = []
    widget = TimelineWidget(base, on_intent=lambda intent: intents.append(intent) or base)
    prompts: list[str] = []
    replies = iter([("Track 3 (tc1_tg2_tr3) - Bass", True), ("Track 4 (tc1_tg2_tr4) - Lead", True)])
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda _parent, _title, label, *_args: prompts.append(label) or next(replies),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.information",
        lambda *args, **kwargs: None,
    )
    try:
        widget._trigger_contract_action(
            InspectorAction(
                action_id="apply_transfer_plan",
                label="Apply Transfer Plan (0 ready rows)",
                params={"plan_id": "push:timeline_selection"},
            )
        )

        assert prompts == ["Target MA3 track for A Layer", "Target MA3 track for B Layer"]
        assert intents == [
            SelectPushTargetTrack(target_track_coord="tc1_tg2_tr3", layer_id=LayerId("layer_a")),
            SelectPushTargetTrack(target_track_coord="tc1_tg2_tr4", layer_id=LayerId("layer_b")),
            ApplyTransferPlan(plan_id="push:timeline_selection"),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_apply_transfer_plan_aborts_when_push_mapping_prompt_is_canceled(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        manual_push_flow=ManualPushFlowPresentation(
            push_mode_active=True,
            available_tracks=[
                ManualPushTrackOptionPresentation(coord="tc1_tg2_tr3", name="Track 3", note="Bass"),
            ],
        ),
        batch_transfer_plan=BatchTransferPlanPresentation(
            plan_id="push:timeline_selection",
            operation_type="push",
            rows=[
                BatchTransferPlanRowPresentation(
                    row_id="push:layer_a",
                    direction="push",
                    source_label="A Layer",
                    source_layer_id=LayerId("layer_a"),
                    target_label="Unmapped",
                    selected_event_ids=[EventId("main_evt")],
                    selected_count=1,
                    status="blocked",
                    issue="Select an MA3 target track",
                ),
            ],
            blocked_count=1,
        ),
    )
    intents: list[object] = []
    widget = TimelineWidget(base, on_intent=lambda intent: intents.append(intent) or base)
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: ("", False),
    )
    try:
        widget._trigger_contract_action(
            InspectorAction(
                action_id="apply_transfer_plan",
                label="Apply Transfer Plan (0 ready rows)",
                params={"plan_id": "push:timeline_selection"},
            )
        )

        assert intents == []
    finally:
        widget.close()
        app.processEvents()


def test_manual_pull_timeline_dialog_keeps_target_selection_in_same_popup():
    app = QApplication.instance() or QApplication([])
    dialog = ManualPullTimelineDialog(
        source_track_label="Track 3 (tc1_tg2_tr3)",
        events=[
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
        selected_event_ids=["ma3_evt_2"],
        available_targets=[
            ManualPullTargetOptionPresentation(layer_id=LayerId("layer_kick"), name="Kick"),
            ManualPullTargetOptionPresentation(layer_id=LayerId("layer_new"), name="Create New Layer"),
        ],
        selected_target_layer_id=LayerId("layer_new"),
        selected_import_mode="main",
    )
    try:
        assert dialog.selected_event_ids() == ["ma3_evt_2"]
        assert dialog.selected_target_layer_id() == LayerId("layer_new")
        assert dialog.selected_import_mode() == "main"
        assert [dialog._target_combo.itemText(index) for index in range(dialog._target_combo.count())] == [
            "Kick",
            "Create New Layer",
        ]
        assert [dialog._import_mode_combo.itemText(index) for index in range(dialog._import_mode_combo.count())] == [
            "Import as New Take",
            "Import to Main",
        ]
        assert dialog._scroll_area.widget() is dialog._canvas
        assert dialog._zoom_value_label.text() == "100%"

        initial_pps = dialog._canvas.pixels_per_second
        initial_min_width = dialog._canvas.minimumWidth()
        dialog._zoom_in_btn.click()
        app.processEvents()
        assert dialog._canvas.pixels_per_second > initial_pps
        assert dialog._canvas.minimumWidth() >= initial_min_width

        dialog._canvas.set_selected_event_ids(["ma3_evt_1", "ma3_evt_2"])

        assert dialog.selected_event_ids() == ["ma3_evt_1", "ma3_evt_2"]
        assert dialog.selected_target_layer_id() == LayerId("layer_new")
        assert dialog._selection_label.text() == "Selected: 2 events"
    finally:
        dialog.close()
        app.processEvents()


def test_batch_plan_shortcuts_dispatch_preview_apply_and_cancel(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt")],
        manual_push_flow=ManualPushFlowPresentation(push_mode_active=True),
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
                    target_track_coord="tc1_tg2_tr3",
                    selected_event_ids=[EventId("main_evt")],
                    selected_count=1,
                    status="ready",
                )
            ],
            ready_count=1,
        ),
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.information",
        lambda *args, **kwargs: None,
    )
    try:
        _render_for_hit_testing(widget)

        QTest.keyClick(
            widget._canvas,
            Qt.Key.Key_P,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        QTest.keyClick(
            widget._canvas,
            Qt.Key.Key_Return,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        QTest.keyClick(
            widget._canvas,
            Qt.Key.Key_Backspace,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        QApplication.processEvents()

        assert harness.intents == [
            PreviewTransferPlan(plan_id="push:timeline_selection"),
            ApplyTransferPlan(plan_id="push:timeline_selection"),
            CancelTransferPlan(plan_id="push:timeline_selection"),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_batch_plan_shortcuts_noop_without_active_plan():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        QTest.keyClick(
            widget._canvas,
            Qt.Key.Key_P,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        QTest.keyClick(
            widget._canvas,
            Qt.Key.Key_Return,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        QTest.keyClick(
            widget._canvas,
            Qt.Key.Key_Backspace,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        QApplication.processEvents()

        assert intents == []
    finally:
        widget.close()
        app.processEvents()


def test_shift_click_event_dispatches_additive_selection_mode():
    app = QApplication.instance() or QApplication([])
    intents: list[SelectEvent] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)
        _click_event_rect(widget, "take_evt", Qt.KeyboardModifier.ShiftModifier)

        assert intents == [
            SelectEvent(
                layer_id=LayerId("layer_kick"),
                take_id=TakeId("take_alt"),
                event_id=EventId("take_evt"),
                mode="additive",
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_ctrl_click_event_dispatches_toggle_selection_mode():
    app = QApplication.instance() or QApplication([])
    intents: list[SelectEvent] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)
        _click_event_rect(widget, "main_evt", Qt.KeyboardModifier.ControlModifier)

        assert intents == [
            SelectEvent(
                layer_id=LayerId("layer_kick"),
                take_id=TakeId("take_main"),
                event_id=EventId("main_evt"),
                mode="toggle",
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_layer_header_click_dispatches_layer_selection_not_seek():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        rect, layer_id = widget._canvas._header_select_rects[0]
        assert layer_id == LayerId("layer_kick")

        _click_rect(widget, rect)

        assert intents == [SelectLayer(LayerId("layer_kick"))]
    finally:
        widget.close()
        app.processEvents()


def test_row_empty_space_click_dispatches_layer_selection_not_seek():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        rect, layer_id = widget._canvas._row_body_select_rects[0]
        assert layer_id == LayerId("layer_kick")

        _click_rect(widget, rect)

        assert intents == [SelectLayer(LayerId("layer_kick"))]
    finally:
        widget.close()
        app.processEvents()


def test_main_rows_expose_mute_solo_hit_targets_without_take_row_duplicates():
    app = QApplication.instance() or QApplication([])
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)
    try:
        _render_for_hit_testing(widget)

        assert len(widget._canvas._mute_rects) == len(presentation.layers)
        assert len(widget._canvas._solo_rects) == len(presentation.layers)
        assert len(widget._canvas._push_rects) == len(presentation.layers)
        assert len(widget._canvas._pull_rects) == len(presentation.layers)
    finally:
        widget.close()
        app.processEvents()


def test_ruler_click_dispatches_seek():
    app = QApplication.instance() or QApplication([])
    presentation = _selection_test_presentation()
    widget, intents = _seek_tracking_widget(presentation)
    try:
        _render_for_hit_testing(widget)

        QTest.mouseClick(widget._ruler, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(520, 12))
        QApplication.processEvents()

        assert intents == [Seek(2.0)]
    finally:
        widget.close()
        app.processEvents()


def test_ruler_click_dispatches_seek_using_scroll_offset():
    app = QApplication.instance() or QApplication([])
    presentation = replace(_selection_test_presentation(), scroll_x=200.0, end_time_label="00:12.00")
    widget, intents = _seek_tracking_widget(presentation)
    try:
        _render_for_hit_testing(widget)

        QTest.mouseClick(widget._ruler, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(520, 12))
        QApplication.processEvents()

        assert intents == [Seek(4.0)]
    finally:
        widget.close()
        app.processEvents()


def test_main_row_mute_and_solo_clicks_dispatch_toggle_intents():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        mute_rect, mute_layer_id = widget._canvas._mute_rects[0]
        solo_rect, solo_layer_id = widget._canvas._solo_rects[0]

        _click_rect(widget, mute_rect)
        _click_rect(widget, solo_rect)

        assert intents == [
            ToggleMute(mute_layer_id),
            ToggleSolo(solo_layer_id),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_layer_header_push_control_dispatches_timeline_push_intent():
    app = QApplication.instance() or QApplication([])
    harness = _ManualPushHarness(
        replace(
            _selection_test_presentation(),
            selected_layer_id=LayerId("layer_kick"),
            selected_layer_ids=[LayerId("layer_kick")],
            selected_take_id=TakeId("take_main"),
            selected_event_ids=[EventId("main_evt")],
        )
    )
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    try:
        _render_for_hit_testing(widget)

        push_rect, _ = widget._canvas._push_rects[0]
        _click_rect(widget, push_rect)

        assert harness.intents == [OpenPushToMA3Dialog(selection_event_ids=[EventId("main_evt")])]
        assert widget.presentation.manual_push_flow.push_mode_active is True
    finally:
        widget.close()
        app.processEvents()


def test_layer_header_pull_control_dispatches_pull_workspace_intent():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
    )
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        pull_rect, _ = widget._canvas._pull_rects[0]
        _click_rect(widget, pull_rect)

        assert intents == [OpenPullFromMA3Dialog()]
    finally:
        widget.close()
        app.processEvents()


def test_layer_header_click_dispatches_toggle_and_range_selection_modes():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _multi_layer_selection_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        first_rect, first_layer_id = widget._canvas._header_select_rects[0]
        second_rect, second_layer_id = widget._canvas._header_select_rects[1]
        third_rect, third_layer_id = widget._canvas._header_select_rects[2]

        _click_rect(widget, first_rect)
        _click_rect(widget, second_rect, Qt.KeyboardModifier.ControlModifier)
        _click_rect(widget, third_rect, Qt.KeyboardModifier.ShiftModifier)

        assert intents == [
            SelectLayer(first_layer_id, mode="replace"),
            SelectLayer(second_layer_id, mode="toggle"),
            SelectLayer(third_layer_id, mode="range"),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_ruler_drag_scrubs_playhead_continuously():
    app = QApplication.instance() or QApplication([])
    presentation = _selection_test_presentation()
    widget, intents = _seek_tracking_widget(presentation)
    try:
        _render_for_hit_testing(widget)

        _mouse_drag(
            widget._ruler,
            [QPoint(420, 12), QPoint(520, 12), QPoint(620, 12)],
        )

        assert intents == [Seek(1.0), Seek(2.0), Seek(3.0)]
    finally:
        widget.close()
        app.processEvents()


def test_playhead_head_drag_dispatches_seek():
    app = QApplication.instance() or QApplication([])
    presentation = replace(_selection_test_presentation(), playhead=1.0)
    widget, intents = _seek_tracking_widget(presentation)
    try:
        _render_for_hit_testing(widget)

        start_x = int(
            timeline_x_for_time(
                widget.presentation.playhead,
                scroll_x=widget.presentation.scroll_x,
                pixels_per_second=widget.presentation.pixels_per_second,
                content_start_x=widget._canvas._header_width,
            )
        )
        y = widget._canvas._top_padding - 4
        _mouse_drag(widget._canvas, [QPoint(start_x, y), QPoint(start_x + 100, y), QPoint(start_x + 200, y)])

        assert intents == [Seek(1.0), Seek(2.0), Seek(3.0)]
    finally:
        widget.close()
        app.processEvents()


def test_playhead_head_drag_dispatches_seek_using_scroll_offset():
    app = QApplication.instance() or QApplication([])
    presentation = replace(
        _selection_test_presentation(),
        playhead=4.0,
        scroll_x=200.0,
        end_time_label="00:12.00",
    )
    widget, intents = _seek_tracking_widget(presentation)
    try:
        _render_for_hit_testing(widget)

        start_x = int(
            timeline_x_for_time(
                widget.presentation.playhead,
                scroll_x=widget.presentation.scroll_x,
                pixels_per_second=widget.presentation.pixels_per_second,
                content_start_x=widget._canvas._header_width,
            )
        )
        y = widget._canvas._top_padding - 4
        _mouse_drag(widget._canvas, [QPoint(start_x, y), QPoint(start_x + 100, y), QPoint(start_x + 200, y)])

        assert intents == [Seek(4.0), Seek(5.0), Seek(6.0)]
    finally:
        widget.close()
        app.processEvents()


def test_canvas_empty_non_selection_space_no_longer_dispatches_seek():

    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        QTest.mouseClick(
            widget._canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(widget._canvas.width() - 10, widget._canvas.height() - 10),
        )
        QApplication.processEvents()

        assert intents == []
    finally:
        widget.close()
        app.processEvents()


def test_escape_dispatches_clear_selection():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        QTest.keyClick(widget._canvas, Qt.Key.Key_Escape)
        QApplication.processEvents()

        assert intents == [ClearSelection()]
    finally:
        widget.close()
        app.processEvents()


def test_ctrl_a_dispatches_select_all_events():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        QTest.keyClick(widget._canvas, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QApplication.processEvents()

        assert intents == [SelectAllEvents()]
    finally:
        widget.close()
        app.processEvents()


def test_arrow_keys_dispatch_nudge_selected_events():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        QTest.keyClick(widget._canvas, Qt.Key.Key_Left)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        QApplication.processEvents()

        assert intents == [
            NudgeSelectedEvents(direction=-1, steps=1),
            NudgeSelectedEvents(direction=1, steps=10),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_ctrl_d_dispatches_duplicate_selected_events():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        QTest.keyClick(widget._canvas, Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(
            widget._canvas,
            Qt.Key.Key_D,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        QApplication.processEvents()

        assert intents == [
            DuplicateSelectedEvents(steps=1),
            DuplicateSelectedEvents(steps=10),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_transport_bar_clicks_dispatch_play_pause_and_stop():
    app = QApplication.instance() or QApplication([])
    demo = build_demo_app()
    widget = TimelineWidget(demo.presentation(), on_intent=demo.dispatch)
    try:
        _render_for_hit_testing(widget)
        widget._transport.repaint()
        QApplication.processEvents()

        _click_transport_rect(widget, "play")
        assert widget.presentation.is_playing is False

        _click_transport_rect(widget, "play")
        assert widget.presentation.is_playing is True

        demo.dispatch(Seek(4.25))
        widget.set_presentation(demo.presentation())
        widget._transport.repaint()
        QApplication.processEvents()

        _click_transport_rect(widget, "stop")
        assert widget.presentation.is_playing is False
        assert widget.presentation.playhead == 0.0
    finally:
        widget.close()
        app.processEvents()


def test_dragging_selected_event_dispatches_move_intent():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _drag_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        for rect, _, _, candidate_event_id in widget._canvas._event_rects:
            if str(candidate_event_id) == "main_evt":
                start = rect.center().toPoint()
                break
        else:
            raise AssertionError("Missing event rect for main_evt")

        _mouse_drag(widget._canvas, [start, QPoint(start.x() + 100, start.y())])

        assert intents == [MoveSelectedEvents(delta_seconds=1.0, target_layer_id=None)]
    finally:
        widget.close()
        app.processEvents()


def test_dragging_selected_event_over_other_event_layer_dispatches_transfer_target():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _drag_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: intents.append(intent) or presentation)
    try:
        _render_for_hit_testing(widget)

        for rect, _, _, candidate_event_id in widget._canvas._event_rects:
            if str(candidate_event_id) == "main_evt":
                start = rect.center().toPoint()
                break
        else:
            raise AssertionError("Missing event rect for main_evt")

        target_rect = next(
            rect for rect, layer_id in widget._canvas._event_drop_rects if layer_id == LayerId("layer_snare")
        )
        target = QPoint(start.x(), int(target_rect.center().y()))

        _mouse_drag(widget._canvas, [start, target])

        assert intents == [MoveSelectedEvents(delta_seconds=0.0, target_layer_id=LayerId("layer_snare"))]
    finally:
        widget.close()
        app.processEvents()
    DeleteTransferPreset,
