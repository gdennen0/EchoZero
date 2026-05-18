"""Transfer and live-sync timeline-shell support cases.
Exists to isolate MA3 operator workflow coverage from layout and direct interaction tests.
Connects the workspace transfer helpers to the bounded transfer support slice.
"""

from tests.ui.timeline_shell_shared_support import *  # noqa: F401,F403
from echozero.application.settings import build_default_app_settings_service
from echozero.application.presentation.models import (
    ManualPushFlowPresentation,
    ManualPushSequenceOptionPresentation,
    ManualPushTimecodeOptionPresentation,
    ManualPushTrackOptionPresentation,
    ManualPushTrackGroupOptionPresentation,
    ManualPullTimecodeOptionPresentation,
    ManualPullTrackGroupOptionPresentation,
)
from echozero.ui.qt.timeline.manual_pull import ManualPullWorkspaceDialog
from echozero.ui.qt.timeline.ma3_cue_matcher import MA3CueOption
from echozero.ui.qt.timeline.manual_push_route import ManualPushRouteDialog
from echozero.ui.qt.timeline.widget_action_transfer_workspace_mixin import (
    TimelineWidgetTransferWorkspaceMixin,
)


def _ma3_push_selection_presentation() -> TimelinePresentation:
    return replace(
        _selection_test_presentation(),
        active_song_id="song_alpha",
        active_song_title="Alpha Song",
        active_song_version_id="song_version_festival",
        active_song_version_label="Festival Edit",
        active_song_version_ma3_timecode_pool_no=1,
    )


def test_selection_contract_exposes_push_first_ma3_actions():
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
    all_actions = [action for section in contract.context_sections for action in section.actions]
    action_ids = {action.action_id for action in all_actions}
    workspace_directions = {
        str(action.params.get("direction", "")).lower()
        for action in all_actions
        if action.action_id == "transfer.workspace_open"
    }

    assert {
        "transfer.route_layer_track",
        "transfer.workspace_open",
        "transfer.send_selection",
        "transfer.match_ma3_cues",
        "transfer.send_to_track_once",
    } <= action_ids
    assert {"pull", "push"} <= workspace_directions
    assert "transfer.plan_apply" not in action_ids


def test_empty_contract_omits_primary_ma3_actions():
    contract = build_timeline_inspector_contract(_selection_test_presentation())
    all_actions = [action for section in contract.context_sections for action in section.actions]
    action_ids = {action.action_id for action in all_actions}
    workspace_directions = {
        str(action.params.get("direction", "")).lower()
        for action in all_actions
        if action.action_id == "transfer.workspace_open"
    }

    assert "transfer.workspace_open" in action_ids
    assert "transfer.send_selection" not in action_ids
    assert workspace_directions == {"pull"}


def test_layer_contract_exposes_send_actions_without_batch_or_pull_actions():
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
    all_actions = [action for section in contract.context_sections for action in section.actions]
    action_ids = {action.action_id for action in all_actions}
    workspace_directions = {
        str(action.params.get("direction", "")).lower()
        for action in all_actions
        if action.action_id == "transfer.workspace_open"
    }

    assert "sync-transfer" in section_ids
    assert {
        "transfer.route_layer_track",
        "transfer.workspace_open",
        "transfer.send_selection",
        "transfer.match_ma3_cues",
        "transfer.send_to_track_once",
    } <= action_ids
    assert {"pull", "push"} <= workspace_directions
    assert "transfer.plan_preview" not in action_ids
    assert "transfer.plan_apply" not in action_ids


def test_transfer_workspace_open_action_prompts_once_and_imports_all_events(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = _selection_test_presentation()
    harness = _ManualPullHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)

    def _accept_workspace_dialog(_router, flow, *, exit_on_cancel):
        if not flow.workspace_active:
            _router._dispatch(OpenPullFromMA3Dialog())
        _router._dispatch(SelectPullSourceTracks(source_track_coords=["tc1_tg2_tr3"]))
        _router._dispatch(SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"))
        _router._dispatch(
            SelectPullSourceEvents(selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2"])
        )
        return True

    monkeypatch.setattr(
        TimelineWidgetTransferWorkspaceMixin,
        "_open_manual_pull_workspace_dialog",
        _accept_workspace_dialog,
    )
    try:
        _render_for_hit_testing(widget)
        pull_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.workspace_open"
            and str(action.params.get("direction", "")).lower() == "pull"
        )

        widget._trigger_contract_action(pull_action)

        assert harness.intents == [
            OpenPullFromMA3Dialog(),
            SelectPullSourceTracks(source_track_coords=["tc1_tg2_tr3"]),
            SelectPullSourceTrack(source_track_coord="tc1_tg2_tr3"),
            SelectPullSourceEvents(selected_ma3_event_ids=["ma3_evt_1", "ma3_evt_2"]),
            ApplyPullFromMA3(),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_transfer_workspace_open_pull_from_section_layer_focuses_layer_without_router_crash(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    base = _selection_test_presentation()
    section_layer = replace(
        base.layers[0],
        layer_id=LayerId("layer_sections"),
        title="Sections",
        kind=LayerKind.SECTION,
    )
    presentation = replace(
        base,
        layers=[section_layer],
        selected_layer_id=None,
        selected_layer_ids=[],
    )
    intents: list[object] = []

    def _dispatch(intent):
        nonlocal presentation
        intents.append(intent)
        if isinstance(intent, SelectLayer):
            presentation = replace(
                presentation,
                selected_layer_id=intent.layer_id,
                selected_layer_ids=[intent.layer_id],
            )
        elif isinstance(intent, OpenPullFromMA3Dialog):
            presentation = replace(
                presentation,
                manual_pull_flow=ManualPullFlowPresentation(workspace_active=False),
            )
        return presentation

    widget = TimelineWidget(presentation, on_intent=_dispatch)

    def _cancel_workspace_dialog(_router, flow, *, exit_on_cancel):
        if not flow.workspace_active:
            _router._dispatch(OpenPullFromMA3Dialog())
        return False

    monkeypatch.setattr(
        TimelineWidgetTransferWorkspaceMixin,
        "_open_manual_pull_workspace_dialog",
        _cancel_workspace_dialog,
    )
    try:
        _render_for_hit_testing(widget)
        pull_action = next(
            action
            for section in build_timeline_inspector_contract(
                widget.presentation,
                hit_target=TimelineInspectorHitTarget(
                    kind="layer",
                    layer_id=LayerId("layer_sections"),
                    time_seconds=1.0,
                ),
            ).context_sections
            for action in section.actions
            if action.action_id == "transfer.workspace_open"
            and str(action.params.get("direction", "")).strip().lower() == "pull"
        )

        widget._trigger_contract_action(pull_action)

        assert intents == [
            SelectLayer(LayerId("layer_sections")),
            OpenPullFromMA3Dialog(),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_live_sync_actions_hidden_when_experimental_flag_disabled():
    presentation = replace(
        _selection_test_presentation(),
        experimental_live_sync_enabled=False,
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
    action_ids = {
        action.action_id for section in contract.context_sections for action in section.actions
    }

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
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
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
            SetLayerLiveSyncState(
                layer_id=LayerId("layer_kick"), live_sync_state=LiveSyncState.OFF
            ),
            SetLayerLiveSyncState(
                layer_id=LayerId("layer_kick"), live_sync_state=LiveSyncState.OBSERVE
            ),
            SetLayerLiveSyncPauseReason(
                layer_id=LayerId("layer_kick"), pause_reason="operator pause"
            ),
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
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
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


def test_transfer_route_layer_track_refreshes_tracks_and_saves_route(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[replace(_ma3_push_selection_presentation().layers[0], is_selected=True)],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin."
        "TimelineWidgetMA3PushActionMixin._open_manual_push_route_popup",
        lambda *_args, **_kwargs: "tc1_tg2_tr3",
    )
    try:
        _render_for_hit_testing(widget)

        route_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.route_layer_track"
        )
        widget._trigger_contract_action(route_action)

        assert harness.intents == [
            RefreshMA3PushTracks(),
            SetLayerMA3Route(
                layer_id=LayerId("layer_kick"),
                target_track_coord="tc1_tg2_tr3",
            ),
        ]
        assert widget.presentation.layers[0].sync_target_label == "tc1_tg2_tr3"
    finally:
        widget.close()
        app.processEvents()


def test_transfer_route_layer_track_prepares_unassigned_target_with_existing_sequence(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[replace(_ma3_push_selection_presentation().layers[0], is_selected=True)],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    prompts: list[str] = []
    picks = iter(
        [
            ("Use existing sequence", True),
            ("15 - Lead Stack", True),
        ]
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin."
        "TimelineWidgetMA3PushActionMixin._open_manual_push_route_popup",
        lambda *_args, **_kwargs: "tc1_tg2_tr9",
    )

    def _get_item(*args, **kwargs):
        prompts.append(args[2])
        return next(picks)

    monkeypatch.setattr("echozero.ui.qt.timeline.widget.QInputDialog.getItem", _get_item)
    try:
        _render_for_hit_testing(widget)

        route_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.route_layer_track"
        )
        widget._trigger_contract_action(route_action)

        assert harness.intents == [
            RefreshMA3PushTracks(),
            RefreshMA3Sequences(),
            SetLayerMA3Route(
                layer_id=LayerId("layer_kick"),
                target_track_coord="tc1_tg2_tr9",
                sequence_action=AssignMA3TrackSequence(
                    target_track_coord="tc1_tg2_tr9",
                    sequence_no=15,
                ),
            ),
        ]
        assert any("Current song range: Song A (12-111)" in prompt for prompt in prompts)
        assert any("Assign an existing MA3 sequence" in prompt for prompt in prompts)
        assert widget.presentation.layers[0].sync_target_label == "tc1_tg2_tr9"
    finally:
        widget.close()
        app.processEvents()


def test_section_route_to_existing_sequence_opens_matcher_and_maps_missing_ma3_cues(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    base = _ma3_push_selection_presentation()
    section_layer = replace(
        base.layers[0],
        kind=LayerKind.SECTION,
        events=[
            replace(
                base.layers[0].events[0],
                event_id=EventId("section_intro"),
                start=1.0,
                label="Intro",
                cue_number=1,
                cue_ref="Cue 1",
            ),
            EventPresentation(
                event_id=EventId("section_verse"),
                start=12.0,
                end=12.5,
                label="Verse",
                cue_number=2,
                cue_ref="Cue 2",
            ),
        ],
        is_selected=True,
    )
    presentation = replace(
        base,
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        layers=[section_layer],
    )
    harness = _ManualPushHarness(presentation)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    captured: dict[str, object] = {}

    class _AcceptedCueMatcher:
        def __init__(self, *, rows, cue_options, **_kwargs):
            captured["rows"] = rows
            captured["cue_options"] = cue_options

        def exec(self):
            return True

        def selected_mappings(self):
            return [
                type(
                    "Mapping",
                    (),
                    {
                        "event_id": row.event_id,
                        "cue_number": option.cue_number,
                        "cue_ref": str(option.cue_number),
                    },
                )()
                for row, option in zip(captured["rows"], captured["cue_options"])
            ]

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin."
        "TimelineWidgetMA3PushActionMixin._open_manual_push_route_popup",
        lambda *_args, **_kwargs: "tc1_tg2_tr3",
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin."
        "TimelineWidgetMA3PushActionMixin._ma3_cue_options_for_track",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin.MA3CueMatcherDialog",
        _AcceptedCueMatcher,
    )
    try:
        _render_for_hit_testing(widget)

        route_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.route_layer_track"
        )
        widget._trigger_contract_action(route_action)

        assert [option.cue_number for option in captured["cue_options"]] == [1, 2]
        assert harness.intents == [
            RefreshMA3PushTracks(),
            SetLayerMA3Route(
                layer_id=LayerId("layer_kick"),
                target_track_coord="tc1_tg2_tr3",
            ),
            UpdateEventCueMappings(
                layer_id=LayerId("layer_kick"),
                take_id=TakeId("take_main"),
                edits=[
                    EventCueMappingEdit(
                        event_id=EventId("section_intro"),
                        cue_number=1,
                        cue_ref="1",
                    ),
                    EventCueMappingEdit(
                        event_id=EventId("section_verse"),
                        cue_number=2,
                        cue_ref="2",
                    ),
                ],
            ),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_manual_push_route_dialog_can_reassign_sequence_on_assigned_track():
    app = QApplication.instance() or QApplication([])
    dialog = ManualPushRouteDialog(
        title="Route Layer to MA3 Track",
        prompt="MA3 track",
    )
    try:
        dialog.configure_sheet(
            show_sequence_controls=True,
            show_apply_mode_controls=False,
        )
        dialog.set_flow(
            ManualPushFlowPresentation(
                available_timecodes=[ManualPushTimecodeOptionPresentation(number=1)],
                selected_timecode_no=1,
                available_track_groups=[
                    ManualPushTrackGroupOptionPresentation(number=2, name="Group")
                ],
                selected_track_group_no=2,
                available_tracks=[
                    ManualPushTrackOptionPresentation(
                        coord="tc1_tg2_tr3",
                        name="Track 3",
                        number=3,
                        sequence_no=7,
                    )
                ],
                available_sequences=[
                    ManualPushSequenceOptionPresentation(number=15, name="Lead Stack")
                ],
                target_track_coord="tc1_tg2_tr3",
            ),
            preferred_track_coord="tc1_tg2_tr3",
        )

        assert dialog.selected_sequence_mode() == (
            ManualPushRouteDialog.SEQUENCE_MODE_KEEP_ASSIGNED
        )
        assert dialog._sequence_mode_combo.isEnabled() is True
        dialog._sequence_mode_combo.setCurrentIndex(
            dialog._sequence_mode_combo.findData(
                ManualPushRouteDialog.SEQUENCE_MODE_ASSIGN_EXISTING
            )
        )

        assert dialog.selected_sequence_mode() == (
            ManualPushRouteDialog.SEQUENCE_MODE_ASSIGN_EXISTING
        )
        assert dialog.selected_sequence_no() == 15
    finally:
        dialog.close()
        app.processEvents()


def test_transfer_workspace_open_uses_saved_route_and_merge(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                is_selected=True,
                sync_target_label="tc1_tg2_tr3",
            )
        ],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: ("Merge", True),
    )
    monkeypatch.setattr(
        widget._action_router,
        "_confirm_send_to_saved_ma3_route",
        lambda **_kwargs: MA3PushApplyMode.MERGE,
    )
    try:
        _render_for_hit_testing(widget)

        send_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.workspace_open"
            and str(action.params.get("direction", "")).lower() == "push"
        )
        widget._trigger_contract_action(send_action)

        assert harness.intents == [
            RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3"),
            SetPushTransferMode(mode="merge"),
            PushLayerToMA3(
                layer_id=LayerId("layer_kick"),
                scope=MA3PushScope.LAYER_MAIN,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
                apply_mode=MA3PushApplyMode.MERGE,
            ),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_transfer_workspace_open_saved_route_can_reroute_before_send(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                is_selected=True,
                sync_target_label="tc1_tg2_tr9",
            )
        ],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    monkeypatch.setattr(
        widget._action_router,
        "_confirm_send_to_saved_ma3_route",
        lambda **_kwargs: "reroute",
    )
    monkeypatch.setattr(
        widget._action_router,
        "_open_manual_push_route_popup",
        lambda **_kwargs: "tc1_tg2_tr3",
    )
    try:
        _render_for_hit_testing(widget)

        send_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.workspace_open"
            and str(action.params.get("direction", "")).lower() == "push"
        )
        widget._trigger_contract_action(send_action)

        assert harness.intents == [
            RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr9"),
            SetLayerMA3Route(
                layer_id=LayerId("layer_kick"),
                target_track_coord="tc1_tg2_tr3",
            ),
            SetPushTransferMode(mode="merge"),
            PushLayerToMA3(
                layer_id=LayerId("layer_kick"),
                scope=MA3PushScope.LAYER_MAIN,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
                apply_mode=MA3PushApplyMode.MERGE,
            ),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_section_send_to_saved_route_matches_cues_before_merge_push(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = _ma3_push_selection_presentation()
    section_layer = replace(
        base.layers[0],
        kind=LayerKind.SECTION,
        is_selected=True,
        sync_target_label="tc1_tg2_tr3",
        events=[
            replace(
                base.layers[0].events[0],
                event_id=EventId("section_intro"),
                start=0.0,
                label="Part 1",
                cue_number=1,
                cue_ref="Cue 1",
            ),
            EventPresentation(
                event_id=EventId("section_verse"),
                start=8.0,
                end=8.5,
                label="Part 2",
                cue_number=2,
                cue_ref="Cue 2",
            ),
        ],
    )
    presentation = replace(
        base,
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        layers=[section_layer],
    )
    harness = _ManualPushHarness(presentation)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    match_calls: list[list[EventId]] = []
    monkeypatch.setattr(
        widget._action_router,
        "_confirm_send_to_saved_ma3_route",
        lambda **_kwargs: MA3PushApplyMode.MERGE,
    )
    monkeypatch.setattr(
        widget._action_router,
        "_ma3_cue_options_for_track",
        lambda *_args, **_kwargs: [],
    )

    def _match_cues(**kwargs):
        match_calls.append(list(kwargs["selected_event_ids"]))
        return True

    monkeypatch.setattr(widget._action_router, "_match_events_to_ma3_cues", _match_cues)
    try:
        _render_for_hit_testing(widget)

        send_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.workspace_open"
            and str(action.params.get("direction", "")).lower() == "push"
        )
        widget._trigger_contract_action(send_action)

        assert match_calls == [[EventId("section_intro"), EventId("section_verse")]]
        assert harness.intents == [
            RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3"),
            SetPushTransferMode(mode="merge"),
            PushLayerToMA3(
                layer_id=LayerId("layer_kick"),
                scope=MA3PushScope.LAYER_MAIN,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
                apply_mode=MA3PushApplyMode.MERGE,
            ),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_section_send_to_saved_route_matches_when_sequence_has_too_few_cues(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = _ma3_push_selection_presentation()
    section_layer = replace(
        base.layers[0],
        kind=LayerKind.SECTION,
        is_selected=True,
        sync_target_label="tc1_tg2_tr3",
        events=[
            replace(
                base.layers[0].events[0],
                event_id=EventId("section_intro"),
                start=0.0,
                label="Intro",
                cue_number=None,
                cue_ref=None,
            ),
            EventPresentation(
                event_id=EventId("section_verse"),
                start=8.0,
                end=8.5,
                label="Verse",
                cue_number=None,
                cue_ref=None,
            ),
        ],
    )
    presentation = replace(
        base,
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        layers=[section_layer],
    )
    harness = _ManualPushHarness(presentation)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    match_calls: list[list[EventId]] = []
    monkeypatch.setattr(
        widget._action_router,
        "_confirm_send_to_saved_ma3_route",
        lambda **_kwargs: MA3PushApplyMode.MERGE,
    )
    monkeypatch.setattr(
        widget._action_router,
        "_ma3_cue_options_for_track",
        lambda *_args, **_kwargs: [MA3CueOption(cue_number=1, name="Only Cue")],
    )

    def _match_cues(**kwargs):
        match_calls.append(list(kwargs["selected_event_ids"]))
        return True

    monkeypatch.setattr(widget._action_router, "_match_events_to_ma3_cues", _match_cues)
    try:
        _render_for_hit_testing(widget)

        send_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.workspace_open"
            and str(action.params.get("direction", "")).lower() == "push"
        )
        widget._trigger_contract_action(send_action)

        assert match_calls == [[EventId("section_intro"), EventId("section_verse")]]
    finally:
        widget.close()
        app.processEvents()


def test_section_overwrite_send_matches_cues_once_before_confirmation(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = _ma3_push_selection_presentation()
    section_layer = replace(
        base.layers[0],
        kind=LayerKind.SECTION,
        is_selected=True,
        sync_target_label="tc1_tg2_tr3",
        events=[
            replace(
                base.layers[0].events[0],
                event_id=EventId("section_intro"),
                start=0.0,
                label="Intro",
                cue_number=None,
                cue_ref=None,
            ),
            EventPresentation(
                event_id=EventId("section_verse"),
                start=8.0,
                end=8.5,
                label="Verse",
                cue_number=None,
                cue_ref=None,
            ),
        ],
    )
    presentation = replace(
        base,
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        layers=[section_layer],
    )
    harness = _ManualPushHarness(presentation)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    match_calls: list[list[EventId]] = []
    overwrite_prompts: list[str] = []
    monkeypatch.setattr(
        widget._action_router,
        "_confirm_send_to_saved_ma3_route",
        lambda **_kwargs: MA3PushApplyMode.OVERWRITE,
    )
    monkeypatch.setattr(
        widget._action_router,
        "_ma3_cue_options_for_track",
        lambda *_args, **_kwargs: [MA3CueOption(cue_number=1, name="Only Cue")],
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.question",
        lambda _parent, _title, text, *_args: overwrite_prompts.append(text)
        or QMessageBox.StandardButton.No,
    )

    def _match_cues(**kwargs):
        match_calls.append(list(kwargs["selected_event_ids"]))
        return True

    monkeypatch.setattr(widget._action_router, "_match_events_to_ma3_cues", _match_cues)
    try:
        _render_for_hit_testing(widget)

        send_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.workspace_open"
            and str(action.params.get("direction", "")).lower() == "push"
        )
        widget._trigger_contract_action(send_action)

        assert match_calls == [[EventId("section_intro"), EventId("section_verse")]]
        assert len(overwrite_prompts) == 1
        assert harness.intents == [
            RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3"),
            SetPushTransferMode(mode="overwrite"),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_transfer_workspace_open_context_menu_uses_single_confirmation_dialog(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                is_selected=True,
                sync_target_label="tc1_tg2_tr3",
            )
        ],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: ("Merge", True),
    )

    monkeypatch.setattr(
        widget._action_router,
        "_confirm_send_to_saved_ma3_route",
        lambda **_kwargs: MA3PushApplyMode.MERGE,
    )

    def _choose_send(menu, *_args, **_kwargs) -> object | None:
        for candidate in menu.actions():
            payload = candidate.data()
            if (
                isinstance(payload, InspectorAction)
                and payload.action_id == "transfer.workspace_open"
                and str(payload.params.get("direction", "")).lower() == "push"
            ):
                return candidate
        raise AssertionError("Context menu did not include transfer.workspace_open")

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMenu.exec",
        _choose_send,
    )

    try:
        _render_for_hit_testing(widget)

        header_rect, _ = widget._canvas._header_select_rects[0]
        QTest.mouseClick(
            widget._canvas,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            header_rect.center().toPoint(),
        )
        app.processEvents()

        assert harness.intents == [
            RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3"),
            SetPushTransferMode(mode="merge"),
            PushLayerToMA3(
                layer_id=LayerId("layer_kick"),
                scope=MA3PushScope.LAYER_MAIN,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
                apply_mode=MA3PushApplyMode.MERGE,
            ),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_transfer_workspace_open_context_menu_fails_gracefully_when_ma3_tracks_cannot_refresh(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                is_selected=True,
                sync_target_label="tc1_tg2_tr3",
            )
        ],
    )
    intents: list[object] = []

    def _dispatch(intent):
        intents.append(intent)
        if isinstance(intent, RefreshMA3PushTracks):
            raise OSError("can't assign requested address")
        return base

    widget = TimelineWidget(base, on_intent=_dispatch)
    warnings: list[str] = []
    hud_calls: list[int] = []
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: ("Merge", True),
    )

    monkeypatch.setattr(
        widget._action_router,
        "_confirm_send_to_saved_ma3_route",
        lambda **_kwargs: MA3PushApplyMode.MERGE,
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.warning",
        lambda _parent, _title, message: warnings.append(message),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin.TimelineWidgetMA3PushActionMixin._open_ma3_connection_hud",
        lambda _self: hud_calls.append(1) or False,
    )

    def _choose_send(menu, *_args, **_kwargs) -> object | None:
        for candidate in menu.actions():
            payload = candidate.data()
            if (
                isinstance(payload, InspectorAction)
                and payload.action_id == "transfer.workspace_open"
                and str(payload.params.get("direction", "")).lower() == "push"
            ):
                return candidate
        raise AssertionError("Context menu did not include transfer.workspace_open")

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMenu.exec",
        _choose_send,
    )

    try:
        _render_for_hit_testing(widget)

        header_rect, _ = widget._canvas._header_select_rects[0]
        QTest.mouseClick(
            widget._canvas,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            header_rect.center().toPoint(),
        )
        app.processEvents()

        assert len(warnings) == 1
        assert "Unable to refresh MA3 tracks" in warnings[0]
        assert "can't assign requested address" in warnings[0]
        assert intents == [RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3")]
        assert len(hud_calls) == 1
    finally:
        widget.close()
        app.processEvents()


def test_transfer_workspace_open_context_menu_retries_after_ma3_connection_overlay_applied(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                is_selected=True,
                sync_target_label="tc1_tg2_tr3",
            )
        ],
    )
    harness = _ManualPushHarness(base)
    intents: list[object] = []
    refresh_attempts = {"count": 0}

    def _dispatch(intent):
        if isinstance(intent, RefreshMA3PushTracks):
            refresh_attempts["count"] += 1
            if refresh_attempts["count"] == 1:
                raise OSError("can't assign requested address")
        return harness.dispatch(intent)

    widget = TimelineWidget(base, on_intent=_dispatch)
    warnings: list[str] = []
    hud_calls: list[int] = []
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: ("Merge", True),
    )

    monkeypatch.setattr(
        widget._action_router,
        "_confirm_send_to_saved_ma3_route",
        lambda **_kwargs: MA3PushApplyMode.MERGE,
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.warning",
        lambda _parent, _title, message: warnings.append(message),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin.TimelineWidgetMA3PushActionMixin._open_ma3_connection_hud",
        lambda _self: hud_calls.append(1) or True,
    )

    def _choose_send(menu, *_args, **_kwargs) -> object | None:
        for candidate in menu.actions():
            payload = candidate.data()
            if (
                isinstance(payload, InspectorAction)
                and payload.action_id == "transfer.workspace_open"
                and str(payload.params.get("direction", "")).lower() == "push"
            ):
                return candidate
        raise AssertionError("Context menu did not include transfer.workspace_open")

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMenu.exec",
        _choose_send,
    )

    try:
        _render_for_hit_testing(widget)

        header_rect, _ = widget._canvas._header_select_rects[0]
        QTest.mouseClick(
            widget._canvas,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            header_rect.center().toPoint(),
        )
        app.processEvents()

        assert refresh_attempts["count"] == 2
        assert harness.intents == [
            RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3"),
            SetPushTransferMode(mode="merge"),
            PushLayerToMA3(
                layer_id=LayerId("layer_kick"),
                scope=MA3PushScope.LAYER_MAIN,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
                apply_mode=MA3PushApplyMode.MERGE,
            ),
        ]
        assert len(hud_calls) == 1
        assert any("Unable to refresh MA3 tracks" in warning for warning in warnings)
    finally:
        widget.close()
        app.processEvents()


def test_transfer_workspace_open_context_menu_retries_when_service_comes_from_runtime_shell(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                is_selected=True,
                sync_target_label="tc1_tg2_tr3",
            )
        ],
    )

    class _Runtime:
        def __init__(self, presentation: TimelinePresentation):
            self.harness = _ManualPushHarness(presentation)
            self.app_settings_service = build_default_app_settings_service()
            self.dispatch_intents: list[object] = []
            self.refresh_count = 0
            self.apply_calls = 0
            self.bridge_reconfigured = False

        def presentation(self) -> TimelinePresentation:
            return self.harness.presentation()

        def apply_ma3_osc_runtime_config(self) -> bool:
            self.apply_calls += 1
            self.bridge_reconfigured = True
            return True

        def dispatch(self, intent) -> TimelinePresentation:
            self.dispatch_intents.append(intent)
            if isinstance(intent, RefreshMA3PushTracks):
                self.refresh_count += 1
                if self.refresh_count == 1 and not self.bridge_reconfigured:
                    raise OSError("can't assign requested address")
            return self.harness.dispatch(intent)

    runtime = _Runtime(base)
    widget = TimelineWidget(base, on_intent=runtime.dispatch)
    warnings: list[str] = []
    hud_services: list[object] = []

    class _FakeHud:
        def __init__(self, settings_service, parent=None):
            hud_services.append(settings_service)

        def exec(self):
            return True

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: ("Merge", True),
    )
    monkeypatch.setattr(
        widget._action_router,
        "_confirm_send_to_saved_ma3_route",
        lambda **_kwargs: MA3PushApplyMode.MERGE,
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.warning",
        lambda _parent, _title, message: warnings.append(message),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin.MA3ConnectionHUD",
        _FakeHud,
    )

    def _choose_send(menu, *_args, **_kwargs) -> object | None:
        for candidate in menu.actions():
            payload = candidate.data()
            if (
                isinstance(payload, InspectorAction)
                and payload.action_id == "transfer.workspace_open"
                and str(payload.params.get("direction", "")).lower() == "push"
            ):
                return candidate
        raise AssertionError("Context menu did not include transfer.workspace_open")

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMenu.exec",
        _choose_send,
    )

    try:
        _render_for_hit_testing(widget)

        header_rect, _ = widget._canvas._header_select_rects[0]
        QTest.mouseClick(
            widget._canvas,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            header_rect.center().toPoint(),
        )
        app.processEvents()

        assert runtime.refresh_count == 2
        assert runtime.dispatch_intents == [
            RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3"),
            RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3"),
            SetPushTransferMode(mode="merge"),
            PushLayerToMA3(
                layer_id=LayerId("layer_kick"),
                scope=MA3PushScope.LAYER_MAIN,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
                apply_mode=MA3PushApplyMode.MERGE,
            ),
        ]
        assert len(hud_services) == 1
        assert hud_services[0] is runtime.app_settings_service
        assert runtime.apply_calls == 1
        assert runtime.bridge_reconfigured is True
        assert any("Unable to refresh MA3 tracks" in warning for warning in warnings)
        assert all("unavailable in this shell" not in warning for warning in warnings)
    finally:
        widget.close()
        app.processEvents()


def test_transfer_workspace_open_context_menu_aborts_when_runtime_reconfigure_is_rejected(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                is_selected=True,
                sync_target_label="tc1_tg2_tr3",
            )
        ],
    )

    class _Runtime:
        def __init__(self, presentation: TimelinePresentation):
            self.harness = _ManualPushHarness(presentation)
            self.app_settings_service = build_default_app_settings_service()
            self.dispatch_intents: list[object] = []
            self.refresh_count = 0
            self.apply_calls = 0
            self.bridge_reconfigured = False

        def presentation(self) -> TimelinePresentation:
            return self.harness.presentation()

        def apply_ma3_osc_runtime_config(self) -> bool:
            self.apply_calls += 1
            return False

        def dispatch(self, intent):
            self.dispatch_intents.append(intent)
            if isinstance(intent, RefreshMA3PushTracks):
                self.refresh_count += 1
                if self.refresh_count == 1:
                    raise OSError("can't assign requested address")
            return self.harness.dispatch(intent)

    runtime = _Runtime(base)
    widget = TimelineWidget(base, on_intent=runtime.dispatch)
    warnings: list[str] = []
    hud_services: list[object] = []

    class _FakeHud:
        def __init__(self, settings_service, parent=None):
            hud_services.append(settings_service)

        def exec(self):
            return True

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: ("Merge", True),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.warning",
        lambda _parent, _title, message: warnings.append(message),
    )
    monkeypatch.setattr(
        widget._action_router,
        "_confirm_send_to_saved_ma3_route",
        lambda **_kwargs: MA3PushApplyMode.MERGE,
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin.MA3ConnectionHUD",
        _FakeHud,
    )

    def _choose_send(menu, *_args, **_kwargs) -> object | None:
        for candidate in menu.actions():
            payload = candidate.data()
            if (
                isinstance(payload, InspectorAction)
                and payload.action_id == "transfer.workspace_open"
                and str(payload.params.get("direction", "")).lower() == "push"
            ):
                return candidate
        raise AssertionError("Context menu did not include transfer.workspace_open")

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMenu.exec",
        _choose_send,
    )

    try:
        _render_for_hit_testing(widget)

        header_rect, _ = widget._canvas._header_select_rects[0]
        QTest.mouseClick(
            widget._canvas,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            header_rect.center().toPoint(),
        )
        app.processEvents()

        assert runtime.refresh_count == 1
        assert runtime.apply_calls == 1
        assert runtime.dispatch_intents == [RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3")]
        assert len(hud_services) == 1
        assert hud_services[0] is runtime.app_settings_service
        assert any("Unable to refresh MA3 tracks" in warning for warning in warnings)
        assert any(
            "Unable to reconfigure the live MA3 connection for this session." in warning
            for warning in warnings
        )
        assert all("already routed to" not in warning for warning in warnings)
        assert "PushLayerToMA3" not in {
            intent.__class__.__name__ for intent in runtime.dispatch_intents
        }
    finally:
        widget.close()
        app.processEvents()


def test_transfer_workspace_open_context_menu_uses_widget_app_settings_service_when_runtime_shell_missing(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                is_selected=True,
                sync_target_label="tc1_tg2_tr3",
            )
        ],
    )
    app_settings_service = build_default_app_settings_service()
    refresh_count = {"count": 0}
    dispatch_intents: list[object] = []

    def _dispatch(intent):
        dispatch_intents.append(intent)
        if isinstance(intent, RefreshMA3PushTracks):
            refresh_count["count"] += 1
            if refresh_count["count"] == 1:
                raise OSError("can't assign requested address")
        return base

    widget = TimelineWidget(
        base,
        on_intent=_dispatch,
        app_settings_service=app_settings_service,
    )
    widget._action_router._resolve_runtime_shell = lambda: None
    warnings: list[str] = []
    hud_services: list[object] = []

    class _FakeHud:
        def __init__(self, settings_service, parent=None):
            hud_services.append(settings_service)

        def exec(self):
            return True

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: ("Merge", True),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.warning",
        lambda _parent, _title, message: warnings.append(message),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin.MA3ConnectionHUD",
        _FakeHud,
    )

    def _choose_send(menu, *_args, **_kwargs) -> object | None:
        for candidate in menu.actions():
            payload = candidate.data()
            if (
                isinstance(payload, InspectorAction)
                and payload.action_id == "transfer.workspace_open"
                and str(payload.params.get("direction", "")).lower() == "push"
            ):
                return candidate
        raise AssertionError("Context menu did not include transfer.workspace_open")

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMenu.exec",
        _choose_send,
    )

    try:
        _render_for_hit_testing(widget)

        header_rect, _ = widget._canvas._header_select_rects[0]
        QTest.mouseClick(
            widget._canvas,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            header_rect.center().toPoint(),
        )
        app.processEvents()

        assert refresh_count["count"] == 1
        assert hud_services == [app_settings_service]
        assert dispatch_intents == [RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3")]
        assert all("unavailable in this shell" not in warning for warning in warnings)
    finally:
        widget.close()
        app.processEvents()


def test_transfer_workspace_open_context_menu_shows_overlay_unavailable_warning_when_no_app_settings_service(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                is_selected=True,
                sync_target_label="tc1_tg2_tr3",
            )
        ],
    )

    def _dispatch(intent):
        if isinstance(intent, RefreshMA3PushTracks):
            raise OSError("can't assign requested address")
        return base

    widget = TimelineWidget(base, on_intent=_dispatch)
    widget._action_router._resolve_runtime_shell = lambda: None
    warnings: list[str] = []

    def _deny_hud(*_args, **_kwargs):
        raise AssertionError("HUD should not be created when AppSettingsService is unavailable")

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: ("Merge", True),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.warning",
        lambda _parent, _title, message: warnings.append(message),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin.MA3ConnectionHUD",
        _deny_hud,
    )

    def _choose_send(menu, *_args, **_kwargs) -> object | None:
        for candidate in menu.actions():
            payload = candidate.data()
            if (
                isinstance(payload, InspectorAction)
                and payload.action_id == "transfer.workspace_open"
                and str(payload.params.get("direction", "")).lower() == "push"
            ):
                return candidate
        raise AssertionError("Context menu did not include transfer.workspace_open")

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMenu.exec",
        _choose_send,
    )

    try:
        _render_for_hit_testing(widget)

        header_rect, _ = widget._canvas._header_select_rects[0]
        QTest.mouseClick(
            widget._canvas,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            header_rect.center().toPoint(),
        )
        app.processEvents()

        assert any(
            "The MA3 OSC connection overlay is unavailable in this shell." in warning
            for warning in warnings
        )
    finally:
        widget.close()
        app.processEvents()


def test_transfer_workspace_open_prepares_saved_route_in_current_song_range(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                is_selected=True,
                sync_target_label="tc1_tg2_tr9",
            )
        ],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    picks = iter(
        [
            ("Create sequence in current song range", True),
            ("Merge", True),
        ]
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: next(picks),
    )
    monkeypatch.setattr(
        widget._action_router,
        "_confirm_send_to_saved_ma3_route",
        lambda **_kwargs: MA3PushApplyMode.MERGE,
    )
    try:
        _render_for_hit_testing(widget)

        send_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.workspace_open"
            and str(action.params.get("direction", "")).lower() == "push"
        )
        widget._trigger_contract_action(send_action)

        assert harness.intents == [
            RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr9"),
            RefreshMA3Sequences(),
            SetPushTransferMode(mode="merge"),
            PushLayerToMA3(
                layer_id=LayerId("layer_kick"),
                scope=MA3PushScope.LAYER_MAIN,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
                apply_mode=MA3PushApplyMode.MERGE,
                sequence_action=CreateMA3Sequence(
                    creation_mode=MA3SequenceCreationMode.CURRENT_SONG_RANGE,
                    preferred_name="Alpha Song - Kick",
                ),
            ),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_transfer_send_selection_uses_selected_main_events(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt")],
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                sync_target_label="tc1_tg2_tr3",
                is_selected=True,
                events=[
                    replace(
                        _ma3_push_selection_presentation().layers[0].events[0],
                        is_selected=True,
                    )
                ],
            )
        ],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin."
        "TimelineWidgetMA3PushActionMixin._open_manual_push_route_popup",
        lambda *_args, **_kwargs: "tc1_tg2_tr3",
    )

    def _unexpected_question(*_args, **_kwargs):
        raise AssertionError("Unexpected MA3 confirmation prompt")

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.question",
        _unexpected_question,
    )
    try:
        _render_for_hit_testing(widget)

        send_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.send_selection"
        )
        widget._trigger_contract_action(send_action)

        assert harness.intents == [
            RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3"),
            SetLayerMA3Route(
                layer_id=LayerId("layer_kick"),
                target_track_coord="tc1_tg2_tr3",
            ),
            SetPushTransferMode(mode="merge"),
            PushLayerToMA3(
                layer_id=LayerId("layer_kick"),
                scope=MA3PushScope.SELECTED_EVENTS,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
                apply_mode=MA3PushApplyMode.MERGE,
                selected_event_ids=[EventId("main_evt")],
            ),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_manual_push_track_label_includes_timecode_name_when_available():
    track = ManualPushTrackOptionPresentation(
        coord="tc7_tg2_tr4",
        name="Track 4",
        timecode_name="Verse",
        note="Lead",
        event_count=1,
    )

    assert (
        TimelineWidgetTransferWorkspaceMixin._manual_push_track_label(track)
        == "TC7 Verse · Track 4 (tc7_tg2_tr4) - Lead [1 existing]"
    )


def test_manual_push_route_dialog_emits_dependent_timecode_and_group_selections():
    app = QApplication.instance() or QApplication([])
    dialog = ManualPushRouteDialog(title="Route Layer to MA3 Track", prompt="MA3 track")
    picked_timecodes: list[int] = []
    picked_groups: list[int] = []
    refreshes: list[bool] = []
    created_timecodes: list[bool] = []
    created_groups: list[bool] = []
    created_tracks: list[bool] = []
    dialog.timecode_selected.connect(lambda no: picked_timecodes.append(int(no)))
    dialog.track_group_selected.connect(lambda no: picked_groups.append(int(no)))
    dialog.refresh_requested.connect(lambda: refreshes.append(True))
    dialog.create_timecode_requested.connect(lambda: created_timecodes.append(True))
    dialog.create_track_group_requested.connect(lambda: created_groups.append(True))
    dialog.create_track_requested.connect(lambda: created_tracks.append(True))
    dialog.set_flow(
        ManualPushFlowPresentation(
            available_timecodes=[
                ManualPushTimecodeOptionPresentation(number=1, name=None),
                ManualPushTimecodeOptionPresentation(number=2, name="Verse"),
            ],
            selected_timecode_no=2,
            available_track_groups=[
                ManualPushTrackGroupOptionPresentation(number=1, name="Drums", track_count=1),
                ManualPushTrackGroupOptionPresentation(number=4, name="FX", track_count=2),
            ],
            selected_track_group_no=4,
            available_tracks=[
                ManualPushTrackOptionPresentation(
                    coord="tc2_tg4_tr7",
                    name="Lasers",
                    number=7,
                    event_count=5,
                ),
                ManualPushTrackOptionPresentation(
                    coord="tc2_tg4_tr8",
                    name="Fire",
                    number=8,
                    event_count=1,
                ),
            ],
        ),
        preferred_track_coord="tc2_tg4_tr8",
    )
    try:
        assert dialog.selected_timecode_no() == 2
        assert dialog.selected_track_group_no() == 4
        assert dialog.selected_track_coord() == "tc2_tg4_tr8"
        assert "TR8 Fire" in dialog._summary.text()
        assert dialog._timecode_combo.itemText(0) == "+ Create New Timecode..."
        assert dialog._track_group_combo.itemText(0) == "+ Create New Track Group..."
        assert dialog._track_combo.itemText(0) == "+ Create New Track..."

        dialog._refresh_button.click()
        dialog._timecode_combo.setCurrentIndex(1)
        dialog._track_group_combo.setCurrentIndex(1)
        dialog._track_combo.setCurrentIndex(0)
        dialog._track_group_combo.setCurrentIndex(0)
        dialog._timecode_combo.setCurrentIndex(0)
        app.processEvents()

        assert picked_timecodes == [1]
        assert picked_groups == [1]
        assert refreshes == [True]
        assert created_tracks == [True]
        assert created_groups == [True]
        assert created_timecodes == [True]
    finally:
        dialog.close()
        app.processEvents()


def test_manual_push_route_popup_refresh_button_dispatches_track_refresh(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                is_selected=True,
                sync_target_label="tc1_tg2_tr3",
            )
        ],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)

    def _emit_refresh_then_cancel(dialog) -> bool:
        dialog.refresh_requested.emit()
        return False

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin.ManualPushRouteDialog.exec",
        _emit_refresh_then_cancel,
    )
    try:
        result = widget._action_router._open_manual_push_route_popup(
            title="Route Layer to MA3 Track",
            prompt="MA3 track",
            reference_track_coord="tc1_tg2_tr3",
        )

        assert result is None
        assert harness.intents == [
            RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3"),
            RefreshMA3PushTracks(
                target_track_coord="tc1_tg2_tr3",
                timecode_no=1,
                track_group_no=2,
            ),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_manual_push_route_dialog_disables_write_mode_for_new_or_empty_track():
    app = QApplication.instance() or QApplication([])
    dialog = ManualPushRouteDialog(title="Send to MA3", prompt="MA3 track")
    dialog.configure_sheet(
        show_sequence_controls=False,
        show_apply_mode_controls=True,
        default_apply_mode="overwrite",
    )
    dialog.set_flow(
        ManualPushFlowPresentation(
            available_timecodes=[ManualPushTimecodeOptionPresentation(number=1, name=None)],
            selected_timecode_no=1,
            available_track_groups=[
                ManualPushTrackGroupOptionPresentation(number=2, name="Group 2", track_count=2)
            ],
            selected_track_group_no=2,
            available_tracks=[
                ManualPushTrackOptionPresentation(
                    coord="tc1_tg2_tr3",
                    name="Track 3",
                    number=3,
                    event_count=6,
                ),
                ManualPushTrackOptionPresentation(
                    coord="tc1_tg2_tr9",
                    name="Track 9",
                    number=9,
                    event_count=0,
                ),
            ],
            target_track_coord="tc1_tg2_tr9",
        ),
        preferred_track_coord="tc1_tg2_tr9",
    )
    try:
        assert dialog.selected_track_coord() == "tc1_tg2_tr9"
        assert dialog._apply_mode_combo.isEnabled() is False
        assert dialog.selected_apply_mode() == "merge"
        assert "automatic" in dialog._apply_mode_hint.text().lower()

        dialog._track_combo.setCurrentIndex(1)
        app.processEvents()

        assert dialog.selected_track_coord() == "tc1_tg2_tr3"
        assert dialog._apply_mode_combo.isEnabled() is True

        dialog._apply_mode_combo.setCurrentIndex(1)
        app.processEvents()

        assert dialog.selected_apply_mode() == "overwrite"
    finally:
        dialog.close()
        app.processEvents()


def test_manual_push_route_sequence_creation_prefers_selected_track_name_for_hitmaker():
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                is_selected=True,
            )
        ],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    dialog = ManualPushRouteDialog(title="Route Layer to MA3 Track", prompt="MA3 track")
    dialog.configure_sheet(
        show_sequence_controls=True,
        show_apply_mode_controls=False,
    )
    flow = ManualPushFlowPresentation(
        available_timecodes=[ManualPushTimecodeOptionPresentation(number=1, name="Song A")],
        selected_timecode_no=1,
        available_track_groups=[
            ManualPushTrackGroupOptionPresentation(number=2, name="Group 2", track_count=1)
        ],
        selected_track_group_no=2,
        available_tracks=[
            ManualPushTrackOptionPresentation(
                coord="tc1_tg2_tr9",
                name="Laser Hits",
                number=9,
                sequence_no=None,
                event_count=0,
            )
        ],
        available_sequences=[],
        target_track_coord="tc1_tg2_tr9",
    )
    dialog.set_flow(flow, preferred_track_coord="tc1_tg2_tr9")

    try:
        target_track = flow.available_tracks[0]
        layer = widget.presentation.layers[0]
        sequence_action = widget._action_router._sequence_action_from_route_popup(
            dialog=dialog,
            target_track=target_track,
            flow=flow,
            layer=layer,
            title="Route Layer to MA3 Track",
        )

        assert isinstance(sequence_action, CreateMA3Sequence)
        assert sequence_action.preferred_name == "Laser Hits"
    finally:
        dialog.close()
        widget.close()
        app.processEvents()


def test_transfer_send_to_track_once_uses_selected_scope_without_saving(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt")],
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                sync_target_label="tc1_tg2_tr3",
                is_selected=True,
                events=[
                    replace(
                        _ma3_push_selection_presentation().layers[0].events[0],
                        is_selected=True,
                    )
                ],
            )
        ],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    picks = iter(
        [
            ("Create next available sequence", True),
            ("Merge", True),
        ]
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin."
        "TimelineWidgetMA3PushActionMixin._open_manual_push_route_popup",
        lambda *_args, **_kwargs: "tc1_tg2_tr9",
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: next(picks),
    )
    try:
        _render_for_hit_testing(widget)

        send_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.send_to_track_once"
        )
        widget._trigger_contract_action(send_action)

        assert harness.intents == [
            RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3"),
            RefreshMA3Sequences(),
            SetPushTransferMode(mode="merge"),
            PushLayerToMA3(
                layer_id=LayerId("layer_kick"),
                scope=MA3PushScope.SELECTED_EVENTS,
                target_mode=MA3PushTargetMode.DIFFERENT_TRACK_ONCE,
                apply_mode=MA3PushApplyMode.MERGE,
                target_track_coord="tc1_tg2_tr9",
                selected_event_ids=[EventId("main_evt")],
                sequence_action=CreateMA3Sequence(
                    creation_mode=MA3SequenceCreationMode.NEXT_AVAILABLE,
                    preferred_name="Alpha Song - Kick",
                ),
            ),
        ]
        assert widget.presentation.layers[0].sync_target_label == "tc1_tg2_tr3"
    finally:
        widget.close()
        app.processEvents()


def test_transfer_workspace_open_without_saved_route_routes_then_sends(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[replace(_ma3_push_selection_presentation().layers[0], is_selected=True)],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    picks = iter(
        [
            ("Merge", True),
        ]
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_ma3_push_mixin."
        "TimelineWidgetMA3PushActionMixin._open_manual_push_route_popup",
        lambda *_args, **_kwargs: "tc1_tg2_tr3",
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: next(picks),
    )
    try:
        _render_for_hit_testing(widget)

        send_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.workspace_open"
            and str(action.params.get("direction", "")).lower() == "push"
        )
        widget._trigger_contract_action(send_action)

        assert harness.intents == [
            RefreshMA3PushTracks(),
            SetLayerMA3Route(
                layer_id=LayerId("layer_kick"),
                target_track_coord="tc1_tg2_tr3",
            ),
            SetPushTransferMode(mode="merge"),
            PushLayerToMA3(
                layer_id=LayerId("layer_kick"),
                scope=MA3PushScope.LAYER_MAIN,
                target_mode=MA3PushTargetMode.SAVED_ROUTE,
                apply_mode=MA3PushApplyMode.MERGE,
            ),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_overwrite_send_requires_confirmation_before_dispatch(monkeypatch):
    app = QApplication.instance() or QApplication([])
    base = replace(
        _ma3_push_selection_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        layers=[
            replace(
                _ma3_push_selection_presentation().layers[0],
                is_selected=True,
                sync_target_label="tc1_tg2_tr3",
            )
        ],
    )
    harness = _ManualPushHarness(base)
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    prompts: list[str] = []
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: ("Overwrite", True),
    )
    monkeypatch.setattr(
        widget._action_router,
        "_confirm_send_to_saved_ma3_route",
        lambda **_kwargs: MA3PushApplyMode.OVERWRITE,
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.question",
        lambda _parent, _title, text, *_args: prompts.append(text)
        or QMessageBox.StandardButton.No,
    )
    try:
        _render_for_hit_testing(widget)

        send_action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "transfer.workspace_open"
            and str(action.params.get("direction", "")).lower() == "push"
        )
        widget._trigger_contract_action(send_action)

        assert harness.intents == [
            RefreshMA3PushTracks(target_track_coord="tc1_tg2_tr3"),
            SetPushTransferMode(mode="overwrite"),
        ]
        assert prompts
        assert any("Overwrite Track 3 (tc1_tg2_tr3)?" in prompt for prompt in prompts)
        assert any("Selected EZ events: 1 event" in prompt for prompt in prompts)
    finally:
        widget.close()
        app.processEvents()


def test_manual_pull_workspace_dialog_shows_all_groups_and_tracks_for_selected_timecode():
    app = QApplication.instance() or QApplication([])
    dialog = ManualPullWorkspaceDialog()
    timecode_picks: list[int] = []
    group_picks: list[int] = []
    track_picks: list[str] = []
    dialog.timecode_selected.connect(lambda timecode_no: timecode_picks.append(int(timecode_no)))
    dialog.track_group_selected.connect(lambda group_no: group_picks.append(int(group_no)))
    dialog.track_selected.connect(lambda coord: track_picks.append(str(coord)))
    dialog.set_flow(
        ManualPullFlowPresentation(
            workspace_active=True,
            available_timecodes=[
                ManualPullTimecodeOptionPresentation(number=1, name=None),
                ManualPullTimecodeOptionPresentation(number=2, name="Verse"),
            ],
            selected_timecode_no=2,
            available_track_groups=[
                ManualPullTrackGroupOptionPresentation(
                    number=1,
                    name="Drums",
                    track_count=1,
                ),
                ManualPullTrackGroupOptionPresentation(
                    number=4,
                    name="FX",
                    track_count=2,
                ),
            ],
            selected_track_group_no=4,
            available_tracks=[
                ManualPullTrackOptionPresentation(
                    coord="tc2_tg1_tr1",
                    name="Kick",
                    number=1,
                    event_count=2,
                ),
                ManualPullTrackOptionPresentation(
                    coord="tc2_tg4_tr7",
                    name="Lasers",
                    number=7,
                    note="Stage Left",
                    event_count=5,
                ),
                ManualPullTrackOptionPresentation(
                    coord="tc2_tg4_tr8",
                    name="Fire",
                    number=8,
                    event_count=1,
                ),
            ],
            selected_source_track_coords=["tc2_tg4_tr7"],
            active_source_track_coord="tc2_tg4_tr7",
            source_track_coord="tc2_tg4_tr7",
            available_events=[
                ManualPullEventOptionPresentation(
                    event_id="ma3_evt_1",
                    label="Cue 1",
                    start=1.0,
                    end=1.5,
                ),
            ],
            selected_ma3_event_ids=["ma3_evt_1"],
            available_target_layers=[
                ManualPullTargetOptionPresentation(layer_id=LayerId("layer_kick"), name="Kick"),
                ManualPullTargetOptionPresentation(
                    layer_id=LayerId("__manual_pull__:create_new_layer"),
                    name="+ Create New Layer...",
                ),
            ],
            target_layer_id=LayerId("__manual_pull__:create_new_layer"),
            import_mode="main",
        )
    )
    try:
        assert sorted(dialog._timecode_picker._buttons_by_number) == [1, 2]
        assert sorted(dialog._source_browser._group_buttons) == [1, 4]
        assert sorted(dialog._source_browser._track_buttons) == [
            "tc2_tg1_tr1",
            "tc2_tg4_tr7",
            "tc2_tg4_tr8",
        ]
        assert dialog._source_browser._group_buttons[4].isChecked() is True
        assert dialog._source_browser._track_buttons["tc2_tg4_tr7"].isChecked() is True
        assert dialog.selected_source_track_coord() == "tc2_tg4_tr7"
        assert dialog.selected_import_mode() == "main"
        assert "TG4 FX" in dialog._source_summary.text()
        assert "TR7 Lasers" in dialog._source_summary.text()
        assert "Create a new EZ event layer" in dialog._destination_summary.text()

        dialog._source_browser._group_buttons[1].click()
        dialog._source_browser._track_buttons["tc2_tg1_tr1"].click()
        dialog._timecode_picker._buttons_by_number[1].click()
        app.processEvents()

        assert group_picks == [1]
        assert track_picks == ["tc2_tg1_tr1"]
        assert timecode_picks == [1]
    finally:
        dialog.close()
        app.processEvents()


def test_manual_pull_workspace_dialog_describes_section_layer_creation():
    app = QApplication.instance() or QApplication([])
    dialog = ManualPullWorkspaceDialog()
    dialog.set_flow(
        ManualPullFlowPresentation(
            workspace_active=True,
            available_timecodes=[ManualPullTimecodeOptionPresentation(number=2, name="Verse")],
            selected_timecode_no=2,
            available_track_groups=[
                ManualPullTrackGroupOptionPresentation(
                    number=4,
                    name="FX",
                    track_count=1,
                )
            ],
            selected_track_group_no=4,
            available_tracks=[
                ManualPullTrackOptionPresentation(
                    coord="tc2_tg4_tr7",
                    name="Lasers",
                    number=7,
                    event_count=5,
                ),
            ],
            selected_source_track_coords=["tc2_tg4_tr7"],
            active_source_track_coord="tc2_tg4_tr7",
            source_track_coord="tc2_tg4_tr7",
            available_events=[
                ManualPullEventOptionPresentation(
                    event_id="ma3_evt_1",
                    label="Cue 1",
                    start=1.0,
                    end=1.5,
                ),
            ],
            selected_ma3_event_ids=["ma3_evt_1"],
            available_target_layers=[
                ManualPullTargetOptionPresentation(
                    layer_id=LayerId("__manual_pull__:create_new_section_layer"),
                    name="+ Create Section Layer...",
                    kind=LayerKind.SECTION,
                ),
            ],
            target_layer_id=LayerId("__manual_pull__:create_new_section_layer"),
            import_mode="main",
        )
    )
    try:
        assert dialog.selected_import_mode() == "main"
        assert "Create a new EZ section layer" in dialog._destination_summary.text()
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
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
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

        assert intents == []
    finally:
        widget.close()
        app.processEvents()


__all__ = [name for name in globals() if name.startswith("test_")]
