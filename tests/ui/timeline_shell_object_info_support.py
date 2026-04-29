"""Object-info and context-menu timeline-shell support cases.
Exists to keep inspector and context hit-target coverage separate from transfer and interaction tests.
Connects the compatibility wrapper to the bounded object-info support slice.
"""

from tests.ui.timeline_shell_shared_support import *  # noqa: F401,F403

def test_main_row_event_click_dispatches_main_take_identity():
    app = QApplication.instance() or QApplication([])
    intents: list[SelectEvent] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
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


def test_object_info_panel_shows_current_song_version_without_selection():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(
        replace(
            _song_switching_presentation(),
            selected_layer_id=None,
            selected_layer_ids=[],
            active_playback_layer_id=None,
        )
    )
    try:
        _render_for_hit_testing(widget)

        info = widget._object_info.text()
        assert "Song Alpha Song" in info
        assert "song id: song_alpha" in info
        assert "version label: Festival Edit" in info
        assert widget._object_info._kind.text() == "Song Version"
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
        assert "playback state: Set Active" in info
        assert "selected identity: Layer Kick (layer_kick)" in info
        assert "playback target: none" in info
    finally:
        widget.close()
        app.processEvents()


def test_object_info_panel_exposes_ma3_routing_button_outside_context_menu():
    app = QApplication.instance() or QApplication([])
    harness = _SelectionInspectorHarness(
        replace(_selection_test_presentation(), selected_layer_id=LayerId("layer_kick"))
    )
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    try:
        _render_for_hit_testing(widget)

        action_ids = set(widget._object_info._action_buttons)

        assert "route_layer_to_ma3_track" in action_ids
        assert "send_layer_to_ma3" not in action_ids
        assert "send_selected_events_to_ma3" not in action_ids
        assert "send_to_different_track_once" not in action_ids
    finally:
        widget.close()
        app.processEvents()


def test_object_info_panel_audio_output_route_selector_dispatches_set_layer_output_bus():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _selection_test_presentation()
    audio_layer = replace(
        base.layers[0],
        kind=LayerKind.AUDIO,
        source_audio_path="kick.wav",
        playback_source_ref="kick.wav",
    )
    presentation = replace(
        base,
        layers=[audio_layer],
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        playback_output_channels=4,
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        output_route_combo = widget._object_info._output_bus_combo
        assert output_route_combo.isVisible() is True
        option_labels = [
            output_route_combo.itemText(index) for index in range(output_route_combo.count())
        ]
        assert option_labels == [
            "Default Output (1/2)",
            "Outputs 1/2",
            "Outputs 3/4",
        ]

        output_route_combo.setCurrentIndex(2)
        widget._object_info._output_bus_apply_btn.click()

        assert intents[-1] == SetLayerOutputBus(
            layer_id=LayerId("layer_kick"),
            output_bus="outputs_3_4",
        )
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
        assert "playback state: Set Active" in info
        assert "selected identity: Event Main (main_evt) on Kick / Main take (take_main)" in info
        assert "playback target: none" in info
    finally:
        widget.close()
        app.processEvents()


def test_object_info_panel_batch_buttons_dispatch_scoped_event_intents():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _selection_test_presentation()
    presentation = replace(
        base,
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt")],
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        assert "selection.select_every_other" in widget._object_info._action_buttons
        assert "selection.renumber_cues_from_one" in widget._object_info._action_buttons

        widget._object_info._action_buttons["selection.select_every_other"].click()
        widget._object_info._action_buttons["selection.renumber_cues_from_one"].click()

        assert intents == [
            SelectEveryOtherEvents(scope=EventBatchScope(mode="selected_events")),
            RenumberEventCueNumbers(
                scope=EventBatchScope(mode="selected_events"),
                start_at=1,
                step=1,
            ),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_take_lane_event_click_dispatches_take_identity():
    app = QApplication.instance() or QApplication([])
    intents: list[SelectEvent] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
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
        assert "playback state: Set Active" in info
        assert "selected identity: Event Take (take_evt) on Kick / Take 2 (take_alt)" in info
        assert "playback target: none" in info
    finally:
        widget.close()
        app.processEvents()


def test_object_info_panel_event_clip_button_routes_to_runtime_preview():
    app = QApplication.instance() or QApplication([])

    class _Runtime(_SelectionInspectorHarness):
        def __init__(self, presentation: TimelinePresentation):
            super().__init__(presentation)
            self.runtime_audio = None
            self.preview_calls: list[tuple[object, object, object]] = []

        def preview_event_clip(self, *, layer_id, take_id=None, event_id):
            self.preview_calls.append((layer_id, take_id, event_id))

    base = _selection_test_presentation()
    presentation = replace(
        base,
        layers=[
            replace(
                base.layers[0],
                source_audio_path="kick.wav",
                playback_source_ref="kick.wav",
            )
        ],
    )
    runtime = _Runtime(presentation)
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        _render_for_hit_testing(widget)
        _click_event_rect(widget, "main_evt")

        assert widget._object_info._event_preview_card.isVisible() is True
        assert "0.50s clip" in widget._object_info._event_preview_meta.text()
        widget._object_info._action_buttons["preview_event_clip"].click()

        assert runtime.preview_calls == [
            (LayerId("layer_kick"), TakeId("take_main"), EventId("main_evt"))
        ]
    finally:
        widget.close()
        app.processEvents()


def test_object_info_panel_shows_playback_target_separately_when_nothing_is_selected():
    app = QApplication.instance() or QApplication([])
    presentation = replace(
        _selection_test_presentation(),
        active_playback_layer_id=LayerId("layer_kick"),
        active_playback_take_id=TakeId("take_main"),
    )
    widget = TimelineWidget(presentation)
    try:
        _render_for_hit_testing(widget)

        info = widget._object_info.text()
        assert info == "\n".join(
            [
                "Timeline",
                "selected identity: none",
                "playback target: Active Kick / Main take (take_main)",
            ]
        )
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


def test_zoom_does_not_rebuild_object_info_contract(monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_selection_test_presentation())
    try:
        _render_for_hit_testing(widget)

        set_contract_calls = 0
        set_plans_calls = 0
        original_set_contract = widget._object_info.set_contract
        original_set_plans = widget._object_info.set_action_settings_plans

        def _set_contract(*args, **kwargs):
            nonlocal set_contract_calls
            set_contract_calls += 1
            return original_set_contract(*args, **kwargs)

        def _set_plans(*args, **kwargs):
            nonlocal set_plans_calls
            set_plans_calls += 1
            return original_set_plans(*args, **kwargs)

        monkeypatch.setattr(widget._object_info, "set_contract", _set_contract)
        monkeypatch.setattr(widget._object_info, "set_action_settings_plans", _set_plans)

        before_pps = widget.presentation.pixels_per_second
        widget._zoom_from_input(120, anchor_x=widget._canvas._header_width + 120.0)
        QApplication.processEvents()

        assert widget.presentation.pixels_per_second > before_pps
        assert set_contract_calls == 0
        assert set_plans_calls == 0
    finally:
        widget.close()
        app.processEvents()


def test_object_info_panel_keeps_no_takes_indication_for_empty_layer():
    app = QApplication.instance() or QApplication([])
    presentation = replace(
        _no_takes_layer_presentation(), selected_layer_id=LayerId("layer_empty")
    )
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
        action_ids = [
            action.action_id for section in contract.context_sections for action in section.actions
        ]
        menu_labels = [action.text() for action in menu.actions() if not action.isSeparator()]

        assert "overwrite_main" not in action_ids
        assert "merge_main" not in action_ids
        assert "delete_take" not in action_ids
        assert "Overwrite Main" not in menu_labels
        assert "Merge Main" not in menu_labels
        assert "Delete Take" not in menu_labels
    finally:
        widget.close()
        app.processEvents()


def test_no_takes_layer_has_no_toggle_takes_intent_path():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _no_takes_layer_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        assert widget._canvas._toggle_rects == []
    finally:
        widget.close()
        app.processEvents()

    assert intents == []


def test_timeline_background_right_click_does_not_crash(monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_selection_test_presentation())
    try:
        _render_for_hit_testing(widget)

        monkeypatch.setattr(
            "echozero.ui.qt.timeline.widget.QMenu.exec",
            lambda self, *_args, **_kwargs: None,
        )

        shown = widget._canvas._show_context_menu(
            QPointF(widget._canvas._header_width + 120.0, widget._canvas._top_padding + 6.0)
        )

        assert isinstance(shown, bool)
    finally:
        widget.close()
        app.processEvents()


def test_context_menu_event_uses_qcontextmenuevent_pos(monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_selection_test_presentation())
    captured: dict[str, object] = {}
    try:
        event = QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse,
            QPoint(40, 50),
            QPoint(400, 500),
        )

        def _show_context_menu(pos: QPointF, *, global_pos=None) -> bool:
            captured["pos"] = pos
            captured["global_pos"] = global_pos
            return True

        monkeypatch.setattr(widget._canvas, "_show_context_menu", _show_context_menu)

        widget._canvas.contextMenuEvent(event)

        assert captured["pos"] == QPointF(40.0, 50.0)
        assert captured["global_pos"] == QPoint(400, 500)
        assert event.isAccepted() is True
    finally:
        widget.close()
        app.processEvents()


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
        contract_labels = [
            action.label for section in contract.context_sections for action in section.actions
        ]

        assert menu_labels == contract_labels
        assert "Overwrite Main" in menu_labels
        assert "Merge Main" in menu_labels
        assert "Delete Take" in menu_labels
    finally:
        widget.close()
        app.processEvents()


def test_context_menu_timeline_hit_is_scoped_to_timeline_actions():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_selection_test_presentation())
    try:
        _render_for_hit_testing(widget)

        contract = build_timeline_inspector_contract(
            widget.presentation,
            hit_target=TimelineInspectorHitTarget(kind="timeline", time_seconds=1.25),
        )
        menu = widget._canvas._build_context_menu(contract, hit_kind="timeline")
        labels = [action.text() for action in menu.actions() if not action.isSeparator()]

        assert "Add Song" in labels
        assert "Add SMPTE Layer" in labels
        assert any(label.startswith("Seek to") for label in labels)
        assert "Push to MA3" not in labels
        assert "Route Audio to Master" not in labels
        assert "Overwrite Main" not in labels
        assert "Delete Take" not in labels
    finally:
        widget.close()
        app.processEvents()


def test_context_menu_layer_hit_is_scoped_to_layer_actions():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_selection_test_presentation())
    try:
        _render_for_hit_testing(widget)

        contract = build_timeline_inspector_contract(
            widget.presentation,
            hit_target=TimelineInspectorHitTarget(
                kind="layer", layer_id=LayerId("layer_kick"), time_seconds=1.0
            ),
        )
        menu = widget._canvas._build_context_menu(contract, hit_kind="layer")
        labels = [action.text() for action in menu.actions() if not action.isSeparator()]

        assert "Route Layer to MA3 Track" in labels
        assert "Send Layer to MA3" in labels
        assert "Send Selected Events to MA3" in labels
        assert "Send to Different Track Once" in labels
        assert "Push to MA3" not in labels
        assert "Pull from MA3" not in labels
        assert "Route Audio to Master" in labels
        assert "Select Every Other in Layer" in labels
        assert "Renumber Cues from 1 in Layer" in labels
        assert "Add Song" not in labels
        assert "Nudge Left" not in labels
        assert "Overwrite Main" not in labels
    finally:
        widget.close()
        app.processEvents()


def test_context_menu_smpte_layer_shows_import_smpte_audio_action():
    app = QApplication.instance() or QApplication([])
    base = _selection_test_presentation()
    smpte_layer = replace(
        base.layers[0],
        kind=LayerKind.AUDIO,
        title="SMPTE Layer",
    )
    presentation = replace(
        base,
        layers=[smpte_layer],
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
    )
    widget = TimelineWidget(presentation)
    try:
        _render_for_hit_testing(widget)

        contract = build_timeline_inspector_contract(
            widget.presentation,
            hit_target=TimelineInspectorHitTarget(
                kind="layer", layer_id=LayerId("layer_kick"), time_seconds=1.0
            ),
        )
        menu = widget._canvas._build_context_menu(contract, hit_kind="layer")
        labels = [action.text() for action in menu.actions() if not action.isSeparator()]

        assert "Import SMPTE Audio" in labels
    finally:
        widget.close()
        app.processEvents()


def test_context_menu_take_hit_is_scoped_to_take_actions():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_selection_test_presentation())
    try:
        _render_for_hit_testing(widget)

        contract = build_timeline_inspector_contract(
            widget.presentation,
            hit_target=TimelineInspectorHitTarget(
                kind="take",
                layer_id=LayerId("layer_kick"),
                take_id=TakeId("take_alt"),
                time_seconds=2.0,
            ),
        )
        menu = widget._canvas._build_context_menu(contract, hit_kind="take")
        labels = [action.text() for action in menu.actions() if not action.isSeparator()]

        assert "Overwrite Main" in labels
        assert "Merge Main" in labels
        assert "Delete Take" in labels
        assert "Select Every Other in Take" in labels
        assert "Renumber Cues from 1 in Take" in labels
        assert "Import Event Layer from MA3" in labels
        assert "Route Layer to MA3 Track" in labels
        assert "Send Layer to MA3" in labels
        assert "Send Selected Events to MA3" in labels
        assert "Send to Different Track Once" in labels
        assert "Route Audio to Master" not in labels
        assert "Add Song" not in labels
    finally:
        widget.close()
        app.processEvents()


def test_context_menu_event_hit_is_scoped_to_event_selection_actions():
    app = QApplication.instance() or QApplication([])
    selected = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt")],
    )
    widget = TimelineWidget(selected)
    try:
        _render_for_hit_testing(widget)

        contract = build_timeline_inspector_contract(
            widget.presentation,
            hit_target=TimelineInspectorHitTarget(
                kind="event",
                layer_id=LayerId("layer_kick"),
                take_id=TakeId("take_main"),
                event_id=EventId("main_evt"),
                time_seconds=1.0,
            ),
        )
        menu = widget._canvas._build_context_menu(contract, hit_kind="event")
        labels = [action.text() for action in menu.actions() if not action.isSeparator()]

        assert "Nudge Left" in labels
        assert "Nudge Right" in labels
        assert "Duplicate" in labels
        assert "Select Every Other" in labels
        assert "Renumber Cues from 1" in labels
        assert "Import Event Layer from MA3" in labels
        assert "Route Layer to MA3 Track" in labels
        assert "Send Layer to MA3" in labels
        assert "Send Selected Events to MA3" in labels
        assert "Send to Different Track Once" in labels
        assert "Push to MA3" not in labels
        assert "Route Audio to Master" not in labels
        assert "Add Song" not in labels
    finally:
        widget.close()
        app.processEvents()


def test_context_menu_unselected_main_event_can_send_single_event_to_ma3():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_selection_test_presentation())
    try:
        _render_for_hit_testing(widget)

        contract = build_timeline_inspector_contract(
            widget.presentation,
            hit_target=TimelineInspectorHitTarget(
                kind="event",
                layer_id=LayerId("layer_kick"),
                take_id=TakeId("take_main"),
                event_id=EventId("main_evt"),
                time_seconds=1.0,
            ),
        )
        menu = widget._canvas._build_context_menu(contract, hit_kind="event")
        labels = [action.text() for action in menu.actions() if not action.isSeparator()]

        assert "Send Event to MA3" in labels
        assert "Send Selected Events to MA3" not in labels
        assert "Import Event Layer from MA3" in labels
        assert "Route Layer to MA3 Track" in labels
        assert "Send Layer to MA3" in labels
        assert "Send to Different Track Once" in labels
    finally:
        widget.close()
        app.processEvents()



__all__ = [name for name in globals() if name.startswith("test_")]
