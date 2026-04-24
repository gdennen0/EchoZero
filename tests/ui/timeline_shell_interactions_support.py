"""Interaction-driven timeline-shell support cases.
Exists to keep click, drag, keyboard, and transport coverage separate from fixtures and transfer tests.
Connects the compatibility wrapper to the bounded interaction support slice.
"""

from tests.ui.timeline_shell_shared_support import *  # noqa: F401,F403


def test_shift_click_event_dispatches_additive_selection_mode():
    app = QApplication.instance() or QApplication([])
    intents: list[SelectEvent] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
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
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
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
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
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
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        rect, layer_id = widget._canvas._row_body_select_rects[0]
        assert layer_id == LayerId("layer_kick")

        _click_rect(widget, rect)

        assert intents == [SelectLayer(LayerId("layer_kick"))]
    finally:
        widget.close()
        app.processEvents()


def test_main_rows_expose_active_hit_targets_without_take_row_duplicates():
    app = QApplication.instance() or QApplication([])
    presentation = _selection_test_presentation()
    presentation.layers[0].header_controls = [
        LayerHeaderControlPresentation(
            control_id="set_active_playback_target",
            label="ACTIVE",
            kind="toggle",
        ),
        LayerHeaderControlPresentation(control_id="send_to_ma3", label="Send"),
    ]
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)
    try:
        _render_for_hit_testing(widget)

        assert len(widget._canvas._active_rects) == len(presentation.layers)
        assert len(widget._canvas._push_rects) == len(
            [
                layer
                for layer in presentation.layers
                if layer.kind is LayerKind.EVENT and layer.main_take_id is not None
            ]
        )
        assert widget._canvas._pull_rects == []
    finally:
        widget.close()
        app.processEvents()


def test_audio_layer_header_hides_transfer_hit_targets():
    app = QApplication.instance() or QApplication([])
    presentation = _audio_pipeline_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)
    try:
        _render_for_hit_testing(widget)

        assert len(widget._canvas._active_rects) == len(presentation.layers)
        assert widget._canvas._push_rects == []
        assert widget._canvas._pull_rects == []
    finally:
        widget.close()
        app.processEvents()


def test_layer_presentation_declares_header_controls():
    presentation = _selection_test_presentation()
    layer = presentation.layers[0]

    layer.header_controls = [
        LayerHeaderControlPresentation(
            control_id="set_active_playback_target", label="ACTIVE", kind="toggle"
        ),
        LayerHeaderControlPresentation(control_id="send_to_ma3", label="Send"),
    ]

    assert [control.control_id for control in layer.header_controls] == [
        "set_active_playback_target",
        "send_to_ma3",
    ]


def test_selected_audio_layer_declares_pipeline_header_control():
    presentation = _audio_pipeline_presentation()

    selected_layer_controls = [
        control.control_id for control in presentation.layers[0].header_controls
    ]
    unselected_layer_controls = [
        control.control_id for control in presentation.layers[1].header_controls
    ]

    assert selected_layer_controls == [
        "set_active_playback_target",
        "layer_pipeline_actions",
    ]
    assert "layer_pipeline_actions" not in unselected_layer_controls


def test_ruler_click_dispatches_seek():
    app = QApplication.instance() or QApplication([])
    presentation = _selection_test_presentation()
    widget, intents = _seek_tracking_widget(presentation)
    try:
        _render_for_hit_testing(widget)

        QTest.mouseClick(
            widget._ruler,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(520, 12),
        )
        QApplication.processEvents()

        assert intents == [Seek(2.0)]
    finally:
        widget.close()
        app.processEvents()


def test_ruler_click_dispatches_seek_using_scroll_offset():
    app = QApplication.instance() or QApplication([])
    presentation = replace(
        _selection_test_presentation(), scroll_x=200.0, end_time_label="00:12.00"
    )
    widget, intents = _seek_tracking_widget(presentation)
    try:
        _render_for_hit_testing(widget)

        QTest.mouseClick(
            widget._ruler,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(520, 12),
        )
        QApplication.processEvents()

        assert intents == [Seek(4.0)]
    finally:
        widget.close()
        app.processEvents()


def test_main_row_active_click_dispatches_playback_target_intent_only():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        selected_take_id=TakeId("take_alt"),
        layers=[
            replace(
                _selection_test_presentation().layers[0],
                is_selected=True,
                is_playback_active=False,
            )
        ],
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        active_rect, active_layer_id = widget._canvas._active_rects[0]

        _click_rect(widget, active_rect)

        assert intents == [
            SetActivePlaybackTarget(layer_id=active_layer_id, take_id=None),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_layer_header_renders_selection_background_and_active_button_independently():
    app = QApplication.instance() or QApplication([])
    base = _selection_test_presentation()
    selected_layer = replace(
        base.layers[0],
        is_selected=True,
        is_playback_active=False,
    )
    playback_layer = replace(
        base.layers[0],
        layer_id=LayerId("layer_snare"),
        title="Snare",
        is_selected=False,
        is_playback_active=True,
    )
    header_controls = [
        LayerHeaderControlPresentation(
            control_id="set_active_playback_target",
            label="ACTIVE",
            kind="toggle",
            active=False,
        ),
        LayerHeaderControlPresentation(control_id="send_to_ma3", label="Send"),
    ]
    selected_layer = replace(selected_layer, header_controls=header_controls)
    playback_layer = replace(
        playback_layer,
        header_controls=[
            replace(header_controls[0], active=True),
            header_controls[1],
        ],
    )
    slots = HeaderSlots(
        rect=QRectF(0, 0, 320, 72),
        title_rect=QRectF(16, 8, 140, 20),
        subtitle_rect=QRectF(16, 30, 140, 16),
        status_rect=QRectF(16, 50, 120, 16),
        controls_rect=QRectF(160, 8, 144, 18),
        toggle_rect=QRectF(292, 50, 16, 16),
        metadata_rect=QRectF(0, 0, 0, 0),
    )

    selected_image = QImage(320, 72, QImage.Format.Format_ARGB32)
    selected_image.fill(QColor("#000000"))
    playback_image = QImage(320, 72, QImage.Format.Format_ARGB32)
    playback_image.fill(QColor("#000000"))

    selected_painter = QPainter(selected_image)
    selected_hit_targets = LayerHeaderBlock().paint(selected_painter, slots, selected_layer)
    selected_painter.end()

    playback_painter = QPainter(playback_image)
    playback_hit_targets = LayerHeaderBlock().paint(playback_painter, slots, playback_layer)
    playback_painter.end()

    selected_header_color = selected_image.pixelColor(12, 12)
    playback_header_color = playback_image.pixelColor(12, 12)
    selected_active_rect = dict(selected_hit_targets.control_rects)["set_active_playback_target"]
    playback_active_rect = dict(playback_hit_targets.control_rects)["set_active_playback_target"]
    selected_button_color = selected_image.pixelColor(
        int(selected_active_rect.left()) + 3, int(selected_active_rect.top()) + 9
    )
    playback_button_color = playback_image.pixelColor(
        int(playback_active_rect.left()) + 3, int(playback_active_rect.top()) + 9
    )

    assert selected_header_color.name() == "#202833"
    assert playback_header_color.name() == "#1b212a"
    assert selected_button_color.name() == "#18202a"
    assert playback_button_color.name() == "#2b6bf0"
    app.processEvents()


def test_layer_header_send_control_routes_then_dispatches_typed_push_intents(monkeypatch):
    app = QApplication.instance() or QApplication([])
    presentation = replace(
        _selection_test_presentation(),
        active_song_id="song_alpha",
        active_song_title="Alpha Song",
        active_song_version_id="song_version_alpha",
        active_song_version_label="Original",
        active_song_version_ma3_timecode_pool_no=101,
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt")],
    )
    presentation.layers[0].header_controls = [
        LayerHeaderControlPresentation(
            control_id="set_active_playback_target",
            label="ACTIVE",
            kind="toggle",
        ),
        LayerHeaderControlPresentation(control_id="send_to_ma3", label="Send"),
    ]
    harness = _ManualPushHarness(
        presentation
    )
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    try:
        picks = iter(
            [
                ("Track 3 (tc1_tg2_tr3) - Bass [8 existing]", True),
                ("Merge", True),
            ]
        )
        monkeypatch.setattr(
            "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
            lambda *args, **kwargs: next(picks),
        )
        _render_for_hit_testing(widget)

        push_rect, _ = widget._canvas._push_rects[0]
        _click_rect(widget, push_rect)

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
        assert widget.presentation.layers[0].sync_target_label == "tc1_tg2_tr3"
    finally:
        widget.close()
        app.processEvents()


def test_layer_header_pipeline_control_opens_workspace_pipeline_menu(monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_audio_pipeline_presentation())
    opened: list[InspectorAction] = []
    menu_labels: list[str] = []
    try:
        _render_for_hit_testing(widget)

        monkeypatch.setattr(
            widget._action_router,
            "open_object_action_settings",
            lambda action: opened.append(action),
        )

        def _choose_settings(menu, *_args, **_kwargs):
            menu_labels.extend(
                action.text() for action in menu.actions() if not action.isSeparator()
            )
            return next(
                action
                for action in menu.actions()
                if action.text() == "Open Extract Stems Settings"
            )

        monkeypatch.setattr(
            "echozero.ui.qt.timeline.widget.QMenu.exec",
            _choose_settings,
        )

        pipeline_rect, layer_id = widget._canvas._pipeline_action_rects[0]
        _click_rect(widget, pipeline_rect)

        assert layer_id == LayerId("layer_song")
        assert menu_labels == [
            "Open Extract Stems Settings",
            "Run Extract Stems",
        ]
        assert opened == [
            InspectorAction(
                action_id="timeline.extract_stems",
                label="Extract Stems",
                kind="settings",
                params={"layer_id": LayerId("layer_song")},
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_layer_header_no_longer_exposes_pull_control():
    app = QApplication.instance() or QApplication([])
    presentation = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        layers=[replace(_selection_test_presentation().layers[0], is_selected=True)],
    )
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)
    try:
        _render_for_hit_testing(widget)
        assert widget._canvas._pull_rects == []
    finally:
        widget.close()
        app.processEvents()


def test_layer_header_click_dispatches_toggle_and_range_selection_modes():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _multi_layer_selection_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
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


def test_ruler_drag_in_region_mode_dispatches_create_region_intent():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["region"].click()
        QApplication.processEvents()

        _mouse_drag(
            widget._ruler,
            [QPoint(420, 12), QPoint(620, 12)],
        )

        assert len(intents) == 1
        assert isinstance(intents[0], CreateRegion)
        assert intents[0].time_range.start == 1.0
        assert intents[0].time_range.end == 3.0
    finally:
        widget.close()
        app.processEvents()


def test_ruler_click_in_region_mode_does_not_dispatch_create_region():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["region"].click()
        QApplication.processEvents()

        QTest.mouseClick(
            widget._ruler,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(520, 12),
        )
        QApplication.processEvents()

        assert intents == []
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
        _mouse_drag(
            widget._canvas,
            [QPoint(start_x, y), QPoint(start_x + 100, y), QPoint(start_x + 200, y)],
        )

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
        _mouse_drag(
            widget._canvas,
            [QPoint(start_x, y), QPoint(start_x + 100, y), QPoint(start_x + 200, y)],
        )

        assert intents == [Seek(4.0), Seek(5.0), Seek(6.0)]
    finally:
        widget.close()
        app.processEvents()


def test_canvas_empty_non_selection_space_no_longer_dispatches_seek():

    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
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
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
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
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        QTest.keyClick(widget._canvas, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QApplication.processEvents()

        assert intents == [SelectAllEvents()]
    finally:
        widget.close()
        app.processEvents()


def test_r_key_switches_canvas_to_region_mode():
    app = QApplication.instance() or QApplication([])
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: presentation
    )
    try:
        _render_for_hit_testing(widget)
        QTest.keyClick(widget._canvas, Qt.Key.Key_R, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()

        assert widget._canvas._edit_mode == "region"
        assert widget._editor_bar._mode_buttons["region"].isChecked() is True
    finally:
        widget.close()
        app.processEvents()


def test_arrow_keys_dispatch_nudge_selected_events():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["move"].click()

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


def test_select_mode_keys_dispatch_navigation_intents():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        QTest.keyClick(widget._canvas, Qt.Key.Key_Period, Qt.KeyboardModifier.ShiftModifier)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Comma, Qt.KeyboardModifier.ShiftModifier)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Down)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Up)
        QApplication.processEvents()

        assert intents == [
            SelectAdjacentEventInSelectedLayer(direction=1),
            SelectAdjacentEventInSelectedLayer(direction=-1),
            SelectAdjacentLayer(direction=1),
            SelectAdjacentLayer(direction=-1),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_move_mode_up_down_dispatch_adjacent_layer_move_intents():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["move"].click()

        QTest.keyClick(widget._canvas, Qt.Key.Key_Up)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Down)
        QApplication.processEvents()

        assert intents == [
            MoveSelectedEventsToAdjacentLayer(direction=-1),
            MoveSelectedEventsToAdjacentLayer(direction=1),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_ctrl_d_dispatches_duplicate_selected_events():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
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
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["move"].click()

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
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["move"].click()

        for rect, _, _, candidate_event_id in widget._canvas._event_rects:
            if str(candidate_event_id) == "main_evt":
                start = rect.center().toPoint()
                break
        else:
            raise AssertionError("Missing event rect for main_evt")

        target_rect = next(
            rect
            for rect, layer_id in widget._canvas._event_drop_rects
            if layer_id == LayerId("layer_snare")
        )
        target = QPoint(start.x(), int(target_rect.center().y()))

        _mouse_drag(widget._canvas, [start, target])

        assert intents == [
            MoveSelectedEvents(delta_seconds=0.0, target_layer_id=LayerId("layer_snare"))
        ]
    finally:
        widget.close()
        app.processEvents()


def test_settings_button_triggers_launcher_preferences_action() -> None:
    app = QApplication.instance() or QApplication([])
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)

    class _FakeAction:
        def __init__(self) -> None:
            self.trigger_calls = 0

        def trigger(self) -> None:
            self.trigger_calls += 1

    action = _FakeAction()
    widget._launcher_actions = {"preferences": action}

    try:
        widget._editor_bar._settings_button.click()

        assert action.trigger_calls == 1
    finally:
        widget.close()
        app.processEvents()


def test_draw_mode_drag_dispatches_create_event_intent():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        widget._editor_bar._mode_buttons["draw"].click()
        QApplication.processEvents()

        lane_rect, _, _ = widget._canvas._event_lane_rects[0]
        y = int(lane_rect.center().y())
        _mouse_drag(
            widget._canvas,
            [
                QPoint(int(lane_rect.left() + 260), y),
                QPoint(int(lane_rect.left() + 330), y),
            ],
        )

        assert len(intents) == 1
        assert isinstance(intents[0], CreateEvent)
        assert intents[0].layer_id == LayerId("layer_kick")
        assert intents[0].take_id == TakeId("take_main")
        assert intents[0].time_range.start < intents[0].time_range.end
    finally:
        widget.close()
        app.processEvents()


def test_erase_mode_click_dispatches_delete_events_intent():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        widget._editor_bar._mode_buttons["erase"].click()
        QApplication.processEvents()
        _click_event_rect(widget, "main_evt")

        assert intents == [
            DeleteEvents(
                event_ids=[EventId("main_evt")],
                event_refs=[
                    EventRef(
                        layer_id=LayerId("layer_kick"),
                        take_id=TakeId("take_main"),
                        event_id=EventId("main_evt"),
                    )
                ],
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_marquee_drag_dispatches_batch_event_selection_intent():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        main_rect = next(
            rect
            for rect, _, _, event_id in widget._canvas._event_rects
            if str(event_id) == "main_evt"
        )
        take_rect = next(
            rect
            for rect, _, _, event_id in widget._canvas._event_rects
            if str(event_id) == "take_evt"
        )
        _mouse_drag(
            widget._canvas,
            [
                QPoint(int(main_rect.left() - 6), int(main_rect.top() - 4)),
                QPoint(int(take_rect.right() + 6), int(take_rect.bottom() + 4)),
            ],
        )

        assert intents == [
            SetSelectedEvents(
                event_ids=[EventId("main_evt"), EventId("take_evt")],
                event_refs=[
                    EventRef(
                        layer_id=LayerId("layer_kick"),
                        take_id=TakeId("take_main"),
                        event_id=EventId("main_evt"),
                    ),
                    EventRef(
                        layer_id=LayerId("layer_kick"),
                        take_id=TakeId("take_alt"),
                        event_id=EventId("take_evt"),
                    ),
                ],
                anchor_layer_id=LayerId("layer_kick"),
                anchor_take_id=TakeId("take_alt"),
                selected_layer_ids=[LayerId("layer_kick")],
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_delete_key_dispatches_delete_events_for_current_selection():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt")],
        layers=[
            replace(
                _selection_test_presentation().layers[0],
                events=[
                    replace(_selection_test_presentation().layers[0].events[0], is_selected=True)
                ],
            )
        ],
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        QTest.keyClick(widget._canvas, Qt.Key.Key_Delete)
        QApplication.processEvents()

        assert intents == [DeleteEvents(event_ids=[EventId("main_evt")])]
    finally:
        widget.close()
        app.processEvents()
    DeleteTransferPreset,

__all__ = [name for name in globals() if name.startswith("test_")]
