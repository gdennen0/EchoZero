"""Interaction-driven timeline-shell support cases.
Exists to keep click, drag, keyboard, and transport coverage separate from fixtures and transfer tests.
Connects the compatibility wrapper to the bounded interaction support slice.
"""

from echozero.application.presentation.models import (
    SectionCuePresentation,
    SectionRegionPresentation,
)
from echozero.application.timeline.intents import ReplaceSectionCues
from echozero.ui.FEEL import TIMELINE_ADD_MODE_DEFAULT_EVENT_DURATION_SECONDS
from echozero.ui.qt.timeline.section_manager import SectionCueDraft

from tests.ui.timeline_shell_shared_support import *  # noqa: F401,F403


def _fix_mode_test_presentation() -> TimelinePresentation:
    return TimelinePresentation(
        timeline_id=TimelineId("timeline_fix_mode"),
        title="Fix Mode",
        layers=[
            LayerPresentation(
                layer_id=LayerId("layer_onsets"),
                title="Onsets",
                main_take_id=TakeId("take_onsets"),
                kind=LayerKind.EVENT,
                events=[
                    EventPresentation(
                        event_id=EventId("onset_a"),
                        start=1.0,
                        end=1.2,
                        label="Onset",
                    ),
                    EventPresentation(
                        event_id=EventId("onset_b"),
                        start=2.0,
                        end=2.2,
                        label="Onset",
                    ),
                ],
                status=LayerStatusPresentation(source_layer_id="source_audio"),
            ),
            LayerPresentation(
                layer_id=LayerId("layer_kick"),
                title="Kick",
                main_take_id=TakeId("take_kick"),
                kind=LayerKind.EVENT,
                is_selected=True,
                events=[
                    EventPresentation(
                        event_id=EventId("kick_evt"),
                        start=1.0,
                        end=1.2,
                        label="Kick",
                        source_event_id="onset_a",
                    )
                ],
                status=LayerStatusPresentation(source_layer_id="source_audio"),
            ),
        ],
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        selected_take_id=TakeId("take_kick"),
        pixels_per_second=120.0,
        end_time_label="00:05.00",
    )


def _section_overlay_scope_presentation() -> TimelinePresentation:
    return TimelinePresentation(
        timeline_id=TimelineId("timeline_section_overlay_scope"),
        title="Section Overlay Scope",
        layers=[
            LayerPresentation(
                layer_id=LayerId("layer_events"),
                title="Events",
                main_take_id=TakeId("take_events"),
                kind=LayerKind.EVENT,
                status=LayerStatusPresentation(),
            ),
            LayerPresentation(
                layer_id=LayerId("layer_sections"),
                title="Sections",
                main_take_id=TakeId("take_sections"),
                kind=LayerKind.SECTION,
                status=LayerStatusPresentation(),
            ),
        ],
        section_regions=[
            SectionRegionPresentation(
                cue_id="cue_intro",
                start=0.0,
                end=4.0,
                cue_ref="Q1",
                name="Intro",
                color="#f0b74f",
            )
        ],
        section_cues=[
            SectionCuePresentation(
                cue_id="cue_intro",
                start=0.0,
                cue_ref="Q1",
                name="Intro",
                color="#f0b74f",
            )
        ],
        pixels_per_second=100.0,
        end_time_label="00:08.00",
    )


def test_section_overlay_renders_only_on_section_layer_rows():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_section_overlay_scope_presentation())
    try:
        _render_for_hit_testing(widget)
        row_rect_by_layer = {
            layer_id: rect
            for rect, layer_id, take_id in widget._canvas._row_body_select_rects
            if take_id is None
        }
        event_row_rect = row_rect_by_layer[LayerId("layer_events")]
        section_row_rect = row_rect_by_layer[LayerId("layer_sections")]

        sample_x = int(event_row_rect.left()) + 43
        image = widget._canvas.grab().toImage()
        event_row_color = image.pixelColor(sample_x, int(event_row_rect.center().y())).name()
        section_row_color = image.pixelColor(sample_x, int(section_row_rect.center().y())).name()
        base_row_color = QColor(widget._canvas._style.canvas.row_fill_hex).name()

        assert event_row_color == base_row_color
        assert section_row_color != base_row_color
    finally:
        widget.close()
        app.processEvents()


def test_ruler_does_not_render_section_region_highlight():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_section_overlay_scope_presentation())
    try:
        _render_for_hit_testing(widget)
        ruler_image = widget._ruler.grab().toImage()
        sample_x = int(widget._canvas._header_width + 150)
        sample_y = int(widget._ruler.height() * 0.5)
        sampled = ruler_image.pixelColor(sample_x, sample_y).name()
        expected_background = QColor(widget._ruler._block.style.background_hex).name()
        assert sampled == expected_background
    finally:
        widget.close()
        app.processEvents()


def test_double_click_section_label_dispatches_replace_section_cues_with_new_name(monkeypatch):
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _section_overlay_scope_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        monkeypatch.setattr(
            "echozero.ui.qt.timeline.widget.QInputDialog.getText",
            lambda *args, **kwargs: ("Verse", True),
        )
        _render_for_hit_testing(widget)
        label_rect, _cue_id = widget._canvas._section_label_rects[0]
        QTest.mouseDClick(
            widget._canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(int(label_rect.center().x()), int(label_rect.center().y())),
        )
        QApplication.processEvents()

        replace_intents = [intent for intent in intents if isinstance(intent, ReplaceSectionCues)]
        assert len(replace_intents) == 1
        assert [cue.name for cue in replace_intents[0].cues] == ["Verse"]
    finally:
        widget.close()
        app.processEvents()


def test_double_click_section_boundary_opens_section_editor_and_dispatches_changes(monkeypatch):
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _section_overlay_scope_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )

    class _FakeSectionManagerDialog:
        class DialogCode:
            Accepted = 1

        def __init__(
            self,
            _presentation,
            parent=None,
            *,
            cues=None,
            worksheet_title=None,
            selected_cue_id=None,
        ):
            del parent
            del cues
            del worksheet_title
            assert str(selected_cue_id) == "cue_intro"

        def exec(self):
            return self.DialogCode.Accepted

        def section_cue_drafts(self):
            return [
                SectionCueDraft(
                    cue_id="cue_intro",
                    start=0.0,
                    cue_ref="Q1",
                    name="Intro",
                    color="#112233",
                    notes="updated",
                )
            ]

    try:
        monkeypatch.setattr(
            "echozero.ui.qt.timeline.widget.SectionManagerDialog",
            _FakeSectionManagerDialog,
        )
        _render_for_hit_testing(widget)
        boundary_rect, _cue_id = widget._canvas._section_boundary_rects[0]
        QTest.mouseDClick(
            widget._canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(int(boundary_rect.center().x()), int(boundary_rect.center().y())),
        )
        QApplication.processEvents()

        replace_intents = [intent for intent in intents if isinstance(intent, ReplaceSectionCues)]
        assert len(replace_intents) == 1
        assert replace_intents[0].cues[0].color == "#112233"
        assert replace_intents[0].cues[0].notes == "updated"
    finally:
        widget.close()
        app.processEvents()


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

        rect, layer_id, take_id = widget._canvas._row_body_select_rects[0]
        assert layer_id == LayerId("layer_kick")
        assert take_id is None

        _click_rect(widget, rect)

        assert intents == [SelectLayer(LayerId("layer_kick"))]
    finally:
        widget.close()
        app.processEvents()


def test_take_row_empty_space_click_dispatches_take_selection_not_seek():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        rect, layer_id, take_id = widget._canvas._row_body_select_rects[1]
        assert layer_id == LayerId("layer_kick")
        assert take_id == TakeId("take_alt")

        _click_rect(widget, rect)

        assert intents == [SelectTake(layer_id=LayerId("layer_kick"), take_id=TakeId("take_alt"))]
    finally:
        widget.close()
        app.processEvents()


def test_main_rows_expose_mix_hit_targets_without_take_row_duplicates():
    app = QApplication.instance() or QApplication([])
    presentation = _selection_test_presentation()
    presentation.layers[0].header_controls = [
        LayerHeaderControlPresentation(
            control_id="set_layer_mute",
            label="M",
            kind="toggle",
        ),
        LayerHeaderControlPresentation(control_id="send_to_ma3", label="Send"),
    ]
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)
    try:
        _render_for_hit_testing(widget)

        assert len(widget._canvas._mute_rects) == len(presentation.layers)
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

        assert len(widget._canvas._mute_rects) == len(presentation.layers)
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
            control_id="set_layer_mute", label="M", kind="toggle"
        ),
        LayerHeaderControlPresentation(control_id="send_to_ma3", label="Send"),
    ]

    assert [control.control_id for control in layer.header_controls] == [
        "set_layer_mute",
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
        "set_layer_mute",
        "set_layer_solo",
        "layer_pipeline_actions",
    ]
    assert unselected_layer_controls == [
        "set_layer_mute",
        "set_layer_solo",
    ]


def test_section_layer_header_control_opens_layer_scoped_section_manager(monkeypatch):
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _section_overlay_scope_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )

    class _FakeSectionManagerDialog:
        class DialogCode:
            Accepted = 1

        def __init__(
            self,
            _presentation,
            parent=None,
            *,
            cues=None,
            worksheet_title=None,
            selected_cue_id=None,
        ):
            del parent
            del selected_cue_id
            assert worksheet_title == "Sections Cue Stack"
            assert cues is not None
            assert len(cues) == 0

        def exec(self):
            return self.DialogCode.Accepted

        def section_cue_drafts(self):
            return [
                SectionCueDraft(
                    cue_id=None,
                    start=1.0,
                    cue_ref="Cue 1",
                    name="Intro",
                    cue_number=1,
                )
            ]

    try:
        monkeypatch.setattr(
            "echozero.ui.qt.timeline.widget.SectionManagerDialog",
            _FakeSectionManagerDialog,
        )
        _render_for_hit_testing(widget)
        assert len(widget._canvas._section_manager_rects) == 1
        rect, layer_id = widget._canvas._section_manager_rects[0]
        assert layer_id == LayerId("layer_sections")
        _click_rect(widget, rect)

        replace_intents = [intent for intent in intents if isinstance(intent, ReplaceSectionCues)]
        assert len(replace_intents) == 1
        assert replace_intents[0].target_layer_id == LayerId("layer_sections")
        assert replace_intents[0].cues[0].cue_number == 1
    finally:
        widget.close()
        app.processEvents()


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


def test_main_row_mute_click_dispatches_layer_mute_intent():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _selection_test_presentation()
    presentation = replace(
        base,
        layers=[replace(base.layers[0], kind=LayerKind.AUDIO)],
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        mute_rect, mute_layer_id = widget._canvas._mute_rects[0]

        _click_rect(widget, mute_rect)

        assert intents == [SetLayerMute(layer_id=mute_layer_id, muted=True)]
    finally:
        widget.close()
        app.processEvents()


def test_main_row_solo_click_dispatches_layer_solo_intent():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _selection_test_presentation()
    presentation = replace(
        base,
        layers=[replace(base.layers[0], kind=LayerKind.AUDIO)],
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        solo_rect, solo_layer_id = widget._canvas._solo_rects[0]

        _click_rect(widget, solo_rect)

        assert intents == [SetLayerSolo(layer_id=solo_layer_id, soloed=True)]
    finally:
        widget.close()
        app.processEvents()


def test_main_row_event_layer_hides_mute_and_solo_controls():
    app = QApplication.instance() or QApplication([])
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)
    try:
        _render_for_hit_testing(widget)
        assert widget._canvas._mute_rects == []
        assert widget._canvas._solo_rects == []
    finally:
        widget.close()
        app.processEvents()


def test_layer_header_renders_selection_background_and_mute_button_independently():
    app = QApplication.instance() or QApplication([])
    base = _selection_test_presentation()
    selected_layer = replace(
        base.layers[0],
        is_selected=True,
        muted=False,
    )
    playback_layer = replace(
        base.layers[0],
        layer_id=LayerId("layer_snare"),
        title="Snare",
        is_selected=False,
        muted=True,
    )
    header_controls = [
        LayerHeaderControlPresentation(
            control_id="set_layer_mute",
            label="M",
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
        controls_rect=QRectF(160, 8, 104, 18),
        active_rect=QRectF(272, 12, 14, 14),
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
    selected_active_rect = dict(selected_hit_targets.control_rects)["set_layer_mute"]
    playback_active_rect = dict(playback_hit_targets.control_rects)["set_layer_mute"]
    selected_center_x = int(selected_active_rect.center().x())
    selected_center_y = int(selected_active_rect.center().y())
    playback_center_x = int(playback_active_rect.center().x())
    playback_center_y = int(playback_active_rect.center().y())
    selected_button_color = selected_image.pixelColor(
        selected_center_x, selected_center_y
    )
    playback_button_color = playback_image.pixelColor(
        playback_center_x, playback_center_y
    )

    assert selected_header_color.name() == "#202833"
    assert playback_header_color.name() == "#1b212a"
    assert selected_button_color.name() != playback_button_color.name()
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
            control_id="set_layer_mute",
            label="M",
            kind="toggle",
        ),
        LayerHeaderControlPresentation(control_id="send_to_ma3", label="Send"),
    ]
    harness = _ManualPushHarness(
        presentation
    )
    widget = TimelineWidget(harness.presentation(), on_intent=harness.dispatch)
    try:
        monkeypatch.setattr(
            widget._action_router,
            "_open_manual_push_route_popup",
            lambda **_kwargs: "tc1_tg2_tr3",
        )
        monkeypatch.setattr(
            "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
            lambda *args, **kwargs: ("Merge", True),
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
        assert "Open Extract Stems Settings" in menu_labels
        assert "Run Extract Stems" in menu_labels
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


def test_layer_header_pipeline_menu_ignores_non_pipeline_timeline_actions(monkeypatch):
    app = QApplication.instance() or QApplication([])
    presentation = replace(
        _audio_pipeline_presentation(),
        selected_event_ids=[EventId("evt_selected")],
    )
    widget = TimelineWidget(presentation)
    menu_labels: list[str] = []
    try:
        _render_for_hit_testing(widget)

        def _capture_only(menu, *_args, **_kwargs):
            menu_labels.extend(
                action.text() for action in menu.actions() if not action.isSeparator()
            )
            return None

        monkeypatch.setattr(
            "echozero.ui.qt.timeline.widget.QMenu.exec",
            _capture_only,
        )

        pipeline_rect, _layer_id = widget._canvas._pipeline_action_rects[0]
        _click_rect(widget, pipeline_rect)

        assert any("Extract Stems" in label for label in menu_labels)
        assert not any("Nudge" in label for label in menu_labels)
        assert not any("Duplicate" in label for label in menu_labels)
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


def test_ruler_click_existing_region_in_region_mode_dispatches_select_region_intent():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = replace(
        _selection_test_presentation(),
        regions=[
            RegionPresentation(
                region_id=RegionId("region_1"),
                start=1.0,
                end=2.0,
                label="Verse",
                color="#99aabb",
            )
        ],
    )
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
            QPoint(470, 12),
        )
        QApplication.processEvents()

        assert intents == [SelectRegion(region_id=RegionId("region_1"))]
    finally:
        widget.close()
        app.processEvents()


def test_ruler_double_click_existing_region_in_region_mode_dispatches_update_region_intent(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = replace(
        _selection_test_presentation(),
        regions=[
            RegionPresentation(
                region_id=RegionId("region_1"),
                start=1.0,
                end=2.0,
                label="Verse",
                color="#99aabb",
                kind="song",
            )
        ],
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )

    class _FakeRegionPropertiesDialog:
        class DialogCode:
            Accepted = 1

        def __init__(self, _draft, parent=None):
            del parent

        def exec(self):
            return self.DialogCode.Accepted

        def values(self):
            class _Values:
                start = 1.25
                end = 2.75
                label = "Verse A"
                color = "#112233"

            return _Values()

    try:
        monkeypatch.setattr(
            "echozero.ui.qt.timeline.widget.RegionPropertiesDialog",
            _FakeRegionPropertiesDialog,
        )
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["region"].click()
        QApplication.processEvents()

        QTest.mouseDClick(
            widget._ruler,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(470, 12),
        )
        QApplication.processEvents()

        assert any(
            isinstance(intent, SelectRegion) and intent.region_id == RegionId("region_1")
            for intent in intents
        )
        update_intents = [
            intent for intent in intents if isinstance(intent, UpdateRegion)
        ]
        assert len(update_intents) == 1
        update_intent = update_intents[0]
        assert update_intent.region_id == RegionId("region_1")
        assert update_intent.time_range.start == 1.25
        assert update_intent.time_range.end == 2.75
        assert update_intent.label == "Verse A"
        assert update_intent.color == "#112233"
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


def test_f_key_switches_canvas_to_fix_mode() -> None:
    app = QApplication.instance() or QApplication([])
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: presentation
    )
    try:
        _render_for_hit_testing(widget)
        QTest.keyClick(widget._canvas, Qt.Key.Key_F, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()

        assert widget._canvas._edit_mode == "fix"
        assert widget._canvas._fix_action == "select"
        assert widget._editor_bar._mode_buttons["fix"].isChecked() is True
        assert widget._editor_bar._fix_action_buttons["select"].isChecked() is True
    finally:
        widget.close()
        app.processEvents()


def test_fix_demoted_toggle_is_visible_only_in_fix_mode() -> None:
    app = QApplication.instance() or QApplication([])
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: presentation
    )
    try:
        _render_for_hit_testing(widget)

        assert not widget._editor_bar._fix_include_demoted_button.isVisible()

        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()
        assert widget._editor_bar._fix_include_demoted_button.isVisible()

        widget._editor_bar._mode_buttons["select"].click()
        QApplication.processEvents()
        assert not widget._editor_bar._fix_include_demoted_button.isVisible()
    finally:
        widget.close()
        app.processEvents()


def test_demoted_events_render_only_in_fix_mode() -> None:
    app = QApplication.instance() or QApplication([])
    base = _selection_test_presentation()
    presentation = replace(
        base,
        layers=[
            replace(
                base.layers[0],
                events=[
                    *base.layers[0].events,
                    EventPresentation(
                        event_id=EventId("demoted_evt"),
                        start=2.7,
                        end=3.0,
                        label="Demoted",
                        badges=["demoted"],
                    ),
                ],
            )
        ],
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: presentation
    )
    try:
        _render_for_hit_testing(widget)
        event_ids = {
            str(event_id)
            for _rect, _layer_id, _take_id, event_id in widget._canvas._event_rects
        }
        assert "demoted_evt" not in event_ids

        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()
        widget._canvas.repaint()
        QApplication.processEvents()
        event_ids = {
            str(event_id)
            for _rect, _layer_id, _take_id, event_id in widget._canvas._event_rects
        }
        assert "demoted_evt" in event_ids

        widget._editor_bar._mode_buttons["move"].click()
        QApplication.processEvents()
        widget._canvas.repaint()
        QApplication.processEvents()
        event_ids = {
            str(event_id)
            for _rect, _layer_id, _take_id, event_id in widget._canvas._event_rects
        }
        assert "demoted_evt" not in event_ids
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_shortcuts_switch_fix_tools() -> None:
    app = QApplication.instance() or QApplication([])
    presentation = _fix_mode_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()

        QTest.keyClick(widget._canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()
        assert widget._canvas._fix_action == "remove"
        assert widget._editor_bar._fix_action_buttons["remove"].isChecked() is True

        QTest.keyClick(widget._canvas, Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()
        assert widget._canvas._fix_action == "select"
        assert widget._editor_bar._fix_action_buttons["select"].isChecked() is True

        QTest.keyClick(widget._canvas, Qt.Key.Key_C, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()
        assert widget._canvas._fix_action == "promote"
        assert widget._editor_bar._fix_action_buttons["promote"].isChecked() is True
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_shift_z_demotes_selected_event() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _fix_mode_test_presentation()
    presentation = replace(
        base,
        selected_event_ids=[EventId("kick_evt")],
        selected_event_refs=[
            EventRef(
                layer_id=LayerId("layer_kick"),
                take_id=TakeId("take_kick"),
                event_id=EventId("kick_evt"),
            )
        ],
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()
        assert widget._canvas._fix_action == "select"

        QTest.keyClick(widget._canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ShiftModifier)
        QApplication.processEvents()

        assert intents == [
            CommitRejectedEventsReview(
                event_refs=[
                    EventRef(
                        layer_id=LayerId("layer_kick"),
                        take_id=TakeId("take_kick"),
                        event_id=EventId("kick_evt"),
                    )
                ],
            )
        ]
        assert widget._canvas._fix_action == "select"
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_shift_c_promotes_selected_event() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _fix_mode_test_presentation()
    presentation = replace(
        base,
        selected_event_ids=[EventId("kick_evt")],
        selected_event_refs=[
            EventRef(
                layer_id=LayerId("layer_kick"),
                take_id=TakeId("take_kick"),
                event_id=EventId("kick_evt"),
            )
        ],
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()
        assert widget._canvas._fix_action == "select"

        QTest.keyClick(widget._canvas, Qt.Key.Key_C, Qt.KeyboardModifier.ShiftModifier)
        QApplication.processEvents()

        assert intents == [
            CommitVerifiedEventsReview(
                event_refs=[
                    EventRef(
                        layer_id=LayerId("layer_kick"),
                        take_id=TakeId("take_kick"),
                        event_id=EventId("kick_evt"),
                    )
                ],
            )
        ]
        assert widget._canvas._fix_action == "select"
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_shift_z_demotes_multiple_selected_events_in_batch() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _selection_test_presentation()
    presentation = replace(
        base,
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt"), EventId("take_evt")],
        selected_event_refs=[
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
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()

        QTest.keyClick(widget._canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ShiftModifier)
        QApplication.processEvents()

        assert intents == [
            CommitRejectedEventsReview(
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
                ]
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_shift_c_promotes_multiple_selected_events_in_batch() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _selection_test_presentation()
    presentation = replace(
        base,
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[EventId("main_evt"), EventId("take_evt")],
        selected_event_refs=[
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
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()

        QTest.keyClick(widget._canvas, Qt.Key.Key_C, Qt.KeyboardModifier.ShiftModifier)
        QApplication.processEvents()

        assert intents == [
            CommitVerifiedEventsReview(
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
                ]
            )
        ]
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
    presentation = replace(
        _selection_test_presentation(),
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

        QTest.keyClick(widget._canvas, Qt.Key.Key_Right)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Left)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Period, Qt.KeyboardModifier.ShiftModifier)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Comma, Qt.KeyboardModifier.ShiftModifier)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Down)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Up)
        QApplication.processEvents()

        assert intents == [
            SelectAdjacentEventInSelectedLayer(direction=1),
            SelectAdjacentEventInSelectedLayer(direction=-1),
            SelectAdjacentEventInSelectedLayer(direction=1),
            SelectAdjacentEventInSelectedLayer(direction=-1),
            SelectAdjacentLayer(direction=1),
            SelectAdjacentLayer(direction=-1),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_keys_dispatch_navigation_intents():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = replace(
        _selection_test_presentation(),
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
        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()

        QTest.keyClick(widget._canvas, Qt.Key.Key_Right)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Left)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Down)
        QApplication.processEvents()

        assert intents == [
            SelectAdjacentEventInSelectedLayer(direction=1),
            SelectAdjacentEventInSelectedLayer(direction=-1),
            SelectTake(layer_id=LayerId("layer_kick"), take_id=TakeId("take_alt")),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_demoted_navigation_toggle_updates_adjacent_event_intent() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = replace(
        _selection_test_presentation(),
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
        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()
        assert widget._editor_bar._fix_include_demoted_button.text().endswith("Off")

        widget._editor_bar._fix_include_demoted_button.click()
        QApplication.processEvents()
        assert widget._editor_bar._fix_include_demoted_button.isChecked()
        assert widget._editor_bar._fix_include_demoted_button.text().endswith("On")

        QTest.keyClick(widget._canvas, Qt.Key.Key_Right)
        QTest.keyClick(widget._canvas, Qt.Key.Key_D)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Left)
        QApplication.processEvents()

        assert not widget._editor_bar._fix_include_demoted_button.isChecked()
        assert widget._editor_bar._fix_include_demoted_button.text().endswith("Off")
        assert intents == [
            SelectAdjacentEventInSelectedLayer(direction=1, include_demoted=True),
            SelectAdjacentEventInSelectedLayer(direction=-1, include_demoted=False),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_arrow_keys_dispatch_event_navigation_even_with_overlay_candidates() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _fix_mode_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()

        QTest.keyClick(widget._canvas, Qt.Key.Key_Right)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Left)
        QApplication.processEvents()

        assert intents == [
            SelectAdjacentEventInSelectedLayer(direction=1),
            SelectAdjacentEventInSelectedLayer(direction=-1),
        ]
        assert widget._canvas._focused_fix_overlay() is None
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_preview_keys_navigate_preview_overlay_candidates_locally() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _fix_mode_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()

        QTest.keyClick(widget._canvas, Qt.Key.Key_Period)
        QApplication.processEvents()
        focused = widget._canvas._focused_fix_overlay()
        assert focused is not None
        assert focused[1] == LayerId("layer_kick")
        assert focused[3] == "onset_a"

        QTest.keyClick(widget._canvas, Qt.Key.Key_Period)
        QApplication.processEvents()
        focused = widget._canvas._focused_fix_overlay()
        assert focused is not None
        assert focused[1] == LayerId("layer_kick")
        assert focused[3] == "onset_a"

        QTest.keyClick(widget._canvas, Qt.Key.Key_Comma)
        QApplication.processEvents()
        focused = widget._canvas._focused_fix_overlay()
        assert focused is not None
        assert focused[1] == LayerId("layer_kick")
        assert focused[3] == "onset_a"

        assert len(intents) == 3
        for intent in intents:
            assert isinstance(intent, SetSelectedEvents)
            assert intent.anchor_layer_id == LayerId("layer_kick")
            assert intent.anchor_take_id == TakeId("take_kick")
            assert intent.selected_layer_ids == [LayerId("layer_kick")]
        assert intents[0].event_refs == [
            EventRef(
                layer_id=LayerId("layer_onsets"),
                take_id=TakeId("take_onsets"),
                event_id=EventId("onset_a"),
            )
        ]
        assert intents[1].event_refs == [
            EventRef(
                layer_id=LayerId("layer_onsets"),
                take_id=TakeId("take_onsets"),
                event_id=EventId("onset_a"),
            )
        ]
        assert intents[2].event_refs == [
            EventRef(
                layer_id=LayerId("layer_onsets"),
                take_id=TakeId("take_onsets"),
                event_id=EventId("onset_a"),
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_select_mode_arrow_keys_dispatch_navigation_with_layer_selected_only():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = replace(
        _selection_test_presentation(),
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        selected_take_id=TakeId("take_main"),
        selected_event_ids=[],
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)

        QTest.keyClick(widget._canvas, Qt.Key.Key_Right)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Left)
        QApplication.processEvents()

        assert intents == [
            SelectAdjacentEventInSelectedLayer(direction=1),
            SelectAdjacentEventInSelectedLayer(direction=-1),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_select_mode_right_arrow_centers_on_newly_selected_event():
    app = QApplication.instance() or QApplication([])
    layer_id = LayerId("layer_kick")
    take_id = TakeId("take_main")
    main_event_id = EventId("main_evt")
    next_event_id = EventId("next_evt")
    initial_presentation = replace(
        _selection_test_presentation(),
        end_time_label="01:30.00",
        selected_layer_id=layer_id,
        selected_layer_ids=[layer_id],
        selected_take_id=take_id,
        selected_event_ids=[main_event_id],
        selected_event_refs=[
            EventRef(layer_id=layer_id, take_id=take_id, event_id=main_event_id)
        ],
        layers=[
            replace(
                _selection_test_presentation().layers[0],
                events=[
                    EventPresentation(
                        event_id=main_event_id,
                        start=1.0,
                        end=1.5,
                        label="Main",
                    ),
                    EventPresentation(
                        event_id=next_event_id,
                        start=30.0,
                        end=31.0,
                        label="Next",
                    ),
                ],
            )
        ],
    )
    presentation_state = initial_presentation

    def _on_intent(intent):
        nonlocal presentation_state
        if isinstance(intent, SelectAdjacentEventInSelectedLayer) and intent.direction > 0:
            presentation_state = replace(
                presentation_state,
                selected_event_ids=[next_event_id],
                selected_event_refs=[
                    EventRef(layer_id=layer_id, take_id=take_id, event_id=next_event_id)
                ],
            )
            return presentation_state
        return presentation_state

    widget = TimelineWidget(presentation_state, on_intent=_on_intent)
    try:
        _render_for_hit_testing(widget)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Right)
        QApplication.processEvents()

        viewport = max(1, widget._scroll.viewport().width())
        header_width = widget._canvas._header_width
        content_center_x = float(header_width) + (max(1.0, float(viewport - header_width)) * 0.5)
        event_center_x = float(header_width) + (
            ((30.0 + 31.0) * 0.5) * widget.presentation.pixels_per_second
        ) - float(widget.presentation.scroll_x)

        assert widget.presentation.selected_event_ids == [next_event_id]
        assert abs(event_center_x - content_center_x) <= 1.5
    finally:
        widget.close()
        app.processEvents()


def test_select_mode_shift_space_triggers_event_clip_preview_action(monkeypatch):
    app = QApplication.instance() or QApplication([])
    layer_id = LayerId("layer_kick")
    take_id = TakeId("take_main")
    event_id = EventId("main_evt")
    selected_ref = EventRef(layer_id=layer_id, take_id=take_id, event_id=event_id)
    base = _selection_test_presentation()
    presentation = replace(
        base,
        selected_layer_id=layer_id,
        selected_layer_ids=[layer_id],
        selected_take_id=take_id,
        selected_event_ids=[event_id],
        selected_event_refs=[selected_ref],
        layers=[
            replace(
                base.layers[0],
                source_audio_path="/tmp/preview.wav",
                playback_source_ref="/tmp/preview.wav",
            )
        ],
    )
    captured_actions: list[object] = []
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)
    try:
        monkeypatch.setattr(
            widget._action_router,
            "trigger_contract_action",
            lambda action: captured_actions.append(action),
        )
        _render_for_hit_testing(widget)
        QTest.keyClick(
            widget._canvas,
            Qt.Key.Key_Space,
            Qt.KeyboardModifier.ShiftModifier,
        )
        QApplication.processEvents()

        assert len(captured_actions) == 1
        action = captured_actions[0]
        assert isinstance(action, InspectorAction)
        assert action.action_id == "preview_event_clip"
        assert action.params["layer_id"] == layer_id
        assert action.params["take_id"] == take_id
        assert action.params["event_id"] == event_id
    finally:
        widget.close()
        app.processEvents()


def test_space_dispatches_stop_transport_intent_when_timeline_is_playing(monkeypatch):
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _selection_test_presentation()
    presentation_state = replace(base, is_playing=True)
    captured_actions: list[object] = []

    def _on_intent(intent):
        nonlocal presentation_state
        intents.append(intent)
        if isinstance(intent, Stop):
            presentation_state = replace(presentation_state, is_playing=False)
        return presentation_state

    widget = TimelineWidget(presentation_state, on_intent=_on_intent)
    try:
        monkeypatch.setattr(
            widget._action_router,
            "trigger_contract_action",
            lambda action: captured_actions.append(action),
        )
        _render_for_hit_testing(widget)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Space)
        QApplication.processEvents()

        assert intents == [Stop()]
        assert captured_actions == []
    finally:
        widget.close()
        app.processEvents()


def test_space_dispatches_play_transport_intent_when_timeline_is_stopped(monkeypatch):
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _selection_test_presentation()
    presentation_state = replace(base, is_playing=False)
    captured_actions: list[object] = []

    def _on_intent(intent):
        nonlocal presentation_state
        intents.append(intent)
        if isinstance(intent, Play):
            presentation_state = replace(presentation_state, is_playing=True)
        return presentation_state

    widget = TimelineWidget(presentation_state, on_intent=_on_intent)
    try:
        monkeypatch.setattr(
            widget._action_router,
            "trigger_contract_action",
            lambda action: captured_actions.append(action),
        )
        _render_for_hit_testing(widget)
        QTest.keyClick(widget._canvas, Qt.Key.Key_Space)
        QApplication.processEvents()

        assert intents == [Play()]
        assert captured_actions == []
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_shift_space_and_enter_preview_selected_event_clip(monkeypatch):
    app = QApplication.instance() or QApplication([])
    layer_id = LayerId("layer_kick")
    take_id = TakeId("take_main")
    event_id = EventId("main_evt")
    selected_ref = EventRef(layer_id=layer_id, take_id=take_id, event_id=event_id)
    base = _selection_test_presentation()
    presentation = replace(
        base,
        selected_layer_id=layer_id,
        selected_layer_ids=[layer_id],
        selected_take_id=take_id,
        selected_event_ids=[event_id],
        selected_event_refs=[selected_ref],
        layers=[
            replace(
                base.layers[0],
                source_audio_path="/tmp/preview.wav",
                playback_source_ref="/tmp/preview.wav",
            )
        ],
    )
    captured_actions: list[object] = []
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)
    try:
        monkeypatch.setattr(
            widget._action_router,
            "trigger_contract_action",
            lambda action: captured_actions.append(action),
        )
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()

        QTest.keyClick(
            widget._canvas,
            Qt.Key.Key_Space,
            Qt.KeyboardModifier.ShiftModifier,
        )
        QTest.keyClick(widget._canvas, Qt.Key.Key_Return)
        QApplication.processEvents()

        assert len(captured_actions) == 2
        for action in captured_actions:
            assert isinstance(action, InspectorAction)
            assert action.action_id == "preview_event_clip"
            assert action.params["layer_id"] == layer_id
            assert action.params["take_id"] == take_id
            assert action.params["event_id"] == event_id
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

        assert widget.presentation.follow_mode == FollowMode.CENTER
        _click_transport_rect(widget, "follow")
        assert widget.presentation.follow_mode == FollowMode.OFF
        _click_transport_rect(widget, "follow")
        assert widget.presentation.follow_mode == FollowMode.CENTER

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


def test_manual_horizontal_scroll_disengages_follow_mode():
    app = QApplication.instance() or QApplication([])
    demo = build_demo_app()
    widget = TimelineWidget(demo.presentation(), on_intent=demo.dispatch)
    try:
        _render_for_hit_testing(widget)

        assert widget.presentation.follow_mode == FollowMode.CENTER
        widget._scroll_horizontally_by_steps(120.0)
        QApplication.processEvents()

        assert widget.presentation.follow_mode == FollowMode.OFF
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


def test_option_dragging_selected_event_dispatches_copy_move_intent():
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

        _mouse_drag(
            widget._canvas,
            [start, QPoint(start.x() + 100, start.y())],
            modifiers=Qt.KeyboardModifier.AltModifier,
        )

        assert intents == [
            MoveSelectedEvents(
                delta_seconds=1.0,
                target_layer_id=None,
                copy_selected=True,
            )
        ]
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


def test_option_dragging_selected_event_over_other_layer_dispatches_copy_transfer_target():
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

        _mouse_drag(
            widget._canvas,
            [start, target],
            modifiers=Qt.KeyboardModifier.AltModifier,
        )

        assert intents == [
            MoveSelectedEvents(
                delta_seconds=0.0,
                target_layer_id=LayerId("layer_snare"),
                copy_selected=True,
            )
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
    fallback_action = _FakeAction()
    widget._launcher_actions = {"preferences": action, "project_settings": fallback_action}

    try:
        widget._editor_bar._settings_button.click()

        assert action.trigger_calls == 1
        assert fallback_action.trigger_calls == 0
    finally:
        widget.close()
        app.processEvents()


def test_settings_button_falls_back_to_launcher_project_settings_action() -> None:
    app = QApplication.instance() or QApplication([])
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)

    class _FakeAction:
        def __init__(self) -> None:
            self.trigger_calls = 0

        def trigger(self) -> None:
            self.trigger_calls += 1

    action = _FakeAction()
    widget._launcher_actions = {"project_settings": action}

    try:
        widget._editor_bar._settings_button.click()

        assert action.trigger_calls == 1
    finally:
        widget.close()
        app.processEvents()


def test_osc_settings_button_triggers_launcher_osc_settings_action() -> None:
    app = QApplication.instance() or QApplication([])
    presentation = _selection_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)

    class _FakeAction:
        def __init__(self) -> None:
            self.trigger_calls = 0

        def trigger(self) -> None:
            self.trigger_calls += 1

    action = _FakeAction()
    widget._launcher_actions = {"osc_settings": action}

    try:
        widget._editor_bar._osc_settings_button.click()

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


def test_draw_mode_shortcut_a_dispatches_create_event_at_playhead() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _selection_test_presentation()
    presentation = replace(
        base,
        playhead=2.25,
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        selected_take_id=TakeId("take_main"),
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["draw"].click()
        QApplication.processEvents()

        QTest.keyClick(widget._canvas, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()

        assert len(intents) == 1
        assert isinstance(intents[0], CreateEvent)
        assert intents[0].layer_id == LayerId("layer_kick")
        assert intents[0].take_id == TakeId("take_main")
        assert intents[0].time_range.start == 2.25
        assert intents[0].time_range.end == (
            2.25 + TIMELINE_ADD_MODE_DEFAULT_EVENT_DURATION_SECONDS
        )
    finally:
        widget.close()
        app.processEvents()


def test_draw_mode_toolbar_add_at_playhead_dispatches_create_event() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _selection_test_presentation()
    presentation = replace(
        base,
        playhead=1.5,
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        selected_take_id=TakeId("take_main"),
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["draw"].click()
        QApplication.processEvents()

        assert widget._editor_bar._add_event_at_playhead_button.isVisible() is True
        widget._editor_bar._add_event_at_playhead_button.click()
        QApplication.processEvents()

        assert len(intents) == 1
        assert isinstance(intents[0], CreateEvent)
        assert intents[0].layer_id == LayerId("layer_kick")
        assert intents[0].take_id == TakeId("take_main")
        assert intents[0].time_range.start == 1.5
        assert intents[0].time_range.end == (
            1.5 + TIMELINE_ADD_MODE_DEFAULT_EVENT_DURATION_SECONDS
        )
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


def test_fix_mode_plus_click_promotes_missing_correlated_onset() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _fix_mode_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QTest.keyClick(widget._canvas, Qt.Key.Key_C, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()

        missing_rect = next(
            rect
            for rect, layer_id, _take_id, source_event_id, _start, _end, matched in widget._canvas._fix_event_rects
            if layer_id == LayerId("layer_kick")
            and source_event_id == "onset_b"
            and matched is False
        )
        _click_rect(widget, missing_rect)

        assert len(intents) == 1
        assert isinstance(intents[0], CommitMissedEventReview)
        assert intents[0].layer_id == LayerId("layer_kick")
        assert intents[0].take_id == TakeId("take_kick")
        assert intents[0].label == "Kick"
        assert intents[0].source_event_id == "onset_b"
        assert intents[0].payload_ref == "onset_b"
        assert intents[0].time_range.start < intents[0].time_range.end
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_plus_click_promotes_existing_event() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _fix_mode_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QTest.keyClick(widget._canvas, Qt.Key.Key_C, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()
        _click_event_rect(widget, "kick_evt")

        assert intents == [
            CommitVerifiedEventsReview(
                event_refs=[
                    EventRef(
                        layer_id=LayerId("layer_kick"),
                        take_id=TakeId("take_kick"),
                        event_id=EventId("kick_evt"),
                    )
                ]
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_overlay_events_align_with_main_event_lane_y() -> None:
    app = QApplication.instance() or QApplication([])
    presentation = _fix_mode_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()

        main_rect = next(
            rect
            for rect, layer_id, _take_id, event_id in widget._canvas._event_rects
            if layer_id == LayerId("layer_kick") and str(event_id) == "kick_evt"
        )
        preview_rect = next(
            rect
            for rect, layer_id, _take_id, source_event_id, _start, _end, _matched in widget._canvas._fix_event_rects
            if layer_id == LayerId("layer_kick") and source_event_id == "onset_a"
        )

        assert abs(float(preview_rect.top()) - float(main_rect.top())) <= 0.5
        assert abs(float(preview_rect.height()) - float(main_rect.height())) <= 0.5
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_overlay_shows_unmatched_onsets_only_in_promote_tool() -> None:
    app = QApplication.instance() or QApplication([])
    presentation = _fix_mode_test_presentation()
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()

        select_mode_rects = [
            (layer_id, source_event_id, matched)
            for _rect, layer_id, _take_id, source_event_id, _start, _end, matched in widget._canvas._fix_event_rects
            if layer_id == LayerId("layer_kick")
        ]
        assert (LayerId("layer_kick"), "onset_a", True) in select_mode_rects
        assert not any(
            layer_id == LayerId("layer_kick")
            and source_event_id == "onset_b"
            for layer_id, source_event_id, _matched in select_mode_rects
        )

        QTest.keyClick(widget._canvas, Qt.Key.Key_C, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()
        widget._canvas.repaint()
        QApplication.processEvents()

        promote_mode_rects = [
            (layer_id, source_event_id, matched)
            for _rect, layer_id, _take_id, source_event_id, _start, _end, matched in widget._canvas._fix_event_rects
            if layer_id == LayerId("layer_kick")
        ]
        assert (LayerId("layer_kick"), "onset_a", True) in promote_mode_rects
        assert (LayerId("layer_kick"), "onset_b", False) in promote_mode_rects
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_does_not_cross_render_kick_snare_overlay_without_onset_lane() -> None:
    app = QApplication.instance() or QApplication([])
    presentation = TimelinePresentation(
        timeline_id=TimelineId("timeline_fix_mode_no_onsets"),
        title="Fix Mode No Onsets",
        layers=[
            LayerPresentation(
                layer_id=LayerId("layer_kick"),
                title="Kick",
                main_take_id=TakeId("take_kick"),
                kind=LayerKind.EVENT,
                is_selected=True,
                events=[
                    EventPresentation(
                        event_id=EventId("kick_evt"),
                        start=1.0,
                        end=1.2,
                        label="Kick",
                        source_event_id="kick_src",
                        badges=["demoted"],
                    )
                ],
                status=LayerStatusPresentation(
                    source_layer_id="source_audio",
                    source_label="binary_drum_classify · kick_events",
                    output_name="kick_events",
                ),
            ),
            LayerPresentation(
                layer_id=LayerId("layer_snare"),
                title="Snare",
                main_take_id=TakeId("take_snare"),
                kind=LayerKind.EVENT,
                events=[
                    EventPresentation(
                        event_id=EventId("snare_evt"),
                        start=2.0,
                        end=2.2,
                        label="Snare",
                        source_event_id="snare_src",
                        badges=["demoted"],
                    )
                ],
                status=LayerStatusPresentation(
                    source_layer_id="source_audio",
                    source_label="binary_drum_classify · snare_events",
                    output_name="snare_events",
                ),
            ),
        ],
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        selected_take_id=TakeId("take_kick"),
        pixels_per_second=120.0,
        end_time_label="00:05.00",
    )
    widget = TimelineWidget(presentation, on_intent=lambda intent: presentation)
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QApplication.processEvents()
        widget._canvas.repaint()
        QApplication.processEvents()

        assert widget._canvas._fix_event_rects == []
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_arrow_keys_navigate_preview_overlay_candidates_locally() -> None:
    """Keep the historical wrapper import stable for preview-overlay navigation coverage."""

    test_fix_mode_preview_keys_navigate_preview_overlay_candidates_locally()


def test_fix_mode_selecting_source_onset_updates_selection_and_preview(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _fix_mode_test_presentation()
    presentation_state = replace(
        base,
        layers=[
            replace(
                base.layers[0],
                source_audio_path="/tmp/onset-preview.wav",
                playback_source_ref="/tmp/onset-preview.wav",
            ),
            base.layers[1],
        ],
    )

    def _on_intent(intent):
        nonlocal presentation_state
        intents.append(intent)
        if isinstance(intent, SetSelectedEvents):
            anchor_layer_ids = list(intent.selected_layer_ids or [])
            if not anchor_layer_ids and intent.anchor_layer_id is not None:
                anchor_layer_ids = [intent.anchor_layer_id]
            presentation_state = replace(
                presentation_state,
                selected_event_ids=list(intent.event_ids),
                selected_event_refs=list(intent.event_refs or []),
                selected_layer_id=intent.anchor_layer_id,
                selected_layer_ids=anchor_layer_ids,
                selected_take_id=intent.anchor_take_id,
            )
        return presentation_state

    captured_actions: list[object] = []
    widget = TimelineWidget(
        presentation_state,
        on_intent=_on_intent,
    )
    try:
        monkeypatch.setattr(
            widget._action_router,
            "trigger_contract_action",
            lambda action: captured_actions.append(action),
        )
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QTest.keyClick(widget._canvas, Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()

        source_rect = next(
            rect
            for rect, layer_id, _take_id, source_event_id, _start, _end, matched in widget._canvas._fix_event_rects
            if layer_id == LayerId("layer_kick")
            and source_event_id == "onset_a"
            and matched is True
        )
        _click_rect(widget, source_rect)
        QApplication.processEvents()

        set_selected_intents = [
            intent for intent in intents if isinstance(intent, SetSelectedEvents)
        ]
        assert len(set_selected_intents) >= 1
        latest_selected = set_selected_intents[-1]
        assert latest_selected.event_refs == [
            EventRef(
                layer_id=LayerId("layer_onsets"),
                take_id=TakeId("take_onsets"),
                event_id=EventId("onset_a"),
            )
        ]
        assert latest_selected.anchor_layer_id == LayerId("layer_kick")
        assert latest_selected.anchor_take_id == TakeId("take_kick")
        assert latest_selected.selected_layer_ids == [LayerId("layer_kick")]

        QTest.keyClick(
            widget._canvas,
            Qt.Key.Key_Space,
            Qt.KeyboardModifier.ShiftModifier,
        )
        QApplication.processEvents()

        assert len(captured_actions) == 1
        action = captured_actions[0]
        assert isinstance(action, InspectorAction)
        assert action.action_id == "preview_event_clip"
        assert action.params["layer_id"] == LayerId("layer_onsets")
        assert action.params["take_id"] == TakeId("take_onsets")
        assert action.params["event_id"] == EventId("onset_a")
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_minus_click_removes_existing_event() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _fix_mode_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QTest.keyClick(widget._canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()
        _click_event_rect(widget, "kick_evt")

        assert intents == [
            CommitRejectedEventReview(
                layer_id=LayerId("layer_kick"),
                take_id=TakeId("take_kick"),
                event_id=EventId("kick_evt"),
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_minus_click_rejects_only_clicked_lane_when_event_ids_collide() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = TimelinePresentation(
        timeline_id=TimelineId("timeline_fix_mode_collision"),
        title="Fix Mode Collision",
        layers=[
            LayerPresentation(
                layer_id=LayerId("layer_snare_a"),
                title="Snare A",
                main_take_id=TakeId("take_snare_a"),
                kind=LayerKind.EVENT,
                is_selected=True,
                events=[
                    EventPresentation(
                        event_id=EventId("shared_evt"),
                        start=1.0,
                        end=1.2,
                        label="Snare",
                    )
                ],
                status=LayerStatusPresentation(source_layer_id="source_audio"),
            ),
            LayerPresentation(
                layer_id=LayerId("layer_snare_b"),
                title="Snare B",
                main_take_id=TakeId("take_snare_b"),
                kind=LayerKind.EVENT,
                events=[
                    EventPresentation(
                        event_id=EventId("shared_evt"),
                        start=1.0,
                        end=1.2,
                        label="Snare",
                    )
                ],
                status=LayerStatusPresentation(source_layer_id="source_audio"),
            ),
        ],
        selected_layer_id=LayerId("layer_snare_a"),
        selected_layer_ids=[LayerId("layer_snare_a")],
        selected_take_id=TakeId("take_snare_a"),
        pixels_per_second=120.0,
        end_time_label="00:05.00",
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QTest.keyClick(widget._canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()

        primary_rect = next(
            rect
            for rect, layer_id, take_id, event_id in widget._canvas._event_rects
            if layer_id == LayerId("layer_snare_a")
            and take_id == TakeId("take_snare_a")
            and str(event_id) == "shared_evt"
        )
        _click_rect(widget, primary_rect)

        assert intents == [
            CommitRejectedEventReview(
                layer_id=LayerId("layer_snare_a"),
                take_id=TakeId("take_snare_a"),
                event_id=EventId("shared_evt"),
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_select_drag_dispatches_batch_event_selection_intent() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QTest.keyClick(widget._canvas, Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()

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


def test_fix_mode_minus_drag_removes_intersected_events() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _fix_mode_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QTest.keyClick(widget._canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()

        lane_rect = next(
            rect
            for rect, layer_id, take_id in widget._canvas._event_lane_rects
            if layer_id == LayerId("layer_kick") and take_id == TakeId("take_kick")
        )
        event_rect = next(
            rect
            for rect, layer_id, take_id, event_id in widget._canvas._event_rects
            if layer_id == LayerId("layer_kick")
            and take_id == TakeId("take_kick")
            and str(event_id) == "kick_evt"
        )
        _mouse_drag(
            widget._canvas,
            [
                QPoint(
                    int(max(float(lane_rect.left() + 2), float(event_rect.left() - 16))),
                    int(lane_rect.top() + 2),
                ),
                QPoint(
                    int(event_rect.right() + 16),
                    int(lane_rect.bottom() - 2),
                ),
            ],
        )

        assert intents == [
            CommitRejectedEventReview(
                layer_id=LayerId("layer_kick"),
                take_id=TakeId("take_kick"),
                event_id=EventId("kick_evt"),
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_plus_drag_promotes_intersected_missing_events() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _fix_mode_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QTest.keyClick(widget._canvas, Qt.Key.Key_C, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()

        lane_rect = next(
            rect
            for rect, layer_id, take_id in widget._canvas._event_lane_rects
            if layer_id == LayerId("layer_kick") and take_id == TakeId("take_kick")
        )
        missing_rect = next(
            rect
            for rect, layer_id, _take_id, source_event_id, _start, _end, matched in widget._canvas._fix_event_rects
            if layer_id == LayerId("layer_kick")
            and source_event_id == "onset_b"
            and matched is False
        )
        _mouse_drag(
            widget._canvas,
            [
                QPoint(
                    int(max(float(lane_rect.left() + 2), float(missing_rect.left() - 16))),
                    int(lane_rect.top() + 2),
                ),
                QPoint(
                    int(missing_rect.right() + 16),
                    int(lane_rect.bottom() - 2),
                ),
            ],
        )

        assert len(intents) == 1
        assert isinstance(intents[0], CommitMissedEventReview)
        assert intents[0].layer_id == LayerId("layer_kick")
        assert intents[0].take_id == TakeId("take_kick")
        assert intents[0].label == "Kick"
        assert intents[0].source_event_id == "onset_b"
        assert intents[0].payload_ref == "onset_b"
        assert intents[0].time_range.start < intents[0].time_range.end
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_plus_drag_promotes_intersected_existing_events_in_batch() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QTest.keyClick(widget._canvas, Qt.Key.Key_C, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()

        main_rect = next(
            rect
            for rect, _layer_id, _take_id, event_id in widget._canvas._event_rects
            if str(event_id) == "main_evt"
        )
        take_rect = next(
            rect
            for rect, _layer_id, _take_id, event_id in widget._canvas._event_rects
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
            CommitVerifiedEventsReview(
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
                ]
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_fix_mode_minus_drag_demotes_intersected_existing_events_in_batch() -> None:
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    presentation = _selection_test_presentation()
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QTest.keyClick(widget._canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()

        main_rect = next(
            rect
            for rect, _layer_id, _take_id, event_id in widget._canvas._event_rects
            if str(event_id) == "main_evt"
        )
        take_rect = next(
            rect
            for rect, _layer_id, _take_id, event_id in widget._canvas._event_rects
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
            CommitRejectedEventsReview(
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
                ]
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


def test_delete_key_dispatches_rejected_review_in_fix_remove_mode():
    app = QApplication.instance() or QApplication([])
    intents: list[object] = []
    base = _fix_mode_test_presentation()
    presentation = replace(
        base,
        selected_layer_id=LayerId("layer_kick"),
        selected_layer_ids=[LayerId("layer_kick")],
        selected_take_id=TakeId("take_kick"),
        selected_event_ids=[EventId("kick_evt")],
        selected_event_refs=[
            EventRef(
                layer_id=LayerId("layer_kick"),
                take_id=TakeId("take_kick"),
                event_id=EventId("kick_evt"),
            )
        ],
    )
    widget = TimelineWidget(
        presentation, on_intent=lambda intent: intents.append(intent) or presentation
    )
    try:
        _render_for_hit_testing(widget)
        widget._editor_bar._mode_buttons["fix"].click()
        QTest.keyClick(widget._canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.NoModifier)
        QApplication.processEvents()

        QTest.keyClick(widget._canvas, Qt.Key.Key_Delete)
        QApplication.processEvents()

        assert intents == [
            CommitRejectedEventReview(
                layer_id=LayerId("layer_kick"),
                take_id=TakeId("take_kick"),
                event_id=EventId("kick_evt"),
            )
        ]
    finally:
        widget.close()
        app.processEvents()

__all__ = [name for name in globals() if name.startswith("test_")]
