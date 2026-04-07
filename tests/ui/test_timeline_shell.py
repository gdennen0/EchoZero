from dataclasses import replace

from PyQt6.QtCore import QPoint, QPointF, QEvent
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    LayerStatusPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId, TimelineId
from echozero.application.timeline.intents import (
    ClearSelection,
    DuplicateSelectedEvents,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    Pause,
    Play,
    Seek,
    SelectAllEvents,
    SelectEvent,
    SelectLayer,
    Stop,
    ToggleLayerExpanded,
    ToggleMute,
    ToggleSolo,
)
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.test_harness import build_variant_presentations, estimate_full_window_height
from echozero.ui.qt.timeline.blocks.ruler import timeline_x_for_time
from echozero.ui.qt.timeline.widget import TimelineWidget, compute_scroll_bounds, estimate_timeline_span_seconds


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


class _SelectionInspectorHarness:
    def __init__(self, presentation: TimelinePresentation):
        self._presentation = presentation

    def presentation(self) -> TimelinePresentation:
        return self._presentation

    def dispatch(self, intent):
        if isinstance(intent, SelectLayer):
            self._presentation = replace(
                self._presentation,
                layers=[
                    replace(layer, is_selected=(layer.layer_id == intent.layer_id))
                    for layer in self._presentation.layers
                ],
                selected_layer_id=intent.layer_id,
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
                selected_take_id=None,
                selected_event_ids=[],
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


def _click_rect(widget: TimelineWidget, rect) -> None:
    center = rect.center().toPoint()
    QTest.mouseClick(widget._canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(center.x(), center.y()))
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
