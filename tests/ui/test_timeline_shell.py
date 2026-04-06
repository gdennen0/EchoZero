from dataclasses import replace

from PyQt6.QtCore import QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt
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
from echozero.application.timeline.intents import Pause, Play, Seek, SelectEvent, SelectLayer, ToggleLayerExpanded
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.test_harness import build_variant_presentations, estimate_full_window_height
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
        assert widget._ruler.parent() is widget
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


def _render_for_hit_testing(widget: TimelineWidget) -> None:
    widget.resize(1200, 320)
    widget.show()
    widget.repaint()
    QApplication.processEvents()
    widget._canvas.repaint()
    QApplication.processEvents()


def _click_event_rect(widget: TimelineWidget, event_id: str) -> None:
    for rect, _, _, candidate_event_id in widget._canvas._event_rects:
        if str(candidate_event_id) == event_id:
            center = rect.center().toPoint()
            QTest.mouseClick(widget._canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(center.x(), center.y()))
            QApplication.processEvents()
            return
    raise AssertionError(f"Missing event rect for {event_id}")


def _click_rect(widget: TimelineWidget, rect) -> None:
    center = rect.center().toPoint()
    QTest.mouseClick(widget._canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(center.x(), center.y()))
    QApplication.processEvents()


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
        )
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
        )
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
