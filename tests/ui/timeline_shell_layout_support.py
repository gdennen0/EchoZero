"""Layout-oriented timeline-shell support cases.
Exists to keep fixture and presentation coverage separate from action and interaction tests.
Connects the compatibility wrapper to the bounded layout support slice.
"""

from PyQt6.QtGui import QAction
from tests.ui.timeline_shell_shared_support import *  # noqa: F401,F403

def test_demo_variants_include_take_lanes_open_and_zoom_states():
    variants = build_variant_presentations()
    assert "take_lanes_open" in variants
    assert "zoomed_in" in variants
    assert "zoomed_out" in variants


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

    assert {"Song", "Drums", "Bass", "Vocals", "Other", "Kick", "Snare", "HiHat", "Clap"} <= titles


def test_take_lanes_exist_without_inline_action_requirements():
    demo = build_demo_app()
    presentation = demo.presentation()
    drums = next(layer for layer in presentation.layers if layer.title == "Drums")
    kick = next(layer for layer in presentation.layers if layer.title == "Kick")

    assert len(drums.takes) >= 1
    assert drums.takes[0].kind.name == "AUDIO"
    assert len(kick.takes) >= 1
    assert kick.takes[0].kind.name == "EVENT"


def test_toggle_layer_expansion_round_trips():
    demo = build_demo_app()
    song = next(layer for layer in demo.presentation().layers if layer.title == "Song")

    expanded = demo.dispatch(ToggleLayerExpanded(song.layer_id))
    expanded_song = next(layer for layer in expanded.layers if layer.title == "Song")
    assert expanded_song.is_expanded is True

    collapsed = demo.dispatch(ToggleLayerExpanded(song.layer_id))
    collapsed_song = next(layer for layer in collapsed.layers if layer.title == "Song")
    assert collapsed_song.is_expanded is False


def test_fixture_has_selected_layer_for_daw_style_audio_routing():
    demo = build_demo_app()
    presentation = demo.presentation()
    assert presentation.selected_layer_id is not None
    assert any(layer.is_selected for layer in presentation.layers)


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
    assert any(
        "sync" in layer.title.lower() or layer.status.sync_label for layer in presentation.layers
    )


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


def test_transport_bar_is_below_timeline_viewport():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(build_demo_app().presentation())
    try:
        widget.resize(1400, 900)
        widget.show()
        app.processEvents()

        assert widget._transport.geometry().top() > widget._scroll.geometry().bottom()
        assert widget._transport.geometry().top() > widget._hscroll.geometry().bottom()
    finally:
        widget.close()
        app.processEvents()


def test_timeline_widget_surfaces_pipeline_status_banner_from_presentation():
    app = QApplication.instance() or QApplication([])
    banner = OperationProgressBannerPresentation(
        operation_id="pipeline_run_1",
        title="Extract Stems",
        status="running",
        message="Executing pipeline",
        fraction_complete=0.2,
        is_error=False,
    )
    widget = TimelineWidget(
        replace(_selection_test_presentation(), operation_progress_banner=banner)
    )
    try:
        widget.show()
        app.processEvents()

        assert widget._pipeline_status.isHidden() is False
        assert "Pipeline Running" in widget._pipeline_status_label.text()
        assert "Extract Stems" in widget._pipeline_status_label.text()
        assert "20%" in widget._pipeline_status_label.text()
        assert (
            widget._pipeline_status_label.textInteractionFlags()
            & Qt.TextInteractionFlag.TextSelectableByMouse
        )

        widget.set_presentation(replace(widget.presentation, operation_progress_banner=None))
        app.processEvents()
        assert widget._pipeline_status.isHidden() is True
    finally:
        widget.close()


def test_timeline_widget_pipeline_status_banner_can_be_closed_manually():
    app = QApplication.instance() or QApplication([])
    banner = OperationProgressBannerPresentation(
        operation_id="pipeline_run_1",
        title="Extract Stems",
        status="running",
        message="Executing pipeline",
        fraction_complete=0.2,
        is_error=False,
    )
    widget = TimelineWidget(
        replace(_selection_test_presentation(), operation_progress_banner=banner)
    )
    try:
        widget.show()
        app.processEvents()
        assert widget._pipeline_status.isHidden() is False

        QTest.mouseClick(
            widget._pipeline_status_close_button,
            Qt.MouseButton.LeftButton,
        )
        app.processEvents()

        assert widget._pipeline_status.isHidden() is True
    finally:
        widget.close()
        app.processEvents()


def test_timeline_widget_auto_dismisses_ma3_success_status_banner(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_runtime_mixin._MA3_PUSH_SUCCESS_BANNER_AUTO_DISMISS_MS",
        20,
    )
    base = _selection_test_presentation()
    widget = TimelineWidget(
        replace(
            base,
            operation_progress_banner=None,
            manual_push_flow=replace(
                base.manual_push_flow,
                operation_status="success",
                operation_message="Sent 2 event(s) to tc1_tg2_tr3",
            ),
        )
    )
    try:
        widget.show()
        app.processEvents()
        assert widget._pipeline_status.isHidden() is False

        QTest.qWait(40)
        app.processEvents()
        assert widget._pipeline_status.isHidden() is True
    finally:
        widget.close()
        app.processEvents()


def test_timeline_widget_file_menu_shows_open_recent_project_submenu():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(build_demo_app().presentation())
    try:
        actions = {
            "new_project": QAction("&New Project", widget),
            "open_project": QAction("&Open Project", widget),
            "open_recent_project::0": QAction("1. alpha.ez (C:/projects/alpha.ez)", widget),
            "open_recent_project::1": QAction("2. bravo.ez (C:/projects/bravo.ez)", widget),
            "save_project": QAction("&Save Project", widget),
            "save_project_as": QAction("Save Project &As...", widget),
        }

        widget.configure_launcher_actions(actions)
        menu_actions = widget._launcher_menu_bar.actions()
        assert menu_actions
        file_menu = menu_actions[0].menu()
        assert file_menu is not None

        recent_menu_entries = [
            action
            for action in file_menu.actions()
            if action.menu() is not None and "Recent" in action.text()
        ]
        assert len(recent_menu_entries) == 1
        recent_menu = recent_menu_entries[0].menu()
        assert recent_menu is not None
        assert [action.text() for action in recent_menu.actions()] == [
            "1. alpha.ez (C:/projects/alpha.ez)",
            "2. bravo.ez (C:/projects/bravo.ez)",
        ]
    finally:
        widget.close()
        app.processEvents()



def test_pipeline_context_actions_include_phase1_ids():
    presentation = _audio_pipeline_presentation()
    section_presentation = replace(
        presentation,
        layers=[
            *presentation.layers,
            LayerPresentation(
                layer_id=LayerId("layer_sections"),
                title="Sections",
                main_take_id=TakeId("take_sections"),
                kind=LayerKind.SECTION,
                status=LayerStatusPresentation(),
            ),
        ],
    )

    empty_contract = build_timeline_inspector_contract(
        replace(
            presentation,
            selected_layer_id=None,
            selected_layer_ids=[],
        )
    )
    song_contract = build_timeline_inspector_contract(
        presentation,
        hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=LayerId("layer_song")),
    )
    drums_contract = build_timeline_inspector_contract(
        presentation,
        hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=LayerId("layer_drums")),
    )
    section_contract = build_timeline_inspector_contract(
        section_presentation,
        hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=LayerId("layer_sections")),
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
    section_action_ids = {
        action.action_id
        for section in section_contract.context_sections
        for action in section.actions
    }

    assert "song.add" in empty_action_ids
    assert "add_event_layer" in empty_action_ids
    assert "add_section_layer" in empty_action_ids
    assert "add_automation_layer" not in empty_action_ids
    assert "add_reference_layer" not in empty_action_ids
    assert "song.add" not in song_action_ids
    assert "song.add" not in drums_action_ids
    assert "timeline.extract_stems" in song_action_ids
    assert "timeline.extract_song_drum_events" in song_action_ids
    assert "timeline.extract_song_sections" in song_action_ids
    assert "timeline.extract_drum_events" not in song_action_ids
    assert "timeline.classify_drum_events" not in song_action_ids
    assert "timeline.extract_stems" in drums_action_ids
    assert "timeline.extract_classified_drums" in drums_action_ids
    assert "timeline.extract_drum_events" in drums_action_ids
    assert "timeline.classify_drum_events" not in drums_action_ids
    assert "timeline.extract_song_sections" in section_action_ids



__all__ = [name for name in globals() if name.startswith("test_")]
