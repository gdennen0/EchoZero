"""
App-shell undo/redo tests: prove canonical runtime history and launcher actions.
Exists because undo must run through AppShellRuntime and the launcher surface, not demos.
Connects editable timeline mutations and take switching to real app-path verification.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import LayerId
from echozero.application.shared.ranges import TimeRange
from echozero.application.timeline.intents import CreateEvent, SelectTake
from echozero.testing.analysis_mocks import build_mock_analysis_service, write_test_wav
from echozero.testing.app_flow import AppFlowHarness
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell

_TEST_TEMP_ROOT = Path("C:/Users/griff/.codex/memories/test_app_shell_undo_redo")


def _repo_local_temp_root() -> Path:
    root = _TEST_TEMP_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _runtime_layer_by_id(runtime: AppShellRuntime, layer_id: LayerId):
    return next(layer for layer in runtime._app.timeline.layers if layer.id == layer_id)


def test_app_shell_runtime_undo_redo_restores_manual_event_edits_and_clears_redo_on_new_edit():
    temp_root = _repo_local_temp_root()
    save_path = temp_root / "undo-redo-manual.ez"
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "manual-events.wav")
        runtime.add_song_from_path("Undo Song", audio_path)

        added_layer = runtime.add_layer(LayerKind.EVENT, "Manual Events")
        manual_layer = next(layer for layer in added_layer.layers if layer.title == "Manual Events")

        created = runtime.dispatch(
            CreateEvent(
                layer_id=manual_layer.layer_id,
                take_id=None,
                time_range=TimeRange(1.0, 1.5),
            )
        )
        created_layer = next(layer for layer in created.layers if layer.title == "Manual Events")
        assert len(created_layer.events) == 1
        assert _runtime_layer_by_id(runtime, manual_layer.layer_id).takes[0].events[0].cue_number == 1
        assert runtime.can_undo() is True
        assert runtime.can_redo() is False

        undone = runtime.undo()
        undone_layer = next(layer for layer in undone.layers if layer.title == "Manual Events")
        assert undone_layer.events == []
        assert runtime.can_redo() is True

        redone = runtime.redo()
        redone_layer = next(layer for layer in redone.layers if layer.title == "Manual Events")
        assert len(redone_layer.events) == 1

        runtime.save_project_as(save_path)
        runtime.open_project(save_path)
        reloaded_layer = next(
            layer for layer in runtime.presentation().layers if layer.title == "Manual Events"
        )
        assert len(reloaded_layer.events) == 1
        assert _runtime_layer_by_id(runtime, manual_layer.layer_id).takes[0].events[0].cue_number == 1

        runtime.undo()
        runtime.dispatch(
            CreateEvent(
                layer_id=manual_layer.layer_id,
                take_id=None,
                time_range=TimeRange(2.0, 2.25),
            )
        )
        assert runtime.can_redo() is False
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_undo_redo_restores_take_switch_selection():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "take-switch.wav")
        runtime.add_song_from_path("Undo Song", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")
        runtime.extract_drum_events(drums_layer.layer_id)
        second_pass = runtime.extract_drum_events(drums_layer.layer_id)

        onsets_layer = next(layer for layer in second_pass.layers if layer.title == "Onsets")
        alt_take = onsets_layer.takes[0]
        assert runtime.presentation().selected_take_id == alt_take.take_id

        switched = runtime.dispatch(
            SelectTake(layer_id=onsets_layer.layer_id, take_id=onsets_layer.main_take_id)
        )
        assert switched.selected_take_id == onsets_layer.main_take_id
        assert runtime.can_undo() is True

        restored = runtime.undo()
        assert restored.selected_take_id == alt_take.take_id

        redone = runtime.redo()
        assert redone.selected_take_id == onsets_layer.main_take_id
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_launcher_undo_redo_actions_use_canonical_runtime_history():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(working_dir_root=temp_root / "working")

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "launcher-undo.wav")
        harness.runtime.add_song_from_path("Harness Undo", audio_path)
        presentation = harness.runtime.add_layer(LayerKind.EVENT, "Launcher Events")
        harness.widget.set_presentation(presentation)
        harness._app.processEvents()

        layer = next(
            layer for layer in harness.presentation().layers if layer.title == "Launcher Events"
        )
        harness.dispatch(
            CreateEvent(
                layer_id=layer.layer_id,
                take_id=None,
                time_range=TimeRange(0.5, 0.75),
            )
        )
        assert (
            len(
                next(
                    layer
                    for layer in harness.presentation().layers
                    if layer.title == "Launcher Events"
                ).events
            )
            == 1
        )

        harness.trigger_action("undo")
        assert (
            len(
                next(
                    layer
                    for layer in harness.presentation().layers
                    if layer.title == "Launcher Events"
                ).events
            )
            == 0
        )

        harness.trigger_action("redo")
        assert (
            len(
                next(
                    layer
                    for layer in harness.presentation().layers
                    if layer.title == "Launcher Events"
                ).events
            )
            == 1
        )
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_launcher_undo_preserves_widget_viewport_for_same_timeline():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(working_dir_root=temp_root / "working")

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "launcher-viewport-undo.wav")
        harness.runtime.add_song_from_path("Viewport Undo", audio_path)
        presentation = harness.runtime.add_layer(LayerKind.EVENT, "Viewport Events")
        harness.widget.set_presentation(presentation)
        harness._app.processEvents()

        layer = next(
            layer for layer in harness.presentation().layers if layer.title == "Viewport Events"
        )
        harness.widget._dispatch(
            CreateEvent(
                layer_id=layer.layer_id,
                take_id=None,
                time_range=TimeRange(30.0, 30.5),
            )
        )
        harness.widget._dispatch(
            CreateEvent(
                layer_id=layer.layer_id,
                take_id=None,
                time_range=TimeRange(42.0, 42.5),
            )
        )
        harness._app.processEvents()

        viewport_layer = next(
            layer for layer in harness.widget.presentation.layers if layer.title == "Viewport Events"
        )
        event_id = viewport_layer.events[0].event_id

        harness.widget._hscroll.setValue(900)
        harness._app.processEvents()
        harness.widget._zoom_from_input(120, 720.0)
        harness._app.processEvents()

        expected_scroll = harness.widget.presentation.scroll_x
        expected_pps = harness.widget.presentation.pixels_per_second

        harness.widget._delete_events([event_id])
        harness._app.processEvents()

        assert abs(harness.widget.presentation.scroll_x - expected_scroll) < 0.5
        assert harness.widget.presentation.pixels_per_second == expected_pps

        harness.trigger_action("undo")

        restored_layer = next(
            layer for layer in harness.widget.presentation.layers if layer.title == "Viewport Events"
        )
        assert len(restored_layer.events) == 2
        assert abs(harness.widget.presentation.scroll_x - expected_scroll) < 0.5
        assert harness.widget.presentation.pixels_per_second == expected_pps

        harness.trigger_action("redo")

        redone_layer = next(
            layer for layer in harness.widget.presentation.layers if layer.title == "Viewport Events"
        )
        assert len(redone_layer.events) == 1
        assert abs(harness.widget.presentation.scroll_x - expected_scroll) < 0.5
        assert harness.widget.presentation.pixels_per_second == expected_pps
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)
