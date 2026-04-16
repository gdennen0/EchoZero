from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from echozero.application.shared.enums import LayerKind
from echozero.application.timeline.intents import Play, Seek, ToggleLayerExpanded
from echozero.application.presentation.inspector_contract import (
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
)
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell
from echozero.ui.qt.timeline.demo_app import DemoTimelineApp
from echozero.testing.analysis_mocks import build_mock_analysis_service, write_test_model, write_test_wav


class _CountedRuntimeAudio:
    def __init__(self):
        self.build_calls = 0
        self.play_calls = 0
        self.is_playing_state = False

    def build_for_presentation(self, _presentation) -> None:
        self.build_calls += 1

    def apply_mix_state(self, _presentation) -> None:
        return None

    def play(self) -> None:
        self.play_calls += 1
        self.is_playing_state = True

    def pause(self) -> None:
        self.is_playing_state = False

    def stop(self) -> None:
        self.is_playing_state = False

    def seek(self, _position_seconds: float) -> None:
        return None

    def current_time_seconds(self) -> float:
        return 0.0

    def is_playing(self) -> bool:
        return self.is_playing_state

    def shutdown(self) -> None:
        return None


def _repo_local_temp_root() -> Path:
    root = Path("C:/Users/griff/.codex/memories/test_app_shell_runtime_flow") / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_app_shell_runtime_new_save_open_reopen_flow():
    temp_root = _repo_local_temp_root()
    working_root = temp_root / "working"
    save_path = temp_root / "runtime-flow.ez"

    runtime = build_app_shell(
        working_dir_root=working_root,
        initial_project_name="Runtime Flow",
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        assert runtime.project_storage.project.name == "Runtime Flow"
        assert runtime.project_path is None
        assert runtime.is_dirty is False
        assert runtime.presentation().layers == []

        runtime.dispatch(Seek(1.25))
        assert runtime.is_dirty is False

        first_audio = write_test_wav(temp_root / "fixtures" / "runtime-flow-1.wav")
        runtime.add_song_from_path("Runtime Flow Song", first_audio)
        assert runtime.is_dirty is True

        layer_id = runtime.presentation().layers[0].layer_id
        runtime.dispatch(ToggleLayerExpanded(layer_id))
        assert runtime.is_dirty is True

        runtime.new_project("Second Runtime Flow")
        assert runtime.project_storage.project.name == "Second Runtime Flow"
        assert runtime.project_path is None
        assert runtime.is_dirty is False
        assert runtime.presentation().layers == []

        second_audio = write_test_wav(temp_root / "fixtures" / "runtime-flow-2.wav")
        runtime.add_song_from_path("Second Runtime Song", second_audio)
        assert runtime.is_dirty is True

        layer_id = runtime.presentation().layers[0].layer_id
        runtime.dispatch(ToggleLayerExpanded(layer_id))
        assert runtime.is_dirty is True

        returned_path = runtime.save_project_as(save_path)
        assert returned_path == save_path
        assert save_path.exists()
        assert runtime.project_path == save_path
        assert runtime.is_dirty is False

        runtime.dispatch(ToggleLayerExpanded(layer_id))
        assert runtime.is_dirty is True
        saved_path = runtime.save_project()
        assert saved_path == save_path
        assert runtime.is_dirty is False

        runtime.open_project(save_path)
        assert runtime.project_path == save_path
        assert runtime.project_storage.working_dir.exists()
        assert runtime.project_storage.project.name == "Second Runtime Flow"
        assert runtime.session.project_id == runtime.project_storage.project.id
        assert runtime.session.active_song_id is not None
        assert runtime.session.active_song_version_id is not None
        assert runtime.session.active_timeline_id == runtime.presentation().timeline_id
        assert runtime.presentation().title == "Second Runtime Flow"
        assert runtime.presentation().layers[0].title == "Second Runtime Song"
        assert runtime.is_dirty is False

        runtime.open_project(save_path)
        assert runtime.project_path == save_path
        assert runtime.project_storage.working_dir.exists()
        assert Path(runtime.project_storage.working_dir).is_dir()
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_exposes_transfer_surface_actions():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        presentation = runtime.presentation()
        assert presentation.layers == []

        empty_contract = build_timeline_inspector_contract(
            presentation,
            hit_target=TimelineInspectorHitTarget(kind="timeline", time_seconds=1.0),
        )
        empty_action_ids = {
            action.action_id
            for section in empty_contract.context_sections
            for action in section.actions
        }
        assert "add_song_from_path" in empty_action_ids

        audio_path = write_test_wav(temp_root / "fixtures" / "transfer-actions.wav")
        presentation = runtime.add_song_from_path("Transfer Song", audio_path)
        first_layer = presentation.layers[0]
        layer_contract = build_timeline_inspector_contract(
            presentation,
            hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=first_layer.layer_id),
        )

        layer_action_ids = {
            action.action_id
            for section in layer_contract.context_sections
            for action in section.actions
        }

        assert "push_to_ma3" not in layer_action_ids
        assert "pull_from_ma3" not in layer_action_ids
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_add_song_from_path_updates_presentation():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")

        presentation = runtime.add_song_from_path("Imported Song", audio_path)

        assert runtime.session.active_song_id is not None
        assert runtime.session.active_song_version_id is not None
        assert presentation.layers[0].title == "Imported Song"
        assert presentation.layers[0].kind.name == "AUDIO"
        assert presentation.layers[0].source_audio_path
        assert presentation.end_time_label == "00:00.10"
        assert runtime.is_dirty is True
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_stems_persists_audio_layers_and_takes():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        runtime.add_song_from_path("Imported Song", audio_path)

        presentation = runtime.extract_stems("source_audio")
        titles = [layer.title for layer in presentation.layers]

        assert titles[:5] == ["Imported Song", "Drums", "Bass", "Vocals", "Other"]
        assert runtime.session.active_song_version_id is not None
        assert runtime.is_dirty is True

        stem_layers = presentation.layers[1:5]
        for layer in stem_layers:
            assert layer.kind.name == "AUDIO"
            assert layer.main_take_id is not None
            assert layer.source_audio_path
            assert layer.status.source_label.startswith("stem_separation")
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_stems_from_derived_audio_layer_is_deferred():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        presentation = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in presentation.layers if layer.title == "Drums")

        try:
            runtime.extract_stems(drums_layer.layer_id)
        except NotImplementedError as exc:
            assert "imported song layer" in str(exc)
        else:
            raise AssertionError("Expected extract_stems on a derived layer to remain deferred")
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_drum_events_persists_event_layers_from_drums_stem():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        presentation = runtime.extract_drum_events(drums_layer.layer_id)

        event_layers = [layer for layer in presentation.layers if layer.kind.name == "EVENT"]
        assert event_layers
        assert any(layer.events for layer in event_layers)
        assert any((layer.status.source_label or "").startswith("onset_detection") for layer in event_layers)
        assert runtime.is_dirty is True
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_drum_events_rejects_non_drum_audio_layers():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        bass_layer = next(layer for layer in after_stems.layers if layer.title == "Bass")

        try:
            runtime.extract_drum_events(bass_layer.layer_id)
        except NotImplementedError as exc:
            assert "drum-derived audio layers" in str(exc)
        else:
            raise AssertionError("Expected extract_drum_events to reject non-drum audio layers")
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_classify_drum_events_persists_classified_layers():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        model_path = write_test_model(temp_root / "fixtures" / "drum-model.pth")
        runtime.add_song_from_path("Imported Song", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        presentation = runtime.classify_drum_events(drums_layer.layer_id, model_path)

        event_layers = [layer for layer in presentation.layers if layer.kind.name == "EVENT"]
        assert event_layers
        assert any("drum" in layer.title.lower() and "classified" in layer.title.lower() for layer in event_layers)
        assert any(layer.events and layer.events[0].label == "Kick" for layer in event_layers)
        assert any((layer.status.source_label or "").startswith("drum_classification") for layer in event_layers)
        assert runtime.is_dirty is True
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_classify_drum_events_rejects_missing_model_path():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        missing_model = temp_root / "fixtures" / "missing-model.pth"
        try:
            runtime.classify_drum_events(drums_layer.layer_id, missing_model)
        except FileNotFoundError as exc:
            assert "existing model path" in str(exc)
        else:
            raise AssertionError("Expected classify_drum_events to reject a missing model path")
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_add_song_from_path_builds_runtime_audio():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    counted = _CountedRuntimeAudio()
    runtime.runtime_audio = counted

    try:
        assert counted.build_calls == 0

        runtime.add_song_from_path("Imported Song", write_test_wav(temp_root / "fixtures" / "import.wav"))

        assert counted.build_calls == 1
        assert runtime.presentation().layers[0].title == "Imported Song"
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_add_layer_after_song_rebuilds_runtime_audio():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    counted = _CountedRuntimeAudio()
    runtime.runtime_audio = counted

    try:
        runtime.add_song_from_path("Imported Song", write_test_wav(temp_root / "fixtures" / "import.wav"))
        counted.build_calls = 0

        runtime.add_layer(LayerKind.EVENT, "Event Layer")

        assert counted.build_calls == 1
        assert any(layer.title == "Event Layer" for layer in runtime.presentation().layers)
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_play_dispatch_rebuilds_runtime_audio():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    counted = _CountedRuntimeAudio()
    runtime.runtime_audio = counted

    try:
        runtime.add_song_from_path("Imported Song", write_test_wav(temp_root / "fixtures" / "import.wav"))
        counted.build_calls = 0
        runtime.dispatch(Play())
        assert counted.build_calls == 1
        assert counted.play_calls == 1
        assert runtime.presentation().is_playing is True
        assert runtime.session.transport_state.is_playing is True
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_canonical_build_does_not_depend_on_fixture_loader(monkeypatch):
    temp_root = _repo_local_temp_root()
    working_root = temp_root / "working"

    def fail_fixture_load():
        raise AssertionError("fixture loader should not be called for canonical app shell")

    monkeypatch.setattr(
        "echozero.ui.qt.app_shell.load_realistic_timeline_fixture",
        fail_fixture_load,
    )

    runtime = build_app_shell(
        use_demo_fixture=False,
        working_dir_root=working_root,
        initial_project_name="Native Baseline",
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        presentation = runtime.presentation()

        assert presentation.title == "Native Baseline"
        assert presentation.timeline_id == f"timeline_{runtime.project_storage.project.id}"
        assert runtime.session.project_id == runtime.project_storage.project.id
        assert runtime.session.active_song_id is None
        assert runtime.session.active_song_version_id is None
        assert runtime.session.active_timeline_id == presentation.timeline_id
        assert presentation.layers == []
        assert presentation.selected_layer_id is None
        assert presentation.selected_layer_ids == []
        assert presentation.playhead == 0.0
        assert presentation.current_time_label == "00:00.00"
        assert presentation.end_time_label == "00:00.00"
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_build_app_shell_demo_fixture_mode_still_returns_demo_runtime():
    runtime = build_app_shell(use_demo_fixture=True)

    assert isinstance(runtime, DemoTimelineApp)
    assert not isinstance(runtime, AppShellRuntime)
