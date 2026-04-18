from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from echozero.application.shared.enums import LayerKind
from echozero.application.timeline.intents import Play, Seek, SetActivePlaybackTarget, ToggleLayerExpanded
from echozero.application.timeline.app import TimelineApplication
from echozero.application.presentation.inspector_contract import (
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
)
from echozero.pipelines.registry import get_registry
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell
from echozero.ui.qt.timeline.waveform_cache import clear_waveform_cache, get_cached_waveform
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


def _assert_waveform_registered(waveform_key: str | None) -> None:
    assert waveform_key is not None
    cached = get_cached_waveform(waveform_key)
    assert cached is not None
    assert cached.peaks.size > 0


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


def test_app_shell_runtime_uses_canonical_timeline_application():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        initial_project_name="Canonical Runtime",
    )

    try:
        assert isinstance(runtime, AppShellRuntime)
        assert isinstance(runtime._app, TimelineApplication)
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


def test_app_shell_runtime_import_song_creates_default_pipeline_configs():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Configured Song",
            write_test_wav(temp_root / "fixtures" / "configured.wav"),
        )

        assert runtime.session.active_song_version_id is not None
        configs = runtime.project_storage.pipeline_configs.list_by_version(
            str(runtime.session.active_song_version_id)
        )

        assert configs
        assert {config.template_id for config in configs} == set(get_registry().ids())
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_add_song_version_copies_configs_and_switches_versions():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Versioned Song",
            write_test_wav(temp_root / "fixtures" / "version-1.wav", frames=4410),
        )
        assert runtime.session.active_song_id is not None
        assert runtime.session.active_song_version_id is not None

        song_id = str(runtime.session.active_song_id)
        version_1_id = str(runtime.session.active_song_version_id)
        version_1_templates = {
            config.template_id
            for config in runtime.project_storage.pipeline_configs.list_by_version(version_1_id)
        }

        presentation = runtime.add_song_version(
            song_id,
            write_test_wav(temp_root / "fixtures" / "version-2.wav", frames=8820),
            label="Festival Edit",
        )

        version_2_id = str(runtime.session.active_song_version_id)
        version_2_record = runtime.project_storage.song_versions.get(version_2_id)
        assert version_2_record is not None
        assert version_2_record.label == "Festival Edit"
        assert version_2_id != version_1_id
        assert {
            config.template_id
            for config in runtime.project_storage.pipeline_configs.list_by_version(version_2_id)
        } == version_1_templates
        assert runtime.project_storage.songs.get(song_id).active_version_id == version_2_id
        assert presentation.end_time_label == "00:00.20"

        switched = runtime.switch_song_version(version_1_id)

        assert runtime.project_storage.songs.get(song_id).active_version_id == version_1_id
        assert str(runtime.session.active_song_version_id) == version_1_id
        assert switched.end_time_label == "00:00.10"
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_select_song_switches_loaded_timeline():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Song One",
            write_test_wav(temp_root / "fixtures" / "song-1.wav", frames=4410),
        )
        assert runtime.session.active_song_id is not None
        song_1_id = str(runtime.session.active_song_id)
        version_1_id = str(runtime.session.active_song_version_id)

        runtime.add_song_from_path(
            "Song Two",
            write_test_wav(temp_root / "fixtures" / "song-2.wav", frames=8820),
        )
        assert runtime.presentation().layers[0].title == "Song Two"
        assert runtime.presentation().end_time_label == "00:00.20"

        presentation = runtime.select_song(song_1_id)

        assert str(runtime.session.active_song_id) == song_1_id
        assert str(runtime.session.active_song_version_id) == version_1_id
        assert presentation.layers[0].title == "Song One"
        assert presentation.end_time_label == "00:00.10"
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_open_project_preserves_playback_target_when_still_valid():
    temp_root = _repo_local_temp_root()
    working_root = temp_root / "working"
    save_path = temp_root / "preserve-target.ez"
    runtime = build_app_shell(
        working_dir_root=working_root,
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Playback Target Song",
            write_test_wav(temp_root / "fixtures" / "preserve-target.wav"),
        )
        second_pass = runtime.extract_stems("source_audio")
        second_pass = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in second_pass.layers if layer.title == "Drums")
        drums_take = drums_layer.takes[0]

        runtime.dispatch(SetActivePlaybackTarget(layer_id=drums_layer.layer_id, take_id=drums_take.take_id))
        before_reload = runtime.presentation()
        assert before_reload.selected_layer_id == second_pass.layers[0].layer_id
        assert before_reload.active_playback_layer_id == drums_layer.layer_id
        assert before_reload.active_playback_take_id == drums_take.take_id

        runtime.save_project_as(save_path)
        runtime.open_project(save_path)
        reloaded = runtime.presentation()

        assert reloaded.selected_layer_id == before_reload.selected_layer_id
        assert reloaded.active_playback_layer_id == before_reload.active_playback_layer_id
        assert reloaded.active_playback_take_id == before_reload.active_playback_take_id
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_refresh_repairs_missing_playback_target_to_baseline_layer():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Repair Target Song",
            write_test_wav(temp_root / "fixtures" / "repair-target-1.wav", frames=4410),
        )
        second_pass = runtime.extract_stems("source_audio")
        second_pass = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in second_pass.layers if layer.title == "Drums")
        drums_take = drums_layer.takes[0]

        runtime.dispatch(SetActivePlaybackTarget(layer_id=drums_layer.layer_id, take_id=drums_take.take_id))
        version_2 = runtime.add_song_version(
            str(runtime.session.active_song_id),
            write_test_wav(temp_root / "fixtures" / "repair-target-2.wav", frames=8820),
            label="Blank V2",
        )

        assert version_2.layers[0].title == "Repair Target Song"
        assert version_2.active_playback_layer_id == version_2.layers[0].layer_id
        assert version_2.active_playback_take_id is None
        assert version_2.selected_layer_id == version_2.layers[0].layer_id
        assert len(version_2.layers) == 1
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


def test_app_shell_runtime_extract_stems_registers_waveforms_for_main_and_take_audio():
    temp_root = _repo_local_temp_root()
    clear_waveform_cache()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        runtime.add_song_from_path("Imported Song", audio_path)

        first_pass = runtime.extract_stems("source_audio")
        for layer in first_pass.layers[1:5]:
            assert layer.source_audio_path and Path(layer.source_audio_path).exists()
            _assert_waveform_registered(layer.waveform_key)

        second_pass = runtime.extract_stems("source_audio")
        for layer in second_pass.layers[1:5]:
            assert layer.takes
            _assert_waveform_registered(layer.waveform_key)
            assert layer.takes[0].source_audio_path and Path(layer.takes[0].source_audio_path).exists()
            _assert_waveform_registered(layer.takes[0].waveform_key)
    finally:
        runtime.shutdown()
        clear_waveform_cache()
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
            assert "does not exist" in str(exc)
        else:
            raise AssertionError("Expected classify_drum_events to reject a missing model path")
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_classify_drum_events_accepts_foundry_manifest_path():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        model_path = write_test_model(temp_root / "exports" / "model.pth")
        manifest_path = temp_root / "exports" / "art_demo.manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "weightsPath": "model.pth",
                    "sharedContractFingerprint": "test-fingerprint",
                    "runtime": {"consumer": "PyTorchAudioClassify"},
                    "classes": ["kick", "snare", "hihat"],
                    "classificationMode": "multiclass",
                    "inferencePreprocessing": {
                        "sampleRate": 22050,
                        "maxLength": 22050,
                        "nFft": 2048,
                        "hopLength": 512,
                        "nMels": 128,
                        "fmax": 8000,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        runtime.add_song_from_path("Imported Song", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        presentation = runtime.classify_drum_events(drums_layer.layer_id, manifest_path)

        event_layers = [layer for layer in presentation.layers if layer.kind.name == "EVENT"]
        assert event_layers
        assert any(layer.events and layer.events[0].label == "Kick" for layer in event_layers)
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_classified_drums_persists_kick_and_snare_layers(monkeypatch):
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    fake_models_root = temp_root / "models"
    fake_models_root.mkdir(parents=True, exist_ok=True)
    kick_manifest = fake_models_root / "kick.manifest.json"
    snare_manifest = fake_models_root / "snare.manifest.json"
    kick_manifest.write_text("{}", encoding="utf-8")
    snare_manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "echozero.ui.qt.app_shell.resolve_installed_binary_drum_bundles",
        lambda: {
            "kick": type("Bundle", (), {"manifest_path": kick_manifest})(),
            "snare": type("Bundle", (), {"manifest_path": snare_manifest})(),
        },
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        presentation = runtime.extract_classified_drums(drums_layer.layer_id)

        event_layers = [layer for layer in presentation.layers if layer.kind.name == "EVENT"]
        titles = {layer.title for layer in event_layers}
        assert "Kick" in titles
        assert "Snare" in titles
        assert any(layer.events and layer.events[0].label == "Kick" for layer in event_layers)
        assert any(layer.events and layer.events[0].label == "Snare" for layer in event_layers)
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


def test_app_shell_runtime_add_song_syncs_backend_playback_state_metadata():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path("Imported Song", write_test_wav(temp_root / "fixtures" / "import.wav"))

        assert runtime.session.playback_state.backend_name == "sounddevice"
        assert runtime.session.playback_state.active_layer_id == runtime.presentation().active_playback_layer_id
        assert runtime.session.playback_state.active_sources
        assert runtime.session.playback_state.output_sample_rate > 0
        assert runtime.session.playback_state.output_channels > 0
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
