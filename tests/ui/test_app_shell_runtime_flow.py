from __future__ import annotations

import shutil
import uuid
import wave
from pathlib import Path

import echozero.pipelines.templates  # noqa: F401
from echozero.application.timeline.intents import Seek, ToggleLayerExpanded
from echozero.domain.types import AudioData, Event as DomainEvent, EventData, Layer as DomainLayer
from echozero.execution import ExecutionContext
from echozero.application.presentation.inspector_contract import (
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
)
from echozero.result import ok
from echozero.services.orchestrator import AnalysisService
from echozero.pipelines.registry import get_registry
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell
from echozero.ui.qt.timeline.demo_app import DemoTimelineApp


def _repo_local_temp_root() -> Path:
    root = Path("C:/Users/griff/.codex/memories/test_app_shell_runtime_flow") / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _write_test_wav(path: Path, *, frames: int = 4410, sample_rate: int = 44100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


class _MockLoadAudioExecutor:
    def execute(self, block_id: str, context: ExecutionContext):
        block = context.graph.blocks[block_id]
        return ok(
            AudioData(
                sample_rate=44100,
                duration=0.1,
                file_path=str(block.settings["file_path"]),
                channel_count=1,
            )
        )


class _MockSeparateAudioExecutor:
    def execute(self, block_id: str, context: ExecutionContext):
        audio = context.get_input(block_id, "audio_in", AudioData)
        assert audio is not None
        base = Path(audio.file_path).parent
        stems = {}
        for name in ("drums", "bass", "vocals", "other"):
            stem_path = _write_test_wav(base / f"{name}.wav")
            stems[f"{name}_out"] = AudioData(
                sample_rate=44100,
                duration=0.1,
                file_path=str(stem_path),
                channel_count=1,
            )
        return ok(stems)


class _MockDetectOnsetsExecutor:
    def execute(self, _block_id: str, _context: ExecutionContext):
        event = DomainEvent(
            id="evt_1",
            time=0.25,
            duration=0.05,
            classifications={"namespace:onset": "hit"},
            metadata={},
            origin="detect_onsets",
        )
        return ok(
            EventData(
                layers=(
                    DomainLayer(
                        id="layer_onsets",
                        name="Onsets",
                        events=(event,),
                    ),
                )
            )
        )


class _MockClassifyExecutor:
    def execute(self, _block_id: str, context: ExecutionContext):
        event_data = context.get_input(_block_id, "events_in", EventData)
        assert event_data is not None
        classified_layers: list[DomainLayer] = []
        for layer in event_data.layers:
            classified_events: list[DomainEvent] = []
            for event in layer.events:
                classified_events.append(
                    DomainEvent(
                        id=event.id,
                        time=event.time,
                        duration=event.duration,
                        classifications={"class": "kick", "confidence": "0.99"},
                        metadata={**event.metadata, "classified": True},
                        origin="classify",
                    )
                )
            classified_layers.append(DomainLayer(id=layer.id, name="Kick", events=tuple(classified_events)))
        return ok(EventData(layers=tuple(classified_layers)))


def _mock_stem_analysis_service() -> AnalysisService:
    return AnalysisService(
        get_registry(),
        {
            "LoadAudio": _MockLoadAudioExecutor(),
            "SeparateAudio": _MockSeparateAudioExecutor(),
            "DetectOnsets": _MockDetectOnsetsExecutor(),
            "PyTorchAudioClassify": _MockClassifyExecutor(),
        },
    )


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

        runtime.dispatch(Seek(1.25))
        assert runtime.is_dirty is False

        layer_id = runtime.presentation().layers[0].layer_id
        runtime.dispatch(ToggleLayerExpanded(layer_id))
        assert runtime.is_dirty is True

        runtime.new_project("Second Runtime Flow")
        assert runtime.project_storage.project.name == "Second Runtime Flow"
        assert runtime.project_path is None
        assert runtime.is_dirty is False

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
        assert runtime.session.active_song_id is None
        assert runtime.session.active_song_version_id is None
        assert runtime.session.active_timeline_id == runtime.presentation().timeline_id
        assert runtime.presentation().title == "Second Runtime Flow"
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
        first_layer = presentation.layers[0]
        contract = build_timeline_inspector_contract(
            presentation,
            hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=first_layer.layer_id),
        )

        action_ids = {
            action.action_id
            for section in contract.context_sections
            for action in section.actions
        }

        assert "push_to_ma3" in action_ids
        assert "pull_from_ma3" in action_ids
        assert "add_song_from_path" in action_ids
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_add_song_from_path_updates_presentation():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = _write_test_wav(temp_root / "fixtures" / "import.wav")

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
        analysis_service=_mock_stem_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = _write_test_wav(temp_root / "fixtures" / "import.wav")
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
        analysis_service=_mock_stem_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = _write_test_wav(temp_root / "fixtures" / "import.wav")
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
        analysis_service=_mock_stem_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = _write_test_wav(temp_root / "fixtures" / "import.wav")
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
        analysis_service=_mock_stem_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = _write_test_wav(temp_root / "fixtures" / "import.wav")
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
        analysis_service=_mock_stem_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = _write_test_wav(temp_root / "fixtures" / "import.wav")
        model_path = (temp_root / "fixtures" / "drum-model.pth")
        model_path.write_bytes(b"fake-model")
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
        analysis_service=_mock_stem_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = _write_test_wav(temp_root / "fixtures" / "import.wav")
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
        assert len(presentation.layers) == 1
        assert presentation.layers[0].events == []
        assert presentation.layers[0].takes == []
        assert all(not layer.events for layer in presentation.layers)
        assert all(not layer.takes for layer in presentation.layers)
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
