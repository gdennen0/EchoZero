from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from echozero.application.presentation.inspector_contract import (
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.intents import (
    Play,
    Seek,
    SetActivePlaybackTarget,
    ToggleLayerExpanded,
)
from echozero.application.timeline.object_actions import (
    ApplyCopySource,
    ChangeSessionScope,
    PreviewCopySource,
    ReplaceSessionValues,
    RunSession,
    SaveAndRunSession,
    SaveSession,
    SetSessionFieldValue,
)
from echozero.pipelines.registry import get_registry
from echozero.testing.analysis_mocks import (
    build_mock_analysis_service,
    write_test_model,
    write_test_wav,
)
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell
from echozero.ui.qt.timeline.waveform_cache import clear_waveform_cache, get_cached_waveform


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
        assert "song.add" in empty_action_ids

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


def test_app_shell_runtime_describes_stem_action_settings():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "describe-stems.wav")
        runtime.add_song_from_path("Describe Stems", audio_path)

        plan = runtime.describe_object_action(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )

        assert plan.pipeline_template_id == "stem_separation"
        assert dict(plan.locked_bindings)["audio_file"]
        assert [field.key for field in plan.editable_fields] == ["model", "device"]
        assert [field.key for field in plan.advanced_fields] == ["shifts", "two_stems"]
        assert plan.has_prior_outputs is False
        assert plan.run_label == "Run"
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_marks_existing_outputs_on_rerun_settings_plan():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "rerun-settings.wav")
        runtime.add_song_from_path("Rerun Settings", audio_path)
        runtime.extract_stems("source_audio")

        plan = runtime.describe_object_action(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )

        assert plan.has_prior_outputs is True
        assert plan.run_label == "Run Again"
        assert "Existing outputs detected" in plan.rerun_hint
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_save_object_action_settings_persists_knobs():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "persist-settings.wav")
        runtime.add_song_from_path("Persist Settings", audio_path)

        refreshed = runtime.save_object_action_settings(
            "timeline.extract_stems",
            {"layer_id": "source_audio", "model": "mdx_extra", "shifts": 3},
            object_id="source_audio",
            object_type="layer",
        )

        assert any(
            field.key == "model" and field.value == "mdx_extra"
            for field in refreshed.editable_fields
        )
        assert any(
            field.key == "shifts" and field.value == 3 for field in refreshed.advanced_fields
        )

        song_version_id = str(runtime.session.active_song_version_id)
        configs = runtime.project_storage.pipeline_configs.list_by_version(song_version_id)
        stem_config = next(config for config in configs if config.template_id == "stem_separation")
        assert stem_config.knob_values["model"] == "mdx_extra"
        assert stem_config.knob_values["shifts"] == 3
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_song_default_scope_edits_song_defaults_only():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "song-default.wav")
        runtime.add_song_from_path("Song Default Scope", audio_path)

        runtime.save_object_action_settings(
            "timeline.extract_stems",
            {"layer_id": "source_audio", "model": "mdx_extra"},
            object_id="source_audio",
            object_type="layer",
            scope="song_default",
        )
        version_plan = runtime.describe_object_action(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
            scope="version",
        )
        default_plan = runtime.describe_object_action(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
            scope="song_default",
        )

        assert any(
            field.key == "model" and field.value == "htdemucs"
            for field in version_plan.editable_fields
        )
        assert any(
            field.key == "model" and field.value == "mdx_extra"
            for field in default_plan.editable_fields
        )
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_copies_settings_from_song_default_to_version():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "copy-settings.wav")
        runtime.add_song_from_path("Copy Settings", audio_path)

        runtime.save_object_action_settings(
            "timeline.extract_stems",
            {"layer_id": "source_audio", "model": "mdx_extra_q", "device": "cpu"},
            object_id="source_audio",
            object_type="layer",
            scope="song_default",
        )

        preview = runtime.preview_object_action_settings_copy(
            "timeline.extract_stems",
            source_scope="song_default",
            target_scope="version",
            keys=["model", "device"],
        )
        assert {item["key"] for item in preview["changes"]} == {"model", "device"}

        runtime.apply_object_action_settings_copy(
            "timeline.extract_stems",
            source_scope="song_default",
            target_scope="version",
            keys=["model", "device"],
        )

        version_plan = runtime.describe_object_action(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
            scope="version",
        )
        assert any(
            field.key == "model" and field.value == "mdx_extra_q"
            for field in version_plan.editable_fields
        )
        assert any(
            field.key == "device" and field.value == "cpu"
            for field in version_plan.editable_fields
        )
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_object_action_session_discovers_scope_and_copy_sources():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "session-discovery.wav")
        runtime.add_song_from_path("Session Discovery", audio_path)

        session = runtime.open_object_action_session(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )

        assert session.scope == "version"
        assert session.available_scopes == ("version", "song_default")
        assert [source.source_id for source in session.copy_sources] == ["song_default"]

        switched = runtime.dispatch_object_action_command(
            session.session_id,
            ChangeSessionScope("song_default"),
        )
        assert switched.scope == "song_default"
        assert [source.source_id for source in switched.copy_sources] == ["this_version"]
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_object_action_session_commands_save_preview_copy_and_run():
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    captured: list[dict[str, object] | None] = []
    original_execute = analysis_service.execute

    def _capture_execute(session, config_id, runtime_bindings=None, on_progress=None):
        captured.append(runtime_bindings)
        return original_execute(
            session, config_id, runtime_bindings=runtime_bindings, on_progress=on_progress
        )

    analysis_service.execute = _capture_execute
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "session-commands.wav")
        runtime.add_song_from_path("Session Commands", audio_path)
        runtime.save_object_action_settings(
            "timeline.extract_stems",
            {"layer_id": "source_audio", "model": "mdx_extra_q", "device": "cpu"},
            object_id="source_audio",
            object_type="layer",
            scope="song_default",
        )

        session = runtime.open_object_action_session(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )
        session = runtime.dispatch_object_action_command(
            session.session_id,
            ReplaceSessionValues({"model": "mdx_extra", "shifts": 3}),
        )
        assert session.values["model"] == "mdx_extra"
        session = runtime.dispatch_object_action_command(session.session_id, SaveSession())
        assert any(
            field.key == "model" and field.value == "mdx_extra"
            for field in session.plan.editable_fields
        )

        session = runtime.dispatch_object_action_command(
            session.session_id, PreviewCopySource("song_default")
        )
        assert session.copy_preview is not None
        assert {change[0] for change in session.copy_preview.changes} >= {"model", "device"}

        session = runtime.dispatch_object_action_command(
            session.session_id, ApplyCopySource("song_default")
        )
        assert any(
            field.key == "model" and field.value == "mdx_extra_q"
            for field in session.plan.editable_fields
        )
        assert any(
            field.key == "device" and field.value == "cpu"
            for field in session.plan.editable_fields
        )

        runtime.dispatch_object_action_command(session.session_id, RunSession())
        assert captured
        assert captured[-1] is not None
        assert "audio_file" in captured[-1]
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_object_action_session_classify_uses_descriptor_bound_runtime_bindings():
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    captured: list[dict[str, object] | None] = []
    original_execute = analysis_service.execute

    def _capture_execute(session, config_id, runtime_bindings=None, on_progress=None):
        captured.append(runtime_bindings)
        return original_execute(
            session, config_id, runtime_bindings=runtime_bindings, on_progress=on_progress
        )

    analysis_service.execute = _capture_execute
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "session-classify.wav")
        model_path = write_test_model(temp_root / "fixtures" / "session-classify-model.pth")
        runtime.add_song_from_path("Session Classify", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        session = runtime.open_object_action_session(
            "timeline.classify_drum_events",
            {"layer_id": drums_layer.layer_id, "model_path": str(model_path)},
            object_id=drums_layer.layer_id,
            object_type="layer",
        )
        runtime.dispatch_object_action_command(session.session_id, RunSession())
        plan = runtime.describe_object_action(
            "timeline.classify_drum_events",
            {"layer_id": drums_layer.layer_id},
            object_id=drums_layer.layer_id,
            object_type="layer",
        )

        assert captured
        assert captured[-1] is not None
        assert captured[-1]["audio_file"] == drums_layer.source_audio_path
        assert "classify_model_path" not in captured[-1]
        assert any(
            field.key == "classify_model_path" and field.value == str(model_path)
            for field in plan.editable_fields
        )
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_object_action_session_save_and_copy_do_not_refresh_presentation():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "session-save-refresh.wav")
        runtime.add_song_from_path("Session Save Refresh", audio_path)
        runtime.save_object_action_settings(
            "timeline.extract_stems",
            {"layer_id": "source_audio", "model": "mdx_extra_q", "device": "cpu"},
            object_id="source_audio",
            object_type="layer",
            scope="song_default",
        )

        refresh_calls: list[tuple[object | None, object | None]] = []
        original_refresh = runtime._refresh_from_storage

        def _spy_refresh(*, active_song_id=None, active_song_version_id=None):
            refresh_calls.append((active_song_id, active_song_version_id))
            return original_refresh(
                active_song_id=active_song_id,
                active_song_version_id=active_song_version_id,
            )

        runtime._refresh_from_storage = _spy_refresh
        session = runtime.open_object_action_session(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )

        runtime.dispatch_object_action_command(
            session.session_id,
            ReplaceSessionValues({"model": "mdx_extra"}),
        )
        runtime.dispatch_object_action_command(session.session_id, SaveSession())
        assert refresh_calls == []

        runtime.dispatch_object_action_command(
            session.session_id, PreviewCopySource("song_default")
        )
        runtime.dispatch_object_action_command(session.session_id, ApplyCopySource("song_default"))
        assert refresh_calls == []

        runtime.dispatch_object_action_command(session.session_id, RunSession())
        assert len(refresh_calls) == 1
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_object_action_session_keeps_scope_local_drafts_and_preview_targets_current_draft():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "session-scope-drafts.wav")
        runtime.add_song_from_path("Session Scope Drafts", audio_path)
        runtime.save_object_action_settings(
            "timeline.extract_stems",
            {"layer_id": "source_audio", "model": "mdx_extra_q"},
            object_id="source_audio",
            object_type="layer",
            scope="song_default",
        )

        session = runtime.open_object_action_session(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )
        session = runtime.dispatch_object_action_command(
            session.session_id,
            SetSessionFieldValue("model", "mdx_extra"),
        )
        assert session.values["model"] == "mdx_extra"
        assert session.has_unsaved_changes is True
        assert session.can_save_and_run is True

        switched = runtime.dispatch_object_action_command(
            session.session_id,
            ChangeSessionScope("song_default"),
        )
        assert switched.scope == "song_default"
        assert switched.values["model"] == "mdx_extra_q"
        assert switched.can_save_and_run is False

        switched = runtime.dispatch_object_action_command(
            switched.session_id,
            ChangeSessionScope("version"),
        )
        assert switched.scope == "version"
        assert switched.values["model"] == "mdx_extra"

        previewed = runtime.dispatch_object_action_command(
            switched.session_id,
            PreviewCopySource("song_default"),
        )
        assert previewed.copy_preview is not None
        assert ("model", "mdx_extra", "mdx_extra_q") in previewed.copy_preview.changes
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_object_action_session_rejects_save_and_run_from_song_default_scope():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "session-song-default-run.wav")
        runtime.add_song_from_path("Session Song Default Run", audio_path)

        session = runtime.open_object_action_session(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
            scope="song_default",
        )

        with pytest.raises(ValueError, match="Switch to This Version to run"):
            runtime.dispatch_object_action_command(session.session_id, SaveAndRunSession())
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

        runtime.dispatch(
            SetActivePlaybackTarget(layer_id=drums_layer.layer_id, take_id=drums_take.take_id)
        )
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

        runtime.dispatch(
            SetActivePlaybackTarget(layer_id=drums_layer.layer_id, take_id=drums_take.take_id)
        )
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


def test_app_shell_runtime_extract_stems_passes_explicit_source_audio_binding():
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    captured: list[dict[str, object] | None] = []
    original_execute = analysis_service.execute

    def _capture_execute(session, config_id, runtime_bindings=None, on_progress=None):
        captured.append(runtime_bindings)
        return original_execute(
            session,
            config_id,
            runtime_bindings=runtime_bindings,
            on_progress=on_progress,
        )

    analysis_service.execute = _capture_execute
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        expected_audio_path = runtime.presentation().layers[0].source_audio_path

        runtime.extract_stems("source_audio")

        assert expected_audio_path is not None
        assert captured == [
            {
                "audio_file": expected_audio_path,
            }
        ]
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
            assert (
                layer.takes[0].source_audio_path
                and Path(layer.takes[0].source_audio_path).exists()
            )
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
        assert any(
            (layer.status.source_label or "").startswith("onset_detection")
            for layer in event_layers
        )
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
        assert any(
            "drum" in layer.title.lower() and "classified" in layer.title.lower()
            for layer in event_layers
        )
        assert any(layer.events and layer.events[0].label == "Kick" for layer in event_layers)
        assert any(
            (layer.status.source_label or "").startswith("drum_classification")
            for layer in event_layers
        )
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

        runtime.add_song_from_path(
            "Imported Song", write_test_wav(temp_root / "fixtures" / "import.wav")
        )

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
        runtime.add_song_from_path(
            "Imported Song", write_test_wav(temp_root / "fixtures" / "import.wav")
        )
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
        runtime.add_song_from_path(
            "Imported Song", write_test_wav(temp_root / "fixtures" / "import.wav")
        )
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
        runtime.add_song_from_path(
            "Imported Song", write_test_wav(temp_root / "fixtures" / "import.wav")
        )

        assert runtime.session.playback_state.backend_name == "sounddevice"
        assert (
            runtime.session.playback_state.active_layer_id
            == runtime.presentation().active_playback_layer_id
        )
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
