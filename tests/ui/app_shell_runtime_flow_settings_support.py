"""Object-action settings runtime-flow support cases.
Exists to isolate settings-session and copy behavior from project and pipeline support coverage.
Connects the compatibility wrapper to the bounded settings support slice.
"""

from tests.ui.app_shell_runtime_flow_shared_support import *  # noqa: F401,F403

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
        model_field = next(field for field in plan.editable_fields if field.key == "model")
        assert model_field.default_value == "latest_model"
        assert "latest_model" in {option.value for option in model_field.options}
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
            field.key == "model" and field.value == "latest_model"
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


def test_app_shell_runtime_object_action_session_without_layer_saves_but_disables_run():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "session-no-layer.wav")
        runtime.add_song_from_path("Session No Layer", audio_path)

        session = runtime.open_object_action_session(
            "timeline.extract_stems",
            {},
            object_type="layer",
        )
        assert session.plan.object_id == ""
        assert session.can_save is True
        assert session.can_save_and_run is False
        assert session.run_disabled_reason == "Select a target layer before running this stage."

        session = runtime.dispatch_object_action_command(
            session.session_id,
            ReplaceSessionValues({"model": "mdx_extra_q"}),
        )
        session = runtime.dispatch_object_action_command(session.session_id, SaveSession())
        refreshed = runtime.describe_object_action(
            "timeline.extract_stems",
            {},
            object_type="layer",
            scope="version",
        )
        assert any(
            field.key == "model" and field.value == "mdx_extra_q"
            for field in refreshed.editable_fields
        )

        with pytest.raises(ValueError, match="Select a target layer before running this stage."):
            runtime.dispatch_object_action_command(session.session_id, RunSession())
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

        session = runtime.dispatch_object_action_command(session.session_id, RunSession())
        assert session.plan.run_id is not None
        runtime.wait_for_pipeline_run(session.plan.run_id, timeout=5.0)
        assert captured
        assert captured[-1] is not None
        assert "audio_file" in captured[-1]
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_object_action_session_reset_defaults_restores_template_values():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "session-reset-defaults.wav")
        runtime.add_song_from_path("Session Reset Defaults", audio_path)

        session = runtime.open_object_action_session(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )
        defaults = {
            field.key: field.default_value
            for field in (*session.plan.editable_fields, *session.plan.advanced_fields)
        }

        session = runtime.dispatch_object_action_command(
            session.session_id,
            ReplaceSessionValues({"model": "mdx_extra", "shifts": 3}),
        )
        assert session.values["model"] == "mdx_extra"
        assert session.values["shifts"] == 3
        assert session.has_unsaved_changes is True

        session = runtime.dispatch_object_action_command(
            session.session_id,
            ResetSessionDefaults(),
        )
        assert session.values["model"] == defaults["model"]
        assert session.values["shifts"] == defaults["shifts"]
        assert session.has_unsaved_changes is False
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_object_action_session_save_to_defaults_persists_song_defaults():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "session-save-defaults.wav")
        runtime.add_song_from_path("Session Save Defaults", audio_path)

        session = runtime.open_object_action_session(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )
        session = runtime.dispatch_object_action_command(
            session.session_id,
            ReplaceSessionValues({"model": "mdx_extra", "device": "cpu"}),
        )

        session = runtime.dispatch_object_action_command(
            session.session_id,
            SaveSessionToDefaults(),
        )
        assert session.scope == "version"
        assert session.values["model"] == "mdx_extra"

        default_plan = runtime.describe_object_action(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
            scope="song_default",
        )
        assert any(
            field.key == "model" and field.value == "mdx_extra"
            for field in default_plan.editable_fields
        )
        assert any(
            field.key == "device" and field.value == "cpu"
            for field in default_plan.editable_fields
        )
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_request_object_action_run_returns_immediately_and_refreshes_on_completion():
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    gate = threading.Event()
    started = threading.Event()
    original_execute = analysis_service.execute

    def _blocking_execute(session, config_id, runtime_bindings=None, on_progress=None):
        if on_progress is not None:
            on_progress("Loading configuration", 0.0)
            on_progress("Preparing pipeline", 0.1)
            on_progress("Executing pipeline", 0.2)
        started.set()
        gate.wait(timeout=5.0)
        return original_execute(
            session,
            config_id,
            runtime_bindings=runtime_bindings,
            on_progress=on_progress,
        )

    analysis_service.execute = _blocking_execute
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "async-run.wav")
        runtime.add_song_from_path("Async Run", audio_path)

        started_at = time.monotonic()
        run_id = runtime.request_object_action_run(
            "timeline.extract_stems",
            object_id="source_audio",
            object_type="layer",
        )
        elapsed = time.monotonic() - started_at

        assert elapsed < 0.2
        assert _wait_until(lambda: started.is_set())
        banner = runtime.presentation().pipeline_run_banner
        assert banner is not None
        assert banner.title == "Extract Stems"
        assert banner.is_error is False
        plan = runtime.describe_object_action(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )
        assert plan.is_running is True
        assert plan.run_label == "Running..."
        assert plan.run_id == run_id

        gate.set()
        final_state = runtime.wait_for_pipeline_run(run_id, timeout=5.0)
        assert final_state.status == "completed"
        assert _wait_until(
            lambda: runtime.consume_pipeline_run_presentation_update() is not None,
            timeout=5.0,
        )
        presentation = runtime.presentation()
        assert [layer.title for layer in presentation.layers][:5] == [
            "Async Run",
            "Drums",
            "Bass",
            "Vocals",
            "Other",
        ]
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_object_action_session_run_requests_background_pipeline():
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    gate = threading.Event()
    started = threading.Event()
    original_execute = analysis_service.execute

    def _blocking_execute(session, config_id, runtime_bindings=None, on_progress=None):
        if on_progress is not None:
            on_progress("Loading configuration", 0.0)
            on_progress("Preparing pipeline", 0.1)
            on_progress("Executing pipeline", 0.2)
        started.set()
        gate.wait(timeout=5.0)
        return original_execute(
            session,
            config_id,
            runtime_bindings=runtime_bindings,
            on_progress=on_progress,
        )

    analysis_service.execute = _blocking_execute
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "session-async.wav")
        runtime.add_song_from_path("Session Async", audio_path)

        session = runtime.open_object_action_session(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )
        session = runtime.dispatch_object_action_command(session.session_id, RunSession())

        assert _wait_until(lambda: started.is_set())
        assert session.plan.is_running is True
        assert session.plan.run_label == "Running..."
        assert session.can_save_and_run is False
        assert session.run_disabled_reason == "This stage is already running."

        gate.set()
        final_state = runtime.wait_for_pipeline_run(session.plan.run_id, timeout=5.0)
        assert final_state.status == "completed"
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_object_action_session_classify_uses_descriptor_bound_runtime_bindings(
    monkeypatch,
):
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
    fake_models_root = temp_root / "models"
    fake_models_root.mkdir(parents=True, exist_ok=True)
    installed_manifest = fake_models_root / "installed-classifier.manifest.json"
    installed_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.ensure_installed_models_dir",
        lambda: fake_models_root,
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
        session = runtime.dispatch_object_action_command(session.session_id, RunSession())
        assert session.plan.run_id is not None
        runtime.wait_for_pipeline_run(session.plan.run_id, timeout=5.0)
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
        assert [field.key for field in plan.editable_fields] == [
            "onset_threshold",
            "onset_min_gap",
            "onset_method",
            "onset_backtrack",
            "onset_timing_offset_ms",
            "classify_model_path",
            "classify_device",
            "classify_batch_size",
        ]
        assert any(
            field.key == "classify_model_path" and field.value == str(model_path)
            for field in plan.editable_fields
        )
        model_field = next(field for field in plan.editable_fields if field.key == "classify_model_path")
        assert model_field.widget == "dropdown"
        assert {option.value for option in model_field.options} >= {
            "",
            str(installed_manifest),
            str(model_path),
        }
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_classified_drums_settings_expose_model_fields(monkeypatch):
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    fake_models_root = temp_root / "models"
    fake_models_root.mkdir(parents=True, exist_ok=True)
    kick_manifest = fake_models_root / "kick.manifest.json"
    snare_manifest = fake_models_root / "snare.manifest.json"
    kick_manifest.write_text("{}", encoding="utf-8")
    snare_manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.ensure_installed_models_dir",
        lambda: fake_models_root,
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.upgrade_installed_runtime_bundles",
        lambda _models_dir: None,
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.resolve_installed_binary_drum_bundles",
        lambda: {
            "kick": type("Bundle", (), {"manifest_path": kick_manifest})(),
            "snare": type("Bundle", (), {"manifest_path": snare_manifest})(),
        },
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "classified-settings.wav")
        runtime.add_song_from_path("Classified Settings", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        plan = runtime.describe_object_action(
            "timeline.extract_classified_drums",
            {"layer_id": drums_layer.layer_id},
            object_id=drums_layer.layer_id,
            object_type="layer",
        )

        assert [field.key for field in plan.editable_fields] == [
            "kick_model_path",
            "snare_model_path",
            "classify_device",
            "kick_positive_threshold",
            "snare_positive_threshold",
            "kick_filter_enabled",
            "kick_filter_freq",
            "kick_onset_threshold",
            "snare_filter_enabled",
            "snare_filter_freq",
            "snare_onset_threshold",
        ]
        assert [field.key for field in plan.advanced_fields] == [
            "kick_filter_type",
            "kick_onset_min_gap",
            "kick_onset_method",
            "kick_onset_backtrack",
            "kick_onset_timing_offset_ms",
            "snare_filter_type",
            "snare_onset_min_gap",
            "snare_onset_method",
            "snare_onset_backtrack",
            "snare_onset_timing_offset_ms",
            "assignment_mode",
            "winner_margin",
            "event_match_window_ms",
        ]
        assert [key for key, _value in plan.locked_bindings] == ["audio_file"]
        assert any(
            field.key == "kick_model_path" and field.value == str(kick_manifest)
            for field in plan.editable_fields
        )
        assert any(
            field.key == "snare_model_path" and field.value == str(snare_manifest)
            for field in plan.editable_fields
        )
        kick_field = next(field for field in plan.editable_fields if field.key == "kick_model_path")
        snare_field = next(field for field in plan.editable_fields if field.key == "snare_model_path")
        assert kick_field.widget == "dropdown"
        assert snare_field.widget == "dropdown"
        assert {option.value for option in kick_field.options} >= {
            "",
            str(kick_manifest),
            str(snare_manifest),
        }
        assert {option.value for option in snare_field.options} >= {
            "",
            str(kick_manifest),
            str(snare_manifest),
        }
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_classified_drums_hydrates_legacy_assignment_mode(monkeypatch):
    from echozero.domain.types import BlockSettings
    from echozero.serialization import deserialize_graph, serialize_graph

    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    fake_models_root = temp_root / "models"
    fake_models_root.mkdir(parents=True, exist_ok=True)
    kick_manifest = fake_models_root / "kick.manifest.json"
    snare_manifest = fake_models_root / "snare.manifest.json"
    kick_manifest.write_text("{}", encoding="utf-8")
    snare_manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.ensure_installed_models_dir",
        lambda: fake_models_root,
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.upgrade_installed_runtime_bundles",
        lambda _models_dir: None,
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.resolve_installed_binary_drum_bundles",
        lambda: {
            "kick": type("Bundle", (), {"manifest_path": kick_manifest})(),
            "snare": type("Bundle", (), {"manifest_path": snare_manifest})(),
        },
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "classified-legacy-assignment.wav")
        runtime.add_song_from_path("Classified Legacy Assignment", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        runtime.describe_object_action(
            "timeline.extract_classified_drums",
            {"layer_id": drums_layer.layer_id},
            object_id=drums_layer.layer_id,
            object_type="layer",
        )

        song_version_id = str(runtime.session.active_song_version_id)
        config = next(
            candidate
            for candidate in runtime.project_storage.pipeline_configs.list_by_version(song_version_id)
            if candidate.template_id == "extract_classified_drums"
        )

        graph = deserialize_graph(json.loads(config.graph_json))
        classify = graph.blocks["classify_drums"]
        graph.replace_block(
            replace(
                classify,
                settings=BlockSettings(
                    {
                        **dict(classify.settings),
                        "assignment_mode": "exclusive_max",
                    }
                ),
            )
        )
        legacy_knobs = dict(config.knob_values)
        legacy_knobs.pop("assignment_mode", None)
        legacy_config = replace(
            config,
            graph_json=json.dumps(serialize_graph(graph)),
            knob_values=legacy_knobs,
        )
        runtime.project_storage.pipeline_configs.update(legacy_config)
        runtime.project_storage.commit()

        refreshed = runtime.describe_object_action(
            "timeline.extract_classified_drums",
            {"layer_id": drums_layer.layer_id},
            object_id=drums_layer.layer_id,
            object_type="layer",
        )
        assignment_field = next(
            field for field in refreshed.advanced_fields if field.key == "assignment_mode"
        )
        assert assignment_field.value == "independent"

        persisted = next(
            candidate
            for candidate in runtime.project_storage.pipeline_configs.list_by_version(song_version_id)
            if candidate.template_id == "extract_classified_drums"
        )
        assert persisted.knob_values.get("assignment_mode") == "independent"
        persisted_pipeline = persisted.to_pipeline()
        assert (
            persisted_pipeline.graph.blocks["classify_drums"].settings.get("assignment_mode")
            == "independent"
        )
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_song_drum_events_settings_expose_model_fields(monkeypatch):
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    fake_models_root = temp_root / "models"
    fake_models_root.mkdir(parents=True, exist_ok=True)
    kick_manifest = fake_models_root / "kick.manifest.json"
    snare_manifest = fake_models_root / "snare.manifest.json"
    kick_manifest.write_text("{}", encoding="utf-8")
    snare_manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.ensure_installed_models_dir",
        lambda: fake_models_root,
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.upgrade_installed_runtime_bundles",
        lambda _models_dir: None,
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.resolve_installed_binary_drum_bundles",
        lambda: {
            "kick": type("Bundle", (), {"manifest_path": kick_manifest})(),
            "snare": type("Bundle", (), {"manifest_path": snare_manifest})(),
        },
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "song-drum-settings.wav")
        runtime.add_song_from_path("Song Drum Settings", audio_path)

        plan = runtime.describe_object_action(
            "timeline.extract_song_drum_events",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )

        assert plan.pipeline_template_id == "extract_song_drum_events"
        assert [field.key for field in plan.editable_fields] == [
            "model",
            "device",
            "include_drums_stem_layer",
            "include_bass_stem_layer",
            "include_vocals_stem_layer",
            "include_other_stem_layer",
            "kick_model_path",
            "snare_model_path",
            "kick_positive_threshold",
            "snare_positive_threshold",
            "kick_filter_enabled",
            "kick_filter_freq",
            "kick_onset_threshold",
            "snare_filter_enabled",
            "snare_filter_freq",
            "snare_onset_threshold",
        ]
        assert [field.key for field in plan.advanced_fields] == [
            "shifts",
            "kick_filter_type",
            "kick_onset_min_gap",
            "kick_onset_method",
            "kick_onset_backtrack",
            "kick_onset_timing_offset_ms",
            "snare_filter_type",
            "snare_onset_min_gap",
            "snare_onset_method",
            "snare_onset_backtrack",
            "snare_onset_timing_offset_ms",
            "assignment_mode",
            "winner_margin",
            "event_match_window_ms",
        ]
        assert [key for key, _value in plan.locked_bindings] == ["audio_file"]
        assert any(
            field.key == "kick_model_path" and field.value == str(kick_manifest)
            for field in plan.editable_fields
        )
        assert any(
            field.key == "snare_model_path" and field.value == str(snare_manifest)
            for field in plan.editable_fields
        )
        kick_field = next(field for field in plan.editable_fields if field.key == "kick_model_path")
        snare_field = next(field for field in plan.editable_fields if field.key == "snare_model_path")
        assert kick_field.widget == "dropdown"
        assert snare_field.widget == "dropdown"
        assert {option.value for option in kick_field.options} >= {
            "",
            str(kick_manifest),
            str(snare_manifest),
        }
        assert {option.value for option in snare_field.options} >= {
            "",
            str(kick_manifest),
            str(snare_manifest),
        }
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_classified_drums_settings_accept_custom_model_selection(monkeypatch):
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    fake_models_root = temp_root / "models"
    fake_models_root.mkdir(parents=True, exist_ok=True)
    default_kick_manifest = fake_models_root / "default-kick.manifest.json"
    default_snare_manifest = fake_models_root / "default-snare.manifest.json"
    custom_kick_manifest = fake_models_root / "custom-kick.manifest.json"
    custom_snare_manifest = fake_models_root / "custom-snare.manifest.json"
    for manifest in (
        default_kick_manifest,
        default_snare_manifest,
        custom_kick_manifest,
        custom_snare_manifest,
    ):
        manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.ensure_installed_models_dir",
        lambda: fake_models_root,
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.upgrade_installed_runtime_bundles",
        lambda _models_dir: None,
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.resolve_installed_binary_drum_bundles",
        lambda: {
            "kick": type("Bundle", (), {"manifest_path": default_kick_manifest})(),
            "snare": type("Bundle", (), {"manifest_path": default_snare_manifest})(),
        },
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "classified-custom.wav")
        runtime.add_song_from_path("Classified Custom", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        runtime.save_object_action_settings(
            "timeline.extract_classified_drums",
            {
                "layer_id": drums_layer.layer_id,
                "kick_model_path": str(custom_kick_manifest),
                "snare_model_path": str(custom_snare_manifest),
            },
            object_id=drums_layer.layer_id,
            object_type="layer",
            scope="version",
        )
        refreshed = runtime.describe_object_action(
            "timeline.extract_classified_drums",
            {"layer_id": drums_layer.layer_id},
            object_id=drums_layer.layer_id,
            object_type="layer",
        )

        assert any(
            field.key == "kick_model_path" and field.value == str(custom_kick_manifest)
            for field in refreshed.editable_fields
        )
        assert any(
            field.key == "snare_model_path" and field.value == str(custom_snare_manifest)
            for field in refreshed.editable_fields
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

        session = runtime.dispatch_object_action_command(session.session_id, RunSession())
        assert session.plan.run_id is not None
        assert refresh_calls == []
        runtime.wait_for_pipeline_run(session.plan.run_id, timeout=5.0)
        assert runtime.consume_pipeline_run_presentation_update() is not None
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



__all__ = [name for name in globals() if name.startswith("test_")]
