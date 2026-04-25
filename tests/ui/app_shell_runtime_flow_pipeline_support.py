"""Pipeline runtime-flow support cases.
Exists to keep extract, classify, and take-persistence coverage separate from project and audio-runtime tests.
Connects the compatibility wrapper to the bounded pipeline support slice.
"""

from tests.ui.app_shell_runtime_flow_shared_support import *  # noqa: F401,F403

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


def test_app_shell_runtime_pipeline_runs_do_not_auto_save_projects(monkeypatch):
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)
    assert runtime.project_storage._autosave_timer is None
    save_calls = 0
    save_as_calls: list[Path] = []

    def _capture_save() -> None:
        nonlocal save_calls
        save_calls += 1

    def _capture_save_as(path) -> None:
        save_as_calls.append(Path(path))

    monkeypatch.setattr(runtime.project_storage, "save", _capture_save)
    monkeypatch.setattr(runtime.project_storage, "save_as", _capture_save_as)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        runtime.extract_stems("source_audio")

        assert save_calls == 0
        assert save_as_calls == []
        assert runtime.project_storage._autosave_timer is None
        assert sorted(temp_root.glob("*.ez")) == []
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


def test_app_shell_runtime_delete_take_persists_numbered_take_labels_on_reload():
    temp_root = _repo_local_temp_root()
    save_path = temp_root / "delete-take.ez"
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "delete-take.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        runtime.extract_stems("source_audio")
        second_pass = runtime.extract_stems("source_audio")

        drums_layer = next(layer for layer in second_pass.layers if layer.title == "Drums")
        assert [take.name for take in drums_layer.takes] == ["Take 2"]

        runtime.dispatch(
            TriggerTakeAction(
                layer_id=drums_layer.layer_id,
                take_id=drums_layer.takes[0].take_id,
                action_id="delete_take",
            )
        )

        after_delete = runtime.presentation()
        deleted_drums = next(layer for layer in after_delete.layers if layer.title == "Drums")
        assert deleted_drums.takes == []

        runtime.save_project_as(save_path)
        runtime.open_project(save_path)

        reloaded_drums = next(layer for layer in runtime.presentation().layers if layer.title == "Drums")
        assert reloaded_drums.takes == []
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_add_selection_to_main_persists_after_reload():
    temp_root = _repo_local_temp_root()
    save_path = temp_root / "selection-to-main.ez"
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "selection-to-main.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        first_pass = runtime.extract_drum_events(drums_layer.layer_id)
        onsets_first = next(layer for layer in first_pass.layers if layer.title == "Onsets")
        second_pass = runtime.extract_drum_events(drums_layer.layer_id)
        onsets_second = next(layer for layer in second_pass.layers if layer.title == "Onsets")
        selected_take = onsets_second.takes[0]
        selected_event = selected_take.events[0]

        selected = runtime.dispatch(
            SelectEvent(
                onsets_second.layer_id,
                selected_take.take_id,
                selected_event.event_id,
            )
        )
        selected_onsets = next(layer for layer in selected.layers if layer.title == "Onsets")
        assert {action.action_id for action in selected_onsets.takes[0].actions} >= {
            "add_selection_to_main",
            "overwrite_main",
            "merge_main",
            "delete_take",
        }

        runtime.dispatch(
            TriggerTakeAction(
                layer_id=selected_onsets.layer_id,
                take_id=selected_take.take_id,
                action_id="add_selection_to_main",
            )
        )

        after_action = next(layer for layer in runtime.presentation().layers if layer.title == "Onsets")
        assert len(after_action.events) == len(onsets_first.events) + 1

        runtime.save_project_as(save_path)
        runtime.open_project(save_path)

        reloaded = next(layer for layer in runtime.presentation().layers if layer.title == "Onsets")
        assert len(reloaded.events) == len(onsets_first.events) + 1
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


def test_app_shell_runtime_extract_song_drum_events_from_source_audio(monkeypatch):
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    detect_executor = _CaptureDetectOnsetsAudioExecutor()
    binary_executor = _CaptureBinaryDrumClassifyAudioExecutor()

    class _TwoStemSeparateAudioExecutor:
        def execute(self, block_id: str, context):
            audio = context.get_input(block_id, "audio_in", AudioData)
            assert audio is not None
            base = Path(str(audio.file_path)).parent
            drums_path = write_test_wav(base / "drums.wav")
            remainder_path = write_test_wav(base / "no_drums.wav")
            return ok(
                {
                    "drums_out": AudioData(
                        sample_rate=44100,
                        duration=0.1,
                        file_path=str(drums_path),
                        channel_count=1,
                    ),
                    "no_drums_out": AudioData(
                        sample_rate=44100,
                        duration=0.1,
                        file_path=str(remainder_path),
                        channel_count=1,
                    ),
                }
            )

    analysis_service._executors["SeparateAudio"] = _TwoStemSeparateAudioExecutor()
    analysis_service._executors["DetectOnsets"] = detect_executor
    analysis_service._executors["BinaryDrumClassify"] = binary_executor
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
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

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "song-drums.wav")
        runtime.add_song_from_path("Song Drums", audio_path)

        presentation = runtime.extract_song_drum_events("source_audio")

        event_layers = [layer for layer in presentation.layers if layer.kind.name == "EVENT"]
        titles = {layer.title for layer in event_layers}
        assert "Kick" in titles
        assert "Snare" in titles
        assert all(layer.status.source_layer_id == "source_audio" for layer in event_layers)
        assert all(layer.source_audio_path for layer in event_layers)
        assert all(
            Path(str(layer.source_audio_path)).name == "drums.wav"
            for layer in event_layers
        )
        detect_calls = {
            (block_id, Path(audio_path).name) for block_id, audio_path in detect_executor.calls
        }
        assert ("kick_onsets", "kick_filter.wav") in detect_calls
        assert ("snare_onsets", "snare_filter.wav") in detect_calls
        assert [(block_id, target_class, Path(audio_path).name) for block_id, target_class, audio_path in binary_executor.calls] == [
            ("classify_drums", "", "drums.wav"),
        ]
        assert runtime.is_dirty is True
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_song_drum_events_adds_selected_stem_layers(monkeypatch):
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    detect_executor = _CaptureDetectOnsetsAudioExecutor()
    binary_executor = _CaptureBinaryDrumClassifyAudioExecutor()
    analysis_service._executors["DetectOnsets"] = detect_executor
    analysis_service._executors["BinaryDrumClassify"] = binary_executor
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
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

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "song-drums-with-stems.wav")
        runtime.add_song_from_path("Song Drums", audio_path)
        runtime.save_object_action_settings(
            "timeline.extract_song_drum_events",
            {
                "layer_id": "source_audio",
                "include_bass_stem_layer": True,
                "include_vocals_stem_layer": True,
            },
            object_id="source_audio",
            object_type="layer",
            scope="version",
        )

        presentation = runtime.extract_song_drum_events("source_audio")

        audio_layers = [layer for layer in presentation.layers if layer.kind.name == "AUDIO"]
        audio_titles = {layer.title for layer in audio_layers}
        assert "Bass" in audio_titles
        assert "Vocals" in audio_titles
        assert "Drums" not in audio_titles
        assert "Other" not in audio_titles

        bass_layer = next(layer for layer in audio_layers if layer.title == "Bass")
        vocals_layer = next(layer for layer in audio_layers if layer.title == "Vocals")
        assert Path(str(bass_layer.source_audio_path)).name == "bass.wav"
        assert Path(str(vocals_layer.source_audio_path)).name == "vocals.wav"
        assert bass_layer.status.source_layer_id == "source_audio"
        assert vocals_layer.status.source_layer_id == "source_audio"

        event_layers = [layer for layer in presentation.layers if layer.kind.name == "EVENT"]
        event_titles = {layer.title for layer in event_layers}
        assert "Kick" in event_titles
        assert "Snare" in event_titles
        assert [(block_id, target_class, Path(audio_path).name) for block_id, target_class, audio_path in binary_executor.calls] == [
            ("classify_drums", "", "drums.wav"),
        ]
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_drum_events_persists_event_layers_from_drums_stem():
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    detect_executor = _CaptureDetectOnsetsAudioExecutor()
    analysis_service._executors["DetectOnsets"] = detect_executor
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
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
        assert all(layer.status.source_layer_id == str(drums_layer.layer_id) for layer in event_layers)
        assert any(
            (layer.status.source_label or "").startswith("onset_detection")
            for layer in event_layers
        )
        assert detect_executor.audio_paths[-1] == str(drums_layer.source_audio_path)
        assert runtime.is_dirty is True
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_drum_events_rerun_surfaces_new_take_with_saved_threshold():
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    detect_executor = _ThresholdAwareDetectOnsetsExecutor()
    analysis_service._executors["DetectOnsets"] = detect_executor
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        first_pass = runtime.extract_drum_events(drums_layer.layer_id)
        onsets_first = next(layer for layer in first_pass.layers if layer.title == "Onsets")
        assert detect_executor.thresholds == [0.3]
        assert len(onsets_first.events) == 7
        assert onsets_first.takes == []

        runtime.save_object_action_settings(
            "timeline.extract_drum_events",
            {"layer_id": drums_layer.layer_id, "threshold": 0.05},
            object_id=drums_layer.layer_id,
            object_type="layer",
            scope="version",
        )

        second_pass = runtime.extract_drum_events(drums_layer.layer_id)
        onsets_second = next(layer for layer in second_pass.layers if layer.title == "Onsets")
        assert detect_executor.thresholds == [0.3, 0.05]
        assert len(onsets_second.events) == 7
        assert len(onsets_second.takes) == 1
        assert len(onsets_second.takes[0].events) == 10
        assert second_pass.selected_layer_id == onsets_second.layer_id
        assert second_pass.selected_take_id == onsets_second.takes[0].take_id
        assert onsets_second.takes[0].is_selected is True
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_drum_events_rerun_clears_stale_selected_event_refs():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "clear-selection-rerun.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        first_pass = runtime.extract_drum_events(drums_layer.layer_id)
        onsets_layer = next(layer for layer in first_pass.layers if layer.title == "Onsets")
        selected = runtime.dispatch(
            SelectEvent(
                onsets_layer.layer_id,
                onsets_layer.main_take_id,
                onsets_layer.events[0].event_id,
            )
        )
        assert selected.selected_event_refs

        rerun = runtime.extract_drum_events(drums_layer.layer_id)
        rerun_onsets = next(layer for layer in rerun.layers if layer.title == "Onsets")

        assert rerun.selected_take_id == rerun_onsets.takes[0].take_id
        assert rerun.selected_event_ids == []
        assert rerun.selected_event_refs == []
        assert all(event.is_selected is False for event in rerun_onsets.events)
        assert all(event.is_selected is False for event in rerun_onsets.takes[0].events)
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_generated_event_layer_preview_resolves_source_audio():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    counted = _CountedRuntimeAudio()
    runtime.runtime_audio = counted

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")
        presentation = runtime.extract_drum_events(drums_layer.layer_id)
        onsets_layer = next(layer for layer in presentation.layers if layer.title == "Onsets")

        selected = runtime.dispatch(
            SelectEvent(
                onsets_layer.layer_id,
                onsets_layer.main_take_id,
                onsets_layer.events[0].event_id,
            )
        )
        contract = build_timeline_inspector_contract(selected)
        preview_action = next(
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id == "preview_event_clip"
        )

        assert preview_action.params["source_ref"] == drums_layer.source_audio_path
        assert preview_action.params["source_audio_path"] == drums_layer.source_audio_path

        runtime.preview_event_clip(
            layer_id=onsets_layer.layer_id,
            take_id=onsets_layer.main_take_id,
            event_id=onsets_layer.events[0].event_id,
        )

        assert counted.preview_calls == [
            (
                str(drums_layer.source_audio_path),
                float(onsets_layer.events[0].start),
                float(onsets_layer.events[0].end),
                0.0,
            )
        ]
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


def test_app_shell_runtime_classify_drum_events_uses_drums_audio_for_classifier():
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    capture_executor = _CapturePyTorchAudioClassifyAudioExecutor()
    analysis_service._executors["PyTorchAudioClassify"] = capture_executor
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import.wav")
        model_path = write_test_model(temp_root / "fixtures" / "drum-model.pth")
        runtime.add_song_from_path("Imported Song", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        runtime.classify_drum_events(drums_layer.layer_id, model_path)

        assert capture_executor.audio_paths == [str(drums_layer.source_audio_path)]
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_classified_drum_selection_stays_on_its_own_layer():
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
        runtime.extract_drum_events(drums_layer.layer_id)
        presentation = runtime.classify_drum_events(drums_layer.layer_id, model_path)

        onsets_layer = next(layer for layer in presentation.layers if layer.title == "Onsets")
        classified_layer = next(
            layer for layer in presentation.layers if layer.title == "Drum_Classified_Events"
        )
        assert onsets_layer.events[0].event_id != classified_layer.events[0].event_id

        selected = runtime.dispatch(
            SelectEvent(
                onsets_layer.layer_id,
                onsets_layer.main_take_id,
                onsets_layer.events[0].event_id,
            )
        )
        selected_onsets_layer = next(layer for layer in selected.layers if layer.title == "Onsets")
        selected_classified_layer = next(
            layer for layer in selected.layers if layer.title == "Drum_Classified_Events"
        )

        assert selected_onsets_layer.events[0].is_selected is True
        assert selected_classified_layer.events[0].is_selected is False
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
    analysis_service = build_mock_analysis_service()
    detect_executor = _CaptureDetectOnsetsAudioExecutor()
    binary_executor = _CaptureBinaryDrumClassifyAudioExecutor()
    analysis_service._executors["DetectOnsets"] = detect_executor
    analysis_service._executors["BinaryDrumClassify"] = binary_executor
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
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
        detect_calls = {
            (block_id, Path(audio_path).name) for block_id, audio_path in detect_executor.calls
        }
        assert ("kick_onsets", "kick_filter.wav") in detect_calls
        assert ("snare_onsets", "snare_filter.wav") in detect_calls
        assert binary_executor.calls == [
            ("classify_drums", "", str(drums_layer.source_audio_path)),
        ]
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_binary_drum_selection_stays_on_selected_class_layer(monkeypatch):
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    analysis_service._executors["BinaryDrumClassify"] = _CollidingBinaryDrumClassifyExecutor()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
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

        kick_layer = next(layer for layer in presentation.layers if layer.title == "Kick")
        snare_layer = next(layer for layer in presentation.layers if layer.title == "Snare")
        assert kick_layer.events[0].event_id != snare_layer.events[0].event_id

        selected = runtime.dispatch(
            SelectEvent(
                kick_layer.layer_id,
                kick_layer.main_take_id,
                kick_layer.events[0].event_id,
            )
        )
        selected_kick_layer = next(layer for layer in selected.layers if layer.title == "Kick")
        selected_snare_layer = next(layer for layer in selected.layers if layer.title == "Snare")

        assert selected_kick_layer.events[0].is_selected is True
        assert selected_snare_layer.events[0].is_selected is False
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)



__all__ = [name for name in globals() if name.startswith("test_")]
