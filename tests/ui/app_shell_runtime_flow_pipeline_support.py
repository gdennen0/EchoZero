"""Pipeline runtime-flow support cases.
Exists to keep extract, classify, and take-persistence coverage separate from project and audio-runtime tests.
Connects the compatibility wrapper to the bounded pipeline support slice.
"""

from tests.ui.app_shell_runtime_flow_shared_support import *  # noqa: F401,F403


def _assert_presentation_object_refs_resolve(runtime, presentation) -> None:
    for layer in presentation.layers:
        if layer.object_id is not None:
            object_record = runtime.project_storage.timeline_objects.get(str(layer.object_id))
            assert object_record is not None
            assert object_record.main_content_id == str(layer.main_content_id)
        if layer.main_content_id is not None:
            content_record = runtime.project_storage.object_contents.get(
                str(layer.main_content_id)
            )
            assert content_record is not None
            assert content_record.revision_id == str(layer.main_revision_id)
            if layer.source_content_ref is not None:
                source_content = runtime.project_storage.object_contents.get(
                    str(layer.source_content_ref.content_id)
                )
                assert source_content is not None
        for take in layer.takes:
            if take.content_id is not None:
                take_content = runtime.project_storage.object_contents.get(str(take.content_id))
                assert take_content is not None
                assert take_content.object_id == str(take.object_id)
                assert take_content.revision_id == str(take.revision_id)
            if take.source_content_ref is not None:
                source_content = runtime.project_storage.object_contents.get(
                    str(take.source_content_ref.content_id)
                )
                assert source_content is not None


def _install_fake_binary_drum_bundles(monkeypatch, temp_root):
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


def test_app_shell_runtime_extract_stems_persists_audio_layers_and_takes():
    from echozero.ui.qt.timeline.layer_rows import build_timeline_layer_rows

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
        hierarchy_rows = build_timeline_layer_rows(presentation.layers)
        assert [row.layer.title for row in hierarchy_rows[:5]] == [
            "Imported Song",
            "Drums",
            "Bass",
            "Vocals",
            "Other",
        ]
        assert hierarchy_rows[0].depth == 0
        assert hierarchy_rows[0].has_child_layers is True
        assert [row.depth for row in hierarchy_rows[1:5]] == [1, 1, 1, 1]

        stem_layers = presentation.layers[1:5]
        for layer in stem_layers:
            assert layer.kind.name == "AUDIO"
            assert layer.main_take_id is not None
            assert layer.source_audio_path
            assert layer.muted is True
            assert layer.status.source_label.startswith("stem_separation")
            assert layer.source_content_ref is not None
            source_content = runtime.project_storage.object_contents.get(
                str(layer.source_content_ref.content_id)
            )
            assert source_content is not None
            assert source_content.object_id == str(layer.source_content_ref.object_id)
            assert source_content.revision_id == str(layer.source_content_ref.revision_id)

            assert layer.parent_layer_id == presentation.layers[0].layer_id
            assert "child" in layer.badges
            layer_record = runtime.project_storage.layers.get(str(layer.layer_id))
            assert layer_record is not None
            assert layer_record.parent_layer_id is None
            assert layer_record.state_flags["mute"] is True
            assert layer_record.provenance["source_layer_id"] == str(
                presentation.layers[0].layer_id
            )

            layer_object = runtime.project_storage.timeline_objects.get(str(layer.object_id))
            assert layer_object is not None
            assert layer_object.main_content_id == str(layer.main_content_id)
            main_content = runtime.project_storage.object_contents.get(str(layer.main_content_id))
            assert main_content is not None
            assert main_content.source_ref == layer.source_content_ref.to_dict()
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


def test_app_shell_runtime_promote_stem_take_updates_main_content_on_child_layer():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "promote-stem.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        runtime.extract_stems("source_audio")
        second_pass = runtime.extract_stems("source_audio")

        drums_layer = next(layer for layer in second_pass.layers if layer.title == "Drums")
        assert drums_layer.parent_layer_id == second_pass.layers[0].layer_id
        assert drums_layer.takes
        promoted_take = drums_layer.takes[0]
        prior_main_take_id = drums_layer.main_take_id
        prior_main_content_id = drums_layer.main_content_id

        runtime.dispatch(
            TriggerTakeAction(
                layer_id=drums_layer.layer_id,
                take_id=promoted_take.take_id,
                action_id="overwrite_main",
            )
        )

        promoted = next(layer for layer in runtime.presentation().layers if layer.title == "Drums")
        assert promoted.parent_layer_id == second_pass.layers[0].layer_id
        assert promoted.main_take_id == promoted_take.take_id
        assert promoted.main_content_id == promoted_take.content_id
        assert promoted.main_content_id != prior_main_content_id
        assert [take.take_id for take in promoted.takes] == [prior_main_take_id]
        assert promoted.takes[0].content_id == prior_main_content_id

        layer_record = runtime.project_storage.layers.get(str(promoted.layer_id))
        assert layer_record is not None
        persisted_takes = runtime.project_storage.takes.list_by_layer(str(promoted.layer_id))
        persisted_main = next(take for take in persisted_takes if take.is_main)
        assert persisted_main.id == str(promoted_take.take_id)
        drums_object = runtime.project_storage.timeline_objects.get(str(promoted.object_id))
        assert drums_object is not None
        assert drums_object.main_content_id == str(promoted_take.content_id)
    finally:
        runtime.shutdown()
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
        drums_object = runtime.project_storage.timeline_objects.get(str(deleted_drums.object_id))
        assert drums_object is not None
        drums_contents = runtime.project_storage.object_contents.list_by_object(drums_object.id)
        assert [content.id for content in drums_contents] == [str(deleted_drums.main_content_id)]
        assert runtime.project_storage.object_candidates.list_by_object(drums_object.id) == []

        runtime.save_project_as(save_path)
        runtime.open_project(save_path)

        reloaded_drums = next(
            layer for layer in runtime.presentation().layers if layer.title == "Drums"
        )
        assert reloaded_drums.takes == []
        assert (
            runtime.project_storage.object_contents.get(str(reloaded_drums.main_content_id))
            is not None
        )
        _assert_presentation_object_refs_resolve(runtime, runtime.presentation())
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

        after_action = next(
            layer for layer in runtime.presentation().layers if layer.title == "Onsets"
        )
        assert len(after_action.events) == len(onsets_first.events) + 1
        onsets_object = runtime.project_storage.timeline_objects.get(str(after_action.object_id))
        assert onsets_object is not None
        assert onsets_object.main_content_id == str(after_action.main_content_id)
        assert (
            runtime.project_storage.object_contents.get(str(after_action.main_content_id))
            is not None
        )

        runtime.save_project_as(save_path)
        runtime.open_project(save_path)

        reloaded = next(
            layer for layer in runtime.presentation().layers if layer.title == "Onsets"
        )
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
        source_layer = next(
            layer
            for layer in presentation.layers
            if layer.object_id and str(layer.object_id).startswith("object_song_")
        )
        assert all(
            layer.status.source_layer_id == str(source_layer.layer_id) for layer in event_layers
        )
        assert all(layer.source_content_ref is not None for layer in event_layers)
        assert all(
            Path(str(layer.source_content_ref.locator)).name == "drums.wav"
            for layer in event_layers
            if layer.source_content_ref is not None
        )
        for layer in event_layers:
            source_ref = layer.source_content_ref
            assert source_ref is not None
            source_content = runtime.project_storage.object_contents.get(
                str(source_ref.content_id)
            )
            assert source_content is not None
            assert source_content.object_id == str(source_ref.object_id)
            assert source_content.revision_id == str(source_ref.revision_id)
        detect_calls = {
            (block_id, Path(audio_path).name) for block_id, audio_path in detect_executor.calls
        }
        assert ("kick_onsets", "kick_filter.wav") in detect_calls
        assert ("snare_onsets", "snare_filter.wav") in detect_calls
        assert [
            (block_id, target_class, Path(audio_path).name)
            for block_id, target_class, audio_path in binary_executor.calls
        ] == [
            ("classify_drums", "", "drums.wav"),
        ]
        assert runtime.is_dirty is True
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_song_drum_events_reuses_existing_drums_stem(monkeypatch):
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    detect_executor = _CaptureDetectOnsetsAudioExecutor()
    binary_executor = _CaptureBinaryDrumClassifyAudioExecutor()

    class _CountingSeparateAudioExecutor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []
            self.fail_after_first_call = False

        def execute(self, block_id: str, context):
            audio = context.get_input(block_id, "audio_in", AudioData)
            assert audio is not None
            if self.fail_after_first_call and self.calls:
                raise AssertionError("SeparateAudio should have reused the persisted drums stem")
            self.calls.append((block_id, str(audio.file_path)))
            base = Path(str(audio.file_path)).parent
            return ok(
                {
                    name + "_out": AudioData(
                        sample_rate=44100,
                        duration=0.1,
                        file_path=str(write_test_wav(base / f"{name}.wav")),
                        channel_count=1,
                    )
                    for name in ("drums", "bass", "vocals", "other")
                }
            )

    separator = _CountingSeparateAudioExecutor()
    analysis_service._executors["SeparateAudio"] = separator
    analysis_service._executors["DetectOnsets"] = detect_executor
    analysis_service._executors["BinaryDrumClassify"] = binary_executor
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
    )
    _install_fake_binary_drum_bundles(monkeypatch, temp_root)

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "reuse-song.wav")
        runtime.add_song_from_path("Reuse Song", audio_path)
        stem_presentation = runtime.extract_stems("source_audio")
        assert [block_id for block_id, _audio_path in separator.calls] == ["separate"]

        drums_layer = next(layer for layer in stem_presentation.layers if layer.title == "Drums")
        assert drums_layer.main_content_id is not None
        runtime.save_object_action_settings(
            "timeline.extract_song_drum_events",
            {"layer_id": "source_audio", "model": "latest_model"},
            object_id="source_audio",
        )

        separator.fail_after_first_call = True
        presentation = runtime.extract_song_drum_events("source_audio")

        assert [block_id for block_id, _audio_path in separator.calls] == ["separate"]
        assert [
            (block_id, target_class, Path(audio_path).name)
            for block_id, target_class, audio_path in binary_executor.calls
        ] == [("classify_drums", "", "drums.wav")]
        event_layers = [layer for layer in presentation.layers if layer.kind.name == "EVENT"]
        assert {layer.title for layer in event_layers} == {"Kick", "Snare"}
        assert all(
            layer.source_content_ref is not None
            and str(layer.source_content_ref.content_id) == str(drums_layer.main_content_id)
            for layer in event_layers
        )
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_song_drum_events_ignores_stale_stem_after_source_change(
    monkeypatch,
):
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    detect_executor = _CaptureDetectOnsetsAudioExecutor()
    binary_executor = _CaptureBinaryDrumClassifyAudioExecutor()

    class _CountingSeparateAudioExecutor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def execute(self, block_id: str, context):
            audio = context.get_input(block_id, "audio_in", AudioData)
            assert audio is not None
            self.calls.append((block_id, str(audio.file_path)))
            base = Path(str(audio.file_path)).parent
            return ok(
                {
                    name + "_out": AudioData(
                        sample_rate=44100,
                        duration=0.1,
                        file_path=str(write_test_wav(base / f"{name}.wav")),
                        channel_count=1,
                    )
                    for name in ("drums", "bass", "vocals", "other")
                }
            )

    separator = _CountingSeparateAudioExecutor()
    analysis_service._executors["SeparateAudio"] = separator
    analysis_service._executors["DetectOnsets"] = detect_executor
    analysis_service._executors["BinaryDrumClassify"] = binary_executor
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
    )
    _install_fake_binary_drum_bundles(monkeypatch, temp_root)

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Changed Source Song",
            write_test_wav(temp_root / "fixtures" / "source-v1.wav", frames=4410),
        )
        stem_presentation = runtime.extract_stems("source_audio")
        assert [block_id for block_id, _audio_path in separator.calls] == ["separate"]
        stale_drums_layer = next(
            layer for layer in stem_presentation.layers if layer.title == "Drums"
        )
        assert runtime.session.active_song_id is not None
        song_id = str(runtime.session.active_song_id)

        runtime.add_song_version(
            song_id,
            write_test_wav(temp_root / "fixtures" / "source-v2.wav", frames=8820),
            label="Source Edit",
            transfer_layers=True,
            transfer_layer_ids=[str(stale_drums_layer.layer_id)],
        )
        runtime.save_object_action_settings(
            "timeline.extract_song_drum_events",
            {"layer_id": "source_audio", "model": "latest_model"},
            object_id="source_audio",
        )

        presentation = runtime.extract_song_drum_events("source_audio")

        assert [block_id for block_id, _audio_path in separator.calls] == [
            "separate",
            "separate_drums",
        ]
        event_layers = [layer for layer in presentation.layers if layer.kind.name == "EVENT"]
        assert {layer.title for layer in event_layers} == {"Kick", "Snare"}
        assert all(layer.source_content_ref is not None for layer in event_layers)
        assert all(
            str(layer.source_content_ref.content_id) != str(stale_drums_layer.main_content_id)
            for layer in event_layers
            if layer.source_content_ref is not None
        )
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_song_drum_events_reuses_existing_unchanged_stem(monkeypatch):
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()

    class _CountingSeparateAudioExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def execute(self, block_id: str, context):
            audio = context.get_input(block_id, "audio_in", AudioData)
            assert audio is not None
            self.calls.append(str(audio.file_path))
            base = Path(str(audio.file_path)).parent
            return ok(
                {
                    stem_name + "_out": AudioData(
                        sample_rate=44100,
                        duration=0.1,
                        file_path=str(write_test_wav(base / f"{stem_name}.wav")),
                        channel_count=1,
                    )
                    for stem_name in ("drums", "bass", "vocals", "other")
                }
            )

    separator = _CountingSeparateAudioExecutor()
    binary_executor = _CaptureBinaryDrumClassifyAudioExecutor()
    analysis_service._executors["SeparateAudio"] = separator
    analysis_service._executors["BinaryDrumClassify"] = binary_executor
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
    )
    _install_fake_binary_drum_bundles(monkeypatch, temp_root)

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "song-with-cached-stems.wav")
        runtime.add_song_from_path("Song With Cached Stems", audio_path)

        stem_presentation = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in stem_presentation.layers if layer.title == "Drums")
        cached_drums_path = Path(str(drums_layer.source_audio_path)).resolve()
        assert len(separator.calls) == 1

        runtime.extract_song_drum_events("source_audio")

        assert len(separator.calls) == 1
        assert binary_executor.calls
        assert Path(binary_executor.calls[-1][2]).resolve() == cached_drums_path

        changed_audio_path = write_test_wav(temp_root / "fixtures" / "changed-song-source.wav")
        runtime.add_song_from_path("Changed Song Source", changed_audio_path)
        runtime.extract_song_drum_events("source_audio")

        assert len(separator.calls) == 2
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
                "include_drums_stem_layer": False,
                "include_bass_stem_layer": True,
                "include_vocals_stem_layer": True,
                "include_other_stem_layer": False,
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
        source_layer = next(
            layer
            for layer in presentation.layers
            if layer.object_id and str(layer.object_id).startswith("object_song_")
        )
        assert Path(str(bass_layer.source_audio_path)).name == "bass.wav"
        assert Path(str(vocals_layer.source_audio_path)).name == "vocals.wav"
        assert bass_layer.status.source_layer_id == str(source_layer.layer_id)
        assert vocals_layer.status.source_layer_id == str(source_layer.layer_id)

        event_layers = [layer for layer in presentation.layers if layer.kind.name == "EVENT"]
        event_titles = {layer.title for layer in event_layers}
        assert "Kick" in event_titles
        assert "Snare" in event_titles
        assert all(layer.parent_layer_id is None for layer in event_layers)
        assert [
            (block_id, target_class, Path(audio_path).name)
            for block_id, target_class, audio_path in binary_executor.calls
        ] == [
            ("classify_drums", "", "drums.wav"),
        ]
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_song_drum_events_source_audio_survives_save_and_reopen():
    temp_root = _repo_local_temp_root()
    save_path = temp_root / "extract-song-drum-events.ez"
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "extract-song-drum-events.wav")
        runtime.add_song_from_path("Extract Song Drum Events", audio_path)
        runtime.extract_song_drum_events("source_audio")
        runtime.save_project_as(save_path)
        runtime.open_project(save_path)

        event_layers = [
            layer for layer in runtime.presentation().layers if layer.kind.name == "EVENT"
        ]
        assert event_layers
        assert all(layer.source_content_ref is not None for layer in event_layers)
        assert all(
            Path(str(layer.source_content_ref.locator)).exists()
            for layer in event_layers
            if layer.source_content_ref is not None
        )
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
        assert all(
            layer.status.source_layer_id == str(drums_layer.layer_id) for layer in event_layers
        )
        assert all(layer.parent_layer_id is None for layer in event_layers)
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
        assert preview_action.params["preview"] == {
            "kind": "audio_event_clip",
            "source_ref": drums_layer.source_audio_path,
            "source_audio_path": drums_layer.source_audio_path,
            "waveform_key": drums_layer.waveform_key,
            "start_seconds": float(onsets_layer.events[0].start),
            "end_seconds": float(onsets_layer.events[0].end),
            "duration_seconds": float(onsets_layer.events[0].duration),
        }

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


def test_app_shell_runtime_extract_drum_events_supports_all_stem_layer_types():
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
        stem_layers = [
            layer
            for layer in after_stems.layers
            if layer.title in {"Drums", "Bass", "Vocals", "Other"}
        ]
        assert len(stem_layers) == 4

        for stem_layer in stem_layers:
            presentation = runtime.extract_drum_events(stem_layer.layer_id)
            event_layers = [layer for layer in presentation.layers if layer.kind.name == "EVENT"]
            assert event_layers
            assert any(layer.events for layer in event_layers)
            assert detect_executor.audio_paths[-1] == str(stem_layer.source_audio_path)
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


class _SelectedLabelBinaryDrumClassifyExecutor:
    def __init__(self) -> None:
        self.target_labels: list[tuple[str, ...]] = []

    def execute(self, block_id: str, context):
        block = context.graph.blocks[block_id]
        raw_labels = block.settings.get("target_labels", ("kick", "snare"))
        if isinstance(raw_labels, str):
            target_labels = tuple(label.strip() for label in raw_labels.split(",") if label.strip())
        else:
            target_labels = tuple(str(label).strip() for label in raw_labels if str(label).strip())
        self.target_labels.append(target_labels)
        input_events = _merged_binary_drum_input_events(block_id, context)
        source_event = input_events[0]
        layers = []
        for index, label in enumerate(target_labels):
            title = label.title()
            layers.append(
                DomainLayer(
                    id=label,
                    name=label,
                    events=(
                        DomainEvent(
                            id=f"{source_event.id}_{label}",
                            time=source_event.time + index * 0.1,
                            duration=source_event.duration,
                            classifications={"class": label, "confidence": "0.99"},
                            metadata={**source_event.metadata, "classified": True},
                            origin=f"binary_classify:{label}",
                        ),
                    ),
                )
            )
        return ok(EventData(layers=tuple(layers)))


def _write_ready_binary_drum_manifest(root: Path, label: str) -> Path:
    bundle_dir = root / label
    bundle_dir.mkdir(parents=True, exist_ok=True)
    weights_path = bundle_dir / f"{label}.pth"
    weights_path.write_bytes(b"fixture-model")
    manifest_path = bundle_dir / f"{label}.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "classes": [label, "other"],
                "weightsPath": weights_path.name,
                "classificationMode": "binary",
                "displayName": f"{label.title()} Fixture",
                "evalSummary": {"macroF1": 0.9},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_app_shell_runtime_extract_classified_drums_persists_all_ready_event_layers(monkeypatch):
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    analysis_service._executors["DetectOnsets"] = _CaptureDetectOnsetsAudioExecutor()
    binary_executor = _SelectedLabelBinaryDrumClassifyExecutor()
    analysis_service._executors["BinaryDrumClassify"] = binary_executor
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
    )

    fake_models_root = temp_root / "models"
    for label in ("kick", "snare", "clap", "cymbal"):
        _write_ready_binary_drum_manifest(fake_models_root, label)

    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.ensure_installed_models_dir",
        lambda: fake_models_root,
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.upgrade_installed_runtime_bundles",
        lambda _models_dir: None,
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "import-all-drums.wav")
        runtime.add_song_from_path("Imported Song", audio_path)
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")

        presentation = runtime.extract_classified_drums(drums_layer.layer_id)

        assert binary_executor.target_labels == [("kick", "snare", "clap", "cymbal")]
        event_layers = [layer for layer in presentation.layers if layer.kind.name == "EVENT"]
        titles = {layer.title for layer in event_layers}
        assert {"Kick", "Snare", "Clap", "Cymbal"} <= titles
        for expected_title in ("Kick", "Snare", "Clap", "Cymbal"):
            layer = next(layer for layer in event_layers if layer.title == expected_title)
            assert layer.events
            assert layer.events[0].label == expected_title
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


def test_app_shell_runtime_extract_song_sections_persists_section_layer():
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    captured_audio_paths: list[str] = []

    class _CaptureSongSectionsExecutor:
        def execute(self, block_id: str, context):
            audio = context.get_input(block_id, "audio_in", AudioData)
            assert audio is not None
            captured_audio_paths.append(str(audio.file_path))
            return ok(
                EventData(
                    layers=(
                        DomainLayer(
                            id="generated_sections",
                            name="Sections",
                            events=(
                                DomainEvent(
                                    id="sec_intro",
                                    time=0.0,
                                    duration=0.0,
                                    classifications={"label": "Intro", "confidence": 0.95},
                                    metadata={"cue_ref": "intro_01", "cue_number": 1},
                                    origin="detect_song_sections",
                                ),
                                DomainEvent(
                                    id="sec_chorus",
                                    time=1.1,
                                    duration=0.0,
                                    classifications={"label": "Chorus", "confidence": 0.9},
                                    metadata={"cue_ref": "chorus_02", "cue_number": 2},
                                    origin="detect_song_sections",
                                ),
                            ),
                        ),
                    )
                )
            )

    analysis_service._executors["DetectSongSections"] = _CaptureSongSectionsExecutor()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "section-source.wav")
        runtime.add_song_from_path("Section Source", audio_path)
        presentation = runtime.extract_song_sections("source_audio")

        section_layer = next(layer for layer in presentation.layers if layer.title == "Sections")
        assert section_layer.kind is LayerKind.SECTION
        assert [event.label for event in section_layer.events] == ["Intro", "Chorus"]
        source_layer = next(
            layer
            for layer in presentation.layers
            if layer.object_id and str(layer.object_id).startswith("object_song_")
        )
        assert section_layer.status.source_layer_id == str(source_layer.layer_id)
        assert section_layer.source_content_ref is not None
        assert Path(str(section_layer.source_content_ref.locator)).exists()
        assert presentation.section_cues
        assert presentation.section_cues[0].cue_ref == "intro_01"
        rerun_presentation = runtime.extract_song_sections(section_layer.layer_id)
        rerun_section_layer = next(
            layer for layer in rerun_presentation.layers if layer.title == "Sections"
        )
        assert rerun_section_layer.kind is LayerKind.SECTION
        assert len(captured_audio_paths) == 2
        assert all(Path(path).exists() for path in captured_audio_paths)
        assert captured_audio_paths[0] == captured_audio_paths[1]
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_extract_note_contour_persists_child_event_layer():
    from echozero.pipelines.registry import get_registry
    from echozero.processors.detect_note_contour import DetectNoteContourProcessor, PitchFrame
    from echozero.processors.load_audio import LoadAudioProcessor
    from echozero.services.orchestrator import Orchestrator
    from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell

    temp_root = _repo_local_temp_root()
    analysis_service = Orchestrator(
        get_registry(),
        {
            "LoadAudio": LoadAudioProcessor(),
            "DetectNoteContour": DetectNoteContourProcessor(
                pitch_track_fn=lambda *args: [
                    PitchFrame(0.00, 65.406),
                    PitchFrame(0.05, 65.406),
                    PitchFrame(0.12, 82.407),
                    PitchFrame(0.18, 82.407),
                ]
            ),
        },
    )
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=analysis_service,
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "note-contour-source.wav")
        runtime.add_song_from_path("Note Contour Source", audio_path)
        presentation = runtime.extract_note_contour("source_audio")

        source_layer = next(
            layer
            for layer in presentation.layers
            if layer.object_id and str(layer.object_id).startswith("object_song_")
        )
        contour_layer = next(layer for layer in presentation.layers if layer.title == "Notes")
        assert contour_layer.status.source_layer_id == str(source_layer.layer_id)
        assert contour_layer.status.pipeline_id == "extract_note_contour"
        assert [event.label for event in contour_layer.events] == ["C2", "E2"]
        assert contour_layer.events[0].detection_metadata["midi_note"] == 36
        assert contour_layer.source_content_ref is not None
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


__all__ = [name for name in globals() if name.startswith("test_")]
