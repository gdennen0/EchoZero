"""Project and shell-contract runtime-flow support cases.
Exists to keep project-open, song-switching, and shell contract coverage separate from settings and pipeline behavior.
Connects the compatibility wrapper to the bounded project support slice.
"""

import echozero.ui.qt.app_shell_project_lifecycle as project_lifecycle
from echozero.application.timeline.ma3_push_intents import SetLayerMA3Route
from echozero.persistence.audio import AudioMetadata, PreparedAudioSource, compute_audio_hash

from tests.ui.app_shell_runtime_flow_shared_support import *  # noqa: F401,F403

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


def test_app_shell_runtime_persists_take_lane_expansion_state_across_save_and_open():
    temp_root = _repo_local_temp_root()
    working_root = temp_root / "working"
    save_path = temp_root / "lane-state.ez"

    runtime = build_app_shell(
        working_dir_root=working_root,
        initial_project_name="Lane State Runtime",
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio = write_test_wav(temp_root / "fixtures" / "lane-state.wav")
        runtime.add_song_from_path("Lane State Song", audio)
        runtime.add_layer(LayerKind.EVENT, "Lane Manual")
        layer = next(
            lane for lane in runtime.presentation().layers if lane.title == "Lane Manual"
        )
        layer_id = layer.layer_id

        assert layer.is_expanded is False

        runtime.dispatch(ToggleLayerExpanded(layer_id))
        expanded_lane = next(
            lane for lane in runtime.presentation().layers if lane.layer_id == layer_id
        )
        assert expanded_lane.is_expanded is True

        runtime.save_project_as(save_path)
        runtime.open_project(save_path)
        reopened_expanded = next(
            lane for lane in runtime.presentation().layers if lane.title == "Lane Manual"
        )
        assert reopened_expanded.is_expanded is True
        layer_id = reopened_expanded.layer_id

        runtime.dispatch(ToggleLayerExpanded(layer_id))
        collapsed_lane = next(
            lane for lane in runtime.presentation().layers if lane.layer_id == layer_id
        )
        assert collapsed_lane.is_expanded is False

        runtime.save_project()
        runtime.open_project(save_path)
        reopened_collapsed = next(
            lane for lane in runtime.presentation().layers if lane.title == "Lane Manual"
        )
        assert reopened_collapsed.is_expanded is False
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_open_project_replaces_live_project_state():
    temp_root = _repo_local_temp_root()
    working_root = temp_root / "working"
    project_a_path = temp_root / "project-a.ez"
    project_b_path = temp_root / "project-b.ez"

    runtime = build_app_shell(
        working_dir_root=working_root,
        initial_project_name="Project A",
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_a = write_test_wav(temp_root / "fixtures" / "project-a.wav", frames=4410)
        runtime.add_song_from_path("Song A", audio_a)
        layer_a = runtime.presentation().layers[0].layer_id
        runtime.dispatch(ToggleLayerExpanded(layer_a))
        runtime.save_project_as(project_a_path)
        project_a_id = runtime.project_storage.project.id

        runtime.new_project("Project B")
        audio_b = write_test_wav(temp_root / "fixtures" / "project-b.wav", frames=8820)
        runtime.add_song_from_path("Song B", audio_b)
        layer_b = runtime.presentation().layers[0].layer_id
        runtime.dispatch(ToggleLayerExpanded(layer_b))
        runtime.save_project_as(project_b_path)
        live_storage = runtime.project_storage
        project_b_id = live_storage.project.id

        runtime.open_project(project_a_path)

        assert live_storage._closed is True
        assert runtime.project_path == project_a_path
        assert runtime.project_storage.project.id != project_b_id
        assert runtime.project_storage.project.id == project_a_id
        assert runtime.project_storage._closed is False
        assert runtime.project_storage.working_dir.exists()
        assert runtime.presentation().title == "Project A"
        assert runtime.presentation().layers[0].title == "Song A"
        assert runtime.presentation().end_time_label == "00:00.10"
        assert runtime.session.project_id == runtime.project_storage.project.id
        assert runtime.session.active_song_id is not None
        assert runtime.session.active_song_version_id is not None
        assert runtime.session.active_timeline_id == runtime.presentation().timeline_id
        assert runtime.is_dirty is False
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_persists_layer_ma3_route_across_save_and_open():
    temp_root = _repo_local_temp_root()
    working_root = temp_root / "working"
    save_path = temp_root / "ma3-route-state.ez"

    runtime = build_app_shell(
        working_dir_root=working_root,
        initial_project_name="MA3 Route Persistence",
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio = write_test_wav(temp_root / "fixtures" / "ma3-route-state.wav")
        runtime.add_song_from_path("MA3 Route Song", audio)
        runtime.add_layer(LayerKind.EVENT, "Route Me")
        route_layer = next(layer for layer in runtime.presentation().layers if layer.title == "Route Me")

        runtime.dispatch(
            SetLayerMA3Route(
                layer_id=route_layer.layer_id,
                target_track_coord="tc1_tg2_tr7",
            )
        )
        routed = next(
            layer for layer in runtime.presentation().layers if layer.layer_id == route_layer.layer_id
        )
        assert routed.sync_target_label == "tc1_tg2_tr7"

        runtime.save_project_as(save_path)
        runtime.open_project(save_path)

        reloaded = next(layer for layer in runtime.presentation().layers if layer.title == "Route Me")
        assert reloaded.sync_target_label == "tc1_tg2_tr7"
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_import_smpte_audio_to_layer_uses_extracted_ltc_when_available(
    monkeypatch,
):
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        initial_project_name="SMPTE Import",
    )

    try:
        source_song = write_test_wav(temp_root / "fixtures" / "smpte-import-song.wav")
        runtime.add_song_from_path("SMPTE Import Song", source_song)
        runtime.add_layer(LayerKind.AUDIO, "SMPTE Layer")
        smpte_layer = next(
            layer for layer in runtime.presentation().layers if layer.title == "SMPTE Layer"
        )
        assert smpte_layer.source_audio_path is None

        printed_dual_track = Path(
            write_test_wav(temp_root / "fixtures" / "printed-dual-track.wav")
        )
        extracted_ltc = Path(
            write_test_wav(temp_root / "fixtures" / "extracted-ltc.wav")
        )

        scanned_paths: list[Path] = []
        cleanup_calls: list[PreparedAudioSource] = []

        def _fake_prepare(
            source_path: Path,
            working_dir: Path,
            *,
            options=None,
            scan_fn=None,
        ) -> PreparedAudioSource:
            del working_dir, scan_fn
            assert source_path == printed_dual_track.resolve()
            assert options is not None and bool(options.strip_ltc_timecode)
            assert options.ltc_detection_mode == "aggressive"
            return PreparedAudioSource(
                source_path=source_path,
                ltc_artifact_path=extracted_ltc,
            )

        def _fake_scan(path: Path, scan_fn=None) -> AudioMetadata:
            del scan_fn
            scanned_paths.append(path)
            return AudioMetadata(duration_seconds=11.0, sample_rate=48000, channel_count=1)

        monkeypatch.setattr(
            "echozero.ui.qt.app_shell_editing_mixin.prepare_audio_for_import",
            _fake_prepare,
        )
        monkeypatch.setattr(
            "echozero.ui.qt.app_shell_editing_mixin.scan_audio_metadata",
            _fake_scan,
        )
        monkeypatch.setattr(
            "echozero.ui.qt.app_shell_editing_mixin.cleanup_prepared_audio",
            lambda prepared: cleanup_calls.append(prepared),
        )

        updated = runtime.import_smpte_audio_to_layer(
            str(smpte_layer.layer_id),
            str(printed_dual_track),
        )
        reloaded_layer = next(
            layer for layer in updated.layers if layer.layer_id == smpte_layer.layer_id
        )

        expected_hash_prefix = compute_audio_hash(extracted_ltc)[:16]
        persisted_take = runtime.project_storage.takes.list_by_layer(str(smpte_layer.layer_id))[0]
        assert isinstance(persisted_take.data, AudioData)
        assert Path(str(persisted_take.data.file_path)).name == f"{expected_hash_prefix}.wav"
        assert scanned_paths == [extracted_ltc]
        assert len(cleanup_calls) == 1
        assert reloaded_layer.source_audio_path is not None
        assert reloaded_layer.source_audio_path.endswith(f"{expected_hash_prefix}.wav")
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_import_smpte_audio_to_layer_uses_source_when_ltc_not_detected(
    monkeypatch,
):
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        initial_project_name="SMPTE Import Fallback",
    )

    try:
        source_song = write_test_wav(temp_root / "fixtures" / "smpte-fallback-song.wav")
        runtime.add_song_from_path("SMPTE Fallback Song", source_song)
        runtime.add_layer(LayerKind.AUDIO, "SMPTE Layer")
        smpte_layer = next(
            layer for layer in runtime.presentation().layers if layer.title == "SMPTE Layer"
        )

        smpte_source = Path(
            write_test_wav(temp_root / "fixtures" / "standalone-smpte.wav")
        )
        scanned_paths: list[Path] = []

        def _fake_prepare(
            source_path: Path,
            working_dir: Path,
            *,
            options=None,
            scan_fn=None,
        ) -> PreparedAudioSource:
            del working_dir, scan_fn
            assert source_path == smpte_source.resolve()
            assert options is not None and bool(options.strip_ltc_timecode)
            assert options.ltc_detection_mode == "aggressive"
            return PreparedAudioSource(source_path=source_path)

        def _fake_scan(path: Path, scan_fn=None) -> AudioMetadata:
            del scan_fn
            scanned_paths.append(path)
            return AudioMetadata(duration_seconds=9.5, sample_rate=44100, channel_count=1)

        monkeypatch.setattr(
            "echozero.ui.qt.app_shell_editing_mixin.prepare_audio_for_import",
            _fake_prepare,
        )
        monkeypatch.setattr(
            "echozero.ui.qt.app_shell_editing_mixin.scan_audio_metadata",
            _fake_scan,
        )

        runtime.import_smpte_audio_to_layer(str(smpte_layer.layer_id), str(smpte_source))

        expected_hash_prefix = compute_audio_hash(smpte_source)[:16]
        persisted_take = runtime.project_storage.takes.list_by_layer(str(smpte_layer.layer_id))[0]
        assert isinstance(persisted_take.data, AudioData)
        assert Path(str(persisted_take.data.file_path)).name == f"{expected_hash_prefix}.wav"
        assert scanned_paths == [smpte_source.resolve()]
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_add_smpte_layer_from_import_split_uses_matching_ltc_artifact():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        initial_project_name="SMPTE Import Split Action",
    )

    try:
        source_song = Path(write_test_wav(temp_root / "fixtures" / "split-action-source.wav"))
        runtime.add_song_from_path("Split Action Song", source_song)

        active_song_version_id = runtime.session.active_song_version_id
        assert active_song_version_id is not None
        version_record = runtime.project_storage.song_versions.get(str(active_song_version_id))
        assert version_record is not None

        split_dir = runtime.project_storage.working_dir / "audio" / "split_channels"
        split_dir.mkdir(parents=True, exist_ok=True)
        split_prefix = "1122334455667788"
        program_artifact = split_dir / f"{split_prefix}_program_left.wav"
        ltc_artifact = split_dir / f"{split_prefix}_ltc_right.wav"
        shutil.copy2(source_song, program_artifact)
        write_test_wav(ltc_artifact)

        assert compute_audio_hash(program_artifact) == version_record.audio_hash

        updated = runtime.add_smpte_layer_from_import_split()

        smpte_layer = next(layer for layer in updated.layers if layer.title == "SMPTE Layer")
        persisted_take = runtime.project_storage.takes.list_by_layer(str(smpte_layer.layer_id))[0]
        assert isinstance(persisted_take.data, AudioData)

        expected_hash_prefix = compute_audio_hash(ltc_artifact)[:16]
        assert Path(str(persisted_take.data.file_path)).name == f"{expected_hash_prefix}.wav"
        assert smpte_layer.source_audio_path is not None
        assert smpte_layer.source_audio_path.endswith(f"{expected_hash_prefix}.wav")
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_open_project_failure_keeps_current_project_live(monkeypatch):
    temp_root = _repo_local_temp_root()
    working_root = temp_root / "working"
    runtime = build_app_shell(
        working_dir_root=working_root,
        initial_project_name="Still Open",
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "still-open.wav", frames=4410)
        runtime.add_song_from_path("Current Song", audio_path)
        original_storage = runtime.project_storage
        original_project_id = original_storage.project.id
        original_presentation = runtime.presentation()

        def _raise_open(*_args, **_kwargs):
            raise RuntimeError("archive exploded")

        monkeypatch.setattr(project_lifecycle.ProjectStorage, "open", _raise_open)

        with pytest.raises(RuntimeError, match="archive exploded"):
            runtime.open_project(temp_root / "broken.ez")

        assert runtime.project_storage is original_storage
        assert runtime.project_storage._closed is False
        assert runtime.project_storage.project.id == original_project_id
        assert runtime.presentation().title == original_presentation.title
        assert runtime.presentation().layers[0].title == "Current Song"
        assert runtime.presentation().end_time_label == "00:00.10"
        assert runtime.session.project_id == original_project_id
        assert runtime.session.active_song_id is not None
        assert runtime.session.active_song_version_id is not None

        runtime.dispatch(ToggleLayerExpanded(runtime.presentation().layers[0].layer_id))
        assert runtime.project_storage._closed is False
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
        assert "add_event_layer" in empty_action_ids
        assert "add_section_layer" in empty_action_ids
        assert "add_automation_layer" not in empty_action_ids
        assert "add_reference_layer" not in empty_action_ids

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
    from dataclasses import replace

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
        version_1_record = runtime.project_storage.song_versions.get(version_1_id)
        assert version_1_record is not None
        runtime.project_storage.song_versions.update(
            replace(version_1_record, ma3_timecode_pool_no=113)
        )
        runtime.project_storage.commit()
        runtime._refresh_from_storage(
            active_song_id=runtime.session.active_song_id,
            active_song_version_id=runtime.session.active_song_version_id,
        )
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
        assert version_2_record.ma3_timecode_pool_no == 113
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


def test_app_shell_runtime_add_song_version_can_transfer_selected_layers():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Versioned Song",
            write_test_wav(temp_root / "fixtures" / "version-1.wav", frames=4410),
        )
        runtime.add_layer(LayerKind.EVENT, "Kick Events")
        runtime.add_layer(LayerKind.EVENT, "Snare Events")
        assert runtime.session.active_song_id is not None
        assert runtime.session.active_song_version_id is not None

        song_id = str(runtime.session.active_song_id)
        version_1_id = str(runtime.session.active_song_version_id)
        source_layers = runtime.project_storage.layers.list_by_version(version_1_id)
        snare_layer = next(layer for layer in source_layers if layer.name == "Snare Events")

        runtime.add_song_version(
            song_id,
            write_test_wav(temp_root / "fixtures" / "version-2.wav", frames=8820),
            label="Transfer Edit",
            transfer_layers=True,
            transfer_layer_ids=[snare_layer.id],
        )

        version_2_id = str(runtime.session.active_song_version_id)
        transferred_layers = runtime.project_storage.layers.list_by_version(version_2_id)
        assert [layer.name for layer in transferred_layers] == ["Snare Events"]
        assert [layer.title for layer in runtime.presentation().layers] == [
            "Versioned Song",
            "Snare Events",
        ]
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


def test_app_shell_runtime_scopes_pipeline_progress_to_active_song_version():
    temp_root = _repo_local_temp_root()
    analysis_service = build_mock_analysis_service()
    gate = threading.Event()
    original_execute = analysis_service.execute

    def _blocking_execute(session, config_id, runtime_bindings=None, on_progress=None):
        if on_progress is not None:
            on_progress(
                OperationProgressUpdate(
                    stage="loading_configuration",
                    message="Loading configuration",
                    fraction_complete=0.0,
                )
            )
            on_progress(
                OperationProgressUpdate(
                    stage="preparing_pipeline",
                    message="Preparing pipeline",
                    fraction_complete=0.1,
                )
            )
            on_progress(
                OperationProgressUpdate(
                    stage="executing_pipeline",
                    message="Executing pipeline",
                    fraction_complete=0.2,
                )
            )
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

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Song One",
            write_test_wav(temp_root / "fixtures" / "progress-song-1.wav", frames=4410),
        )
        song_1_id = str(runtime.session.active_song_id)
        version_1_id = str(runtime.session.active_song_version_id)

        runtime.add_song_from_path(
            "Song Two",
            write_test_wav(temp_root / "fixtures" / "progress-song-2.wav", frames=8820),
        )
        song_2_id = str(runtime.session.active_song_id)
        version_2_id = str(runtime.session.active_song_version_id)

        runtime.select_song(song_1_id)
        run_1_id = runtime.request_object_action_run(
            "timeline.extract_stems",
            object_id="source_audio",
            object_type="layer",
        )
        assert _wait_until(
            lambda: (
                runtime.get_operation_state(run_1_id) is not None
                and runtime.get_operation_state(run_1_id).status in {"resolving", "running"}
            )
        )

        banner = runtime.presentation().operation_progress_banner
        assert banner is not None
        assert banner.operation_id == run_1_id
        assert runtime.presentation().active_song_version_id == version_1_id

        runtime.select_song(song_2_id)

        assert runtime.presentation().active_song_version_id == version_2_id
        assert runtime.presentation().operation_progress_banner is None
        song_2_plan = runtime.describe_object_action(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )
        assert song_2_plan.is_running is False
        assert song_2_plan.operation_id is None

        run_2_id = runtime.request_object_action_run(
            "timeline.extract_stems",
            object_id="source_audio",
            object_type="layer",
        )
        assert _wait_until(
            lambda: (
                runtime.get_operation_state(run_2_id) is not None
                and runtime.get_operation_state(run_2_id).status in {"resolving", "running"}
            )
        )

        song_2_banner = runtime.presentation().operation_progress_banner
        assert song_2_banner is not None
        assert song_2_banner.operation_id == run_2_id
        song_2_plan = runtime.describe_object_action(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )
        assert song_2_plan.is_running is True
        assert song_2_plan.operation_id == run_2_id

        runtime.select_song(song_1_id)

        song_1_banner = runtime.presentation().operation_progress_banner
        assert song_1_banner is not None
        assert song_1_banner.operation_id == run_1_id
        song_1_plan = runtime.describe_object_action(
            "timeline.extract_stems",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )
        assert song_1_plan.is_running is True
        assert song_1_plan.operation_id == run_1_id

        gate.set()
        assert runtime.wait_for_operation(run_1_id, timeout=5.0).status == "completed"
        assert runtime.wait_for_operation(run_2_id, timeout=5.0).status == "completed"
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_delete_song_switches_to_neighbor_and_clears_empty_project():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Song One",
            write_test_wav(temp_root / "fixtures" / "delete-song-1.wav", frames=4410),
        )
        song_1_id = str(runtime.session.active_song_id)

        runtime.add_song_from_path(
            "Song Two",
            write_test_wav(temp_root / "fixtures" / "delete-song-2.wav", frames=8820),
        )
        song_2_id = str(runtime.session.active_song_id)

        presentation = runtime.delete_song(song_2_id)

        assert str(runtime.session.active_song_id) == song_1_id
        assert presentation.layers[0].title == "Song One"
        assert runtime.project_storage.songs.get(song_2_id) is None

        emptied = runtime.delete_song(song_1_id)

        assert runtime.session.active_song_id is None
        assert runtime.session.active_song_version_id is None
        assert emptied.layers == []
        assert runtime.project_storage.songs.list_by_project(runtime.project_storage.project.id) == []
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_delete_song_version_switches_versions_and_deletes_last_song_version():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Versioned Song",
            write_test_wav(temp_root / "fixtures" / "delete-version-1.wav", frames=4410),
        )
        song_id = str(runtime.session.active_song_id)
        version_1_id = str(runtime.session.active_song_version_id)

        runtime.add_song_version(
            song_id,
            write_test_wav(temp_root / "fixtures" / "delete-version-2.wav", frames=8820),
            label="Festival Edit",
        )
        version_2_id = str(runtime.session.active_song_version_id)

        switched = runtime.delete_song_version(version_2_id)

        assert str(runtime.session.active_song_id) == song_id
        assert str(runtime.session.active_song_version_id) == version_1_id
        assert switched.active_song_version_id == version_1_id
        assert runtime.project_storage.song_versions.get(version_2_id) is None

        emptied = runtime.delete_song_version(version_1_id)

        assert runtime.session.active_song_id is None
        assert runtime.session.active_song_version_id is None
        assert emptied.layers == []
        assert runtime.project_storage.songs.get(song_id) is None
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_move_song_and_reorder_songs_update_setlist_order():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Song One",
            write_test_wav(temp_root / "fixtures" / "setlist-move-1.wav", frames=4410),
        )
        song_1_id = str(runtime.session.active_song_id)
        runtime.add_song_from_path(
            "Song Two",
            write_test_wav(temp_root / "fixtures" / "setlist-move-2.wav", frames=4410),
        )
        song_2_id = str(runtime.session.active_song_id)
        runtime.add_song_from_path(
            "Song Three",
            write_test_wav(temp_root / "fixtures" / "setlist-move-3.wav", frames=4410),
        )
        song_3_id = str(runtime.session.active_song_id)

        moved = runtime.move_song(song_3_id, steps=-2)
        assert [song.song_id for song in moved.available_songs] == [song_3_id, song_1_id, song_2_id]

        reordered = runtime.reorder_songs([song_2_id, song_3_id, song_1_id])
        assert [song.song_id for song in reordered.available_songs] == [song_2_id, song_3_id, song_1_id]
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_widget_contract_switches_song_and_song_version():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(
        working_dir_root=temp_root / "working",
        initial_project_name="Song Switching App Path",
    )

    try:
        harness.runtime.add_song_from_path(
            "Song One",
            write_test_wav(temp_root / "fixtures" / "song-one.wav", frames=4410),
        )
        song_1_id = str(harness.runtime.session.active_song_id)
        version_1_id = str(harness.runtime.session.active_song_version_id)
        harness.runtime.add_song_version(
            song_1_id,
            write_test_wav(temp_root / "fixtures" / "song-one-festival.wav", frames=8820),
            label="Festival Edit",
        )
        version_2_id = str(harness.runtime.session.active_song_version_id)
        harness.runtime.add_song_from_path(
            "Song Two",
            write_test_wav(temp_root / "fixtures" / "song-two.wav", frames=13230),
        )
        harness.widget.set_presentation(harness.runtime.presentation())

        select_action = next(
            action
            for section in build_timeline_inspector_contract(
                harness.widget.presentation
            ).context_sections
            for action in section.actions
            if action.action_id == "song.select"
        )
        harness.widget._trigger_contract_action(
            replace(select_action, params={**select_action.params, "song_id": song_1_id})
        )

        assert str(harness.runtime.session.active_song_id) == song_1_id
        assert str(harness.runtime.session.active_song_version_id) == version_2_id
        assert harness.runtime.presentation().layers[0].title == "Song One"
        assert harness.runtime.presentation().end_time_label == "00:00.20"

        version_action = next(
            action
            for section in build_timeline_inspector_contract(
                harness.widget.presentation
            ).context_sections
            for action in section.actions
            if action.action_id == "song.version.switch"
        )
        harness.widget._trigger_contract_action(
            replace(
                version_action,
                params={**version_action.params, "song_version_id": version_1_id},
            )
        )

        assert str(harness.runtime.session.active_song_version_id) == version_1_id
        assert harness.runtime.presentation().layers[0].title == "Song One"
        assert harness.runtime.presentation().end_time_label == "00:00.10"
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_open_project_preserves_selected_take_when_still_valid():
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

        runtime.dispatch(SelectTake(layer_id=drums_layer.layer_id, take_id=drums_take.take_id))
        before_reload = runtime.presentation()
        assert before_reload.selected_layer_id == second_pass.selected_layer_id
        assert before_reload.selected_layer_id == drums_layer.layer_id
        assert before_reload.selected_take_id == drums_take.take_id

        runtime.save_project_as(save_path)
        runtime.open_project(save_path)
        reloaded = runtime.presentation()

        assert reloaded.selected_layer_id == before_reload.selected_layer_id
        assert reloaded.selected_layer_id == before_reload.selected_layer_id
        assert reloaded.selected_take_id == before_reload.selected_take_id
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_open_project_restores_saved_song_view_and_playhead_state():
    temp_root = _repo_local_temp_root()
    working_root = temp_root / "working"
    save_path = temp_root / "restore-runtime-state.ez"
    runtime = build_app_shell(
        working_dir_root=working_root,
        initial_project_name="Runtime State Restore",
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Song One",
            write_test_wav(temp_root / "fixtures" / "restore-state-1.wav", frames=441000),
        )
        song_1_id = str(runtime.session.active_song_id)
        runtime.add_song_from_path(
            "Song Two",
            write_test_wav(temp_root / "fixtures" / "restore-state-2.wav", frames=882000),
        )
        song_2_id = str(runtime.session.active_song_id)
        assert song_2_id != song_1_id

        runtime.dispatch(Seek(6.25))
        runtime._app.timeline.viewport.pixels_per_second = 180.0
        runtime._app.timeline.viewport.scroll_x = 720.0
        runtime._app.timeline.viewport.scroll_y = 0.0

        runtime.save_project_as(save_path)
        runtime.open_project(save_path)

        reloaded = runtime.presentation()
        assert str(runtime.session.active_song_id) == song_2_id
        assert reloaded.active_song_id == song_2_id
        assert runtime.session.transport_state.playhead == pytest.approx(6.25)
        assert reloaded.playhead == pytest.approx(6.25)
        assert reloaded.pixels_per_second == pytest.approx(180.0)
        assert reloaded.scroll_x == pytest.approx(720.0)
        assert reloaded.scroll_y == pytest.approx(0.0)
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_open_project_clears_stale_selected_event_refs():
    temp_root = _repo_local_temp_root()
    working_root = temp_root / "working"
    save_path = temp_root / "clear-selection-on-open.ez"
    runtime = build_app_shell(
        working_dir_root=working_root,
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Clear Selection Song",
            write_test_wav(temp_root / "fixtures" / "clear-selection-open.wav"),
        )
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

        runtime.save_project_as(save_path)
        runtime.open_project(save_path)
        reloaded = next(
            layer for layer in runtime.presentation().layers if layer.title == "Onsets"
        )

        assert runtime.presentation().selected_event_ids == []
        assert runtime.presentation().selected_event_refs == []
        assert all(event.is_selected is False for event in reloaded.events)
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_refresh_repairs_missing_selection_to_baseline_layer():
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

        runtime.dispatch(SelectTake(layer_id=drums_layer.layer_id, take_id=drums_take.take_id))
        version_2 = runtime.add_song_version(
            str(runtime.session.active_song_id),
            write_test_wav(temp_root / "fixtures" / "repair-target-2.wav", frames=8820),
            label="Blank V2",
        )

        assert version_2.layers[0].title == "Repair Target Song"
        assert version_2.selected_layer_id == version_2.layers[0].layer_id
        assert version_2.selected_take_id is None
        assert version_2.selected_layer_id == version_2.layers[0].layer_id
        assert len(version_2.layers) == 1
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)



__all__ = [name for name in globals() if name.startswith("test_")]
