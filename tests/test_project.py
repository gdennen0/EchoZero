"""
Project class tests: lifecycle, graph mutations, execution, analysis, song management,
save/load roundtrip, and naming/identity.

Tests use Project.create() with real processors (LoadAudio, DetectOnsets) or mock executors,
and use tmp_path for isolation.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.events import (
    FoundryArtifactFinalizedEvent,
    FoundryArtifactValidatedEvent,
    FoundryRunCreatedEvent,
    FoundryRunStartedEvent,
)
from echozero.editor.commands import AddBlockCommand, ChangeBlockSettingsCommand
from echozero.persistence.audio import AudioMetadata
from echozero.persistence.session import ProjectStorage
from echozero.project import Project
from echozero.result import Ok, Err, is_ok, is_err, unwrap
from tests.foundry.audio_fixtures import write_percussion_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_scan(path) -> AudioMetadata:
    """Mock audio scanner that returns realistic metadata without reading real audio."""
    return AudioMetadata(duration_seconds=120.0, sample_rate=44100, channel_count=2)


def _create_audio_file(tmp_path: Path, name: str = "test.wav") -> Path:
    """Create a minimal fake audio file for import tests."""
    audio = tmp_path / name
    audio.write_bytes(b"RIFF" + b"\x00" * 100)
    return audio


def _make_add_block_command(block_id: str = None, name: str = "TestBlock") -> AddBlockCommand:
    """Create an AddBlockCommand with minimal port setup."""
    return AddBlockCommand(
        block_id=block_id or uuid.uuid4().hex,
        name=name,
        block_type="TestType",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(("out", "EVENT", "OUTPUT"),),
        control_ports=(),
        settings_entries=(),
    )


def _create_project(tmp_path: Path, name: str = "Test Project") -> Project:
    """Create a minimal project in tmp_path."""
    return Project.create(
        name=name,
        working_dir_root=tmp_path / "working",
    )


def _create_foundry_version_for_project(p: Project, tmp_path: Path, *, label_counts: dict[str, int] | None = None) -> Any:
    """Create and plan a tiny foundry dataset version for project-level run tests."""
    counts = label_counts or {"kick": 2, "snare": 2}
    dataset_root = tmp_path / f"foundry_dataset_{uuid.uuid4().hex[:8]}"
    write_percussion_dataset(dataset_root, sample_count=max(counts.values()), sample_rate=22050)

    ds_result = p.foundry_create_dataset("Drums")
    assert is_ok(ds_result)
    dataset = unwrap(ds_result)

    version_result = p.foundry_ingest_dataset_folder(dataset.id, dataset_root)
    assert is_ok(version_result)
    version = unwrap(version_result)

    assert is_ok(
        p.foundry_plan_dataset_version(
            version.id,
            validation_split=0.5,
            test_split=0.0,
            balance_strategy="none",
        )
    )
    return version


# ---------------------------------------------------------------------------
# 1. Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_create_returns_working_project(self, tmp_path):
        """Project.create() returns a live, usable Project."""
        p = _create_project(tmp_path)
        try:
            assert p is not None
            assert p.name == "Test Project"
            assert p.graph is not None
        finally:
            p.close()

    def test_open_db_reopens_created_project(self, tmp_path):
        """Project.open_db() can reopen a project created by create()."""
        working_root = tmp_path / "working"
        p = Project.create(name="ReopenMe", working_dir_root=working_root)
        working_dir = p.storage.working_dir
        p.close()

        p2 = Project.open_db(working_dir=working_dir)
        try:
            assert p2.name == "ReopenMe"
        finally:
            p2.close()

    def test_close_releases_storage(self, tmp_path):
        """close() marks storage as closed; subsequent operations should raise."""
        p = _create_project(tmp_path)
        p.close()
        # Storage should be closed
        assert p.storage._closed

    def test_context_manager(self, tmp_path):
        """Context manager calls close() on exit."""
        working_root = tmp_path / "working"
        with Project.create(name="CtxMgr", working_dir_root=working_root) as p:
            assert p.name == "CtxMgr"
        # After __exit__, storage should be closed
        assert p.storage._closed

    def test_close_stops_autosave(self, tmp_path):
        """close() stops the autosave timer."""
        p = _create_project(tmp_path)
        p.close()
        # autosave timer should be None after close
        assert p.storage._autosave_timer is None


# ---------------------------------------------------------------------------
# 2. Graph mutations
# ---------------------------------------------------------------------------


class TestGraphMutations:
    def test_dispatch_adds_block(self, tmp_path):
        """dispatch(AddBlockCommand) adds block to the graph."""
        with _create_project(tmp_path) as p:
            cmd = _make_add_block_command(block_id="b1", name="Alpha")
            result = p.dispatch(cmd)
            assert is_ok(result)
            assert "b1" in p.graph.blocks
            assert p.graph.blocks["b1"].name == "Alpha"

    def test_dispatch_persists_graph_on_success(self, tmp_path):
        """Successful dispatch() persists the graph to the DB (committed on close)."""
        working_root = tmp_path / "working"
        p = Project.create(name="PersistTest", working_dir_root=working_root)
        working_dir = p.storage.working_dir

        cmd = _make_add_block_command(block_id="b_persist")
        p.dispatch(cmd)
        p.close()  # close() commits pending changes

        # Reopen and verify block is present
        p2 = Project.open_db(working_dir=working_dir)
        try:
            assert "b_persist" in p2.graph.blocks
        finally:
            p2.close()

    def test_dispatch_failed_command_does_not_persist(self, tmp_path):
        """Failed dispatch() does not call save_graph (graph restored to prior state)."""
        with _create_project(tmp_path) as p:
            # Dispatch AddBlockCommand — should succeed
            cmd1 = _make_add_block_command(block_id="b_good")
            p.dispatch(cmd1)

            # Try to change settings on a nonexistent block — should fail
            cmd_bad = ChangeBlockSettingsCommand(
                block_id="nonexistent_block",
                setting_key="x",
                new_value=42,
            )
            result = p.dispatch(cmd_bad)
            assert is_err(result)
            # b_good should still be there, nonexistent_block shouldn't
            assert "b_good" in p.graph.blocks
            assert "nonexistent_block" not in p.graph.blocks

    def test_graph_survives_close_and_reopen(self, tmp_path):
        """Graph state (blocks) persists through close + open_db."""
        working_root = tmp_path / "working"
        p = Project.create(name="RoundTrip", working_dir_root=working_root)
        working_dir = p.storage.working_dir

        cmd = _make_add_block_command(block_id="b_survivor", name="SurvivorBlock")
        p.dispatch(cmd)
        p.close()

        p2 = Project.open_db(working_dir=working_dir)
        try:
            assert "b_survivor" in p2.graph.blocks
            assert p2.graph.blocks["b_survivor"].name == "SurvivorBlock"
        finally:
            p2.close()

    def test_multiple_dispatches_cumulate(self, tmp_path):
        """Multiple dispatches build up the graph correctly."""
        with _create_project(tmp_path) as p:
            for i in range(5):
                cmd = _make_add_block_command(block_id=f"b{i}", name=f"Block{i}")
                assert is_ok(p.dispatch(cmd))
            assert len(p.graph.blocks) == 5


# ---------------------------------------------------------------------------
# 3. Execution
# ---------------------------------------------------------------------------


class TestExecution:
    def test_run_executes_empty_graph(self, tmp_path):
        """run() on an empty graph succeeds (returns execution_id)."""
        with _create_project(tmp_path) as p:
            result = p.run()
            # Empty graph execution should succeed
            assert is_ok(result)

    def test_run_async_returns_execution_handle(self, tmp_path):
        """run_async() returns an ExecutionHandle."""
        from echozero.editor.coordinator import ExecutionHandle
        with _create_project(tmp_path) as p:
            result = p.run_async()
            assert is_ok(result)
            handle = unwrap(result)
            assert isinstance(handle, ExecutionHandle)
            # Wait for it to finish
            handle.future.result(timeout=5)

    def test_cancel_signals_cancellation(self, tmp_path):
        """cancel() sets the cancel event on the coordinator."""
        with _create_project(tmp_path) as p:
            p.cancel()
            assert p._coordinator._cancel_event.is_set()

    def test_is_executing_false_when_idle(self, tmp_path):
        """is_executing is False when no execution is in progress."""
        with _create_project(tmp_path) as p:
            assert p.is_executing is False

    def test_run_second_time_after_first_completes(self, tmp_path):
        """Can call run() multiple times in sequence."""
        with _create_project(tmp_path) as p:
            r1 = p.run()
            r2 = p.run()
            assert is_ok(r1)
            assert is_ok(r2)


# ---------------------------------------------------------------------------
# 4. Analysis
# ---------------------------------------------------------------------------


class TestAnalysis:
    def test_analyze_delegates_to_orchestrator(self, tmp_path):
        """analyze() delegates to Orchestrator and propagates the result."""
        with _create_project(tmp_path) as p:
            from echozero.services.orchestrator import AnalysisResult

            mock_result = AnalysisResult(
                song_version_id="sv1",
                pipeline_id="tmpl1",
                layer_ids=["l1"],
                take_ids=["t1"],
                duration_ms=100.0,
            )
            p._orchestrator.analyze = MagicMock(return_value=Ok(value=mock_result))

            result = p.analyze(
                song_version_id="sv1",
                template_id="tmpl1",
            )

            assert is_ok(result)
            assert unwrap(result) is mock_result
            p._orchestrator.analyze.assert_called_once_with(
                session=p._storage,
                song_version_id="sv1",
                pipeline_id="tmpl1",
                bindings=None,
                on_progress=None,
            )

    def test_execute_config_delegates_to_orchestrator(self, tmp_path):
        """execute_config() delegates to Orchestrator."""
        with _create_project(tmp_path) as p:
            from echozero.services.orchestrator import AnalysisResult

            mock_result = AnalysisResult(
                song_version_id="sv1",
                pipeline_id="tmpl1",
                layer_ids=[],
                take_ids=[],
                duration_ms=50.0,
            )
            p._orchestrator.execute = MagicMock(return_value=Ok(value=mock_result))

            result = p.execute_config(config_id="cfg1")

            assert is_ok(result)
            p._orchestrator.execute.assert_called_once_with(
                session=p._storage,
                config_id="cfg1",
                on_progress=None,
            )


# ---------------------------------------------------------------------------
# 5. Song management
# ---------------------------------------------------------------------------


class TestSongManagement:
    def test_import_song_creates_song_and_version(self, tmp_path):
        """import_song() creates a SongRecord and SongVersionRecord."""
        audio = _create_audio_file(tmp_path)
        with _create_project(tmp_path) as p:
            song, version = p.import_song(
                title="My Song",
                audio_source=audio,
                scan_fn=_mock_scan,
                default_templates=[],
            )
            assert song.title == "My Song"
            assert version.song_id == song.id
            assert version.duration_seconds == 120.0

    def test_add_song_version_adds_new_version(self, tmp_path):
        """add_song_version() adds a new version to an existing song."""
        audio1 = _create_audio_file(tmp_path, "song_v1.wav")
        audio2 = _create_audio_file(tmp_path, "song_v2.wav")
        with _create_project(tmp_path) as p:
            song, v1 = p.import_song(
                title="VersionedSong",
                audio_source=audio1,
                scan_fn=_mock_scan,
                default_templates=[],
            )
            v2 = p.add_song_version(
                song_id=song.id,
                audio_source=audio2,
                label="Remix",
                scan_fn=_mock_scan,
            )
            assert v2.song_id == song.id
            assert v2.label == "Remix"

    def test_songs_survive_close_and_reopen(self, tmp_path):
        """Songs survive close + reopen via open_db."""
        audio = _create_audio_file(tmp_path)
        working_root = tmp_path / "working"
        p = Project.create(name="SongPersist", working_dir_root=working_root)
        working_dir = p.storage.working_dir

        song, version = p.import_song(
            title="PersistentSong",
            audio_source=audio,
            scan_fn=_mock_scan,
            default_templates=[],
        )
        song_id = song.id
        p.close()

        p2 = Project.open_db(working_dir=working_dir)
        try:
            loaded_song = p2.songs.get(song_id)
            assert loaded_song is not None
            assert loaded_song.title == "PersistentSong"
        finally:
            p2.close()

    def test_repo_shortcuts_accessible(self, tmp_path):
        """songs, song_versions, layers, takes, pipeline_configs are accessible."""
        with _create_project(tmp_path) as p:
            # These should not raise
            _ = p.songs
            _ = p.song_versions
            _ = p.layers
            _ = p.takes
            _ = p.pipeline_configs


# ---------------------------------------------------------------------------
# 6. Save / load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_save_as_and_open(self, tmp_path):
        """Create project, add blocks, save_as .ez, close, open .ez, verify."""
        working_root = tmp_path / "working"
        ez_path = tmp_path / "project.ez"

        # Create and populate
        p = Project.create(name="ArchiveProject", working_dir_root=working_root)
        cmd = _make_add_block_command(block_id="b_archive", name="ArchiveBlock")
        p.dispatch(cmd)
        p.save_as(ez_path)
        p.close()

        assert ez_path.exists()

        # Reopen from .ez
        working_root2 = tmp_path / "working2"
        p2 = Project.open(ez_path=ez_path, working_dir_root=working_root2)
        try:
            assert p2.name == "ArchiveProject"
            assert "b_archive" in p2.graph.blocks
            assert p2.graph.blocks["b_archive"].name == "ArchiveBlock"
        finally:
            p2.close()

    def test_save_as_with_song_roundtrip(self, tmp_path):
        """save_as() preserves songs in the .ez archive."""
        audio = _create_audio_file(tmp_path)
        working_root = tmp_path / "working"
        ez_path = tmp_path / "with_song.ez"

        p = Project.create(name="SongArchive", working_dir_root=working_root)
        song, version = p.import_song(
            title="ArchiveSong",
            audio_source=audio,
            scan_fn=_mock_scan,
            default_templates=[],
        )
        song_id = song.id
        p.save_as(ez_path)
        p.close()

        working_root2 = tmp_path / "working2"
        p2 = Project.open(ez_path=ez_path, working_dir_root=working_root2)
        try:
            loaded_song = p2.songs.get(song_id)
            assert loaded_song is not None
            assert loaded_song.title == "ArchiveSong"
        finally:
            p2.close()

    def test_save_then_reopen_db(self, tmp_path):
        """save() flushes changes; open_db() restores them."""
        working_root = tmp_path / "working"
        p = Project.create(name="SaveReopen", working_dir_root=working_root)
        working_dir = p.storage.working_dir
        cmd = _make_add_block_command(block_id="b_saved", name="SavedBlock")
        p.dispatch(cmd)
        p.save()
        p.close()

        p2 = Project.open_db(working_dir=working_dir)
        try:
            assert "b_saved" in p2.graph.blocks
        finally:
            p2.close()


# ---------------------------------------------------------------------------
# 7. Naming / identity
# ---------------------------------------------------------------------------


class TestFoundryIntegration:
    def test_foundry_dataset_to_run_v1_flow(self, tmp_path):
        with _create_project(tmp_path) as p:
            version = _create_foundry_version_for_project(p, tmp_path)
            planned_result = p.foundry_plan_dataset_version(
                version.id,
                validation_split=0.5,
                test_split=0.0,
                balance_strategy="undersample_min",
            )
            assert is_ok(planned_result)

            run_spec = {
                "schema": "foundry.train_run_spec.v1",
                "classificationMode": "multiclass",
                "data": {
                    "datasetVersionId": version.id,
                    "sampleRate": 22050,
                    "maxLength": 22050,
                    "nFft": 2048,
                    "hopLength": 512,
                    "nMels": 128,
                    "fmax": 8000,
                },
                "training": {
                    "epochs": 1,
                    "batchSize": 2,
                    "learningRate": 0.001,
                    "seed": 17,
                },
            }
            run = unwrap(p.foundry_create_run(version.id, run_spec))
            run = unwrap(p.foundry_start_run(run.id))
            if run.status.value != "completed":
                run = unwrap(p.foundry_complete_run(run.id, metrics={"f1": 0.9}))
                assert run.status.value == "completed"
            unwrap(p.foundry_save_checkpoint(run.id, epoch=1, metric_snapshot={"loss": 0.1}))
            if run.status.value != "completed":
                run = unwrap(p.foundry_complete_run(run.id, metrics={"f1": 0.9}))

            eval_result = p.foundry_record_eval(
                run.id,
                classification_mode="multiclass",
                metrics={"accuracy": 0.9},
            )
            assert is_ok(eval_result)

            artifact_result = p.foundry_finalize_artifact_checked(
                run.id,
                {
                    "weightsPath": "exports/model.pth",
                    "classes": ["kick", "snare"],
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
            )
            assert is_ok(artifact_result)

    def test_foundry_publishes_lifecycle_events(self, tmp_path):
        with _create_project(tmp_path) as p:
            version = _create_foundry_version_for_project(p, tmp_path)
            assert version is not None

            seen: list[type] = []

            p.event_bus.subscribe(FoundryRunCreatedEvent, lambda e: seen.append(type(e)))
            p.event_bus.subscribe(FoundryRunStartedEvent, lambda e: seen.append(type(e)))
            p.event_bus.subscribe(FoundryArtifactFinalizedEvent, lambda e: seen.append(type(e)))
            p.event_bus.subscribe(FoundryArtifactValidatedEvent, lambda e: seen.append(type(e)))

            run_spec = {
                "schema": "foundry.train_run_spec.v1",
                "classificationMode": "multiclass",
                "data": {
                "datasetVersionId": version.id,
                    "sampleRate": 22050,
                    "maxLength": 22050,
                    "nFft": 2048,
                    "hopLength": 512,
                    "nMels": 128,
                    "fmax": 8000,
                },
                "training": {
                    "epochs": 1,
                    "batchSize": 2,
                    "learningRate": 0.001,
                    "seed": 17,
                },
            }

            run = unwrap(p.foundry_create_run(version.id, run_spec))
            unwrap(p.foundry_start_run(run.id))
            unwrap(
                p.foundry_finalize_artifact_checked(
                    run.id,
                    {
                        "weightsPath": "exports/model.pth",
                        "classes": ["kick", "snare"],
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
                )
            )

            assert FoundryRunCreatedEvent in seen
            assert FoundryRunStartedEvent in seen
            assert FoundryArtifactFinalizedEvent in seen
            assert FoundryArtifactValidatedEvent in seen

    def test_foundry_run_and_artifact_checked_gate_passes(self, tmp_path):
        with _create_project(tmp_path) as p:
            version = _create_foundry_version_for_project(p, tmp_path)
            assert version is not None

            run_spec = {
                "schema": "foundry.train_run_spec.v1",
                "classificationMode": "multiclass",
                "data": {
                    "datasetVersionId": version.id,
                    "sampleRate": 22050,
                    "maxLength": 22050,
                    "nFft": 2048,
                    "hopLength": 512,
                    "nMels": 128,
                    "fmax": 8000,
                },
                "training": {
                    "epochs": 1,
                    "batchSize": 2,
                    "learningRate": 0.001,
                    "seed": 17,
                },
            }

            run_result = p.foundry_create_run(version.id, run_spec)
            assert is_ok(run_result)
            run = unwrap(run_result)

            assert is_ok(p.foundry_start_run(run.id))

            artifact_result = p.foundry_finalize_artifact_checked(
                run.id,
                {
                    "weightsPath": "exports/model.pth",
                    "classes": ["kick", "snare"],
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
            )
            assert is_ok(artifact_result)

    def test_foundry_artifact_checked_gate_fails_on_missing_weights(self, tmp_path):
        with _create_project(tmp_path) as p:
            version = _create_foundry_version_for_project(p, tmp_path)
            assert version is not None

            run_spec = {
                "schema": "foundry.train_run_spec.v1",
                "classificationMode": "multiclass",
                "data": {
                    "datasetVersionId": version.id,
                    "sampleRate": 22050,
                    "maxLength": 22050,
                    "nFft": 2048,
                    "hopLength": 512,
                    "nMels": 128,
                    "fmax": 8000,
                },
                "training": {
                    "epochs": 1,
                    "batchSize": 2,
                    "learningRate": 0.001,
                    "seed": 17,
                },
            }

            run = unwrap(p.foundry_create_run(version.id, run_spec))
            unwrap(p.foundry_start_run(run.id))

            artifact_result = p.foundry_finalize_artifact_checked(
                run.id,
                {
                    "classes": ["kick", "snare"],
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
            )
            assert is_err(artifact_result)


class TestNamingIdentity:
    def test_project_name(self, tmp_path):
        """project.name returns the project name."""
        with _create_project(tmp_path, name="My Awesome Project") as p:
            assert p.name == "My Awesome Project"

    def test_project_graph_property(self, tmp_path):
        """project.graph returns the Graph instance."""
        from echozero.domain.graph import Graph
        with _create_project(tmp_path) as p:
            assert isinstance(p.graph, Graph)

    def test_project_is_dirty_false_initially(self, tmp_path):
        """is_dirty is False on a fresh project (no uncommitted changes)."""
        with _create_project(tmp_path) as p:
            # After create, no uncommitted changes yet
            assert p.is_dirty is False

    def test_project_is_dirty_after_dispatch(self, tmp_path):
        """is_dirty reflects dirty state after dispatch (save_graph marks dirty via DirtyTracker)."""
        with _create_project(tmp_path) as p:
            # Dispatch sets dirty via save_graph → mark_dirty
            cmd = _make_add_block_command(block_id="b_dirty")
            p.dispatch(cmd)
            assert p.is_dirty is True

    def test_is_dirty_clears_after_save(self, tmp_path):
        """is_dirty is False after calling save()."""
        with _create_project(tmp_path) as p:
            cmd = _make_add_block_command(block_id="b_to_save")
            p.dispatch(cmd)
            p.save()
            assert p.is_dirty is False

    def test_storage_property(self, tmp_path):
        """project.storage returns the ProjectStorage."""
        with _create_project(tmp_path) as p:
            assert isinstance(p.storage, ProjectStorage)

    def test_event_bus_property(self, tmp_path):
        """project.event_bus returns the EventBus."""
        from echozero.event_bus import EventBus
        with _create_project(tmp_path) as p:
            assert isinstance(p.event_bus, EventBus)

    def test_stale_tracker_property(self, tmp_path):
        """project.stale_tracker returns the StaleTracker."""
        from echozero.editor.staleness import StaleTracker
        with _create_project(tmp_path) as p:
            assert isinstance(p.stale_tracker, StaleTracker)
