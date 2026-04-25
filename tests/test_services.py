"""
Tests for the services layer: Orchestrator and SetlistProcessor.
Exists because the services layer bridges engine execution and persistence — tests
verify the full orchestration from template lookup through pipeline execution to
layer/take persistence, using mock executors for isolation from real audio.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

import echozero.pipelines.templates  # noqa: F401 — register templates
from echozero.domain.types import AudioData, Event, EventData, Layer
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.persistence.entities import PipelineConfigRecord, SongRecord, SongVersionRecord
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.registry import get_registry
from echozero.result import Err, Ok, err, ok
from echozero.services.orchestrator import AnalysisResult, Orchestrator
from echozero.services.setlist import SetlistProcessor, SetlistResult


# ---------------------------------------------------------------------------
# Mock executors
# ---------------------------------------------------------------------------


class MockLoadAudioExecutor:
    """Returns fake AudioData without reading a file."""

    def execute(self, block_id: str, context: ExecutionContext) -> Any:
        return ok(AudioData(
            sample_rate=44100,
            duration=180.0,
            file_path="test.wav",
            channel_count=2,
        ))


class MockDetectOnsetsExecutor:
    """Returns fake EventData with some test events."""

    def execute(self, block_id: str, context: ExecutionContext) -> Any:
        return ok(EventData(
            layers=(
                Layer(
                    id="onsets_layer",
                    name="onsets",
                    events=(
                        Event(id="e1", time=1.0, duration=0.1,
                              classifications={"type": "onset"}, metadata={},
                              origin="pipeline"),
                        Event(id="e2", time=2.5, duration=0.1,
                              classifications={"type": "onset"}, metadata={},
                              origin="pipeline"),
                        Event(id="e3", time=4.0, duration=0.1,
                              classifications={"type": "onset"}, metadata={},
                              origin="pipeline"),
                    ),
                ),
            ),
        ))


class SettingsCapturingExecutor:
    """Mock detect_onsets executor that captures block settings for verification."""

    def __init__(self) -> None:
        self.captured_settings: dict[str, Any] | None = None

    def execute(self, block_id: str, context: ExecutionContext) -> Any:
        block = context.graph.blocks.get(block_id)
        if block:
            self.captured_settings = dict(block.settings)
        return ok(EventData(
            layers=(Layer(id="onsets_layer", name="onsets", events=()),),
        ))


class FailingExecutor:
    """Always returns an error."""

    def execute(self, block_id: str, context: ExecutionContext) -> Any:
        return err(ExecutionError("Simulated failure"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_executors() -> dict[str, Any]:
    """Return mock executors keyed by block type (matching onset_detection template)."""
    return {
        "LoadAudio": MockLoadAudioExecutor(),
        "DetectOnsets": MockDetectOnsetsExecutor(),
    }


def _create_session_with_song(
    tmp_path: Any,
    audio_file: str = "/path/to/test.wav",
) -> tuple[ProjectStorage, SongRecord, SongVersionRecord]:
    """Create a ProjectStorage with one song and one version."""
    session = ProjectStorage.create_new("Test ProjectRecord", working_dir_root=tmp_path)
    now = datetime.now(timezone.utc)

    song = SongRecord(
        id=uuid.uuid4().hex,
        project_id=session.project.id,
        title="Test SongRecord",
        artist="Test Artist",
        order=0,
    )
    session.songs.create(song)

    version = SongVersionRecord(
        id=uuid.uuid4().hex,
        song_id=song.id,
        label="Studio Mix",
        audio_file=audio_file,
        duration_seconds=180.0,
        original_sample_rate=44100,
        audio_hash="abc123",
        created_at=now,
    )
    session.song_versions.create(version)
    session.commit()

    return session, song, version


def _create_session_with_songs(
    tmp_path: Any,
    count: int = 3,
) -> tuple[ProjectStorage, list[SongRecord], list[SongVersionRecord]]:
    """Create a ProjectStorage with multiple songs and versions."""
    session = ProjectStorage.create_new("Test ProjectRecord", working_dir_root=tmp_path)
    now = datetime.now(timezone.utc)
    songs: list[SongRecord] = []
    versions: list[SongVersionRecord] = []

    for i in range(count):
        song = SongRecord(
            id=uuid.uuid4().hex,
            project_id=session.project.id,
            title=f"SongRecord {i + 1}",
            artist="Test Artist",
            order=i,
        )
        session.songs.create(song)
        songs.append(song)

        version = SongVersionRecord(
            id=uuid.uuid4().hex,
            song_id=song.id,
            label=f"Mix {i + 1}",
            audio_file=f"/path/to/song_{i + 1}.wav",
            duration_seconds=180.0 + i * 30,
            original_sample_rate=44100,
            audio_hash=f"hash_{i}",
            created_at=now,
        )
        session.song_versions.create(version)
        versions.append(version)

    session.commit()
    return session, songs, versions


# ---------------------------------------------------------------------------
# Orchestrator — success path
# ---------------------------------------------------------------------------


class TestOrchestratorSuccess:
    """Verify successful analysis creates layers and takes."""

    def test_analyze_creates_layers_and_takes(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        service = Orchestrator(get_registry(), _default_executors())

        result = service.analyze(session, version.id, "onset_detection")

        assert isinstance(result, Ok)
        ar = result.value
        assert ar.song_version_id == version.id
        assert ar.pipeline_id == "onset_detection"
        assert len(ar.layer_ids) == 1
        assert len(ar.take_ids) == 1
        assert ar.duration_ms >= 0

        # Verify layer created in DB
        layers = session.layers.list_by_version(version.id)
        assert len(layers) == 1
        assert layers[0].name == "onsets"
        assert layers[0].layer_type == "analysis"
        assert layers[0].visible is True

        # Verify take created in DB
        takes = session.takes.list_by_layer(layers[0].id)
        assert len(takes) == 1
        assert takes[0].is_main is True
        assert takes[0].origin == "pipeline"

        session.close()

    def test_analyze_take_contains_correct_events(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        service = Orchestrator(get_registry(), _default_executors())

        result = service.analyze(session, version.id, "onset_detection")
        assert isinstance(result, Ok)

        layers = session.layers.list_by_version(version.id)
        takes = session.takes.list_by_layer(layers[0].id)
        take = takes[0]

        assert isinstance(take.data, EventData)
        assert len(take.data.layers) == 1
        assert len(take.data.layers[0].events) == 3
        assert take.data.layers[0].events[0].time == 1.0
        assert take.data.layers[0].events[1].time == 2.5
        assert take.data.layers[0].events[2].time == 4.0

        session.close()

    def test_analyze_take_source_records_provenance(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        service = Orchestrator(get_registry(), _default_executors())

        result = service.analyze(session, version.id, "onset_detection")
        assert isinstance(result, Ok)

        layers = session.layers.list_by_version(version.id)
        takes = session.takes.list_by_layer(layers[0].id)
        take = takes[0]

        assert take.source is not None
        assert take.source.block_id == "detect_onsets"
        assert take.source.block_type == "DetectOnsets"
        assert take.source.run_id  # Non-empty execution ID

        session.close()

    def test_analyze_result_ids_match_persisted(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        service = Orchestrator(get_registry(), _default_executors())

        result = service.analyze(session, version.id, "onset_detection")
        assert isinstance(result, Ok)
        ar = result.value

        layers = session.layers.list_by_version(version.id)
        assert [lr.id for lr in layers] == ar.layer_ids

        takes = session.takes.list_by_layer(layers[0].id)
        assert [t.id for t in takes] == ar.take_ids

        session.close()


# ---------------------------------------------------------------------------
# Orchestrator — error handling
# ---------------------------------------------------------------------------


class TestOrchestratorErrors:
    """Verify error handling for invalid inputs."""

    def test_nonexistent_song_version(self, tmp_path: Any) -> None:
        session = ProjectStorage.create_new("Test", working_dir_root=tmp_path)
        service = Orchestrator(get_registry(), _default_executors())

        result = service.analyze(session, "nonexistent_id", "onset_detection")

        assert isinstance(result, Err)
        assert isinstance(result.error, ValidationError)
        assert "SongVersionRecord not found" in str(result.error)

        session.close()

    def test_nonexistent_pipeline(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        service = Orchestrator(get_registry(), _default_executors())

        result = service.analyze(session, version.id, "nonexistent_pipeline")

        assert isinstance(result, Err)
        assert isinstance(result.error, ValidationError)
        assert "Pipeline template not found" in str(result.error)

        session.close()

    def test_invalid_bindings_wrong_type(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        service = Orchestrator(get_registry(), _default_executors())

        result = service.analyze(
            session, version.id, "onset_detection",
            bindings={"threshold": "not_a_float"},
        )

        assert isinstance(result, Err)
        assert isinstance(result.error, ValidationError)

        session.close()

    def test_engine_failure(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        executors = {
            "LoadAudio": FailingExecutor(),
            "DetectOnsets": MockDetectOnsetsExecutor(),
        }
        service = Orchestrator(get_registry(), executors)

        result = service.analyze(session, version.id, "onset_detection")

        assert isinstance(result, Err)
        assert isinstance(result.error, ExecutionError)

        session.close()

    def test_missing_executor_for_block_type(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        # Only register load_audio, missing detect_onsets
        executors: dict[str, Any] = {"LoadAudio": MockLoadAudioExecutor()}
        service = Orchestrator(get_registry(), executors)

        result = service.analyze(session, version.id, "onset_detection")

        assert isinstance(result, Err)

        session.close()


# ---------------------------------------------------------------------------
# Orchestrator — bindings
# ---------------------------------------------------------------------------


class TestOrchestratorBindings:
    """Verify binding application and auto-binding."""

    def test_custom_bindings_applied_to_graph(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        capturing_executor = SettingsCapturingExecutor()
        executors: dict[str, Any] = {
            "LoadAudio": MockLoadAudioExecutor(),
            "DetectOnsets": capturing_executor,
        }
        service = Orchestrator(get_registry(), executors)

        result = service.analyze(
            session, version.id, "onset_detection",
            bindings={"threshold": 0.7},
        )

        assert isinstance(result, Ok)
        assert capturing_executor.captured_settings is not None
        assert capturing_executor.captured_settings["threshold"] == 0.7

        session.close()

    def test_auto_binding_sets_audio_file(self, tmp_path: Any) -> None:
        """The service auto-binds audio_file from SongVersionRecord."""
        session, song, version = _create_session_with_song(
            tmp_path, audio_file="/my/custom/audio.wav",
        )

        class AudioPathCapturingExecutor:
            def __init__(self) -> None:
                self.captured_file_path: str | None = None

            def execute(self, block_id: str, context: ExecutionContext) -> Any:
                block = context.graph.blocks.get(block_id)
                if block:
                    self.captured_file_path = block.settings.get("file_path")
                return ok(AudioData(
                    sample_rate=44100, duration=180.0,
                    file_path="test.wav", channel_count=2,
                ))

        capturing = AudioPathCapturingExecutor()
        executors: dict[str, Any] = {
            "LoadAudio": capturing,
            "DetectOnsets": MockDetectOnsetsExecutor(),
        }
        service = Orchestrator(get_registry(), executors)

        result = service.analyze(session, version.id, "onset_detection")

        assert isinstance(result, Ok)
        assert capturing.captured_file_path == os.path.abspath("/my/custom/audio.wav")

        session.close()

    def test_unknown_binding_key_fails_validation(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        service = Orchestrator(get_registry(), _default_executors())

        result = service.analyze(
            session, version.id, "onset_detection",
            bindings={"bogus_key": 42},
        )

        assert isinstance(result, Err)
        assert isinstance(result.error, ValidationError)
        assert "Unknown parameter" in str(result.error)

        session.close()


# ---------------------------------------------------------------------------
# Orchestrator — re-run behavior
# ---------------------------------------------------------------------------


class TestOrchestratorReRun:
    """Verify re-run behavior with existing layers."""

    def test_first_run_creates_main_take(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        service = Orchestrator(get_registry(), _default_executors())

        service.analyze(session, version.id, "onset_detection")

        layers = session.layers.list_by_version(version.id)
        assert len(layers) == 1
        takes = session.takes.list_by_layer(layers[0].id)
        assert len(takes) == 1
        assert takes[0].is_main is True

        session.close()

    def test_rerun_adds_non_main_take(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        service = Orchestrator(get_registry(), _default_executors())

        # First run
        result1 = service.analyze(session, version.id, "onset_detection")
        assert isinstance(result1, Ok)

        # Second run
        result2 = service.analyze(session, version.id, "onset_detection")
        assert isinstance(result2, Ok)

        # Should still have 1 layer but 2 takes
        layers = session.layers.list_by_version(version.id)
        assert len(layers) == 1

        takes = session.takes.list_by_layer(layers[0].id)
        assert len(takes) == 2

        main_takes = [t for t in takes if t.is_main]
        non_main_takes = [t for t in takes if not t.is_main]
        assert len(main_takes) == 1
        assert len(non_main_takes) == 1

        session.close()

    def test_rerun_does_not_duplicate_layer(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        service = Orchestrator(get_registry(), _default_executors())

        # Run three times
        service.analyze(session, version.id, "onset_detection")
        service.analyze(session, version.id, "onset_detection")
        service.analyze(session, version.id, "onset_detection")

        # Still only 1 layer, but 3 takes
        layers = session.layers.list_by_version(version.id)
        assert len(layers) == 1
        takes = session.takes.list_by_layer(layers[0].id)
        assert len(takes) == 3

        session.close()


# ---------------------------------------------------------------------------
# Orchestrator — progress callback
# ---------------------------------------------------------------------------


class TestOrchestratorProgress:
    """Verify progress callback behavior."""

    def test_progress_callback_called(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        service = Orchestrator(get_registry(), _default_executors())

        progress_calls: list[tuple[str, float]] = []

        def on_progress(message: str, percent: float) -> None:
            progress_calls.append((message, percent))

        result = service.analyze(
            session, version.id, "onset_detection",
            on_progress=on_progress,
        )

        assert isinstance(result, Ok)
        assert len(progress_calls) >= 3
        assert progress_calls[0][1] == 0.0
        assert progress_calls[-1][1] == 1.0
        assert progress_calls[-1][0] == "Complete"

        session.close()


# ---------------------------------------------------------------------------
# Orchestrator — persistence round-trip
# ---------------------------------------------------------------------------


class TestOrchestratorRoundTrip:
    """Verify full persistence round-trip: create, analyze, close, reopen, verify."""

    def test_full_round_trip(self, tmp_path: Any) -> None:
        # Create and analyze
        session, song, version = _create_session_with_song(tmp_path)
        service = Orchestrator(get_registry(), _default_executors())

        result = service.analyze(session, version.id, "onset_detection")
        assert isinstance(result, Ok)
        ar = result.value
        working_dir = session.working_dir
        session.close()

        # Reopen and verify
        session2 = ProjectStorage.open_db(working_dir)

        layers = session2.layers.list_by_version(version.id)
        assert len(layers) == 1
        assert layers[0].name == "onsets"
        assert layers[0].id == ar.layer_ids[0]

        takes = session2.takes.list_by_layer(layers[0].id)
        assert len(takes) == 1
        assert takes[0].id == ar.take_ids[0]
        assert takes[0].is_main is True
        assert isinstance(takes[0].data, EventData)
        assert len(takes[0].data.layers[0].events) == 3

        session2.close()


# ---------------------------------------------------------------------------
# SetlistProcessor — success path
# ---------------------------------------------------------------------------


def _create_pipeline_config(session, song_version_id, template_id="onset_detection"):
    """Helper: create a PipelineConfigRecord via Orchestrator.create_config and return the ID."""
    from echozero.result import unwrap
    service = Orchestrator(get_registry(), _default_executors())
    result = service.create_config(session, song_version_id, template_id)
    config = unwrap(result)
    return config.id


class TestSetlistProcessorSuccess:
    """Verify successful setlist processing."""

    def test_process_all_songs_succeed(self, tmp_path: Any) -> None:
        session, songs, versions = _create_session_with_songs(tmp_path, count=3)
        service = Orchestrator(get_registry(), _default_executors())
        processor = SetlistProcessor(service)

        config_ids = [_create_pipeline_config(session, v.id) for v in versions]

        result = processor.process_setlist(session, config_ids)

        assert result.total == 3
        assert result.succeeded == 3
        assert result.failed == 0
        assert len(result.results) == 3
        assert all(isinstance(r, Ok) for r in result.results)
        assert result.duration_ms >= 0

        session.close()

    def test_single_song(self, tmp_path: Any) -> None:
        session, song, version = _create_session_with_song(tmp_path)
        service = Orchestrator(get_registry(), _default_executors())
        processor = SetlistProcessor(service)

        config_ids = [_create_pipeline_config(session, version.id)]

        result = processor.process_setlist(session, config_ids)

        assert result.total == 1
        assert result.succeeded == 1
        assert result.failed == 0

        session.close()


# ---------------------------------------------------------------------------
# SetlistProcessor — failure isolation
# ---------------------------------------------------------------------------


class TestSetlistProcessorFailure:
    """Verify error isolation — one failure doesn't stop the others."""

    def test_one_failure_continues_processing(self, tmp_path: Any) -> None:
        session, songs, versions = _create_session_with_songs(tmp_path, count=3)
        service = Orchestrator(get_registry(), _default_executors())
        processor = SetlistProcessor(service)

        # Create valid configs for songs 0 and 2, use a bad ID for song 1
        config_ids = [
            _create_pipeline_config(session, versions[0].id),
            "nonexistent_config_id",
            _create_pipeline_config(session, versions[2].id),
        ]

        result = processor.process_setlist(session, config_ids)

        assert result.total == 3
        assert result.succeeded == 2
        assert result.failed == 1
        assert isinstance(result.results[0], Ok)
        assert isinstance(result.results[1], Err)
        assert isinstance(result.results[2], Ok)

        session.close()

    def test_all_fail(self, tmp_path: Any) -> None:
        session = ProjectStorage.create_new("Test", working_dir_root=tmp_path)
        service = Orchestrator(get_registry(), _default_executors())
        processor = SetlistProcessor(service)

        config_ids = [f"bad_id_{i}" for i in range(3)]

        result = processor.process_setlist(session, config_ids)

        assert result.total == 3
        assert result.succeeded == 0
        assert result.failed == 3
        assert all(isinstance(r, Err) for r in result.results)

        session.close()


# ---------------------------------------------------------------------------
# SetlistProcessor — progress
# ---------------------------------------------------------------------------


class TestSetlistProcessorProgress:
    """Verify progress callback behavior."""

    def test_progress_called_per_song(self, tmp_path: Any) -> None:
        session, songs, versions = _create_session_with_songs(tmp_path, count=3)
        service = Orchestrator(get_registry(), _default_executors())
        processor = SetlistProcessor(service)

        config_ids = [_create_pipeline_config(session, v.id) for v in versions]

        progress_calls: list[tuple[str, int, int]] = []

        def on_progress(message: str, current: int, total: int) -> None:
            progress_calls.append((message, current, total))

        processor.process_setlist(session, config_ids, on_progress=on_progress)

        assert len(progress_calls) == 3
        assert progress_calls[0] == ("Processing song 1/3", 0, 3)
        assert progress_calls[1] == ("Processing song 2/3", 1, 3)
        assert progress_calls[2] == ("Processing song 3/3", 2, 3)

        session.close()


# ---------------------------------------------------------------------------
# SetlistProcessor — empty
# ---------------------------------------------------------------------------


class TestSetlistProcessorEmpty:
    """Verify empty setlist behavior."""

    def test_empty_list_returns_zero_summary(self, tmp_path: Any) -> None:
        session = ProjectStorage.create_new("Test", working_dir_root=tmp_path)
        service = Orchestrator(get_registry(), _default_executors())
        processor = SetlistProcessor(service)

        result = processor.process_setlist(session, config_ids=[])

        assert result.total == 0
        assert result.succeeded == 0
        assert result.failed == 0
        assert result.results == []
        assert result.duration_ms >= 0

        session.close()


# ---------------------------------------------------------------------------
# SetlistProcessor — persistence round-trip
# ---------------------------------------------------------------------------


class TestSetlistProcessorRoundTrip:
    """Verify full setlist persistence round-trip."""

    def test_process_all_and_verify_persisted(self, tmp_path: Any) -> None:
        session, songs, versions = _create_session_with_songs(tmp_path, count=3)
        service = Orchestrator(get_registry(), _default_executors())
        processor = SetlistProcessor(service)

        config_ids = [_create_pipeline_config(session, v.id) for v in versions]

        result = processor.process_setlist(session, config_ids)
        assert result.succeeded == 3

        working_dir = session.working_dir
        session.close()

        # Reopen and verify all songs have layers and takes
        session2 = ProjectStorage.open_db(working_dir)

        for version in versions:
            layers = session2.layers.list_by_version(version.id)
            assert len(layers) == 1, f"Expected 1 layer for version {version.id}"
            assert layers[0].name == "onsets"

            takes = session2.takes.list_by_layer(layers[0].id)
            assert len(takes) == 1
            assert takes[0].is_main is True
            assert isinstance(takes[0].data, EventData)

        session2.close()
