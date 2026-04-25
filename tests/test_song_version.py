"""
SongVersionRecord tests: Verify unified audio import, metadata scanning, config copy (D278),
and default template application on first song add.
Exists because song version management is a critical vertical — audio in, metadata out,
configs copied or created, all through one clean path.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from echozero.domain.types import AudioData
from echozero.persistence.audio import AudioImportOptions, AudioMetadata
from echozero.persistence.entities import (
    LayerRecord,
    PipelineConfigRecord,
    SongDefaultPipelineConfigRecord,
    SongRecord,
    SongVersionRecord,
)
from echozero.persistence.session import ProjectStorage
from echozero.takes import Take

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_scan(path) -> AudioMetadata:
    """Mock audio scanner that returns realistic metadata without reading real audio."""
    return AudioMetadata(duration_seconds=180.5, sample_rate=44100, channel_count=2)


def _mock_scan_48k(path) -> AudioMetadata:
    """Mock scanner returning 48kHz mono metadata."""
    return AudioMetadata(duration_seconds=240.0, sample_rate=48000, channel_count=1)


def _create_session(tmp_path: Path) -> ProjectStorage:
    """Create a fresh ProjectStorage in a temp directory."""
    return ProjectStorage.create_new(
        name="Test ProjectRecord",
        working_dir_root=tmp_path / "working",
    )


def _create_audio_file(tmp_path: Path, name: str = "test.wav") -> Path:
    """Create a fake audio file for import."""
    audio = tmp_path / name
    audio.write_bytes(b"RIFF" + b"\x00" * 100)
    return audio


def _make_pipeline_config(
    song_version_id: str,
    template_id: str = "onset_detection",
    name: str = "Onset Detection",
    knob_values: dict[str, Any] | None = None,
    block_overrides: dict[str, list[str]] | None = None,
) -> PipelineConfigRecord:
    """Create a PipelineConfigRecord for testing."""
    now = datetime.now(timezone.utc)
    graph = {
        "blocks": [
            {
                "id": "load1",
                "name": "Load Audio",
                "block_type": "LoadAudio",
                "category": "PROCESSOR",
                "input_ports": [],
                "output_ports": [
                    {"name": "audio_out", "port_type": "AUDIO", "direction": "OUTPUT"}
                ],
                "settings": {"file_path": ""},
            },
            {
                "id": "detect1",
                "name": "Detect Onsets",
                "block_type": "DetectOnsets",
                "category": "PROCESSOR",
                "input_ports": [{"name": "audio_in", "port_type": "AUDIO", "direction": "INPUT"}],
                "output_ports": [
                    {"name": "events_out", "port_type": "EVENT", "direction": "OUTPUT"}
                ],
                "settings": {"threshold": 0.3, "min_gap": 0.05},
            },
        ],
        "connections": [
            {
                "source_block_id": "load1",
                "source_output_name": "audio_out",
                "target_block_id": "detect1",
                "target_input_name": "audio_in",
            }
        ],
    }
    return PipelineConfigRecord(
        id=uuid.uuid4().hex,
        song_version_id=song_version_id,
        template_id=template_id,
        name=name,
        graph_json=json.dumps(graph),
        outputs_json=json.dumps([]),
        knob_values=knob_values or {"threshold": 0.3, "min_gap": 0.05},
        created_at=now,
        updated_at=now,
        block_overrides=block_overrides or {},
    )


def _make_layer_record(
    *,
    song_version_id: str,
    name: str,
    order: int,
    parent_layer_id: str | None = None,
) -> LayerRecord:
    now = datetime.now(timezone.utc)
    return LayerRecord(
        id=uuid.uuid4().hex,
        song_version_id=song_version_id,
        name=name,
        layer_type="manual",
        color="#445566",
        order=order,
        visible=True,
        locked=False,
        parent_layer_id=parent_layer_id,
        source_pipeline=None,
        created_at=now,
        state_flags={"manual_kind": "event"},
        provenance={},
    )


def _make_audio_take(
    *,
    label: str,
    is_main: bool,
) -> Take:
    return Take(
        id=uuid.uuid4().hex,
        label=label,
        data=AudioData(
            sample_rate=44100,
            duration=12.0,
            file_path=f"audio/{uuid.uuid4().hex}.wav",
            channel_count=2,
        ),
        origin="user",
        source=None,
        created_at=datetime.now(timezone.utc),
        is_main=is_main,
        is_archived=False,
        notes="",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAudioMetadata:
    """Verify that _create_version populates real metadata from scanned audio."""

    def test_version_has_real_duration(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)
        song, v1 = session.import_song("Test", audio, scan_fn=_mock_scan)

        assert v1.duration_seconds == 180.5
        session.close()

    def test_version_has_real_sample_rate(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)
        song, v1 = session.import_song("Test", audio, scan_fn=_mock_scan)

        assert v1.original_sample_rate == 44100
        session.close()

    def test_version_metadata_from_48k_file(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)
        song, v1 = session.import_song("Test", audio, scan_fn=_mock_scan_48k)

        assert v1.original_sample_rate == 48000
        assert v1.duration_seconds == 240.0
        session.close()

    def test_add_version_also_scans_metadata(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test", audio1, scan_fn=_mock_scan)
        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan_48k)

        assert v2.duration_seconds == 240.0
        assert v2.original_sample_rate == 48000
        session.close()


class TestLtcImportPreprocessing:
    """Verify import-time LTC stripping is applied through the shared version path."""

    def test_import_song_strips_detected_ltc_channel_before_copy(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = _create_session(tmp_path)
        source = _create_audio_file(tmp_path, "ltc-stereo.wav")
        staged_program = tmp_path / "ltc-program-staged.wav"
        staged_program.write_bytes(b"RIFF_PROGRAM")
        staged_ltc = tmp_path / "ltc-timecode-staged.wav"
        staged_ltc.write_bytes(b"RIFF_TIMECODE")

        monkeypatch.setattr(
            "echozero.persistence.audio.detect_ltc_channel",
            lambda _path: "left",
        )
        monkeypatch.setattr(
            "echozero.persistence.audio.compute_audio_hash",
            lambda _path: "a" * 64,
        )

        def _fake_write(_path: Path, *, working_dir: Path, channel_index: int) -> Path:
            if channel_index == 1:
                return staged_program
            return staged_ltc

        monkeypatch.setattr(
            "echozero.persistence.audio._write_import_channel_copy",
            _fake_write,
        )

        def _scan(path: Path) -> AudioMetadata:
            if path == source:
                return AudioMetadata(duration_seconds=180.0, sample_rate=48000, channel_count=2)
            return AudioMetadata(duration_seconds=180.0, sample_rate=48000, channel_count=1)

        song, version = session.import_song(
            "LTC Song",
            source,
            default_templates=[],
            audio_import_options=AudioImportOptions(strip_ltc_timecode=True),
            scan_fn=_scan,
        )
        imported_path = session.working_dir / version.audio_file
        split_dir = session.working_dir / "audio" / "split_channels"
        retained_program = split_dir / f"{'a' * 16}_program_right.wav"
        retained_ltc = split_dir / f"{'a' * 16}_ltc_left.wav"

        assert imported_path.read_bytes() == b"RIFF_PROGRAM"
        assert retained_program.read_bytes() == b"RIFF_PROGRAM"
        assert retained_ltc.read_bytes() == b"RIFF_TIMECODE"
        assert not staged_program.exists()
        assert not staged_ltc.exists()
        assert song.active_version_id == version.id
        session.close()

    def test_ltc_channel_flip_extracts_left_audio_when_ltc_is_right(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = _create_session(tmp_path)
        source = _create_audio_file(tmp_path, "flipped-ltc.wav")
        staged_program = tmp_path / "flipped-program-staged.wav"
        staged_program.write_bytes(b"RIFF_LEFT_PROGRAM")
        staged_ltc = tmp_path / "flipped-timecode-staged.wav"
        staged_ltc.write_bytes(b"RIFF_RIGHT_TIMECODE")
        captured_channel_indices: list[int] = []

        monkeypatch.setattr(
            "echozero.persistence.audio.detect_ltc_channel",
            lambda _path: "right",
        )
        monkeypatch.setattr(
            "echozero.persistence.audio.compute_audio_hash",
            lambda _path: "b" * 64,
        )

        def _fake_write(_path: Path, *, working_dir: Path, channel_index: int) -> Path:
            captured_channel_indices.append(channel_index)
            if channel_index == 0:
                return staged_program
            return staged_ltc

        monkeypatch.setattr(
            "echozero.persistence.audio._write_import_channel_copy",
            _fake_write,
        )

        song, version = session.import_song(
            "Flipped LTC Song",
            source,
            default_templates=[],
            audio_import_options=AudioImportOptions(strip_ltc_timecode=True),
            scan_fn=_mock_scan,
        )
        imported_path = session.working_dir / version.audio_file
        split_dir = session.working_dir / "audio" / "split_channels"
        retained_program = split_dir / f"{'b' * 16}_program_left.wav"
        retained_ltc = split_dir / f"{'b' * 16}_ltc_right.wav"

        assert captured_channel_indices == [0, 1]
        assert imported_path.read_bytes() == b"RIFF_LEFT_PROGRAM"
        assert retained_program.read_bytes() == b"RIFF_LEFT_PROGRAM"
        assert retained_ltc.read_bytes() == b"RIFF_RIGHT_TIMECODE"
        assert not staged_program.exists()
        assert not staged_ltc.exists()
        assert song.active_version_id == version.id
        session.close()


class TestDefaultTemplates:
    """Verify that import_song creates default pipeline configs from templates."""

    def test_import_creates_configs_from_registered_templates(self, tmp_path: Path) -> None:
        # Ensure templates are registered
        import echozero.pipelines.templates  # noqa: F401
        from echozero.pipelines.registry import get_registry

        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)
        song, v1 = session.import_song("Test", audio, scan_fn=_mock_scan)

        configs = session.pipeline_configs.list_by_version(v1.id)
        default_configs = session.song_default_pipeline_configs.list_by_song(song.id)
        registered = get_registry().list()

        assert len(configs) == len(registered)
        assert len(default_configs) == len(registered)
        assert len(configs) > 0  # at least onset_detection exists
        session.close()

    def test_import_with_specific_templates(self, tmp_path: Path) -> None:
        import echozero.pipelines.templates  # noqa: F401

        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)
        song, v1 = session.import_song(
            "Test",
            audio,
            scan_fn=_mock_scan,
            default_templates=["onset_detection"],
        )

        configs = session.pipeline_configs.list_by_version(v1.id)
        assert len(configs) == 1
        assert configs[0].template_id == "onset_detection"
        session.close()

    def test_import_with_empty_templates_creates_none(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)
        song, v1 = session.import_song(
            "Test",
            audio,
            scan_fn=_mock_scan,
            default_templates=[],
        )

        configs = session.pipeline_configs.list_by_version(v1.id)
        assert len(configs) == 0
        session.close()
    def test_add_version_copies_song_defaults_not_active_version(self, tmp_path: Path) -> None:
        """add_song_version uses song defaults as the source of effective settings."""
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        # Import with no default templates
        song, v1 = session.import_song(
            "Test",
            audio1,
            scan_fn=_mock_scan,
            default_templates=[],
        )
        default_like = SongDefaultPipelineConfigRecord.from_version_config(
            _make_pipeline_config(v1.id, "custom", "My Custom Pipeline"),
            song_id=song.id,
        )
        session.song_default_pipeline_configs.create(default_like)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert len(v2_configs) == 1
        assert v2_configs[0].template_id == "custom"
        session.close()


class TestSetlistReorder:
    """Verify project-level song reordering persists through versioning storage."""

    def test_reorder_songs_updates_song_order(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio_a = _create_audio_file(tmp_path, "a.wav")
        audio_b = _create_audio_file(tmp_path, "b.wav")
        audio_c = _create_audio_file(tmp_path, "c.wav")

        song_a, _ = session.import_song("A", audio_a, default_templates=[], scan_fn=_mock_scan)
        song_b, _ = session.import_song("B", audio_b, default_templates=[], scan_fn=_mock_scan)
        song_c, _ = session.import_song("C", audio_c, default_templates=[], scan_fn=_mock_scan)

        session.reorder_songs([song_c.id, song_a.id, song_b.id])
        ordered = session.songs.list_by_project(session.project.id)

        assert [song.id for song in ordered] == [song_c.id, song_a.id, song_b.id]
        session.close()

    def test_reorder_songs_rejects_missing_ids(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio_a = _create_audio_file(tmp_path, "a.wav")
        audio_b = _create_audio_file(tmp_path, "b.wav")

        song_a, _ = session.import_song("A", audio_a, default_templates=[], scan_fn=_mock_scan)
        song_b, _ = session.import_song("B", audio_b, default_templates=[], scan_fn=_mock_scan)

        with pytest.raises(ValueError, match="requires one ID for every song"):
            session.reorder_songs([song_a.id])

        with pytest.raises(ValueError, match="requires the same song IDs"):
            session.reorder_songs([song_a.id, "ghost-song-id"])
        session.close()


class TestAddSongVersion:
    """Verify the D278 add_song_version flow."""

    def test_creates_new_version(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song(
            "Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        v2 = session.add_song_version(song.id, audio2, label="Festival Edit", scan_fn=_mock_scan)

        assert v2.song_id == song.id
        assert v2.label == "Festival Edit"
        assert v2.id != v1.id
        session.close()

    def test_new_version_has_different_audio_file(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = tmp_path / "v2.wav"
        audio2.write_bytes(b"RIFF" + b"\xff" * 100)

        song, v1 = session.import_song(
            "Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)

        assert v2.audio_file != v1.audio_file
        assert v2.audio_hash != v1.audio_hash
        session.close()

    def test_activates_new_version_by_default(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song(
            "Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)

        updated_song = session.songs.get(song.id)
        assert updated_song.active_version_id == v2.id
        session.close()

    def test_does_not_activate_when_flag_false(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song(
            "Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        v2 = session.add_song_version(song.id, audio2, activate=False, scan_fn=_mock_scan)

        updated_song = session.songs.get(song.id)
        assert updated_song.active_version_id == v1.id
        session.close()

    def test_copies_song_default_pipeline_configs(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song(
            "Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        config1 = SongDefaultPipelineConfigRecord.from_version_config(
            _make_pipeline_config(v1.id, "onset", "Onset Detection"),
            song_id=song.id,
        )
        config2 = SongDefaultPipelineConfigRecord.from_version_config(
            _make_pipeline_config(v1.id, "classify", "Classification"),
            song_id=song.id,
        )
        session.song_default_pipeline_configs.create(config1)
        session.song_default_pipeline_configs.create(config2)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)
        assert len(v2_configs) == 2
        session.close()

    def test_copied_configs_have_new_ids(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song(
            "Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        config = SongDefaultPipelineConfigRecord.from_version_config(
            _make_pipeline_config(v1.id), song_id=song.id
        )
        session.song_default_pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert v2_configs[0].id != config.id
        session.close()

    def test_copied_configs_preserve_knob_values(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song(
            "Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        config = SongDefaultPipelineConfigRecord.from_version_config(
            _make_pipeline_config(v1.id, knob_values={"threshold": 0.7, "min_gap": 0.02}),
            song_id=song.id,
        )
        session.song_default_pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert v2_configs[0].knob_values == {"threshold": 0.7, "min_gap": 0.02}
        session.close()

    def test_copied_configs_preserve_block_overrides(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song(
            "Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        config = SongDefaultPipelineConfigRecord.from_version_config(
            _make_pipeline_config(v1.id, block_overrides={"detect1": ["threshold"]}),
            song_id=song.id,
        )
        session.song_default_pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert v2_configs[0].block_overrides == {"detect1": ["threshold"]}
        session.close()

    def test_copied_configs_point_to_new_version(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song(
            "Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        config = SongDefaultPipelineConfigRecord.from_version_config(
            _make_pipeline_config(v1.id), song_id=song.id
        )
        session.song_default_pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v2_configs = session.pipeline_configs.list_by_version(v2.id)

        assert v2_configs[0].song_version_id == v2.id
        session.close()

    def test_original_configs_unchanged(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song(
            "Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        config = SongDefaultPipelineConfigRecord.from_version_config(
            _make_pipeline_config(v1.id), song_id=song.id
        )
        session.song_default_pipeline_configs.create(config)
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        v1_configs = session.pipeline_configs.list_by_version(v1.id)

        assert len(v1_configs) == 0
        session.close()

    def test_auto_label_when_none_provided(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song(
            "Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)

        assert v2.label == "v2"
        session.close()

    def test_version_listed_in_song_versions(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song(
            "Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)

        versions = session.song_versions.list_by_song(song.id)
        assert len(versions) == 2
        session.close()

    def test_new_version_carries_forward_ma3_timecode_pool(self, tmp_path: Path) -> None:
        from dataclasses import replace

        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song(
            "Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        session.song_versions.update(replace(v1, ma3_timecode_pool_no=113))
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)

        assert v2.ma3_timecode_pool_no == 113
        session.close()

    def test_import_song_assigns_next_unused_ma3_timecode_pool(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "song_a.wav")
        audio2 = _create_audio_file(tmp_path, "song_b.wav")

        _song_a, version_a = session.import_song(
            "Song A", audio1, scan_fn=_mock_scan, default_templates=[]
        )
        _song_b, version_b = session.import_song(
            "Song B", audio2, scan_fn=_mock_scan, default_templates=[]
        )

        assert version_a.ma3_timecode_pool_no == 1
        assert version_b.ma3_timecode_pool_no == 2
        session.close()

    def test_new_version_does_not_copy_layers_by_default(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[])
        source_layer = _make_layer_record(song_version_id=v1.id, name="Drums", order=0)
        session.layers.create(source_layer)
        session.takes.create(source_layer.id, _make_audio_take(label="Take 1", is_main=True))
        session.commit()

        v2 = session.add_song_version(song.id, audio2, scan_fn=_mock_scan)

        assert session.layers.list_by_version(v2.id) == []
        session.close()

    def test_can_transfer_all_layers_to_new_version(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[])
        parent_layer = _make_layer_record(song_version_id=v1.id, name="Drums", order=0)
        child_layer = _make_layer_record(
            song_version_id=v1.id,
            name="Snare Hits",
            order=1,
            parent_layer_id=parent_layer.id,
        )
        session.layers.create(parent_layer)
        session.layers.create(child_layer)
        session.takes.create(parent_layer.id, _make_audio_take(label="Main", is_main=True))
        session.takes.create(parent_layer.id, _make_audio_take(label="Alt", is_main=False))
        session.takes.create(child_layer.id, _make_audio_take(label="Main", is_main=True))
        session.commit()

        v2 = session.add_song_version(song.id, audio2, transfer_layers=True, scan_fn=_mock_scan)

        copied_layers = session.layers.list_by_version(v2.id)
        assert [layer.name for layer in copied_layers] == ["Drums", "Snare Hits"]
        assert all(layer.id not in {parent_layer.id, child_layer.id} for layer in copied_layers)
        copied_by_name = {layer.name: layer for layer in copied_layers}
        assert copied_by_name["Snare Hits"].parent_layer_id == copied_by_name["Drums"].id
        copied_parent_takes = session.takes.list_by_layer(copied_by_name["Drums"].id)
        assert [take.label for take in copied_parent_takes] == ["Main", "Alt"]
        assert [take.is_main for take in copied_parent_takes] == [True, False]
        session.close()

    def test_can_transfer_selected_layers_to_new_version(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio1 = _create_audio_file(tmp_path, "v1.wav")
        audio2 = _create_audio_file(tmp_path, "v2.wav")

        song, v1 = session.import_song("Test SongRecord", audio1, scan_fn=_mock_scan, default_templates=[])
        drums = _make_layer_record(song_version_id=v1.id, name="Drums", order=0)
        bass = _make_layer_record(song_version_id=v1.id, name="Bass", order=1)
        fx = _make_layer_record(song_version_id=v1.id, name="FX", order=2)
        session.layers.create(drums)
        session.layers.create(bass)
        session.layers.create(fx)
        session.takes.create(drums.id, _make_audio_take(label="Take 1", is_main=True))
        session.takes.create(bass.id, _make_audio_take(label="Take 1", is_main=True))
        session.takes.create(fx.id, _make_audio_take(label="Take 1", is_main=True))
        session.commit()

        v2 = session.add_song_version(
            song.id,
            audio2,
            transfer_layers=True,
            transfer_layer_ids=[bass.id, "missing-layer-id"],
            scan_fn=_mock_scan,
        )

        copied_layers = session.layers.list_by_version(v2.id)
        assert [layer.name for layer in copied_layers] == ["Bass"]
        copied_takes = session.takes.list_by_layer(copied_layers[0].id)
        assert len(copied_takes) == 1
        assert copied_takes[0].label == "Take 1"
        session.close()

    def test_nonexistent_song_raises(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)

        with pytest.raises(ValueError, match="SongRecord not found"):
            session.add_song_version("ghost_id", audio, scan_fn=_mock_scan)
        session.close()

    def test_song_without_active_version_raises(self, tmp_path: Path) -> None:
        session = _create_session(tmp_path)
        audio = _create_audio_file(tmp_path)

        from dataclasses import replace

        song, v1 = session.import_song(
            "Test SongRecord", audio, scan_fn=_mock_scan, default_templates=[]
        )
        session.songs.update(replace(song, active_version_id=None))
        session.commit()

        audio2 = _create_audio_file(tmp_path, "v2.wav")
        with pytest.raises(ValueError, match="no active version"):
            session.add_song_version(song.id, audio2, scan_fn=_mock_scan)
        session.close()
