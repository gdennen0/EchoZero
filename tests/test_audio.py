"""
Audio import tests: content-addressed storage, dedup, hash verification.
Exercises the audio module against real temp files using pytest tmp_path fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from echozero.persistence.audio import (
    AudioImportOptions,
    compute_audio_hash,
    import_audio,
    prepare_audio_for_import,
    resolve_audio_path,
    verify_audio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_audio_file(path: Path, content: bytes = b"fake wav data 12345") -> Path:
    """Write a fake audio file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# compute_audio_hash
# ---------------------------------------------------------------------------


class TestComputeAudioHash:
    def test_deterministic(self, tmp_path):
        """Same file content always produces the same hash."""
        f = _write_audio_file(tmp_path / "song.wav", b"hello audio world")
        h1 = compute_audio_hash(f)
        h2 = compute_audio_hash(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        """Different content produces different hashes."""
        f1 = _write_audio_file(tmp_path / "a.wav", b"content A")
        f2 = _write_audio_file(tmp_path / "b.wav", b"content B")
        assert compute_audio_hash(f1) != compute_audio_hash(f2)

    def test_returns_hex_string(self, tmp_path):
        """Hash is a hex string of expected length (64 chars for SHA-256)."""
        f = _write_audio_file(tmp_path / "song.wav", b"data")
        h = compute_audio_hash(f)
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_large_file(self, tmp_path):
        """Hash works for files larger than the chunk size (8192)."""
        content = b"x" * 20000
        f = _write_audio_file(tmp_path / "large.wav", content)
        h = compute_audio_hash(f)
        assert len(h) == 64

    def test_empty_file(self, tmp_path):
        """Hash works for empty files."""
        f = _write_audio_file(tmp_path / "empty.wav", b"")
        h = compute_audio_hash(f)
        assert len(h) == 64


# ---------------------------------------------------------------------------
# import_audio
# ---------------------------------------------------------------------------


class TestImportAudio:
    def test_copies_file_with_content_addressed_name(self, tmp_path):
        """import_audio copies the file into audio/ with a hash-based name."""
        source = _write_audio_file(tmp_path / "source" / "my_song.wav")
        working_dir = tmp_path / "project"
        working_dir.mkdir()

        rel_path, audio_hash = import_audio(source, working_dir)

        # File should exist in audio/
        dest = working_dir / rel_path
        assert dest.exists()
        assert dest.read_bytes() == source.read_bytes()

    def test_returns_correct_relative_path(self, tmp_path):
        """Relative path starts with 'audio/' and uses hash prefix + extension."""
        source = _write_audio_file(tmp_path / "song.wav", b"test data")
        working_dir = tmp_path / "project"
        working_dir.mkdir()

        rel_path, audio_hash = import_audio(source, working_dir)

        assert rel_path.startswith("audio/")
        assert rel_path.endswith(".wav")
        # Filename is first 16 chars of hash + extension
        expected_name = f"{audio_hash[:16]}.wav"
        assert rel_path == f"audio/{expected_name}"

    def test_returns_correct_hash(self, tmp_path):
        """The returned hash matches an independent computation."""
        content = b"some audio bytes"
        source = _write_audio_file(tmp_path / "song.wav", content)
        working_dir = tmp_path / "project"
        working_dir.mkdir()

        _, audio_hash = import_audio(source, working_dir)

        assert audio_hash == compute_audio_hash(source)

    def test_dedup_same_file_twice(self, tmp_path):
        """Importing the same file twice doesn't create a second copy."""
        source = _write_audio_file(tmp_path / "song.wav", b"dedup test")
        working_dir = tmp_path / "project"
        working_dir.mkdir()

        rel_path1, hash1 = import_audio(source, working_dir)
        rel_path2, hash2 = import_audio(source, working_dir)

        assert rel_path1 == rel_path2
        assert hash1 == hash2

        # Only one file in audio/
        audio_dir = working_dir / "audio"
        assert len(list(audio_dir.iterdir())) == 1

    def test_different_files_get_different_names(self, tmp_path):
        """Two different audio files get different content-addressed names."""
        source_a = _write_audio_file(tmp_path / "a.wav", b"audio file A")
        source_b = _write_audio_file(tmp_path / "b.wav", b"audio file B")
        working_dir = tmp_path / "project"
        working_dir.mkdir()

        rel_a, hash_a = import_audio(source_a, working_dir)
        rel_b, hash_b = import_audio(source_b, working_dir)

        assert rel_a != rel_b
        assert hash_a != hash_b

        # Both files exist
        assert (working_dir / rel_a).exists()
        assert (working_dir / rel_b).exists()

    def test_creates_audio_dir_if_missing(self, tmp_path):
        """audio/ directory is created automatically."""
        source = _write_audio_file(tmp_path / "song.wav")
        working_dir = tmp_path / "project"
        working_dir.mkdir()

        assert not (working_dir / "audio").exists()
        import_audio(source, working_dir)
        assert (working_dir / "audio").exists()

    def test_preserves_file_extension(self, tmp_path):
        """File extension from source is preserved."""
        source_wav = _write_audio_file(tmp_path / "song.wav", b"wav data")
        source_mp3 = _write_audio_file(tmp_path / "song.mp3", b"mp3 data")
        working_dir = tmp_path / "project"
        working_dir.mkdir()

        rel_wav, _ = import_audio(source_wav, working_dir)
        rel_mp3, _ = import_audio(source_mp3, working_dir)

        assert rel_wav.endswith(".wav")
        assert rel_mp3.endswith(".mp3")


class TestImportPreprocessing:
    def test_prepare_audio_for_import_uses_requested_detection_mode(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        source = _write_audio_file(tmp_path / "printed.wav", b"fake stereo data")
        working_dir = tmp_path / "project"
        working_dir.mkdir()
        captured_modes: list[str] = []

        def _fake_detect(source_path: Path, *, mode: str = "strict") -> None:
            assert source_path == source
            captured_modes.append(mode)
            return None

        monkeypatch.setattr(
            "echozero.persistence.audio.detect_ltc_channel",
            _fake_detect,
        )

        prepared = prepare_audio_for_import(
            source,
            working_dir,
            options=AudioImportOptions(
                strip_ltc_timecode=True,
                ltc_detection_mode="aggressive",
            ),
        )

        assert prepared.source_path == source
        assert captured_modes == ["aggressive"]

    def test_prepare_audio_for_import_honors_channel_override(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        source = _write_audio_file(tmp_path / "printed.wav", b"fake stereo data")
        working_dir = tmp_path / "project"
        working_dir.mkdir()
        write_calls: list[int] = []

        monkeypatch.setattr(
            "echozero.persistence.audio.detect_ltc_channel",
            lambda _path, *, mode="strict": (_ for _ in ()).throw(
                AssertionError("detect_ltc_channel should not be called when override is set")
            ),
        )
        monkeypatch.setattr(
            "echozero.persistence.audio.compute_audio_hash",
            lambda _path: "c" * 64,
        )

        def _fake_write(_path: Path, *, working_dir: Path, channel_index: int) -> Path:
            write_calls.append(channel_index)
            staged = working_dir / f"stage_{channel_index}.wav"
            staged.write_bytes(f"RIFF{channel_index}".encode("utf-8"))
            return staged

        monkeypatch.setattr(
            "echozero.persistence.audio._write_import_channel_copy",
            _fake_write,
        )

        prepared = prepare_audio_for_import(
            source,
            working_dir,
            options=AudioImportOptions(
                strip_ltc_timecode=True,
                ltc_channel_override="right",
            ),
        )

        assert write_calls == [0, 1]
        assert prepared.ltc_artifact_path is not None
        assert prepared.program_artifact_path is not None
        assert prepared.ltc_artifact_path.name.endswith("_ltc_right.wav")
        assert prepared.program_artifact_path.name.endswith("_program_left.wav")


# ---------------------------------------------------------------------------
# verify_audio
# ---------------------------------------------------------------------------


class TestVerifyAudio:
    def test_valid_file_returns_true(self, tmp_path):
        """verify_audio returns True when file exists and hash matches."""
        source = _write_audio_file(tmp_path / "song.wav", b"verify me")
        working_dir = tmp_path / "project"
        working_dir.mkdir()

        rel_path, audio_hash = import_audio(source, working_dir)
        assert verify_audio(working_dir, rel_path, audio_hash) is True

    def test_missing_file_returns_false(self, tmp_path):
        """verify_audio returns False when file doesn't exist."""
        working_dir = tmp_path / "project"
        working_dir.mkdir()

        assert verify_audio(working_dir, "audio/nonexistent.wav", "abc123") is False

    def test_tampered_file_returns_false(self, tmp_path):
        """verify_audio returns False when file content doesn't match hash."""
        source = _write_audio_file(tmp_path / "song.wav", b"original")
        working_dir = tmp_path / "project"
        working_dir.mkdir()

        rel_path, audio_hash = import_audio(source, working_dir)

        # Tamper with the file
        (working_dir / rel_path).write_bytes(b"tampered")

        assert verify_audio(working_dir, rel_path, audio_hash) is False

    def test_wrong_hash_returns_false(self, tmp_path):
        """verify_audio returns False when expected hash is wrong."""
        source = _write_audio_file(tmp_path / "song.wav", b"data")
        working_dir = tmp_path / "project"
        working_dir.mkdir()

        rel_path, _ = import_audio(source, working_dir)
        assert verify_audio(working_dir, rel_path, "wrong_hash") is False


# ---------------------------------------------------------------------------
# resolve_audio_path
# ---------------------------------------------------------------------------


class TestResolveAudioPath:
    def test_returns_correct_absolute_path(self, tmp_path):
        """resolve_audio_path joins working_dir and relative path."""
        working_dir = tmp_path / "project"
        result = resolve_audio_path(working_dir, "audio/abc123.wav")
        assert result == working_dir / "audio" / "abc123.wav"
        assert result.is_absolute()

    def test_works_with_imported_file(self, tmp_path):
        """resolve_audio_path returns a path that exists after import."""
        source = _write_audio_file(tmp_path / "song.wav", b"data")
        working_dir = tmp_path / "project"
        working_dir.mkdir()

        rel_path, _ = import_audio(source, working_dir)
        resolved = resolve_audio_path(working_dir, rel_path)
        assert resolved.exists()
