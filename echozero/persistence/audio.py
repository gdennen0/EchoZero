"""
Audio import: Content-addressed audio storage for EchoZero projects.
Exists because audio files must be copied into the project working directory with
deterministic, hash-based filenames for deduplication and integrity verification.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def compute_audio_hash(source_path: Path) -> str:
    """Compute SHA-256 hash of an audio file. Returns hex digest."""
    h = hashlib.sha256()
    with open(source_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def import_audio(source_path: Path, working_dir: Path) -> tuple[str, str]:
    """Copy an audio file into the project's audio directory with content-addressed naming.

    Args:
        source_path: Absolute path to the source audio file
        working_dir: Project working directory

    Returns:
        (project_relative_path, sha256_hash)
        e.g. ('audio/a3f2c8d1e9b04f71.wav', 'a3f2c8d1e9...')

    If the file already exists (same hash), skips the copy (dedup).
    """
    audio_dir = working_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    audio_hash = compute_audio_hash(source_path)
    dest_name = f"{audio_hash[:16]}{source_path.suffix}"
    dest_path = audio_dir / dest_name

    if not dest_path.exists():
        shutil.copy2(source_path, dest_path)

    return f"audio/{dest_name}", audio_hash


def verify_audio(working_dir: Path, audio_file: str, expected_hash: str) -> bool:
    """Verify an audio file's integrity by checking its hash.

    Args:
        working_dir: Project working directory
        audio_file: Project-relative path (e.g. 'audio/a3f2c8d1e9b04f71.wav')
        expected_hash: Expected SHA-256 hash

    Returns True if file exists and hash matches, False otherwise.
    Non-blocking — returns False on missing file, doesn't raise.
    """
    full_path = working_dir / audio_file
    if not full_path.exists():
        return False
    actual_hash = compute_audio_hash(full_path)
    return actual_hash == expected_hash


def resolve_audio_path(working_dir: Path, audio_file: str) -> Path:
    """Resolve a project-relative audio path to an absolute path."""
    return working_dir / audio_file


# ---------------------------------------------------------------------------
# Audio metadata scanning
# ---------------------------------------------------------------------------


class AudioMetadata:
    """Audio file metadata discovered during import.

    Populated by scan_audio_metadata(). Carries all info we can extract from
    the file itself — duration, sample rate, channels, format. Extensible
    for future metadata (bit depth, codec, loudness, etc.).
    """

    __slots__ = ("duration_seconds", "sample_rate", "channel_count")

    def __init__(
        self,
        duration_seconds: float,
        sample_rate: int,
        channel_count: int,
    ) -> None:
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.channel_count = channel_count

    def __repr__(self) -> str:
        return (
            f"AudioMetadata(duration={self.duration_seconds:.2f}s, "
            f"sr={self.sample_rate}, ch={self.channel_count})"
        )


def scan_audio_metadata(
    audio_path: Path,
    scan_fn=None,
) -> AudioMetadata:
    """Read audio file metadata without loading samples.

    Args:
        audio_path: Path to the audio file.
        scan_fn: Optional injectable function(path) -> AudioMetadata for testing.

    Returns:
        AudioMetadata with duration, sample rate, and channel count.

    Raises:
        RuntimeError: If metadata cannot be read.
    """
    if scan_fn is not None:
        return scan_fn(audio_path)

    try:
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "Audio metadata scanning requires soundfile. "
            "Install with: pip install soundfile"
        )

    try:
        info = sf.info(str(audio_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio metadata from '{audio_path}': {exc}")

    return AudioMetadata(
        duration_seconds=info.duration,
        sample_rate=info.samplerate,
        channel_count=info.channels,
    )
