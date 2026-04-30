"""
Audio import: Content-addressed audio storage for EchoZero projects.
Exists because audio files must be copied into the project working directory with
deterministic, hash-based filenames for deduplication and integrity verification.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(slots=True, frozen=True)
class AudioImportOptions:
    """Optional import-time audio preprocessing settings."""

    strip_ltc_timecode: bool = False
    ltc_detection_mode: Literal["strict", "aggressive"] = "strict"
    ltc_channel_override: Literal["left", "right"] | None = None


@dataclass(slots=True, frozen=True)
class PreparedAudioSource:
    """Prepared import source plus temporary files to clean up after import."""

    source_path: Path
    cleanup_paths: tuple[Path, ...] = ()
    retained_paths: tuple[Path, ...] = ()
    program_artifact_path: Path | None = None
    ltc_artifact_path: Path | None = None


def compute_audio_hash(source_path: Path) -> str:
    """Compute SHA-256 hash of an audio file. Returns hex digest."""

    h = hashlib.sha256()
    with open(source_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def import_audio(
    source_path: Path,
    working_dir: Path,
    *,
    audio_import_options: AudioImportOptions | None = None,
) -> tuple[str, str]:
    """Copy an audio file into the project's audio directory with content-addressed naming.

    Args:
        source_path: Absolute path to the source audio file
        working_dir: ProjectRecord working directory
        audio_import_options: Optional preprocessing options.

    Returns:
        (project_relative_path, sha256_hash)
        e.g. ('audio/a3f2c8d1e9b04f71.wav', 'a3f2c8d1e9...')

    If the file already exists (same hash), skips the copy (dedup).
    """

    if audio_import_options is None:
        return _import_audio_file(source_path, working_dir)

    prepared = prepare_audio_for_import(
        source_path,
        working_dir,
        options=audio_import_options,
    )
    try:
        return _import_audio_file(prepared.source_path, working_dir)
    finally:
        cleanup_prepared_audio(prepared)


def _import_audio_file(source_path: Path, working_dir: Path) -> tuple[str, str]:
    audio_dir = working_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    audio_hash = compute_audio_hash(source_path)
    suffix = source_path.suffix or ".wav"
    dest_name = f"{audio_hash[:16]}{suffix}"
    dest_path = audio_dir / dest_name

    if not dest_path.exists():
        shutil.copy2(source_path, dest_path)

    return f"audio/{dest_name}", audio_hash


def prepare_audio_for_import(
    source_path: Path,
    working_dir: Path,
    *,
    options: AudioImportOptions | None = None,
    scan_fn=None,
) -> PreparedAudioSource:
    """Apply import-time preprocessing and return the prepared source path."""

    _ = scan_fn
    resolved_options = options or AudioImportOptions()
    if not resolved_options.strip_ltc_timecode:
        return PreparedAudioSource(source_path=source_path)

    ltc_channel = resolved_options.ltc_channel_override
    if ltc_channel is None:
        ltc_channel = detect_ltc_channel(
            source_path,
            mode=resolved_options.ltc_detection_mode,
        )
    if ltc_channel is None:
        return PreparedAudioSource(source_path=source_path)

    # If LTC is left, program audio is right, and vice versa.
    ltc_channel_index = 0 if ltc_channel == "left" else 1
    program_channel_index = 1 - ltc_channel_index
    program_channel_name = "right" if ltc_channel == "left" else "left"

    program_artifact_path, ltc_artifact_path = _split_artifact_paths(
        source_path=source_path,
        working_dir=working_dir,
        ltc_channel=ltc_channel,
        program_channel=program_channel_name,
    )

    staged_program_path: Path | None = None
    staged_ltc_path: Path | None = None
    cleanup_paths: list[Path] = []
    try:
        staged_program_path = _write_import_channel_copy(
            source_path,
            working_dir=working_dir,
            channel_index=program_channel_index,
        )
        cleanup_paths.append(staged_program_path)

        staged_ltc_path = _write_import_channel_copy(
            source_path,
            working_dir=working_dir,
            channel_index=ltc_channel_index,
        )
        cleanup_paths.append(staged_ltc_path)

        _persist_split_artifact(
            staged_path=staged_program_path,
            artifact_path=program_artifact_path,
        )
        _persist_split_artifact(
            staged_path=staged_ltc_path,
            artifact_path=ltc_artifact_path,
        )
    except Exception:
        for path in cleanup_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
        raise

    normalized_cleanup = tuple(
        path
        for path in cleanup_paths
        if path not in {program_artifact_path, ltc_artifact_path}
    )
    return PreparedAudioSource(
        source_path=program_artifact_path,
        cleanup_paths=normalized_cleanup,
        retained_paths=(program_artifact_path, ltc_artifact_path),
        program_artifact_path=program_artifact_path,
        ltc_artifact_path=ltc_artifact_path,
    )


def cleanup_prepared_audio(prepared: PreparedAudioSource) -> None:
    """Best-effort cleanup for temporary prepared audio files."""

    for path in prepared.cleanup_paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def detect_ltc_channel(
    source_path: Path,
    *,
    mode: Literal["strict", "aggressive"] = "strict",
) -> str | None:
    """Detect whether a stereo file appears to contain LTC on left or right channel.

    Returns:
        "left" if the left channel appears to be LTC,
        "right" if the right channel appears to be LTC,
        None when no confident LTC channel is detected.
    """

    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return None

    try:
        info = sf.info(str(source_path))
    except Exception:
        return None
    if int(getattr(info, "channels", 0)) < 2:
        return None

    frames_to_scan = int(min(max(int(info.samplerate) * 20, 0), int(info.frames)))
    if frames_to_scan <= 0:
        return None

    try:
        samples, sample_rate = sf.read(
            str(source_path),
            frames=frames_to_scan,
            dtype="float32",
            always_2d=True,
        )
    except Exception:
        return None

    if getattr(samples, "ndim", 0) != 2 or samples.shape[1] < 2:
        return None

    left_score = _ltc_score(np.asarray(samples[:, 0], dtype=np.float32), int(sample_rate))
    right_score = _ltc_score(np.asarray(samples[:, 1], dtype=np.float32), int(sample_rate))
    score_delta = abs(left_score - right_score)
    best_score = max(left_score, right_score)

    # Require one clearly LTC-like channel and a meaningful left/right separation.
    min_best_score, min_score_delta = _ltc_detection_thresholds(mode)
    if best_score < min_best_score or score_delta < min_score_delta:
        return None
    return "left" if left_score > right_score else "right"


def _ltc_detection_thresholds(mode: Literal["strict", "aggressive"]) -> tuple[float, float]:
    """Resolve LTC channel detection thresholds for one import mode."""

    if mode == "aggressive":
        return (0.50, 0.06)
    return (0.62, 0.12)


def _ltc_score(channel_samples, sample_rate: int) -> float:
    """Heuristic LTC likeness score in [0, 1]. Higher means more LTC-like."""

    import numpy as np

    if channel_samples.size < 2048:
        return 0.0

    peak = float(np.max(np.abs(channel_samples)))
    if peak <= 1e-6:
        return 0.0

    normalized = channel_samples / peak
    sign_changes = np.not_equal(np.signbit(normalized[1:]), np.signbit(normalized[:-1]))
    zero_cross_rate = float(np.mean(sign_changes))
    edge_component = min(1.0, zero_cross_rate * 12.0)

    high_level_ratio = float(np.mean(np.abs(normalized) >= 0.6))

    window = normalized[: min(normalized.size, sample_rate * 8)]
    if window.size < 1024:
        return 0.0
    windowed = window * np.hanning(window.size)
    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    if power.size <= 1:
        return 0.0
    freqs = np.fft.rfftfreq(window.size, d=1.0 / max(sample_rate, 1))
    total_power = float(np.sum(power)) + 1e-12
    high_power = float(np.sum(power[freqs >= 2000.0]))
    high_frequency_ratio = high_power / total_power

    score = (
        0.45 * edge_component
        + 0.35 * min(1.0, high_level_ratio)
        + 0.20 * min(1.0, high_frequency_ratio)
    )
    return float(max(0.0, min(1.0, score)))


def _write_import_channel_copy(
    source_path: Path,
    *,
    working_dir: Path,
    channel_index: int,
) -> Path:
    """Write one mono channel copy from a stereo source for import preprocessing."""

    import soundfile as sf

    temp_dir = working_dir / ".import_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        prefix="ltc_strip_",
        dir=temp_dir,
        delete=False,
    ) as handle:
        output_path = Path(handle.name)

    with sf.SoundFile(str(source_path), mode="r") as source_file:
        if int(source_file.channels) < 2:
            raise RuntimeError(
                f"Expected stereo source for LTC stripping: {source_path}"
            )

        writer_kwargs: dict[str, object] = {
            "mode": "w",
            "samplerate": int(source_file.samplerate),
            "channels": 1,
            "format": "WAV",
        }
        if source_file.subtype in {
            "PCM_16",
            "PCM_24",
            "PCM_32",
            "FLOAT",
            "DOUBLE",
        }:
            writer_kwargs["subtype"] = source_file.subtype

        with sf.SoundFile(str(output_path), **writer_kwargs) as output_file:
            for block in source_file.blocks(
                blocksize=65536,
                dtype="float32",
                always_2d=True,
            ):
                output_file.write(block[:, channel_index])

    return output_path


def _split_artifact_paths(
    *,
    source_path: Path,
    working_dir: Path,
    ltc_channel: str,
    program_channel: str,
) -> tuple[Path, Path]:
    """Resolve deterministic project-scoped paths for split LTC/program channels."""

    audio_dir = working_dir / "audio" / "split_channels"
    audio_dir.mkdir(parents=True, exist_ok=True)
    source_hash = compute_audio_hash(source_path)[:16]
    program_artifact = audio_dir / f"{source_hash}_program_{program_channel}.wav"
    ltc_artifact = audio_dir / f"{source_hash}_ltc_{ltc_channel}.wav"
    return program_artifact, ltc_artifact


def _persist_split_artifact(*, staged_path: Path, artifact_path: Path) -> None:
    """Persist one staged split channel artifact into project scope."""

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        return
    shutil.copy2(staged_path, artifact_path)


def verify_audio(working_dir: Path, audio_file: str, expected_hash: str) -> bool:
    """Verify an audio file's integrity by checking its hash.

    Args:
        working_dir: ProjectRecord working directory
        audio_file: ProjectRecord-relative path (e.g. 'audio/a3f2c8d1e9b04f71.wav')
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
