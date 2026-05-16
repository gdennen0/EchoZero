"""
Video import helpers for song timeline reference media.
Exists to store one project-local video and extract its audio as a non-playback reference.
Connects project storage to timeline presentation without teaching the engine about video.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Media metadata needed to render and sync a video reference."""

    duration_seconds: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    has_audio: bool = False


@dataclass(frozen=True, slots=True)
class ImportedVideo:
    """Project-local video import result with optional extracted reference audio."""

    video_file: str
    video_hash: str
    metadata: VideoMetadata
    extracted_audio_file: str | None
    extracted_audio_hash: str | None


def compute_video_hash(source_path: Path) -> str:
    """Compute SHA-256 for a video or extracted reference file."""

    h = hashlib.sha256()
    with open(source_path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def import_video(source_path: Path, working_dir: Path) -> ImportedVideo:
    """Copy a video into project storage and extract mono reference audio when present."""

    resolved = source_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Video file not found: {resolved}")

    metadata = scan_video_metadata(resolved)
    video_rel_path, video_hash = _copy_video_file(resolved, working_dir)
    extracted_audio_file: str | None = None
    extracted_audio_hash: str | None = None
    if metadata.has_audio:
        extracted_audio_path = _extract_reference_audio(resolved, working_dir, video_hash)
        extracted_audio_hash = compute_video_hash(extracted_audio_path)
        extracted_audio_file = _relative_to_working_dir(working_dir, extracted_audio_path)

    return ImportedVideo(
        video_file=video_rel_path,
        video_hash=video_hash,
        metadata=metadata,
        extracted_audio_file=extracted_audio_file,
        extracted_audio_hash=extracted_audio_hash,
    )


def scan_video_metadata(source_path: Path) -> VideoMetadata:
    """Read video metadata through ffprobe."""

    payload = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,width,height,r_frame_rate,duration",
            "-of",
            "json",
            str(source_path),
        ]
    )
    streams = payload.get("streams") if isinstance(payload, dict) else None
    stream_rows = streams if isinstance(streams, list) else []
    format_payload = payload.get("format") if isinstance(payload, dict) else {}
    format_duration = _optional_float(
        format_payload.get("duration") if isinstance(format_payload, dict) else None
    )
    video_stream = next(
        (row for row in stream_rows if isinstance(row, dict) and row.get("codec_type") == "video"),
        None,
    )
    audio_stream = next(
        (row for row in stream_rows if isinstance(row, dict) and row.get("codec_type") == "audio"),
        None,
    )
    duration = _optional_float(video_stream.get("duration") if video_stream else None)
    return VideoMetadata(
        duration_seconds=max(
            0.0,
            float(duration if duration is not None else format_duration or 0.0),
        ),
        width=_optional_int(video_stream.get("width") if video_stream else None),
        height=_optional_int(video_stream.get("height") if video_stream else None),
        fps=_fps_from_rate(video_stream.get("r_frame_rate") if video_stream else None),
        has_audio=audio_stream is not None,
    )


def resolve_project_video_path(working_dir: Path, video_file: str) -> Path:
    """Resolve a stored video path against the project working directory."""

    raw_path = Path(video_file)
    if raw_path.is_absolute():
        return raw_path
    return (working_dir / raw_path).resolve()


def _copy_video_file(source_path: Path, working_dir: Path) -> tuple[str, str]:
    video_dir = working_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_hash = compute_video_hash(source_path)
    suffix = source_path.suffix or ".mov"
    dest_path = video_dir / f"{video_hash[:16]}{suffix}"
    if not dest_path.exists():
        shutil.copy2(source_path, dest_path)
    return _relative_to_working_dir(working_dir, dest_path), video_hash


def _extract_reference_audio(source_path: Path, working_dir: Path, video_hash: str) -> Path:
    audio_dir = working_dir / "audio" / "video_refs"
    audio_dir.mkdir(parents=True, exist_ok=True)
    dest_path = audio_dir / f"{video_hash[:16]}_video_ref.wav"
    if dest_path.exists():
        return dest_path
    _run_checked(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-f",
            "wav",
            str(dest_path),
        ]
    )
    return dest_path


def _run_json(args: list[str]) -> dict[str, object]:
    completed = _run_checked(args)
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON while scanning video.") from exc
    return payload if isinstance(payload, dict) else {}


def _run_checked(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Required video tool '{args[0]}' was not found. Install ffmpeg/ffprobe."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"{args[0]} failed while processing video: {detail}") from exc


def _relative_to_working_dir(working_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(working_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _fps_from_rate(value: object) -> float | None:
    text = str(value or "").strip()
    if not text or text == "0/0":
        return None
    try:
        parsed = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return _optional_float(text)
    if parsed.denominator == 0:
        return None
    return float(parsed)


def _optional_int(value: object) -> int | None:
    try:
        return None if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None
