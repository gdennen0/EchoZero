"""ReviewAudioClipService materializes deterministic event clips for Foundry review.
Exists because phone review must replay one explicit event window, not a full song file.
Connects project-backed review queues to cached clip files with safe staleness keys.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class ReviewAudioClipService:
    """Builds deterministic cached audio clips for review items."""

    def __init__(self) -> None:
        self._source_cache: dict[tuple[Path, int, int], tuple[Any, int]] = {}

    def materialize_event_clip(
        self,
        *,
        source_audio_path: Path,
        clip_cache_dir: Path,
        clip_stem: str,
        start_seconds: float,
        end_seconds: float,
    ) -> Path | None:
        """Materialize one event window to a deterministic cached clip file."""
        resolved_source = source_audio_path.expanduser().resolve()
        normalized_start = max(0.0, float(start_seconds))
        normalized_end = max(normalized_start, float(end_seconds))
        if normalized_end <= normalized_start:
            return None
        if not resolved_source.exists() or not resolved_source.is_file():
            return None

        source_stat = resolved_source.stat()
        clip_path = self._clip_path(
            source_audio_path=resolved_source,
            source_stat=source_stat,
            clip_cache_dir=clip_cache_dir,
            clip_stem=clip_stem,
            start_seconds=normalized_start,
            end_seconds=normalized_end,
        )
        if clip_path.exists() and clip_path.stat().st_size > 0:
            return clip_path

        samples, sample_rate = self._load_audio(
            source_audio_path=resolved_source,
            source_stat=source_stat,
        )
        total_frames = int(samples.shape[0]) if getattr(samples, "ndim", 0) >= 1 else 0
        if total_frames <= 0 or sample_rate <= 0:
            return None

        start_frame = min(total_frames, max(0, int(round(normalized_start * sample_rate))))
        end_frame = min(total_frames, max(start_frame, int(round(normalized_end * sample_rate))))
        if end_frame <= start_frame:
            return None

        clip = samples[start_frame:end_frame]
        if getattr(clip, "size", 0) <= 0:
            return None

        clip_path.parent.mkdir(parents=True, exist_ok=True)
        self._soundfile().write(str(clip_path), clip, sample_rate, format="WAV")
        return clip_path

    def _load_audio(self, *, source_audio_path: Path, source_stat: Any) -> tuple[Any, int]:
        cache_key = (
            source_audio_path,
            int(source_stat.st_mtime_ns),
            int(source_stat.st_size),
        )
        cached = self._source_cache.get(cache_key)
        if cached is not None:
            return cached

        samples, sample_rate = self._soundfile().read(
            str(source_audio_path),
            always_2d=False,
            dtype="float32",
        )
        payload = (samples, int(sample_rate))
        self._source_cache[cache_key] = payload
        return payload

    @classmethod
    def _clip_path(
        cls,
        *,
        source_audio_path: Path,
        source_stat: Any,
        clip_cache_dir: Path,
        clip_stem: str,
        start_seconds: float,
        end_seconds: float,
    ) -> Path:
        digest = hashlib.sha1(
            (
                f"{source_audio_path.resolve()}|{source_stat.st_mtime_ns}|"
                f"{source_stat.st_size}|{start_seconds:.9f}|{end_seconds:.9f}"
            ).encode("utf-8")
        ).hexdigest()[:20]
        safe_stem = cls._safe_stem(clip_stem)
        return clip_cache_dir / f"{safe_stem}_{digest}.wav"

    @staticmethod
    def _safe_stem(value: str) -> str:
        text = str(value).strip() or "review_clip"
        return "".join(character if character.isalnum() else "_" for character in text)

    @staticmethod
    def _soundfile() -> Any:
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "Review clip materialization requires soundfile. Install with: pip install soundfile"
            ) from exc
        return sf
