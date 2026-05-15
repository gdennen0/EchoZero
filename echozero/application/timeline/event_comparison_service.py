"""Event comparison service for selecting similar timeline events."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.event_similarity_audio import (
    audio_shape_preview,
    compare_shape_similarity,
    read_mono_audio_slice,
)
from echozero.application.timeline.models import Event, EventRef, Layer, Take


@dataclass(frozen=True, slots=True)
class TimbreFingerprintSettings:
    sample_count: int = 64
    padding_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class SavedTimbreMiniModelSettings:
    """Settings needed to score candidates against a saved local timbre prototype."""

    artifact_path: str | Path
    sample_count: int = 64
    padding_ms: float = 20.0
    centroid: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class EventComparisonCandidateRecord:
    layer_id: LayerId
    take_id: TakeId
    event: Event
    layer: Layer
    take: Take


@dataclass(frozen=True, slots=True)
class EventComparisonRequest:
    anchor_event_id: EventId
    comparison_mode: str = "shape_envelope"
    similarity_threshold: float = 0.85
    comparison_settings: object | None = None


PreviewBuilder = Callable[[tuple[np.ndarray, int], object], tuple[float, ...]]
ScoreBuilder = Callable[[tuple[float, ...], tuple[float, ...], object], float]


@dataclass(frozen=True, slots=True)
class EventComparisonMethod:
    """Registered strategy for one find-similar comparison method."""

    mode: str
    label: str
    preview_builder: PreviewBuilder
    score_builder: ScoreBuilder
    settings_factory: Callable[[object | None], object]
    requires_anchor_preview: bool = True


class EventComparisonService:
    """Select events whose audio comparison fingerprint matches an anchor."""

    def select_matching_event_refs(
        self,
        *,
        anchor_layer: Layer,
        anchor_take: Take,
        candidate_records: list[EventComparisonCandidateRecord],
        request: EventComparisonRequest,
    ) -> tuple[EventRef, ...]:
        anchor = next((event for event in anchor_take.events if event.id == request.anchor_event_id), None)
        if anchor is None:
            return ()
        method = comparison_method_for_mode(request.comparison_mode)
        settings = method.settings_factory(request.comparison_settings)
        cache: dict[str, tuple[np.ndarray, int]] = {}
        anchor_preview: tuple[float, ...] | None = None
        if method.requires_anchor_preview:
            anchor_preview = _event_preview(anchor_take, anchor, method=method, settings=settings, cache=cache)
            if anchor_preview is None:
                return (EventRef(anchor_layer.id, anchor_take.id, anchor.id),)
        elif isinstance(settings, SavedTimbreMiniModelSettings):
            anchor_preview = _saved_model_centroid(settings)
            if not anchor_preview:
                return (EventRef(anchor_layer.id, anchor_take.id, anchor.id),)

        matches: list[EventRef] = []
        threshold = max(0.0, min(1.0, float(request.similarity_threshold)))
        for record in candidate_records:
            preview = _event_preview(record.take, record.event, method=method, settings=settings, cache=cache)
            is_anchor_candidate = record.event.id == anchor.id and record.take.id == anchor_take.id
            score = 1.0 if is_anchor_candidate else 0.0
            if preview is not None and anchor_preview is not None and not is_anchor_candidate:
                score = method.score_builder(anchor_preview, preview, settings)
            if score >= threshold:
                matches.append(EventRef(record.layer_id, record.take_id, record.event.id))
        if not matches:
            matches.append(EventRef(anchor_layer.id, anchor_take.id, anchor.id))
        return tuple(matches)


def comparison_method_for_mode(mode: str | None) -> EventComparisonMethod:
    normalized = normalize_comparison_mode(mode)
    return COMPARISON_METHODS[normalized]


def normalize_comparison_mode(mode: str | None) -> str:
    normalized = (mode or "").strip().lower()
    if not normalized:
        normalized = "shape_envelope"
    if normalized not in COMPARISON_METHODS:
        allowed = ", ".join(sorted(COMPARISON_METHODS))
        raise ValueError(f"Unsupported comparison_mode {mode!r}; expected one of: {allowed}")
    return normalized


def build_timbre_fingerprint_preview(
    *,
    audio_path: str | None,
    start_seconds: float,
    end_seconds: float,
    settings: TimbreFingerprintSettings,
    audio_cache: dict[str, tuple[np.ndarray, int]] | None = None,
) -> tuple[float, ...] | None:
    audio = _read_cached_slice(
        audio_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        settings=settings,
        audio_cache=audio_cache,
    )
    if audio is None:
        return None
    samples, sample_rate = audio
    return _timbre_fingerprint(samples, sample_rate=sample_rate, sample_count=max(8, int(settings.sample_count)))


def compare_timbre_fingerprint_similarity(
    reference: tuple[float, ...], candidate: tuple[float, ...]
) -> float:
    ref = _unit_vector(reference)
    cand = _unit_vector(candidate)
    if ref.size == 0 or cand.size == 0:
        return 0.0
    if cand.size != ref.size:
        cand = np.interp(
            np.linspace(0.0, 1.0, ref.size),
            np.linspace(0.0, 1.0, cand.size),
            cand,
        ).astype(np.float32)
        cand = _unit_array(cand)
    return max(0.0, min(1.0, float(np.dot(ref, cand))))


def _coerce_fingerprint_settings(settings: object | None) -> TimbreFingerprintSettings:
    if isinstance(settings, TimbreFingerprintSettings):
        return settings
    if isinstance(settings, SavedTimbreMiniModelSettings):
        return TimbreFingerprintSettings(sample_count=settings.sample_count, padding_ms=settings.padding_ms)
    if isinstance(settings, dict):
        return TimbreFingerprintSettings(
            sample_count=int(settings.get("sample_count", 64)),
            padding_ms=float(settings.get("padding_ms", 0.0)),
        )
    return TimbreFingerprintSettings()


def _coerce_saved_model_settings(settings: object | None) -> SavedTimbreMiniModelSettings:
    if isinstance(settings, SavedTimbreMiniModelSettings):
        return settings
    if isinstance(settings, (str, Path)):
        return _settings_from_saved_model(Path(settings))
    if isinstance(settings, dict):
        artifact_path = settings.get("artifact_path") or settings.get("mini_model_path")
        if not artifact_path:
            raise ValueError("Saved mini-model comparison requires an artifact_path")
        if settings.get("centroid"):
            return SavedTimbreMiniModelSettings(
                artifact_path=artifact_path,
                sample_count=int(settings.get("sample_count", 64)),
                padding_ms=float(settings.get("padding_ms", 20.0)),
                centroid=tuple(float(value) for value in settings.get("centroid", ())),
            )
        return _settings_from_saved_model(Path(str(artifact_path)))
    raise ValueError("Saved mini-model comparison requires SavedTimbreMiniModelSettings or artifact_path")


def _settings_from_saved_model(artifact_path: Path) -> SavedTimbreMiniModelSettings:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    if payload.get("schema") != "echozero.find-similar-mini-model.v1":
        raise ValueError(f"Unsupported mini-model schema: {payload.get('schema')!r}")
    if payload.get("model_kind") != "timbre_prototype":
        raise ValueError(f"Unsupported mini-model kind: {payload.get('model_kind')!r}")
    centroid = payload.get("centroid")
    if not isinstance(centroid, list) or not centroid:
        raise ValueError("Mini-model artifact is missing a centroid")
    settings = payload.get("settings") or {}
    return SavedTimbreMiniModelSettings(
        artifact_path=artifact_path,
        sample_count=int(settings.get("sample_count", 64)),
        padding_ms=float(settings.get("padding_ms", 20.0)),
        centroid=tuple(float(value) for value in centroid),
    )


def _event_preview(
    take: Take,
    event: Event,
    *,
    method: EventComparisonMethod,
    settings: object,
    cache: dict[str, tuple[np.ndarray, int]],
) -> tuple[float, ...] | None:
    audio = _read_cached_slice(
        _take_audio_path(take),
        start_seconds=float(event.start),
        end_seconds=float(event.end),
        settings=_coerce_fingerprint_settings(settings),
        audio_cache=cache,
    )
    if audio is None:
        return None
    return method.preview_builder(audio, settings)


def _read_cached_slice(
    audio_path: str | None,
    *,
    start_seconds: float,
    end_seconds: float,
    settings: TimbreFingerprintSettings,
    audio_cache: dict[str, tuple[np.ndarray, int]] | None = None,
) -> tuple[np.ndarray, int] | None:
    if not audio_path:
        return None
    padding = max(0.0, float(settings.padding_ms) / 1000.0)
    cache_key = f"{audio_path}|{max(0.0, float(start_seconds) - padding):.6f}|{max(float(start_seconds), float(end_seconds) + padding):.6f}"
    if audio_cache is not None and cache_key in audio_cache:
        return audio_cache[cache_key]
    sliced = read_mono_audio_slice(
        audio_path,
        start_seconds=max(0.0, float(start_seconds) - padding),
        end_seconds=max(float(start_seconds), float(end_seconds) + padding),
    )
    if sliced is not None and audio_cache is not None:
        audio_cache[cache_key] = sliced
    return sliced


def _shape_preview_from_audio(audio: tuple[np.ndarray, int], settings: object) -> tuple[float, ...]:
    samples, _sample_rate = audio
    resolved = _coerce_fingerprint_settings(settings)
    return audio_shape_preview(samples, sample_count=max(8, int(resolved.sample_count)))


def _timbre_preview_from_audio(audio: tuple[np.ndarray, int], settings: object) -> tuple[float, ...]:
    samples, sample_rate = audio
    resolved = _coerce_fingerprint_settings(settings)
    return _timbre_fingerprint(samples, sample_rate=sample_rate, sample_count=max(8, int(resolved.sample_count)))


def _shape_score(anchor_preview: tuple[float, ...], candidate_preview: tuple[float, ...], _settings: object) -> float:
    return compare_shape_similarity(anchor_preview, candidate_preview)


def _timbre_score(anchor_preview: tuple[float, ...], candidate_preview: tuple[float, ...], _settings: object) -> float:
    return compare_timbre_fingerprint_similarity(anchor_preview, candidate_preview)


def _saved_model_score(
    anchor_preview: tuple[float, ...],
    candidate_preview: tuple[float, ...],
    settings: object,
) -> float:
    centroid = _saved_model_centroid(settings)
    return compare_timbre_fingerprint_similarity(centroid or anchor_preview, candidate_preview)


def _saved_model_centroid(settings: object) -> tuple[float, ...]:
    if isinstance(settings, SavedTimbreMiniModelSettings):
        return tuple(float(value) for value in settings.centroid)
    return ()


def _timbre_fingerprint(
    audio: np.ndarray,
    *,
    sample_rate: int,
    sample_count: int,
) -> tuple[float, ...]:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return ()
    envelope = np.asarray(audio_shape_preview(arr, sample_count=sample_count), dtype=np.float32)
    spectrum = np.abs(np.fft.rfft(arr * np.hanning(arr.size))).astype(np.float32)
    if spectrum.size <= 1 or float(np.sum(spectrum)) <= 1e-9:
        spectral = np.zeros(sample_count, dtype=np.float32)
    else:
        freqs = np.fft.rfftfreq(arr.size, d=1.0 / max(1, int(sample_rate))).astype(np.float32)
        # Compress frequency energy into fixed bands. Log-ish interpolation keeps one-shot
        # low/high timbres distinct while staying cheap and dependency-free.
        band_x = np.linspace(0.0, 1.0, sample_count)
        source_x = np.linspace(0.0, 1.0, spectrum.size)
        spectral = np.interp(band_x, source_x, spectrum).astype(np.float32)
        centroid = float(np.sum(freqs * spectrum) / max(1e-9, float(np.sum(spectrum))))
        spectral = _unit_array(spectral) * 0.75
        centroid_feature = np.full(sample_count // 4 or 1, centroid / max(1.0, sample_rate / 2.0), dtype=np.float32)
        return tuple(float(x) for x in _unit_array(np.concatenate((envelope, spectral, centroid_feature))))
    return tuple(float(x) for x in _unit_array(np.concatenate((envelope, spectral))))


def _unit_vector(value: tuple[float, ...]) -> np.ndarray:
    return _unit_array(np.asarray(value, dtype=np.float32).reshape(-1))


def _unit_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    arr = arr - float(arr.min())
    peak = float(arr.max())
    if peak > 1e-9:
        arr = arr / peak
    norm = float(np.linalg.norm(arr))
    if norm > 1e-9:
        arr = arr / norm
    return arr.astype(np.float32, copy=False)


def _take_audio_path(take: Take) -> str | None:
    source_ref = getattr(take, "source_content_ref", None)
    locator = getattr(source_ref, "locator", None)
    if locator and Path(str(locator)).exists():
        return str(locator)
    return None


COMPARISON_METHODS: dict[str, EventComparisonMethod] = {
    "shape_envelope": EventComparisonMethod(
        mode="shape_envelope",
        label="Shape Envelope",
        preview_builder=_shape_preview_from_audio,
        score_builder=_shape_score,
        settings_factory=_coerce_fingerprint_settings,
    ),
    "timbre_fingerprint": EventComparisonMethod(
        mode="timbre_fingerprint",
        label="Timbre Fingerprint",
        preview_builder=_timbre_preview_from_audio,
        score_builder=_timbre_score,
        settings_factory=_coerce_fingerprint_settings,
    ),
    "timbre_mini_model": EventComparisonMethod(
        mode="timbre_mini_model",
        label="Saved Mini-model",
        preview_builder=_timbre_preview_from_audio,
        score_builder=_saved_model_score,
        settings_factory=_coerce_saved_model_settings,
        requires_anchor_preview=False,
    ),
}


__all__ = [
    "COMPARISON_METHODS",
    "EventComparisonCandidateRecord",
    "EventComparisonMethod",
    "EventComparisonRequest",
    "EventComparisonService",
    "SavedTimbreMiniModelSettings",
    "TimbreFingerprintSettings",
    "build_timbre_fingerprint_preview",
    "compare_timbre_fingerprint_similarity",
    "comparison_method_for_mode",
    "normalize_comparison_mode",
]
