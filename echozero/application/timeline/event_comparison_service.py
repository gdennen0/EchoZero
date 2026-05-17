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


def _hybrid_mir_preview_from_audio(audio: tuple[np.ndarray, int], settings: object) -> tuple[float, ...]:
    return _timbre_preview_from_audio(audio, settings)


def _shape_score(anchor_preview: tuple[float, ...], candidate_preview: tuple[float, ...], _settings: object) -> float:
    return compare_shape_similarity(anchor_preview, candidate_preview)


def _timbre_score(anchor_preview: tuple[float, ...], candidate_preview: tuple[float, ...], _settings: object) -> float:
    return compare_timbre_fingerprint_similarity(anchor_preview, candidate_preview)


def _hybrid_mir_score(
    anchor_preview: tuple[float, ...], candidate_preview: tuple[float, ...], settings: object
) -> float:
    resolved = _coerce_fingerprint_settings(settings)
    return _compare_hybrid_mir_similarity(
        anchor_preview,
        candidate_preview,
        sample_count=max(8, int(resolved.sample_count)),
    )


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
    transient = _transient_preview(arr, sample_count=sample_count)
    spectrum = np.abs(np.fft.rfft(arr * np.hanning(arr.size))).astype(np.float32)
    percussive = _percussive_component(arr)
    percussive_spectrum = np.abs(np.fft.rfft(percussive * np.hanning(percussive.size))).astype(np.float32)
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / max(1, int(sample_rate))).astype(np.float32)
    spectral = _compress_spectrum(spectrum, sample_count=sample_count)
    percussive_bands = _compress_spectrum(percussive_spectrum, sample_count=sample_count)
    stats = _spectral_statistics(
        arr=arr,
        envelope=envelope,
        transient=transient,
        spectrum=spectrum,
        percussive_spectrum=percussive_spectrum,
        freqs=freqs,
        sample_rate=sample_rate,
    )
    return tuple(
        float(x)
        for x in _unit_array(np.concatenate((envelope, transient, spectral, percussive_bands, stats)))
    )


def _transient_preview(audio: np.ndarray, *, sample_count: int) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(sample_count, dtype=np.float32)
    onset = np.maximum(0.0, np.diff(np.abs(arr), prepend=np.abs(arr[:1])))
    return np.asarray(audio_shape_preview(onset, sample_count=sample_count), dtype=np.float32)


def _compress_spectrum(spectrum: np.ndarray, *, sample_count: int) -> np.ndarray:
    values = np.asarray(spectrum, dtype=np.float32).reshape(-1)
    if values.size == 0 or float(np.sum(values)) <= 1e-9:
        return np.zeros(sample_count, dtype=np.float32)
    compressed = np.interp(
        np.linspace(0.0, 1.0, sample_count),
        np.linspace(0.0, 1.0, values.size),
        np.log1p(values),
    ).astype(np.float32)
    return _unit_array(compressed)


def _percussive_component(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size <= 1:
        return arr
    kernel = max(5, min(129, (arr.size // 32) | 1))
    smooth = np.convolve(arr, np.ones(kernel, dtype=np.float32) / float(kernel), mode="same")
    return (arr - smooth).astype(np.float32, copy=False)


def _spectral_statistics(
    *,
    arr: np.ndarray,
    envelope: np.ndarray,
    transient: np.ndarray,
    spectrum: np.ndarray,
    percussive_spectrum: np.ndarray,
    freqs: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    energy = float(np.sum(spectrum))
    if spectrum.size <= 1 or energy <= 1e-9:
        return np.zeros(12, dtype=np.float32)
    centroid = float(np.sum(freqs * spectrum) / energy)
    spread = float(np.sqrt(np.sum(np.square(freqs - centroid) * spectrum) / energy))
    stats = np.asarray(
        (
            centroid / max(1.0, sample_rate / 2.0),
            spread / max(1.0, sample_rate / 2.0),
            _spectral_rolloff_hz(freqs, spectrum, percentile=0.85) / max(1.0, sample_rate / 2.0),
            _spectral_rolloff_hz(freqs, spectrum, percentile=0.95) / max(1.0, sample_rate / 2.0),
            _spectral_flatness(spectrum),
            _band_energy_ratio(freqs, spectrum, 40.0, 180.0),
            _band_energy_ratio(freqs, spectrum, 180.0, 900.0),
            _band_energy_ratio(freqs, spectrum, 1500.0, 6000.0),
            _band_energy_ratio(freqs, spectrum, 6000.0, max(6000.0, sample_rate / 2.0)),
            _band_energy_ratio(freqs, percussive_spectrum, 1800.0, 8000.0),
            float(np.mean(np.abs(np.diff(np.signbit(arr).astype(np.float32))))) if arr.size > 1 else 0.0,
            _attack_descriptor(envelope, transient),
        ),
        dtype=np.float32,
    )
    return np.clip(stats, 0.0, 1.0)


def _spectral_rolloff_hz(freqs: np.ndarray, spectrum: np.ndarray, *, percentile: float) -> float:
    cumulative = np.cumsum(np.asarray(spectrum, dtype=np.float32))
    if cumulative.size == 0 or float(cumulative[-1]) <= 1e-9:
        return 0.0
    target = float(percentile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(freqs[max(0, min(index, len(freqs) - 1))])


def _spectral_flatness(spectrum: np.ndarray) -> float:
    values = np.maximum(np.asarray(spectrum, dtype=np.float32).reshape(-1), 1e-10)
    if values.size == 0:
        return 0.0
    geometric = float(np.exp(np.mean(np.log(values))))
    arithmetic = float(np.mean(values))
    if arithmetic <= 1e-10:
        return 0.0
    return float(max(0.0, min(1.0, geometric / arithmetic)))


def _band_energy_ratio(freqs: np.ndarray, spectrum: np.ndarray, low_hz: float, high_hz: float) -> float:
    if high_hz <= low_hz:
        return 0.0
    total = float(np.sum(spectrum))
    if total <= 1e-9:
        return 0.0
    mask = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(mask):
        return 0.0
    return float(np.sum(spectrum[mask]) / total)


def _attack_descriptor(envelope: np.ndarray, transient: np.ndarray) -> float:
    env = np.asarray(envelope, dtype=np.float32).reshape(-1)
    trans = np.asarray(transient, dtype=np.float32).reshape(-1)
    if env.size == 0:
        return 0.0
    attack_window = max(1, env.size // 5)
    early_peak = float(np.max(env[:attack_window]))
    transient_peak = float(np.max(trans[:attack_window])) if trans.size else 0.0
    return float(max(0.0, min(1.0, 0.65 * early_peak + 0.35 * transient_peak)))


def _compare_hybrid_mir_similarity(
    reference: tuple[float, ...],
    candidate: tuple[float, ...],
    *,
    sample_count: int,
) -> float:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float32).reshape(-1)
    section_count = sample_count * 4 + 12
    if ref.size == 0 or cand.size == 0:
        return 0.0
    if ref.size < section_count or cand.size < section_count:
        return compare_timbre_fingerprint_similarity(reference, candidate)
    if ref.size != cand.size:
        cand = np.interp(
            np.linspace(0.0, 1.0, ref.size),
            np.linspace(0.0, 1.0, cand.size),
            cand,
        ).astype(np.float32)
    ref_env, ref_transient, ref_spectral, ref_percussive, ref_stats = _split_hybrid_sections(
        ref, sample_count=sample_count
    )
    cand_env, cand_transient, cand_spectral, cand_percussive, cand_stats = _split_hybrid_sections(
        cand, sample_count=sample_count
    )
    stats_gap = float(np.mean(np.abs(ref_stats - cand_stats)))
    return float(
        max(
            0.0,
            min(
                1.0,
                0.18 * float(np.dot(_unit_array(ref_env), _unit_array(cand_env)))
                + 0.17 * float(np.dot(_unit_array(ref_transient), _unit_array(cand_transient)))
                + 0.30 * float(np.dot(_unit_array(ref_spectral), _unit_array(cand_spectral)))
                + 0.23 * float(np.dot(_unit_array(ref_percussive), _unit_array(cand_percussive)))
                + 0.12 * (1.0 - stats_gap),
            ),
        )
    )


def _split_hybrid_sections(vector: np.ndarray, *, sample_count: int) -> tuple[np.ndarray, ...]:
    first = sample_count
    second = first + sample_count
    third = second + sample_count
    fourth = third + sample_count
    return (
        vector[:first],
        vector[first:second],
        vector[second:third],
        vector[third:fourth],
        vector[fourth : fourth + 12],
    )


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
    "hybrid_mir": EventComparisonMethod(
        mode="hybrid_mir",
        label="Hybrid MIR",
        preview_builder=_hybrid_mir_preview_from_audio,
        score_builder=_hybrid_mir_score,
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
