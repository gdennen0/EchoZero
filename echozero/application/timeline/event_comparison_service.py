"""
Generic event comparison service for timeline selection and review workflows.
Exists to keep event-comparison orchestration mode-agnostic while allowing one strategy per comparison type.
Connects scoped event resolution to strategy-based extraction, scoring, preview payloads, and selection results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import librosa
import numpy as np

from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.event_similarity_audio import (
    EventShapeBundle,
    ShapeNormalizationSettings,
    align_shape_to_reference,
    compare_shape_similarity,
    load_event_shape_bundle,
)
from echozero.application.timeline.models import Event, EventRef, Layer, Take

_DEFAULT_COMPARISON_MODE = "shape_envelope"
_TIMBRE_COMPARISON_MODE = "timbre_fingerprint"
_GRAPH_PREVIEW_KIND = "normalized_graph"
_FEATURE_SAMPLE_RATE = 22050
_MIN_EVENT_DURATION_SECONDS = 0.08


@dataclass(slots=True)
class EventComparisonCandidateRecord:
    """One candidate event that can be scored by a comparison strategy."""

    layer_id: LayerId
    take_id: TakeId
    event: Event
    layer: Layer
    take: Take


@dataclass(slots=True)
class EventComparisonRequest:
    """Operator-selected options for one event comparison run."""

    anchor_event_id: EventId
    similarity_threshold: float = 0.78
    comparison_mode: str = _DEFAULT_COMPARISON_MODE
    comparison_settings: object | None = None

    def __post_init__(self) -> None:
        self.similarity_threshold = max(0.0, min(1.0, float(self.similarity_threshold)))
        mode = str(self.comparison_mode or "").strip().lower()
        self.comparison_mode = mode or _DEFAULT_COMPARISON_MODE


@dataclass(slots=True)
class EventComparisonPreviewArtifact:
    """Optional strategy-specific preview payload for one scored event."""

    kind: str
    payload: object | None


@dataclass(slots=True)
class EventComparisonScoredCandidate:
    """One scored candidate with a strategy-specific preview artifact."""

    event_ref: EventRef
    score: float | None
    is_selected: bool
    preview_artifact: EventComparisonPreviewArtifact | None

    @property
    def similarity_percentage(self) -> float | None:
        if self.score is None:
            return None
        return self.score * 100.0

    @property
    def normalized_shape(self) -> tuple[float, ...] | None:
        """Compatibility alias for the current shape-envelope preview."""

        if self.preview_artifact is None or self.preview_artifact.kind != _GRAPH_PREVIEW_KIND:
            return None
        payload = self.preview_artifact.payload
        if isinstance(payload, tuple):
            return payload
        return None


@dataclass(slots=True)
class TimbreFingerprintSettings:
    """Controls how one event clip becomes a compact timbre descriptor."""

    sample_count: int = 64
    padding_ms: float = 20.0

    def __post_init__(self) -> None:
        self.sample_count = max(16, min(256, int(self.sample_count)))
        self.padding_ms = max(0.0, min(250.0, float(self.padding_ms)))


class EventComparisonStrategy(Protocol):
    """One pluggable event-comparison implementation."""

    mode_id: str

    def prepare_anchor(
        self,
        *,
        anchor_record: EventComparisonCandidateRecord,
        request: EventComparisonRequest,
        audio_cache: dict[str, tuple[np.ndarray, int]],
    ) -> object:
        ...

    def score_candidate(
        self,
        *,
        anchor_state: object,
        anchor_record: EventComparisonCandidateRecord,
        candidate_record: EventComparisonCandidateRecord,
        request: EventComparisonRequest,
        audio_cache: dict[str, tuple[np.ndarray, int]],
        is_anchor: bool,
    ) -> tuple[float | None, EventComparisonPreviewArtifact | None]:
        ...


@dataclass(slots=True)
class _ShapeEnvelopeAnchorState:
    bundle: EventShapeBundle | None


class ShapeEnvelopeComparisonStrategy:
    """Current normalized-envelope graph comparison strategy."""

    mode_id = _DEFAULT_COMPARISON_MODE

    def prepare_anchor(
        self,
        *,
        anchor_record: EventComparisonCandidateRecord,
        request: EventComparisonRequest,
        audio_cache: dict[str, tuple[np.ndarray, int]],
    ) -> _ShapeEnvelopeAnchorState:
        return _ShapeEnvelopeAnchorState(
            bundle=_load_candidate_shape_bundle(
                anchor_record,
                settings=self._settings_from_request(request),
                audio_cache=audio_cache,
            )
        )

    def score_candidate(
        self,
        *,
        anchor_state: object,
        anchor_record: EventComparisonCandidateRecord,
        candidate_record: EventComparisonCandidateRecord,
        request: EventComparisonRequest,
        audio_cache: dict[str, tuple[np.ndarray, int]],
        is_anchor: bool,
    ) -> tuple[float | None, EventComparisonPreviewArtifact | None]:
        if not isinstance(anchor_state, _ShapeEnvelopeAnchorState):
            return None, None
        if is_anchor:
            anchor_samples = (
                None if anchor_state.bundle is None else anchor_state.bundle.normalized_samples
            )
            return 1.0, EventComparisonPreviewArtifact(
                kind=_GRAPH_PREVIEW_KIND,
                payload=anchor_samples,
            )
        candidate_bundle = _load_candidate_shape_bundle(
            candidate_record,
            settings=self._settings_from_request(request),
            audio_cache=audio_cache,
        )
        if anchor_state.bundle is None or candidate_bundle is None:
            return None, None
        aligned_shape = align_shape_to_reference(
            anchor_state.bundle.normalized_samples,
            candidate_bundle.normalized_samples,
        )
        return compare_shape_similarity(
            anchor_state.bundle.normalized_samples,
            candidate_bundle.normalized_samples,
        ), EventComparisonPreviewArtifact(
            kind=_GRAPH_PREVIEW_KIND,
            payload=aligned_shape,
        )

    @staticmethod
    def _settings_from_request(request: EventComparisonRequest) -> ShapeNormalizationSettings:
        if isinstance(request.comparison_settings, ShapeNormalizationSettings):
            return request.comparison_settings
        return ShapeNormalizationSettings()


@dataclass(slots=True)
class _TimbreFingerprintAnchorState:
    fingerprint: tuple[float, ...] | None


class TimbreFingerprintComparisonStrategy:
    """Compact timbre-descriptor comparison for one-shot matching."""

    mode_id = _TIMBRE_COMPARISON_MODE

    def prepare_anchor(
        self,
        *,
        anchor_record: EventComparisonCandidateRecord,
        request: EventComparisonRequest,
        audio_cache: dict[str, tuple[np.ndarray, int]],
    ) -> _TimbreFingerprintAnchorState:
        return _TimbreFingerprintAnchorState(
            fingerprint=load_event_timbre_fingerprint(
                record=anchor_record,
                settings=self._settings_from_request(request),
                audio_cache=audio_cache,
            )
        )

    def score_candidate(
        self,
        *,
        anchor_state: object,
        anchor_record: EventComparisonCandidateRecord,
        candidate_record: EventComparisonCandidateRecord,
        request: EventComparisonRequest,
        audio_cache: dict[str, tuple[np.ndarray, int]],
        is_anchor: bool,
    ) -> tuple[float | None, EventComparisonPreviewArtifact | None]:
        if not isinstance(anchor_state, _TimbreFingerprintAnchorState):
            return None, None
        if is_anchor:
            return 1.0, EventComparisonPreviewArtifact(
                kind=_GRAPH_PREVIEW_KIND,
                payload=anchor_state.fingerprint,
            )
        candidate_fingerprint = load_event_timbre_fingerprint(
            record=candidate_record,
            settings=self._settings_from_request(request),
            audio_cache=audio_cache,
        )
        if anchor_state.fingerprint is None or candidate_fingerprint is None:
            return None, None
        return compare_timbre_fingerprint_similarity(
            anchor_state.fingerprint,
            candidate_fingerprint,
        ), EventComparisonPreviewArtifact(
            kind=_GRAPH_PREVIEW_KIND,
            payload=candidate_fingerprint,
        )

    @staticmethod
    def _settings_from_request(request: EventComparisonRequest) -> TimbreFingerprintSettings:
        settings = request.comparison_settings
        if isinstance(settings, TimbreFingerprintSettings):
            return settings
        if isinstance(settings, ShapeNormalizationSettings):
            return TimbreFingerprintSettings(
                sample_count=settings.sample_count,
                padding_ms=settings.padding_ms,
            )
        return TimbreFingerprintSettings()


class EventComparisonService:
    """Compute similar-event selections using a registered comparison strategy."""

    def __init__(
        self,
        *,
        strategies: tuple[EventComparisonStrategy, ...] | None = None,
    ) -> None:
        resolved_strategies = strategies or (
            ShapeEnvelopeComparisonStrategy(),
            TimbreFingerprintComparisonStrategy(),
        )
        self._strategies = {
            str(strategy.mode_id).strip().lower(): strategy
            for strategy in resolved_strategies
        }

    def analyze_candidates(
        self,
        *,
        anchor_layer: Layer,
        anchor_take: Take,
        candidate_records: list[EventComparisonCandidateRecord],
        request: EventComparisonRequest,
        on_progress: Callable[
            [int, int, EventComparisonScoredCandidate, EventComparisonCandidateRecord], None
        ]
        | None = None,
    ) -> list[EventComparisonScoredCandidate]:
        anchor_record = next(
            (
                record
                for record in candidate_records
                if record.layer_id == _layer_id(anchor_layer)
                and record.take_id == _take_id(anchor_take)
                and _event_id(record.event) == request.anchor_event_id
            ),
            None,
        )
        if anchor_record is None:
            return []
        strategy = self._strategies.get(request.comparison_mode)
        if strategy is None:
            raise ValueError(
                f"Unsupported event comparison mode: {request.comparison_mode!r}"
            )

        audio_cache: dict[str, tuple[np.ndarray, int]] = {}
        anchor_state = strategy.prepare_anchor(
            anchor_record=anchor_record,
            request=request,
            audio_cache=audio_cache,
        )
        total = len(candidate_records)
        results: list[EventComparisonScoredCandidate] = []
        selected_count = 0
        for index, record in enumerate(candidate_records, start=1):
            is_anchor = (
                record.layer_id == anchor_record.layer_id
                and record.take_id == anchor_record.take_id
                and _event_id(record.event) == _event_id(anchor_record.event)
            )
            score, preview_artifact = strategy.score_candidate(
                anchor_state=anchor_state,
                anchor_record=anchor_record,
                candidate_record=record,
                request=request,
                audio_cache=audio_cache,
                is_anchor=is_anchor,
            )
            is_selected = is_anchor or (
                score is not None and score >= request.similarity_threshold
            )
            result = EventComparisonScoredCandidate(
                event_ref=EventRef(
                    layer_id=record.layer_id,
                    take_id=record.take_id,
                    event_id=_event_id(record.event),
                ),
                score=score,
                is_selected=is_selected,
                preview_artifact=preview_artifact,
            )
            if is_selected:
                selected_count += 1
            results.append(result)
            if on_progress is not None:
                on_progress(index, total, result, record)

        if selected_count > 0:
            return results
        anchor_preview = None
        if isinstance(anchor_state, _ShapeEnvelopeAnchorState) and anchor_state.bundle is not None:
            anchor_preview = EventComparisonPreviewArtifact(
                kind=_GRAPH_PREVIEW_KIND,
                payload=anchor_state.bundle.normalized_samples,
            )
        elif (
            isinstance(anchor_state, _TimbreFingerprintAnchorState)
            and anchor_state.fingerprint is not None
        ):
            anchor_preview = EventComparisonPreviewArtifact(
                kind=_GRAPH_PREVIEW_KIND,
                payload=anchor_state.fingerprint,
            )
        return [
            EventComparisonScoredCandidate(
                event_ref=EventRef(
                    layer_id=anchor_record.layer_id,
                    take_id=anchor_record.take_id,
                    event_id=_event_id(anchor_record.event),
                ),
                score=1.0,
                is_selected=True,
                preview_artifact=anchor_preview,
            )
        ]

    def select_matching_event_refs(
        self,
        *,
        anchor_layer: Layer,
        anchor_take: Take,
        candidate_records: list[EventComparisonCandidateRecord],
        request: EventComparisonRequest,
    ) -> list[EventRef]:
        results = self.analyze_candidates(
            anchor_layer=anchor_layer,
            anchor_take=anchor_take,
            candidate_records=candidate_records,
            request=request,
        )
        return [result.event_ref for result in results if result.is_selected]


def _load_candidate_shape_bundle(
    record: EventComparisonCandidateRecord,
    *,
    settings: ShapeNormalizationSettings,
    audio_cache: dict[str, tuple[np.ndarray, int]],
) -> EventShapeBundle | None:
    audio_path = _resolve_audio_path(record.layer, record.take)
    if audio_path is None:
        return None
    try:
        return load_event_shape_bundle(
            audio_path=audio_path,
            start_seconds=record.event.start,
            end_seconds=record.event.end,
            settings=settings,
            audio_cache=audio_cache,
        )
    except Exception:
        return None


def load_event_timbre_fingerprint(
    *,
    record: EventComparisonCandidateRecord,
    settings: TimbreFingerprintSettings,
    audio_cache: dict[str, tuple[np.ndarray, int]],
) -> tuple[float, ...] | None:
    audio_path = _resolve_audio_path(record.layer, record.take)
    if audio_path is None:
        return None
    try:
        audio, sample_rate = _load_audio(audio_path, audio_cache)
        if audio.size == 0:
            return None
        segment = _slice_audio(
            audio=audio,
            sample_rate=sample_rate,
            start_seconds=record.event.start,
            end_seconds=record.event.end,
            padding_seconds=settings.padding_ms / 1000.0,
        )
        if segment is None:
            return None
        return _build_timbre_fingerprint(
            segment=segment,
            sample_rate=sample_rate,
            target_size=settings.sample_count,
        )
    except Exception:
        return None


def build_timbre_fingerprint_preview(
    *,
    audio_path: str,
    start_seconds: float,
    end_seconds: float,
    settings: TimbreFingerprintSettings,
    audio_cache: dict[str, tuple[np.ndarray, int]],
) -> tuple[float, ...] | None:
    try:
        audio, sample_rate = _load_audio(audio_path, audio_cache)
        if audio.size == 0:
            return None
        segment = _slice_audio(
            audio=audio,
            sample_rate=sample_rate,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            padding_seconds=settings.padding_ms / 1000.0,
        )
        if segment is None:
            return None
        return _build_timbre_fingerprint(
            segment=segment,
            sample_rate=sample_rate,
            target_size=settings.sample_count,
        )
    except Exception:
        return None


def compare_timbre_fingerprint_similarity(
    anchor_fingerprint: tuple[float, ...],
    candidate_fingerprint: tuple[float, ...],
) -> float:
    anchor, candidate = _coerce_comparable_vectors(
        anchor_fingerprint,
        candidate_fingerprint,
    )
    if anchor is None or candidate is None:
        return 0.0
    anchor_norm = float(np.linalg.norm(anchor))
    candidate_norm = float(np.linalg.norm(candidate))
    if anchor_norm <= 1e-8 or candidate_norm <= 1e-8:
        return 0.0
    cosine = float(np.dot(anchor, candidate) / (anchor_norm * candidate_norm))
    return max(0.0, min(1.0, cosine))


def _resolve_audio_path(layer: Layer, take: Take) -> str | None:
    for source_ref in (take.source_content_ref, layer.source_content_ref):
        if source_ref is not None and source_ref.locator:
            return str(Path(source_ref.locator).expanduser())
    return None


def _load_audio(
    audio_path: str,
    audio_cache: dict[str, tuple[np.ndarray, int]],
) -> tuple[np.ndarray, int]:
    cached = audio_cache.get(audio_path)
    if cached is not None:
        return cached
    resolved = str(Path(audio_path).expanduser())
    audio, sample_rate = librosa.load(resolved, sr=_FEATURE_SAMPLE_RATE, mono=True)
    cached = (audio.astype(np.float32, copy=False), int(sample_rate))
    audio_cache[audio_path] = cached
    return cached


def _slice_audio(
    *,
    audio: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
    padding_seconds: float,
) -> np.ndarray | None:
    padded_start = max(0.0, float(start_seconds) - padding_seconds)
    padded_end = max(
        padded_start + _MIN_EVENT_DURATION_SECONDS,
        float(end_seconds) + padding_seconds,
    )
    start_sample = int(round(padded_start * sample_rate))
    end_sample = min(len(audio), int(round(padded_end * sample_rate)))
    if end_sample <= start_sample:
        return None
    segment = audio[start_sample:end_sample].astype(np.float32, copy=False)
    return segment if segment.size > 0 else None


def _build_timbre_fingerprint(
    *,
    segment: np.ndarray,
    sample_rate: int,
    target_size: int,
) -> tuple[float, ...]:
    if segment.size <= 0:
        return tuple(0.0 for _ in range(max(1, target_size)))
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sample_rate,
        n_mels=max(16, int(target_size)),
        n_fft=512,
        hop_length=128,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel + 1e-10)
    profile = np.mean(log_mel, axis=1).astype(np.float32, copy=False)
    profile = np.nan_to_num(profile, copy=False)
    profile = profile - float(np.mean(profile))
    norm = float(np.linalg.norm(profile))
    if norm <= 1e-8:
        return tuple(0.0 for _ in range(max(1, target_size)))
    profile = profile / norm
    resampled = _resample_vector(profile, target_size)
    return tuple(float(value) for value in resampled)


def _resample_vector(values: np.ndarray, target_size: int) -> np.ndarray:
    if target_size <= 0:
        return np.zeros(0, dtype=np.float32)
    if values.size <= 1:
        return np.zeros(target_size, dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_size, dtype=np.float32)
    return np.interp(target_x, source_x, values).astype(np.float32, copy=False)


def _coerce_comparable_vectors(
    anchor_values: tuple[float, ...],
    candidate_values: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if not anchor_values or not candidate_values:
        return None, None
    anchor = np.asarray(anchor_values, dtype=np.float32)
    candidate = np.asarray(candidate_values, dtype=np.float32)
    if anchor.shape != candidate.shape:
        target_size = min(anchor.size, candidate.size)
        if target_size <= 0:
            return None, None
        anchor = _resample_vector(anchor, target_size)
        candidate = _resample_vector(candidate, target_size)
    return anchor, candidate


def _event_id(event: Event | object) -> EventId:
    direct = getattr(event, "id", None)
    if direct is not None:
        return direct
    presented = getattr(event, "event_id", None)
    if presented is not None:
        return presented
    raise AttributeError("Comparison event must expose id or event_id")


def _layer_id(layer: Layer | object) -> LayerId:
    direct = getattr(layer, "id", None)
    if direct is not None:
        return direct
    presented = getattr(layer, "layer_id", None)
    if presented is not None:
        return presented
    raise AttributeError("Comparison layer must expose id or layer_id")


def _take_id(take: Take | object) -> TakeId:
    direct = getattr(take, "id", None)
    if direct is not None:
        return direct
    presented = getattr(take, "take_id", None)
    if presented is not None:
        return presented
    raise AttributeError("Comparison take must expose id or take_id")
