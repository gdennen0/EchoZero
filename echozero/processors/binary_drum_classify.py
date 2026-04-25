"""
BinaryDrumClassifyProcessor: classify drum onsets into per-class layers using shared runtime models.
Exists because Stage Zero needs a real app-path "extract classified drums" pipeline, not a preview heuristic.
Used by the analysis service to turn drum-stem onset detections into kick/snare event layers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from echozero.domain.types import AudioData, Event, EventData, Layer
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok
from echozero.runtime_models.loader import (
    LoadedRuntimeModel,
    build_model_artifact_reference,
    build_feature_tensor,
    load_runtime_model,
    predict_probabilities,
    resolve_device,
)

_DEFAULT_MIN_EVENT_PEAK = 1e-3
_DEFAULT_MIN_EVENT_RMS = 2e-4


@dataclass(frozen=True, slots=True)
class DrumLabelInferenceInput:
    """Resolved input bundle for one drum label's audio, events, and threshold."""

    label: str
    audio_file: str
    events: tuple[Event, ...]
    model_path: str
    positive_threshold: float


@dataclass(frozen=True, slots=True)
class BinaryAssignmentConfig:
    """Controls how overlapping kick/snare candidates become final outputs."""

    assignment_mode: str = "independent"
    winner_margin: float = 0.0
    event_match_window_seconds: float = 0.04
    min_event_peak: float = _DEFAULT_MIN_EVENT_PEAK
    min_event_rms: float = _DEFAULT_MIN_EVENT_RMS


@dataclass(frozen=True, slots=True)
class ClassifiedCandidate:
    """Scored candidate event before cross-label assignment is resolved."""

    label: str
    event: Event
    score: float


BinaryClassifyFn = Callable[
    [
        tuple[DrumLabelInferenceInput, ...],
        str,
        BinaryAssignmentConfig,
    ],
    dict[str, list[tuple[Event, float]]],
]

_VALID_ASSIGNMENT_MODES = frozenset({"independent", "exclusive_max"})


def _default_binary_classify(
    inputs: tuple[DrumLabelInferenceInput, ...],
    device: str,
    assignment: BinaryAssignmentConfig,
) -> dict[str, list[tuple[Event, float]]]:
    try:
        import librosa
        import soundfile as sf
    except ImportError as exc:
        raise ExecutionError(
            "Binary drum classification requires librosa and soundfile."
        ) from exc

    if not inputs:
        return {}

    resolved_device = resolve_device(device)
    model_runtimes = {
        label_input.label: _load_runtime_model_for_label(
            label=label_input.label,
            model_path=label_input.model_path,
            device=resolved_device,
        )
        for label_input in inputs
    }
    model_artifacts = {
        label: build_model_artifact_reference(runtime_model)
        for label, runtime_model in model_runtimes.items()
    }
    audio_cache: dict[tuple[str, int], np.ndarray] = {}
    classified: dict[str, list[tuple[Event, float]]] = {
        label_input.label: [] for label_input in inputs
    }

    for label_input in inputs:
        runtime_model = model_runtimes[label_input.label]
        audio = _load_runtime_audio(
            audio_cache=audio_cache,
            audio_file=label_input.audio_file,
            output_sample_rate=runtime_model.sample_rate,
            librosa_module=librosa,
            soundfile_module=sf,
        )
        for event in label_input.events:
            event_peak, event_rms = _event_window_energy(
                audio=audio,
                event_time=float(event.time),
                sample_rate=runtime_model.sample_rate,
                max_length=runtime_model.max_length,
            )
            if (
                event_peak < assignment.min_event_peak
                and event_rms < assignment.min_event_rms
            ):
                continue
            feature = build_feature_tensor(
                audio=audio,
                event_time=float(event.time),
                sample_rate=runtime_model.sample_rate,
                max_length=runtime_model.max_length,
                n_fft=runtime_model.n_fft,
                hop_length=runtime_model.hop_length,
                n_mels=runtime_model.n_mels,
                fmax=runtime_model.fmax,
            )
            score = _predict_positive_probability(runtime_model, feature, label_input.label)
            if score < label_input.positive_threshold:
                continue
            classified_event = replace(
                event,
                classifications={
                    "class": label_input.label,
                    "confidence": round(score, 4),
                    "model_artifact": model_artifacts[label_input.label],
                },
                metadata={
                    **event.metadata,
                    "classified": True,
                    "positive_threshold": round(label_input.positive_threshold, 4),
                    "source_audio": Path(label_input.audio_file).name,
                    "source_model": Path(label_input.model_path).name,
                    "model_artifact": model_artifacts[label_input.label],
                    "assignment_mode": assignment.assignment_mode,
                    "event_peak": round(event_peak, 6),
                    "event_rms": round(event_rms, 6),
                    "min_event_peak": round(assignment.min_event_peak, 6),
                    "min_event_rms": round(assignment.min_event_rms, 6),
                },
                origin=f"binary_drum_classify:{label_input.label}",
            )
            classified[label_input.label].append((classified_event, score))
    return _apply_assignment_config(classified, assignment)


class BinaryDrumClassifyProcessor:
    """Classify drum stem onsets into one layer per target drum class."""

    def __init__(self, classify_fn: BinaryClassifyFn | None = None) -> None:
        self._classify_fn = classify_fn or _default_binary_classify

    def execute(self, block_id: str, context: ExecutionContext) -> Result[EventData]:
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="binary_drum_classify",
                percent=0.0,
                message="Starting classified drum extraction",
            )
        )

        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        settings = block.settings
        kick_model_path = settings.get("kick_model_path")
        snare_model_path = settings.get("snare_model_path")
        if not isinstance(kick_model_path, str) or not kick_model_path.strip():
            return err(ValidationError("kick_model_path is required"))
        if not isinstance(snare_model_path, str) or not snare_model_path.strip():
            return err(ValidationError("snare_model_path is required"))

        device = str(settings.get("device", "cpu"))
        positive_threshold = _validated_probability(
            settings.get("positive_threshold", 0.5),
            setting_name="positive_threshold",
        )
        assignment_mode = str(settings.get("assignment_mode", "independent")).strip().lower()
        if assignment_mode not in _VALID_ASSIGNMENT_MODES:
            return err(
                ValidationError(
                    "assignment_mode must be one of: independent, exclusive_max"
                )
            )
        winner_margin = float(settings.get("winner_margin", 0.0))
        if winner_margin < 0.0 or winner_margin > 1.0:
            return err(ValidationError("winner_margin must be between 0.0 and 1.0"))
        event_match_window_ms = float(settings.get("event_match_window_ms", 40.0))
        if event_match_window_ms < 0.0:
            return err(ValidationError("event_match_window_ms must be >= 0.0"))
        min_event_peak = _validated_non_negative(
            settings.get("min_event_peak", _DEFAULT_MIN_EVENT_PEAK),
            setting_name="min_event_peak",
        )
        min_event_rms = _validated_non_negative(
            settings.get("min_event_rms", _DEFAULT_MIN_EVENT_RMS),
            setting_name="min_event_rms",
        )

        shared_event_data = context.get_input(block_id, "events_in", EventData)
        shared_audio = context.get_input(block_id, "audio_in", AudioData)
        assignment = BinaryAssignmentConfig(
            assignment_mode=assignment_mode,
            winner_margin=winner_margin,
            event_match_window_seconds=event_match_window_ms / 1000.0,
            min_event_peak=min_event_peak,
            min_event_rms=min_event_rms,
        )

        try:
            label_inputs = (
                _resolve_label_inference_input(
                    context,
                    block_id=block_id,
                    label="kick",
                    model_path=kick_model_path,
                    shared_audio=shared_audio,
                    shared_event_data=shared_event_data,
                    fallback_positive_threshold=positive_threshold,
                ),
                _resolve_label_inference_input(
                    context,
                    block_id=block_id,
                    label="snare",
                    model_path=snare_model_path,
                    shared_audio=shared_audio,
                    shared_event_data=shared_event_data,
                    fallback_positive_threshold=positive_threshold,
                ),
            )
        except (ExecutionError, ValidationError) as exc:
            return err(exc)

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="binary_drum_classify",
                percent=0.2,
                message="Loading drum classifier bundles",
            )
        )

        try:
            classified = self._classify_fn(label_inputs, device, assignment)
        except (ExecutionError, ValidationError, FileNotFoundError) as exc:
            return err(exc)
        except Exception as exc:
            return err(
                ExecutionError(
                    f"Binary drum classification failed: {type(exc).__name__}: {exc}"
                )
            )

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="binary_drum_classify",
                percent=0.9,
                message="Building kick/snare layers",
            )
        )

        output_layers: list[Layer] = []
        for label in ("kick", "snare"):
            scored_events = classified.get(label, [])
            ordered_events = tuple(
                event
                for event, _score in sorted(scored_events, key=lambda item: item[0].time)
            )
            output_layers.append(Layer(id=label, name=label, events=ordered_events))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="binary_drum_classify",
                percent=1.0,
                message="Classified drum extraction complete",
            )
        )
        return ok(EventData(layers=tuple(output_layers)))


def _validated_probability(raw_value: object, *, setting_name: str) -> float:
    value = float(raw_value)
    if value < 0.0 or value > 1.0:
        raise ValidationError(f"{setting_name} must be between 0.0 and 1.0")
    return value


def _validated_non_negative(raw_value: object, *, setting_name: str) -> float:
    value = float(raw_value)
    if value < 0.0:
        raise ValidationError(f"{setting_name} must be >= 0.0")
    return value


def _flatten_events(event_data: EventData | None) -> tuple[Event, ...]:
    if event_data is None:
        return ()
    return tuple(event for layer in event_data.layers for event in layer.events)


def _resolve_label_inference_input(
    context: ExecutionContext,
    *,
    block_id: str,
    label: str,
    model_path: str,
    shared_audio: AudioData | None,
    shared_event_data: EventData | None,
    fallback_positive_threshold: float,
) -> DrumLabelInferenceInput:
    label_audio = context.get_input(block_id, f"{label}_audio_in", AudioData) or shared_audio
    if label_audio is None or not label_audio.file_path:
        raise ExecutionError(
            f"Block '{block_id}' requires '{label}_audio_in' or 'audio_in'."
        )

    label_event_data = context.get_input(block_id, f"{label}_events_in", EventData) or shared_event_data
    if label_event_data is None:
        raise ExecutionError(
            f"Block '{block_id}' requires '{label}_events_in' or 'events_in'."
        )

    block = context.graph.blocks[block_id]
    positive_threshold = _validated_probability(
        block.settings.get(f"{label}_positive_threshold", fallback_positive_threshold),
        setting_name=f"{label}_positive_threshold",
    )
    return DrumLabelInferenceInput(
        label=label,
        audio_file=str(label_audio.file_path),
        events=_flatten_events(label_event_data),
        model_path=model_path,
        positive_threshold=positive_threshold,
    )


def _load_runtime_audio(
    *,
    audio_cache: dict[tuple[str, int], np.ndarray],
    audio_file: str,
    output_sample_rate: int,
    librosa_module: object,
    soundfile_module: object,
) -> np.ndarray:
    cache_key = (audio_file, output_sample_rate)
    cached = audio_cache.get(cache_key)
    if cached is not None:
        return cached

    audio, file_sample_rate = soundfile_module.read(
        audio_file,
        dtype="float32",
        always_2d=False,
    )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sample_rate != output_sample_rate:
        audio = librosa_module.resample(
            audio,
            orig_sr=file_sample_rate,
            target_sr=output_sample_rate,
        )
    resolved = audio.astype(np.float32)
    audio_cache[cache_key] = resolved
    return resolved


def _event_window_energy(
    *,
    audio: np.ndarray,
    event_time: float,
    sample_rate: int,
    max_length: int,
) -> tuple[float, float]:
    start_index = max(0, int(round(event_time * sample_rate)))
    segment = audio[start_index : start_index + max_length]
    if len(segment) < max_length:
        segment = np.pad(segment, (0, max_length - len(segment)))
    if segment.size == 0:
        return 0.0, 0.0
    peak = float(np.max(np.abs(segment)))
    rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
    return peak, rms


def _apply_assignment_config(
    classified: dict[str, list[tuple[Event, float]]],
    assignment: BinaryAssignmentConfig,
) -> dict[str, list[tuple[Event, float]]]:
    if assignment.assignment_mode == "independent":
        return classified

    candidates = [
        ClassifiedCandidate(label=label, event=event, score=score)
        for label, scored_events in classified.items()
        for event, score in scored_events
    ]
    resolved: dict[str, list[tuple[Event, float]]] = {label: [] for label in classified}
    for group in _group_candidates_by_time(
        candidates,
        window_seconds=assignment.event_match_window_seconds,
    ):
        winner, runner_up = _resolve_group_winner(group)
        if (
            runner_up is not None
            and (winner.score - runner_up.score) < assignment.winner_margin
        ):
            continue
        resolved[winner.label].append((winner.event, winner.score))

    return {
        label: sorted(scored_events, key=lambda item: item[0].time)
        for label, scored_events in resolved.items()
    }


def _group_candidates_by_time(
    candidates: list[ClassifiedCandidate],
    *,
    window_seconds: float,
) -> tuple[tuple[ClassifiedCandidate, ...], ...]:
    if not candidates:
        return ()

    ordered = sorted(candidates, key=lambda item: (item.event.time, -item.score))
    groups: list[list[ClassifiedCandidate]] = [[ordered[0]]]
    group_end_time = ordered[0].event.time
    for candidate in ordered[1:]:
        if candidate.event.time - group_end_time <= window_seconds:
            groups[-1].append(candidate)
            group_end_time = max(group_end_time, candidate.event.time)
            continue
        groups.append([candidate])
        group_end_time = candidate.event.time
    return tuple(tuple(group) for group in groups)


def _resolve_group_winner(
    group: tuple[ClassifiedCandidate, ...],
) -> tuple[ClassifiedCandidate, ClassifiedCandidate | None]:
    ordered = sorted(
        group,
        key=lambda item: (item.score, -item.event.time),
        reverse=True,
    )
    winner = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    return winner, runner_up


def _load_runtime_model_for_label(*, label: str, model_path: str, device: str) -> LoadedRuntimeModel:
    runtime_model = load_runtime_model(model_path, device=device)
    if label not in runtime_model.classes:
        raise ValidationError(
            f"Runtime model {runtime_model.source_path} does not include class '{label}'."
        )
    return runtime_model


def _predict_positive_probability(
    runtime_model: LoadedRuntimeModel,
    feature: np.ndarray,
    label: str,
) -> float:
    probabilities = predict_probabilities(runtime_model, feature)
    return float(probabilities[runtime_model.classes.index(label)])
