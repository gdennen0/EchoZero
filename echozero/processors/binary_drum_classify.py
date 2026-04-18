"""
BinaryDrumClassifyProcessor: classify drum onsets into per-class layers using shared runtime models.
Exists because Stage Zero needs a real app-path "extract classified drums" pipeline, not a preview heuristic.
Used by the analysis service to turn drum-stem onset detections into kick/snare event layers.
"""

from __future__ import annotations

from dataclasses import replace
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
    build_feature_tensor,
    load_runtime_model,
    predict_probabilities,
    resolve_device,
)

BinaryClassifyFn = Callable[
    [
        list[Event],
        str,
        dict[str, str],
        str,
        float,
    ],
    dict[str, list[tuple[Event, float]]],
]


def _default_binary_classify(
    events: list[Event],
    audio_file: str,
    model_paths: dict[str, str],
    device: str,
    positive_threshold: float,
) -> dict[str, list[tuple[Event, float]]]:
    try:
        import librosa
        import soundfile as sf
    except ImportError as exc:
        raise ExecutionError(
            "Binary drum classification requires librosa and soundfile."
        ) from exc

    if not audio_file:
        raise ValidationError("Binary drum classification requires an audio file path.")

    resolved_device = resolve_device(device)
    model_runtimes = {
        label: _load_runtime_model_for_label(label=label, model_path=path, device=resolved_device)
        for label, path in model_paths.items()
    }
    if not model_runtimes:
        return {}

    output_sample_rate = next(iter(model_runtimes.values())).sample_rate
    audio, file_sample_rate = sf.read(audio_file, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sample_rate != output_sample_rate:
        audio = librosa.resample(audio, orig_sr=file_sample_rate, target_sr=output_sample_rate)
    audio = audio.astype(np.float32)

    classified: dict[str, list[tuple[Event, float]]] = {label: [] for label in model_paths}
    for event in events:
        for label, runtime in model_runtimes.items():
            feature = build_feature_tensor(
                audio=audio,
                event_time=float(event.time),
                sample_rate=runtime.sample_rate,
                max_length=runtime.max_length,
                n_fft=runtime.n_fft,
                hop_length=runtime.hop_length,
                n_mels=runtime.n_mels,
                fmax=runtime.fmax,
            )
            score = _predict_positive_probability(runtime, feature, label)
            if score < positive_threshold:
                continue
            classified_event = replace(
                event,
                classifications={
                    "class": label,
                    "confidence": round(score, 4),
                },
                metadata={
                    **event.metadata,
                    "classified": True,
                    "source_model": Path(model_paths[label]).name,
                },
                origin=f"binary_drum_classify:{label}",
            )
            classified[label].append((classified_event, score))
    return classified


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

        event_data = context.get_input(block_id, "events_in", EventData)
        if event_data is None:
            return err(ExecutionError(f"Block '{block_id}' requires 'events_in'."))
        audio = context.get_input(block_id, "audio_in", AudioData)
        if audio is None or not audio.file_path:
            return err(ExecutionError(f"Block '{block_id}' requires 'audio_in'."))

        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        settings = block.settings
        kick_model_path = settings.get("kick_model_path")
        snare_model_path = settings.get("snare_model_path")
        device = str(settings.get("device", "cpu"))
        positive_threshold = float(settings.get("positive_threshold", 0.5))
        if positive_threshold < 0.0 or positive_threshold > 1.0:
            return err(ValidationError("positive_threshold must be between 0.0 and 1.0"))
        if not isinstance(kick_model_path, str) or not kick_model_path.strip():
            return err(ValidationError("kick_model_path is required"))
        if not isinstance(snare_model_path, str) or not snare_model_path.strip():
            return err(ValidationError("snare_model_path is required"))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="binary_drum_classify",
                percent=0.2,
                message="Loading drum classifier bundles",
            )
        )

        all_events: list[Event] = []
        for layer in event_data.layers:
            all_events.extend(layer.events)

        try:
            classified = self._classify_fn(
                list(all_events),
                str(audio.file_path),
                {
                    "kick": kick_model_path,
                    "snare": snare_model_path,
                },
                device,
                positive_threshold,
            )
        except (ExecutionError, ValidationError, FileNotFoundError) as exc:
            return err(exc)
        except Exception as exc:
            return err(ExecutionError(f"Binary drum classification failed: {type(exc).__name__}: {exc}"))

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
                event for event, _score in sorted(scored_events, key=lambda item: item[0].time)
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
