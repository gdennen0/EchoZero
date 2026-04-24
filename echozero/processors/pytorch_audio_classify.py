"""
PyTorchAudioClassifyProcessor: Event classification via a pre-trained PyTorch model.
Exists because classification is the next step after detection — you detect onsets,
then classify them (kick vs snare vs hihat, etc.).
Used by ExecutionEngine when running blocks of type 'PyTorchAudioClassify'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from echozero.domain.types import AudioData, Event, EventData, Layer
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok
from echozero.runtime_models.loader import (
    build_feature_tensor,
    load_runtime_model,
    predict_probabilities_batch,
    resolve_device,
)

# Classifier function signature for DI
ClassifyFn = Callable[
    [
        list[Event],  # events to classify
        str | None,  # audio file path (optional context)
        str,  # model_path
        str,  # device
        int,  # batch_size
    ],
    list[Event],  # classified events (same events with classifications updated)
]


def _default_classify(
    events: list[Event],
    audio_file: str | None,
    model_path: str,
    device: str,
    batch_size: int,
) -> list[Event]:
    """Load a runtime model and classify events from onset-aligned audio windows."""
    try:
        import librosa
        import numpy as np
        import soundfile as sf
    except ImportError:
        raise ExecutionError(
            "PyTorch classification requires librosa and soundfile."
        )

    if not audio_file:
        raise ValidationError("PyTorchAudioClassify requires an audio file path.")
    if batch_size <= 0:
        raise ValidationError("batch_size must be a positive integer.")

    resolved_device = resolve_device(device)
    runtime_model = load_runtime_model(model_path, device=resolved_device)

    audio, file_sample_rate = sf.read(audio_file, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sample_rate != runtime_model.sample_rate:
        audio = librosa.resample(
            audio,
            orig_sr=file_sample_rate,
            target_sr=runtime_model.sample_rate,
        )
    audio = audio.astype(np.float32)

    classified_events: list[Event] = []
    for start_index in range(0, len(events), batch_size):
        batch_events = events[start_index : start_index + batch_size]
        batch_features = np.concatenate(
            [
                build_feature_tensor(
                    audio=audio,
                    event_time=float(event.time),
                    sample_rate=runtime_model.sample_rate,
                    max_length=runtime_model.max_length,
                    n_fft=runtime_model.n_fft,
                    hop_length=runtime_model.hop_length,
                    n_mels=runtime_model.n_mels,
                    fmax=runtime_model.fmax,
                )
                for event in batch_events
            ],
            axis=0,
        )
        probabilities = predict_probabilities_batch(runtime_model, batch_features)

        for event, probability_row in zip(batch_events, probabilities, strict=False):
            predicted_index = int(np.argmax(probability_row))
            predicted_class = runtime_model.classes[predicted_index]
            confidence = float(probability_row[predicted_index])
            classified_events.append(
                Event(
                    id=event.id,
                    time=event.time,
                    duration=event.duration,
                    classifications={
                        "class": predicted_class,
                        "confidence": round(confidence, 4),
                    },
                    metadata={
                        **event.metadata,
                        "classified": True,
                        "source_model": Path(runtime_model.source_path).name,
                    },
                    origin=event.origin,
                )
            )
    return classified_events


class PyTorchAudioClassifyProcessor:
    """Classifies events using a pre-trained PyTorch model."""

    def __init__(self, classify_fn: ClassifyFn | None = None) -> None:
        self._classify_fn = classify_fn or _default_classify

    def execute(self, block_id: str, context: ExecutionContext) -> Result[EventData]:
        """Read upstream events, classify with PyTorch model, return classified EventData."""
        # Report start
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="pytorch_audio_classify",
                percent=0.0,
                message="Starting event classification",
            )
        )

        # Read event input
        event_data = context.get_input(block_id, "events_in", EventData)
        if event_data is None:
            return err(
                ExecutionError(
                    f"Block '{block_id}' has no event input — "
                    f"connect an event source to 'events_in'"
                )
            )

        # Optionally read audio for context
        audio = context.get_input(block_id, "audio_in", AudioData)
        audio_file = audio.file_path if audio else None

        # Read settings
        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        settings = block.settings
        model_path = settings.get("model_path")
        device = settings.get("device", "cpu")
        batch_size = settings.get("batch_size", 32)

        # Validate required settings
        if model_path is None:
            return err(
                ValidationError(f"Block '{block_id}' is missing required setting 'model_path'")
            )

        if not isinstance(batch_size, int) or batch_size <= 0:
            return err(ValidationError(f"batch_size must be a positive integer, got {batch_size}"))

        # Report progress
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="pytorch_audio_classify",
                percent=0.1,
                message=f"Loading model from {model_path}",
            )
        )

        # Flatten all events from all layers
        all_events = []
        for layer in event_data.layers:
            all_events.extend(layer.events)

        if not all_events:
            # No events to classify, return empty
            context.progress_bus.publish(
                ProgressReport(
                    block_id=block_id,
                    phase="pytorch_audio_classify",
                    percent=1.0,
                    message="No events to classify",
                )
            )
            return ok(event_data)

        # Report progress
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="pytorch_audio_classify",
                percent=0.3,
                message=f"Classifying {len(all_events)} events",
            )
        )

        # Run classification
        try:
            classified_events = self._classify_fn(
                events=list(all_events),
                audio_file=audio_file,
                model_path=model_path,
                device=device,
                batch_size=batch_size,
            )
        except (ValidationError, ExecutionError) as exc:
            return err(exc)
        except Exception as exc:
            return err(
                ExecutionError(
                    f"Classification failed for block '{block_id}': "
                    f"{type(exc).__name__}: {exc}"
                )
            )

        # Report near-complete
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="pytorch_audio_classify",
                percent=0.9,
                message="Rebuilding layers with classified events",
            )
        )

        # Rebuild layers with classified events (preserve layer structure)
        # Map events back to their layers by matching IDs or maintaining order
        output_layers: list[Layer] = []
        event_idx = 0
        for input_layer in event_data.layers:
            layer_events: list[Event] = []
            for _ in input_layer.events:
                if event_idx < len(classified_events):
                    layer_events.append(classified_events[event_idx])
                    event_idx += 1
            output_layers.append(
                Layer(
                    id=input_layer.id,
                    name=input_layer.name,
                    events=tuple(layer_events),
                )
            )

        # Report complete
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="pytorch_audio_classify",
                percent=1.0,
                message="Classification complete",
            )
        )

        return ok(EventData(layers=tuple(output_layers)))
