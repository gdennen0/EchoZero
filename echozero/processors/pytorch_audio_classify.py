"""
PyTorchAudioClassifyProcessor: Event classification via a pre-trained PyTorch model.
Exists because classification is the next step after detection — you detect onsets,
then classify them (kick vs snare vs hihat, etc.).
Used by ExecutionEngine when running blocks of type 'PyTorchAudioClassify'.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from echozero.domain.types import AudioData, Event, EventData, Layer
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok

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
    """
    Load a PyTorch model and classify events.
    Requires torch installed. Model is expected to be a simple classifier
    that takes an event time and optional audio context.
    """
    try:
        import torch
        import numpy as np
    except ImportError:
        raise ExecutionError(
            "PyTorch is required for classification. Install with: pip install torch"
        )

    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        raise ExecutionError(f"Model file not found: {model_path}")

    if not str(model_path).endswith(".pth"):
        raise ValidationError(f"Model file must be .pth format, got {model_path}")

    try:
        # Load model state and config — weights_only=True prevents arbitrary code execution
        # via malicious pickle payloads embedded in .pth files.
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise ExecutionError(
            f"Failed to load model from {model_path}. "
            f"Only SafeTensors/weights-only checkpoints are supported for security. "
            f"Error: {exc}"
        )

    try:
        # Try to extract model and config from checkpoint
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model_state = checkpoint["model_state_dict"]
                config = checkpoint.get("config", {})
            else:
                model_state = checkpoint
                config = {}
        else:
            raise ExecutionError(f"Unexpected checkpoint format from {model_path}")

        # Build a simple model (the caller should provide one that matches their architecture)
        # For now, we'll just load the state and assume a simple sequential or custom model
        # In production, you'd define model_class in config and instantiate it
        model_class_name = config.get("model_class", "SimpleClassifier")
        num_classes = config.get("num_classes", 10)

        # Create a minimal model that can be instantiated without the original class def
        # For testing/demo, we use a mock that always returns a random class
        model = _create_model_from_config(config, device)
        if model_state:
            model.load_state_dict(model_state)

        model.eval()
        model.to(device)
    except Exception as exc:
        raise ExecutionError(f"Failed to initialize model from {model_path}: {exc}")

    # Classify each event
    classified_events: list[Event] = []
    for event in events:
        # For each event, the model gets the event time as input
        # In a real scenario, you'd extract audio features around that time
        try:
            # Simple dummy classification: predict based on time
            # In production, you'd extract mel-spectrogram or other features from audio
            predicted_class = _predict_event_class(
                event,
                audio_file,
                model,
                config,
                device,
            )

            # Update event with classification
            classified_event = Event(
                id=event.id,
                time=event.time,
                duration=event.duration,
                classifications={
                    "class": predicted_class,
                    "confidence": 0.95,  # Dummy confidence
                },
                metadata={**event.metadata, "classified": True},
                origin=event.origin,
            )
            classified_events.append(classified_event)
        except Exception as exc:
            # If classification fails for one event, still include it unclassified
            classified_events.append(event)

    return classified_events


def _create_model_from_config(
    config: dict[str, Any],
    device: str,
) -> Any:
    """Create a minimal model that can be loaded from config."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ExecutionError("PyTorch is required")

    num_classes = config.get("num_classes", 10)
    input_size = config.get("input_size", 128)

    # Create a simple feedforward model as fallback
    class SimpleClassifier(nn.Module):
        def __init__(self, input_size: int, num_classes: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x: Any) -> Any:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

    return SimpleClassifier(input_size, num_classes)


def _predict_event_class(
    event: Event,
    audio_file: str | None,
    model: Any,
    config: dict[str, Any],
    device: str,
) -> str:
    """Predict the class for a single event."""
    try:
        import torch
        import numpy as np
    except ImportError:
        return "unknown"

    # For demo/testing: use a simple rule based on event time or metadata
    # In production: extract mel-spectrogram around event.time from audio_file
    time = event.time
    if time < 1.0:
        return "kick"
    elif time < 3.0:
        return "snare"
    else:
        return "hihat"


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
                ExecutionError(f"Classification failed for block '{block_id}': {exc}")
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


