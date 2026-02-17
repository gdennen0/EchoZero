"""
Export Audio Dataset Block Processor

Extracts audio clips from source audio at event time regions and exports
them as individual audio files to a selected directory.

Use case: Create audio datasets from EchoZero event data. For each event,
the block extracts the audio between event.time and event.time + event.duration
from the source audio, then writes it as a standalone audio file.
"""
from typing import Dict, Optional, Any, List, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import os

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem, AudioDataItem, EventDataItem, Event
from src.application.blocks import register_processor_class
from src.features.execution.application.progress_helpers import (
    IncrementalProgress, get_progress_tracker
)
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade

try:
    import numpy as np
    import soundfile as sf
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    Log.warning("Audio libraries not available - ExportAudioDataset block will not work")


# Supported export formats
SUPPORTED_FORMATS = {"wav", "mp3", "flac", "ogg"}

# Naming schemes for exported clips
NAMING_SCHEMES = {
    "index": {
        "name": "Index",
        "description": "Sequential numbers: clip_001.wav, clip_002.wav, ...",
    },
    "timestamp": {
        "name": "Timestamp",
        "description": "Event time: clip_1.234s.wav, clip_5.678s.wav, ...",
    },
    "class_index": {
        "name": "Class + Index",
        "description": "Classification prefix: kick_001.wav, snare_002.wav, ...",
    },
    "prefix": {
        "name": "Custom Prefix",
        "description": "User-defined prefix + number: myprefix_001.wav, myprefix_002.wav, ...",
    },
}


def _normalise_audio_shape(audio: np.ndarray) -> np.ndarray:
    """
    Normalise audio to 2D array with shape (channels, samples).

    Handles 1D mono, 2D channels-first, and 2D samples-first layouts.
    """
    if audio.ndim == 1:
        return audio.reshape(1, -1)
    if audio.ndim == 2:
        if audio.shape[0] > audio.shape[1] and audio.shape[1] <= 16:
            return audio.T
        return audio
    raise ValueError(f"Unsupported audio shape: {audio.shape}")


class ExportAudioDatasetBlockProcessor(BlockProcessor):
    """
    Processor for ExportAudioDataset block type.

    Takes EventData and SourceAudio as inputs. For every event, extracts the
    audio between event.time and event.time + event.duration from the source
    audio, then exports each clip as an individual audio file.
    """

    def can_process(self, block: Block) -> bool:
        return block.type == "ExportAudioDataset"

    def get_block_type(self) -> str:
        return "ExportAudioDataset"

    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Status levels for ExportAudioDataset block.

        - Error (0): Output directory not configured
        - Warning (1): Audio or events input not connected
        - Ready (2): All requirements met
        """
        from src.features.blocks.domain import BlockStatusLevel

        def check_output_dir(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if output directory is configured and writable."""
            output_dir = blk.metadata.get("output_dir")
            if not output_dir:
                return False
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception:
                    return False
            return os.access(output_dir, os.W_OK)

        def check_inputs_connected(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if both audio and events inputs are connected."""
            if not hasattr(f, "connection_service"):
                return False
            connections = f.connection_service.list_connections_by_block(blk.id)
            incoming = [c for c in connections if c.target_block_id == blk.id]
            connected_ports = {c.target_input_name for c in incoming}
            return "audio" in connected_ports and "events" in connected_ports

        return [
            BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[check_output_dir],
            ),
            BlockStatusLevel(
                priority=1,
                name="warning",
                display_name="Warning",
                color="#ffd43b",
                conditions=[check_inputs_connected],
            ),
            BlockStatusLevel(
                priority=2,
                name="ready",
                display_name="Ready",
                color="#51cf66",
                conditions=[],
            ),
        ]

    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DataItem]:
        """
        Extract audio clips from source audio at event time regions and
        export them as individual files.
        """
        if not HAS_AUDIO_LIBS:
            raise ProcessingError(
                "Audio libraries (numpy, soundfile) not installed.",
                block_id=block.id,
                block_name=block.name,
            )

        # -- Read settings from block metadata --------------------------------
        output_dir = block.metadata.get("output_dir")
        fmt = (block.metadata.get("audio_format") or "wav").lower()
        naming = block.metadata.get("naming_scheme", "index")
        zero_pad = int(block.metadata.get("zero_pad_digits", 4))
        group_by_class = bool(block.metadata.get("group_by_class", False))
        unclassified_folder = block.metadata.get("unclassified_folder", "unclassified")
        filename_prefix = block.metadata.get("filename_prefix", "clip")

        if not output_dir:
            raise ProcessingError(
                "ExportAudioDataset block requires an output directory. "
                "Set one in the block panel before executing.",
                block_id=block.id,
                block_name=block.name,
            )

        if fmt not in SUPPORTED_FORMATS:
            raise ProcessingError(
                f"Unsupported export format '{fmt}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
                block_id=block.id,
                block_name=block.name,
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # -- Validate audio input ---------------------------------------------
        audio_input = inputs.get("audio")
        if audio_input is None:
            raise ProcessingError(
                "ExportAudioDataset requires audio input. "
                "Connect a source audio block to the 'audio' input.",
                block_id=block.id,
                block_name=block.name,
            )

        audio_items = self._normalize_to_list(audio_input)
        if not audio_items:
            raise ProcessingError(
                "ExportAudioDataset: No audio items received on 'audio' input.",
                block_id=block.id,
                block_name=block.name,
            )

        # -- Validate events input --------------------------------------------
        events_input = inputs.get("events")
        if events_input is None:
            raise ProcessingError(
                "ExportAudioDataset requires event data. "
                "Connect an event source to the 'events' input.",
                block_id=block.id,
                block_name=block.name,
            )

        event_items = self._normalize_to_list(events_input)
        all_events: List[Event] = []
        for item in event_items:
            if isinstance(item, EventDataItem):
                all_events.extend(item.get_events())

        # Keep only events with positive duration
        valid_events = [e for e in all_events if e.duration > 0]

        if not valid_events:
            raise ProcessingError(
                "ExportAudioDataset: No events with positive duration found. "
                "Events need duration > 0 to define audio regions.",
                block_id=block.id,
                block_name=block.name,
            )

        Log.info(
            f"ExportAudioDataset: Exporting {len(valid_events)} event clips "
            f"to {output_path} as .{fmt}"
            f"{' (grouped by class)' if group_by_class else ''}"
        )

        # -- Load source audio ------------------------------------------------
        source_audio, sample_rate = self._load_source_audio(audio_items)
        if source_audio is None:
            raise ProcessingError(
                "ExportAudioDataset: Could not load audio data from source.",
                block_id=block.id,
                block_name=block.name,
            )

        audio_2d = _normalise_audio_shape(source_audio).astype(np.float64)
        n_channels, total_samples = audio_2d.shape

        Log.info(
            f"ExportAudioDataset: Source audio shape={audio_2d.shape}, "
            f"sr={sample_rate}, duration={total_samples / sample_rate:.2f}s"
        )

        # -- Progress ---------------------------------------------------------
        progress_tracker = get_progress_tracker(metadata)
        progress = IncrementalProgress(
            progress_tracker,
            "Exporting audio dataset",
            total=len(valid_events),
        )

        # -- Export each event clip -------------------------------------------
        exported = 0
        skipped = 0
        classification_counts: Dict[str, int] = {}
        # Per-classification index counters for zero-padded naming
        class_index_counters: Dict[str, int] = {}

        for idx, event in enumerate(valid_events):
            start_sample = max(0, int(event.time * sample_rate))
            end_sample = min(total_samples, int((event.time + event.duration) * sample_rate))

            if start_sample >= end_sample or start_sample >= total_samples:
                skipped += 1
                progress.step(f"Skipped event {idx + 1}")
                continue

            # Extract clip (channels, samples)
            clip = audio_2d[:, start_sample:end_sample]

            # Determine destination directory
            if group_by_class:
                classification = (
                    event.classification.strip() if event.classification else ""
                )
                folder_name = (
                    self._sanitize_name(classification)
                    if classification
                    else self._sanitize_name(unclassified_folder)
                )
                clip_dir = output_path / folder_name
                clip_dir.mkdir(parents=True, exist_ok=True)

                # Track per-class index for naming
                if folder_name not in class_index_counters:
                    class_index_counters[folder_name] = 0
                clip_index = class_index_counters[folder_name]
                class_index_counters[folder_name] += 1

                # Track classification counts for summary
                display_class = classification or unclassified_folder
                classification_counts[display_class] = (
                    classification_counts.get(display_class, 0) + 1
                )
            else:
                clip_dir = output_path
                clip_index = exported

            # Build filename based on naming scheme
            filename = self._build_filename(
                event=event,
                index=clip_index,
                naming=naming,
                fmt=fmt,
                zero_pad=zero_pad,
                prefix=filename_prefix,
            )
            dest = clip_dir / filename

            # Write clip to disk
            try:
                # soundfile expects (samples, channels) for multi-channel
                if clip.shape[0] == 1:
                    sf.write(str(dest), clip[0], sample_rate)
                else:
                    sf.write(str(dest), clip.T, sample_rate)
                exported += 1
            except Exception as e:
                Log.warning(
                    f"ExportAudioDataset: Failed to write clip {filename}: {e}"
                )
                skipped += 1

            progress.step(f"Exported {exported}/{len(valid_events)}")

        # -- Summary ----------------------------------------------------------
        if group_by_class and classification_counts:
            for cls, count in sorted(classification_counts.items()):
                Log.info(f"  - {cls}: {count} clips")
            folder_count = len(classification_counts)
            progress.complete(
                f"Exported {exported} clips to {folder_count} class folders"
            )
        else:
            progress.complete(f"Exported {exported} clips to {output_path.name}")

        Log.info(
            f"ExportAudioDataset: Done. Exported {exported} clips, "
            f"skipped {skipped}."
        )

        if exported == 0:
            raise ProcessingError(
                "ExportAudioDataset: No clips were exported. "
                "Check that events fall within the source audio duration.",
                block_id=block.id,
                block_name=block.name,
            )

        return {}

    # =========================================================================
    # Helpers
    # =========================================================================

    def _normalize_to_list(self, data) -> List:
        """Normalize input data to list format."""
        if data is None:
            return []
        if isinstance(data, list):
            return data
        return [data]

    def _load_source_audio(self, audio_items: List[AudioDataItem]):
        """
        Load audio data from the first available audio item.

        Returns (numpy_array, sample_rate) or (None, None).
        """
        for item in audio_items:
            if not isinstance(item, AudioDataItem):
                continue

            # Try in-memory data first
            audio_data = item.get_audio_data()
            sr = item.sample_rate or 44100

            if audio_data is not None:
                return audio_data, sr

            # Fallback: load from file
            if item.file_path and os.path.exists(item.file_path):
                try:
                    import librosa
                    audio_data, sr = librosa.load(item.file_path, sr=None)
                    return audio_data, sr
                except Exception as e:
                    Log.warning(
                        f"ExportAudioDataset: Failed to load {item.file_path}: {e}"
                    )

        return None, None

    def _build_filename(
        self,
        event: Event,
        index: int,
        naming: str,
        fmt: str,
        zero_pad: int,
        prefix: str = "clip",
    ) -> str:
        """Build a filename for an exported clip based on the naming scheme."""
        padded_idx = str(index).zfill(zero_pad)
        safe_prefix = self._sanitize_name(prefix) if prefix else "clip"

        if naming == "timestamp":
            return f"{safe_prefix}_{event.time:.3f}s.{fmt}"
        elif naming == "class_index":
            classification = (
                event.classification.strip() if event.classification else "unknown"
            )
            safe_class = self._sanitize_name(classification)
            return f"{safe_class}_{padded_idx}.{fmt}"
        elif naming == "prefix":
            return f"{safe_prefix}_{padded_idx}.{fmt}"
        else:
            # Default: index
            return f"clip_{padded_idx}.{fmt}"

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a string for use in filenames."""
        invalid_chars = '<>:"/\\|?* '
        result = name
        for char in invalid_chars:
            result = result.replace(char, "_")
        return result.strip("_")[:80] or "unknown"

    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None,
    ) -> List[str]:
        """Validate block configuration before execution."""
        errors = []

        output_dir = block.metadata.get("output_dir")
        if not output_dir:
            errors.append("Output directory not set.")

        fmt = block.metadata.get("audio_format", "wav")
        if fmt not in SUPPORTED_FORMATS:
            errors.append(f"Unsupported format: '{fmt}'")

        naming = block.metadata.get("naming_scheme", "index")
        if naming not in NAMING_SCHEMES:
            errors.append(f"Unknown naming scheme: '{naming}'")

        return errors


register_processor_class(ExportAudioDatasetBlockProcessor)
