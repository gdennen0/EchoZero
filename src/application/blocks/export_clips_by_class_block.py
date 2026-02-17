"""
ExportClipsByClass Block Processor

Exports EventData audio clips into folders organized by classification.
Each event's audio clip is saved to a subfolder named after its classification.
"""
from typing import Dict, Optional, Any, List, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import os

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.domain.entities import AudioDataItem
from src.shared.domain.entities import EventDataItem, Event
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
    import librosa
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    Log.warning("Audio libraries not available - ExportClipsByClass block will not work")


class ExportClipsByClassBlockProcessor(BlockProcessor):
    """
    Processor for ExportClipsByClass block type.
    
    Exports event audio clips organized by classification into subfolders.
    Requires both audio and events inputs to extract clips from source audio.
    
    Input ports:
        - audio: Source AudioDataItem(s) to extract clips from
        - events: EventDataItem(s) with classification and timing info
    
    Settings (via block.metadata.export_clips_settings):
        - output_dir: Base directory for export
        - format: Audio format (wav, mp3, flac, etc.)
        - include_unclassified: Whether to export events without classification
        - unclassified_folder: Folder name for unclassified events
    """

    _SUPPORTED_FORMATS = {"wav", "mp3", "flac", "ogg"}

    def can_process(self, block: Block) -> bool:
        return block.type == "ExportClipsByClass"

    def get_block_type(self) -> str:
        return "ExportClipsByClass"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for ExportClipsByClass block.
        
        Status levels:
        - Error (0): Output directory not configured or invalid
        - Warning (1): Audio or events inputs not connected
        - Ready (2): All requirements met
        
        Args:
            block: Block entity to get status levels for
            facade: ApplicationFacade for accessing services
            
        Returns:
            List of BlockStatusLevel instances in priority order
        """
        from src.features.blocks.domain import BlockStatusLevel
        import os
        
        def check_output_dir(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if output directory is configured and writable."""
            settings = blk.metadata.get("export_clips_settings", {})
            output_dir = settings.get("output_dir")
            if not output_dir:
                return False
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception:
                    return False
            return os.access(output_dir, os.W_OK)
        
        def check_inputs_connected(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if audio and events inputs are connected."""
            if not hasattr(f, 'connection_service'):
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
                conditions=[check_output_dir]
            ),
            BlockStatusLevel(
                priority=1,
                name="warning",
                display_name="Warning",
                color="#ffd43b",
                conditions=[check_inputs_connected]
            ),
            BlockStatusLevel(
                priority=2,
                name="ready",
                display_name="Ready",
                color="#51cf66",
                conditions=[]
            )
        ]

    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DataItem]:
        """
        Process ExportClipsByClass block.
        
        Extracts audio clips from source audio based on event timing,
        and exports them to folders organized by classification.
        """
        if not HAS_AUDIO_LIBS:
            raise ProcessingError(
                "Audio libraries (numpy, soundfile, librosa) not installed.",
                block_id=block.id,
                block_name=block.name
            )

        # Get settings from block metadata (top-level keys)
        output_dir = block.metadata.get("output_dir")
        fmt = (block.metadata.get("audio_format") or "wav").lower()
        include_unclassified = block.metadata.get("include_unclassified", True)
        unclassified_folder = block.metadata.get("unclassified_folder", "unclassified")

        if not output_dir:
            raise ProcessingError(
                "ExportClipsByClass block requires output directory to be set before executing",
                block_id=block.id,
                block_name=block.name
            )

        if fmt not in self._SUPPORTED_FORMATS:
            raise ProcessingError(
                f"Unsupported export format '{fmt}'. Supported: {', '.join(self._SUPPORTED_FORMATS)}",
                block_id=block.id,
                block_name=block.name
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        Log.info(f"ExportClipsByClass: Exporting to directory: {output_path.absolute()}")

        # Get audio and events inputs
        audio_input = inputs.get("audio")
        events_input = inputs.get("events")

        if not events_input:
            Log.info(f"ExportClipsByClass: No events input for block {block.name}")
            return {}

        # Normalize inputs to lists
        audio_items = self._normalize_to_list(audio_input)
        event_items = self._normalize_to_list(events_input)

        if not audio_items:
            raise ProcessingError(
                "ExportClipsByClass requires audio input to extract clips",
                block_id=block.id,
                block_name=block.name
            )

        Log.info(f"ExportClipsByClass: Processing {len(event_items)} event item(s) with {len(audio_items)} audio source(s)")

        # Get progress tracker from metadata
        progress_tracker = get_progress_tracker(metadata)

        # Count total events for progress tracking
        total_events = sum(len(ei.get_events()) for ei in event_items if hasattr(ei, 'get_events'))
        
        # Use IncrementalProgress for manual step tracking
        progress = IncrementalProgress(progress_tracker, "Exporting clips by classification", total=total_events)

        exported_count = 0
        skipped_count = 0
        classification_counts = {}

        # Process each event item
        current_event = 0
        for event_item in event_items:
            if not hasattr(event_item, 'get_events'):
                Log.warning(f"ExportClipsByClass: Item {event_item.name} is not an EventDataItem, skipping")
                continue

            events = event_item.get_events()
            Log.info(f"ExportClipsByClass: Processing {len(events)} events from {event_item.name}")

            for event in events:
                current_event += 1
                
                # Update progress
                progress.step(f"Exported clip {current_event}/{total_events}")

                # Get classification for folder organization
                classification = event.classification.strip() if event.classification else ""
                
                if not classification:
                    if not include_unclassified:
                        skipped_count += 1
                        continue
                    classification = unclassified_folder

                # Sanitize classification for folder name
                safe_classification = self._sanitize_folder_name(classification)

                # Try to export the clip
                success = self._export_event_clip(
                    event=event,
                    audio_items=audio_items,
                    output_path=output_path,
                    classification_folder=safe_classification,
                    fmt=fmt,
                    event_item=event_item
                )

                if success:
                    exported_count += 1
                    classification_counts[classification] = classification_counts.get(classification, 0) + 1
                else:
                    skipped_count += 1

        # Log summary
        Log.info(f"ExportClipsByClass: Exported {exported_count} clips, skipped {skipped_count}")
        for cls, count in sorted(classification_counts.items()):
            Log.info(f"  - {cls}: {count} clips")

        progress.complete(f"Exported {exported_count} clips to {len(classification_counts)} folders")

        if exported_count == 0:
            raise ProcessingError(
                "ExportClipsByClass did not export any clips. Ensure events have valid timing metadata.",
                block_id=block.id,
                block_name=block.name
            )

        return {}

    def _normalize_to_list(self, data) -> List:
        """Normalize input to list format."""
        if data is None:
            return []
        if isinstance(data, list):
            return data
        return [data]

    def _sanitize_folder_name(self, name: str) -> str:
        """Sanitize classification name for use as folder name."""
        # Replace problematic characters
        invalid_chars = '<>:"/\\|?*'
        result = name
        for char in invalid_chars:
            result = result.replace(char, '_')
        # Limit length and strip whitespace
        return result.strip()[:100] or "unknown"

    def _export_event_clip(
        self,
        event: Event,
        audio_items: List[AudioDataItem],
        output_path: Path,
        classification_folder: str,
        fmt: str,
        event_item: EventDataItem
    ) -> bool:
        """
        Export a single event's audio clip.
        
        Returns True if export succeeded, False otherwise.
        """
        # Create classification subfolder
        class_folder = output_path / classification_folder
        class_folder.mkdir(parents=True, exist_ok=True)

        # Strategy 1: Event has audio_path metadata (pre-saved clip)
        audio_path = (
            event.metadata.get("audio_path") or
            event.metadata.get("file_path") or
            event.metadata.get("audio_file")
        )
        
        if audio_path and os.path.exists(audio_path):
            return self._copy_existing_clip(audio_path, class_folder, event, fmt)

        # Strategy 2: Extract clip from source audio using timing metadata
        clip_start = event.metadata.get("clip_start_time")
        clip_end = event.metadata.get("clip_end_time")
        
        # Fallback to event.time and duration if clip times not available
        if clip_start is None:
            clip_start = event.time
        if clip_end is None and event.duration > 0:
            clip_end = event.time + event.duration

        if clip_start is not None and clip_end is not None and clip_end > clip_start:
            return self._extract_and_save_clip(
                event=event,
                audio_items=audio_items,
                class_folder=class_folder,
                clip_start=clip_start,
                clip_end=clip_end,
                fmt=fmt,
                event_item=event_item
            )

        Log.warning(f"ExportClipsByClass: Event at time {event.time}s has no valid audio data or timing")
        return False

    def _copy_existing_clip(
        self,
        source_path: str,
        class_folder: Path,
        event: Event,
        fmt: str
    ) -> bool:
        """Copy an existing audio clip file to the classification folder."""
        import shutil
        
        try:
            source = Path(source_path)
            if not source.is_file():
                return False

            # Generate unique filename
            timestamp = datetime.utcnow().strftime("%H%M%S%f")[:10]
            filename = f"clip_{event.time:.3f}s_{timestamp}.{fmt}"
            dest = class_folder / filename

            # If source format matches target, just copy
            if source.suffix.lower() == f".{fmt}":
                shutil.copy2(source, dest)
            else:
                # Need to convert format
                audio_data, sr = librosa.load(str(source), sr=None)
                sf.write(str(dest), audio_data, sr)

            return True
        except Exception as e:
            Log.warning(f"ExportClipsByClass: Failed to copy clip: {e}")
            return False

    def _extract_and_save_clip(
        self,
        event: Event,
        audio_items: List[AudioDataItem],
        class_folder: Path,
        clip_start: float,
        clip_end: float,
        fmt: str,
        event_item: EventDataItem
    ) -> bool:
        """Extract audio clip from source and save to classification folder."""
        
        # Find matching source audio
        source_audio = self._find_source_audio(event, audio_items, event_item)
        
        if source_audio is None:
            Log.warning(f"ExportClipsByClass: Could not find source audio for event at {event.time}s")
            return False

        try:
            # Get audio data and sample rate
            audio_data = None
            sample_rate = event.metadata.get("sample_rate", 44100)

            # Try to get audio from the item
            if hasattr(source_audio, '_audio_data') and source_audio._audio_data is not None:
                audio_data = source_audio._audio_data
                sample_rate = source_audio.sample_rate or sample_rate
            elif hasattr(source_audio, 'audio_data') and source_audio.audio_data is not None:
                audio_data = source_audio.audio_data
                sample_rate = source_audio.sample_rate or sample_rate
            elif source_audio.file_path and os.path.exists(source_audio.file_path):
                audio_data, sample_rate = librosa.load(source_audio.file_path, sr=None)

            if audio_data is None:
                Log.warning(f"ExportClipsByClass: Could not load audio data for event at {event.time}s")
                return False

            # Ensure mono for extraction
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)

            # Extract clip
            start_sample = int(clip_start * sample_rate)
            end_sample = int(clip_end * sample_rate)
            
            # Bounds checking
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            if end_sample <= start_sample:
                Log.warning(f"ExportClipsByClass: Invalid clip range for event at {event.time}s")
                return False

            clip_audio = audio_data[start_sample:end_sample]

            # Generate unique filename
            timestamp = datetime.utcnow().strftime("%H%M%S%f")[:10]
            duration = clip_end - clip_start
            filename = f"clip_{clip_start:.3f}s_{duration:.3f}dur_{timestamp}.{fmt}"
            dest = class_folder / filename

            # Save clip
            sf.write(str(dest), clip_audio, sample_rate)
            
            return True

        except Exception as e:
            Log.warning(f"ExportClipsByClass: Failed to extract clip: {e}")
            return False

    def _find_source_audio(
        self,
        event: Event,
        audio_items: List[AudioDataItem],
        event_item: EventDataItem
    ) -> Optional[AudioDataItem]:
        """Find the source audio item for an event."""
        
        # Strategy 1: Match by audio_id in event metadata
        audio_id = event.metadata.get("audio_id")
        if audio_id:
            for audio in audio_items:
                if audio.id == audio_id:
                    return audio

        # Strategy 2: Match by audio_name in event metadata
        audio_name = event.metadata.get("audio_name")
        if audio_name:
            for audio in audio_items:
                if audio.name == audio_name:
                    return audio

        # Strategy 3: Match by source_audio in event_item metadata
        source_audio_path = event_item.metadata.get("source_audio")
        if source_audio_path:
            for audio in audio_items:
                if audio.file_path == source_audio_path:
                    return audio

        # Strategy 4: Use first audio item if only one available
        if len(audio_items) == 1:
            return audio_items[0]

        # Strategy 5: Return first audio item as fallback
        if audio_items:
            return audio_items[0]

        return None

    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None
    ) -> List[str]:
        """Validate block configuration before execution."""
        errors = []
        
        output_dir = block.metadata.get("output_dir")
        
        if not output_dir:
            errors.append("Output directory not set. Use 'Set Output Directory' action.")
        
        return errors


register_processor_class(ExportClipsByClassBlockProcessor)

