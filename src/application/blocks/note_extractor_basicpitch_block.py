"""
Note Extractor Block (Basic-Pitch)

Uses Spotify's Basic-Pitch for deep learning-based note transcription.
Extracts individual notes with precise start/end times from audio (especially bass stems).
"""
from pathlib import Path
from typing import Dict, List, Any, TYPE_CHECKING
from datetime import datetime
import json
import numpy as np

from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem, AudioDataItem
from src.shared.domain.entities import EventDataItem, Event, EventLayer
from src.application.processing.block_processor import BlockProcessor
from src.application.blocks import register_processor_class
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade


class NoteExtractorBasicPitchProcessor(BlockProcessor):
    """
    Processor for Basic-Pitch note extraction block.
    
    Uses Spotify's Basic-Pitch deep learning model to extract notes from audio.
    Outputs note events with start time, duration, pitch (MIDI number and name), and velocity.
    """
    
    def get_block_type(self) -> str:
        """Return the block type this processor handles"""
        return "TranscribeNote"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for NoteExtractorBasicPitch block.
        
        Status levels:
        - Error (0): Audio input not connected
        - Warning (1): Basic-Pitch not available
        - Ready (2): All requirements met
        
        Args:
            block: Block entity to get status levels for
            facade: ApplicationFacade for accessing services
            
        Returns:
            List of BlockStatusLevel instances in priority order
        """
        from src.features.blocks.domain import BlockStatusLevel
        
        def check_audio_input(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if audio input is connected."""
            if not hasattr(f, 'connection_service'):
                return False
            connections = f.connection_service.list_connections_by_block(blk.id)
            incoming = [c for c in connections if c.target_block_id == blk.id and c.target_input_name == "audio"]
            return len(incoming) > 0
        
        def check_basicpitch_available(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if Basic-Pitch is available."""
            try:
                import basicpitch
                return True
            except ImportError:
                return False
        
        return [
            BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[check_audio_input]
            ),
            BlockStatusLevel(
                priority=1,
                name="warning",
                display_name="Warning",
                color="#ffd43b",
                conditions=[check_basicpitch_available]
            ),
            BlockStatusLevel(
                priority=2,
                name="ready",
                display_name="Ready",
                color="#51cf66",
                conditions=[]
            )
        ]
    
    def can_process(self, block: Block) -> bool:
        """Check if this processor can handle the given block"""
        return block.type == "TranscribeNote"
    
    def process(self, block: Block, inputs: Dict[str, DataItem], metadata: Dict = None) -> Dict[str, DataItem]:
        """
        Process audio through Basic-Pitch to extract notes.
        
        Args:
            block: Block entity with configuration
            inputs: Dictionary of input data items (expects 'audio' key)
            metadata: Optional metadata for processing
            
        Returns:
            Dictionary with 'events' key containing EventDataItem with note events
        """
        try:
            import basic_pitch
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
        except ImportError:
            raise RuntimeError(
                "basic-pitch library not installed. "
                "Install with: pip install basic-pitch"
            )
        
        if not inputs or "audio" not in inputs:
            Log.warning(f"NoteExtractorBasicPitch: No audio input for block {block.name}")
            return {}
        
        audio_input = inputs["audio"]
        
        # Handle both single audio item and list of audio items
        audio_items = [audio_input] if isinstance(audio_input, DataItem) else audio_input
        
        # Get configuration from block metadata
        onset_threshold = float(block.metadata.get("onset_threshold", 0.5))
        frame_threshold = float(block.metadata.get("frame_threshold", 0.3))
        minimum_note_length = float(block.metadata.get("minimum_note_length", 0.058))  # ~58ms default
        minimum_frequency = float(block.metadata.get("minimum_frequency", 27.5))  # A0 (bass range)
        maximum_frequency = float(block.metadata.get("maximum_frequency", 1000.0))  # Upper bass range
        
        # Get progress tracker from metadata
        progress_tracker = metadata.get("progress_tracker") if metadata else None
        
        Log.info(f"NoteExtractorBasicPitch: Processing {len(audio_items)} audio item(s)")
        Log.info(f"NoteExtractorBasicPitch: onset_threshold={onset_threshold}, "
                f"frame_threshold={frame_threshold}, min_note_length={minimum_note_length}s")
        
        all_note_events: List[EventDataItem] = []
        
        # Initialize progress tracking
        if progress_tracker and len(audio_items) > 1:
            progress_tracker.start(f"Extracting notes from {len(audio_items)} audio files...", total=len(audio_items))
        
        for item_idx, audio_item in enumerate(audio_items, 1):
            # Update progress
            if progress_tracker and len(audio_items) > 1:
                progress_tracker.update(
                    current=item_idx,
                    message=f"Extracting notes from audio {item_idx}/{len(audio_items)}"
                )
            
            if not audio_item.file_path:
                Log.warning(f"NoteExtractorBasicPitch: Audio item has no file_path")
                continue
            
            audio_path = Path(audio_item.file_path)
            if not audio_path.exists():
                Log.warning(f"NoteExtractorBasicPitch: Audio file not found: {audio_path}")
                continue
            
            Log.info(f"NoteExtractorBasicPitch: Analyzing {audio_path.name}...")
            
            # Update progress for model inference
            if progress_tracker:
                progress_tracker.update(message=f"Running Basic-Pitch model on {audio_path.name}...")
            
            # Run Basic-Pitch inference
            model_output, midi_data, note_events = predict(
                audio_path=str(audio_path),
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                minimum_note_length=minimum_note_length,
                minimum_frequency=minimum_frequency,
                maximum_frequency=maximum_frequency
            )
            
            # Collect events by note name for EventLayers
            # Structure: EventDataItem -> EventLayers -> Events
            from collections import defaultdict
            events_by_note = defaultdict(list)
            
            # Convert note_events (numpy array) to Event objects
            # note_events format: [start_time, end_time, pitch_midi, velocity, (optional) pitch_bend]
            note_count = 0
            for note in note_events:
                start_time = float(note[0])
                end_time = float(note[1])
                pitch_midi = int(note[2])
                velocity = int(note[3]) if len(note) > 3 else 127
                
                duration = end_time - start_time
                
                # Convert MIDI number to note name
                note_name = self._midi_to_note_name(pitch_midi)
                
                # Create event with note information
                note_event = Event(
                    time=start_time,
                    classification=note_name,
                    duration=duration,
                    metadata={
                        "midi_note": pitch_midi,
                        "velocity": velocity,
                        "end_time": end_time,
                        "frequency_hz": self._midi_to_frequency(pitch_midi),
                        "_original_source_block_id": block.id
                    }
                )
                events_by_note[note_name].append(note_event)
                note_count += 1
            
            # Create EventLayers from grouped notes
            layers = []
            for note_name, layer_events in events_by_note.items():
                if layer_events:
                    layer = EventLayer(
                        name=note_name,
                        events=layer_events,
                        metadata={
                            "source": "basic-pitch",
                            "event_count": len(layer_events)
                        }
                    )
                    layers.append(layer)
            
            # Create EventDataItem with EventLayers
            event_item = EventDataItem(
                id="",
                block_id=block.id,
                name=f"{block.name}_{audio_path.stem}_notes",
                type="Event",
                created_at=datetime.utcnow(),
                metadata={
                    "source_audio": str(audio_path),
                    "source_audio_name": audio_item.name,
                    "onset_threshold": onset_threshold,
                    "frame_threshold": frame_threshold,
                    "minimum_note_length": minimum_note_length,
                    "frequency_range": [minimum_frequency, maximum_frequency],
                    "extractor": "basic-pitch"
                },
                layers=layers  # SINGLE SOURCE OF TRUTH: EventLayers
            )
            
            Log.info(
                f"NoteExtractorBasicPitch: Extracted {note_count} notes in {len(layers)} layers "
                f"from {audio_path.name}"
            )
            
            # Optionally save as JSON file
            if block.metadata.get("save_to_file", False):
                output_dir = Path(block.metadata.get("output_dir", "data/note_extractions"))
                output_dir.mkdir(parents=True, exist_ok=True)
                
                json_path = output_dir / f"{audio_path.stem}_notes.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(event_item.to_dict(), f, indent=2)
                
                event_item.file_path = str(json_path)
                Log.info(f"NoteExtractorBasicPitch: Saved notes to {json_path}")
            
            all_note_events.append(event_item)
        
        # Complete progress tracking
        if progress_tracker and len(audio_items) > 1:
            progress_tracker.complete(f"Extracted notes from {len(all_note_events)} audio file(s)")
        
        # Return all note event items on the 'events' port
        return {"events": all_note_events}
    
    def _midi_to_note_name(self, midi_number: int) -> str:
        """Convert MIDI number to note name (e.g., 60 -> C4)"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_number // 12) - 1
        note = note_names[midi_number % 12]
        return f"{note}{octave}"
    
    def _midi_to_frequency(self, midi_number: int) -> float:
        """Convert MIDI number to frequency in Hz"""
        return 440.0 * (2.0 ** ((midi_number - 69) / 12.0))


# Auto-register this processor
    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None
    ) -> List[str]:
        """
        Validate NoteExtractorBasicPitch block configuration before execution.

        Args:
            block: Block to validate
            data_item_repo: Data item repository (for checking upstream data)
            connection_repo: Connection repository (for checking connections)
            block_registry: Block registry (for getting expected input types)

        Returns:
            List of error messages (empty if valid)
        """
        # NoteExtractorBasicPitchProcessor doesn't have specific validation requirements
        return []


register_processor_class(NoteExtractorBasicPitchProcessor)

