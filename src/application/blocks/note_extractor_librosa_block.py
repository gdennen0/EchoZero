"""
Note Extractor Block (Librosa)

Uses Librosa's onset detection and pitch tracking for note transcription.
More manual/configurable approach compared to Basic-Pitch.
Extracts individual notes with start/end times from audio (especially bass stems).
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, TYPE_CHECKING
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


class NoteExtractorLibrosaProcessor(BlockProcessor):
    """
    Processor for Librosa-based note extraction block.
    
    Uses Librosa's onset detection + pitch tracking (pYIN) to extract notes from audio.
    More configurable but requires more parameter tuning than Basic-Pitch.
    """
    
    def get_block_type(self) -> str:
        """Return the block type this processor handles"""
        return "TranscribeLib"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for NoteExtractorLibrosa block.
        
        Status levels:
        - Error (0): Audio input not connected
        - Warning (1): librosa not available
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
        
        def check_librosa_available(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if librosa is available."""
            try:
                import librosa
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
                conditions=[check_librosa_available]
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
        return block.type == "TranscribeLib"
    
    def process(self, block: Block, inputs: Dict[str, DataItem], metadata: Dict = None) -> Dict[str, DataItem]:
        """
        Process audio through Librosa to extract notes.
        
        Args:
            block: Block entity with configuration
            inputs: Dictionary of input data items (expects 'audio' key)
            metadata: Optional metadata for processing
            
        Returns:
            Dictionary with 'events' key containing EventDataItem with note events
        """
        try:
            import librosa
        except ImportError:
            raise RuntimeError(
                "librosa library not installed. "
                "Install with: pip install librosa"
            )
        
        if not inputs or "audio" not in inputs:
            Log.warning(f"NoteExtractorLibrosa: No audio input for block {block.name}")
            return {}
        
        audio_input = inputs["audio"]
        
        # Handle both single audio item and list of audio items
        audio_items = [audio_input] if isinstance(audio_input, DataItem) else audio_input
        
        # Get configuration from block metadata
        hop_length = int(block.metadata.get("hop_length", 512))
        onset_threshold = float(block.metadata.get("onset_threshold", 0.5))
        min_note_duration = float(block.metadata.get("min_note_duration", 0.05))  # 50ms
        fmin = float(block.metadata.get("fmin", 27.5))  # A0 for bass
        fmax = float(block.metadata.get("fmax", 1000.0))  # Upper bass range
        
        # Get progress tracker from metadata
        progress_tracker = metadata.get("progress_tracker") if metadata else None
        
        Log.info(f"NoteExtractorLibrosa: Processing {len(audio_items)} audio item(s)")
        Log.info(f"NoteExtractorLibrosa: onset_threshold={onset_threshold}, "
                f"min_duration={min_note_duration}s, freq_range=[{fmin}, {fmax}]Hz")
        
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
                Log.warning(f"NoteExtractorLibrosa: Audio item has no file_path")
                continue
            
            audio_path = Path(audio_item.file_path)
            if not audio_path.exists():
                Log.warning(f"NoteExtractorLibrosa: Audio file not found: {audio_path}")
                continue
            
            Log.info(f"NoteExtractorLibrosa: Analyzing {audio_path.name}...")
            
            # Load audio
            if progress_tracker:
                progress_tracker.update(message=f"Loading audio: {audio_path.name}...")
            y, sr = librosa.load(str(audio_path), sr=None)
            Log.info(f"NoteExtractorLibrosa: Loaded audio - sr={sr}Hz, duration={len(y)/sr:.2f}s")
            
            # 1. Detect onsets (note starts)
            if progress_tracker:
                progress_tracker.update(message="Detecting onsets...")
            onset_frames = librosa.onset.onset_detect(
                y=y,
                sr=sr,
                hop_length=hop_length,
                backtrack=True,
                delta=onset_threshold
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
            
            Log.info(f"NoteExtractorLibrosa: Detected {len(onset_times)} onsets")
            
            # 2. Track pitch using pYIN (probabilistic YIN)
            if progress_tracker:
                progress_tracker.update(message="Tracking pitch...")
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y=y,
                sr=sr,
                fmin=fmin,
                fmax=fmax,
                hop_length=hop_length
            )
            
            # Convert frame times
            times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
            
            # 3. Pair onsets with pitches to create notes
            if progress_tracker:
                progress_tracker.update(message="Creating notes...")
            notes = self._pair_onsets_with_pitches(
                onset_times=onset_times,
                pitch_times=times,
                pitch_values=f0,
                voiced_flag=voiced_flag,
                min_duration=min_note_duration
            )
            
            # Collect events by note name for EventLayers
            # Structure: EventDataItem -> EventLayers -> Events
            from collections import defaultdict
            events_by_note = defaultdict(list)
            
            # Add notes as events
            note_count = 0
            for note_start, note_end, note_freq in notes:
                if note_freq is None or np.isnan(note_freq):
                    continue
                
                duration = note_end - note_start
                midi_note = self._frequency_to_midi(note_freq)
                note_name = self._midi_to_note_name(midi_note)
                
                note_event = Event(
                    time=note_start,
                    classification=note_name,
                    duration=duration,
                    metadata={
                        "midi_note": midi_note,
                        "frequency_hz": float(note_freq),
                        "end_time": note_end,
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
                            "source": "librosa",
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
                    "sample_rate": int(sr),
                    "hop_length": hop_length,
                    "onset_threshold": onset_threshold,
                    "min_note_duration": min_note_duration,
                    "frequency_range": [fmin, fmax],
                    "extractor": "librosa"
                },
                layers=layers  # SINGLE SOURCE OF TRUTH: EventLayers
            )
            
            Log.info(
                f"NoteExtractorLibrosa: Extracted {note_count} notes in {len(layers)} layers "
                f"from {audio_path.name}"
            )
            
            # Optionally save as JSON file
            if block.metadata.get("save_to_file", False):
                output_dir = Path(block.metadata.get("output_dir", "data/note_extractions"))
                output_dir.mkdir(parents=True, exist_ok=True)
                
                json_path = output_dir / f"{audio_path.stem}_notes_librosa.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(event_item.to_dict(), f, indent=2)
                
                event_item.file_path = str(json_path)
                Log.info(f"NoteExtractorLibrosa: Saved notes to {json_path}")
            
            all_note_events.append(event_item)
        
        # Complete progress tracking
        if progress_tracker and len(audio_items) > 1:
            progress_tracker.complete(f"Extracted notes from {len(all_note_events)} audio file(s)")
        
        # Return all note event items on the 'events' port
        return {"events": all_note_events}
    
    def _pair_onsets_with_pitches(
        self,
        onset_times: np.ndarray,
        pitch_times: np.ndarray,
        pitch_values: np.ndarray,
        voiced_flag: np.ndarray,
        min_duration: float
    ) -> List[Tuple[float, float, float]]:
        """
        Pair detected onsets with tracked pitches to create notes.
        
        Returns:
            List of (start_time, end_time, frequency_hz) tuples
        """
        notes = []
        
        for i, onset in enumerate(onset_times):
            # Find the next onset (or end of audio) for note end time
            if i < len(onset_times) - 1:
                next_onset = onset_times[i + 1]
            else:
                next_onset = pitch_times[-1]
            
            # Find pitch values between this onset and the next
            mask = (pitch_times >= onset) & (pitch_times < next_onset) & voiced_flag
            
            if not np.any(mask):
                continue
            
            # Use median pitch in this segment
            segment_pitches = pitch_values[mask]
            median_pitch = np.nanmedian(segment_pitches)
            
            duration = next_onset - onset
            
            # Filter by minimum duration
            if duration >= min_duration:
                notes.append((float(onset), float(next_onset), float(median_pitch)))
        
        return notes
    
    def _frequency_to_midi(self, frequency: float) -> int:
        """Convert frequency in Hz to MIDI note number"""
        if frequency <= 0:
            return 0
        return int(round(69 + 12 * np.log2(frequency / 440.0)))
    
    def _midi_to_note_name(self, midi_number: int) -> str:
        """Convert MIDI number to note name (e.g., 60 -> C4)"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_number // 12) - 1
        note = note_names[midi_number % 12]
        return f"{note}{octave}"


# Auto-register this processor
    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None
    ) -> List[str]:
        """
        Validate NoteExtractorLibrosa block configuration before execution.

        Args:
            block: Block to validate
            data_item_repo: Data item repository (for checking upstream data)
            connection_repo: Connection repository (for checking connections)
            block_registry: Block registry (for getting expected input types)

        Returns:
            List of error messages (empty if valid)
        """
        # NoteExtractorLibrosaProcessor doesn't have specific validation requirements
        return []


register_processor_class(NoteExtractorLibrosaProcessor)

