"""
DetectOnsets Block Processor

Processes DetectOnsets blocks - detects onset times in audio using librosa.
"""
import os
from typing import Dict, Optional, Any, List, TYPE_CHECKING
from datetime import datetime
import numpy as np

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.domain.entities import AudioDataItem
from src.shared.domain.entities import EventDataItem, Event, EventLayer
from src.application.blocks import register_processor_class
from src.application.blocks.clip_end_detector import ClipEndDetector
from src.application.blocks.onset_detector import OnsetDetector
from src.application.blocks.audio_preprocessor import AudioPreprocessor
from src.application.settings.detect_onsets_settings import DetectOnsetsBlockSettings
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    Log.warning("librosa not available - DetectOnsets block will not work")


class DetectOnsetsBlockProcessor(BlockProcessor):
    """
    Processor for DetectOnsets block type.
    
    Modular two-phase detection process:
    
    Phase 1: Onset Detection (finding where events start)
    - Uses OnsetDetector module to detect onset times
    - Finds when audio events begin (attack/transient start)
    - Settings: onset_method, onset_threshold, min_silence, use_backtrack, energy_hop_length
    
    Phase 2: Clip End Detection (finding where events end) - Always runs
    - Uses ClipEndDetector module to determine when each event stops
    - Finds decay/end time for each detected onset
    - Settings: detection_mode, energy_decay_threshold/peak_decay_ratio, decay_detection_method, etc.
    
    Execution Workflow:
    1. For each audio item:
       a. Phase 1: onset_times = OnsetDetector.detect_onsets(audio, settings)
       b. Phase 2: For each onset: end_time = ClipEndDetector.detect_clip_end(onset, settings)
       c. Create events with duration from onset to end_time
       d. Set render_as_marker property based on output_mode setting (visual distinction only)
    
    All events have duration >= MIN_EVENT_DURATION. Visual distinction (marker vs clip) is
    controlled by render_as_marker property in event metadata.
    """
    
    def can_process(self, block: Block) -> bool:
        """Check if this processor can handle the block"""
        return block.type == "DetectOnsets"
    
    def get_block_type(self) -> str:
        """Get the block type this processor handles"""
        return "DetectOnsets"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for DetectOnsets block.
        
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
            return HAS_LIBROSA
        
        from src.shared.domain.data_state import DataState
        
        def check_data_fresh(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if block data is fresh (not stale)."""
            if not hasattr(f, 'data_state_service') or not f.data_state_service:
                return True  # If no data state service, assume fresh
            try:
                project_id = getattr(f, 'current_project_id', None) if hasattr(f, 'current_project_id') else None
                data_state = f.data_state_service.get_block_data_state(blk.id, project_id)
                return data_state != DataState.STALE
            except Exception:
                return True  # On error, assume fresh
        
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
                name="stale",
                display_name="Stale",
                color="#ffa94d",
                conditions=[check_data_fresh]
            ),
            BlockStatusLevel(
                priority=3,
                name="ready",
                display_name="Ready",
                color="#51cf66",
                conditions=[]
            )
        ]
    
    def get_expected_outputs(self, block: Block) -> Dict[str, List[str]]:
        """
        Get expected output names for DetectOnsets block.
        
        Returns default static output. Connection-based calculation is handled
        by ExpectedOutputsService.
        
        Args:
            block: Block entity
            connection_repo: Ignored (handled by ExpectedOutputsService)
            block_repo: Ignored (handled by ExpectedOutputsService)
            facade: Ignored (handled by ExpectedOutputsService)
        """
        # Return empty dict - ExpectedOutputsService handles all output calculation
        # This prevents validation warnings since actual outputs are connection-based
        return {}
    
    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DataItem]:
        """
        Process DetectOnsets block.
        
        Supports both single AudioDataItem and list of AudioDataItems.
        For multiple audio items, creates separate EventDataItem for each.
        
        Args:
            block: Block entity to process
            inputs: Input data items (should contain "audio" AudioDataItem or list)
            metadata: Optional metadata (not used currently)
            
        Returns:
            Dictionary with "events" port containing EventDataItem or list of EventDataItems
            
        Raises:
            ProcessingError: If audio input missing or processing fails
        """
        if not HAS_LIBROSA:
            raise ProcessingError(
                "librosa library not installed. Install with: pip install librosa",
                block_id=block.id,
                block_name=block.name
            )
        
        # Get audio input
        audio_input = inputs.get("audio")
        if not audio_input:
            raise ProcessingError(
                "Audio input required for DetectOnsets block",
                block_id=block.id,
                block_name=block.name
            )
        
        # Handle both single audio item and list of audio items
        if isinstance(audio_input, list):
            audio_items = audio_input
        elif isinstance(audio_input, AudioDataItem):
            audio_items = [audio_input]
        else:
            raise ProcessingError(
                f"Audio input must be AudioDataItem or list of AudioDataItems, got {type(audio_input)}",
                block_id=block.id,
                block_name=block.name
            )
        
        # Get progress tracker from metadata
        progress_tracker = metadata.get("progress_tracker") if metadata else None
        
        # Track execution summaries for all audio items
        all_execution_summaries = []
        
        Log.info(f"DetectOnsetsBlockProcessor: Processing {len(audio_items)} audio item(s)")
        
        # Load settings into DetectOnsetsBlockSettings object
        # This provides structured access to all settings for both phases
        settings = DetectOnsetsBlockSettings.from_dict(block.metadata)
        
        Log.info(
            f"DetectOnsetsBlockProcessor: Processing with "
            f"onset_method={settings.onset_method}, "
            f"onset_threshold={settings.onset_threshold}, "
            f"min_silence={settings.min_silence}s, "
            f"output_mode={settings.output_mode}"
        )
        
        all_event_items: List[EventDataItem] = []
        
        # Process items with progress tracking
        from src.features.execution.application.progress_helpers import track_progress, get_progress_tracker
        progress_tracker = get_progress_tracker(metadata)
        
        for audio_item in track_progress(audio_items, progress_tracker, "Processing audio items"):
            if not isinstance(audio_item, AudioDataItem):
                Log.warning(
                    f"DetectOnsetsBlockProcessor: Skipping non-AudioDataItem input: {type(audio_item)}"
                )
                continue
            
            # Log progress (track_progress handles progress updates automatically)
            Log.info(f"DetectOnsetsBlockProcessor: Processing audio '{audio_item.name}'")
            
            try:
                # Get audio data - load from file if not already in memory
                audio_data = audio_item.get_audio_data()
                if audio_data is None:
                    # Try to load from file_path if available
                    if audio_item.file_path and os.path.exists(audio_item.file_path):
                        Log.info(
                            f"DetectOnsetsBlockProcessor: Loading audio data from file: {audio_item.file_path}"
                        )
                        if not audio_item.load_audio(audio_item.file_path):
                            Log.warning(
                                f"DetectOnsetsBlockProcessor: Failed to load audio from file for '{audio_item.name}'. "
                                "Skipping this item."
                            )
                            continue
                        audio_data = audio_item.get_audio_data()
                    else:
                        Log.warning(
                            f"DetectOnsetsBlockProcessor: Audio data not loaded and no valid file_path "
                            f"for '{audio_item.name}'. Skipping this item."
                        )
                        continue
                
                sample_rate = audio_item.sample_rate
                if sample_rate is None:
                    Log.warning(
                        f"DetectOnsetsBlockProcessor: Audio sample rate not available for '{audio_item.name}'. "
                        "Skipping this item."
                    )
                    continue
                
                # ============================================================
                # PHASE 1: ONSET DETECTION (Finding where events start)
                # ============================================================
                # Use dedicated OnsetDetector module
                # librosa expects mono audio, so convert if needed
                if len(audio_data.shape) > 1:
                    # Multi-channel: convert to mono by averaging
                    audio_mono = np.mean(audio_data, axis=0)
                else:
                    audio_mono = audio_data
                
                # Apply audio preprocessing to improve onset detection
                # Preprocessing emphasizes transients and improves signal quality
                # This is especially helpful for closely-spaced onsets
                if settings.preprocessing_enabled:
                    preprocessor = AudioPreprocessor()
                    audio_mono = preprocessor.preprocess_audio(
                        audio=audio_mono,
                        sample_rate=sample_rate,
                        settings=settings
                    )
                    Log.debug(
                        f"DetectOnsetsBlockProcessor: Applied audio preprocessing "
                        f"(preemphasis={settings.preemphasis_enabled}, "
                        f"dc_removal={settings.remove_dc_offset}, "
                        f"highpass={settings.highpass_enabled})"
                    )
                
                # Detect onsets using OnsetDetector module
                # Settings already loaded above
                onset_detector = OnsetDetector()
                onset_times = onset_detector.detect_onsets(
                    audio_mono=audio_mono,
                    sample_rate=sample_rate,
                    settings=settings
                )
                # Phase 1 complete: We now have onset times (when events start)
                
                Log.info(
                    f"DetectOnsetsBlockProcessor: Detected {len(onset_times)} unique onsets "
                    f"in audio '{audio_item.name}'"
                )
                
                # Get render mode setting (determines visual display, not data structure)
                # Legacy: output_mode is now used to set render_as_marker property
                output_mode = block.metadata.get("output_mode", "markers")
                render_as_marker = bool(output_mode == "markers")  # Convert to Python bool for JSON serialization
                clip_classification = block.metadata.get("clip_classification", "clip")
                
                # Get clip detection parameters (always used - clip end detection always runs)
                max_clip_duration = float(block.metadata.get("max_clip_duration", 2.0))
                min_clip_duration = float(block.metadata.get("min_clip_duration", 0.01))
                
                # Import minimum duration constant
                from ui.qt_gui.widgets.timeline.types import MIN_EVENT_DURATION
                
                # Settings already loaded above, reuse for ClipEndDetector
                
                # Post-processing: Split clips with multiple onsets
                split_clips_with_multiple_onsets = bool(block.metadata.get("split_clips_with_multiple_onsets", False))
                
                Log.info(
                    f"DetectOnsetsBlockProcessor: Clip detection parameters - "
                    f"energy_decay_threshold={settings.energy_decay_threshold}, "
                    f"adaptive_threshold_enabled={settings.adaptive_threshold_enabled}, "
                    f"min_separation_time={settings.min_separation_time}s, "
                    f"early_cut_mode={settings.early_cut_mode}"
                )
                
                # Get audio duration for clip mode
                audio_duration = audio_item.length_ms / 1000.0 if audio_item.length_ms else 0.0
                
                # Initialize energy frames (used by both clips and markers modes)
                energy_frames = None
                energy_times = None
                
                # Collect events in a list, then create EventLayer and EventDataItem at the end
                # Structure: EventDataItem -> EventLayers -> Events
                collected_events: List[Event] = []
                
                # Determine layer name based on source audio
                # Always use clip classification (visual distinction is via render_as_marker property)
                audio_base_name = audio_item.name
                layer_name = f"{audio_base_name}-{clip_classification}"
                
                Log.debug(
                    f"DetectOnsetsBlockProcessor: Layer name determined - "
                    f"audio_item.name='{audio_item.name}', "
                    f"render_as_marker={render_as_marker}, "
                    f"layer_name='{layer_name}'"
                )
                
                # Determine output_name from audio input
                audio_output_name = audio_item.metadata.get('output_name')
                if audio_output_name:
                    from src.application.processing.output_name_helpers import parse_output_name, make_output_name
                    try:
                        port_name, item_name = parse_output_name(audio_output_name)
                        # Convert to events port with same item name
                        event_output_name = make_output_name("events", item_name)
                    except ValueError:
                        # Invalid format, use default
                        from src.application.processing.output_name_helpers import make_default_output_name
                        event_output_name = make_default_output_name("events")
                else:
                    # No output_name on input, use default
                    from src.application.processing.output_name_helpers import make_default_output_name
                    event_output_name = make_default_output_name("events")
                
                # Always run clip end detection - all events have duration
                # Compute RMS energy frames for accurate tail cutting
                    
                try:
                    # Compute RMS energy using librosa
                    energy_frames = librosa.feature.rms(
                        y=audio_mono,
                        frame_length=settings.energy_frame_length,
                        hop_length=settings.energy_hop_length
                    )[0]  # Get 1D array
                    
                    # Convert frame indices to time
                    frame_times = librosa.frames_to_time(
                        np.arange(len(energy_frames)),
                        sr=sample_rate,
                        hop_length=settings.energy_hop_length
                    )
                    energy_times = frame_times
                    
                    Log.debug(
                        f"DetectOnsetsBlockProcessor: Computed {len(energy_frames)} energy frames "
                        f"for clip end detection (frame_length={settings.energy_frame_length}, "
                        f"hop_length={settings.energy_hop_length})"
                    )
                except Exception as e:
                    Log.warning(
                        f"DetectOnsetsBlockProcessor: Failed to compute energy frames: {e}. "
                        "Falling back to simple duration calculation."
                    )
                
                # ============================================================
                # PHASE 2: CLIP END DETECTION (Finding where events end)
                # ============================================================
                # Always runs - all events have duration
                # For each detected onset, determine when the sound stops
                # Uses ClipEndDetector module to detect energy decay
                
                # Initialize clip end detector
                clip_end_detector = ClipEndDetector()
                
                # Process each onset to find accurate end time
                for i, onset_time in enumerate(onset_times):
                    # Initialize metadata values (will be populated during processing)
                    onset_amplitude = 0.0
                    onset_energy = 0.0
                    onset_strength_value = 0.0
                    peak_amplitude = 0.0
                    peak_energy_value = 0.0
                    peak_energy_time_value = onset_time
                    avg_amplitude = 0.0
                    avg_energy_value = 0.0
                    threshold_energy_value = 0.0
                    threshold_type_value = "absolute"
                    clip_scope_energy_for_avg = None
                    
                    # Default end time (next onset or end of audio)
                    if i < len(onset_times) - 1:
                        default_end_time = onset_times[i + 1]
                    else:
                        # Last onset: use end of audio
                        default_end_time = audio_duration if audio_duration > 0 else onset_time + 0.5
                    
                    # Detect clip end using dedicated module
                    end_time = default_end_time
                    next_onset_time_for_detector = onset_times[i + 1] if i < len(onset_times) - 1 else None
                    
                    if energy_frames is not None and energy_times is not None:
                        try:
                            # Find frames within this clip's potential duration
                            onset_frame_idx = np.searchsorted(energy_times, onset_time, side='left')
                            
                            # Calculate onset-time characteristics (before processing clip)
                            onset_energy = float(energy_frames[onset_frame_idx]) if onset_frame_idx < len(energy_frames) else 0.0
                            
                            # Onset strength is no longer available (computed inside OnsetDetector)
                            # Can be added back if needed by having OnsetDetector return it
                            onset_strength_value = 0.0
                            
                            # Calculate amplitude at onset time
                            onset_sample_idx = int(onset_time * sample_rate)
                            if onset_sample_idx < len(audio_mono):
                                onset_amplitude = float(np.abs(audio_mono[onset_sample_idx]))
                            
                            # Use ClipEndDetector to find clip end
                            end_time, detection_metadata = clip_end_detector.detect_clip_end(
                                onset_time=onset_time,
                                energy_frames=energy_frames,
                                energy_times=energy_times,
                                onset_frame_idx=onset_frame_idx,
                                next_onset_time=next_onset_time_for_detector,
                                settings=settings,
                                sample_rate=sample_rate,
                                audio_duration=audio_duration,
                                default_end_time=default_end_time
                            )
                            
                            # Extract metadata from detection
                            peak_energy_value = detection_metadata.get("peak_energy", 0.0)
                            peak_energy_time_value = detection_metadata.get("peak_energy_time", onset_time)
                            threshold_energy_value = detection_metadata.get("threshold_energy", 0.0)
                            threshold_type_value = detection_metadata.get("threshold_type", "absolute")
                            clip_scope_energy_for_avg = detection_metadata.get("clip_scope_energy_for_avg")
                            
                            # Apply safety constraints (don't override detection, just ensure bounds)
                            original_end_time = end_time
                            
                            # Safety: Don't extend into next onset
                            if next_onset_time_for_detector is not None:
                                max_end_time = next_onset_time_for_detector - settings.min_separation_time
                                if end_time > max_end_time:
                                    end_time = max_end_time
                                    Log.debug(
                                        f"DetectOnsetsBlockProcessor: Safety constraint adjusted end_time from {original_end_time:.3f}s "
                                        f"to {end_time:.3f}s (prevented extension into next onset at {next_onset_time_for_detector:.3f}s)"
                                    )
                            
                            # Safety: Don't exceed audio duration
                            if audio_duration > 0 and end_time > audio_duration:
                                original_end_time = end_time
                                end_time = audio_duration
                                Log.debug(
                                    f"DetectOnsetsBlockProcessor: Safety constraint adjusted end_time from {original_end_time:.3f}s "
                                    f"to {end_time:.3f}s (prevented exceeding audio duration)"
                                )
                            
                            # Calculate duration from final end_time
                            duration = end_time - onset_time
                            
                            # Enforce minimum duration (use MIN_EVENT_DURATION as absolute minimum)
                            if duration < MIN_EVENT_DURATION:
                                duration = MIN_EVENT_DURATION
                                end_time = onset_time + duration
                            elif duration < min_clip_duration:
                                duration = min_clip_duration
                                end_time = onset_time + duration
                            
                            # Enforce maximum duration
                            if duration > max_clip_duration:
                                duration = max_clip_duration
                                end_time = onset_time + duration
                            
                        except Exception as e:
                            Log.warning(
                                f"DetectOnsetsBlockProcessor: Error in clip end detection for "
                                f"onset at {onset_time:.3f}s: {e}. Using default end time."
                            )
                            # Use default end time on error
                            end_time = default_end_time
                            duration = end_time - onset_time
                            # Enforce minimum duration
                            if duration < MIN_EVENT_DURATION:
                                duration = MIN_EVENT_DURATION
                                end_time = onset_time + duration
                    else:
                        # No energy frames available, use default
                        duration = end_time - onset_time
                        # Enforce minimum duration
                        if duration < MIN_EVENT_DURATION:
                            duration = MIN_EVENT_DURATION
                            end_time = onset_time + duration
                    
                    # Calculate remaining metadata values (peak amplitude, averages)
                    # (Onset-time characteristics already calculated above if energy_frames available)
                    
                    # Calculate peak amplitude within clip
                    clip_start_sample = int(onset_time * sample_rate)
                    clip_end_sample = int(end_time * sample_rate)
                    clip_end_sample = min(clip_end_sample, len(audio_mono))
                    if clip_start_sample < clip_end_sample:
                        clip_audio_segment = audio_mono[clip_start_sample:clip_end_sample]
                        if len(clip_audio_segment) > 0:
                            peak_amplitude = float(np.max(np.abs(clip_audio_segment)))
                            avg_amplitude = float(np.mean(np.abs(clip_audio_segment)))
                    
                    # Calculate average energy (use clip_scope_energy if available, otherwise use clip segment)
                    if clip_scope_energy_for_avg is not None and len(clip_scope_energy_for_avg) > 0:
                        avg_energy_value = float(np.mean(clip_scope_energy_for_avg))
                    elif energy_frames is not None:
                        # Fallback: calculate from energy frames in clip range
                        onset_frame_idx = np.searchsorted(energy_times, onset_time, side='left') if energy_times is not None else 0
                        end_frame_idx = np.searchsorted(energy_times, end_time, side='right') if energy_times is not None else len(energy_frames)
                        if onset_frame_idx < end_frame_idx and end_frame_idx <= len(energy_frames):
                            clip_energy_segment = energy_frames[onset_frame_idx:end_frame_idx]
                            if len(clip_energy_segment) > 0:
                                avg_energy_value = float(np.mean(clip_energy_segment))
                    
                    # Create event with comprehensive metadata
                    # All events have duration - visual distinction via render_as_marker
                    # All values are Python native types (no sanitization needed)
                    event_metadata = {
                        "source": "DetectOnsets",
                        "render_as_marker": render_as_marker,  # Already Python bool
                        "audio_name": str(audio_item.name),
                        "audio_id": str(audio_item.id) if hasattr(audio_item, 'id') and audio_item.id else None,
                        "clip_start_time": float(onset_time),  # Convert numpy float to Python float
                        "clip_end_time": float(end_time),  # Convert numpy float to Python float
                        "sample_rate": int(sample_rate),
                        "_original_source_block_id": str(block.id),
                        # Onset-time characteristics (already Python floats)
                        "onset_amplitude": float(onset_amplitude),
                        "onset_energy": float(onset_energy),
                        "onset_strength": float(onset_strength_value),
                        # Peak characteristics (already Python floats from detection_metadata)
                        "peak_amplitude": float(peak_amplitude),
                        "peak_energy": float(peak_energy_value),
                        "peak_energy_time": float(peak_energy_time_value),
                        # Average characteristics (already Python floats)
                        "avg_amplitude": float(avg_amplitude),
                        "avg_energy": float(avg_energy_value),
                        # Threshold information (already Python types from detection_metadata)
                        "threshold_energy": float(threshold_energy_value),
                        "threshold_type": str(threshold_type_value)
                    }
                    
                    event = Event(
                        time=float(onset_time),
                        classification=str(clip_classification),
                        duration=float(duration),
                        metadata=event_metadata
                    )
                    
                    collected_events.append(event)
                    
                Log.info(
                    f"DetectOnsetsBlockProcessor: Created {len(onset_times)} events "
                    f"with duration (classification: '{clip_classification}', "
                    f"render_as_marker={render_as_marker})"
                )
                    
                # Post-processing: Split clips with multiple onsets if enabled
                if split_clips_with_multiple_onsets:
                    original_event_count = len(collected_events)
                    split_count = 0
                    total_splits = 0
                    
                    # Process all collected events for potential splitting
                    events_to_split = collected_events.copy()
                    collected_events.clear()
                    
                    for original_event in events_to_split:
                        clip_start = original_event.time
                        clip_end = original_event.time + original_event.duration
                        clip_duration = clip_end - clip_start
                        
                        # Only check clips with reasonable duration (skip very short clips)
                        if clip_duration < 0.01:  # Less than 10ms, likely fine
                            collected_events.append(original_event)
                            continue
                        
                        try:
                            # Extract audio segment for this clip
                            start_sample = int(clip_start * sample_rate)
                            end_sample = int(clip_end * sample_rate)
                            start_sample = max(0, min(start_sample, len(audio_mono) - 1))
                            end_sample = max(start_sample + 1, min(end_sample, len(audio_mono)))
                            
                            if end_sample <= start_sample:
                                # Invalid segment, keep original
                                collected_events.append(original_event)
                                continue
                            
                            clip_audio = audio_mono[start_sample:end_sample]
                            
                            if len(clip_audio) < 512:  # Too short for reliable onset detection
                                collected_events.append(original_event)
                                continue
                            
                            # Run onset detection on clip segment (use simpler parameters for local detection)
                            clip_onset_times = librosa.onset.onset_detect(
                                y=clip_audio,
                                sr=sample_rate,
                                units='time',
                                backtrack=True,
                                delta=settings.onset_threshold * 0.7,  # Slightly lower threshold for sub-clip detection
                                hop_length=settings.energy_hop_length
                            )
                            
                            # Filter onsets to be within clip bounds (with small tolerance)
                            clip_onset_times = [t for t in clip_onset_times if 0 <= t < clip_duration]
                            
                            if len(clip_onset_times) > 1:
                                # Multiple onsets found - split the clip
                                split_count += 1
                                total_splits += len(clip_onset_times) - 1  # Count additional splits
                                
                                # Convert local times to absolute times
                                absolute_onset_times = [clip_start + t for t in clip_onset_times]
                                
                                # Create split events
                                for i, split_onset_time in enumerate(absolute_onset_times):
                                    # Determine end time: next split onset or original clip end
                                    if i + 1 < len(absolute_onset_times):
                                        split_end_time = absolute_onset_times[i + 1]
                                    else:
                                        split_end_time = clip_end
                                    
                                    split_duration = split_end_time - split_onset_time
                                    
                                    # Enforce minimum duration
                                    if split_duration < MIN_EVENT_DURATION:
                                        split_duration = MIN_EVENT_DURATION
                                        split_end_time = split_onset_time + split_duration
                                    
                                    # Copy original metadata and update timing
                                    # Original metadata already contains Python native types, just update timing fields
                                    split_metadata = original_event.metadata.copy()
                                    split_metadata["clip_start_time"] = float(split_onset_time)
                                    split_metadata["clip_end_time"] = float(split_end_time)
                                    split_metadata["_split_from_clip"] = True
                                    split_metadata["_split_index"] = int(i)
                                    split_metadata["_split_total"] = int(len(absolute_onset_times))
                                    
                                    split_event = Event(
                                        time=float(split_onset_time),
                                        classification=str(original_event.classification),
                                        duration=float(split_duration),
                                        metadata=split_metadata
                                    )
                                    collected_events.append(split_event)
                                
                                Log.debug(
                                    f"DetectOnsetsBlockProcessor: Split clip at {clip_start:.3f}s "
                                    f"({clip_duration:.3f}s) into {len(absolute_onset_times)} clips "
                                    f"(found {len(clip_onset_times)} local onsets)"
                                )
                            else:
                                # Single onset (or none) - keep original clip
                                collected_events.append(original_event)
                        except Exception as e:
                            Log.warning(
                                f"DetectOnsetsBlockProcessor: Failed to split clip at {clip_start:.3f}s: {e}. "
                                f"Keeping original clip."
                            )
                            # On error, keep original clip
                            collected_events.append(original_event)
                    
                    if split_count > 0:
                        Log.info(
                            f"DetectOnsetsBlockProcessor: Post-processing split {split_count} clips "
                            f"with multiple onsets ({original_event_count} -> {len(collected_events)} events, "
                            f"{total_splits} additional splits)"
                        )
                
                # Create EventLayer from collected events
                # Structure: EventDataItem -> EventLayers -> Events
                event_layer = EventLayer(
                    name=layer_name,  # CRITICAL: Layer name must match what we search for
                    events=collected_events,
                    metadata={
                        "source": "DetectOnsets",
                        "audio_name": audio_item.name,
                        "audio_id": audio_item.id if hasattr(audio_item, 'id') else None
                    }
                )
                
                # Create EventDataItem with the layer
                event_item = EventDataItem(
                    id="",  # Will be generated
                    block_id=block.id,
                    name=f"{block.name}_{audio_item.name}_events",
                    type="Event",
                    metadata={"output_name": event_output_name},
                    layers=[event_layer]  # EXPLICIT EventLayers - single source of truth
                )
                
                # Instrumentation: Log layer naming for debugging
                Log.info(
                    f"DetectOnsetsBlockProcessor: Created EventLayer - "
                    f"layer_name='{layer_name}', "
                    f"source_audio='{audio_item.name}', "
                    f"event_count={len(collected_events)}, "
                    f"EventDataItem_name='{event_item.name}'"
                )
                
                # Store execution summary statistics for this audio item
                # Convert all numpy types to Python native types for JSON serialization
                execution_summary = {
                    "audio_name": str(audio_item.name),
                    "audio_id": str(audio_item.id) if hasattr(audio_item, 'id') and audio_item.id else None,
                    "onset_count": int(len(onset_times)),
                    "event_count": int(len(collected_events)),
                    "render_as_marker": bool(render_as_marker),
                    "sample_rate": int(sample_rate),
                    "audio_duration": float(audio_duration),
                    "split_clips_enabled": bool(split_clips_with_multiple_onsets)
                }
                
                # Count splits if enabled
                if split_clips_with_multiple_onsets:
                    # Try to count splits from events metadata
                    split_clips_count = sum(
                        1 for event in collected_events
                        if event.metadata.get("_split_from_clip", False) and event.metadata.get("_split_index", 0) == 0
                    )
                    execution_summary["split_clips_count"] = split_clips_count
                
                Log.info(
                    f"DetectOnsetsBlockProcessor: Created EventDataItem '{event_item.name}' with "
                    f"{len(collected_events)} events in layer '{layer_name}' from audio '{audio_item.name}'"
                )
                
                all_event_items.append(event_item)
                all_execution_summaries.append(execution_summary)
                
            except Exception as e:
                Log.error(
                    f"DetectOnsetsBlockProcessor: Failed to process audio '{audio_item.name}': {e}"
                )
                # Continue processing other audio items instead of failing completely
                continue
        
        if not all_event_items:
            raise ProcessingError(
                "No event items created. Check audio inputs and try again.",
                block_id=block.id,
                block_name=block.name
            )
        
        # Store execution summary in block metadata for UI display
        if all_execution_summaries:
            total_events = sum(s["event_count"] for s in all_execution_summaries)
            total_onsets = sum(s["onset_count"] for s in all_execution_summaries)
            summary_data = {
                "last_execution": {
                    "timestamp": datetime.now().isoformat(),
                    "audio_items_processed": len(all_execution_summaries),
                    "total_onsets_detected": total_onsets,
                    "total_events_created": total_events,
                    "details": all_execution_summaries
                }
            }
            # Merge with existing metadata
            if block.metadata:
                block.metadata.update(summary_data)
            else:
                block.metadata = summary_data
            Log.debug(f"DetectOnsetsBlockProcessor: Stored execution summary: {total_onsets} onsets, {total_events} events")
        
        # Note: track_progress() handles completion automatically
        
        # Return single item if only one, otherwise return list
        if len(all_event_items) == 1:
            return {"events": all_event_items[0]}
        else:
            Log.info(
                f"DetectOnsetsBlockProcessor: Created {len(all_event_items)} event items "
                f"from {len(audio_items)} audio items"
            )
            return {"events": all_event_items}


    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None
    ) -> List[str]:
        """
        Validate DetectOnsets block configuration before execution.

        Args:
            block: Block to validate
            data_item_repo: Data item repository (for checking upstream data)
            connection_repo: Connection repository (for checking connections)
            block_registry: Block registry (for getting expected input types)

        Returns:
            List of error messages (empty if valid)
        """
        # DetectOnsetsBlock doesn't have specific validation requirements
        return []


# Auto-register this processor class
register_processor_class(DetectOnsetsBlockProcessor)
