"""
Block Type Registry

Manages available block types and their metadata.
Provides block type discovery and registration.
"""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from src.shared.domain.value_objects.port_type import PortType, AUDIO_TYPE, EVENT_TYPE, MANIPULATOR_TYPE
from src.utils.message import Log


@dataclass
class BlockTypeMetadata:
    """Metadata for a block type"""
    name: str  # Display name
    type_id: str  # Unique identifier (e.g., "LoadAudio", "DetectOnsets")
    description: str = ""
    inputs: Dict[str, PortType] = field(default_factory=dict)  # Input port definitions
    outputs: Dict[str, PortType] = field(default_factory=dict)  # Output port definitions
    bidirectional: Dict[str, PortType] = field(default_factory=dict)  # Bidirectional ports (both in/out)
    execution_mode: str = "executable"  # "executable" | "live" | "passthrough"
    icon: Optional[str] = None  # Icon path or identifier
    tags: List[str] = field(default_factory=list)  # Search tags
    commands: List[Dict[str, Any]] = field(default_factory=list)  # CLI commands/help info


class BlockTypeRegistry:
    """
    Registry for block types.
    
    Manages available block types and their metadata.
    Block types define what inputs/outputs they have and other metadata.
    """
    
    def __init__(self):
        """Initialize block type registry"""
        self._block_types: Dict[str, BlockTypeMetadata] = {}
        self._initialized = False
        Log.info("BlockTypeRegistry: Initialized")
    
    def register(self, block_type: BlockTypeMetadata) -> None:
        """
        Register a block type.
        
        Args:
            block_type: BlockTypeMetadata instance
        """
        if block_type.type_id in self._block_types:
            Log.warning(f"BlockTypeRegistry: Overwriting existing block type: {block_type.type_id}")
        
        self._block_types[block_type.type_id] = block_type
        Log.info(f"BlockTypeRegistry: Registered block type '{block_type.type_id}' ({block_type.name})")
    
    def get(self, type_id: str) -> Optional[BlockTypeMetadata]:
        """
        Get block type metadata by ID (case-insensitive).
        
        Args:
            type_id: Block type identifier
            
        Returns:
            BlockTypeMetadata or None if not found
        """
        # Try exact match first (for backwards compatibility)
        exact_match = self._block_types.get(type_id)
        if exact_match:
            return exact_match
        
        # If no exact match, try case-insensitive search
        type_id_lower = type_id.lower()
        for registered_type_id, metadata in self._block_types.items():
            if registered_type_id.lower() == type_id_lower:
                return metadata
        
        return None
    
    def list_all(self) -> List[BlockTypeMetadata]:
        """
        List all registered block types.
        
        Returns:
            List of BlockTypeMetadata objects
        """
        return list(self._block_types.values())
    
    def get_execution_mode(self, type_id: str) -> str:
        """
        Get execution mode for a block type.
        
        Args:
            type_id: Block type identifier
            
        Returns:
            Execution mode string ("executable", "live", or "passthrough")
            Defaults to "executable" if block type not found
        """
        metadata = self.get(type_id)
        return metadata.execution_mode if metadata else "executable"
    
    def search(self, query: str) -> List[BlockTypeMetadata]:
        """
        Search block types by name, description, or tags.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching BlockTypeMetadata objects
        """
        query_lower = query.lower()
        results = []
        
        for block_type in self._block_types.values():
            # Search in name
            if query_lower in block_type.name.lower():
                results.append(block_type)
                continue
            
            # Search in type_id
            if query_lower in block_type.type_id.lower():
                results.append(block_type)
                continue
            
            # Search in description
            if query_lower in block_type.description.lower():
                results.append(block_type)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in block_type.tags):
                results.append(block_type)
                continue
        
        return results
    
    def initialize_default_types(self):
        """Initialize default block types"""
        if self._initialized:
            return
        
        # Load Audio - Executable block
        self.register(BlockTypeMetadata(
            name="Load Audio",
            type_id="LoadAudio",
            description="Load audio file into the project",
            execution_mode="executable",
            outputs={"audio": AUDIO_TYPE},
            tags=["audio", "load", "file", "import"],
            commands=[
                {
                    "name": "set_path",
                    "usage": "set_path <path>",
                    "description": "Register the audio file to load for this block",
                    "arguments": [
                        {
                            "name": "path",
                            "required": True,
                            "source": "positional",
                            "description": "Path to the audio file"
                        }
                    ]
                }
            ]
        ))
        
        # Setlist Audio Input - Executable block (for setlist processing)
        self.register(BlockTypeMetadata(
            name="Setlist Audio Input",
            type_id="SetlistAudioInput",
            description="Audio input block for setlist processing. Automatically loads the current setlist song's audio file.",
            execution_mode="executable",
            outputs={"audio": AUDIO_TYPE},
            tags=["audio", "setlist", "batch", "input"],
            commands=[
                {
                    "name": "set_audio_file",
                    "usage": "set_audio_file <path>",
                    "description": "Set the audio file path for this setlist song",
                    "arguments": [
                        {
                            "name": "path",
                            "required": True,
                            "source": "positional",
                            "description": "Path to the audio file"
                        }
                    ]
                }
            ]
        ))
        
        # Detect Onsets - Executable block
        self.register(BlockTypeMetadata(
            name="Detect Onsets",
            type_id="DetectOnsets",
            description="Detect onset times in audio",
            execution_mode="executable",
            inputs={"audio": AUDIO_TYPE},
            outputs={"events": EVENT_TYPE},
            tags=["onset", "detection", "analysis", "audio"],
            commands=[
                {
                    "name": "tune_sensitivity",
                    "usage": "tune_sensitivity <float>",
                    "description": "Adjust onset sensitivity threshold for detection",
                    "arguments": [
                        {
                            "name": "float",
                            "required": True,
                            "source": "positional",
                            "description": "Sensitivity value"
                        }
                    ]
                }
            ]
        ))

        # Learned Onset Detector - Executable block
        self.register(BlockTypeMetadata(
            name="Learned Onset Detector",
            type_id="LearnedOnsetDetector",
            description="Detect onset times in audio using a learned probability curve (with optional fallback)",
            execution_mode="executable",
            inputs={"audio": AUDIO_TYPE},
            outputs={"events": EVENT_TYPE},
            tags=["onset", "detection", "audio", "pytorch", "cnn", "learned"],
            commands=[]
        ))
        
        # Drum Classify - Executable block
        # TensorFlowClassify - Executable block
        self.register(BlockTypeMetadata(
            name="TensorFlow Classify",
            type_id="TensorFlowClassify",
            description="Classify events using TensorFlow/Keras models (.h5, .keras, SavedModel)",
            execution_mode="executable",
            inputs={"events": EVENT_TYPE},
            outputs={"events": EVENT_TYPE},
            tags=["classification", "tensorflow", "keras", "ai", "ml", "deep-learning"]
        ))
        
        # PyTorchAudioTrainer - Executable block
        self.register(BlockTypeMetadata(
            name="PyTorch Audio Trainer",
            type_id="PyTorchAudioTrainer",
            description="Advanced PyTorch audio classification trainer with multiple architectures, transfer learning, and hyperparameter optimization",
            execution_mode="executable",
            outputs={"model": PortType("Model")},
            tags=["training", "pytorch", "ai", "ml", "deep-learning", "audio", "classification", "cnn", "rnn", "transformer", "transfer-learning"]
        ))

        # Learned Onset Trainer - Executable block
        self.register(BlockTypeMetadata(
            name="Learned Onset Trainer",
            type_id="LearnedOnsetTrainer",
            description="Train a CNN onset detector from frame-level drum onset annotations",
            execution_mode="executable",
            outputs={"model": PortType("Model")},
            tags=["training", "onset", "pytorch", "audio", "cnn", "drums"]
        ))
        
        # PyTorchAudioClassify - Executable block
        self.register(BlockTypeMetadata(
            name="PyTorch Audio Classify",
            type_id="PyTorchAudioClassify",
            description="Classify events using PyTorch models created by PyTorch Audio Trainer. Automatically loads model architecture, classes, and preprocessing config. Connect a Trainer's model output or set model_path in settings.",
            execution_mode="executable",
            inputs={"events": EVENT_TYPE, "audio": AUDIO_TYPE, "model": PortType("Model")},
            outputs={"events": EVENT_TYPE},
            tags=["classification", "pytorch", "ai", "ml", "deep-learning", "audio", "audio-trainer"]
        ))
        
        # Editor - Live block (maintains editable state)
        self.register(BlockTypeMetadata(
            name="Editor",
            type_id="Editor",
            description="Audio editor and visualization",
            execution_mode="live",  # Live block: maintains editable state
            inputs={"audio": AUDIO_TYPE, "events": EVENT_TYPE},
            outputs={"audio": AUDIO_TYPE, "events": EVENT_TYPE},
            bidirectional={"manipulator": MANIPULATOR_TYPE},  # Bidirectional command port
            tags=["editor", "visualization", "audio", "timeline"],
            commands=[
                {
                    "name": "zoom",
                    "usage": "zoom <level>",
                    "description": "Adjust the editor zoom level",
                    "arguments": [
                        {
                            "name": "level",
                            "required": True,
                            "source": "positional",
                            "description": "Zoom level multiplier"
                        }
                    ]
                }
            ]
        ))
        
        # Export Audio - Executable block
        self.register(BlockTypeMetadata(
            name="Export Audio",
            type_id="ExportAudio",
            description="Export processed audio items to disk in a chosen format (accepts multiple items on single port)",
            execution_mode="executable",
            inputs={"audio": AUDIO_TYPE},  # Single port accepts all audio items
            tags=["export", "audio", "stems", "file"],
            commands=[
                {
                    "name": "set_output_dir",
                    "usage": "set_output_dir <directory>",
                    "description": "Directory where exported audio files will be written",
                    "arguments": [
                        {
                            "name": "directory",
                            "required": True,
                            "source": "positional",
                            "description": "Filesystem path for exported audio"
                        }
                    ]
                },
                {
                    "name": "set_format",
                    "usage": "set_format <wav|mp3|flac|ogg|m4a|aiff>",
                    "description": "Choose the output audio format",
                    "arguments": [
                        {
                            "name": "format",
                            "required": True,
                            "source": "positional",
                            "description": "Audio format/extension"
                        }
                    ]
                }
            ]
        ))
        
        # Export Clips By Class - Executable block
        self.register(BlockTypeMetadata(
            name="Export Clips By Class",
            type_id="ExportClipsByClass",
            description="Export event audio clips organized by classification into subfolders",
            execution_mode="executable",
            inputs={"audio": AUDIO_TYPE, "events": EVENT_TYPE},
            tags=["export", "clips", "classification", "organize", "audio"],
            commands=[
                {
                    "name": "set_output_dir",
                    "usage": "set_output_dir <directory>",
                    "description": "Base directory for exported clips (subfolders created per class)",
                    "arguments": [
                        {
                            "name": "directory",
                            "required": True,
                            "source": "positional",
                            "description": "Filesystem path for exported clips"
                        }
                    ]
                },
                {
                    "name": "set_format",
                    "usage": "set_format <wav|mp3|flac|ogg>",
                    "description": "Choose the output audio format",
                    "arguments": [
                        {
                            "name": "format",
                            "required": True,
                            "source": "positional",
                            "description": "Audio format/extension"
                        }
                    ]
                }
            ]
        ))
        
        # Demucs Separator - Executable block
        self.register(BlockTypeMetadata(
            name="Demucs Separator",
            type_id="Separator",
            description="Separate audio into stems using Demucs (outputs multiple stems on single port)",
            execution_mode="executable",
            inputs={"audio": AUDIO_TYPE},
            outputs={"audio": AUDIO_TYPE},  # Single port carries all stems
            tags=["separator", "demucs", "stem", "audio"],
            commands=[
                {
                    "name": "set_model",
                    "usage": "set_model <demucs_model>",
                    "description": "Choose which Demucs model to run",
                    "arguments": [
                        {
                            "name": "demucs_model",
                            "required": True,
                            "source": "positional",
                            "description": "Model name (e.g., htdemucs, htdemucs_ft, htdemucs_6s)"
                        }
                    ]
                },
                {
                    "name": "list_models",
                    "usage": "list_models",
                    "description": "List available Demucs models with descriptions",
                    "arguments": []
                },
                {
                    "name": "set_two_stems",
                    "usage": "set_two_stems <stem_name>",
                    "description": "Output only 2 stems instead of 4 (selected stem + everything else combined)",
                    "arguments": [
                        {
                            "name": "stem_name",
                            "required": True,
                            "source": "positional",
                            "description": "Stem to isolate: drums, vocals, bass, or other (Note: All stems still processed internally)"
                        }
                    ]
                },
                {
                    "name": "clear_two_stems",
                    "usage": "clear_two_stems",
                    "description": "Disable 2-stem mode and return to full 4-stem separation",
                    "arguments": []
                },
                {
                    "name": "set_device",
                    "usage": "set_device <device>",
                    "description": "Set processing device (default: auto)",
                    "arguments": [
                        {
                            "name": "device",
                            "required": True,
                            "source": "positional",
                            "description": "Device: auto (recommended), cpu, or cuda"
                        }
                    ]
                },
                {
                    "name": "set_output_format",
                    "usage": "set_output_format <format>",
                    "description": "Set output audio format (default: wav)",
                    "arguments": [
                        {
                            "name": "format",
                            "required": True,
                            "source": "positional",
                            "description": "Format: wav or mp3"
                        }
                    ]
                },
                {
                    "name": "set_shifts",
                    "usage": "set_shifts <number>",
                    "description": "Set number of random shifts for quality (default: 1, higher=better quality but slower)",
                    "arguments": [
                        {
                            "name": "number",
                            "required": True,
                            "source": "positional",
                            "description": "Number of shifts (0=fastest/lowest quality, 1=default, 10=paper recommendation)"
                        }
                    ]
                }
            ]
        ))
        
        # Demucs Separator Cloud - Executable block (AWS)
        self.register(BlockTypeMetadata(
            name="Demucs Separator (Cloud)",
            type_id="SeparatorCloud",
            description="Separate audio into stems using Demucs on AWS Batch (configure credentials in block settings)",
            execution_mode="executable",
            inputs={"audio": AUDIO_TYPE},
            outputs={"audio": AUDIO_TYPE},  # Single port carries all stems
            tags=["separator", "demucs", "stem", "audio", "cloud", "aws"],
            commands=[]
        ))
        
        # TranscribeNote - Executable block
        self.register(BlockTypeMetadata(
            name="TranscribeNote",
            type_id="TranscribeNote",
            description="Extract notes from audio using Spotify's Basic-Pitch AI model (best for bass/single instruments)",
            execution_mode="executable",
            inputs={"audio": AUDIO_TYPE},
            outputs={"events": EVENT_TYPE},
            tags=["note", "transcription", "pitch", "bass", "basicpitch", "ai"],
            commands=[
                {
                    "name": "set_onset_threshold",
                    "usage": "set_onset_threshold <value>",
                    "description": "Set onset detection threshold (0.0-1.0, default: 0.5)",
                    "arguments": [
                        {
                            "name": "value",
                            "required": True,
                            "source": "positional",
                            "description": "Threshold value (higher = fewer onsets)"
                        }
                    ]
                },
                {
                    "name": "set_frame_threshold",
                    "usage": "set_frame_threshold <value>",
                    "description": "Set frame threshold for note activation (0.0-1.0, default: 0.3)",
                    "arguments": [
                        {
                            "name": "value",
                            "required": True,
                            "source": "positional",
                            "description": "Threshold value"
                        }
                    ]
                },
                {
                    "name": "set_min_note_length",
                    "usage": "set_min_note_length <seconds>",
                    "description": "Set minimum note length in seconds (default: 0.058)",
                    "arguments": [
                        {
                            "name": "seconds",
                            "required": True,
                            "source": "positional",
                            "description": "Minimum duration in seconds"
                        }
                    ]
                },
                {
                    "name": "set_frequency_range",
                    "usage": "set_frequency_range <min_hz> <max_hz>",
                    "description": "Set frequency range for detection (default: 27.5-1000 Hz for bass)",
                    "arguments": [
                        {
                            "name": "min_hz",
                            "required": True,
                            "source": "positional",
                            "description": "Minimum frequency in Hz"
                        },
                        {
                            "name": "max_hz",
                            "required": True,
                            "source": "positional",
                            "description": "Maximum frequency in Hz"
                        }
                    ]
                },
                {
                    "name": "save_to_file",
                    "usage": "save_to_file <true|false>",
                    "description": "Save extracted notes to JSON file",
                    "arguments": [
                        {
                            "name": "enabled",
                            "required": True,
                            "source": "positional",
                            "description": "true or false"
                        }
                    ]
                }
            ]
        ))
        
        # TranscribeLib - Executable block
        self.register(BlockTypeMetadata(
            name="TranscribeLib",
            type_id="TranscribeLib",
            description="Extract notes from audio using Librosa onset detection + pitch tracking (more configurable)",
            execution_mode="executable",
            inputs={"audio": AUDIO_TYPE},
            outputs={"events": EVENT_TYPE},
            tags=["note", "transcription", "pitch", "bass", "librosa", "onset"],
            commands=[
                {
                    "name": "set_hop_length",
                    "usage": "set_hop_length <samples>",
                    "description": "Set hop length for analysis (default: 512 samples)",
                    "arguments": [
                        {
                            "name": "samples",
                            "required": True,
                            "source": "positional",
                            "description": "Hop length in samples"
                        }
                    ]
                },
                {
                    "name": "set_onset_threshold",
                    "usage": "set_onset_threshold <value>",
                    "description": "Set onset detection threshold (default: 0.5)",
                    "arguments": [
                        {
                            "name": "value",
                            "required": True,
                            "source": "positional",
                            "description": "Threshold value"
                        }
                    ]
                },
                {
                    "name": "set_min_duration",
                    "usage": "set_min_duration <seconds>",
                    "description": "Set minimum note duration in seconds (default: 0.05)",
                    "arguments": [
                        {
                            "name": "seconds",
                            "required": True,
                            "source": "positional",
                            "description": "Minimum duration in seconds"
                        }
                    ]
                },
                {
                    "name": "set_frequency_range",
                    "usage": "set_frequency_range <min_hz> <max_hz>",
                    "description": "Set frequency range for pitch detection (default: 27.5-1000 Hz for bass)",
                    "arguments": [
                        {
                            "name": "min_hz",
                            "required": True,
                            "source": "positional",
                            "description": "Minimum frequency (fmin)"
                        },
                        {
                            "name": "max_hz",
                            "required": True,
                            "source": "positional",
                            "description": "Maximum frequency (fmax)"
                        }
                    ]
                },
                {
                    "name": "save_to_file",
                    "usage": "save_to_file <true|false>",
                    "description": "Save extracted notes to JSON file",
                    "arguments": [
                        {
                            "name": "enabled",
                            "required": True,
                            "source": "positional",
                            "description": "true or false"
                        }
                    ]
                }
            ]
        ))
        
        # Audio Filter - Executable block
        self.register(BlockTypeMetadata(
            name="Audio Filter",
            type_id="AudioFilter",
            description="Apply audio filters (low-pass, high-pass, band-pass, band-stop, shelf, peak/bell EQ). Accepts audio or event data (filters source audio referenced by events).",
            execution_mode="executable",
            inputs={"audio": AUDIO_TYPE, "events": EVENT_TYPE},
            outputs={"audio": AUDIO_TYPE},
            tags=["filter", "eq", "equalizer", "lowpass", "highpass", "bandpass", "audio", "frequency"],
            commands=[
                {
                    "name": "set_filter_type",
                    "usage": "set_filter_type <type>",
                    "description": "Set the filter type (lowpass, highpass, bandpass, bandstop, lowshelf, highshelf, peak)",
                    "arguments": [
                        {
                            "name": "type",
                            "required": True,
                            "source": "positional",
                            "description": "Filter type"
                        }
                    ]
                },
                {
                    "name": "set_cutoff",
                    "usage": "set_cutoff <frequency>",
                    "description": "Set the cutoff/center frequency in Hz (20-20000)",
                    "arguments": [
                        {
                            "name": "frequency",
                            "required": True,
                            "source": "positional",
                            "description": "Frequency in Hz"
                        }
                    ]
                },
                {
                    "name": "set_order",
                    "usage": "set_order <order>",
                    "description": "Set the Butterworth filter order (1-8)",
                    "arguments": [
                        {
                            "name": "order",
                            "required": True,
                            "source": "positional",
                            "description": "Filter order (higher = steeper rolloff)"
                        }
                    ]
                },
                {
                    "name": "set_gain",
                    "usage": "set_gain <db>",
                    "description": "Set gain in dB for shelf/peak filters (-24 to +24)",
                    "arguments": [
                        {
                            "name": "db",
                            "required": True,
                            "source": "positional",
                            "description": "Gain in decibels"
                        }
                    ]
                }
            ]
        ))
        
        # ShowManager - Orchestrator block for MA3 communication
        self.register(BlockTypeMetadata(
            name="Show Manager",
            type_id="ShowManager",
            description="Orchestrates bidirectional communication between EchoZero and grandMA3",
            execution_mode="live",  # Live block: maintains connection state
            inputs={},
            outputs={},
            bidirectional={"manipulator": MANIPULATOR_TYPE},  # Bidirectional command port
            tags=["ma3", "grandma", "lighting", "sync", "osc", "orchestrator"],
            commands=[
                {
                    "name": "push_to_ma3",
                    "usage": "push_to_ma3",
                    "description": "Push current events to grandMA3",
                    "arguments": []
                },
                {
                    "name": "pull_from_ma3",
                    "usage": "pull_from_ma3",
                    "description": "Pull events from grandMA3",
                    "arguments": []
                },
                {
                    "name": "set_target_timecode",
                    "usage": "set_target_timecode <number>",
                    "description": "Set target timecode pool index in MA3",
                    "arguments": [
                        {
                            "name": "number",
                            "required": True,
                            "source": "positional",
                            "description": "Timecode pool index (1-based)"
                        }
                    ]
                },
                {
                    "name": "connect",
                    "usage": "connect",
                    "description": "Establish connection to grandMA3",
                    "arguments": []
                },
                {
                    "name": "disconnect",
                    "usage": "disconnect",
                    "description": "Disconnect from grandMA3",
                    "arguments": []
                }
            ]
        ))
        
        # Audio Negate - Negate/cancel audio at event time regions
        self.register(BlockTypeMetadata(
            name="Audio Negate",
            type_id="AudioNegate",
            description="Negate/cancel audio at event time regions (silence, attenuate, or phase-subtract)",
            execution_mode="executable",
            inputs={"audio": AUDIO_TYPE, "subtract_audio": AUDIO_TYPE, "events": EVENT_TYPE},
            outputs={"audio": AUDIO_TYPE},
            tags=["negate", "cancel", "noise", "subtract", "silence", "audio", "events"],
        ))
        
        # Audio Player - Sink/endpoint block for inline audio preview
        self.register(BlockTypeMetadata(
            name="Audio Player",
            type_id="AudioPlayer",
            description="Inline audio player with embedded playback controls in the node editor",
            execution_mode="passthrough",
            inputs={"audio": AUDIO_TYPE},
            tags=["audio", "player", "playback", "preview"],
            commands=[]
        ))
        
        # EQ Bands - Multi-band parametric equalizer
        self.register(BlockTypeMetadata(
            name="EQ Bands",
            type_id="EQBands",
            description="Multi-band parametric EQ with configurable frequency ranges and gain per band",
            execution_mode="executable",
            inputs={"audio": AUDIO_TYPE},
            outputs={"audio": AUDIO_TYPE},
            tags=["eq", "equalizer", "bands", "frequency", "gain", "boost", "cut", "audio", "parametric"],
            commands=[]
        ))
        
        # Export Audio Dataset - Extract event clips from source audio
        self.register(BlockTypeMetadata(
            name="Export Audio Dataset",
            type_id="ExportAudioDataset",
            description="Extract audio clips at event time regions and export as individual files for dataset creation",
            execution_mode="executable",
            inputs={"audio": AUDIO_TYPE, "events": EVENT_TYPE},
            tags=["export", "dataset", "clips", "events", "audio", "extract", "ml", "training"],
            commands=[
                {
                    "name": "set_output_dir",
                    "usage": "set_output_dir <directory>",
                    "description": "Set directory where audio clips will be exported",
                    "arguments": [
                        {
                            "name": "directory",
                            "required": True,
                            "source": "positional",
                            "description": "Filesystem path for exported clips"
                        }
                    ]
                },
                {
                    "name": "set_format",
                    "usage": "set_format <wav|mp3|flac|ogg>",
                    "description": "Choose the output audio format",
                    "arguments": [
                        {
                            "name": "format",
                            "required": True,
                            "source": "positional",
                            "description": "Audio format/extension"
                        }
                    ]
                }
            ]
        ))

        # Dataset Viewer - Manual audit of directory of audio clips
        self.register(BlockTypeMetadata(
            name="Dataset Viewer",
            type_id="DatasetViewer",
            description="Manually audit a directory of audio clips: play, remove (move to removed/), prev/next sample",
            execution_mode="executable",
            tags=["dataset", "audio", "audit", "clips", "viewer"],
            commands=[]
        ))

        self._initialized = True
        Log.info(f"BlockTypeRegistry: Initialized {len(self._block_types)} default block types")


# Global registry instance
_block_registry = None


def get_block_registry() -> BlockTypeRegistry:
    """
    Get the global block type registry instance.
    
    Returns:
        BlockTypeRegistry instance
    """
    global _block_registry
    if _block_registry is None:
        _block_registry = BlockTypeRegistry()
        _block_registry.initialize_default_types()
    return _block_registry

