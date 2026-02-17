"""
DetectOnsets Block Settings

Settings schema and manager for DetectOnsets blocks.

This settings class demonstrates the new validation framework:
- Uses validated_field() for declarative validation on critical fields
- Maintains backwards compatibility with existing property setters
- Validation is defined once at the schema level
- Registered via @register_block_settings decorator for auto-discovery
"""
from dataclasses import dataclass

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings, validated_field
from src.shared.application.settings import register_block_settings
from src.utils.message import Log


@register_block_settings(
    "DetectOnsets",
    description="Onset detection settings for audio event detection",
    version=1,
    tags=["audio", "detection"]
)
@dataclass
class DetectOnsetsBlockSettings(BaseSettings):
    """
    Settings schema for DetectOnsets blocks.
    
    Settings are organized by responsibility:
    
    Phase 1: Onset Detection (finding where events start)
    - onset_method, onset_threshold, min_silence, use_backtrack
    
    Phase 2: Clip End Detection (finding where events end) - Only in "clips" mode
    - detection_mode, energy_decay_threshold, peak_decay_ratio, decay_detection_method
    - adaptive_threshold_enabled, adaptive_threshold_factor
    - onset_lookahead_time, energy_rise_threshold, min_separation_time
    
    Duration Constraints (safety limits applied after end detection)
    - max_clip_duration, min_clip_duration
    
    All fields have default values for backwards compatibility.
    Settings are stored in block.metadata at the top level.
    
    Validation is defined declaratively using validated_field().
    Use settings.validate() to check all fields, or settings.is_valid() for a quick check.
    """
    # ============================================================
    # PHASE 1: ONSET DETECTION (Finding where events start)
    # ============================================================
    # All settings for detecting when audio events begin
    
    # Detection method - determines how onset strength is computed
    # Validated: must be one of the allowed methods
    onset_method: str = validated_field(
        "default",
        choices=["default", "energy", "flux", "hfc", "complex"]
    )
    
    # Detection sensitivity and filtering
    # Validated: must be between 0.0 and 1.0
    onset_threshold: float = validated_field(0.5, min_value=0.0, max_value=1.0)
    
    # Minimum seconds between onsets (prevents duplicates)
    # Validated: must be >= 0
    min_silence: float = validated_field(0.02, min_value=0.0)
    
    use_backtrack: bool = True  # Use librosa backtrack to align onsets to energy minima
    
    # Analysis parameters (used by both onset and clip end detection)
    # Validated: must be > 0
    energy_frame_length: int = validated_field(2048, min_value=1)
    energy_hop_length: int = validated_field(512, min_value=1)
    
    # Note: librosa onset detection also uses pre_max, post_max, pre_avg, post_avg
    # These are currently hardcoded to 3 in OnsetDetector but can be made configurable
    
    # Output mode: "markers" (point events) or "clips" (events with duration)
    # Validated: must be one of the allowed modes
    output_mode: str = validated_field("markers", choices=["markers", "clips"])
    
    # Classification name for clip events
    # Validated: must be non-empty
    clip_classification: str = validated_field("clip", required=True, min_length=1)
    
    # ============================================================
    # AUDIO PREPROCESSING (Signal enhancement before onset detection)
    # ============================================================
    # Preprocessing improves onset detection accuracy, especially for closely-spaced onsets
    
    preprocessing_enabled: bool = True  # Master switch for all preprocessing
    preemphasis_enabled: bool = True  # Pre-emphasis filter (emphasizes transients)
    # Validated: must be between 0.0 and 1.0
    preemphasis_coefficient: float = validated_field(0.97, min_value=0.0, max_value=1.0)
    remove_dc_offset: bool = True  # Remove DC offset (always safe, minimal overhead)
    highpass_enabled: bool = False  # High-pass filter (removes low-frequency noise)
    # Validated: must be > 0
    highpass_cutoff: float = validated_field(80.0, min_value=0.1)
    normalize_audio: bool = False  # Normalize audio levels (optional)
    # Validated: must be one of the allowed methods
    normalization_method: str = validated_field("peak", choices=["peak", "rms"])
    
    # ============================================================
    # DURATION CONSTRAINTS (Safety limits)
    # ============================================================
    # Validated: must be > 0
    max_clip_duration: float = validated_field(2.0, min_value=0.001)
    min_clip_duration: float = validated_field(0.01, min_value=0.0)
    
    # ============================================================
    # PHASE 2: CLIP END DETECTION (Finding where events end)
    # ============================================================
    # All settings for detecting when audio events stop
    # Only used when output_mode == "clips"
    
    # Detection mode - determines how clip end is detected
    early_cut_mode: bool = False  # Enable peak decay detection (Peak Decay mode)
    # When False: Uses Sustained Silence mode (energy_decay_threshold)
    # When True: Uses Peak Decay mode (peak_decay_ratio)
    
    # Sustained Silence mode settings (when early_cut_mode == False)
    # Validated: must be between 0.0 and 1.0
    energy_decay_threshold: float = validated_field(0.1, min_value=0.0, max_value=1.0)
    
    # Peak Decay mode settings (when early_cut_mode == True)
    # Validated: must be between 0.01 and 1.0
    peak_decay_ratio: float = validated_field(0.5, min_value=0.01, max_value=1.0)
    
    # Validated: must be one of the allowed methods
    decay_detection_method: str = validated_field(
        "threshold",
        choices=["threshold", "rate_of_change", "slope", "confirmed_threshold", "all"]
    )
    
    # Advanced clip end detection settings
    adaptive_threshold_enabled: bool = True  # Use adaptive threshold for quiet events
    # Validated: must be between 0.0 and 1.0
    adaptive_threshold_factor: float = validated_field(0.5, min_value=0.0, max_value=1.0)
    onset_lookahead_time: float = validated_field(0.1, min_value=0.0)
    # Validated: must be >= 1.0
    energy_rise_threshold: float = validated_field(1.5, min_value=1.0)
    min_separation_time: float = validated_field(0.02, min_value=0.0)
    use_numba_acceleration: bool = True  # Use numba JIT acceleration (faster) or librosa only
    
    
    # Post-processing: Split clips with multiple onsets
    split_clips_with_multiple_onsets: bool = False  # If enabled, run onset detection on each clip and split if multiple onsets found
    
    # Legacy support: map "delta" to "onset_threshold"
    @classmethod
    def from_dict(cls, data: dict):
        """
        Load settings from block metadata with backwards compatibility.
        
        Handles legacy "delta" key mapping to "onset_threshold".
        """
        merged = dict(data)
        
        # Map legacy "delta" to "onset_threshold" if present
        if "delta" in merged and "onset_threshold" not in merged:
            merged["onset_threshold"] = merged["delta"]
        
        return super().from_dict(merged)
    
    def to_dict(self) -> dict:
        """
        Convert to metadata format.
        
        Stores onset_threshold (not legacy "delta").
        """
        return super().to_dict()

class DetectOnsetsSettingsManager(BlockSettingsManager):
    """
    Settings manager for DetectOnsets blocks.
    
    Provides type-safe property accessors with validation.
    All settings changes go through this manager (single pathway).
    """
    SETTINGS_CLASS = DetectOnsetsBlockSettings
    
    @property
    def onset_method(self) -> str:
        """Get onset detection method."""
        return self._settings.onset_method
    
    @onset_method.setter
    def onset_method(self, value: str):
        """Set onset detection method with validation."""
        valid_methods = {"default", "energy", "flux", "hfc", "complex"}
        if value not in valid_methods:
            raise ValueError(
                f"Invalid onset method: '{value}'. "
                f"Valid options: {', '.join(sorted(valid_methods))}"
            )
        
        if value != self._settings.onset_method:
            self._settings.onset_method = value
            self._save_setting('onset_method')
    
    @property
    def onset_threshold(self) -> float:
        """Get onset detection threshold (0.0 to 1.0)."""
        return self._settings.onset_threshold
    
    @onset_threshold.setter
    def onset_threshold(self, value: float):
        """Set onset detection threshold with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Threshold must be a number, got {type(value).__name__}")
        
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Onset threshold must be between 0.0 and 1.0, got {value}"
            )
        
        # Update if changed (use small epsilon for float comparison)
        if abs(value - self._settings.onset_threshold) > 0.001:
            self._settings.onset_threshold = float(value)
            self._save_setting('onset_threshold')
    
    @property
    def min_silence(self) -> float:
        """Get minimum silence duration between onsets (in seconds)."""
        return self._settings.min_silence
    
    @min_silence.setter
    def min_silence(self, value: float):
        """Set minimum silence duration with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Min silence must be a number, got {type(value).__name__}")
        
        if value < 0.0:
            raise ValueError(f"Min silence must be >= 0.0, got {value}")
        
        # Update if changed (use small epsilon for float comparison)
        if abs(value - self._settings.min_silence) > 0.0001:
            self._settings.min_silence = float(value)
            self._save_setting('min_silence')
    
    @property
    def output_mode(self) -> str:
        """Get output mode: 'markers' (point events) or 'clips' (events with duration)."""
        return self._settings.output_mode
    
    @output_mode.setter
    def output_mode(self, value: str):
        """Set output mode with validation."""
        valid_modes = {"markers", "clips"}
        if value not in valid_modes:
            raise ValueError(
                f"Invalid output mode: '{value}'. "
                f"Valid options: {', '.join(sorted(valid_modes))}"
            )
        
        if value != self._settings.output_mode:
            self._settings.output_mode = value
            self._save_setting('output_mode')
    
    @property
    def clip_classification(self) -> str:
        """Get classification name for clip events."""
        return self._settings.clip_classification
    
    @clip_classification.setter
    def clip_classification(self, value: str):
        """Set classification name for clip events."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Clip classification must be a non-empty string")
        
        value = value.strip()
        if value != self._settings.clip_classification:
            self._settings.clip_classification = value
            self._save_setting('clip_classification')
    
    @property
    def max_clip_duration(self) -> float:
        """Get maximum clip duration (failsafe, in seconds)."""
        return self._settings.max_clip_duration
    
    @max_clip_duration.setter
    def max_clip_duration(self, value: float):
        """Set maximum clip duration with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Max clip duration must be a number, got {type(value).__name__}")
        
        if value <= 0.0:
            raise ValueError(f"Max clip duration must be > 0.0, got {value}")
        
        if abs(value - self._settings.max_clip_duration) > 0.001:
            self._settings.max_clip_duration = float(value)
            self._save_setting('max_clip_duration')
    
    @property
    def min_clip_duration(self) -> float:
        """Get minimum clip duration (in seconds)."""
        return self._settings.min_clip_duration
    
    @min_clip_duration.setter
    def min_clip_duration(self, value: float):
        """Set minimum clip duration with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Min clip duration must be a number, got {type(value).__name__}")
        
        if value < 0.0:
            raise ValueError(f"Min clip duration must be >= 0.0, got {value}")
        
        if abs(value - self._settings.min_clip_duration) > 0.0001:
            self._settings.min_clip_duration = float(value)
            self._save_setting('min_clip_duration')
    
    @property
    def energy_decay_threshold(self) -> float:
        """Get energy decay threshold (0.0 to 1.0, as fraction of peak)."""
        return self._settings.energy_decay_threshold
    
    @energy_decay_threshold.setter
    def energy_decay_threshold(self, value: float):
        """Set energy decay threshold with validation."""
        
        if not isinstance(value, (int, float)):
            raise ValueError(f"Energy decay threshold must be a number, got {type(value).__name__}")
        
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Energy decay threshold must be between 0.0 and 1.0, got {value}"
            )
        
        if abs(value - self._settings.energy_decay_threshold) > 0.001:
            self._settings.energy_decay_threshold = float(value)
            self._save_setting('energy_decay_threshold')
        else:
            pass  # No change needed if value is within threshold
    
    @property
    def energy_frame_length(self) -> int:
        """Get frame length for RMS energy computation."""
        return self._settings.energy_frame_length
    
    @energy_frame_length.setter
    def energy_frame_length(self, value: int):
        """Set frame length for RMS energy computation with validation."""
        if not isinstance(value, int):
            raise ValueError(f"Energy frame length must be an integer, got {type(value).__name__}")
        
        if value <= 0:
            raise ValueError(f"Energy frame length must be > 0, got {value}")
        
        if value != self._settings.energy_frame_length:
            self._settings.energy_frame_length = value
            self._save_setting('energy_frame_length')
    
    @property
    def energy_hop_length(self) -> int:
        """Get hop length for RMS energy computation."""
        return self._settings.energy_hop_length
    
    @energy_hop_length.setter
    def energy_hop_length(self, value: int):
        """Set hop length for RMS energy computation with validation."""
        if not isinstance(value, int):
            raise ValueError(f"Energy hop length must be an integer, got {type(value).__name__}")
        
        if value <= 0:
            raise ValueError(f"Energy hop length must be > 0, got {value}")
        
        if value != self._settings.energy_hop_length:
            self._settings.energy_hop_length = value
            self._save_setting('energy_hop_length')
    
    @property
    def use_numba_acceleration(self) -> bool:
        """Get whether to use numba JIT acceleration for decay detection."""
        return self._settings.use_numba_acceleration
    
    @use_numba_acceleration.setter
    def use_numba_acceleration(self, value: bool):
        """Set whether to use numba JIT acceleration with validation."""
        if not isinstance(value, bool):
            raise ValueError(f"use_numba_acceleration must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.use_numba_acceleration:
            self._settings.use_numba_acceleration = value
            self._save_setting('use_numba_acceleration')
    
    @property
    def use_backtrack(self) -> bool:
        """Get whether to use librosa backtrack for onset alignment."""
        return self._settings.use_backtrack
    
    @use_backtrack.setter
    def use_backtrack(self, value: bool):
        """Set whether to use librosa backtrack with validation."""
        if not isinstance(value, bool):
            raise ValueError(f"use_backtrack must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.use_backtrack:
            self._settings.use_backtrack = value
            self._save_setting('use_backtrack')
    
    @property
    def adaptive_threshold_enabled(self) -> bool:
        """Get whether adaptive threshold is enabled for quiet events."""
        return self._settings.adaptive_threshold_enabled
    
    @adaptive_threshold_enabled.setter
    def adaptive_threshold_enabled(self, value: bool):
        """Set whether adaptive threshold is enabled."""
        if not isinstance(value, bool):
            raise ValueError(f"adaptive_threshold_enabled must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.adaptive_threshold_enabled:
            self._settings.adaptive_threshold_enabled = value
            if not value:
                Log.debug(
                    "DetectOnsetsSettingsManager: Adaptive threshold disabled - "
                    "adaptive_threshold_factor setting will be ignored during processing"
                )
            self._save_setting('adaptive_threshold_enabled')
    
    @property
    def adaptive_threshold_factor(self) -> float:
        """Get adaptive threshold factor (0.0 to 1.0)."""
        return self._settings.adaptive_threshold_factor
    
    @adaptive_threshold_factor.setter
    def adaptive_threshold_factor(self, value: float):
        """Set adaptive threshold factor with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Adaptive threshold factor must be a number, got {type(value).__name__}")
        
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Adaptive threshold factor must be between 0.0 and 1.0, got {value}"
            )
        
        if abs(value - self._settings.adaptive_threshold_factor) > 0.001:
            self._settings.adaptive_threshold_factor = float(value)
            self._save_setting('adaptive_threshold_factor')
    
    @property
    def onset_lookahead_time(self) -> float:
        """Get onset lookahead time (in seconds)."""
        return self._settings.onset_lookahead_time
    
    @onset_lookahead_time.setter
    def onset_lookahead_time(self, value: float):
        """Set onset lookahead time with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Onset lookahead time must be a number, got {type(value).__name__}")
        
        if value < 0.0:
            raise ValueError(f"Onset lookahead time must be >= 0.0, got {value}")
        
        if abs(value - self._settings.onset_lookahead_time) > 0.001:
            self._settings.onset_lookahead_time = float(value)
            self._save_setting('onset_lookahead_time')
    
    @property
    def energy_rise_threshold(self) -> float:
        """Get energy rise threshold (1.0+)."""
        return self._settings.energy_rise_threshold
    
    @energy_rise_threshold.setter
    def energy_rise_threshold(self, value: float):
        """Set energy rise threshold with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Energy rise threshold must be a number, got {type(value).__name__}")
        
        if value < 1.0:
            raise ValueError(f"Energy rise threshold must be >= 1.0, got {value}")
        
        if abs(value - self._settings.energy_rise_threshold) > 0.001:
            self._settings.energy_rise_threshold = float(value)
            self._save_setting('energy_rise_threshold')
    
    @property
    def min_separation_time(self) -> float:
        """Get minimum separation time before next onset (in seconds)."""
        return self._settings.min_separation_time
    
    @min_separation_time.setter
    def min_separation_time(self, value: float):
        """Set minimum separation time with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Min separation time must be a number, got {type(value).__name__}")
        
        if value < 0.0:
            raise ValueError(f"Min separation time must be >= 0.0, got {value}")
        
        if abs(value - self._settings.min_separation_time) > 0.001:
            self._settings.min_separation_time = float(value)
            self._save_setting('min_separation_time')
    
    @property
    def early_cut_mode(self) -> bool:
        """Get whether early cut mode (peak decay detection) is enabled."""
        return self._settings.early_cut_mode
    
    @early_cut_mode.setter
    def early_cut_mode(self, value: bool):
        """Set whether early cut mode is enabled."""
        if not isinstance(value, bool):
            raise ValueError(f"early_cut_mode must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.early_cut_mode:
            self._settings.early_cut_mode = value
            if value:
                Log.debug(
                    "DetectOnsetsSettingsManager: Early cut mode enabled - "
                    "clips will cut immediately when energy drops to peak_decay_ratio of peak"
                )
            self._save_setting('early_cut_mode')
    
    @property
    def peak_decay_ratio(self) -> float:
        """Get peak decay ratio (0.01 to 1.0)."""
        return self._settings.peak_decay_ratio
    
    @peak_decay_ratio.setter
    def peak_decay_ratio(self, value: float):
        """Set peak decay ratio with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"peak_decay_ratio must be a number, got {type(value).__name__}")
        
        if not 0.01 <= value <= 1.0:
            raise ValueError(f"peak_decay_ratio must be between 0.01 and 1.0, got {value}")
        
        if abs(value - self._settings.peak_decay_ratio) > 0.001:
            self._settings.peak_decay_ratio = float(value)
            self._save_setting('peak_decay_ratio')
    
    @property
    def decay_detection_method(self) -> str:
        """Get decay detection method."""
        return self._settings.decay_detection_method
    
    @decay_detection_method.setter
    def decay_detection_method(self, value: str):
        """Set decay detection method with validation."""
        valid_methods = ["threshold", "rate_of_change", "slope", "confirmed_threshold", "all"]
        if value not in valid_methods:
            raise ValueError(f"decay_detection_method must be one of {valid_methods}, got {value}")
        
        if value != self._settings.decay_detection_method:
            self._settings.decay_detection_method = value
            Log.debug(f"DetectOnsetsSettingsManager: Decay detection method set to '{value}'")
            self._save_setting('decay_detection_method')
    
    @property
    def split_clips_with_multiple_onsets(self) -> bool:
        """Get whether to split clips with multiple onsets."""
        return self._settings.split_clips_with_multiple_onsets
    
    @split_clips_with_multiple_onsets.setter
    def split_clips_with_multiple_onsets(self, value: bool):
        """Set whether to split clips with multiple onsets."""
        if not isinstance(value, bool):
            raise ValueError(f"split_clips_with_multiple_onsets must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.split_clips_with_multiple_onsets:
            self._settings.split_clips_with_multiple_onsets = value
            Log.debug(f"DetectOnsetsSettingsManager: Split clips with multiple onsets set to {value}")
            self._save_setting('split_clips_with_multiple_onsets')
    
    # ============================================================
    # PREPROCESSING SETTINGS
    # ============================================================
    
    @property
    def preprocessing_enabled(self) -> bool:
        """Get whether audio preprocessing is enabled."""
        return self._settings.preprocessing_enabled
    
    @preprocessing_enabled.setter
    def preprocessing_enabled(self, value: bool):
        """Set whether audio preprocessing is enabled."""
        if not isinstance(value, bool):
            raise ValueError(f"preprocessing_enabled must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.preprocessing_enabled:
            self._settings.preprocessing_enabled = value
            if not value:
                Log.debug("DetectOnsetsSettingsManager: Preprocessing disabled - all preprocessing steps will be skipped")
            self._save_setting('preprocessing_enabled')
    
    @property
    def preemphasis_enabled(self) -> bool:
        """Get whether pre-emphasis filter is enabled."""
        return self._settings.preemphasis_enabled
    
    @preemphasis_enabled.setter
    def preemphasis_enabled(self, value: bool):
        """Set whether pre-emphasis filter is enabled."""
        if not isinstance(value, bool):
            raise ValueError(f"preemphasis_enabled must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.preemphasis_enabled:
            self._settings.preemphasis_enabled = value
            self._save_setting('preemphasis_enabled')
    
    @property
    def preemphasis_coefficient(self) -> float:
        """Get pre-emphasis coefficient (0.0 to 1.0)."""
        return self._settings.preemphasis_coefficient
    
    @preemphasis_coefficient.setter
    def preemphasis_coefficient(self, value: float):
        """Set pre-emphasis coefficient with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"preemphasis_coefficient must be a number, got {type(value).__name__}")
        
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"preemphasis_coefficient must be between 0.0 and 1.0, got {value}")
        
        if abs(value - self._settings.preemphasis_coefficient) > 0.001:
            self._settings.preemphasis_coefficient = float(value)
            self._save_setting('preemphasis_coefficient')
    
    @property
    def remove_dc_offset(self) -> bool:
        """Get whether DC offset removal is enabled."""
        return self._settings.remove_dc_offset
    
    @remove_dc_offset.setter
    def remove_dc_offset(self, value: bool):
        """Set whether DC offset removal is enabled."""
        if not isinstance(value, bool):
            raise ValueError(f"remove_dc_offset must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.remove_dc_offset:
            self._settings.remove_dc_offset = value
            self._save_setting('remove_dc_offset')
    
    @property
    def highpass_enabled(self) -> bool:
        """Get whether high-pass filter is enabled."""
        return self._settings.highpass_enabled
    
    @highpass_enabled.setter
    def highpass_enabled(self, value: bool):
        """Set whether high-pass filter is enabled."""
        if not isinstance(value, bool):
            raise ValueError(f"highpass_enabled must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.highpass_enabled:
            self._settings.highpass_enabled = value
            self._save_setting('highpass_enabled')
    
    @property
    def highpass_cutoff(self) -> float:
        """Get high-pass cutoff frequency in Hz."""
        return self._settings.highpass_cutoff
    
    @highpass_cutoff.setter
    def highpass_cutoff(self, value: float):
        """Set high-pass cutoff frequency with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"highpass_cutoff must be a number, got {type(value).__name__}")
        
        if value <= 0.0:
            raise ValueError(f"highpass_cutoff must be > 0.0, got {value}")
        
        if abs(value - self._settings.highpass_cutoff) > 0.1:
            self._settings.highpass_cutoff = float(value)
            self._save_setting('highpass_cutoff')
    
    @property
    def normalize_audio(self) -> bool:
        """Get whether audio normalization is enabled."""
        return self._settings.normalize_audio
    
    @normalize_audio.setter
    def normalize_audio(self, value: bool):
        """Set whether audio normalization is enabled."""
        if not isinstance(value, bool):
            raise ValueError(f"normalize_audio must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.normalize_audio:
            self._settings.normalize_audio = value
            self._save_setting('normalize_audio')
    
    @property
    def normalization_method(self) -> str:
        """Get normalization method ('peak' or 'rms')."""
        return self._settings.normalization_method
    
    @normalization_method.setter
    def normalization_method(self, value: str):
        """Set normalization method with validation."""
        valid_methods = {"peak", "rms"}
        if value not in valid_methods:
            raise ValueError(
                f"Invalid normalization method: '{value}'. "
                f"Valid options: {', '.join(sorted(valid_methods))}"
            )
        
        if value != self._settings.normalization_method:
            self._settings.normalization_method = value
            self._save_setting('normalization_method')
