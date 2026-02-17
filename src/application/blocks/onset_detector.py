"""
Onset Detector Module

Dedicated module for detecting onset times (where events start).
Separates onset detection logic from clip end detection.

This module handles Phase 1: Finding where audio events begin.
"""
from typing import List, Optional
import numpy as np
from src.utils.message import Log
from src.application.settings.detect_onsets_settings import DetectOnsetsBlockSettings

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class OnsetDetector:
    """
    Detects onset times in audio (where events start).
    
    This module handles Phase 1 of onset detection: finding when audio events begin.
    It is separate from Phase 2 (ClipEndDetector) which finds where events end.
    
    Responsibility: Given audio, determine when events start/happen.
    
    Supports multiple detection methods:
    - default: librosa's default (spectral flux)
    - energy: RMS energy-based detection
    - flux: Spectral flux (same as default)
    - hfc: High Frequency Content (emphasizes high frequencies)
    - complex: Complex domain using chroma features
    """
    
    def detect_onsets(
        self,
        audio_mono: np.ndarray,
        sample_rate: int,
        settings: DetectOnsetsBlockSettings
    ) -> List[float]:
        """
        Detect onset times in audio.
        
        Args:
            audio_mono: Mono audio signal (1D numpy array)
            sample_rate: Audio sample rate in Hz
            settings: DetectOnsetsBlockSettings instance containing:
                - onset_method: Detection method to use
                - onset_threshold: Detection threshold (delta parameter)
                - min_silence: Minimum seconds between onsets
                - use_backtrack: Whether to use backtrack alignment
                - energy_frame_length: Frame length for energy-based methods
                - energy_hop_length: Hop length for analysis
        
        Returns:
            List of onset times in seconds (sorted, no duplicates)
        """
        if not HAS_LIBROSA:
            raise RuntimeError(
                "librosa library not installed. Install with: pip install librosa"
            )
        
        # Extract onset detection settings
        onset_method = settings.onset_method
        onset_threshold = settings.onset_threshold
        min_silence = settings.min_silence
        use_backtrack = settings.use_backtrack
        energy_hop_length = settings.energy_hop_length
        
        # Convert min_silence (seconds) to wait (frames) for librosa
        # wait is minimum frames between onsets
        wait = max(1, int(min_silence * sample_rate / energy_hop_length))
        
        # librosa onset detection parameters (can be made configurable later)
        pre_max = 3  # Pre-maximum filter size
        post_max = 3  # Post-maximum filter size
        pre_avg = 3  # Pre-average filter size
        post_avg = 3  # Post-average filter size
        
        # Detect onsets using specified method
        try:
            onset_strength = self._compute_onset_strength(
                audio_mono=audio_mono,
                sample_rate=sample_rate,
                hop_length=energy_hop_length,
                method=onset_method
            )
            
            # Detect onsets from the strength envelope
            # When using onset_envelope, units="time" returns times directly
            onset_times = librosa.onset.onset_detect(
                onset_envelope=onset_strength,
                sr=sample_rate,
                hop_length=energy_hop_length,
                units="time",
                delta=onset_threshold,
                pre_max=pre_max,
                post_max=post_max,
                pre_avg=pre_avg,
                post_avg=post_avg,
                wait=wait,
                backtrack=use_backtrack
            )
            
        except Exception as e:
            Log.warning(
                f"OnsetDetector: Error with method '{onset_method}': {e}. "
                "Falling back to default method."
            )
            # Fallback to default method
            onset_times = librosa.onset.onset_detect(
                y=audio_mono,
                sr=sample_rate,
                units="time",
                delta=onset_threshold,
                hop_length=energy_hop_length,
                pre_max=pre_max,
                post_max=post_max,
                pre_avg=pre_avg,
                post_avg=post_avg,
                wait=wait,
                backtrack=use_backtrack
            )
        
        # Remove duplicate onsets (within small tolerance - 1ms)
        # This prevents librosa from returning duplicate times due to rounding
        unique_onsets = []
        seen_times = set()
        tolerance = 0.001  # 1ms tolerance
        for onset_time in onset_times:
            # Round to nearest millisecond to check for duplicates
            rounded_time = round(onset_time, 3)
            if rounded_time not in seen_times:
                unique_onsets.append(onset_time)
                seen_times.add(rounded_time)
        
        if len(unique_onsets) != len(onset_times):
            Log.debug(
                f"OnsetDetector: Removed {len(onset_times) - len(unique_onsets)} "
                f"duplicate onsets (within 1ms tolerance)"
            )
        
        Log.debug(
            f"OnsetDetector: Detected {len(unique_onsets)} unique onsets "
            f"using method '{onset_method}'"
        )
        
        return unique_onsets
    
    def _compute_onset_strength(
        self,
        audio_mono: np.ndarray,
        sample_rate: int,
        hop_length: int,
        method: str
    ) -> np.ndarray:
        """
        Compute onset strength envelope using specified method.
        
        Args:
            audio_mono: Mono audio signal
            sample_rate: Audio sample rate
            hop_length: Hop length for analysis
            method: Detection method ("default", "energy", "flux", "hfc", "complex")
        
        Returns:
            Onset strength envelope (1D numpy array)
        """
        if method == "default":
            # Default: use librosa's default (spectral flux)
            return librosa.onset.onset_strength(
                y=audio_mono,
                sr=sample_rate,
                hop_length=hop_length
            )
        
        elif method == "energy":
            # Energy-based: use RMS energy
            return librosa.onset.onset_strength(
                y=audio_mono,
                sr=sample_rate,
                hop_length=hop_length,
                feature=librosa.feature.rms
            )
        
        elif method == "flux":
            # Spectral flux (librosa's default, same as "default")
            return librosa.onset.onset_strength(
                y=audio_mono,
                sr=sample_rate,
                hop_length=hop_length
            )
        
        elif method == "hfc":
            # High Frequency Content: emphasize high frequencies
            # Use mel spectrogram with emphasis on high mel bands
            S = librosa.feature.melspectrogram(
                y=audio_mono,
                sr=sample_rate,
                hop_length=hop_length,
                n_mels=128
            )
            # Emphasize high frequency content
            hfc_weights = np.linspace(0.5, 2.0, S.shape[0])
            S_weighted = S * hfc_weights[:, np.newaxis]
            return librosa.onset.onset_strength(
                S=librosa.power_to_db(S_weighted),
                sr=sample_rate,
                hop_length=hop_length
            )
        
        elif method == "complex":
            # Complex domain: use phase information
            # librosa doesn't have direct complex domain support,
            # but we can use chroma which captures harmonic content
            return librosa.onset.onset_strength(
                y=audio_mono,
                sr=sample_rate,
                hop_length=hop_length,
                feature=librosa.feature.chroma_stft
            )
        
        else:
            # Fallback to default
            Log.warning(f"OnsetDetector: Unknown method '{method}', using default")
            return librosa.onset.onset_strength(
                y=audio_mono,
                sr=sample_rate,
                hop_length=hop_length
            )










