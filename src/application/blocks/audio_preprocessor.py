"""
Audio Preprocessor Module

Provides signal enhancement techniques for improved onset detection.
Preprocessing emphasizes transients and improves signal quality before onset detection.
"""
from typing import Optional
import numpy as np
from src.utils.message import Log
from src.application.settings.detect_onsets_settings import DetectOnsetsBlockSettings

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    Log.warning("scipy not available - high-pass filtering will use librosa fallback")


class AudioPreprocessor:
    """
    Preprocesses audio signals to improve onset detection accuracy.
    
    Applies signal enhancement techniques to:
    - Emphasize transients/attacks (pre-emphasis)
    - Remove DC offset
    - Filter low-frequency noise (optional high-pass)
    - Normalize signal levels (optional)
    
    All preprocessing steps are configurable and can be disabled.
    """
    
    def preprocess_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        settings: DetectOnsetsBlockSettings
    ) -> np.ndarray:
        """
        Apply preprocessing pipeline to audio signal.
        
        Args:
            audio: Mono audio signal (1D numpy array)
            sample_rate: Audio sample rate in Hz
            settings: DetectOnsetsBlockSettings instance containing preprocessing settings
        
        Returns:
            Preprocessed audio signal (same shape as input)
        """
        if not settings.preprocessing_enabled:
            return audio
        
        processed = audio.copy()
        
        # Step 1: Remove DC offset (always safe, minimal overhead)
        if settings.remove_dc_offset:
            processed = self._remove_dc_offset(processed)
        
        # Step 2: Apply pre-emphasis filter (emphasizes transients)
        if settings.preemphasis_enabled:
            processed = self._apply_preemphasis(
                processed,
                coefficient=settings.preemphasis_coefficient
            )
        
        # Step 3: Apply high-pass filter (optional, removes low-frequency noise)
        if settings.highpass_enabled:
            processed = self._apply_highpass(
                processed,
                sample_rate=sample_rate,
                cutoff=settings.highpass_cutoff
            )
        
        # Step 4: Normalize audio (optional, ensures consistent levels)
        if settings.normalize_audio:
            processed = self._normalize_audio(
                processed,
                method=settings.normalization_method
            )
        
        return processed
    
    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove DC offset by subtracting the mean.
        
        Args:
            audio: Audio signal
        
        Returns:
            Audio with DC offset removed
        """
        return audio - np.mean(audio)
    
    def _apply_preemphasis(
        self,
        audio: np.ndarray,
        coefficient: float = 0.97
    ) -> np.ndarray:
        """
        Apply pre-emphasis filter to emphasize transients.
        
        Formula: y[n] = x[n] - α * x[n-1]
        where α is the pre-emphasis coefficient (typically 0.95-0.97)
        
        This high-pass filter emphasizes high-frequency content (transients/attacks)
        which helps separate closely-spaced onsets.
        
        Args:
            audio: Audio signal
            coefficient: Pre-emphasis coefficient (0.0-1.0)
        
        Returns:
            Pre-emphasized audio signal
        """
        if coefficient == 0.0:
            return audio
        
        # Apply pre-emphasis: y[n] = x[n] - α * x[n-1]
        preemphasized = np.zeros_like(audio)
        preemphasized[0] = audio[0]
        preemphasized[1:] = audio[1:] - coefficient * audio[:-1]
        
        return preemphasized
    
    def _apply_highpass(
        self,
        audio: np.ndarray,
        sample_rate: int,
        cutoff: float = 80.0
    ) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency content.
        
        Low-frequency noise can mask transients, so removing it can improve
        onset detection accuracy.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate in Hz
            cutoff: Cutoff frequency in Hz (default: 80 Hz)
        
        Returns:
            High-pass filtered audio signal
        """
        if cutoff <= 0 or cutoff >= sample_rate / 2:
            Log.warning(
                f"AudioPreprocessor: Invalid high-pass cutoff {cutoff} Hz "
                f"(sample_rate={sample_rate}). Skipping high-pass filter."
            )
            return audio
        
        try:
            if HAS_SCIPY:
                # Use scipy for high-pass filtering (more reliable)
                nyquist = sample_rate / 2.0
                normalized_cutoff = cutoff / nyquist
                
                # Design Butterworth high-pass filter
                # Order 4 provides good rolloff without excessive phase distortion
                b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
                filtered = signal.filtfilt(b, a, audio)
                
                return filtered
            else:
                # Fallback: use librosa's preemphasis (less precise but available)
                Log.debug(
                    "AudioPreprocessor: scipy not available, using librosa preemphasis "
                    "as high-pass approximation"
                )
                try:
                    import librosa
                    # librosa.preemphasis is similar to high-pass but not exactly the same
                    # Use a coefficient that approximates the cutoff
                    # This is a rough approximation
                    alpha = 1.0 - (cutoff / (sample_rate / 2.0))
                    alpha = max(0.0, min(0.99, alpha))
                    return librosa.effects.preemphasis(audio, coef=alpha)
                except ImportError:
                    Log.warning(
                        "AudioPreprocessor: Neither scipy nor librosa.effects available. "
                        "Skipping high-pass filter."
                    )
                    return audio
        except Exception as e:
            Log.warning(
                f"AudioPreprocessor: Error applying high-pass filter: {e}. "
                "Returning unfiltered audio."
            )
            return audio
    
    def _normalize_audio(
        self,
        audio: np.ndarray,
        method: str = "peak"
    ) -> np.ndarray:
        """
        Normalize audio to ensure consistent signal levels.
        
        Args:
            audio: Audio signal
            method: Normalization method ("peak" or "rms")
        
        Returns:
            Normalized audio signal
        """
        if method == "peak":
            # Peak normalization: scale to [-1, 1] range
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0:
                return audio / max_amplitude
            return audio
        
        elif method == "rms":
            # RMS normalization: scale to target RMS level
            rms = np.sqrt(np.mean(audio ** 2))
            target_rms = 0.1  # Conservative target RMS level
            if rms > 0:
                return audio * (target_rms / rms)
            return audio
        
        else:
            Log.warning(
                f"AudioPreprocessor: Unknown normalization method '{method}'. "
                "Using peak normalization."
            )
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0:
                return audio / max_amplitude
            return audio









