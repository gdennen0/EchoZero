"""
Audio Data Item

Concrete implementation of AudioDataItem with audio loading/saving capabilities.
"""
import os
import numpy as np
from typing import Optional
from datetime import datetime
import uuid

from src.shared.domain.entities.data_item import AudioDataItem as BaseAudioDataItem
from src.utils.message import Log

try:
    import soundfile as sf
    import librosa
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    Log.warning("soundfile or librosa not available - audio loading/saving disabled")


class AudioDataItem(BaseAudioDataItem):
    """
    Audio data item with audio loading/saving capabilities.
    
    Handles:
    - Loading audio files (librosa/soundfile)
    - Saving audio files
    - Audio metadata (sample_rate, channels, duration)
    - Audio data storage (in-memory or on disk)
    """
    
    def __init__(
        self,
        id: str = "",
        block_id: str = "",
        name: str = "",
        type: str = "Audio",
        created_at: Optional[datetime] = None,
        file_path: Optional[str] = None,
        sample_rate: Optional[int] = None,
        length_ms: Optional[float] = None,
        metadata: Optional[dict] = None,
        audio_data: Optional[np.ndarray] = None,
        channels: Optional[int] = None
    ):
        """
        Initialize audio data item.
        
        Args:
            audio_data: Optional numpy array of audio data
            channels: Number of audio channels
            Other args: See base class
        """
        if not id:
            id = str(uuid.uuid4())
        if not created_at:
            created_at = datetime.utcnow()
        if metadata is None:
            metadata = {}
        
        super().__init__(
            id=id,
            block_id=block_id,
            name=name or "AudioData",
            type=type,
            created_at=created_at,
            file_path=file_path,
            sample_rate=sample_rate,
            length_ms=length_ms,
            metadata=metadata
        )
        self._audio_data = audio_data  # In-memory audio data
        self.channels = channels
        
        # Infer properties from audio data if provided
        if audio_data is not None:
            if self.sample_rate is None:
                self.sample_rate = metadata.get('sample_rate', 44100)  # Default
            if self.channels is None:
                if len(audio_data.shape) == 1:
                    self.channels = 1
                else:
                    self.channels = audio_data.shape[0]
            if self.length_ms is None and self.sample_rate:
                duration_samples = audio_data.shape[-1] if len(audio_data.shape) > 0 else 0
                self.length_ms = (duration_samples / self.sample_rate) * 1000.0
    
    def load_audio(self, file_path: str, target_sr: Optional[int] = None) -> bool:
        """
        Load audio from file.
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (resamples if different)
            
        Returns:
            True if loaded successfully
        """
        if not HAS_AUDIO_LIBS:
            Log.error("Audio libraries not available - cannot load audio")
            return False
        
        if not os.path.exists(file_path):
            Log.error(f"Audio file not found: {file_path}")
            return False
        
        try:
            # Load audio
            if target_sr:
                audio_data, sample_rate = librosa.load(file_path, sr=target_sr, mono=False)
            else:
                audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)
            
            # Handle mono audio (librosa returns 1D array)
            if len(audio_data.shape) == 1:
                channels = 1
            else:
                channels = audio_data.shape[0]
            
            # Store audio data
            self._audio_data = audio_data
            self.sample_rate = int(sample_rate)
            self.channels = channels
            self.file_path = file_path
            
            # Calculate duration
            duration_samples = audio_data.shape[-1] if len(audio_data.shape) > 0 else 0
            self.length_ms = (duration_samples / self.sample_rate) * 1000.0
            
            # Extract file info (only non-duplicate metadata)
            try:
                info = sf.info(file_path)
                self.metadata['file_format'] = info.format
                self.metadata['subtype'] = info.subtype
            except Exception:
                pass
            
            Log.info(f"Loaded audio: {file_path} ({self.channels}ch, {self.sample_rate}Hz, {self.length_ms:.2f}ms)")
            return True
            
        except Exception as e:
            Log.error(f"Failed to load audio from {file_path}: {e}")
            return False
    
    def save_audio(self, file_path: str, format: str = "WAV", subtype: str = "PCM_24") -> bool:
        """
        Save audio to file.
        
        Args:
            file_path: Path to save audio file
            format: Audio format (WAV, FLAC, etc.)
            subtype: Audio subtype (PCM_24, PCM_16, etc.)
            
        Returns:
            True if saved successfully
        """
        if not HAS_AUDIO_LIBS:
            Log.error("Audio libraries not available - cannot save audio")
            return False
        
        if self._audio_data is None:
            Log.error("No audio data to save")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert to appropriate format for soundfile
            audio_to_save = self._audio_data
            
            # If mono (1D array), keep as is. If multi-channel (2D array), transpose for soundfile
            if len(audio_to_save.shape) == 2:
                audio_to_save = audio_to_save.T  # soundfile expects (samples, channels)
            
            # Save audio
            sf.write(file_path, audio_to_save, self.sample_rate or 44100, format=format, subtype=subtype)
            
            self.file_path = file_path
            Log.info(f"Saved audio: {file_path}")
            return True
            
        except Exception as e:
            Log.error(f"Failed to save audio to {file_path}: {e}")
            return False
    
    def get_audio_data(self) -> Optional[np.ndarray]:
        """
        Get audio data array. Lazy-loads from file_path if not in memory.
        
        After block execution, _audio_data is cleared to free RAM.
        This method reloads from disk on demand (rare -- only during re-processing).
        Playback uses librosa.load(file_path, offset, duration) directly, so this
        method is not called during normal playback.
        
        Returns:
            Audio data as numpy array, or None if unavailable
        """
        if self._audio_data is not None:
            return self._audio_data
        
        # Lazy-load from file if available
        if self.file_path and os.path.exists(self.file_path) and HAS_AUDIO_LIBS:
            try:
                audio_data, sample_rate = librosa.load(
                    self.file_path, sr=self.sample_rate, mono=False
                )
                self._audio_data = audio_data
                self.sample_rate = int(sample_rate)
                Log.debug(f"AudioDataItem: Lazy-loaded audio from {self.file_path}")
                return self._audio_data
            except Exception as e:
                Log.warning(f"AudioDataItem: Failed to lazy-load audio from {self.file_path}: {e}")
                return None
        
        return None
    
    def release_audio_data(self):
        """
        Release in-memory audio data to free RAM.
        
        Called after block execution completes and outputs are persisted.
        The audio file on disk remains available for lazy reloading if needed.
        """
        if self._audio_data is not None:
            Log.debug(f"AudioDataItem: Releasing in-memory audio for '{self.name}'")
            self._audio_data = None
    
    def set_audio_data(self, audio_data: np.ndarray, sample_rate: int):
        """
        Set audio data directly.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate in Hz
        """
        self._audio_data = audio_data
        self.sample_rate = sample_rate
        
        # Update channels
        if len(audio_data.shape) == 1:
            self.channels = 1
        else:
            self.channels = audio_data.shape[0]
        
        # Update duration
        duration_samples = audio_data.shape[-1] if len(audio_data.shape) > 0 else 0
        self.length_ms = (duration_samples / self.sample_rate) * 1000.0
        
        # Note: sample_rate, channels, duration are top-level attributes -- not duplicated in metadata
    
    def get_waveform_data(self) -> Optional[np.ndarray]:
        """
        Get stored waveform data.
        
        Returns:
            Waveform amplitude array (normalized to [-1, 1]) or None if not available
        """
        try:
            from src.shared.application.services.waveform_service import get_waveform_service
            service = get_waveform_service()
            return service.load_waveform(self)
        except Exception as e:
            Log.warning(f"AudioDataItem: Failed to load waveform: {e}")
            return None
    
    def get_waveform_slice(
        self,
        start_time: float,
        end_time: float,
        target_resolution: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Get waveform slice for a time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            target_resolution: Optional target resolution (points per second)
            
        Returns:
            Waveform slice array or None if not available
        """
        try:
            from src.shared.application.services.waveform_service import get_waveform_service
            service = get_waveform_service()
            return service.get_waveform_slice(self, start_time, end_time, target_resolution)
        except Exception as e:
            Log.warning(f"AudioDataItem: Failed to get waveform slice: {e}")
            return None
    
    def has_waveform(self) -> bool:
        """
        Check if waveform is available.
        
        Returns:
            True if waveform exists and is accessible
        """
        try:
            from src.shared.application.services.waveform_service import get_waveform_service
            service = get_waveform_service()
            return service.has_waveform(self)
        except Exception:
            return False
    
    @property
    def waveform_resolution(self) -> Optional[int]:
        """
        Get stored waveform resolution.
        
        Returns:
            Resolution in points per second, or None if not available
        """
        try:
            from src.shared.application.services.waveform_service import get_waveform_service
            service = get_waveform_service()
            return service.get_waveform_resolution(self)
        except Exception:
            return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary including audio-specific fields"""
        data = super().to_dict()
        data["channels"] = self.channels
        # Note: audio_data is not serialized (too large), only metadata
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AudioDataItem':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            block_id=data["block_id"],
            name=data["name"],
            type=data.get("type", "Audio"),
            created_at=datetime.fromisoformat(data["created_at"]),
            file_path=data.get("file_path"),
            sample_rate=data.get("sample_rate"),
            length_ms=data.get("length_ms"),
            channels=data.get("channels"),
            metadata=data.get("metadata", {})
        )

