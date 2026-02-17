"""
Clip Audio Player

Utility for playing audio clips extracted from source audio files.
Used by EventInspector to preview detected clip events.
"""

import json
import os
import tempfile
import time
from typing import Optional, Callable
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

try:
    import librosa
    import soundfile as sf
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False

from ..playback.controller import SimpleAudioPlayer
from ..logging import TimelineLog as Log



class ClipAudioPlayer(QObject):
    """
    Plays audio clips extracted from source audio files.
    
    Features:
    - Extracts audio segments using librosa
    - Creates temporary WAV files
    - Plays clips using SimpleAudioPlayer
    - Handles cleanup of temporary files
    - Position tracking and signals
    - Loop playback support
    
    Audio lookup uses data_item_repo.get(audio_id) directly.
    No callback. No fallback. No searching across blocks.
    """
    
    # Signal emitted when playback position changes
    position_changed = pyqtSignal(float)  # Position in seconds
    
    # Signal emitted when playback finishes (for loop handling)
    playback_finished = pyqtSignal()
    
    def __init__(self, data_item_repo=None, audio_lookup_callback: Optional[Callable] = None, parent=None):
        """
        Initialize clip audio player.
        
        Args:
            data_item_repo: DataItemRepository for direct audio item lookup by ID.
            audio_lookup_callback: Deprecated. Kept for backward compatibility but ignored
                                   if data_item_repo is provided.
            parent: Parent QObject for Qt parent-child relationship
        """
        super().__init__(parent)
        self._data_item_repo = data_item_repo
        self._audio_lookup_callback = audio_lookup_callback
        self._current_player: Optional[SimpleAudioPlayer] = None
        self._temp_file: Optional[str] = None
        self._is_playing = False
        self._is_looping = False
        self._duration = 0.0
        self._clip_start_time = 0.0
        self._clip_end_time = 0.0
        
        # Timer for position updates
        self._position_timer = QTimer(self)
        self._position_timer.timeout.connect(self._update_position)
        self._position_timer.setInterval(50)  # Update every 50ms for smooth playhead
    
    def is_playing(self) -> bool:
        """Check if currently playing a clip."""
        return self._is_playing and self._current_player is not None and self._current_player.is_playing()
    
    def play_clip(
        self,
        audio_id: Optional[str],
        audio_name: Optional[str],
        start_time: float,
        end_time: float
    ) -> tuple[bool, str]:
        """
        Extract and play audio clip.
        
        Args:
            audio_id: Source audio item ID (preferred)
            audio_name: Source audio item name (fallback)
            start_time: Start time in seconds within source audio
            end_time: End time in seconds within source audio
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not HAS_AUDIO_LIBS:
            return False, "Audio libraries (librosa/soundfile) not available"
        
        if not self._data_item_repo and not self._audio_lookup_callback:
            return False, "Audio lookup not available (no repo or callback)"
        
        # Stop any current playback
        self.stop()
        
        # Look up audio item -- direct repo lookup (preferred) or legacy callback
        audio_item = None
        if self._data_item_repo and audio_id:
            audio_item = self._data_item_repo.get(audio_id)
        elif self._audio_lookup_callback:
            audio_item = self._audio_lookup_callback(audio_id, audio_name)
        
        if not audio_item:
            return False, f"Audio source not found (id: {audio_id}, name: {audio_name})"
        
        if not audio_item.file_path or not os.path.exists(audio_item.file_path):
            return False, f"Audio file not found: {audio_item.file_path}"
        
        # Validate time range
        duration = end_time - start_time
        if duration <= 0:
            return False, f"Invalid time range: {start_time:.3f}s to {end_time:.3f}s"
        
        # Extract audio segment
        temp_file = self._extract_clip(
            audio_item.file_path,
            start_time,
            end_time,
            getattr(audio_item, 'sample_rate', None)
        )
        if not temp_file:
            return False, "Failed to extract audio clip"
        
        # Store temp file path for cleanup
        self._temp_file = temp_file
        
        # Store clip timing
        self._clip_start_time = start_time
        self._clip_end_time = end_time
        self._duration = duration
        
        # Play extracted clip
        try:
            self._current_player = SimpleAudioPlayer()
            if self._current_player.load(temp_file):
                self._duration = self._current_player.get_duration()
                self._current_player.play()
                self._is_playing = True
                
                # Start position update timer
                self._position_timer.start()
                
                Log.info(f"ClipAudioPlayer: Playing clip from {start_time:.3f}s to {end_time:.3f}s")
                return True, f"Playing clip ({duration:.3f}s)"
            else:
                return False, "Failed to load extracted clip for playback"
        except Exception as e:
            Log.error(f"ClipAudioPlayer: Playback error: {e}")
            return False, f"Playback error: {str(e)}"
    
    def stop(self):
        """Stop current playback and cleanup temporary file."""
        self._position_timer.stop()
        
        if self._current_player:
            try:
                self._current_player.stop()
                self._current_player.cleanup()
            except Exception as e:
                Log.warning(f"ClipAudioPlayer: Error stopping player: {e}")
            self._current_player = None
        
        self._is_playing = False
        self._is_looping = False
        
        # Clean up temporary file
        if self._temp_file and os.path.exists(self._temp_file):
            try:
                os.remove(self._temp_file)
                Log.debug(f"ClipAudioPlayer: Cleaned up temp file: {self._temp_file}")
            except Exception as e:
                Log.warning(f"ClipAudioPlayer: Failed to remove temp file: {e}")
            self._temp_file = None
    
    def get_position(self) -> float:
        """Get current playback position in seconds."""
        if self._current_player:
            return self._current_player.get_position()
        return 0.0
    
    def set_position(self, seconds: float):
        """
        Seek to position.
        
        Args:
            seconds: Position in seconds (0 to duration)
        """
        if self._current_player:
            seconds = max(0.0, min(seconds, self._duration))
            self._current_player.set_position(seconds)
            self.position_changed.emit(seconds)
    
    def get_duration(self) -> float:
        """Get clip duration in seconds."""
        return self._duration
    
    def set_loop(self, enabled: bool):
        """
        Enable/disable loop playback.
        
        Args:
            enabled: True to loop, False to play once
        """
        self._is_looping = enabled
        Log.debug(f"ClipAudioPlayer: Loop {'enabled' if enabled else 'disabled'}")
    
    def is_looping(self) -> bool:
        """Check if loop is enabled."""
        return self._is_looping
    
    def pause(self):
        """Pause playback."""
        if self._current_player:
            self._current_player.pause()
            self._position_timer.stop()
        self._is_playing = False
    
    def resume(self):
        """Resume playback."""
        if self._current_player:
            self._current_player.play()
            self._is_playing = True
            self._position_timer.start()
    
    def _update_position(self):
        """Update position and emit signal. Handle loop if enabled."""
        if not self._current_player or not self._is_playing:
            return
        
        position = self._current_player.get_position()
        
        # Check if playback finished
        if not self._current_player.is_playing():
            if self._is_looping:
                # Restart from beginning
                self._current_player.set_position(0.0)
                self._current_player.play()
                position = 0.0
            else:
                # Playback finished
                self._is_playing = False
                self._position_timer.stop()
                self.playback_finished.emit()
                return
        
        # Emit position update
        self.position_changed.emit(position)
    
    def _extract_clip(
        self,
        source_file: str,
        start_time: float,
        end_time: float,
        sample_rate: Optional[int] = None
    ) -> Optional[str]:
        """
        Extract audio segment to temporary file.
        
        Args:
            source_file: Path to source audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            sample_rate: Optional target sample rate (uses source rate if None)
            
        Returns:
            Path to temporary file, or None on error
        """
        try:
            duration = end_time - start_time
            
            # Load audio segment using librosa
            # librosa.load with offset and duration is efficient (doesn't load entire file)
            audio_data, sr = librosa.load(
                source_file,
                sr=sample_rate,
                offset=start_time,
                duration=duration,
                mono=False  # Preserve stereo if available
            )
            
            # Handle mono audio (librosa returns 1D array)
            if len(audio_data.shape) == 1:
                # Mono: convert to 2D (1 channel, N samples)
                audio_data = audio_data.reshape(1, -1)
            
            # Transpose for soundfile (soundfile expects shape: [samples, channels])
            # librosa returns shape: [channels, samples]
            if len(audio_data.shape) == 2:
                audio_data = audio_data.T
            
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)  # Close file descriptor, we'll use the path
            
            # Save extracted segment as WAV
            sf.write(temp_path, audio_data, sr)
            
            Log.debug(f"ClipAudioPlayer: Extracted clip to {temp_path} ({duration:.3f}s, {sr}Hz)")
            return temp_path
            
        except Exception as e:
            Log.error(f"ClipAudioPlayer: Failed to extract clip from {source_file}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """Clean up all resources (called on panel close)."""
        self.stop()
        self._position_timer.stop()





